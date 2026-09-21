"""Signed-cookie identity and Gradio transport guards."""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import os
import re
import secrets
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import parse_qs, urlsplit

from fastapi import Request
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from .observability import Observability


logger = logging.getLogger(__name__)

COOKIE_SCHEMA = 1
COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60
_MAX_JSON_CONTROL_BODY = 2 * 1024 * 1024
_COOKIE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{3,96}$")
_EVENT_PATH_RE = re.compile(r"/gradio_api/call(?:/v2)?/[^/]+/([^/]+)$")
_OPERATIONAL_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})
_CURRENT_OWNER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sam3_session_cookie_owner",
    default=None,
)


class SessionWebError(ValueError):
    """Invalid deployment configuration or unauthorized transport identity."""


@dataclass(frozen=True)
class SessionCookieSettings:
    deployment_id: str
    cookie_name: str
    secret: bytes
    allowed_origins: frozenset[str]
    secure_cookie: bool

    @property
    def signer_secret(self) -> str:
        return base64.urlsafe_b64encode(self.secret).decode("ascii")


def _origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise SessionWebError(f"invalid exact origin: {value!r}")
    default_port = 80 if parsed.scheme == "http" else 443
    port = parsed.port or default_port
    return f"{parsed.scheme}://{parsed.hostname.lower()}:{port}"


def _read_secret(path_text: str) -> bytes:
    path = Path(path_text)
    if not path.is_absolute():
        raise SessionWebError("SAM3_SESSION_SECRET_FILE must be absolute")
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise SessionWebError("session cookie secret file is unavailable") from exc
    if mode != 0o600:
        raise SessionWebError("session cookie secret file must have mode 0600")
    try:
        secret = path.read_bytes().rstrip(b"\r\n")
    except OSError as exc:
        raise SessionWebError("session cookie secret file cannot be read") from exc
    if len(secret) < 32:
        raise SessionWebError("session cookie secret must contain at least 32 bytes")
    return secret


def load_session_cookie_settings(
    environ: dict[str, str] | None = None,
) -> SessionCookieSettings:
    env = os.environ if environ is None else environ
    deployment_id = env.get("SAM3_SESSION_DEPLOYMENT_ID", "").strip()
    cookie_name = env.get("SAM3_SESSION_COOKIE_NAME", "").strip()
    secret_file = env.get("SAM3_SESSION_SECRET_FILE", "").strip()
    origins_text = env.get("SAM3_ALLOWED_ORIGINS", "").strip()
    secure_text = env.get("SAM3_SESSION_COOKIE_SECURE", "").strip()
    if not _IDENTIFIER_RE.fullmatch(deployment_id):
        raise SessionWebError("SAM3_SESSION_DEPLOYMENT_ID is required and invalid")
    if not _COOKIE_NAME_RE.fullmatch(cookie_name):
        raise SessionWebError("SAM3_SESSION_COOKIE_NAME is required and invalid")
    if not secret_file:
        raise SessionWebError("SAM3_SESSION_SECRET_FILE is required")
    if not origins_text:
        raise SessionWebError("SAM3_ALLOWED_ORIGINS is required")
    origins = frozenset(_origin(item) for item in origins_text.split(",") if item.strip())
    if not origins:
        raise SessionWebError("at least one exact allowed origin is required")
    if secure_text not in {"0", "1"}:
        raise SessionWebError("SAM3_SESSION_COOKIE_SECURE must be exactly 0 or 1")
    secure_cookie = secure_text == "1"
    if not secure_cookie and env.get("SAM3_ALLOW_INSECURE_COOKIE", "") != "1":
        raise SessionWebError(
            "HTTP requires explicit SAM3_ALLOW_INSECURE_COOKIE=1"
        )
    return SessionCookieSettings(
        deployment_id=deployment_id,
        cookie_name=cookie_name,
        secret=_read_secret(secret_file),
        allowed_origins=origins,
        secure_cookie=secure_cookie,
    )


def _valid_owner_session(
    session: Any,
    settings: SessionCookieSettings,
) -> str | None:
    if not isinstance(session, dict):
        return None
    if set(session) != {"owner_id", "schema", "deployment_id"}:
        return None
    owner_id = session.get("owner_id")
    if (
        session.get("schema") != COOKIE_SCHEMA
        or session.get("deployment_id") != settings.deployment_id
        or not isinstance(owner_id, str)
        or len(owner_id) < 43
        or len(owner_id) > 128
    ):
        return None
    return owner_id


def owner_from_request(
    request: Request,
    settings: SessionCookieSettings,
) -> str | None:
    return _valid_owner_session(request.scope.get("session"), settings)


class OwnerClaimRegistry:
    """Process-local authorization claims for Gradio hashes and queue events."""

    def __init__(self, *, max_hashes: int = 4096) -> None:
        self.max_hashes = max_hashes
        self._hash_owners: dict[str, str] = {}
        self._event_owners: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _value(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise SessionWebError(f"invalid {label}")
        return value

    def claim_hash(self, owner_id: str, session_hash: str) -> None:
        owner_id = self._value(owner_id, "owner")
        session_hash = self._value(session_hash, "session hash")
        with self._lock:
            existing = self._hash_owners.get(session_hash)
            if existing is not None and not secrets.compare_digest(existing, owner_id):
                raise SessionWebError("session hash belongs to another browser owner")
            if existing is None:
                if len(self._hash_owners) >= self.max_hashes:
                    raise SessionWebError("session claim capacity is full")
                self._hash_owners[session_hash] = owner_id

    def require_hash(self, owner_id: str, session_hash: str) -> None:
        owner_id = self._value(owner_id, "owner")
        session_hash = self._value(session_hash, "session hash")
        with self._lock:
            existing = self._hash_owners.get(session_hash)
            if existing is None or not secrets.compare_digest(existing, owner_id):
                raise SessionWebError("session hash is not owned by this browser")

    def bind_event(self, owner_id: str, session_hash: str, event_id: str) -> None:
        self.require_hash(owner_id, session_hash)
        event_id = self._value(event_id, "event id")
        with self._lock:
            existing = self._event_owners.get(event_id)
            expected = (owner_id, session_hash)
            if existing is not None and existing != expected:
                raise SessionWebError("event belongs to another browser owner")
            self._event_owners[event_id] = expected

    def require_event(
        self,
        owner_id: str,
        event_id: str,
        session_hash: str | None = None,
    ) -> None:
        owner_id = self._value(owner_id, "owner")
        event_id = self._value(event_id, "event id")
        with self._lock:
            existing = self._event_owners.get(event_id)
            if existing is None or not secrets.compare_digest(existing[0], owner_id):
                raise SessionWebError("event is not owned by this browser")
            if session_hash is not None and existing[1] != session_hash:
                raise SessionWebError("event does not match session hash")

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "hash_claims": len(self._hash_owners),
                "event_claims": len(self._event_owners),
            }


class OwnerBoundStateHolder:
    """Protect request-time Gradio state reads without patching Gradio."""

    def __init__(self, wrapped: Any, claims: OwnerClaimRegistry) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_claims", claims)

    def _check(self, session_hash: str) -> None:
        owner_id = _CURRENT_OWNER.get()
        if owner_id is not None:
            self._claims.require_hash(owner_id, session_hash)

    def __getitem__(self, session_hash: str) -> Any:
        self._check(session_hash)
        return self._wrapped[session_hash]

    def __contains__(self, session_hash: str) -> bool:
        self._check(session_hash)
        return session_hash in self._wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._wrapped, name, value)


def _error(
    code: str,
    status_code: int,
    message: str,
    *,
    error_id: str | None = None,
) -> JSONResponse:
    payload = {"code": code, "message": message}
    if error_id is not None:
        payload["error_id"] = error_id
    return JSONResponse(
        {"error": payload},
        status_code=status_code,
    )


def _referer_origin(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        default_port = 80 if parsed.scheme == "http" else 443
        return f"{parsed.scheme}://{parsed.hostname.lower()}:{parsed.port or default_port}"
    except ValueError:
        return None


def _write_origin(headers: Headers, allowed_origins: frozenset[str]) -> bool:
    origin = headers.get("origin")
    if origin is not None:
        try:
            return _origin(origin) in allowed_origins
        except SessionWebError:
            return False
    referer = headers.get("referer")
    if referer is None:
        return False
    return _referer_origin(referer) in allowed_origins


async def _read_json_body(receive: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    body = bytearray()
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            continue
        body.extend(message.get("body", b""))
        if len(body) > _MAX_JSON_CONTROL_BODY:
            raise SessionWebError("JSON control request is too large")
        if not message.get("more_body", False):
            break
    try:
        data = json.loads(body or b"{}")
    except (TypeError, ValueError) as exc:
        raise SessionWebError("invalid JSON control request") from exc
    if not isinstance(data, dict):
        raise SessionWebError("JSON control request must be an object")
    return data, messages


def _replay(messages: Iterable[dict[str, Any]]):
    iterator = iter(messages)

    async def receive() -> dict[str, Any]:
        try:
            return next(iterator)
        except StopIteration:
            return {"type": "http.disconnect"}

    return receive


class SessionSecurityMiddleware:
    """Issue owner cookies and guard Gradio state/queue identifiers."""

    def __init__(
        self,
        app: Any,
        *,
        settings: SessionCookieSettings,
        claims: OwnerClaimRegistry,
        observability: "Observability | None" = None,
    ) -> None:
        self.app = app
        self.settings = settings
        self.claims = claims
        self.observability = observability

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")
        if method == "GET" and path in _OPERATIONAL_PATHS:
            await self.app(scope, receive, send)
            return
        session = scope.get("session")
        owner_id = _valid_owner_session(session, self.settings)
        if owner_id is None and method == "GET" and path == "/":
            owner_id = secrets.token_urlsafe(32)
            session.clear()
            session.update(
                owner_id=owner_id,
                schema=COOKIE_SCHEMA,
                deployment_id=self.settings.deployment_id,
            )
        if owner_id is None:
            error_id = None
            if self.observability is not None:
                error_id = self.observability.record_session_rejection(
                    "SESSION_COOKIE_REQUIRED"
                )
            response = _error(
                "SESSION_COOKIE_REQUIRED",
                401,
                "会话身份缺失或已失效，请重新打开首页。",
                error_id=error_id,
            )
            await response(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if method in {"POST", "PUT", "PATCH", "DELETE"} and not _write_origin(
            headers,
            self.settings.allowed_origins,
        ):
            error_id = None
            if self.observability is not None:
                error_id = self.observability.record_session_rejection(
                    "SESSION_ORIGIN_REJECTED",
                    owner_id=owner_id,
                )
            response = _error(
                "SESSION_ORIGIN_REJECTED",
                403,
                "请求来源校验失败，请从当前工作台重试。",
                error_id=error_id,
            )
            await response(scope, receive, send)
            return

        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        session_hash = query.get("session_hash", [None])[0]
        event_id: str | None = None
        body_data: dict[str, Any] | None = None
        original_receive = receive
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        try:
            if method in {"POST", "PUT", "PATCH", "DELETE"} and content_type == "application/json":
                body_data, messages = await _read_json_body(receive)
                receive = _replay(messages)
                session_hash = body_data.get("session_hash", session_hash)
                event_id = body_data.get("event_id")
            heartbeat_prefix = "/gradio_api/heartbeat/"
            if path.startswith(heartbeat_prefix):
                session_hash = path[len(heartbeat_prefix) :].split("/", 1)[0]
            match = _EVENT_PATH_RE.fullmatch(path)
            if match:
                event_id = match.group(1)
            if path.startswith("/gradio_api/stream/"):
                tail = path[len("/gradio_api/stream/") :].split("/")
                if len(tail) >= 3:
                    session_hash = tail[0]
                elif tail:
                    event_id = tail[0]
            if session_hash is not None:
                self.claims.claim_hash(owner_id, session_hash)
            if event_id is not None and path.endswith("/cancel"):
                self.claims.require_event(owner_id, event_id, session_hash)
            elif event_id is not None and method == "GET":
                self.claims.require_event(owner_id, event_id, session_hash)
            elif event_id is not None and path.endswith("/reset"):
                self.claims.require_event(owner_id, event_id, session_hash)
            elif event_id is not None and path.startswith("/gradio_api/stream/"):
                self.claims.require_event(owner_id, event_id, session_hash)
        except SessionWebError as exc:
            logger.warning(
                "session transport rejected code=SESSION_OWNER_MISMATCH"
            )
            error_id = None
            if self.observability is not None:
                error_id = self.observability.record_session_rejection(
                    "SESSION_OWNER_MISMATCH",
                    owner_id=owner_id,
                    session_hash=session_hash,
                    exc=exc,
                )
            response = _error(
                "SESSION_OWNER_MISMATCH",
                403,
                "该页面或任务不属于当前浏览器会话。",
                error_id=error_id,
            )
            await response(scope, original_receive, send)
            return

        token = _CURRENT_OWNER.set(owner_id)
        response_content_type = ""
        response_body = bytearray()

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_content_type
            if message["type"] == "http.response.start":
                response_headers = Headers(raw=message.get("headers", []))
                response_content_type = response_headers.get("content-type", "")
            elif (
                message["type"] == "http.response.body"
                and response_content_type.startswith("application/json")
                and len(response_body) <= 65536
            ):
                response_body.extend(message.get("body", b""))
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and session_hash is not None
                and len(response_body) <= 65536
            ):
                try:
                    payload = json.loads(response_body)
                    returned_event = payload.get("event_id")
                    if isinstance(returned_event, str):
                        self.claims.bind_event(owner_id, session_hash, returned_event)
                except (SessionWebError, TypeError, ValueError):
                    pass

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _CURRENT_OWNER.reset(token)


def protect_gradio_state_holder(gradio_app: Any, claims: OwnerClaimRegistry) -> None:
    if isinstance(gradio_app.state_holder, OwnerBoundStateHolder):
        return
    wrapped = OwnerBoundStateHolder(gradio_app.state_holder, claims)
    gradio_app.state_holder = wrapped
    blocks = gradio_app.get_blocks()
    blocks.state_holder = wrapped


__all__ = [
    "COOKIE_MAX_AGE_SECONDS",
    "OwnerClaimRegistry",
    "SessionCookieSettings",
    "SessionSecurityMiddleware",
    "SessionWebError",
    "load_session_cookie_settings",
    "owner_from_request",
    "protect_gradio_state_holder",
]
