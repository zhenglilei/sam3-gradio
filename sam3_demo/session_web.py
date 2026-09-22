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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
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
_OPERATIONAL_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})
_CURRENT_OWNER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sam3_session_cookie_owner",
    default=None,
)


class SessionWebError(ValueError):
    """Invalid deployment configuration or unauthorized transport identity."""


class SessionRequestError(SessionWebError):
    """A malformed request body that must not reach Gradio."""

    code = "SESSION_REQUEST_INVALID"
    status_code = 400


class SessionRequestTooLarge(SessionRequestError):
    code = "SESSION_REQUEST_TOO_LARGE"
    status_code = 413


class SessionClientDisconnected(SessionWebError):
    """The client disconnected before its control request body completed."""


class SessionClaimCapacityError(SessionWebError):
    """The bounded ownership registry cannot safely admit another claim."""

    code = "SESSION_CLAIM_CAPACITY"
    status_code = 503


@dataclass
class _HashClaim:
    owner_id: str
    last_seen: float


@dataclass
class _EventClaim:
    owner_id: str
    session_hash: str
    last_seen: float


@dataclass
class _RequestLease:
    session_hash: str | None
    event_id: str | None
    reserved_event: bool
    created_hash: bool


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

    def __init__(
        self,
        *,
        max_hashes: int = 4096,
        max_events: int | None = None,
        claim_ttl_seconds: float = 60 * 60,
        hash_live_probe: Callable[[str], bool] | None = None,
        event_live_probe: Callable[[str], bool] | None = None,
        reclaim_callback: Callable[[str | None, tuple[str, ...]], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_hashes < 1 or (max_events is not None and max_events < 1):
            raise ValueError("claim registry capacities must be positive")
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be positive")
        self.max_hashes = max_hashes
        self.max_events = max_events or max_hashes * 4
        self.claim_ttl_seconds = claim_ttl_seconds
        self._hash_live_probe = hash_live_probe
        self._event_live_probe = event_live_probe
        self._reclaim_callback = reclaim_callback
        self._clock = clock
        self._hash_owners: dict[str, _HashClaim] = {}
        self._event_owners: dict[str, _EventClaim] = {}
        self._inflight_hashes: dict[str, int] = {}
        self._inflight_events: dict[str, int] = {}
        self._event_reservations = 0
        self._last_reap = 0.0
        self._lock = threading.RLock()

    def set_liveness_probes(
        self,
        *,
        hash_live_probe: Callable[[str], bool],
        event_live_probe: Callable[[str], bool],
        reclaim_callback: Callable[[str | None, tuple[str, ...]], bool] | None = None,
    ) -> None:
        """Install synchronous probes; missing or uncertain liveness fails closed."""
        with self._lock:
            self._hash_live_probe = hash_live_probe
            self._event_live_probe = event_live_probe
            self._reclaim_callback = reclaim_callback

    def _reclaim(self, session_hash: str | None, event_ids: tuple[str, ...]) -> bool:
        if self._reclaim_callback is None:
            return False
        try:
            return self._reclaim_callback(session_hash, event_ids) is True
        except Exception:
            logger.exception("session claim cleanup failed; ownership retained")
            return False

    @staticmethod
    def _value(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise SessionWebError(f"invalid {label}")
        return value

    @staticmethod
    def _is_live(probe: Callable[[str], bool] | None, value: str) -> bool:
        if probe is None:
            return True
        try:
            result = probe(value)
        except Exception:
            logger.exception("session claim liveness probe failed")
            return True
        return result is not False

    def _hash_reclaimable(self, session_hash: str) -> bool:
        if self._inflight_hashes.get(session_hash, 0):
            return False
        if self._is_live(self._hash_live_probe, session_hash):
            return False
        for event_id, event in self._event_owners.items():
            if event.session_hash != session_hash:
                continue
            if self._inflight_events.get(event_id, 0):
                return False
            reclaimable = (
                self._reclaim(None, (event_id,)) if self._reclaim_callback is not None
                else not self._is_live(self._event_live_probe, event_id)
            )
            if not reclaimable:
                return False
        return True

    def _reap_locked(self, now: float, *, force: bool = False) -> int:
        if not force and now - self._last_reap < 60:
            return 0
        self._last_reap = now
        removed = 0
        for event_id, claim in list(self._event_owners.items()):
            if now - claim.last_seen < self.claim_ttl_seconds:
                continue
            if self._inflight_events.get(event_id, 0):
                claim.last_seen = now
                continue
            reclaimable = (
                self._reclaim(None, (event_id,)) if self._reclaim_callback is not None
                else not self._is_live(self._event_live_probe, event_id)
            )
            if not reclaimable:
                claim.last_seen = now
                continue
            self._event_owners.pop(event_id, None)
            removed += 1
        for session_hash, claim in list(self._hash_owners.items()):
            if now - claim.last_seen < self.claim_ttl_seconds:
                continue
            event_ids = tuple(
                event_id for event_id, event in self._event_owners.items()
                if event.session_hash == session_hash
            )
            if (self._inflight_hashes.get(session_hash, 0)
                    or any(self._inflight_events.get(event_id, 0) for event_id in event_ids)):
                claim.last_seen = now
                continue
            reclaimable = (
                self._reclaim(session_hash, event_ids) if self._reclaim_callback is not None
                else self._hash_reclaimable(session_hash)
            )
            if not reclaimable:
                claim.last_seen = now
                continue
            self._remove_hash_locked(session_hash)
            removed += 1
        return removed

    def _remove_hash_locked(self, session_hash: str) -> None:
        self._hash_owners.pop(session_hash, None)
        for event_id, claim in list(self._event_owners.items()):
            if claim.session_hash == session_hash:
                self._event_owners.pop(event_id, None)

    def reap_expired_claims(self) -> int:
        """Reclaim only expired hashes and events proven unreachable by probes."""
        with self._lock:
            return self._reap_locked(self._clock(), force=True)

    def _claim_hash_locked(self, owner_id: str, session_hash: str, now: float) -> bool:
        self._reap_locked(now)
        existing = self._hash_owners.get(session_hash)
        if existing is not None and now - existing.last_seen >= self.claim_ttl_seconds:
            self._reap_locked(now, force=True)
            existing = self._hash_owners.get(session_hash)
        if existing is not None:
            if not secrets.compare_digest(existing.owner_id, owner_id):
                raise SessionWebError("session hash belongs to another browser owner")
            existing.last_seen = now
            return False
        self._reap_locked(now)
        if len(self._hash_owners) >= self.max_hashes:
            self._reap_locked(now, force=True)
        if len(self._hash_owners) >= self.max_hashes:
            raise SessionClaimCapacityError("session claim capacity is full")
        if self._hash_live_probe is not None and self._is_live(self._hash_live_probe, session_hash):
            raise SessionWebError("unclaimed backend state cannot be adopted")
        self._hash_owners[session_hash] = _HashClaim(owner_id, now)
        return True

    def begin_request(
        self,
        owner_id: str,
        *,
        session_hash: str | None = None,
        event_id: str | None = None,
        claim_hash: bool = False,
        require_event: bool = False,
        reserve_event: bool = False,
    ) -> _RequestLease:
        owner_id = self._value(owner_id, "owner")
        if session_hash is not None:
            session_hash = self._value(session_hash, "session hash")
        if event_id is not None:
            event_id = self._value(event_id, "event id")
        if require_event and event_id is None:
            raise SessionWebError("event id is required")
        now = self._clock()
        with self._lock:
            self._reap_locked(now)
            hash_claim = self._hash_owners.get(session_hash) if session_hash else None
            event_claim = self._event_owners.get(event_id) if event_id else None
            if hash_claim is not None and now - hash_claim.last_seen >= self.claim_ttl_seconds:
                self._reap_locked(now, force=True)
                hash_claim = self._hash_owners.get(session_hash)
                event_claim = self._event_owners.get(event_id) if event_id else None
            if hash_claim is not None and not secrets.compare_digest(
                hash_claim.owner_id, owner_id
            ):
                raise SessionWebError("session hash belongs to another browser owner")
            if event_claim is not None and not secrets.compare_digest(
                event_claim.owner_id, owner_id
            ):
                raise SessionWebError("event belongs to another browser owner")
            if require_event and event_claim is None:
                raise SessionWebError("event is not owned by this browser")
            if (
                event_claim is not None
                and session_hash is not None
                and event_claim.session_hash != session_hash
            ):
                raise SessionWebError("event does not match session hash")
            if session_hash is not None and hash_claim is None and not claim_hash:
                raise SessionWebError("session hash is not owned by this browser")
            if session_hash is not None and hash_claim is None:
                self._reap_locked(now, force=True)
                if len(self._hash_owners) >= self.max_hashes:
                    raise SessionClaimCapacityError("session claim capacity is full")
                if self._hash_live_probe is not None and self._is_live(self._hash_live_probe, session_hash):
                    raise SessionWebError("unclaimed backend state cannot be adopted")
            if reserve_event:
                self._reap_locked(now, force=True)
                if len(self._event_owners) + self._event_reservations >= self.max_events:
                    raise SessionClaimCapacityError("event claim capacity is full")

            created_hash = False
            if session_hash is not None and hash_claim is None:
                self._hash_owners[session_hash] = _HashClaim(owner_id, now)
                created_hash = True
            if session_hash is not None:
                self._hash_owners[session_hash].last_seen = now
            effective_hash = session_hash or (
                event_claim.session_hash if event_claim is not None else None
            )
            if event_claim is not None:
                event_claim.last_seen = now
            if effective_hash is not None:
                self._inflight_hashes[effective_hash] = (
                    self._inflight_hashes.get(effective_hash, 0) + 1
                )
            if event_id is not None and event_claim is not None:
                self._inflight_events[event_id] = self._inflight_events.get(event_id, 0) + 1
            if reserve_event:
                self._event_reservations += 1
            return _RequestLease(
                session_hash=effective_hash,
                event_id=event_id if event_claim is not None else None,
                reserved_event=reserve_event,
                created_hash=created_hash,
            )

    def bind_reserved_event(
        self,
        lease: _RequestLease,
        owner_id: str,
        session_hash: str,
        event_id: str,
    ) -> None:
        owner_id = self._value(owner_id, "owner")
        session_hash = self._value(session_hash, "session hash")
        event_id = self._value(event_id, "event id")
        now = self._clock()
        with self._lock:
            if not lease.reserved_event:
                self._bind_event_locked(owner_id, session_hash, event_id, now)
                return
            self._event_reservations -= 1
            lease.reserved_event = False
            try:
                self._bind_event_locked(owner_id, session_hash, event_id, now)
            except Exception:
                self._event_reservations += 1
                lease.reserved_event = True
                raise
            lease.event_id = event_id
            self._inflight_events[event_id] = self._inflight_events.get(event_id, 0) + 1

    def end_request(
        self,
        lease: _RequestLease,
        *,
        response_status: int | None = None,
    ) -> None:
        now = self._clock()
        with self._lock:
            if lease.reserved_event:
                self._event_reservations -= 1
                lease.reserved_event = False
            if lease.event_id is not None:
                self._decrement(self._inflight_events, lease.event_id)
                event = self._event_owners.get(lease.event_id)
                if event is not None:
                    event.last_seen = now
            if lease.session_hash is not None:
                self._decrement(self._inflight_hashes, lease.session_hash)
                claim = self._hash_owners.get(lease.session_hash)
                if claim is not None:
                    claim.last_seen = now
            if (
                response_status == 404
                and lease.created_hash
                and lease.session_hash is not None
                and self._hash_reclaimable(lease.session_hash)
            ):
                self._remove_hash_locked(lease.session_hash)

    @staticmethod
    def _decrement(counts: dict[str, int], key: str) -> None:
        count = counts.get(key, 0)
        if count <= 1:
            counts.pop(key, None)
        else:
            counts[key] = count - 1

    def _bind_event_locked(
        self,
        owner_id: str,
        session_hash: str,
        event_id: str,
        now: float,
    ) -> None:
        self._reap_locked(now)
        claim = self._hash_owners.get(session_hash)
        if claim is None or not secrets.compare_digest(claim.owner_id, owner_id):
            raise SessionWebError("session hash is not owned by this browser")
        existing = self._event_owners.get(event_id)
        expected = (owner_id, session_hash)
        if existing is not None:
            if (existing.owner_id, existing.session_hash) != expected:
                raise SessionWebError("event belongs to another browser owner")
            existing.last_seen = now
            return
        if len(self._event_owners) + self._event_reservations >= self.max_events:
            self._reap_locked(now, force=True)
        if len(self._event_owners) + self._event_reservations >= self.max_events:
            raise SessionClaimCapacityError("event claim capacity is full")
        self._event_owners[event_id] = _EventClaim(owner_id, session_hash, now)

    def claim_hash(self, owner_id: str, session_hash: str) -> bool:
        owner_id = self._value(owner_id, "owner")
        session_hash = self._value(session_hash, "session hash")
        with self._lock:
            return self._claim_hash_locked(owner_id, session_hash, self._clock())

    def require_hash(self, owner_id: str, session_hash: str) -> None:
        owner_id = self._value(owner_id, "owner")
        session_hash = self._value(session_hash, "session hash")
        with self._lock:
            now = self._clock()
            existing = self._hash_owners.get(session_hash)
            if existing is not None and now - existing.last_seen >= self.claim_ttl_seconds:
                self._reap_locked(now, force=True)
            else:
                self._reap_locked(now)
            existing = self._hash_owners.get(session_hash)
            if existing is None or not secrets.compare_digest(existing.owner_id, owner_id):
                raise SessionWebError("session hash is not owned by this browser")
            existing.last_seen = now

    def bind_event(self, owner_id: str, session_hash: str, event_id: str) -> None:
        self.require_hash(owner_id, session_hash)
        event_id = self._value(event_id, "event id")
        with self._lock:
            self._bind_event_locked(owner_id, session_hash, event_id, self._clock())

    def require_event(
        self,
        owner_id: str,
        event_id: str,
        session_hash: str | None = None,
    ) -> None:
        owner_id = self._value(owner_id, "owner")
        event_id = self._value(event_id, "event id")
        with self._lock:
            now = self._clock()
            existing = self._event_owners.get(event_id)
            if existing is not None and now - existing.last_seen >= self.claim_ttl_seconds:
                self._reap_locked(now, force=True)
            else:
                self._reap_locked(now)
            existing = self._event_owners.get(event_id)
            if existing is None or not secrets.compare_digest(existing.owner_id, owner_id):
                raise SessionWebError("event is not owned by this browser")
            if session_hash is not None and existing.session_hash != session_hash:
                raise SessionWebError("event does not match session hash")
            existing.last_seen = now

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
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = {"code": code, "message": message}
    if error_id is not None:
        payload["error_id"] = error_id
    return JSONResponse(
        {"error": payload},
        status_code=status_code,
        headers=headers,
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
        message_type = message.get("type")
        if message_type == "http.disconnect":
            raise SessionClientDisconnected("client disconnected during JSON body")
        if message_type != "http.request":
            raise SessionRequestError("unexpected ASGI message in JSON body")
        messages.append(message)
        chunk = message.get("body", b"")
        if not isinstance(chunk, bytes):
            raise SessionRequestError("invalid JSON control request body")
        body.extend(chunk)
        if len(body) > _MAX_JSON_CONTROL_BODY:
            raise SessionRequestTooLarge("JSON control request is too large")
        if not message.get("more_body", False):
            break
    try:
        data = json.loads(body or b"{}")
    except (TypeError, ValueError) as exc:
        raise SessionRequestError("invalid JSON control request") from exc
    if not isinstance(data, dict):
        raise SessionRequestError("JSON control request must be an object")
    return data, messages


def _replay(messages: Iterable[dict[str, Any]]):
    iterator = iter(messages)

    async def receive() -> dict[str, Any]:
        try:
            return next(iterator)
        except StopIteration:
            return {"type": "http.disconnect"}

    return receive


def _session_route_kind(method: str, path: str) -> str | None:
    """Whitelist Gradio routes that carry session or event identifiers."""
    prefix = "/gradio_api/"
    if not path.startswith(prefix):
        return None
    parts = path[len(prefix) :].strip("/").split("/")
    if not parts or not parts[0]:
        return None
    method = method.upper()
    if method == "POST":
        if len(parts) == 2 and parts[0] in {"run", "api"}:
            return "json_hash"
        if parts == ["queue", "join"]:
            return "json_create_event"
        if len(parts) == 2 and parts[0] == "call":
            return "json_create_event"
        if len(parts) == 3 and parts[:2] == ["call", "v2"]:
            return "json_create_event"
        if parts in (["cancel"], ["reset"]):
            return "json_control"
        if len(parts) == 2 and parts[0] == "stream":
            return "stream_event"
        if len(parts) == 3 and parts[0] == "stream" and parts[2] == "close":
            return "stream_event"
    elif method == "GET":
        if parts == ["queue", "data"]:
            return "query_hash"
        if len(parts) == 2 and parts[0] == "heartbeat":
            return "path_hash"
        if len(parts) == 3 and parts[0] == "call":
            return "path_event"
        if len(parts) == 4 and parts[:2] == ["call", "v2"]:
            return "path_event"
        if len(parts) >= 5 and parts[0] == "stream":
            return "path_hash"
    return None


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
        route_kind = _session_route_kind(method, path)
        session_hash = None
        event_id: str | None = None
        body_data: dict[str, Any] = {}
        original_receive = receive
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        lease: _RequestLease | None = None
        try:
            if route_kind is None:
                pass
            elif route_kind in {
                "json_hash",
                "json_create_event",
                "json_control",
                "stream_event",
            } and (not content_type or content_type == "application/json"
                   or content_type.endswith("+json")):
                body_data, messages = await _read_json_body(receive)
                receive = _replay(messages)
                if route_kind in {"json_hash", "json_create_event", "json_control", "stream_event"}:
                    session_hash = body_data.get("session_hash")
                if route_kind == "json_control":
                    event_id = body_data.get("event_id")
            elif route_kind in {"json_hash", "json_create_event", "json_control"}:
                raise SessionRequestError("control requests require JSON")
            if route_kind == "query_hash":
                session_hash = query.get("session_hash", [None])[0]
            elif route_kind == "path_hash":
                parts = path[len("/gradio_api/") :].strip("/").split("/")
                session_hash = parts[1]
            elif route_kind == "path_event":
                event_id = path.rstrip("/").rsplit("/", 1)[-1]
            elif route_kind == "stream_event":
                parts = path[len("/gradio_api/") :].strip("/").split("/")
                event_id = parts[1]

            if route_kind is not None:
                create_hash = route_kind in {"json_hash", "json_create_event", "path_hash"}
                require_event = event_id is not None and route_kind in {
                    "json_control",
                    "path_event",
                    "stream_event",
                }
                reserve_event = (
                    route_kind == "json_create_event" and session_hash is not None
                )
                lease = self.claims.begin_request(
                    owner_id,
                    session_hash=session_hash,
                    event_id=event_id,
                    claim_hash=create_hash,
                    require_event=require_event,
                    reserve_event=reserve_event,
                )
        except SessionClientDisconnected:
            return
        except SessionWebError as exc:
            if isinstance(exc, SessionClaimCapacityError):
                code = exc.code
                status_code = exc.status_code
                message = "会话资源暂时繁忙，请稍后重试。"
                error_type = code
            elif isinstance(exc, SessionRequestError):
                code = exc.code
                status_code = exc.status_code
                message = "请求格式无效或超过允许大小。"
                error_type = code
            else:
                code = "SESSION_OWNER_MISMATCH"
                status_code = 403
                message = "该页面或任务不属于当前浏览器会话。"
                error_type = code
                logger.warning("session transport rejected code=%s", code)
            error_id = None
            if self.observability is not None:
                error_id = self.observability.record_session_rejection(
                    error_type,
                    owner_id=owner_id,
                    session_hash=session_hash,
                    exc=exc,
                )
            response_headers = {"Retry-After": "5"} if status_code == 503 else None
            response = _error(
                code,
                status_code,
                message,
                error_id=error_id,
                headers=response_headers,
            )
            await response(scope, original_receive, send)
            return

        token = _CURRENT_OWNER.set(owner_id)
        response_status: int | None = None
        response_content_type = ""
        response_body = bytearray()

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_content_type, response_status
            if message["type"] == "http.response.start":
                response_headers = Headers(raw=message.get("headers", []))
                response_content_type = response_headers.get("content-type", "")
                response_status = message.get("status")
            elif (
                message["type"] == "http.response.body"
                and response_content_type.startswith("application/json")
                and len(response_body) <= 65536
            ):
                response_body.extend(message.get("body", b""))
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and route_kind == "json_create_event"
                and lease is not None
                and lease.reserved_event
                and session_hash is not None
                and response_status is not None
                and 200 <= response_status < 300
                and len(response_body) <= 65536
            ):
                try:
                    payload = json.loads(response_body)
                    returned_event = payload.get("event_id")
                    if isinstance(returned_event, str):
                        self.claims.bind_reserved_event(
                            lease,
                            owner_id,
                            session_hash,
                            returned_event,
                        )
                except (SessionWebError, TypeError, ValueError):
                    pass
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _CURRENT_OWNER.reset(token)
            if lease is not None:
                self.claims.end_request(lease, response_status=response_status)


def protect_gradio_state_holder(gradio_app: Any, claims: OwnerClaimRegistry) -> None:
    def raw_holder():
        holder = gradio_app.state_holder
        while isinstance(holder, OwnerBoundStateHolder):
            holder = holder._wrapped
        return holder

    def state_holder_contains(session_hash: str) -> bool:
        return session_hash in raw_holder()

    def hash_is_live(session_hash: str) -> bool:
        if state_holder_contains(session_hash):
            return True
        queue = getattr(gradio_app.get_blocks(), "_queue", None)
        if queue is None:
            return True
        if session_hash in queue.pending_messages_per_session:
            return True
        if queue.pending_event_ids_session.get(session_hash):
            return True
        return any(
            getattr(event, "session_hash", None) == session_hash
            for event in queue.event_ids_to_events.values()
        )

    def event_is_live(event_id: str) -> bool:
        queue = getattr(gradio_app.get_blocks(), "_queue", None)
        if queue is None:
            return True
        if event_id in queue.event_ids_to_events:
            return True
        if event_id in gradio_app.iterators:
            return True
        return any(
            event_id in event_ids
            for event_ids in queue.pending_event_ids_session.values()
        )

    def reclaim(session_hash: str | None, event_ids: tuple[str, ...]) -> bool:
        # Gradio 6 retains finished events and even empty SessionState keys.
        # A TTL alone must not drop authorization while these remain readable.
        # This synchronous callback runs under the claim lock on the ASGI loop.
        queue = gradio_app.get_blocks()._queue
        holder = raw_holder()
        targets = set(event_ids)
        if session_hash is not None:
            targets.update(queue.pending_event_ids_session.get(session_hash, ()))
            targets.update(
                event_id for event_id, event in queue.event_ids_to_events.items()
                if event.session_hash == session_hash
            )

        def matches(event):
            return event._id in targets or (
                session_hash is not None and event.session_hash == session_hash
            )

        if any(matches(event) for jobs in queue.active_jobs if jobs for event in jobs):
            return False
        if any(matches(event) for lane in queue.event_queue_per_concurrency_id.values()
               for event in lane.queue):
            return False
        for event_id in targets:
            if gradio_app.iterators.get(event_id) is not None:
                return False
            event = queue.event_ids_to_events.get(event_id)
            if event is not None and event.streaming and event.alive and not event.closed:
                return False

        # No await between the checks and invalidation: a queue job cannot start
        # in this interval. In-flight direct requests are pinned by the registry.
        if session_hash is not None:
            holder.delete_state(session_hash)
            with holder.lock:
                old_state = holder.session_data.pop(session_hash, None)
                if old_state is not None:
                    old_state.state_data.clear()
                holder.time_last_used.pop(session_hash, None)
            queue.pending_messages_per_session.pop(session_hash, None)
            queue.pending_event_ids_session.pop(session_hash, None)
        for event_id in targets:
            queue.event_ids_to_events.pop(event_id, None)
            queue.event_analytics.pop(event_id, None)
            gradio_app.iterators.pop(event_id, None)
            gradio_app.iterators_to_reset.discard(event_id)
        for pending in queue.pending_event_ids_session.values():
            pending.difference_update(targets)
        return (
            (session_hash is None or session_hash not in holder)
            and all(event_id not in queue.event_ids_to_events
                    and event_id not in gradio_app.iterators for event_id in targets)
        )

    claims.set_liveness_probes(
        hash_live_probe=hash_is_live,
        event_live_probe=event_is_live,
        reclaim_callback=reclaim,
    )
    if isinstance(gradio_app.state_holder, OwnerBoundStateHolder):
        return
    wrapped = OwnerBoundStateHolder(gradio_app.state_holder, claims)
    gradio_app.state_holder = wrapped
    blocks = gradio_app.get_blocks()
    blocks.state_holder = wrapped


__all__ = [
    "COOKIE_MAX_AGE_SECONDS",
    "SessionClaimCapacityError",
    "SessionClientDisconnected",
    "OwnerClaimRegistry",
    "SessionRequestError",
    "SessionRequestTooLarge",
    "SessionCookieSettings",
    "SessionSecurityMiddleware",
    "SessionWebError",
    "load_session_cookie_settings",
    "owner_from_request",
    "protect_gradio_state_holder",
]
