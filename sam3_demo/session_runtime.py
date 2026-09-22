"""Gradio-independent session identity and lifecycle primitives."""
from __future__ import annotations
import hashlib
import hmac
import ipaddress
import logging
import math
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
SCHEMA_VERSION = 2
_MAX_HASH = 512
_DIGEST_TAG = b"sam3-session-digest-v2"
_TOKEN_TAG = b"sam3-session-owner-v2"
_RESUME_TAG = b"sam3-session-resume-v2:"
logger = logging.getLogger(__name__)


class SessionError(ValueError):
    """Invalid, stale, or unauthorized session identity."""
class SessionExpired(SessionError):
    """A previously valid session is closed or has expired."""


@dataclass(frozen=True)
class RequestIdentity:
    owner_id: str
    session_hash: str
    client_ip: str


@dataclass
class SessionRecord:
    session_id: str
    owner_id_digest: str
    session_hash_digest: str
    client_ip: str                 # latest raw address is memory-only and diagnostic
    client_ip_digest: str          # initial diagnostic digest retained in browser state
    last_client_ip_digest: str
    generation: int
    owner_token: str
    created_at: float
    last_seen: float
    in_flight: int = 0
    close_requested: bool = False
    resume_id: str = ""
    closed: bool = False
    operation_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )
def _ip(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("client IP must be a non-empty string")
    try:
        return ipaddress.ip_address(value.strip()).compressed
    except ValueError as exc:
        raise ValueError(f"invalid client IP: {value!r}") from exc
def _session_hash(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_HASH:
        raise ValueError("session_hash must be 1..512 characters")
    if value.strip() != value or not value.strip() or "\x00" in value:
        raise ValueError("session_hash contains invalid whitespace or NUL")
    return value


def _owner_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_HASH:
        raise ValueError("owner_id must be 1..512 characters")
    if value.strip() != value or not value.strip() or "\x00" in value:
        raise ValueError("owner_id contains invalid whitespace or NUL")
    return value


def normalize_identity(identity: RequestIdentity) -> RequestIdentity:
    if not isinstance(identity, RequestIdentity):
        raise ValueError("request identity is required")
    return RequestIdentity(
        owner_id=_owner_id(identity.owner_id),
        session_hash=_session_hash(identity.session_hash),
        client_ip=_ip(identity.client_ip),
    )


def _trusted(value: Optional[Sequence[str] | str]) -> tuple[ipaddress._BaseNetwork, ...]:
    if value is None:
        return ()
    if isinstance(value, str) and not value.strip():
        return ()
    items = [x.strip() for x in value.split(",")] if isinstance(value, str) else list(value)
    result = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("trusted proxy CIDR must be non-empty")
        try:
            result.append(ipaddress.ip_network(item.strip(), strict=False))
        except ValueError as exc:
            raise ValueError(f"invalid trusted proxy CIDR: {item!r}") from exc
    return tuple(result)


def validate_trusted_proxy_cidrs(value: Optional[Sequence[str] | str]) -> None:
    _trusted(value)


def _is_trusted(address: ipaddress._BaseAddress, networks: tuple[ipaddress._BaseNetwork, ...]) -> bool:
    return any(address in network for network in networks)
def _header_values(headers: Optional[Mapping[str, Any]], name: str) -> list[str]:
    if headers is None:
        return []
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a mapping")
    values: list[str] = []
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != name.lower():
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            if not all(isinstance(item, str) for item in value):
                return ["\x00invalid\x00"]
            values.extend(value)
        else:
            return ["\x00invalid\x00"]
    return values
def resolve_client_ip(
    peer_ip: str,
    headers: Optional[Mapping[str, Any]] = None,
    trusted_proxy_cidrs: Optional[Sequence[str] | str] = None,
) -> str:
    """Normalize a peer and safely resolve X-Forwarded-For when trusted.
    X-Real-IP is ignored.  A malformed forwarded chain always falls back to
    the normalized immediate peer address.
    """
    peer_text = _ip(peer_ip)
    peer = ipaddress.ip_address(peer_text)
    networks = _trusted(trusted_proxy_cidrs)
    if not networks or not _is_trusted(peer, networks):
        return peer_text
    values = _header_values(headers, "X-Forwarded-For")
    if not values:
        return peer_text
    forwarded: list[ipaddress._BaseAddress] = []
    try:
        for value in values:
            parts = value.split(",")
            if any(not part.strip() for part in parts):
                return peer_text
            forwarded.extend(ipaddress.ip_address(part.strip()) for part in parts)
    except (AttributeError, ValueError):
        return peer_text
    for address in reversed(forwarded):
        if not _is_trusted(address, networks):
            return address.compressed
    return peer_text


def _resume_id(
    session_hash: str,
    owner_id: str,
    deployment_id: str,
    secret: bytes,
) -> str:
    hash_bytes = session_hash.encode("utf-8")
    owner_bytes = owner_id.encode("utf-8")
    deployment_bytes = deployment_id.encode("utf-8")
    payload = b"".join(
        (
            _RESUME_TAG,
            len(deployment_bytes).to_bytes(4, "big"),
            deployment_bytes,
            len(owner_bytes).to_bytes(4, "big"),
            owner_bytes,
            len(hash_bytes).to_bytes(4, "big"),
            hash_bytes,
        )
    )
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def resume_id_for_identity(
    session_hash: str,
    owner_id: str,
    *,
    deployment_id: str,
    secret: bytes | str,
) -> str:
    """Return the stable, non-authorizing key used for same-browser drafts."""
    secret_bytes = secret.encode() if isinstance(secret, str) else secret
    if not isinstance(secret_bytes, bytes) or not secret_bytes:
        raise ValueError("secret must be non-empty bytes or string")
    return _resume_id(
        _session_hash(session_hash),
        _owner_id(owner_id),
        _owner_id(deployment_id),
        secret_bytes,
    )


def _digest(value: str, secret: bytes) -> str:
    return hmac.new(secret, _DIGEST_TAG + value.encode(), hashlib.sha256).hexdigest()
def _owner_token(secret: bytes, record: SessionRecord) -> str:
    return _token_for_fields(
        secret, record.session_id, record.generation,
        record.owner_id_digest, record.session_hash_digest,
    )
def _token_for_fields(secret: bytes, session_id: str, generation: int,
                      owner_digest: str, hash_digest: str) -> str:
    text = "|".join(
        (
            session_id,
            str(generation),
            owner_digest,
            hash_digest,
        )
    ).encode("ascii")
    return hmac.new(secret, _TOKEN_TAG + text, hashlib.sha256).hexdigest()
class SessionRegistry:
    """Thread-safe registry keyed by signed-cookie owner and page hash."""
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        idle_seconds: float = 3600,
        max_sessions: int = 1000,
        cleanup_callback: Optional[Callable[[SessionRecord], None]] = None,
        secret: Optional[bytes | str] = None,
        deployment_id: str = "test",
        start_sweeper: bool = False,
        sweep_interval: float = 5,
    ) -> None:
        if not callable(clock):
            raise ValueError("clock must be callable")
        if (
            isinstance(idle_seconds, bool)
            or not isinstance(idle_seconds, (int, float))
            or not math.isfinite(idle_seconds)
            or idle_seconds <= 0
        ):
            raise ValueError("idle_seconds must be positive finite numeric")
        if not isinstance(max_sessions, int) or isinstance(max_sessions, bool) or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        if (
            isinstance(sweep_interval, bool)
            or not isinstance(sweep_interval, (int, float))
            or not math.isfinite(sweep_interval)
            or sweep_interval <= 0
        ):
            raise ValueError("sweep_interval must be positive finite numeric")
        if cleanup_callback is not None and not callable(cleanup_callback):
            raise ValueError("cleanup_callback must be callable")
        if secret is None:
            secret_bytes = secrets.token_bytes(32)
        elif isinstance(secret, str) and secret:
            secret_bytes = secret.encode()
        elif isinstance(secret, bytes) and secret:
            secret_bytes = secret
        else:
            raise ValueError("secret must be non-empty bytes or string")
        self._clock = clock
        self.idle_seconds = float(idle_seconds)
        self.max_sessions = max_sessions
        self._cleanup_callback = cleanup_callback
        self._secret = secret_bytes
        self.deployment_id = _owner_id(deployment_id)
        self._sweep_interval = float(sweep_interval)
        self._lock = threading.RLock()
        self._records: dict[str, SessionRecord] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._local = threading.local()
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        self._shutdown = False
        if start_sweeper:
            self._sweeper = threading.Thread(target=self._sweep, name="sam3-session-sweeper", daemon=True)
            self._sweeper.start()
    def _depths(self) -> dict[str, int]:
        depths = getattr(self._local, "depths", None)
        if depths is None:
            depths = {}
            self._local.depths = depths
        return depths
    def _state(self, record: SessionRecord) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": record.session_id,
            "generation": record.generation,
            "owner_token": record.owner_token,
            "owner_id_digest": record.owner_id_digest,
            "session_hash_digest": record.session_hash_digest,
            "client_ip_digest": record.client_ip_digest,
            "resume_id": record.resume_id,
        }
    def _expired(self, record: SessionRecord, now: float) -> bool:
        return not record.close_requested and record.in_flight == 0 and now - record.last_seen >= self.idle_seconds
    def _remove(self, record: SessionRecord) -> Optional[SessionRecord]:
        if record.closed:
            return None
        record.close_requested = True
        key = (record.owner_id_digest, record.session_hash_digest)
        if self._keys.get(key) == record.session_id:
            self._keys.pop(key, None)
        if record.in_flight:
            return None
        self._records.pop(record.session_id, None)
        record.closed = True
        return record
    def _callbacks(self, records: Sequence[SessionRecord]) -> None:
        if self._cleanup_callback is None:
            return
        for record in records:
            try:
                self._cleanup_callback(record)
            except Exception:
                logger.exception("Session cleanup callback failed for %s", record.session_id)
    def _drop_idle(self, now: float) -> list[SessionRecord]:
        removed = []
        with self._lock:
            for record in list(self._records.values()):
                if self._expired(record, now):
                    item = self._remove(record)
                    if item is not None:
                        removed.append(item)
        return removed
    @staticmethod
    def _state_fields(
        state: Mapping[str, Any],
    ) -> tuple[str, int, str, str, str, str, Optional[str]]:
        if not isinstance(state, Mapping):
            raise SessionError("session state must be a mapping")
        try:
            version = state["schema_version"]
            session_id = state["session_id"]
            generation = state["generation"]
            token = state["owner_token"]
            owner_digest = state["owner_id_digest"]
            hash_digest = state["session_hash_digest"]
            ip_digest = state["client_ip_digest"]
            resume_id = state["resume_id"] if "resume_id" in state else None
        except (KeyError, TypeError) as exc:
            raise SessionError("incomplete session state") from exc
        if version != SCHEMA_VERSION or not isinstance(session_id, str) or not session_id:
            raise SessionError("invalid session state schema or id")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise SessionError("invalid session generation")
        if not all(
            isinstance(value, str)
            for value in (token, owner_digest, hash_digest, ip_digest)
        ):
            raise SessionError("invalid session state fields")
        if (
            resume_id is not None
            and (
                not isinstance(resume_id, str)
                or len(resume_id) != hashlib.sha256().digest_size * 2
                or any(char not in "0123456789abcdef" for char in resume_id)
            )
        ):
            raise SessionError("invalid session resume id")
        return (
            session_id,
            generation,
            token,
            owner_digest,
            hash_digest,
            ip_digest,
            resume_id,
        )
    def _record_from_state(self, state: Mapping[str, Any]) -> SessionRecord:
        (
            session_id,
            generation,
            token,
            owner_digest,
            hash_digest,
            ip_digest,
            resume_id,
        ) = self._state_fields(state)
        record = self._records.get(session_id)
        if record is None or record.closed or record.close_requested:
            raise SessionExpired("session is closed or unknown")
        if (
            record.generation != generation
            or record.owner_id_digest != owner_digest
            or record.session_hash_digest != hash_digest
            or record.client_ip_digest != ip_digest
        ):
            raise SessionError("session state does not match record")
        if not hmac.compare_digest(token, _owner_token(self._secret, record)):
            raise SessionError("invalid session owner token")
        if resume_id is not None and not hmac.compare_digest(resume_id, record.resume_id):
            raise SessionError("session resume id does not match record")
        return record
    def _record_for_identity(
        self,
        state: Mapping[str, Any],
        identity: RequestIdentity,
    ) -> SessionRecord:
        identity = normalize_identity(identity)
        record = self._record_from_state(state)
        if record.owner_id_digest != _digest(identity.owner_id, self._secret):
            raise SessionError("cookie owner does not match session")
        if record.session_hash_digest != _digest(identity.session_hash, self._secret):
            raise SessionError("session hash does not match owner")
        record.client_ip = identity.client_ip
        record.last_client_ip_digest = _digest(identity.client_ip, self._secret)
        return record
    def _touch(self, record: SessionRecord, now: float) -> tuple[Optional[SessionRecord], list[SessionRecord]]:
        if self._expired(record, now):
            removed = self._remove(record)
            return None, [removed] if removed is not None else []
        record.last_seen = now
        return record, []
    def bind(self, identity: RequestIdentity) -> dict[str, Any]:
        identity = normalize_identity(identity)
        owner_digest = _digest(identity.owner_id, self._secret)
        hash_value = identity.session_hash
        ip_value = identity.client_ip
        resume_id = _resume_id(
            hash_value,
            identity.owner_id,
            self.deployment_id,
            self._secret,
        )
        hash_digest = _digest(hash_value, self._secret)
        now = self._clock()
        cleanups = self._drop_idle(now)
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionError("session registry is shut down")
                key = (owner_digest, hash_digest)
                existing = self._records.get(self._keys.get(key, ""))
                if existing is not None and not existing.close_requested and not existing.closed:
                    existing.last_seen = now
                    state = self._state(existing)
                else:
                    while len(self._records) >= self.max_sessions:
                        candidates = [r for r in self._records.values() if not r.in_flight and not r.close_requested]
                        if not candidates:
                            raise SessionError("session capacity is temporarily full")
                        victim = min(candidates, key=lambda r: r.last_seen)
                        removed = self._remove(victim)
                        if removed is not None:
                            cleanups.append(removed)
                    record = SessionRecord(
                        session_id=uuid.uuid4().hex,
                        owner_id_digest=owner_digest,
                        session_hash_digest=hash_digest,
                        resume_id=resume_id,
                        client_ip=ip_value,
                        client_ip_digest=_digest(ip_value, self._secret),
                        last_client_ip_digest=_digest(ip_value, self._secret),
                        generation=1,
                        owner_token="",
                        created_at=now,
                        last_seen=now,
                    )
                    record.owner_token = _owner_token(self._secret, record)
                    self._records[record.session_id] = record
                    self._keys[key] = record.session_id
                    state = self._state(record)
        finally:
            self._callbacks(cleanups)
        return state

    def authenticate_state(
        self, state: Mapping[str, Any], identity: RequestIdentity,
    ) -> None:
        """Check ownership even after eviction; never authorize stale business data."""
        identity = normalize_identity(identity)
        if not isinstance(state, Mapping):
            raise SessionError("session state must be a mapping")
        session_id = state.get("session_id")
        token = state.get("owner_token")
        # Business states issued before this fix contain no generation field.
        generation = state.get("generation", 1)
        if (
            not isinstance(session_id, str) or not session_id.isascii()
            or not session_id or "|" in session_id
            or not isinstance(token, str) or not token.isascii()
            or not isinstance(generation, int) or isinstance(generation, bool)
            or generation < 1
        ):
            raise SessionError("invalid recovery identity")
        owner_digest = _digest(identity.owner_id, self._secret)
        hash_digest = _digest(identity.session_hash, self._secret)
        expected = _token_for_fields(
            self._secret, session_id, generation, owner_digest, hash_digest,
        )
        if not hmac.compare_digest(token, expected):
            raise SessionError("stale session does not belong to this browser")
        if "generation" in state:
            self._state_fields(state)
            if (state.get("owner_id_digest") != owner_digest
                    or state.get("session_hash_digest") != hash_digest):
                raise SessionError("stale session does not belong to this browser")
        resume_id = state.get("resume_id")
        if resume_id is not None:
            expected_resume = _resume_id(
                identity.session_hash, identity.owner_id,
                self.deployment_id, self._secret,
            )
            if (not isinstance(resume_id, str) or not resume_id.isascii()
                    or not hmac.compare_digest(resume_id, expected_resume)):
                raise SessionError("stale session does not belong to this browser")

    def ensure(
        self,
        state: Mapping[str, Any] | None,
        identity: RequestIdentity,
    ) -> tuple[dict[str, Any], bool]:
        """Validate a live state or safely bind a replacement after restart."""

        identity = normalize_identity(identity)
        if state is None or (isinstance(state, Mapping) and not state):
            return self.bind(identity), True
        if not isinstance(state, Mapping):
            raise SessionError("session state must be a mapping")
        self.authenticate_state(state, identity)
        try:
            if "generation" in state:
                record = self.validate(state, identity)
            else:
                record = self.validate_owner(state["session_id"], identity)
        except SessionExpired:
            return self.bind(identity), True
        return self._state(record), False

    def validate(
        self,
        state: Mapping[str, Any],
        identity: RequestIdentity,
    ) -> SessionRecord:
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionExpired("session registry is shut down")
                record = self._record_for_identity(state, identity)
                record, cleanups = self._touch(record, self._clock())
                if record is None:
                    raise SessionExpired("session has expired")
        finally:
            self._callbacks(cleanups)
        return record
    def validate_owner(
        self,
        session_id: str,
        identity: RequestIdentity,
    ) -> SessionRecord:
        if not isinstance(session_id, str) or not session_id:
            raise SessionError("session_id must be a non-empty string")
        identity = normalize_identity(identity)
        hash_digest = _digest(identity.session_hash, self._secret)
        owner_digest = _digest(identity.owner_id, self._secret)
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionExpired("session registry is shut down")
                record = self._records.get(session_id)
                if record is None or record.closed or record.close_requested:
                    raise SessionExpired("session is closed or unknown")
                if (
                    record.session_hash_digest != hash_digest
                    or record.owner_id_digest != owner_digest
                ):
                    raise SessionError("session owner does not match")
                record.client_ip = identity.client_ip
                record.last_client_ip_digest = _digest(identity.client_ip, self._secret)
                record, cleanups = self._touch(record, self._clock())
                if record is None:
                    raise SessionExpired("session has expired")
        finally:
            self._callbacks(cleanups)
        return record
    @contextmanager
    def _lease_record(self, record: SessionRecord) -> Iterator[SessionRecord]:
        depths = self._depths()
        cleanups: list[SessionRecord] = []
        record.operation_lock.acquire()
        try:
            with self._lock:
                if self._records.get(record.session_id) is not record or record.closed or record.close_requested:
                    raise SessionExpired("session is closed")
                depth = depths.get(record.session_id, 0)
                if depth == 0:
                    record.in_flight += 1
                depths[record.session_id] = depth + 1
                record.last_seen = self._clock()
            try:
                yield record
            finally:
                with self._lock:
                    depth = depths.get(record.session_id, 0)
                    if depth <= 1:
                        depths.pop(record.session_id, None)
                        record.in_flight = max(0, record.in_flight - 1)
                        record.last_seen = self._clock()
                        if record.close_requested and record.in_flight == 0:
                            removed = self._remove(record)
                            if removed is not None:
                                cleanups.append(removed)
                    else:
                        depths[record.session_id] = depth - 1
                self._callbacks(cleanups)
        finally:
            record.operation_lock.release()
    @contextmanager
    def lease(
        self,
        state: Mapping[str, Any],
        identity: RequestIdentity,
    ) -> Iterator[SessionRecord]:
        with self._lease_record(self.validate(state, identity)) as record:
            yield record
    @contextmanager
    def owner_lease(
        self,
        session_id: str,
        identity: RequestIdentity,
    ) -> Iterator[SessionRecord]:
        with self._lease_record(self.validate_owner(session_id, identity)) as record:
            yield record
    def _close(
        self,
        state: Mapping[str, Any],
        identity: Optional[RequestIdentity] = None,
    ) -> bool:
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    return False
                try:
                    record = self._record_from_state(state)
                    if identity is not None:
                        record = self._record_for_identity(state, identity)
                except SessionError:
                    return False
                if self._expired(record, self._clock()):
                    removed = self._remove(record)
                    if removed is not None:
                        cleanups.append(removed)
                    return False
                removed = self._remove(record)
                if removed is not None:
                    cleanups.append(removed)
                return True
        finally:
            self._callbacks(cleanups)
    def close(
        self,
        state: Mapping[str, Any],
        identity: Optional[RequestIdentity] = None,
    ) -> bool:
        return self._close(state, identity)
    def close_state(self, state: Mapping[str, Any]) -> bool:
        return self._close(state)
    def reap_expired(self) -> int:
        cleanups = self._drop_idle(self._clock())
        self._callbacks(cleanups)
        return len(cleanups)
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._records),
                "max_sessions": self.max_sessions,
                "idle_seconds": self.idle_seconds,
                "shutdown": self._shutdown,
                "sessions": [
                    {
                        "session_id": r.session_id,
                        "generation": r.generation,
                        "session_hash_digest": r.session_hash_digest,
                        "client_ip_digest": r.client_ip_digest,
                        "created_at": r.created_at,
                        "last_seen": r.last_seen,
                        "in_flight": r.in_flight,
                        "close_requested": r.close_requested,
                        "closed": r.closed,
                    }
                    for r in self._records.values()
                ],
            }
    def _sweep(self) -> None:
        while not self._stop.wait(self._sweep_interval):
            try:
                self.reap_expired()
            except Exception:
                logger.exception("Session sweeper failed")
    def shutdown(self) -> None:
        self._stop.set()
        if self._sweeper is not None and self._sweeper is not threading.current_thread():
            self._sweeper.join(timeout=max(1.0, self._sweep_interval * 2))
        cleanups = []
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for record in list(self._records.values()):
                record.close_requested = True
                self._keys.pop(
                    (record.owner_id_digest, record.session_hash_digest),
                    None,
                )
                if record.in_flight == 0:
                    self._records.pop(record.session_id, None)
                    record.closed = True
                    cleanups.append(record)
        self._callbacks(cleanups)
__all__ = [
    "SCHEMA_VERSION",
    "SessionError",
    "SessionExpired",
    "RequestIdentity",
    "SessionRecord",
    "SessionRegistry",
    "normalize_identity",
    "resume_id_for_identity",
    "resolve_client_ip",
    "validate_trusted_proxy_cidrs",
]
