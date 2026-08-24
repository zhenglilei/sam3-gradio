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
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
SCHEMA_VERSION = 1
_MAX_HASH = 512
_DIGEST_TAG = b"sam3-session-digest-v1"
_TOKEN_TAG = b"sam3-session-owner-v1"
logger = logging.getLogger(__name__)


class SessionError(ValueError):
    """Invalid, stale, or unauthorized session identity."""
class SessionExpired(SessionError):
    """A previously valid session is closed or has expired."""
@dataclass
class SessionRecord:
    session_id: str
    session_hash_digest: str
    client_ip: str                 # raw address is memory-only
    client_ip_digest: str
    generation: int
    owner_token: str
    created_at: float
    last_seen: float
    in_flight: int = 0
    close_requested: bool = False
    closed: bool = False
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
def _digest(value: str, secret: bytes) -> str:
    return hmac.new(secret, _DIGEST_TAG + value.encode(), hashlib.sha256).hexdigest()
def _owner_token(secret: bytes, record: SessionRecord) -> str:
    text = "|".join(
        (record.session_id, str(record.generation), record.session_hash_digest, record.client_ip_digest)
    ).encode("ascii")
    return hmac.new(secret, _TOKEN_TAG + text, hashlib.sha256).hexdigest()
class SessionRegistry:
    """Thread-safe registry keyed by (session-hash digest, normalized IP)."""
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        idle_seconds: float = 3600,
        max_sessions: int = 1000,
        cleanup_callback: Optional[Callable[[SessionRecord], None]] = None,
        secret: Optional[bytes | str] = None,
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
            "session_hash_digest": record.session_hash_digest,
            "client_ip_digest": record.client_ip_digest,
        }
    def _expired(self, record: SessionRecord, now: float) -> bool:
        return not record.close_requested and record.in_flight == 0 and now - record.last_seen >= self.idle_seconds
    def _remove(self, record: SessionRecord) -> Optional[SessionRecord]:
        if record.closed:
            return None
        record.close_requested = True
        key = (record.session_hash_digest, record.client_ip)
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
    def _state_fields(state: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
        if not isinstance(state, Mapping):
            raise SessionError("session state must be a mapping")
        try:
            version = state["schema_version"]
            session_id = state["session_id"]
            generation = state["generation"]
            token = state["owner_token"]
            hash_digest = state["session_hash_digest"]
            ip_digest = state["client_ip_digest"]
        except (KeyError, TypeError) as exc:
            raise SessionError("incomplete session state") from exc
        if version != SCHEMA_VERSION or not isinstance(session_id, str) or not session_id:
            raise SessionError("invalid session state schema or id")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise SessionError("invalid session generation")
        if not all(isinstance(value, str) for value in (token, hash_digest, ip_digest)):
            raise SessionError("invalid session state fields")
        return session_id, generation, token, hash_digest, ip_digest
    def _record_from_state(self, state: Mapping[str, Any]) -> SessionRecord:
        session_id, generation, token, hash_digest, ip_digest = self._state_fields(state)
        record = self._records.get(session_id)
        if record is None or record.closed or record.close_requested:
            raise SessionExpired("session is closed or unknown")
        if record.generation != generation or record.session_hash_digest != hash_digest or record.client_ip_digest != ip_digest:
            raise SessionError("session state does not match record")
        if not hmac.compare_digest(token, _owner_token(self._secret, record)):
            raise SessionError("invalid session owner token")
        return record
    def _record_for_identity(self, state: Mapping[str, Any], session_hash: str, client_ip: str) -> SessionRecord:
        record = self._record_from_state(state)
        if record.session_hash_digest != _digest(_session_hash(session_hash), self._secret):
            raise SessionError("session hash does not match owner")
        if record.client_ip != _ip(client_ip):
            raise SessionError("client IP does not match owner")
        return record
    def _touch(self, record: SessionRecord, now: float) -> tuple[Optional[SessionRecord], list[SessionRecord]]:
        if self._expired(record, now):
            removed = self._remove(record)
            return None, [removed] if removed is not None else []
        record.last_seen = now
        return record, []
    def bind(self, session_hash: str, client_ip: str) -> dict[str, Any]:
        hash_value = _session_hash(session_hash)
        ip_value = _ip(client_ip)
        hash_digest = _digest(hash_value, self._secret)
        now = self._clock()
        cleanups = self._drop_idle(now)
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionError("session registry is shut down")
                key = (hash_digest, ip_value)
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
                        session_hash_digest=hash_digest,
                        client_ip=ip_value,
                        client_ip_digest=_digest(ip_value, self._secret),
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
    def validate(self, state: Mapping[str, Any], session_hash: str, client_ip: str) -> SessionRecord:
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionExpired("session registry is shut down")
                record = self._record_for_identity(state, session_hash, client_ip)
                record, cleanups = self._touch(record, self._clock())
                if record is None:
                    raise SessionExpired("session has expired")
        finally:
            self._callbacks(cleanups)
        return record
    def validate_owner(self, session_id: str, session_hash: str, client_ip: str) -> SessionRecord:
        if not isinstance(session_id, str) or not session_id:
            raise SessionError("session_id must be a non-empty string")
        hash_digest = _digest(_session_hash(session_hash), self._secret)
        ip_value = _ip(client_ip)
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    raise SessionExpired("session registry is shut down")
                record = self._records.get(session_id)
                if record is None or record.closed or record.close_requested:
                    raise SessionExpired("session is closed or unknown")
                if record.session_hash_digest != hash_digest or record.client_ip != ip_value:
                    raise SessionError("session owner does not match")
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
    @contextmanager
    def lease(self, state: Mapping[str, Any], session_hash: str, client_ip: str) -> Iterator[SessionRecord]:
        with self._lease_record(self.validate(state, session_hash, client_ip)) as record:
            yield record
    @contextmanager
    def owner_lease(self, session_id: str, session_hash: str, client_ip: str) -> Iterator[SessionRecord]:
        with self._lease_record(self.validate_owner(session_id, session_hash, client_ip)) as record:
            yield record
    def _close(self, state: Mapping[str, Any], session_hash: Optional[str] = None, client_ip: Optional[str] = None) -> bool:
        cleanups: list[SessionRecord] = []
        try:
            with self._lock:
                if self._shutdown:
                    return False
                try:
                    record = self._record_from_state(state)
                    if session_hash is not None or client_ip is not None:
                        if session_hash is None or client_ip is None:
                            return False
                        record = self._record_for_identity(state, session_hash, client_ip)
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
    def close(self, state: Mapping[str, Any], session_hash: Optional[str] = None, client_ip: Optional[str] = None) -> bool:
        return self._close(state, session_hash, client_ip)
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
                self._keys.pop((record.session_hash_digest, record.client_ip), None)
                if record.in_flight == 0:
                    self._records.pop(record.session_id, None)
                    record.closed = True
                    cleanups.append(record)
        self._callbacks(cleanups)
__all__ = [
    "SCHEMA_VERSION",
    "SessionError",
    "SessionExpired",
    "SessionRecord",
    "SessionRegistry",
    "resolve_client_ip",
    "validate_trusted_proxy_cidrs",
]
