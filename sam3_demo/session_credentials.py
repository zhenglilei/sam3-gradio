"""Bounded, process-local browser credentials; no signing or deployment key."""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable


IDLE_SECONDS = 24 * 60 * 60
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}\Z")


class CredentialCapacityError(ValueError):
    """No expired, safely reclaimable browser credential is available."""


@dataclass
class _Credential:
    owner_id: str
    deployment_id: str
    last_seen: float
    inflight: int = 0


@dataclass(frozen=True)
class CredentialLease:
    digest: bytes
    owner_id: str


class BrowserCredentialRegistry:
    """Store token digests, never raw bearer tokens or client IP identities.

    A lease pins uploads, downloads and SSE through completion/disconnect.
    The owner probe must also retain queued/running work after submission ends.
    Probes run synchronously under this lock; claim code must not call back here.
    """

    def __init__(
        self,
        deployment_id: str,
        *,
        capacity: int = 512,
        idle_seconds: float = IDLE_SECONDS,
        owner_live_probe: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("credential capacity must be a positive integer")
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("credential idle timeout must be positive and finite")
        self.deployment_id = deployment_id
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._owner_live_probe = owner_live_probe
        self._records: dict[bytes, _Credential] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _digest(token: str | None) -> bytes | None:
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            return None
        return hashlib.sha256(token.encode("ascii")).digest()

    def _expired(self, record: _Credential, now: float) -> bool:
        if record.inflight or now - record.last_seen < self.idle_seconds:
            return False
        if self._owner_live_probe is None:
            return False  # Missing or uncertain task liveness fails closed.
        try:
            return self._owner_live_probe(record.owner_id) is False
        except Exception:
            return False

    def acquire(
        self, token: str | None, *, create: bool = False,
    ) -> tuple[CredentialLease, str] | None:
        """Only the homepage may set create=True. Existing credentials are reused."""
        digest = self._digest(token)
        with self._lock:
            now = self._clock()
            record = self._records.get(digest)
            if record is not None and self._expired(record, now):
                del self._records[digest]
                record = None
            if record is None:
                if not create:
                    return None
                if len(self._records) >= self.capacity:
                    for old_digest, old in list(self._records.items()):
                        if self._expired(old, now):
                            del self._records[old_digest]
                if len(self._records) >= self.capacity:
                    raise CredentialCapacityError("browser credential capacity is full")
                while True:
                    token = secrets.token_urlsafe(32)
                    digest = self._digest(token)
                    if digest not in self._records:
                        break
                record = _Credential(secrets.token_urlsafe(32), self.deployment_id, now)
                self._records[digest] = record
            if record.deployment_id != self.deployment_id:
                return None
            record.inflight += 1
            return CredentialLease(digest, record.owner_id), token

    def release(self, lease: CredentialLease, *, renew: bool) -> None:
        with self._lock:
            record = self._records.get(lease.digest)
            if record is None or record.owner_id != lease.owner_id or record.inflight < 1:
                raise RuntimeError("invalid browser credential lease")
            record.inflight -= 1
            if renew:
                record.last_seen = self._clock()

    def reap(self) -> int:
        with self._lock:
            now = self._clock()
            expired = [key for key, record in self._records.items() if self._expired(record, now)]
            for key in expired:
                del self._records[key]
            return len(expired)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"count": len(self._records), "capacity": self.capacity,
                    "inflight": sum(record.inflight for record in self._records.values())}
