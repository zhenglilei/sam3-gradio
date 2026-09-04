"""Lazy process-wide access to the shared overlap layout repository."""

from __future__ import annotations

import atexit
import threading

from sam3_demo.config import overlap_layout_db_path
from sam3_demo.overlap_service import OverlapPredictionService
from sam3_demo.overlap_store import OverlapLayoutStore


_LOCK = threading.RLock()
_STORE: OverlapLayoutStore | None = None
_SERVICE: OverlapPredictionService | None = None


def get_overlap_store() -> OverlapLayoutStore:
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = OverlapLayoutStore(overlap_layout_db_path)
        return _STORE


def get_overlap_service() -> OverlapPredictionService:
    global _SERVICE
    with _LOCK:
        if _SERVICE is None:
            _SERVICE = OverlapPredictionService(get_overlap_store())
        return _SERVICE


def close_overlap_runtime() -> None:
    global _SERVICE, _STORE
    with _LOCK:
        store = _STORE
        _SERVICE = None
        _STORE = None
        if store is not None:
            store.close()


atexit.register(close_overlap_runtime)


__all__ = [
    "close_overlap_runtime",
    "get_overlap_service",
    "get_overlap_store",
]
