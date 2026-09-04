"""SQLite-backed persistence for reusable O3 overlap layouts.

The store is deliberately independent of Gradio and of the model runtime. It
is shared by the display and optional manual-authoring components, so all
mutations use a short, explicit SQLite transaction and optimistic revisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from numbers import Integral, Real
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# 32 MiB is large enough for the board images used by the demo while keeping
# accidental database growth bounded.
MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_LAYERS = 32
MIN_LAYERS = 1
BUSY_TIMEOUT_MS = 30_000

_LAYER_TYPES = frozenset(("BSM", "ACT", "GE1", "GE2"))
_TRANSFORM_FIELDS = (
    "center_x",
    "center_y",
    "pivot_x",
    "pivot_y",
    "scale",
    "rotation_deg",
    "preview_alpha",
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


class OverlapStoreError(Exception):
    """Base class for errors raised by :class:`OverlapLayoutStore`."""


class NotFoundError(OverlapStoreError):
    """The requested layout does not exist in the requested visibility set."""


class ConflictError(OverlapStoreError):
    """An optimistic revision check failed."""

    def __init__(
        self,
        message: str,
        *,
        expected_revision: Optional[int] = None,
        current_revision: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.expected_revision = expected_revision
        self.current_revision = current_revision


class ValidationError(OverlapStoreError):
    """The caller supplied a value outside the store's data contract."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _normalise_name(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field} must not be empty")
    if len(value) > 128:
        raise ValidationError(f"{field} must be at most 128 characters")
    return value


def _normalise_png(value: Any, field: str) -> bytes:
    if isinstance(value, memoryview):
        value = value.tobytes()
    elif isinstance(value, bytearray):
        value = bytes(value)
    elif isinstance(value, bytes):
        # A memoryview round-trip guarantees that a returned BLOB is not an
        # alias of a caller-owned mutable buffer in implementations that use
        # a bytes subclass.
        value = memoryview(value).tobytes()
    else:
        raise ValidationError(f"{field} must be bytes-like")
    if not value:
        raise ValidationError(f"{field} must not be empty")
    if len(value) > MAX_PNG_BYTES:
        raise ValidationError(
            f"{field} exceeds the maximum PNG size of {MAX_PNG_BYTES} bytes"
        )
    if not value.startswith(PNG_SIGNATURE):
        raise ValidationError(f"{field} does not have a valid PNG signature")
    return value


def _normalise_dimension(value: Any, field: str) -> int:
    if not _is_positive_integer(value):
        raise ValidationError(f"{field} must be a positive integer")
    return int(value)


def _finite_number(value: Any, field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"transform {field} must be a finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValidationError(f"transform {field} must be a finite number")
    return value


def _normalise_transform(value: Any, field: str) -> Tuple[Dict[str, Any], str]:
    """Validate and canonicalise a transform, omitting the derived matrix."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field} must contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be an object")

    # Copy through a plain dict and drop the derived value before serialising.
    # This also prevents a caller's mutable mapping from being retained.
    transform = {key: item for key, item in value.items() if key != "matrix_2x3"}
    version = transform.get("transform_version")
    if not isinstance(version, Integral) or isinstance(version, bool) or version != 2:
        raise ValidationError("transform transform_version must be exactly 2")
    for key in _TRANSFORM_FIELDS:
        if key not in transform:
            raise ValidationError(f"transform {key} is required")
        _finite_number(transform[key], key)
    if transform["scale"] <= 0:
        raise ValidationError("transform scale must be greater than zero")

    try:
        encoded = json.dumps(
            transform,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field} must be JSON serialisable") from exc
    if not isinstance(canonical, dict):  # pragma: no cover - guarded above
        raise ValidationError(f"{field} must be an object")
    return canonical, encoded


def _normalise_revision(value: Any) -> int:
    if not _is_positive_integer(value):
        raise ValidationError("expected_revision must be a positive integer")
    return int(value)


class OverlapLayoutStore:
    """A small, process-shared SQLite gallery of complete overlap layouts."""

    board_type = "O3"

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS layouts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            board_type TEXT NOT NULL DEFAULT 'O3'
                CHECK (board_type = 'O3'),
            base_image_png BLOB NOT NULL,
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            revision INTEGER NOT NULL CHECK (revision > 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS layers (
            id TEXT PRIMARY KEY,
            layout_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            layer_type TEXT NOT NULL
                CHECK (layer_type IN ('BSM', 'ACT', 'GE1', 'GE2')),
            name TEXT NOT NULL,
            mask_png BLOB NOT NULL,
            mask_width INTEGER NOT NULL CHECK (mask_width > 0),
            mask_height INTEGER NOT NULL CHECK (mask_height > 0),
            mask_sha256 TEXT NOT NULL CHECK (length(mask_sha256) = 64),
            transform_json TEXT NOT NULL,
            UNIQUE (layout_id, ordinal),
            FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_layers_layout_ordinal "
        "ON layers(layout_id, ordinal)",
        "CREATE INDEX IF NOT EXISTS idx_layouts_updated_name "
        "ON layouts(updated_at DESC, name ASC)",
    )

    def __init__(self, db_path: Any) -> None:
        try:
            path = os.fspath(db_path)
        except TypeError as exc:
            raise ValidationError("db_path must be a filesystem path") from exc
        self._lock = threading.RLock()
        self._closed = False
        self._conn = None  # type: ignore[assignment]
        try:
            self._conn = sqlite3.connect(
                path,
                timeout=BUSY_TIMEOUT_MS / 1000.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._initialise_connection()
            self._initialise_schema()
        except OverlapStoreError:
            self.close()
            raise
        except sqlite3.Error as exc:
            self.close()
            raise OverlapStoreError("could not initialise overlap store") from exc

    def _initialise_connection(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            # WAL is persistent for file-backed databases and harmlessly falls
            # back to memory mode for :memory: stores.
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def _initialise_schema(self) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                for statement in self._SCHEMA:
                    self._conn.execute(statement)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise OverlapStoreError("could not initialise overlap schema") from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise OverlapStoreError("overlap store is closed")

    def _rollback_quietly(self) -> None:
        try:
            self._conn.rollback()
        except sqlite3.Error:
            pass

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        self._ensure_open()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise OverlapStoreError("could not begin overlap write") from exc
            try:
                yield
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise OverlapStoreError("overlap write failed") from exc
            except Exception:
                self._rollback_quietly()
                raise
            else:
                try:
                    self._conn.commit()
                except sqlite3.Error as exc:
                    self._rollback_quietly()
                    raise OverlapStoreError("could not commit overlap write") from exc

    @contextmanager
    def _read_transaction(self) -> Iterator[None]:
        self._ensure_open()
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield
                self._conn.commit()
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise OverlapStoreError("overlap read failed") from exc
            except Exception:
                self._rollback_quietly()
                raise

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    def _normalise_layers(
        self,
        layers: Any,
        *,
        existing_layer_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(layers, (str, bytes, bytearray)) or not isinstance(
            layers, Sequence
        ):
            raise ValidationError("layers must be a sequence")
        if not MIN_LAYERS <= len(layers) <= MAX_LAYERS:
            raise ValidationError("layers must contain between 1 and 32 entries")

        entries = list(layers)
        supplied_ordinals = [
            entry.get("ordinal") if isinstance(entry, Mapping) else None
            for entry in entries
        ]
        has_ordinal = [
            isinstance(entry, Mapping) and "ordinal" in entry for entry in entries
        ]
        if any(has_ordinal) and not all(has_ordinal):
            raise ValidationError("every layer must provide ordinal, or none may")
        if all(has_ordinal):
            if not all(_is_nonnegative_integer(value) for value in supplied_ordinals):
                raise ValidationError("layer ordinal must be a non-negative integer")
            if sorted(int(value) for value in supplied_ordinals) != list(
                range(len(entries))
            ):
                raise ValidationError("layer ordinals must be contiguous from zero")
            entries = [
                entry
                for _, entry in sorted(
                    zip(supplied_ordinals, entries), key=lambda item: int(item[0])
                )
            ]

        existing_layer_ids = existing_layer_ids or set()
        used_input_ids: Set[str] = set()
        prepared: List[Dict[str, Any]] = []
        for ordinal, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValidationError(f"layer {ordinal} must be an object")

            supplied_id = entry.get("id")
            if supplied_id is not None:
                if not isinstance(supplied_id, str) or not supplied_id:
                    raise ValidationError(f"layer {ordinal} id must be a string")
                if supplied_id in used_input_ids:
                    raise ValidationError("layer ids must be unique")
                used_input_ids.add(supplied_id)
            if supplied_id in existing_layer_ids:
                layer_id = supplied_id
            else:
                layer_id = self._new_id()

            layer_type = entry.get("layer_type")
            if not isinstance(layer_type, str) or layer_type not in _LAYER_TYPES:
                raise ValidationError(
                    f"layer {ordinal} layer_type must be one of "
                    "BSM, ACT, GE1, GE2"
                )
            layer_name = _normalise_name(entry.get("name"), f"layer {ordinal} name")
            mask_png = _normalise_png(entry.get("mask_png"), f"layer {ordinal} mask_png")
            mask_width = _normalise_dimension(
                entry.get("mask_width"), f"layer {ordinal} mask_width"
            )
            mask_height = _normalise_dimension(
                entry.get("mask_height"), f"layer {ordinal} mask_height"
            )

            calculated_hash = hashlib.sha256(mask_png).hexdigest()
            supplied_hash = entry.get("mask_sha256")
            if supplied_hash is None:
                mask_sha256 = calculated_hash
            elif not isinstance(supplied_hash, str) or not _SHA256_RE.fullmatch(
                supplied_hash
            ):
                raise ValidationError(
                    f"layer {ordinal} mask_sha256 must be 64 hexadecimal characters"
                )
            else:
                mask_sha256 = supplied_hash.lower()
                if mask_sha256 != calculated_hash:
                    raise ValidationError(
                        f"layer {ordinal} mask_sha256 does not match mask_png"
                    )

            transform_value = entry.get("transform", entry.get("transform_json"))
            transform, transform_json = _normalise_transform(
                transform_value, f"layer {ordinal} transform"
            )
            prepared.append(
                {
                    "id": layer_id,
                    "ordinal": ordinal,
                    "layer_type": layer_type,
                    "name": layer_name,
                    "mask_png": mask_png,
                    "mask_width": mask_width,
                    "mask_height": mask_height,
                    "mask_sha256": mask_sha256,
                    "transform": transform,
                    "transform_json": transform_json,
                }
            )
        return prepared

    def _normalise_layout_values(
        self,
        name: Any,
        base_image_png: Any,
        width: Any,
        height: Any,
    ) -> Tuple[str, bytes, int, int]:
        return (
            _normalise_name(name, "name"),
            _normalise_png(base_image_png, "base_image_png"),
            _normalise_dimension(width, "width"),
            _normalise_dimension(height, "height"),
        )

    @staticmethod
    def _snapshot_from_rows(
        layout_row: sqlite3.Row, layer_rows: Sequence[sqlite3.Row]
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "id": str(layout_row["id"]),
            "name": str(layout_row["name"]),
            "board_type": str(layout_row["board_type"]),
            "base_image_png": memoryview(layout_row["base_image_png"]).tobytes(),
            "width": int(layout_row["width"]),
            "height": int(layout_row["height"]),
            "revision": int(layout_row["revision"]),
            "created_at": str(layout_row["created_at"]),
            "updated_at": str(layout_row["updated_at"]),
            "deleted_at": layout_row["deleted_at"],
            "layers": [],
        }
        for row in layer_rows:
            try:
                transform = json.loads(row["transform_json"])
            except (TypeError, ValueError) as exc:  # pragma: no cover - DB invariant
                raise OverlapStoreError("stored transform JSON is invalid") from exc
            snapshot["layers"].append(
                {
                    "id": str(row["id"]),
                    "ordinal": int(row["ordinal"]),
                    "layer_type": str(row["layer_type"]),
                    "name": str(row["name"]),
                    "mask_png": memoryview(row["mask_png"]).tobytes(),
                    "mask_width": int(row["mask_width"]),
                    "mask_height": int(row["mask_height"]),
                    "mask_sha256": str(row["mask_sha256"]),
                    "transform": transform,
                }
            )
        return snapshot

    def _fetch_snapshot_locked(
        self, layout_id: str, *, include_deleted: bool
    ) -> Dict[str, Any]:
        layout_row = self._conn.execute(
            "SELECT id, name, board_type, base_image_png, width, height, revision, "
            "created_at, updated_at, deleted_at FROM layouts WHERE id = ?",
            (layout_id,),
        ).fetchone()
        if layout_row is None:
            raise NotFoundError(f"layout {layout_id!r} was not found")
        if layout_row["deleted_at"] is not None and not include_deleted:
            raise NotFoundError(f"layout {layout_id!r} was not found")
        layer_rows = self._conn.execute(
            "SELECT id, ordinal, layer_type, name, mask_png, mask_width, "
            "mask_height, mask_sha256, transform_json FROM layers "
            "WHERE layout_id = ? ORDER BY ordinal ASC",
            (layout_id,),
        ).fetchall()
        return self._snapshot_from_rows(layout_row, layer_rows)

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "board_type": str(row["board_type"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "revision": int(row["revision"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "deleted_at": row["deleted_at"],
        }

    def create_layout(
        self,
        name: Any,
        base_image_png: Any,
        width: Any,
        height: Any,
        layers: Any,
    ) -> Dict[str, Any]:
        """Create and return a complete, revision-1 layout snapshot."""

        name, base_image_png, width, height = self._normalise_layout_values(
            name, base_image_png, width, height
        )
        prepared_layers = self._normalise_layers(layers)
        layout_id = self._new_id()
        now = _utc_timestamp()
        with self._write_transaction():
            self._conn.execute(
                "INSERT INTO layouts (id, name, board_type, base_image_png, width, "
                "height, revision, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, 'O3', ?, ?, ?, 1, ?, ?, NULL)",
                (layout_id, name, base_image_png, width, height, now, now),
            )
            self._conn.executemany(
                "INSERT INTO layers (id, layout_id, ordinal, layer_type, name, "
                "mask_png, mask_width, mask_height, mask_sha256, transform_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        layer["id"],
                        layout_id,
                        layer["ordinal"],
                        layer["layer_type"],
                        layer["name"],
                        layer["mask_png"],
                        layer["mask_width"],
                        layer["mask_height"],
                        layer["mask_sha256"],
                        layer["transform_json"],
                    )
                    for layer in prepared_layers
                ],
            )
            return self._fetch_snapshot_locked(layout_id, include_deleted=True)

    def list_layouts(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """Return non-BLOB summaries ordered by most recently updated first."""

        if not isinstance(include_deleted, bool):
            raise ValidationError("include_deleted must be a boolean")
        self._ensure_open()
        with self._read_transaction():
            if include_deleted:
                rows = self._conn.execute(
                    "SELECT id, name, board_type, width, height, revision, "
                    "created_at, updated_at, deleted_at FROM layouts "
                    "ORDER BY updated_at DESC, name ASC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, name, board_type, width, height, revision, "
                    "created_at, updated_at, deleted_at FROM layouts "
                    "WHERE deleted_at IS NULL ORDER BY updated_at DESC, name ASC"
                ).fetchall()
            return [self._summary_from_row(row) for row in rows]

    def get_layout(self, layout_id: Any, include_deleted: bool = False) -> Dict[str, Any]:
        """Return a complete layout snapshot, including BLOBs and layers."""

        if not isinstance(layout_id, str) or not layout_id:
            raise ValidationError("layout_id must be a non-empty string")
        if not isinstance(include_deleted, bool):
            raise ValidationError("include_deleted must be a boolean")
        with self._read_transaction():
            return self._fetch_snapshot_locked(
                layout_id, include_deleted=include_deleted
            )

    def update_layout(
        self,
        layout_id: Any,
        expected_revision: Any,
        *,
        name: Any,
        base_image_png: Any,
        width: Any,
        height: Any,
        layers: Any,
    ) -> Dict[str, Any]:
        """Atomically replace a layout when its revision is current."""

        if not isinstance(layout_id, str) or not layout_id:
            raise ValidationError("layout_id must be a non-empty string")
        expected_revision = _normalise_revision(expected_revision)
        name, base_image_png, width, height = self._normalise_layout_values(
            name, base_image_png, width, height
        )
        with self._write_transaction():
            current = self._conn.execute(
                "SELECT revision, deleted_at FROM layouts WHERE id = ?",
                (layout_id,),
            ).fetchone()
            if current is None or current["deleted_at"] is not None:
                raise NotFoundError(f"layout {layout_id!r} was not found")
            current_revision = int(current["revision"])
            if current_revision != expected_revision:
                raise ConflictError(
                    f"layout {layout_id!r} revision conflict: expected "
                    f"{expected_revision}, current {current_revision}",
                    expected_revision=expected_revision,
                    current_revision=current_revision,
                )
            existing_ids = {
                str(row["id"])
                for row in self._conn.execute(
                    "SELECT id FROM layers WHERE layout_id = ?", (layout_id,)
                ).fetchall()
            }
            prepared_layers = self._normalise_layers(
                layers, existing_layer_ids=existing_ids
            )
            now = _utc_timestamp()
            updated = self._conn.execute(
                "UPDATE layouts SET name = ?, base_image_png = ?, width = ?, "
                "height = ?, revision = revision + 1, updated_at = ? "
                "WHERE id = ? AND revision = ? AND deleted_at IS NULL",
                (
                    name,
                    base_image_png,
                    width,
                    height,
                    now,
                    layout_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise ConflictError(
                    f"layout {layout_id!r} changed during update",
                    expected_revision=expected_revision,
                )
            self._conn.execute("DELETE FROM layers WHERE layout_id = ?", (layout_id,))
            self._conn.executemany(
                "INSERT INTO layers (id, layout_id, ordinal, layer_type, name, "
                "mask_png, mask_width, mask_height, mask_sha256, transform_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        layer["id"],
                        layout_id,
                        layer["ordinal"],
                        layer["layer_type"],
                        layer["name"],
                        layer["mask_png"],
                        layer["mask_width"],
                        layer["mask_height"],
                        layer["mask_sha256"],
                        layer["transform_json"],
                    )
                    for layer in prepared_layers
                ],
            )
            return self._fetch_snapshot_locked(layout_id, include_deleted=True)

    def delete_layout(
        self, layout_id: Any, expected_revision: Any
    ) -> Dict[str, Any]:
        """Soft-delete a layout and return its deleted complete snapshot."""

        if not isinstance(layout_id, str) or not layout_id:
            raise ValidationError("layout_id must be a non-empty string")
        expected_revision = _normalise_revision(expected_revision)
        with self._write_transaction():
            current = self._conn.execute(
                "SELECT revision, deleted_at FROM layouts WHERE id = ?",
                (layout_id,),
            ).fetchone()
            if current is None or current["deleted_at"] is not None:
                raise NotFoundError(f"layout {layout_id!r} was not found")
            current_revision = int(current["revision"])
            if current_revision != expected_revision:
                raise ConflictError(
                    f"layout {layout_id!r} revision conflict: expected "
                    f"{expected_revision}, current {current_revision}",
                    expected_revision=expected_revision,
                    current_revision=current_revision,
                )
            now = _utc_timestamp()
            deleted = self._conn.execute(
                "UPDATE layouts SET deleted_at = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ? AND revision = ? "
                "AND deleted_at IS NULL",
                (now, now, layout_id, expected_revision),
            )
            if deleted.rowcount != 1:
                raise ConflictError(
                    f"layout {layout_id!r} changed during delete",
                    expected_revision=expected_revision,
                )
            return self._fetch_snapshot_locked(layout_id, include_deleted=True)

    def close(self) -> None:
        """Close the SQLite connection; repeated calls are harmless."""

        if getattr(self, "_closed", True):
            return
        with self._lock:
            if not self._closed:
                try:
                    if self._conn is not None:
                        self._conn.close()
                finally:
                    self._closed = True

    def __enter__(self) -> "OverlapLayoutStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "BUSY_TIMEOUT_MS",
    "MAX_LAYERS",
    "MAX_PNG_BYTES",
    "NotFoundError",
    "OverlapLayoutStore",
    "OverlapStoreError",
    "ValidationError",
    "ConflictError",
]
