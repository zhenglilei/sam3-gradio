"""Independent persistence and mask utilities for layout screenshot Regions."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


REGIONS_SCHEMA_VERSION = 1
CATEGORIES_SCHEMA_VERSION = 1
MAX_LASSO_POINTS = 4096
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RegionValidationError(ValueError):
    """Raised when a Region payload or persisted document is invalid."""


class StaleRegionsRevisionError(RegionValidationError):
    """Raised when a save/delete/preview request uses an old revision."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_path_component(value: Any, field: str) -> str:
    text = str(value or "")
    if not _SAFE_COMPONENT_RE.fullmatch(text):
        raise RegionValidationError(f"invalid {field}")
    return text


def mask_pixel_sha256(mask: Any) -> str:
    array = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_layout_categories(path: str | Path) -> list[str]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegionValidationError(f"cannot load layout category config: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CATEGORIES_SCHEMA_VERSION:
        raise RegionValidationError("unsupported layout category schema")
    values = payload.get("categories")
    if not isinstance(values, list) or not values:
        raise RegionValidationError("layout category config must contain a non-empty categories list")
    categories: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RegionValidationError("layout categories must be non-empty strings")
        category = value.strip()
        if category in seen:
            raise RegionValidationError(f"duplicate layout category: {category}")
        categories.append(category)
        seen.add(category)
    return categories


def validate_class_label(value: Any, allowed_categories: Iterable[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegionValidationError("region category is required")
    class_label = value.strip()
    if class_label not in set(allowed_categories):
        raise RegionValidationError(f"region category is not configured: {class_label}")
    return class_label


def normalize_region_name(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RegionValidationError("region name must be a string")
    return value.strip()


def normalize_lasso_points(
    points: Any,
    image_shape: Sequence[int],
    *,
    max_points: int = MAX_LASSO_POINTS,
) -> np.ndarray:
    if not isinstance(points, (list, tuple)):
        raise RegionValidationError("lasso_polygon must be a point list")
    if len(points) > max_points:
        raise RegionValidationError(f"lasso_polygon exceeds {max_points} points")
    if len(image_shape) < 2:
        raise RegionValidationError("invalid source mask shape")
    height, width = int(image_shape[0]), int(image_shape[1])
    if width <= 0 or height <= 0:
        raise RegionValidationError("invalid source mask dimensions")

    cleaned: list[tuple[float, float]] = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RegionValidationError("lasso_polygon points must be [x, y]")
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError) as exc:
            raise RegionValidationError("lasso_polygon coordinates must be numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise RegionValidationError("lasso_polygon coordinates must be finite")
        point = (float(np.clip(x, 0.0, width - 1.0)), float(np.clip(y, 0.0, height - 1.0)))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)

    if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
        cleaned.pop()
    if len(cleaned) < 3 or len(set(cleaned)) < 3:
        raise RegionValidationError("lasso_polygon needs at least three unique points")
    return np.asarray(cleaned, dtype=np.float32)


def rasterize_region_mask(source_mask: Any, points: Any) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    if source.ndim != 2:
        raise RegionValidationError("source mask must be two-dimensional")
    polygon = normalize_lasso_points(points, source.shape)
    polygon_int = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
    lasso_mask = np.zeros(source.shape, dtype=np.uint8)
    cv2.fillPoly(lasso_mask, [polygon_int], 1)
    return np.logical_and(lasso_mask.astype(bool), source)


def encode_binary_mask(mask: Any) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise RegionValidationError("region mask must be two-dimensional")
    encoded = coco_mask.encode(np.asfortranarray(binary))
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": str(counts)}


def decode_binary_mask(rle: Any, expected_shape: Sequence[int] | None = None) -> np.ndarray:
    if not isinstance(rle, dict):
        raise RegionValidationError("mask_rle must be an object")
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or not all(isinstance(v, int) and v > 0 for v in size):
        raise RegionValidationError("mask_rle size is invalid")
    if not isinstance(counts, str) or not counts:
        raise RegionValidationError("mask_rle counts is invalid")
    if expected_shape is not None and [int(expected_shape[0]), int(expected_shape[1])] != size:
        raise RegionValidationError("mask_rle size does not match source mask")
    try:
        decoded = coco_mask.decode({"size": size, "counts": counts.encode("ascii")})
    except Exception as exc:
        raise RegionValidationError(f"cannot decode mask_rle: {exc}") from exc
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    result = np.asarray(decoded, dtype=bool)
    if result.shape != (size[0], size[1]):
        raise RegionValidationError("decoded mask_rle shape is invalid")
    return result


def mask_metadata(mask: Any) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise RegionValidationError("region mask must be two-dimensional")
    area = int(binary.sum())
    if area <= 0:
        raise RegionValidationError("lasso does not contain source mask pixels")
    ys, xs = np.nonzero(binary)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, count):
        x, y, width, height, component_area = stats[label].tolist()
        components.append(
            {
                "id": int(label),
                "area": int(component_area),
                "bbox_xywh": [int(x), int(y), int(width), int(height)],
            }
        )
    return {
        "area": area,
        "bbox_xywh": bbox,
        "component_count": len(components),
        "components": components,
    }


def region_record_from_mask(
    region_id: int,
    class_label: str,
    name: str,
    region_mask: Any,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    if int(region_id) <= 0:
        raise RegionValidationError("region_id must be positive")
    metadata = mask_metadata(region_mask)
    return {
        "region_id": int(region_id),
        "class_label": str(class_label),
        "name": str(name),
        "mask_rle": encode_binary_mask(region_mask),
        **metadata,
        "created_at": created_at or utc_now_iso(),
        "deleted_at": None,
    }


def new_regions_document(session_id: str, layout_id: str, source_mask_hash: str) -> dict[str, Any]:
    return {
        "schema_version": REGIONS_SCHEMA_VERSION,
        "session_id": safe_path_component(session_id, "session_id"),
        "layout_id": safe_path_component(layout_id, "layout_id"),
        "source_mask_hash": str(source_mask_hash),
        "regions_revision": 0,
        "next_region_id": 1,
        "regions": [],
    }


def _metadata_matches(record: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return all(record.get(field) == metadata[field] for field in ("area", "bbox_xywh", "component_count", "components"))


def validate_regions_document(
    payload: Any,
    *,
    session_id: str,
    layout_id: str,
    source_mask_hash: str,
    source_mask: Any,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != REGIONS_SCHEMA_VERSION:
        raise RegionValidationError("unsupported regions schema")
    expected_session = safe_path_component(session_id, "session_id")
    expected_layout = safe_path_component(layout_id, "layout_id")
    if payload.get("session_id") != expected_session or payload.get("layout_id") != expected_layout:
        raise RegionValidationError("regions identity does not match current layout")
    if payload.get("source_mask_hash") != source_mask_hash:
        raise RegionValidationError("regions source mask hash does not match current layout")
    revision = payload.get("regions_revision")
    next_region_id = payload.get("next_region_id")
    regions = payload.get("regions")
    if not isinstance(revision, int) or revision < 0:
        raise RegionValidationError("regions_revision is invalid")
    if not isinstance(next_region_id, int) or next_region_id <= 0:
        raise RegionValidationError("next_region_id is invalid")
    if not isinstance(regions, list):
        raise RegionValidationError("regions must be a list")

    source = np.asarray(source_mask, dtype=bool)
    normalized_regions: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    active_coverage = np.zeros(source.shape, dtype=bool)
    for raw in regions:
        if not isinstance(raw, dict):
            raise RegionValidationError("region record must be an object")
        region_id = raw.get("region_id")
        if not isinstance(region_id, int) or region_id <= 0 or region_id in seen_ids:
            raise RegionValidationError("region_id is invalid or duplicated")
        class_label = raw.get("class_label")
        name = raw.get("name")
        if not isinstance(class_label, str) or not class_label.strip():
            raise RegionValidationError("persisted class_label is invalid")
        if not isinstance(name, str):
            raise RegionValidationError("persisted region name is invalid")
        if not isinstance(raw.get("created_at"), str) or not raw.get("created_at"):
            raise RegionValidationError("persisted created_at is invalid")
        if raw.get("deleted_at") is not None and not isinstance(raw.get("deleted_at"), str):
            raise RegionValidationError("persisted deleted_at is invalid")
        decoded = decode_binary_mask(raw.get("mask_rle"), source.shape)
        if np.any(np.logical_and(decoded, np.logical_not(source))):
            raise RegionValidationError(f"R{region_id} is not a subset of the source mask")
        if raw.get("deleted_at") is None:
            if np.logical_and(decoded, active_coverage).any():
                raise RegionValidationError(
                    f"R{region_id} overlaps another active Region"
                )
            active_coverage = np.logical_or(active_coverage, decoded)
        metadata = mask_metadata(decoded)
        if not _metadata_matches(raw, metadata):
            raise RegionValidationError(f"R{region_id} derived metadata does not match its RLE")
        record = copy.deepcopy(raw)
        record["class_label"] = class_label.strip()
        record["name"] = name.strip()
        normalized_regions.append(record)
        seen_ids.add(region_id)
    if seen_ids and next_region_id <= max(seen_ids):
        raise RegionValidationError("next_region_id would reuse an existing Region id")

    document = copy.deepcopy(payload)
    document["regions"] = normalized_regions
    return document


def active_regions(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [region for region in document.get("regions", []) if region.get("deleted_at") is None]


def active_regions_for_class(
    document: dict[str, Any],
    class_label: Any,
) -> list[dict[str, Any]]:
    """Return active Regions for one persisted class, ordered by Region id."""
    if not isinstance(class_label, str) or not class_label.strip():
        raise RegionValidationError("region category is required")
    target = class_label.strip()
    matches = [
        record
        for record in active_regions(document)
        if record.get("class_label") == target
    ]
    return sorted(matches, key=lambda record: int(record["region_id"]))


def decode_region_masks(
    region_records: Iterable[dict[str, Any]],
    expected_shape: Sequence[int],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Decode Region RLEs independently while preserving record order."""
    return [
        (record, decode_binary_mask(record.get("mask_rle"), expected_shape))
        for record in region_records
    ]


def class_region_preview_mask(
    document: dict[str, Any],
    class_label: Any,
    expected_shape: Sequence[int],
) -> np.ndarray:
    """Build a display-only union of all active Regions in one class."""
    if len(expected_shape) < 2:
        raise RegionValidationError("invalid source mask shape")
    height, width = int(expected_shape[0]), int(expected_shape[1])
    if height <= 0 or width <= 0:
        raise RegionValidationError("invalid source mask dimensions")
    preview = np.zeros((height, width), dtype=bool)
    records = active_regions_for_class(document, class_label)
    for _, mask in decode_region_masks(records, (height, width)):
        preview = np.logical_or(preview, mask)
    return preview


def rasterize_uncovered_region_mask(
    source_mask: Any,
    points: Any,
    document: dict[str, Any],
) -> np.ndarray:
    region_mask = rasterize_region_mask(source_mask, points)
    for record in active_regions(document):
        region_mask = np.logical_and(
            region_mask,
            np.logical_not(decode_binary_mask(record.get("mask_rle"), region_mask.shape)),
        )
    return region_mask


def append_region(
    document: dict[str, Any],
    *,
    class_label: Any,
    name: Any,
    region_mask: Any,
    allowed_categories: Iterable[str],
    created_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(document)
    category = validate_class_label(class_label, allowed_categories)
    normalized_name = normalize_region_name(name)
    region_id = int(updated.get("next_region_id") or 0)
    record = region_record_from_mask(region_id, category, normalized_name, region_mask, created_at=created_at)
    updated.setdefault("regions", []).append(record)
    updated["next_region_id"] = region_id + 1
    updated["regions_revision"] = int(updated.get("regions_revision") or 0) + 1
    return updated, record


def soft_delete_region(
    document: dict[str, Any],
    region_id: int,
    *,
    deleted_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(document)
    for record in updated.get("regions", []):
        if record.get("region_id") == int(region_id) and record.get("deleted_at") is None:
            record["deleted_at"] = deleted_at or utc_now_iso()
            updated["regions_revision"] = int(updated.get("regions_revision") or 0) + 1
            return updated, record
    raise RegionValidationError(f"active Region R{region_id} does not exist")


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    replace: Callable[[str, str], None] = os.replace,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary_path, str(target))
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _draw_region_label(canvas: np.ndarray, mask: np.ndarray, text: str, color: tuple[int, int, int, int]) -> None:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] <= 0:
        return
    x = int(round(moments["m10"] / moments["m00"]))
    y = int(round(moments["m01"] / moments["m00"]))
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 230), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_saved_region_overlay(
    regions: Iterable[dict[str, Any]],
    shape: Sequence[int],
    *,
    selected_region_id: int | None = None,
) -> Image.Image:
    height, width = int(shape[0]), int(shape[1])
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for record in regions:
        if record.get("deleted_at") is not None:
            continue
        mask = decode_binary_mask(record.get("mask_rle"), (height, width))
        canvas[mask, :3] = (35, 200, 85)
        canvas[mask, 3] = np.maximum(canvas[mask, 3], 90)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 4 if int(record.get("region_id")) == int(selected_region_id or -1) else 2
        cv2.drawContours(canvas, contours, -1, (45, 255, 105, 235), thickness, cv2.LINE_AA)
        _draw_region_label(canvas, mask, f"R{int(record['region_id'])}", (75, 255, 125, 255))
    return Image.fromarray(canvas, mode="RGBA")


def render_draft_region_overlay(region_mask: Any) -> Image.Image:
    mask = np.asarray(region_mask, dtype=bool)
    if mask.ndim != 2:
        raise RegionValidationError("draft region mask must be two-dimensional")
    canvas = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    if mask.any():
        canvas[mask, :3] = (255, 210, 0)
        canvas[mask, 3] = 120
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 230, 0, 255), 3, cv2.LINE_AA)
        _draw_region_label(canvas, mask, "Draft", (255, 235, 40, 255))
    return Image.fromarray(canvas, mode="RGBA")


class LayoutRegionStore:
    """Disk-backed Region store isolated by session_id and layout_id."""

    def __init__(
        self,
        *,
        layout_masks_root: str | Path,
        layout_regions_root: str | Path,
        categories_path: str | Path,
    ) -> None:
        self.layout_masks_root = Path(layout_masks_root)
        self.layout_regions_root = Path(layout_regions_root)
        self.categories_path = Path(categories_path)
        self._lock = threading.RLock()

    def _identity(self, session_id: Any, layout_id: Any) -> tuple[str, str]:
        return safe_path_component(session_id, "session_id"), safe_path_component(layout_id, "layout_id")

    def regions_path(self, session_id: Any, layout_id: Any) -> Path:
        sid, lid = self._identity(session_id, layout_id)
        return self.layout_regions_root / sid / lid / "regions.json"

    @contextmanager
    def _layout_write_lock(self, session_id: Any, layout_id: Any):
        sid, lid = self._identity(session_id, layout_id)
        lock_path = self.layout_regions_root / sid / lid / ".regions.lock"
        with self._lock:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_source_mask(
        self,
        session_id: Any,
        layout_id: Any,
        expected_source_mask_hash: str,
    ) -> tuple[np.ndarray, str]:
        sid, lid = self._identity(session_id, layout_id)
        mask_path = self.layout_masks_root / sid / lid / "source_mask.png"
        meta_path = self.layout_masks_root / sid / lid / "layout_meta.json"
        gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RegionValidationError("source_mask.png does not exist or cannot be read")
        source_mask = gray >= 128
        actual_hash = mask_pixel_sha256(source_mask.astype(np.uint8))
        if not expected_source_mask_hash or actual_hash != str(expected_source_mask_hash):
            raise RegionValidationError("source mask hash does not match layout state")
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RegionValidationError(f"cannot read layout_meta.json: {exc}") from exc
            meta_hash = meta.get("source_mask_pixel_sha256")
            if meta_hash and meta_hash != actual_hash:
                raise RegionValidationError("layout_meta.json source mask hash does not match source_mask.png")
        return source_mask, actual_hash

    def load_document(
        self,
        session_id: Any,
        layout_id: Any,
        expected_source_mask_hash: str,
    ) -> tuple[dict[str, Any], np.ndarray]:
        sid, lid = self._identity(session_id, layout_id)
        source_mask, actual_hash = self.load_source_mask(sid, lid, expected_source_mask_hash)
        path = self.regions_path(sid, lid)
        if not path.exists():
            return new_regions_document(sid, lid, actual_hash), source_mask
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegionValidationError(f"cannot read regions.json: {exc}") from exc
        return (
            validate_regions_document(
                payload,
                session_id=sid,
                layout_id=lid,
                source_mask_hash=actual_hash,
                source_mask=source_mask,
            ),
            source_mask,
        )

    @staticmethod
    def _check_revision(document: dict[str, Any], expected_revision: Any) -> None:
        if not isinstance(expected_revision, int) or expected_revision != document.get("regions_revision"):
            raise StaleRegionsRevisionError(
                f"stale regions revision: expected {expected_revision}, current {document.get('regions_revision')}"
            )

    def preview_region(
        self,
        *,
        session_id: Any,
        layout_id: Any,
        source_mask_hash: str,
        expected_revision: int,
        lasso_polygon: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            document, source_mask = self.load_document(session_id, layout_id, source_mask_hash)
            self._check_revision(document, expected_revision)
            region_mask = rasterize_uncovered_region_mask(source_mask, lasso_polygon, document)
            mask_metadata(region_mask)
            return region_mask, document

    def save_region(
        self,
        *,
        session_id: Any,
        layout_id: Any,
        source_mask_hash: str,
        expected_revision: int,
        lasso_polygon: Any,
        class_label: Any,
        name: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._layout_write_lock(session_id, layout_id):
            document, source_mask = self.load_document(session_id, layout_id, source_mask_hash)
            self._check_revision(document, expected_revision)
            region_mask = rasterize_uncovered_region_mask(source_mask, lasso_polygon, document)
            categories = load_layout_categories(self.categories_path)
            updated, record = append_region(
                document,
                class_label=class_label,
                name=name,
                region_mask=region_mask,
                allowed_categories=categories,
            )
            write_json_atomic(self.regions_path(session_id, layout_id), updated)
            return updated, record

    def delete_region(
        self,
        *,
        session_id: Any,
        layout_id: Any,
        source_mask_hash: str,
        expected_revision: int,
        region_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._layout_write_lock(session_id, layout_id):
            document, _ = self.load_document(session_id, layout_id, source_mask_hash)
            self._check_revision(document, expected_revision)
            updated, record = soft_delete_region(document, int(region_id))
            write_json_atomic(self.regions_path(session_id, layout_id), updated)
            return updated, record
