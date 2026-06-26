
"""Pure helpers for Layout Mask transform metadata and affine geometry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = _SAFE_ID_RE.sub("_", text)
    return text[:128] or fallback


def mask_pixel_sha256(mask: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_pixel_sha256(image: Image.Image | np.ndarray) -> str:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def foreground_bbox_xyxy(mask: Any) -> list[float]:
    arr = np.asarray(mask, dtype=bool)
    ys, xs = np.where(arr)
    if len(xs) == 0:
        raise ValueError("layout source mask has no foreground pixels")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def pivot_from_bbox_xyxy(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def normalize_rotation_deg(rotation_deg: float) -> float:
    value = (float(rotation_deg) + 180.0) % 360.0 - 180.0
    if value == -180.0:
        return 180.0
    return value


def build_layout_affine_matrix(transform: dict[str, Any]) -> np.ndarray:
    """Build a source->target affine matrix with y-down, clockwise-positive angles."""
    center_x = float(transform["center_x"])
    center_y = float(transform["center_y"])
    pivot_x = float(transform["pivot_x"])
    pivot_y = float(transform["pivot_y"])
    scale = float(transform["scale"])
    rotation_deg = float(transform.get("rotation_deg") or 0.0)
    if not np.isfinite([center_x, center_y, pivot_x, pivot_y, scale, rotation_deg]).all():
        raise ValueError("layout transform contains non-finite values")
    if scale <= 0:
        raise ValueError("layout transform scale must be positive")
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    a = scale * cos_t
    b = scale * sin_t
    return np.asarray(
        [
            [a, -b, center_x - a * pivot_x + b * pivot_y],
            [b, a, center_y - b * pivot_x - a * pivot_y],
        ],
        dtype=np.float32,
    )


def apply_affine_to_point(point_xy: tuple[float, float] | list[float], matrix_2x3: Any) -> tuple[float, float]:
    x, y = float(point_xy[0]), float(point_xy[1])
    m = np.asarray(matrix_2x3, dtype=np.float64)
    return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))


def inverse_transform_point(target_xy: tuple[float, float] | list[float], transform: dict[str, Any]) -> tuple[float, float]:
    matrix = build_layout_affine_matrix(transform)
    inv = cv2.invertAffineTransform(matrix.astype(np.float64))
    return apply_affine_to_point(target_xy, inv)


def warp_layout_mask(source_mask: Any, matrix_2x3: Any, target_size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target size must be positive")
    mask = np.asarray(source_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("source mask must be 2D")
    transformed = cv2.warpAffine(
        mask,
        np.asarray(matrix_2x3, dtype=np.float32),
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed.astype(bool)


def derive_legacy_tx_ty(transform: dict[str, Any], target_size: tuple[int, int]) -> tuple[float, float]:
    target_width, target_height = int(target_size[0]), int(target_size[1])
    return float(transform["center_x"]) - target_width / 2.0, float(transform["center_y"]) - target_height / 2.0


def old_cv2_layout_matrix(old_state: dict[str, Any], source_shape: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    target_w, target_h = int(target_size[0]), int(target_size[1])
    cx, cy = src_w / 2.0, src_h / 2.0
    scale = float(old_state.get("scale") or 1.0)
    rotation_deg = float(old_state.get("rotation_deg") or 0.0)
    tx = float(old_state.get("tx") or 0.0)
    ty = float(old_state.get("ty") or 0.0)
    matrix = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
    matrix[0, 2] += (target_w / 2.0 - cx) + tx
    matrix[1, 2] += (target_h / 2.0 - cy) + ty
    return np.asarray(matrix, dtype=np.float32)


def migrate_layout_transform_v1_to_v2(
    old_state: dict[str, Any],
    source_mask: Any,
    target_size: tuple[int, int],
    *,
    session_id: str,
    layout_id: str,
    image_id: str | None = None,
    target_image_sha256: str | None = None,
    source_mask_pixel_sha256: str | None = None,
    revision: int = 0,
) -> dict[str, Any]:
    mask = np.asarray(source_mask, dtype=bool)
    bbox = foreground_bbox_xyxy(mask)
    pivot_x, pivot_y = pivot_from_bbox_xyxy(bbox)
    old_matrix = old_cv2_layout_matrix(old_state, mask.shape[:2], target_size)
    center_x, center_y = apply_affine_to_point((pivot_x, pivot_y), old_matrix)
    transform = {
        "transform_version": 2,
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "image_id": image_id,
        "revision": int(revision),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "pivot_x": float(pivot_x),
        "pivot_y": float(pivot_y),
        "scale": float(old_state.get("scale") or 1.0),
        # Old cv2 UI was visual counter-clockwise positive; v2 is clockwise positive.
        "rotation_deg": normalize_rotation_deg(-float(old_state.get("rotation_deg") or 0.0)),
        "preview_alpha": float(old_state.get("preview_alpha") or 0.35),
        "source_mask_pixel_sha256": source_mask_pixel_sha256 or mask_pixel_sha256(mask.astype(np.uint8)),
        "target_image_sha256": target_image_sha256,
    }
    return transform


def make_layout_transform_v2(
    *,
    session_id: str,
    layout_id: str,
    image_id: str | None,
    target_size: tuple[int, int],
    source_mask: Any,
    center_x: float | None = None,
    center_y: float | None = None,
    pivot_xy: list[float] | tuple[float, float] | None = None,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    preview_alpha: float = 0.35,
    revision: int = 0,
    source_mask_pixel_sha256: str | None = None,
    target_image_sha256: str | None = None,
) -> dict[str, Any]:
    mask = np.asarray(source_mask, dtype=bool)
    if pivot_xy is None:
        pivot_xy = pivot_from_bbox_xyxy(foreground_bbox_xyxy(mask))
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if center_x is None:
        center_x = target_width / 2.0
    if center_y is None:
        center_y = target_height / 2.0
    transform = {
        "transform_version": 2,
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "image_id": image_id,
        "revision": int(revision),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "pivot_x": float(pivot_xy[0]),
        "pivot_y": float(pivot_xy[1]),
        "scale": float(scale),
        "rotation_deg": normalize_rotation_deg(float(rotation_deg or 0.0)),
        "preview_alpha": float(preview_alpha),
        "source_mask_pixel_sha256": source_mask_pixel_sha256 or mask_pixel_sha256(mask.astype(np.uint8)),
        "target_image_sha256": target_image_sha256,
    }
    build_layout_affine_matrix(transform)
    return transform


def transform_with_derived_fields(transform: dict[str, Any], target_size: tuple[int, int]) -> dict[str, Any]:
    out = json.loads(json.dumps(transform))
    matrix = build_layout_affine_matrix(out)
    tx, ty = derive_legacy_tx_ty(out, target_size)
    out["matrix_2x3"] = matrix.astype(float).tolist()
    out["tx"] = float(tx)
    out["ty"] = float(ty)
    return out
