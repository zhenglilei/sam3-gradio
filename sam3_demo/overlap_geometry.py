"""Geometry helpers for authored multi-layer overlap layouts.

The authored transforms are the source of truth. When a current layer is
observed at a different pose, the same canvas-space delta is propagated to the
next layer before both masks are warped and intersected.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from layout_transform_utils import (
    build_layout_affine_matrix,
    normalize_rotation_deg,
    warp_layout_mask,
)


DEFAULT_LAYER_ORDER = ("BSM", "ACT", "GE1", "GE2")


class OverlapGeometryError(ValueError):
    """Raised when an authored layer or transform is geometrically invalid."""


def default_layer_ordinal(layer_type: str) -> int:
    """Return the nominal process order for a supported layer type."""
    normalized = str(layer_type or "").strip().upper()
    try:
        return DEFAULT_LAYER_ORDER.index(normalized)
    except ValueError as exc:
        raise OverlapGeometryError(f"unsupported layer type: {layer_type!r}") from exc


def ordered_layers(layers: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and sort authored layers by their explicit order."""
    if not layers:
        raise OverlapGeometryError("at least one layer is required")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    ordinals: set[int] = set()
    for raw in layers:
        if not isinstance(raw, dict):
            raise OverlapGeometryError("each layer must be an object")
        layer_id = str(raw.get("id") or raw.get("layer_id") or "").strip()
        if not layer_id or layer_id in ids:
            raise OverlapGeometryError("layer ids must be non-empty and unique")
        try:
            ordinal = int(raw["ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OverlapGeometryError("layer ordinal must be an integer") from exc
        if ordinal < 0 or ordinal in ordinals:
            raise OverlapGeometryError("layer ordinals must be unique and non-negative")
        layer_type = str(raw.get("layer_type") or "").strip()
        if not layer_type:
            raise OverlapGeometryError("layer type must not be empty")
        item = dict(raw)
        item["id"] = layer_id
        item["ordinal"] = ordinal
        item["layer_type"] = layer_type
        validated.append(item)
        ids.add(layer_id)
        ordinals.add(ordinal)
    validated.sort(key=lambda item: item["ordinal"])
    actual = [item["ordinal"] for item in validated]
    if actual != list(range(len(validated))):
        raise OverlapGeometryError(
            f"layer ordinals must be contiguous from zero; got {actual}"
        )
    return validated


def current_and_next_layer(
    layers: Sequence[dict[str, Any]], current_layer_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the selected layer and its authored successor."""
    ordered = ordered_layers(layers)
    selected = str(current_layer_id or "").strip()
    for index, layer in enumerate(ordered):
        if layer["id"] != selected:
            continue
        if index + 1 >= len(ordered):
            raise OverlapGeometryError("the selected layer has no next layer")
        return layer, ordered[index + 1]
    raise OverlapGeometryError(f"unknown current layer id: {current_layer_id!r}")


def _homogeneous(matrix_2x3: Any) -> np.ndarray:
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise OverlapGeometryError("affine matrix must be finite with shape (2, 3)")
    return np.vstack([matrix, np.asarray([0.0, 0.0, 1.0])])


def _validated_transform(transform: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(transform, dict):
        raise OverlapGeometryError("layer transform must be an object")
    try:
        matrix = build_layout_affine_matrix(transform)
    except (KeyError, TypeError, ValueError) as exc:
        raise OverlapGeometryError(f"invalid layer transform: {exc}") from exc
    out = dict(transform)
    out["matrix_2x3"] = np.asarray(matrix, dtype=float).tolist()
    return out


def predict_next_transform(
    reference_current_transform: dict[str, Any],
    reference_next_transform: dict[str, Any],
    observed_current_transform: dict[str, Any],
) -> dict[str, Any]:
    """Propagate the current layer canvas-space pose delta to the next layer.

    With authored current/next matrices Rc/Rn and observed current matrix Oc,
    the predicted next matrix is Oc multiplied by inverse(Rc) and then Rn.
    """
    reference_current = _validated_transform(reference_current_transform)
    reference_next = _validated_transform(reference_next_transform)
    observed_current = _validated_transform(observed_current_transform)
    ref_current = _homogeneous(reference_current["matrix_2x3"])
    ref_next = _homogeneous(reference_next["matrix_2x3"])
    observed = _homogeneous(observed_current["matrix_2x3"])
    try:
        predicted = observed @ np.linalg.inv(ref_current) @ ref_next
    except np.linalg.LinAlgError as exc:
        raise OverlapGeometryError("current reference transform is singular") from exc

    linear = predicted[:2, :2]
    scale = float(math.hypot(linear[0, 0], linear[1, 0]))
    if not math.isfinite(scale) or scale <= 0:
        raise OverlapGeometryError("predicted next-layer scale is invalid")
    expected = np.asarray(
        [[linear[0, 0], -linear[1, 0]], [linear[1, 0], linear[0, 0]]],
        dtype=np.float64,
    )
    if not np.allclose(linear, expected, rtol=1e-5, atol=1e-6):
        raise OverlapGeometryError("predicted transform is not a similarity transform")

    pivot = np.asarray(
        [
            float(reference_next["pivot_x"]),
            float(reference_next["pivot_y"]),
            1.0,
        ],
        dtype=np.float64,
    )
    center = predicted @ pivot
    out = {
        key: value
        for key, value in reference_next.items()
        if key not in {"matrix_2x3", "tx", "ty"}
    }
    out["center_x"] = float(center[0])
    out["center_y"] = float(center[1])
    out["scale"] = scale
    out["rotation_deg"] = normalize_rotation_deg(
        math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    )
    out["predicted_from_current"] = True
    out["matrix_2x3"] = predicted[:2, :].astype(float).tolist()
    return out


def normalize_binary_mask(mask: Any, *, name: str = "mask") -> np.ndarray:
    """Normalize a grayscale/RGB/PIL annotation into a non-empty bool mask."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise OverlapGeometryError(f"{name} must have 1, 3, or 4 channels")
        array = np.max(array[:, :, :3], axis=2)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise OverlapGeometryError(f"{name} must be a non-empty 2D mask")
    normalized = np.ascontiguousarray(array > 0)
    if not normalized.any():
        raise OverlapGeometryError(f"{name} has no foreground pixels")
    return normalized


def warp_layer_mask(
    source_mask: Any,
    transform: dict[str, Any],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Warp one authored layer into the common O3 canvas."""
    mask = normalize_binary_mask(source_mask)
    validated = _validated_transform(transform)
    return warp_layout_mask(mask, validated["matrix_2x3"], canvas_size)


def overlap_metrics(current_mask: Any, next_mask: Any) -> dict[str, Any]:
    """Compute pairwise intersection statistics for two canvas-space masks."""
    current = np.asarray(current_mask, dtype=bool)
    following = np.asarray(next_mask, dtype=bool)
    if current.ndim != 2 or current.shape != following.shape:
        raise OverlapGeometryError("warped masks must be 2D with identical shapes")
    intersection = current & following
    union = current | following
    current_area = int(current.sum())
    next_area = int(following.sum())
    intersection_area = int(intersection.sum())
    union_area = int(union.sum())
    ys, xs = np.where(intersection)
    bbox = (
        None
        if len(xs) == 0
        else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    )
    return {
        "current_area": current_area,
        "next_area": next_area,
        "intersection_area": intersection_area,
        "union_area": union_area,
        "iou": intersection_area / union_area if union_area else 0.0,
        "current_coverage": intersection_area / current_area if current_area else 0.0,
        "next_coverage": intersection_area / next_area if next_area else 0.0,
        "intersection_bbox_xyxy": bbox,
    }


def calculate_next_layer_overlap(
    layers: Sequence[dict[str, Any]],
    current_layer_id: str,
    canvas_size: tuple[int, int],
    *,
    observed_current_transform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict the next layer and calculate its overlap with the current layer."""
    current, following = current_and_next_layer(layers, current_layer_id)
    try:
        current_mask = current["mask"]
        next_mask = following["mask"]
        reference_current = current["transform"]
        reference_next = following["transform"]
    except KeyError as exc:
        raise OverlapGeometryError(f"layer is missing required field: {exc.args[0]}") from exc

    active_current = (
        _validated_transform(observed_current_transform)
        if observed_current_transform is not None
        else _validated_transform(reference_current)
    )
    predicted_next = (
        predict_next_transform(reference_current, reference_next, active_current)
        if observed_current_transform is not None
        else _validated_transform(reference_next)
    )
    warped_current = warp_layer_mask(current_mask, active_current, canvas_size)
    warped_next = warp_layer_mask(next_mask, predicted_next, canvas_size)
    intersection = warped_current & warped_next
    return {
        "current_layer": {key: value for key, value in current.items() if key != "mask"},
        "next_layer": {key: value for key, value in following.items() if key != "mask"},
        "current_transform": active_current,
        "predicted_next_transform": predicted_next,
        "current_mask": warped_current,
        "next_mask": warped_next,
        "intersection_mask": intersection,
        "metrics": overlap_metrics(warped_current, warped_next),
    }


def render_overlap_preview(
    base_image: Image.Image | np.ndarray,
    current_mask: Any,
    next_mask: Any,
    *,
    current_color: tuple[int, int, int] = (0, 183, 235),
    next_color: tuple[int, int, int] = (211, 73, 177),
    intersection_color: tuple[int, int, int] = (255, 205, 48),
) -> Image.Image:
    """Render current/next contours and a strongly highlighted intersection."""
    if isinstance(base_image, Image.Image):
        base = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    else:
        base = np.asarray(base_image, dtype=np.uint8)
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        if base.ndim != 3 or base.shape[2] not in (3, 4):
            raise OverlapGeometryError("base image must be grayscale, RGB, or RGBA")
        base = base[:, :, :3]
    current = np.asarray(current_mask, dtype=bool)
    following = np.asarray(next_mask, dtype=bool)
    if current.shape != base.shape[:2] or following.shape != base.shape[:2]:
        raise OverlapGeometryError("preview masks must match the base image size")

    rendered = np.ascontiguousarray(base.copy())
    for mask, color, alpha in (
        (current, current_color, 0.26),
        (following, next_color, 0.26),
        (current & following, intersection_color, 0.78),
    ):
        if mask.any():
            color_array = np.asarray(color, dtype=np.float32)
            pixels = rendered[mask].astype(np.float32)
            rendered[mask] = np.clip(
                pixels * (1.0 - alpha) + color_array * alpha, 0, 255
            ).astype(np.uint8)

    for mask, color, thickness in (
        (current, current_color, 2),
        (following, next_color, 2),
        (current & following, intersection_color, 3),
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(rendered, contours, -1, color, thickness)
    return Image.fromarray(rendered, mode="RGB")
