"""Pure orchestration helpers for full-image periodic template matching.

The workflow maps crop-local PVS masks back to the uploaded source image,
delegates candidate discovery to :mod:`periodic_template_matching`, and builds
independent full-image masks and preview metadata.  It never invokes SAM3 or
modifies the supplied PVS state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from periodic_template_matching import match_periodic_instances


def _binary_mask(
    mask: Any,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Validate a real, finite 2-D mask before converting it to bool."""

    try:
        array = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-dimensional numeric array") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} shape does not match the expected shape: "
            f"expected {expected_shape}, got {array.shape}"
        )
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(bool, copy=False)


def _image_shape(source_image: np.ndarray) -> tuple[int, int]:
    if (
        not isinstance(source_image, np.ndarray)
        or source_image.ndim not in (2, 3)
        or source_image.size == 0
    ):
        raise ValueError("source_image must be a non-empty NumPy image")
    height, width = source_image.shape[:2]
    return int(height), int(width)


def _crop_box(
    crop_bbox_xyxy: Sequence[Any],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    values = np.asarray(crop_bbox_xyxy)
    if values.shape != (4,) or not np.isfinite(values.astype(np.float64)).all():
        raise ValueError("crop_bbox_xyxy must contain four finite coordinates")
    converted = tuple(int(value) for value in values.tolist())
    if any(float(value) != converted[index] for index, value in enumerate(values.tolist())):
        raise ValueError("crop_bbox_xyxy coordinates must be integers")
    x1, y1, x2, y2 = converted
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError("crop_bbox_xyxy must lie inside source_image")
    return converted


def map_crop_mask_to_source(
    crop_mask: Any,
    source_shape_hw: Sequence[int],
    crop_bbox_xyxy: Sequence[Any],
) -> np.ndarray:
    """Place a crop-local binary mask into full-source coordinates."""

    if len(source_shape_hw) != 2:
        raise ValueError("source_shape_hw must be [height, width]")
    source_height, source_width = (int(value) for value in source_shape_hw)
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_shape_hw must be positive")
    x1, y1, x2, y2 = _crop_box(
        crop_bbox_xyxy,
        source_width,
        source_height,
    )
    expected_shape = (y2 - y1, x2 - x1)
    mask = _binary_mask(
        crop_mask,
        name="crop mask",
        expected_shape=expected_shape,
    )
    full_mask = np.zeros((source_height, source_width), dtype=bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def mask_bbox_xyxy(mask: Any) -> list[int]:
    """Return an exclusive ``[x1, y1, x2, y2]`` foreground bounding box."""

    binary = _binary_mask(mask, name="mask")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("mask must contain foreground pixels")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def _bbox_polygon(bbox_xyxy: Sequence[int]) -> list[list[int]]:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("template mask bbox must be at least 2 x 2 pixels")
    return [
        [x1, y1],
        [x2 - 1, y1],
        [x2 - 1, y2 - 1],
        [x1, y2 - 1],
    ]


def translate_source_mask(mask: Any, dx: int, dy: int) -> np.ndarray:
    """Translate a full-source mask without changing its internal geometry."""

    binary = _binary_mask(mask, name="mask")
    source_area = int(np.count_nonzero(binary))
    matrix = np.array([[1.0, 0.0, int(dx)], [0.0, 1.0, int(dy)]])
    translated = cv2.warpAffine(
        binary.astype(np.uint8),
        matrix,
        (binary.shape[1], binary.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = translated.astype(bool)
    if int(np.count_nonzero(result)) != source_area:
        raise ValueError("translated mask would be clipped by source bounds")
    return result


def _instance(instances: Mapping[Any, Any], instance_id: Any) -> Mapping[str, Any]:
    candidates = (instance_id, str(instance_id))
    try:
        candidates = (*candidates, int(instance_id))
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        value = instances.get(candidate)
        if isinstance(value, Mapping):
            return value
    raise ValueError(f"PVS instance {instance_id!r} does not exist")


def _instance_id(instance: Mapping[str, Any], fallback: Any) -> int | str:
    value = instance.get("id", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _source_rgb(source_image: np.ndarray) -> np.ndarray:
    if source_image.ndim == 2:
        return np.repeat(source_image[..., None], 3, axis=2).astype(np.uint8)
    if source_image.shape[2] == 1:
        return np.repeat(source_image, 3, axis=2).astype(np.uint8)
    if source_image.shape[2] not in (3, 4):
        raise ValueError("source_image must have 1, 3, or 4 channels")
    return np.clip(source_image[..., :3], 0, 255).astype(np.uint8).copy()


def _draw_match_label(
    overlay: np.ndarray,
    bbox_xyxy: Sequence[int],
    match_id: Any,
    score: Any,
) -> None:
    """Draw one fixed-size readable label next to a match bbox."""

    try:
        label = f"M{int(match_id)} {float(score):.2f}"
    except (TypeError, ValueError) as exc:
        raise ValueError("match metadata must contain numeric match_id and score") from exc

    x1, y1, _, _ = (int(value) for value in bbox_xyxy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )
    box_width = text_width + 2 * padding
    box_height = text_height + baseline + 2 * padding
    height, width = overlay.shape[:2]
    left = min(max(0, x1), max(0, width - box_width))
    top = y1 - box_height if y1 >= box_height else y1
    top = min(max(0, top), max(0, height - box_height))
    right = min(width - 1, left + box_width)
    bottom = min(height - 1, top + box_height)
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 80, 0), -1)
    cv2.putText(
        overlay,
        label,
        (left + padding, min(height - 1, top + padding + text_height)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_template_overlay(
    source_image: np.ndarray,
    seed_mask: Any,
    match_masks: Sequence[Any],
    match_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Render a source-size RGB overlay (yellow seed, green matches)."""

    overlay = _source_rgb(source_image)
    seed = _binary_mask(
        seed_mask,
        name="seed mask",
        expected_shape=overlay.shape[:2],
    )

    metadata = None if match_metadata is None else list(match_metadata)
    if metadata is not None and len(metadata) != len(match_masks):
        raise ValueError("match_metadata must align with match_masks")

    for match_index, match_mask in enumerate(match_masks):
        match = _binary_mask(
            match_mask,
            name="match mask",
            expected_shape=overlay.shape[:2],
        )
        overlay[match] = np.rint(
            overlay[match].astype(np.float32) * 0.65
            + np.array([0.0, 255.0, 80.0], dtype=np.float32) * 0.35
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            match.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 80), 1)
        if metadata is not None:
            match_item = metadata[match_index]
            if not isinstance(match_item, Mapping):
                raise ValueError("each match metadata item must be a mapping")
            _draw_match_label(
                overlay,
                mask_bbox_xyxy(match),
                match_item.get("match_id", match_index + 1),
                match_item.get("score", 0.0),
            )

    contours, _ = cv2.findContours(
        seed.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 215, 0), 2)
    return overlay


def run_template_match_workflow(
    source_image: np.ndarray,
    crop_bbox_xyxy: Sequence[Any],
    pvs_state: Mapping[str, Any],
    *,
    active_instance_id: Any | None = None,
    match_threshold: float = 0.7,
    expand_threshold: int = 20,
    nms_threshold: float = 0.3,
    label: str = "template",
) -> dict[str, Any]:
    """Match one active crop-local PVS mask across the full source image.

    The current seed always blocks itself.  Other instances block candidates
    only when their status is ``accepted``; draft and deleted instances are
    deliberately ignored.
    """

    source_height, source_width = _image_shape(source_image)
    crop_bbox = _crop_box(crop_bbox_xyxy, source_width, source_height)
    if not isinstance(pvs_state, Mapping):
        raise ValueError("pvs_state must be a mapping")
    instances = pvs_state.get("instances")
    if not isinstance(instances, Mapping):
        raise ValueError("pvs_state.instances must be a mapping")
    if active_instance_id is None:
        active_instance_id = pvs_state.get("active_instance_id")
    if active_instance_id is None:
        raise ValueError("an active PVS instance is required")

    seed_instance = _instance(instances, active_instance_id)
    if seed_instance.get("status") == "deleted":
        raise ValueError("the active PVS instance is deleted")
    seed_mask = map_crop_mask_to_source(
        seed_instance.get("mask_fullres_bool"),
        (source_height, source_width),
        crop_bbox,
    )
    seed_bbox = mask_bbox_xyxy(seed_mask)
    seed_polygon = _bbox_polygon(seed_bbox)
    seed_id = _instance_id(seed_instance, active_instance_id)

    accepted_polygons: list[list[list[int]]] = []
    exact_blocker_bboxes = [tuple(seed_bbox)]
    accepted_ids: list[int | str] = []
    for key, candidate in instances.items():
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = _instance_id(candidate, key)
        if str(candidate_id) == str(seed_id) or candidate.get("status") != "accepted":
            continue
        accepted_mask = map_crop_mask_to_source(
            candidate.get("mask_fullres_bool"),
            (source_height, source_width),
            crop_bbox,
        )
        if not np.any(accepted_mask):
            continue
        accepted_bbox = mask_bbox_xyxy(accepted_mask)
        accepted_polygons.append(_bbox_polygon(accepted_bbox))
        exact_blocker_bboxes.append(tuple(accepted_bbox))
        accepted_ids.append(candidate_id)

    raw_matches = match_periodic_instances(
        source_image,
        seed_polygon,
        label=str(label),
        match_threshold=match_threshold,
        expand_threshold=expand_threshold,
        nms_threshold=nms_threshold,
        all_segmentations=accepted_polygons,
    )

    match_masks: list[np.ndarray] = []
    serializable_matches: list[dict[str, Any]] = []
    for raw_match in raw_matches:
        candidate_points = np.asarray(raw_match.get("segmentation"), dtype=np.int32)
        if candidate_points.ndim != 2 or candidate_points.shape[1] != 2:
            raise ValueError("matcher returned an invalid segmentation")
        candidate_bbox = mask_bbox_xyxy(
            cv2.fillPoly(
                np.zeros((source_height, source_width), dtype=np.uint8),
                [candidate_points],
                1,
            )
        )
        if tuple(candidate_bbox) in exact_blocker_bboxes:
            continue
        dx = int(candidate_bbox[0] - seed_bbox[0])
        dy = int(candidate_bbox[1] - seed_bbox[1])
        try:
            translated = translate_source_mask(seed_mask, dx, dy)
        except ValueError:
            continue
        if not np.any(translated):
            continue
        translated_bbox = mask_bbox_xyxy(translated)
        match_masks.append(translated)
        serializable_matches.append(
            {
                "match_id": len(serializable_matches) + 1,
                "label": str(raw_match.get("label", label)),
                "score": float(raw_match.get("matchScore", 0.0)),
                "translation_xy": [dx, dy],
                "bbox_xyxy": translated_bbox,
                "area": int(np.count_nonzero(translated)),
            }
        )

    result = {
        "schema_version": 1,
        "source_size_wh": [source_width, source_height],
        "crop_bbox_xyxy": list(crop_bbox),
        "seed": {
            "instance_id": seed_id,
            "status": str(seed_instance.get("status", "draft")),
            "bbox_xyxy": seed_bbox,
            "area": int(np.count_nonzero(seed_mask)),
            "mask_pixel_sha256": hashlib.sha256(
                np.ascontiguousarray(seed_mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
        },
        "blockers": {
            "seed_instance_id": seed_id,
            "accepted_instance_ids": accepted_ids,
        },
        "parameters": {
            "match_threshold": float(match_threshold),
            "expand_threshold": int(expand_threshold),
            "nms_threshold": float(nms_threshold),
        },
        "match_count": len(serializable_matches),
        "matches": serializable_matches,
    }
    return {
        "result": result,
        "seed_mask_fullres_bool": seed_mask,
        "match_masks_fullres_bool": match_masks,
        "overlay_rgb": render_template_overlay(
            source_image,
            seed_mask,
            match_masks,
            serializable_matches,
        ),
    }


__all__ = [
    "map_crop_mask_to_source",
    "mask_bbox_xyxy",
    "render_template_overlay",
    "run_template_match_workflow",
    "translate_source_mask",
]
