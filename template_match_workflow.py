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


_EDGE_REFINE_MAX_RADIUS = 10
_EDGE_REFINE_MIN_GAIN = 0.03
_EDGE_REFINE_MIN_TEMPLATE_COVERAGE = 0.80
_EDGE_REFINE_MIN_MASK_VISIBLE_RATIO = 0.90


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


def translate_source_mask(
    mask: Any,
    dx: int,
    dy: int,
    *,
    allow_clip: bool = False,
    min_visible_ratio: float = _EDGE_REFINE_MIN_MASK_VISIBLE_RATIO,
) -> np.ndarray:
    """Translate a full-source mask without changing its internal geometry."""

    binary = _binary_mask(mask, name="mask")
    source_area = int(np.count_nonzero(binary))
    min_visible_ratio = float(min_visible_ratio)
    if not 0.0 <= min_visible_ratio <= 1.0:
        raise ValueError("min_visible_ratio must be between 0 and 1")
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
    visible_area = int(np.count_nonzero(result))
    if visible_area != source_area and not allow_clip:
        raise ValueError("translated mask would be clipped by source bounds")
    if allow_clip and source_area and visible_area / source_area < min_visible_ratio:
        raise ValueError("translated mask retains too little visible area")
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


def _grayscale_u8(source_image: np.ndarray) -> np.ndarray:
    if source_image.ndim == 2:
        return np.clip(source_image, 0, 255).astype(np.uint8, copy=False)
    return cv2.cvtColor(_source_rgb(source_image), cv2.COLOR_RGB2GRAY)


def _partial_ccoeff_at(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    x: int,
    y: int,
    *,
    min_coverage: float,
) -> tuple[float, float]:
    """Score one possibly clipped template placement using real pixels only."""

    image_height, image_width = image_gray.shape
    template_height, template_width = template_gray.shape
    image_x1 = max(0, int(x))
    image_y1 = max(0, int(y))
    image_x2 = min(image_width, int(x) + template_width)
    image_y2 = min(image_height, int(y) + template_height)
    if image_x1 >= image_x2 or image_y1 >= image_y2:
        return -1.0, 0.0

    template_x1 = image_x1 - int(x)
    template_y1 = image_y1 - int(y)
    overlap_width = image_x2 - image_x1
    overlap_height = image_y2 - image_y1
    coverage = float(
        overlap_width * overlap_height / (template_width * template_height)
    )
    if coverage < float(min_coverage):
        return -1.0, coverage

    image_patch = image_gray[image_y1:image_y2, image_x1:image_x2]
    template_patch = template_gray[
        template_y1 : template_y1 + overlap_height,
        template_x1 : template_x1 + overlap_width,
    ]
    if (
        float(np.std(image_patch)) < 1e-6
        or float(np.std(template_patch)) < 1e-6
    ):
        return -1.0, coverage
    score = float(
        cv2.matchTemplate(
            image_patch,
            template_patch,
            cv2.TM_CCOEFF_NORMED,
        )[0, 0]
    )
    return (score if np.isfinite(score) else -1.0), coverage


def _refine_horizontal_edge_translation(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    candidate_bbox_xyxy: Sequence[int],
    *,
    edge_margin: int,
    search_radius: int,
    min_score_gain: float = _EDGE_REFINE_MIN_GAIN,
    min_template_coverage: float = _EDGE_REFINE_MIN_TEMPLATE_COVERAGE,
) -> dict[str, Any]:
    """Refine only the x coordinate of candidates close to a vertical edge."""

    x1, y1, x2, _ = (int(value) for value in candidate_bbox_xyxy)
    image_width = int(image_gray.shape[1])
    left_distance = x1
    right_distance = image_width - x2
    if min(left_distance, right_distance) > int(edge_margin):
        return {"delta_x": 0, "score_gain": 0.0, "template_coverage": 1.0}

    current_score, current_coverage = _partial_ccoeff_at(
        image_gray,
        template_gray,
        x1,
        y1,
        min_coverage=min_template_coverage,
    )
    best_score = current_score
    best_x = x1
    best_coverage = current_coverage
    for delta_x in range(-int(search_radius), int(search_radius) + 1):
        candidate_x = x1 + delta_x
        score, coverage = _partial_ccoeff_at(
            image_gray,
            template_gray,
            candidate_x,
            y1,
            min_coverage=min_template_coverage,
        )
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(delta_x) < abs(best_x - x1)
        ):
            best_score = score
            best_x = candidate_x
            best_coverage = coverage

    score_gain = best_score - current_score
    if best_x == x1 or score_gain < float(min_score_gain):
        return {
            "delta_x": 0,
            "score_gain": max(0.0, float(score_gain)),
            "template_coverage": current_coverage,
        }
    return {
        "delta_x": int(best_x - x1),
        "score_gain": float(score_gain),
        "template_coverage": float(best_coverage),
    }


def _draw_match_label(
    overlay: np.ndarray,
    bbox_xyxy: Sequence[int],
    match_id: Any,
    score: Any,
) -> None:
    """Draw one fixed-size readable label next to a match bbox."""

    try:
        score_value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("match metadata must contain a numeric score") from exc
    label_id = str(match_id).strip()
    if not label_id:
        raise ValueError("match metadata must contain a display id")
    if label_id.isdigit():
        label_id = f"M{int(label_id)}"
    label = f"{label_id} {score_value:.2f}"

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
                match_item.get(
                    "display_id",
                    match_item.get("match_id", match_index + 1),
                ),
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

    The current seed always blocks itself. Every other generated instance also
    blocks duplicate candidates immediately; only deleted instances are ignored.
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
    seed_area = int(np.count_nonzero(seed_mask))
    seed_polygon = _bbox_polygon(seed_bbox)
    seed_id = _instance_id(seed_instance, active_instance_id)
    source_gray = _grayscale_u8(source_image)
    seed_template_gray = source_gray[
        seed_bbox[1] : seed_bbox[3],
        seed_bbox[0] : seed_bbox[2],
    ]
    edge_refine_radius = min(
        _EDGE_REFINE_MAX_RADIUS,
        max(0, int(expand_threshold)),
    )

    blocker_polygons: list[list[list[int]]] = []
    exact_blocker_bboxes = [tuple(seed_bbox)]
    blocker_ids: list[int | str] = []
    for key, candidate in instances.items():
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = _instance_id(candidate, key)
        if str(candidate_id) == str(seed_id) or candidate.get("status") == "deleted":
            continue
        blocker_mask = map_crop_mask_to_source(
            candidate.get("mask_fullres_bool"),
            (source_height, source_width),
            crop_bbox,
        )
        if not np.any(blocker_mask):
            continue
        blocker_bbox = mask_bbox_xyxy(blocker_mask)
        blocker_polygons.append(_bbox_polygon(blocker_bbox))
        exact_blocker_bboxes.append(tuple(blocker_bbox))
        blocker_ids.append(candidate_id)

    raw_matches = match_periodic_instances(
        source_image,
        seed_polygon,
        label=str(label),
        match_threshold=match_threshold,
        expand_threshold=expand_threshold,
        nms_threshold=nms_threshold,
        all_segmentations=blocker_polygons,
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
        coarse_dx = int(candidate_bbox[0] - seed_bbox[0])
        coarse_dy = int(candidate_bbox[1] - seed_bbox[1])
        refinement = {
            "delta_x": 0,
            "score_gain": 0.0,
            "template_coverage": 1.0,
        }
        if edge_refine_radius:
            refinement = _refine_horizontal_edge_translation(
                source_gray,
                seed_template_gray,
                candidate_bbox,
                edge_margin=max(int(expand_threshold), edge_refine_radius),
                search_radius=edge_refine_radius,
            )
        refine_delta_x = int(refinement["delta_x"])
        dx = coarse_dx + refine_delta_x
        dy = coarse_dy
        try:
            translated = translate_source_mask(
                seed_mask,
                dx,
                dy,
                allow_clip=bool(refine_delta_x),
                min_visible_ratio=_EDGE_REFINE_MIN_MASK_VISIBLE_RATIO,
            )
        except ValueError:
            if not refine_delta_x:
                continue
            # A refinement must never discard a legal coarse match. Fall back
            # when the refined mask would retain too little visible area.
            refine_delta_x = 0
            dx = coarse_dx
            translated = translate_source_mask(seed_mask, dx, dy)
        if not np.any(translated):
            continue
        translated_bbox = mask_bbox_xyxy(translated)
        visible_ratio = float(
            np.count_nonzero(translated) / seed_area
        )
        match_masks.append(translated)
        match_item = {
            "match_id": len(serializable_matches) + 1,
            "label": str(raw_match.get("label", label)),
            "score": float(raw_match.get("matchScore", 0.0)),
            "translation_xy": [dx, dy],
            "bbox_xyxy": translated_bbox,
            "area": int(np.count_nonzero(translated)),
        }
        if refine_delta_x:
            match_item.update(
                {
                    "edge_refined": True,
                    "coarse_translation_xy": [coarse_dx, coarse_dy],
                    "edge_refine_delta_xy": [refine_delta_x, 0],
                    "edge_refine_score_gain": float(refinement["score_gain"]),
                    "edge_refine_template_coverage": float(
                        refinement["template_coverage"]
                    ),
                    "visible_ratio": visible_ratio,
                }
            )
        serializable_matches.append(match_item)

    result = {
        "schema_version": 1,
        "source_size_wh": [source_width, source_height],
        "crop_bbox_xyxy": list(crop_bbox),
        "seed": {
            "instance_id": seed_id,
            "bbox_xyxy": seed_bbox,
            "area": seed_area,
            "mask_pixel_sha256": hashlib.sha256(
                np.ascontiguousarray(seed_mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
        },
        "blockers": {
            "seed_instance_id": seed_id,
            "instance_ids": blocker_ids,
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
