"""Backend-only template matching for repeated image annotations.

The matcher extracts a rectangular template around one seed polygon, locates
translated copies with OpenCV template matching, and copies the seed polygon
to each retained location. It intentionally does not perform segmentation or
connect to the Gradio UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAX_LOCAL_MATCH_CANDIDATES = 4096


def _polygon_points(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{name} must contain at least three [x, y] points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains a non-finite coordinate")

    # Keep compatibility with the reference smart_annotation implementation:
    # floating-point LabelMe coordinates are truncated before matching.
    points = points.astype(np.int32)
    if len(np.unique(points, axis=0)) < 3 or cv2.contourArea(points) == 0:
        raise ValueError(f"{name} must describe a non-degenerate polygon")
    return points


def _validate_threshold(value: Any, name: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return threshold


def _template_has_spatial_variation(template: np.ndarray) -> bool:
    values = template.astype(np.float32)
    if values.ndim == 2:
        return float(np.std(values)) >= 1e-6
    channel_std = np.std(values, axis=(0, 1))
    return bool(np.any(channel_std >= 1e-6))


def _local_peak_candidates(
    score_map: np.ndarray,
    match_threshold: float,
    *,
    max_candidates: int = MAX_LOCAL_MATCH_CANDIDATES,
) -> list[tuple[int, int, float]]:
    finite = np.isfinite(score_map)
    working = np.where(finite, score_map, -2.0).astype(np.float32, copy=False)
    neighborhood_max = cv2.dilate(
        working,
        np.ones((3, 3), dtype=np.uint8),
    )
    peak_mask = (
        finite
        & (working >= match_threshold)
        & (working == neighborhood_max)
    ).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        peak_mask,
        connectivity=8,
    )
    candidate_count = component_count - 1
    if candidate_count > max_candidates:
        raise ValueError(
            "too many local template matches "
            f"({candidate_count} > {max_candidates}); increase match_threshold "
            "or use a more distinctive seed template"
        )

    candidates: list[tuple[int, int, float]] = []
    for component_id in range(1, component_count):
        x, y, width, height, _ = stats[component_id].tolist()
        component = labels[y : y + height, x : x + width] == component_id
        component_scores = np.where(
            component,
            working[y : y + height, x : x + width],
            -np.inf,
        )
        offset_y, offset_x = np.unravel_index(
            int(np.argmax(component_scores)),
            component_scores.shape,
        )
        match_x = int(x + offset_x)
        match_y = int(y + offset_y)
        candidates.append((match_x, match_y, float(working[match_y, match_x])))
    return candidates


def _padded_bbox(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(points)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + width + padding)
    y2 = min(image_height, y + height + padding)
    return x1, y1, x2 - x1, y2 - y1


def _bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _greedy_nms(
    candidates: list[tuple[tuple[int, int, int, int], float]],
    blocker_boxes: Sequence[Sequence[int]],
    nms_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Apply deterministic NMS while treating existing annotations as blockers."""

    candidates = sorted(
        candidates,
        key=lambda item: (-item[1], item[0][1], item[0][0]),
    )
    kept: list[tuple[tuple[int, int, int, int], float]] = []
    for bbox, score in candidates:
        if any(_bbox_iou(bbox, blocker) > nms_threshold for blocker in blocker_boxes):
            continue
        if any(_bbox_iou(bbox, kept_bbox) > nms_threshold for kept_bbox, _ in kept):
            continue
        kept.append((bbox, score))
    return kept


def match_periodic_instances(
    image: np.ndarray,
    segmentation: Any,
    *,
    label: str,
    match_threshold: float = 0.7,
    expand_threshold: int = 20,
    nms_threshold: float = 0.3,
    all_segmentations: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Find translated copies of a seed polygon in a periodic image.

    The returned polygons have the same shape, scale, and orientation as the
    seed. Only their x/y translation changes.
    """

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy image")

    match_threshold = _validate_threshold(match_threshold, "match_threshold")
    nms_threshold = _validate_threshold(nms_threshold, "nms_threshold")
    expand_threshold = int(expand_threshold)
    if expand_threshold < 0:
        raise ValueError("expand_threshold must be non-negative")

    image_height, image_width = image.shape[:2]
    seed_points = _polygon_points(segmentation, "segmentation")
    if (
        np.any(seed_points[:, 0] < 0)
        or np.any(seed_points[:, 0] >= image_width)
        or np.any(seed_points[:, 1] < 0)
        or np.any(seed_points[:, 1] >= image_height)
    ):
        raise ValueError("segmentation coordinates must lie inside the image")

    template_bbox = _padded_bbox(
        seed_points,
        image_width,
        image_height,
        expand_threshold,
    )
    template_x, template_y, template_width, template_height = template_bbox
    template = image[
        template_y : template_y + template_height,
        template_x : template_x + template_width,
    ]
    if template.size == 0:
        raise ValueError("the seed polygon produced an empty template")
    if not _template_has_spatial_variation(template):
        raise ValueError("the template has no intensity variation")

    score_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    candidates = [
        (
            (int(x), int(y), template_width, template_height),
            score,
        )
        for x, y, score in _local_peak_candidates(
            score_map,
            match_threshold,
        )
    ]

    # Always block the seed template itself. Existing annotations are padded in
    # the same way as the seed and suppress overlapping new candidates.
    blocker_boxes: list[tuple[int, int, int, int]] = [template_bbox]
    for index, existing in enumerate(all_segmentations):
        existing_points = _polygon_points(existing, f"all_segmentations[{index}]")
        blocker_boxes.append(
            _padded_bbox(
                existing_points,
                image_width,
                image_height,
                expand_threshold,
            )
        )

    relative_points = seed_points - np.array([template_x, template_y], dtype=np.int32)
    matches = []
    for bbox, score in _greedy_nms(candidates, blocker_boxes, nms_threshold):
        x, y, _, _ = bbox
        matched_points = relative_points + np.array([x, y], dtype=np.int32)
        matches.append(
            {
                "segmentation": matched_points.tolist(),
                "label": str(label),
                "matchScore": round(score, 2),
            }
        )
    return matches


def smart_annotation(dict_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compatibility entry point for the original ``smart_annotation`` API."""

    required = ("picPath", "label", "matchThreshold", "segmentation")
    missing = [name for name in required if name not in dict_params]
    if missing:
        raise KeyError(f"missing required parameters: {', '.join(missing)}")

    image_path = Path(dict_params["picPath"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    return match_periodic_instances(
        image,
        dict_params["segmentation"],
        label=dict_params["label"],
        match_threshold=dict_params["matchThreshold"],
        expand_threshold=dict_params.get("expandThreshold", 20),
        nms_threshold=dict_params.get("nmsThreshold", 0.3),
        all_segmentations=dict_params.get("allSegmentation", ()),
    )
