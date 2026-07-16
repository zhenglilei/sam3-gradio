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
    if float(np.std(template.astype(np.float32))) < 1e-6:
        raise ValueError("the template has no intensity variation")

    score_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    candidate_y, candidate_x = np.where(
        np.isfinite(score_map) & (score_map >= match_threshold)
    )
    candidates = [
        (
            (int(x), int(y), template_width, template_height),
            float(score_map[y, x]),
        )
        for y, x in zip(candidate_y, candidate_x)
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
