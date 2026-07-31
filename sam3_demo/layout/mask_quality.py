"""Topology and pixel-change checks for layout-mask assistant candidates."""

from __future__ import annotations

import cv2
import numpy as np


def _labels(mask):
    return cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)


def _hole_labels(mask):
    background = ~np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), 8)
    border_ids = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
    holes = []
    for component_id in range(1, count):
        if component_id not in border_ids:
            holes.append((component_id, int(stats[component_id, cv2.CC_STAT_AREA])))
    return labels, holes


def mask_metrics(mask):
    mask = np.asarray(mask, dtype=bool)
    foreground = int(mask.sum())
    component_count = max(0, int(_labels(mask)[0]) - 1)
    _, holes = _hole_labels(mask)
    return {
        "foreground_pixels": foreground,
        "foreground_ratio": float(mask.mean()),
        "component_count": component_count,
        "hole_count": len(holes),
        "hole_area": int(sum(area for _, area in holes)),
    }


def compare_masks(previous_mask, candidate_mask):
    previous = np.asarray(previous_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if previous.shape != candidate.shape:
        raise ValueError("Candidate mask shape does not match the current mask")
    added = candidate & ~previous
    removed = previous & ~candidate
    _, previous_labels = _labels(previous)
    candidate_count, candidate_labels = _labels(candidate)
    merge_count = 0
    for component_id in range(1, candidate_count):
        overlapped = np.unique(previous_labels[candidate_labels == component_id])
        foreground_ids = overlapped[overlapped > 0]
        if len(foreground_ids) > 1:
            merge_count += int(len(foreground_ids) - 1)

    hole_labels, holes = _hole_labels(previous)
    large_hole_min_area = max(64, int(round(previous.size * 0.0005)))
    lost_large_holes = []
    for component_id, area in holes:
        if area < large_hole_min_area:
            continue
        pixels = hole_labels == component_id
        if float(candidate[pixels].mean()) > 0.5:
            lost_large_holes.append({"area": area})

    denominator = max(1, int(previous.sum()))
    added_ratio = float(added.mean())
    removed_ratio = float(removed.mean())
    changed_relative_to_foreground = float((added.sum() + removed.sum()) / denominator)
    return {
        "previous": mask_metrics(previous),
        "candidate": mask_metrics(candidate),
        "added_pixels": int(added.sum()),
        "removed_pixels": int(removed.sum()),
        "added_ratio": added_ratio,
        "removed_ratio": removed_ratio,
        "changed_relative_to_foreground": changed_relative_to_foreground,
        "component_merge_count": merge_count,
        "large_holes_lost": lost_large_holes,
        "empty": not bool(candidate.any()),
        "full": bool(candidate.all()),
        "manual_review": bool(
            added_ratio > 0.05
            or removed_ratio > 0.05
            or changed_relative_to_foreground > 0.25
        ),
    }


def validate_candidate_transition(previous_mask, candidate_mask):
    report = compare_masks(previous_mask, candidate_mask)
    if report["empty"]:
        raise ValueError("Candidate mask is empty")
    if report["full"]:
        raise ValueError("Candidate mask covers the full image")
    if report["component_merge_count"]:
        raise ValueError("Candidate merges previously separate components")
    if report["large_holes_lost"]:
        raise ValueError("Candidate removes a large hole")
    return report


def render_mask_delta(image, previous_mask, candidate_mask, *, manual_review=False):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    previous = np.asarray(previous_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    added = candidate & ~previous
    removed = previous & ~candidate
    output = rgb.astype(np.float32)
    output[added] = output[added] * 0.35 + np.array([255, 40, 40], dtype=np.float32) * 0.65
    output[removed] = output[removed] * 0.35 + np.array([40, 90, 255], dtype=np.float32) * 0.65
    if manual_review:
        changed = (added | removed).astype(np.uint8)
        risk_outline = cv2.dilate(
            changed,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool) & ~changed.astype(bool)
        output[risk_outline] = np.array([255, 220, 0], dtype=np.float32)
        cv2.rectangle(output, (1, 1), (max(1, rgb.shape[1] - 2), max(1, rgb.shape[0] - 2)), (255, 220, 0), 4)
    return np.clip(output, 0, 255).astype(np.uint8)
