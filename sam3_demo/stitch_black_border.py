"""Conservative black-border trimming for the periodic stitch workflow.

The stitcher receives tiles that may contain a thin black frame introduced by
the camera or by a previous export.  This module deliberately does not try to
remove arbitrary dark pixels: only near-black pixels connected to an outer
edge and covering most of that edge are considered a removable border.  The
operation is CPU-only and returns an audit record so callers can show what was
changed without keeping a second copy of every original tile.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from PIL import Image


# These defaults are intentionally stricter than the legacy T4 implementation
# (which accepted an edge black-pixel ratio of 0.25).  A border should be a
# broad strip along an edge; a local dark component should not be cropped.
# JPEG DCT ringing can lift a nominally black frame into roughly the 30--45
# range.  48 catches that near-black halo while the edge-span and
# edge-connected checks below keep ordinary dark content out.
DEFAULT_BLACK_THRESHOLD = 48
DEFAULT_MIN_EDGE_COVERAGE = 0.85
DEFAULT_MAX_TRIM_FRACTION = 0.20
DEFAULT_MIN_RETAINED_FRACTION = 0.50
DEFAULT_MIN_BORDER_WIDTH = 2
DEFAULT_MIN_RETAINED_SIZE = 8
_LOW_DYNAMIC_RANGE = 8


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Return a detached PIL image while accepting the common array forms."""

    if isinstance(image, Image.Image):
        return image.copy()

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a PIL image or a 2-D/3-D NumPy array")
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError("a color image must have 3 (RGB) or 4 (RGBA) channels")
    if array.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(array), mode="L")
    mode = "RGB" if array.shape[2] == 3 else "RGBA"
    return Image.fromarray(np.ascontiguousarray(array), mode=mode)


def _record(
    *,
    original_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    trim: tuple[int, int, int, int],
    applied: bool,
    reason: str,
    detected_sides: tuple[str, ...] = (),
    edge_coverage: dict[str, float] | None = None,
    candidate_trim: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly, stable audit record."""

    left, top, right, bottom = trim
    result: dict[str, Any] = {
        "original_size": [int(original_size[0]), int(original_size[1])],
        "crop_bbox": [int(value) for value in bbox],
        "trim": {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        },
        "applied": bool(applied),
        "reason": str(reason),
    }
    if detected_sides:
        result["detected_sides"] = list(detected_sides)
    if edge_coverage:
        result["edge_coverage"] = {
            side: round(float(value), 4)
            for side, value in edge_coverage.items()
        }
    if candidate_trim is not None:
        result["candidate_trim"] = [int(value) for value in candidate_trim]
    return result


def _edge_connected_mask(rgb: np.ndarray, threshold: int) -> np.ndarray:
    """Find near-black pixels belonging to a component touching an image edge."""

    # Requiring every channel to be dark avoids classifying a saturated dark
    # green/blue circuit feature as a black frame.
    near_black = np.max(rgb, axis=2) <= int(threshold)
    if not np.any(near_black):
        return np.zeros(near_black.shape, dtype=bool)

    _count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        near_black.astype(np.uint8), connectivity=8
    )
    edge_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )
    edge_labels = edge_labels[edge_labels != 0]
    if edge_labels.size == 0:
        return np.zeros(near_black.shape, dtype=bool)
    return near_black & np.isin(labels, edge_labels)


def _leading_run(values: np.ndarray, minimum: float) -> int:
    """Count a contiguous qualifying run from the first element."""

    count = 0
    for value in values:
        if float(value) < minimum:
            break
        count += 1
    return count


def _detect_trim(
    rgb: np.ndarray,
    *,
    threshold: int,
    min_edge_coverage: float,
    min_border_width: int,
) -> tuple[tuple[int, int, int, int], dict[str, float], tuple[str, ...]]:
    connected = _edge_connected_mask(rgb, threshold)
    h, w = connected.shape
    row_coverage = connected.mean(axis=1)
    col_coverage = connected.mean(axis=0)

    top = _leading_run(row_coverage, min_edge_coverage)
    bottom = _leading_run(row_coverage[::-1], min_edge_coverage)
    left = _leading_run(col_coverage, min_edge_coverage)
    right = _leading_run(col_coverage[::-1], min_edge_coverage)

    # A one-pixel isolated dark line is much more likely to be image content
    # or compression noise than a removable frame.  Requiring a short
    # contiguous run keeps the operation conservative while still handling the
    # usual 2+ pixel camera border.
    if top < min_border_width:
        top = 0
    if bottom < min_border_width:
        bottom = 0
    if left < min_border_width:
        left = 0
    if right < min_border_width:
        right = 0

    # Do not let opposing scans consume the same rows/columns.
    if top + bottom >= h:
        top = bottom = 0
    if left + right >= w:
        left = right = 0

    coverage = {
        "top": float(row_coverage[0]) if h else 0.0,
        "bottom": float(row_coverage[-1]) if h else 0.0,
        "left": float(col_coverage[0]) if w else 0.0,
        "right": float(col_coverage[-1]) if w else 0.0,
    }
    trim = (left, top, right, bottom)
    sides = tuple(
        side
        for side, amount in zip(("left", "top", "right", "bottom"), trim)
        if amount > 0
    )
    return trim, coverage, sides


def trim_black_border(
    image: Image.Image | np.ndarray,
    *,
    threshold: int = DEFAULT_BLACK_THRESHOLD,
    min_edge_coverage: float = DEFAULT_MIN_EDGE_COVERAGE,
    max_trim_fraction: float = DEFAULT_MAX_TRIM_FRACTION,
    min_retained_fraction: float = DEFAULT_MIN_RETAINED_FRACTION,
    min_border_width: int = DEFAULT_MIN_BORDER_WIDTH,
    min_retained_size: int = DEFAULT_MIN_RETAINED_SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Safely crop a near-black border from one tile.

    ``crop_bbox`` uses PIL's half-open ``(left, top, right, bottom)``
    convention.  If the evidence is weak or the proposed crop is unsafe, the
    detached original image is returned and ``record["applied"]`` is false.
    """

    source = _as_pil(image)
    width, height = source.size
    original_size = (width, height)
    whole_bbox = (0, 0, width, height)
    no_trim = (0, 0, 0, 0)

    if width < 1 or height < 1:
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="empty_image",
        )
        return source, record

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    # Completely black or nearly uniform tiles have no reliable frame/content
    # distinction.  Returning them unchanged is safer than deleting most of
    # the tile.
    if not np.any(np.max(rgb, axis=2) > int(threshold)):
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="all_black",
        )
        return source, record
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if int(np.ptp(luma)) < _LOW_DYNAMIC_RANGE:
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="low_dynamic_range",
        )
        return source, record

    threshold = max(0, min(255, int(threshold)))
    min_edge_coverage = max(0.5, min(1.0, float(min_edge_coverage)))
    max_trim_fraction = max(0.0, min(1.0, float(max_trim_fraction)))
    min_retained_fraction = max(0.0, min(1.0, float(min_retained_fraction)))
    min_border_width = max(1, int(min_border_width))
    min_retained_size = max(1, int(min_retained_size))

    trim, coverage, sides = _detect_trim(
        rgb,
        threshold=threshold,
        min_edge_coverage=min_edge_coverage,
        min_border_width=min_border_width,
    )
    left, top, right, bottom = trim
    if not sides:
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="no_black_border",
            edge_coverage=coverage,
        )
        return source, record

    # Reject an implausibly large candidate even if the remaining dimensions
    # would technically be valid.  This specifically protects edge-local dark
    # structures and malformed/mostly-black tiles.
    if any(
        (
            left > width * max_trim_fraction,
            right > width * max_trim_fraction,
            top > height * max_trim_fraction,
            bottom > height * max_trim_fraction,
        )
    ):
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="border_crop_too_large",
            detected_sides=sides,
            edge_coverage=coverage,
            candidate_trim=trim,
        )
        return source, record

    new_width = width - left - right
    new_height = height - top - bottom
    if (
        new_width < min_retained_size
        or new_height < min_retained_size
        or new_width < width * min_retained_fraction
        or new_height < height * min_retained_fraction
    ):
        record = _record(
            original_size=original_size,
            bbox=whole_bbox,
            trim=no_trim,
            applied=False,
            reason="retained_size_too_small",
            detected_sides=sides,
            edge_coverage=coverage,
            candidate_trim=trim,
        )
        return source, record

    bbox = (left, top, width - right, height - bottom)
    cropped = source.crop(bbox)
    record = _record(
        original_size=original_size,
        bbox=bbox,
        trim=trim,
        applied=True,
        reason="trimmed_black_border",
        detected_sides=sides,
        edge_coverage=coverage,
    )
    return cropped, record


def trim_black_borders(
    images: Iterable[Image.Image | np.ndarray],
    **kwargs: Any,
) -> tuple[list[Image.Image], list[dict[str, Any]], list[str]]:
    """Trim a batch of tiles and return images, audit records, and warnings."""

    output: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, image in enumerate(images, start=1):
        try:
            processed, record = trim_black_border(image, **kwargs)
        except Exception as exc:
            processed = _as_pil(image)
            width, height = processed.size
            record = _record(
                original_size=(width, height),
                bbox=(0, 0, width, height),
                trim=(0, 0, 0, 0),
                applied=False,
                reason="processing_error",
            )
            warnings.append(f"第{index}张图去黑边失败，已保留原图：{exc}")
        output.append(processed)
        records.append(record)
        if not record["applied"] and record["reason"] not in {
            "no_black_border",
            "all_black",
            "low_dynamic_range",
            "processing_error",
        }:
            warnings.append(f"第{index}张图未去黑边：{record['reason']}")
    return output, records, warnings


__all__ = [
    "DEFAULT_BLACK_THRESHOLD",
    "DEFAULT_MIN_EDGE_COVERAGE",
    "DEFAULT_MAX_TRIM_FRACTION",
    "DEFAULT_MIN_RETAINED_FRACTION",
    "trim_black_border",
    "trim_black_borders",
]
