"""Pure helpers for authoritative image crop coordinates."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image


def normalize_crop_box(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    image_width: int,
    image_height: int,
    *,
    min_size: int = 4,
) -> tuple[int, int, int, int]:
    """Return a clamped ``[x1, y1, x2, y2)`` crop box."""

    width = int(image_width)
    height = int(image_height)
    minimum = int(min_size)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if minimum <= 0:
        raise ValueError("min_size must be positive")
    if len(start_xy) != 2 or len(end_xy) != 2:
        raise ValueError("crop gesture must contain two [x, y] points")
    values = [float(start_xy[0]), float(start_xy[1]), float(end_xy[0]), float(end_xy[1])]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("crop coordinates must be finite")
    left = max(0, min(width, math.floor(min(values[0], values[2]))))
    top = max(0, min(height, math.floor(min(values[1], values[3]))))
    right = max(0, min(width, math.ceil(max(values[0], values[2]))))
    bottom = max(0, min(height, math.ceil(max(values[1], values[3]))))
    if right - left < minimum or bottom - top < minimum:
        raise ValueError(f"crop rectangle must be at least {minimum}x{minimum} pixels")
    return left, top, right, bottom


def whole_image_crop_box(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return 0, 0, width, height


def crop_pil_image(image: Image.Image, crop_box_xyxy: Sequence[int]) -> Image.Image:
    """Crop a PIL image after validating an exclusive-end crop box."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if len(crop_box_xyxy) != 4:
        raise ValueError("crop_box_xyxy must contain four coordinates")
    x1, y1, x2, y2 = (int(value) for value in crop_box_xyxy)
    width, height = image.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("crop_box_xyxy lies outside the source image")
    return image.convert("RGB").crop((x1, y1, x2, y2))
