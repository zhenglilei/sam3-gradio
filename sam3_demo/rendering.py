"""Pure display and image-rendering helpers."""

from __future__ import annotations

from PIL import Image


def _layout_preview_alpha(state):
    value = (state or {}).get("preview_alpha")
    return float(0.35 if value is None else value)


def _result_placeholder(image_state):
    if not image_state or not image_state.get("image_id"):
        return None
    width = max(1, int(image_state.get("width") or 1))
    height = max(1, int(image_state.get("height") or 1))
    return Image.new("RGB", (width, height), (248, 250, 252))
