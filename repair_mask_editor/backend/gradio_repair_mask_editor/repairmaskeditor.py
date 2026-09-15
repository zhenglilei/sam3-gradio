"""Gradio custom component for browser-side repair mask editing."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from gradio.components.base import Component
from gradio.components.json_component import JsonData
from gradio.events import Events
from gradio.i18n import I18nData

if TYPE_CHECKING:
    from gradio.components import Timer


_TOOLS = frozenset({"brush", "eraser", "rect_add", "rect_erase"})
_MAX_DIMENSION = 100_000
_MAX_DATA_URL_LENGTH = 100_000_000
_MAX_STATUS_LENGTH = 1_024


def _unwrap(payload: Any) -> Any:
    return payload.root if isinstance(payload, JsonData) else payload


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive_integer(value: Any, maximum: int = _MAX_DIMENSION) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    integer = math.trunc(number)
    if integer < 1 or integer > maximum:
        return None
    return integer


def _nonnegative_integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    integer = math.trunc(number)
    return integer if integer >= 0 else None


def _string(value: Any, *, maximum: int = _MAX_DATA_URL_LENGTH) -> str | None:
    if not isinstance(value, str) or len(value) > maximum:
        return None
    return value


def _data_url(value: Any) -> str | None:
    text = _string(value)
    if text is None or not text.startswith("data:image/png;base64,"):
        return None
    return text


class RepairMaskEditor(Component):
    """Canvas editor whose value contains a source-sized repair mask PNG."""

    EVENTS = [Events.change]
    data_model = None

    def __init__(
        self,
        value: dict | str | None = None,
        *,
        label: str | I18nData | None = None,
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool | Literal["hidden"] = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
        preserved_by_key: list[str] | str | None = "value",
        height: int | str = 520,
    ):
        self.height = height
        super().__init__(
            label=label,
            every=every,
            inputs=inputs,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            preserved_by_key=preserved_by_key,
            value=value,
        )

    @staticmethod
    def _sanitize(raw: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}

        image_id = _string(raw.get("image_id"), maximum=512)
        if image_id is not None:
            result["image_id"] = image_id

        revision = _nonnegative_integer(raw.get("revision"))
        if revision is not None:
            result["revision"] = revision

        for field in ("source_width", "source_height"):
            dimension = _positive_integer(raw.get(field))
            if dimension is not None:
                result[field] = dimension

        base_image = raw.get("base_image")
        if base_image is None:
            result["base_image"] = None
        else:
            base_text = _string(base_image)
            if base_text is not None:
                result["base_image"] = base_text

        mask_png = _data_url(raw.get("mask_png"))
        if mask_png is not None:
            result["mask_png"] = mask_png

        alpha = _finite_number(raw.get("preview_alpha"))
        if alpha is not None:
            result["preview_alpha"] = max(0.0, min(1.0, alpha))

        tool = raw.get("tool")
        if isinstance(tool, str) and tool in _TOOLS:
            result["tool"] = tool

        brush_size = _finite_number(raw.get("brush_size"))
        if brush_size is not None and brush_size > 0:
            result["brush_size"] = max(1.0, min(10_000.0, brush_size))

        status = raw.get("status")
        if isinstance(status, str) and len(status) <= _MAX_STATUS_LENGTH:
            result["status"] = status
        return result

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = _unwrap(payload)
        if not isinstance(raw, dict):
            return {}
        return self._sanitize(raw)

    def postprocess(self, value: dict | list | str | None) -> JsonData | None:
        if value is None:
            return None
        if isinstance(value, str):
            return JsonData(root=json.loads(value))
        if isinstance(value, (dict, list)):
            return JsonData(root=value)
        return JsonData(root={"value": value})

    def api_info(self) -> dict[str, Any]:
        return {
            "type": {},
            "description": "repair mask editor JSON payload with a PNG data URL",
        }

    def example_payload(self) -> Any:
        return {
            "image_id": "example",
            "revision": 0,
            "source_width": 1,
            "source_height": 1,
            "base_image": None,
            "mask_png": None,
            "preview_alpha": 0.45,
            "tool": "brush",
            "brush_size": 24,
        }

    def example_value(self) -> Any:
        return self.example_payload()
