"""Gradio custom component for browser-side tile-stitch alignment preview."""

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

_TILE_FIELDS = {"index", "x", "y", "width", "height"}
_REQUIRED_TILE_FIELDS = frozenset(_TILE_FIELDS)
_MAX_TILE_INDEX = 1_000_000
_MAX_TILE_COORDINATE = 10_000_000
_MAX_TILE_SIZE = 10_000_000
_MAX_STATUS_LENGTH = 1_024


class StitchPreviewCanvas(Component):
    """Canvas editor whose value is a JSON payload with tile images and alignment metadata."""

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
    def _unwrap(payload: Any) -> Any:
        return payload.root if isinstance(payload, JsonData) else payload

    @staticmethod
    def _coerce_integer(value: Any, *, minimum: int, maximum: int) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        integer = math.trunc(number)
        if integer < minimum or integer > maximum:
            return None
        return integer

    @staticmethod
    def _normalize_rotation(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            angle = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(angle):
            return None
        angle = (angle + 180.0) % 360.0 - 180.0
        if angle == -180.0:
            angle = 180.0
        return 0.0 if abs(angle) < 1e-9 else angle

    @classmethod
    def _sanitize_tile(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        if not _REQUIRED_TILE_FIELDS.issubset(value):
            return None
        index = cls._coerce_integer(
            value["index"], minimum=0, maximum=_MAX_TILE_INDEX
        )
        x = cls._coerce_integer(
            value["x"], minimum=-_MAX_TILE_COORDINATE, maximum=_MAX_TILE_COORDINATE
        )
        y = cls._coerce_integer(
            value["y"], minimum=-_MAX_TILE_COORDINATE, maximum=_MAX_TILE_COORDINATE
        )
        width = cls._coerce_integer(
            value["width"], minimum=1, maximum=_MAX_TILE_SIZE
        )
        height = cls._coerce_integer(
            value["height"], minimum=1, maximum=_MAX_TILE_SIZE
        )
        rotation = cls._normalize_rotation(value.get("rotation_deg", 0.0))
        if None in (index, x, y, width, height, rotation):
            return None
        return {
            "index": index,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation_deg": rotation,
        }

    @classmethod
    def _sanitize_value_fields(cls, raw: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        selected = cls._coerce_integer(
            raw.get("selected"), minimum=0, maximum=_MAX_TILE_INDEX
        )
        if selected is not None:
            result["selected"] = selected

        nudge_step = cls._coerce_integer(raw.get("nudge_step"), minimum=1, maximum=10)
        if nudge_step in (1, 5, 10):
            result["nudge_step"] = nudge_step

        if isinstance(raw.get("diff_mode"), bool):
            result["diff_mode"] = raw["diff_mode"]
        if isinstance(raw.get("show_loupe"), bool):
            result["show_loupe"] = raw["show_loupe"]

        drag_gain = raw.get("drag_gain")
        if not isinstance(drag_gain, bool):
            try:
                drag_gain_value = float(drag_gain)
            except (TypeError, ValueError, OverflowError):
                drag_gain_value = None
            if drag_gain_value is not None and math.isfinite(drag_gain_value):
                if 0.0 < drag_gain_value <= 10.0:
                    result["drag_gain"] = drag_gain_value

        status = raw.get("status")
        if isinstance(status, str) and len(status) <= _MAX_STATUS_LENGTH:
            result["status"] = status
        return result

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            return {}

        result = self._sanitize_value_fields(raw)

        tiles = []
        seen_indices: set[int] = set()
        for item in raw.get("tiles") or []:
            sanitized = self._sanitize_tile(item)
            if sanitized is None:
                continue
            index = sanitized["index"]
            if index in seen_indices:
                continue
            seen_indices.add(index)
            tiles.append(sanitized)
        if tiles:
            result["tiles"] = tiles
        return result

    def postprocess(self, value: dict | list | str | None) -> JsonData | None:
        if value is None:
            return None
        if isinstance(value, str):
            return JsonData(root=json.loads(value))
        if isinstance(value, (dict, list)):
            return JsonData(root=value)
        return JsonData(root={"value": value})

    def api_info(self) -> dict[str, Any]:
        return {"type": {}, "description": "stitch preview canvas json payload"}

    def example_payload(self) -> Any:
        return {"tiles": [], "selected": 0, "status": "no tiles loaded"}

    def example_value(self) -> Any:
        return {"tiles": [], "selected": 0, "status": "no tiles loaded"}
