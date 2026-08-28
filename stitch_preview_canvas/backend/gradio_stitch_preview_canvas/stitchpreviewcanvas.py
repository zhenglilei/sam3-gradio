"""Gradio custom component for browser-side tile-stitch alignment preview."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from gradio.components.base import Component
from gradio.components.json_component import JsonData
from gradio.events import Events
from gradio.i18n import I18nData

if TYPE_CHECKING:
    from gradio.components import Timer

_TILE_FIELDS = {"index", "x", "y", "width", "height"}
_VALUE_FIELDS = {
    "selected",
    "nudge_step",
    "diff_mode",
    "show_loupe",
    "drag_gain",
    "status",
}


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
    def _sanitize_tile(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        if "index" not in value:
            return None
        try:
            index = int(value["index"])
        except (TypeError, ValueError):
            return None
        tile: dict[str, Any] = {"index": index}
        for key in _TILE_FIELDS:
            if key != "index" and key in value:
                tile[key] = value[key]
        return tile

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            return {}

        result: dict[str, Any] = {}
        for field in _VALUE_FIELDS:
            if field in raw:
                result[field] = raw[field]

        tiles = []
        for item in raw.get("tiles") or []:
            sanitized = self._sanitize_tile(item)
            if sanitized is not None:
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
