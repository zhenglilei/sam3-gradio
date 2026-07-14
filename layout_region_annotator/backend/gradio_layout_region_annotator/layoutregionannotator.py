"""Gradio component for source-space layout Region lasso annotation."""

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


class LayoutRegionAnnotator(Component):
    """Canvas annotator whose browser input is restricted to client intent fields."""

    EVENTS = [Events.input]
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

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            return {
                "tool_mode": "browse",
                "lasso_polygon": [],
                "expected_regions_revision": None,
                "session_id": "",
                "layout_id": "",
                "source_mask_hash": "",
            }
        intent = raw.get("client_intent")
        if not isinstance(intent, dict):
            intent = raw
        tool_mode = intent.get("tool_mode")
        if tool_mode not in {"browse", "lasso"}:
            tool_mode = "browse"
        polygon = intent.get("lasso_polygon")
        if not isinstance(polygon, list):
            polygon = []
        revision = intent.get("expected_regions_revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            revision = None
        return {
            "tool_mode": tool_mode,
            "lasso_polygon": polygon,
            "expected_regions_revision": revision,
            "session_id": str(intent.get("session_id") or ""),
            "layout_id": str(intent.get("layout_id") or ""),
            "source_mask_hash": str(intent.get("source_mask_hash") or ""),
        }

    def postprocess(self, value: dict | list | str | None) -> JsonData | None:
        if value is None:
            return None
        if isinstance(value, str):
            return JsonData(root=json.loads(value))
        if isinstance(value, (dict, list)):
            return JsonData(root=value)
        return JsonData(root={"value": value})

    def api_info(self) -> dict[str, Any]:
        return {"type": {}, "description": "layout Region annotator client intent"}

    def example_payload(self) -> Any:
        return {"client_intent": {"tool_mode": "browse", "lasso_polygon": []}}

    def example_value(self) -> Any:
        return {
            "server_view": {"enabled": False, "status": "no source layout mask loaded"},
            "client_intent": {"tool_mode": "browse", "lasso_polygon": []},
        }
