"""Transparent Gradio image gesture overlay with a client-intent boundary."""

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


class ImageGestureOverlay(Component):
    """Overlay that emits sanitized click or drag intent without image pixels."""

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
        container: bool = False,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool | Literal["hidden"] = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
        preserved_by_key: list[str] | str | None = "value",
        height: int | str = 320,
        target_elem_id: str = "",
    ):
        self.height = height
        self.target_elem_id = target_elem_id
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
    def _point(value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return []
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            return []
        point = [float(value[0]), float(value[1])]
        return point if all(math.isfinite(item) for item in point) else []

    def preprocess(self, payload: Any) -> dict[str, Any]:
        """Return only browser-owned gesture intent fields."""

        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            raw = {}
        intent = raw.get("client_intent")
        if not isinstance(intent, dict):
            intent = raw

        gesture = intent.get("gesture")
        if gesture not in {"click", "drag"}:
            gesture = ""
        revision = intent.get("expected_revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            revision = None

        return {
            "gesture": gesture,
            "start_xy": self._point(intent.get("start_xy")),
            "end_xy": self._point(intent.get("end_xy")),
            "expected_revision": revision,
            "image_id": str(intent.get("image_id") or ""),
            "image_sha256": str(intent.get("image_sha256") or ""),
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
        return {"type": {}, "description": "sanitized image click or drag intent"}

    def example_payload(self) -> Any:
        return {"client_intent": {"gesture": "click", "start_xy": [10, 20], "end_xy": [10, 20]}}

    def example_value(self) -> Any:
        return {
            "server_view": {
                "enabled": False,
                "natural_width": 1,
                "natural_height": 1,
                "revision": 0,
                "interaction": "disabled",
            },
            "client_intent": {},
        }
