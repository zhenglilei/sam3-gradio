"""Gradio custom component for browser-side layout mask transform editing."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from gradio.components.base import Component
from gradio.components.json_component import JsonData
from gradio.events import Events
from gradio.i18n import I18nData

if TYPE_CHECKING:
    from gradio.components import Timer


_TRANSFORM_INTENT_FIELDS = {
    "transform_version",
    "session_id",
    "layout_id",
    "image_id",
    "revision",
    "center_x",
    "center_y",
    "pivot_x",
    "pivot_y",
    "scale",
    "rotation_deg",
    "preview_alpha",
    "source_mask_pixel_sha256",
    "target_image_sha256",
    "origin",
    "group_id",
    "region_id",
    "label",
    "group_mask_pixel_sha256",
}


class LayoutTransformEditor(Component):
    """Canvas editor whose value is a JSON payload with image URLs and transform metadata."""

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
    def _sanitize_transform(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        return {
            key: value[key]
            for key in _TRANSFORM_INTENT_FIELDS
            if key in value
        }

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            return {}

        result: dict[str, Any] = {}
        for field in ("enabled", "target_width", "target_height"):
            if field in raw:
                result[field] = raw[field]
        transform = self._sanitize_transform(raw.get("transform"))
        if transform is not None:
            result["transform"] = transform

        if raw.get("transform_mode") != "label_groups":
            return result
        result["transform_mode"] = "label_groups"
        intent = raw.get("group_intent")
        if not isinstance(intent, dict):
            return result
        transforms = []
        for item in intent.get("transforms") or []:
            if not isinstance(item, dict) or "group_id" not in item:
                continue
            sanitized = self._sanitize_transform(item.get("transform"))
            if sanitized is None:
                continue
            transforms.append(
                {
                    "group_id": item["group_id"],
                    "transform": sanitized,
                }
            )
        sanitized_intent = {
            "selection_signature": intent.get("selection_signature"),
            "transform_set_revision": intent.get("transform_set_revision"),
            "active_group_id": intent.get("active_group_id"),
            "transforms": transforms,
        }
        changed_group_ids = intent.get("changed_group_ids")
        if isinstance(changed_group_ids, list):
            sanitized_intent["changed_group_ids"] = [
                value
                for value in changed_group_ids
                if isinstance(value, (str, int))
                and not isinstance(value, bool)
                and value != ""
            ]
        result["group_intent"] = sanitized_intent
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
        return {"type": {}, "description": "layout transform editor json payload"}

    def example_payload(self) -> Any:
        return {"enabled": False, "status": "no layout mask loaded"}

    def example_value(self) -> Any:
        return {"enabled": False, "status": "no layout mask loaded"}
