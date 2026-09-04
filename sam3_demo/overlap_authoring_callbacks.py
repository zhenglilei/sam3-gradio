"""Gradio adapters for the optional manual overlap-layout authoring flow.

The callbacks in this module are deliberately independent from the viewer,
PVS state, and the application event binder.  A draft is kept in a single
``gr.State`` value; the service owns validation and returns immutable copies.
The editor payload is rebuilt from that draft on every response, so the
browser never becomes the source of truth for images or masks.
"""

from __future__ import annotations

import copy
import uuid
from io import BytesIO
from typing import Any, Callable, Mapping

import gradio as gr
from PIL import Image, UnidentifiedImageError

import sam3_demo.overlap_runtime as overlap_runtime
from sam3_demo.overlap_authoring import (
    AUTHORING_SCHEMA_VERSION,
    OverlapAuthoringError,
    OverlapAuthoringService,
)
from sam3_demo.overlap_store import OverlapStoreError


# The first six values are intentionally stable so an authoring panel can bind
# all draft controls through one helper.  Save/load/delete append a seventh
# value containing the shared viewer's layout-selector update.
AUTHORING_VIEW_OUTPUTS = 6
AUTHORING_PERSIST_OUTPUTS = 7
_EMPTY_STATUS = "已新建空白制作草稿，请上传整体版图。"


def _service() -> OverlapAuthoringService:
    """Resolve the lazy authoring service at callback time, not import time."""

    return overlap_runtime.get_overlap_authoring_service()


def _offline_empty_draft() -> dict[str, Any]:
    """Create a valid empty state even while the runtime is unavailable."""

    return {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "draft_id": str(uuid.uuid4()),
        "layout_id": None,
        "store_revision": None,
        "draft_revision": 0,
        "name": "",
        "base_image_png": None,
        "width": 0,
        "height": 0,
        "layers": [],
        "active_layer_id": None,
    }


def _as_empty_draft(service: OverlapAuthoringService, draft: Any) -> dict[str, Any]:
    """Return a fresh draft only for the explicit empty-state inputs."""

    if draft is None:
        return service.new_draft()
    if isinstance(draft, dict) and not draft:
        return service.new_draft()
    if not isinstance(draft, dict):
        raise OverlapAuthoringError("制作草稿无效")
    return copy.deepcopy(draft)


def _base_image(draft: Mapping[str, Any]) -> Image.Image | None:
    payload = draft.get("base_image_png")
    if not payload:
        return None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise OverlapAuthoringError("整体版图数据无效")
    try:
        with Image.open(BytesIO(bytes(payload))) as source:
            if source.format != "PNG":
                raise OverlapAuthoringError("整体版图不是 PNG")
            source.load()
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError, TypeError) as exc:
        raise OverlapAuthoringError("整体版图无法显示") from exc


def _layer_choices(draft: Mapping[str, Any]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ordinal, layer in enumerate(draft.get("layers") or []):
        if not isinstance(layer, Mapping):
            raise OverlapAuthoringError("制作草稿层数据无效")
        layer_id = layer.get("id")
        if not isinstance(layer_id, str) or not layer_id:
            raise OverlapAuthoringError("制作草稿层 ID 无效")
        choices.append(
            (
                f"{ordinal + 1}. {layer.get('name', '')} "
                f"({layer.get('layer_type', '')})",
                layer_id,
            )
        )
    return choices


def _with_status(view: tuple[Any, ...], status: str) -> tuple[Any, ...]:
    values = list(view)
    values[-1] = str(status)
    return tuple(values)


def draft_to_view(
    draft: Any,
    *,
    service: OverlapAuthoringService | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any], str]:
    """Convert one draft into all controls needed by the authoring panel.

    Returns ``(draft_state, name, base_image, layer_selector_update,
    editor_payload, status_text)``.  ``None`` and ``{}`` are stable empty
    inputs.  Images and masks remain in the server-side draft; only the
    service-created editor payload contains data URLs for display.
    """

    svc = service or _service()
    current = _as_empty_draft(svc, draft)
    editor_payload = svc.editor_payload(current)
    choices = _layer_choices(current)
    valid_ids = {value for _label, value in choices}
    active_id = current.get("active_layer_id")
    if active_id not in valid_ids:
        active_id = choices[0][1] if choices else None
    selector_update = gr.update(choices=choices, value=active_id)
    status = str(editor_payload.get("status") or _EMPTY_STATUS)
    return (
        current,
        str(current.get("name") or ""),
        _base_image(current),
        selector_update,
        editor_payload,
        status,
    )


def _failed_view(
    previous: Any,
    service: OverlapAuthoringService | None,
    operation: str,
    error: Exception,
) -> tuple[Any, ...]:
    """Return the previous draft with an explanatory status after a failure."""

    try:
        if service is None:
            raise OverlapStoreError("authoring runtime unavailable")
        view = draft_to_view(previous, service=service)
    except (OverlapAuthoringError, OverlapStoreError):
        # A malformed external value cannot be preserved as a usable draft;
        # keep the UI recoverable while still reporting the original failure.
        view = _fallback_view(previous)
    return _with_status(view, f"{operation}失败：{error}")


def _fallback_view(previous: Any) -> tuple[Any, ...]:
    """Render a safe, local-only view when the runtime cannot be resolved."""

    current = (
        copy.deepcopy(previous)
        if isinstance(previous, dict)
        else _offline_empty_draft()
    )
    try:
        choices = _layer_choices(current)
    except (OverlapAuthoringError, KeyError, TypeError):
        choices = []
    valid_ids = {value for _label, value in choices}
    active_id = current.get("active_layer_id")
    if active_id not in valid_ids:
        active_id = choices[0][1] if choices else None
    try:
        base = _base_image(current)
    except OverlapAuthoringError:
        base = None
    payload = {"enabled": False, "status": "制作服务暂时不可用，请稍后重试。"}
    return (
        current,
        str(current.get("name") or ""),
        base,
        gr.update(choices=choices, value=active_id),
        payload,
        payload["status"],
    )


def _mutate(
    previous: Any,
    operation: str,
    fn: Callable[[OverlapAuthoringService, Any], dict[str, Any]],
) -> tuple[Any, ...]:
    service: OverlapAuthoringService | None = None
    try:
        service = _service()
        updated = fn(service, previous)
        return draft_to_view(updated, service=service)
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return _failed_view(previous, service, operation, exc)


def new_authoring_draft() -> tuple[Any, ...]:
    service: OverlapAuthoringService | None = None
    try:
        service = _service()
        return _with_status(
            draft_to_view(service.new_draft(), service=service), _EMPTY_STATUS
        )
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return _failed_view(None, service, "新建草稿", exc)


def set_authoring_name(name: Any, draft: Any) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "设置版图名称",
        lambda service, previous: service.set_name(previous, name),
    )


def set_authoring_base_image(image: Any, draft: Any) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "上传整体版图",
        lambda service, previous: service.set_base_image(previous, image),
    )


def add_authoring_layer(
    layer_name: Any,
    layer_type: Any,
    mask: Any,
    draft: Any,
) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "添加叠层",
        lambda service, previous: service.add_layer(
            previous, mask, layer_type, layer_name
        ),
    )


def select_authoring_layer(layer_id: Any, draft: Any) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "选择叠层",
        lambda service, previous: service.set_active_layer(previous, layer_id),
    )


def remove_authoring_layer(layer_id: Any, draft: Any) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "删除叠层",
        lambda service, previous: service.remove_layer(previous, layer_id),
    )


def apply_authoring_editor(editor_payload: Any, draft: Any) -> tuple[Any, ...]:
    return _mutate(
        draft,
        "应用叠层变换",
        lambda service, previous: service.apply_editor_intent(
            previous, editor_payload
        ),
    )


def _layout_choices_update(
    selected_layout_id: str | None = None,
) -> dict[str, Any]:
    """Return a Gradio update sourced from the shared read-only service."""

    choices = overlap_runtime.get_overlap_service().layout_choices()
    valid_ids = {value for _label, value in choices}
    selected = (
        selected_layout_id
        if selected_layout_id in valid_ids
        else (choices[0][1] if choices else None)
    )
    return gr.update(choices=choices, value=selected)


def refresh_authoring_layouts(
    selected_layout_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Refresh persisted-layout choices without touching the current draft."""

    try:
        return (
            _layout_choices_update(selected_layout_id),
            "已刷新已保存版图列表。",
        )
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return gr.update(), f"刷新版图列表失败：{exc}"


def load_authoring_layout(
    layout_id: str | None,
    draft: Any = None,
) -> tuple[Any, ...]:
    """Load a persisted layout into a new draft and refresh the selector."""

    service: OverlapAuthoringService | None = None
    try:
        service = _service()
        if not layout_id:
            raise OverlapAuthoringError("请选择要加载的版图")
        loaded = service.load_layout(str(layout_id))
        view = draft_to_view(loaded, service=service)
        view = _with_status(view, f"已加载版图：{view[1]}。")
        return (*view, _layout_choices_update(str(layout_id)))
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return (
            *_failed_view(draft, service, "加载版图", exc),
            gr.update(),
        )


def save_authoring_layout(
    draft: Any,
    layout_name: Any = None,
) -> tuple[Any, ...]:
    """Create/update a layout and return the latest shared selector update."""

    service: OverlapAuthoringService | None = None
    try:
        service = _service()
        current = (
            service.set_name(draft, layout_name)
            if layout_name is not None
            else draft
        )
        saved = service.save(current)
        view = draft_to_view(saved, service=service)
        view = _with_status(view, f"版图已保存：{view[1]}。")
        return (*view, _layout_choices_update(saved.get("layout_id")))
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return (*_failed_view(draft, service, "保存版图", exc), gr.update())


def delete_authoring_layout(draft: Any) -> tuple[Any, ...]:
    """Soft-delete the current persisted layout and clear its draft."""

    service: OverlapAuthoringService | None = None
    try:
        service = _service()
        cleared = service.delete(draft)
        view = draft_to_view(cleared, service=service)
        view = _with_status(view, "已删除当前已保存版图，制作草稿已清空。")
        return (*view, _layout_choices_update(None))
    except (OverlapAuthoringError, OverlapStoreError) as exc:
        return (*_failed_view(draft, service, "删除版图", exc), gr.update())


__all__ = [
    "AUTHORING_PERSIST_OUTPUTS",
    "AUTHORING_VIEW_OUTPUTS",
    "add_authoring_layer",
    "apply_authoring_editor",
    "delete_authoring_layout",
    "draft_to_view",
    "load_authoring_layout",
    "new_authoring_draft",
    "refresh_authoring_layouts",
    "remove_authoring_layer",
    "save_authoring_layout",
    "select_authoring_layer",
    "set_authoring_base_image",
    "set_authoring_name",
]
