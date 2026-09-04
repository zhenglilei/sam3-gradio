"""Gradio callbacks for the read-only overlap prediction view."""

from __future__ import annotations

from typing import Any

import gradio as gr

from sam3_demo.overlap_runtime import get_overlap_service
from sam3_demo.overlap_service import OverlapServiceError
from sam3_demo.overlap_store import OverlapStoreError


_EMPTY_MESSAGE = (
    "尚无已制作的叠层版图。展示模块可独立运行；录入数据后刷新列表即可查看。"
)


def _render_or_error(layout_id: str | None, layer_id: str | None):
    if not layout_id:
        return None, _EMPTY_MESSAGE
    service = get_overlap_service()
    choices = service.layer_choices(layout_id)
    valid_ids = {value for _, value in choices}
    selected = layer_id if layer_id in valid_ids else (choices[0][1] if choices else None)
    if not selected:
        return None, "该版图至少需要两层标注，才能预测下一层交叠位置。"
    result = service.render(layout_id, selected)
    return result["preview"], result["report"]


def refresh_overlap_view(selected_layout_id: str | None):
    """Refresh shared layouts and render a stable or first selection."""
    try:
        service = get_overlap_service()
        choices = service.layout_choices()
        valid_ids = {value for _, value in choices}
        selected = (
            selected_layout_id
            if selected_layout_id in valid_ids
            else (choices[0][1] if choices else None)
        )
        if not selected:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                None,
                _EMPTY_MESSAGE,
            )
        layer_choices = service.layer_choices(selected)
        layer_id = layer_choices[0][1] if layer_choices else None
        preview, report = _render_or_error(selected, layer_id)
        return (
            gr.update(choices=choices, value=selected),
            gr.update(choices=layer_choices, value=layer_id),
            preview,
            report,
        )
    except (OverlapStoreError, OverlapServiceError, ValueError) as exc:
        return (
            gr.update(),
            gr.update(choices=[], value=None),
            None,
            f"叠层版图读取失败：{exc}",
        )


def select_overlap_layout(layout_id: str | None):
    """Load the first valid current layer for one persisted layout."""
    if not layout_id:
        return gr.update(choices=[], value=None), None, _EMPTY_MESSAGE
    try:
        service = get_overlap_service()
        choices = service.layer_choices(layout_id)
        layer_id = choices[0][1] if choices else None
        preview, report = _render_or_error(layout_id, layer_id)
        return gr.update(choices=choices, value=layer_id), preview, report
    except (OverlapStoreError, OverlapServiceError, ValueError) as exc:
        return (
            gr.update(choices=[], value=None),
            None,
            f"叠层版图读取失败：{exc}",
        )


def render_overlap_layer(layout_id: str | None, layer_id: str | None):
    """Render a user-selected current-to-next pair."""
    try:
        return _render_or_error(layout_id, layer_id)
    except (OverlapStoreError, OverlapServiceError, ValueError) as exc:
        return None, f"叠层预测失败：{exc}"


def bind_overlap_prediction_events(*, demo: gr.Blocks, refs: Any) -> None:
    """Bind the standalone view without touching existing demo callbacks."""
    event_kwargs = {
        "queue": True,
        "show_progress": "minimal",
        "concurrency_limit": 2,
        "concurrency_id": "overlap-read-view",
        "api_visibility": "private",
    }
    demo.load(
        fn=refresh_overlap_view,
        inputs=[refs.layout_selector],
        outputs=[
            refs.layout_selector,
            refs.layer_selector,
            refs.preview,
            refs.report,
        ],
        **event_kwargs,
    )
    refs.refresh_btn.click(
        fn=refresh_overlap_view,
        inputs=[refs.layout_selector],
        outputs=[
            refs.layout_selector,
            refs.layer_selector,
            refs.preview,
            refs.report,
        ],
        **event_kwargs,
    )
    refs.layout_selector.input(
        fn=select_overlap_layout,
        inputs=[refs.layout_selector],
        outputs=[refs.layer_selector, refs.preview, refs.report],
        **event_kwargs,
    )
    refs.layer_selector.input(
        fn=render_overlap_layer,
        inputs=[refs.layout_selector, refs.layer_selector],
        outputs=[refs.preview, refs.report],
        **event_kwargs,
    )


__all__ = [
    "bind_overlap_prediction_events",
    "refresh_overlap_view",
    "render_overlap_layer",
    "select_overlap_layout",
]
