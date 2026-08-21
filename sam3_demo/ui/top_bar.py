"""Global SAM3 model lifecycle status bar.

The top bar deliberately has no dependency on the model runtime.  The two
callbacks supplied by the application are invoked only by the Gradio events:
the timer reads a cheap status snapshot and the button requests an
asynchronous model start.  In particular, constructing the UI never loads a
model or probes a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from html import escape
from typing import Any

import gradio as gr

from .refs import ComponentRefs


_RED_STATES = frozenset({"UNLOADED", "ERROR"})
_YELLOW_STATES = frozenset(
    {
        "SEARCHING_GPU",
        "WAITING_GPU",
        "LOADING",
        "WARMUP",
        "REHYDRATING",
        "STOPPING",
    }
)
_GREEN_STATES = frozenset({"READY", "RUNNING"})
_MAX_ERROR_LENGTH = 160


def _compact_error(value: Any, *, limit: int = _MAX_ERROR_LENGTH) -> str:
    """Return a short, single-line representation safe for HTML output."""

    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _status_presentation(snapshot: Mapping[str, Any] | None) -> tuple[str, str, str, bool]:
    """Map a Supervisor snapshot to ``(state, label, colour, enabled)``."""

    data = snapshot if isinstance(snapshot, Mapping) else {}
    state = str(data.get("state") or "UNLOADED").upper()
    if state in _RED_STATES:
        colour = "#dc2626"
        enabled = True
        if state == "ERROR":
            detail = _compact_error(data.get("last_error"))
            label = f"模型加载失败：{detail}" if detail else "模型加载失败"
        else:
            label = "模型未加载"
    elif state in _YELLOW_STATES:
        colour = "#d97706"
        enabled = False
        label = "等待 GPU" if state == "WAITING_GPU" else "启动中"
    elif state in _GREEN_STATES:
        colour = "#16a34a"
        enabled = False
        label = "推理中" if state == "RUNNING" else "模型已加载"
    else:
        colour = "#dc2626"
        enabled = True
        label = f"模型状态未知：{_compact_error(state)}"
    return state, label, colour, enabled


def _render_status(snapshot: Mapping[str, Any] | None) -> str:
    """Render a compact status light with accessible text."""

    state, label, colour, _enabled = _status_presentation(snapshot)
    safe_label = escape(label, quote=True)
    safe_colour = escape(colour, quote=True)
    tone = "green" if state in _GREEN_STATES else "yellow" if state in _YELLOW_STATES else "red"
    return (
        f'<div class="sam3-model-status sam3-model-status--{tone}" '
        f'style="--sam3-status-colour:{safe_colour};" '
        f'aria-label="{safe_label}">'
        '<span class="sam3-model-status-light" aria-hidden="true"></span>'
        f'<span class="sam3-model-status-text">{safe_label}</span>'
        "</div>"
    )


def _button_update(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a Gradio update for the single start/restart button."""

    state, _label, _colour, enabled = _status_presentation(snapshot)
    value = "重新启动" if state == "ERROR" else "启动模型"
    if state in _YELLOW_STATES:
        value = "启动中"
    elif state in _GREEN_STATES:
        value = "模型已加载" if state == "READY" else "推理中"
    return gr.update(value=value, interactive=enabled)


def _model_status_view(snapshot_fn: Callable[[], Mapping[str, Any] | None]):
    """Timer callback; only reads the injected O(1) snapshot function."""

    snapshot = snapshot_fn()
    return _render_status(snapshot), _button_update(snapshot)


def _request_model_start(request_start_fn: Callable[[], Mapping[str, Any] | None]):
    """Start-button callback; the injected function must be non-blocking."""

    snapshot = request_start_fn()
    return _render_status(snapshot), _button_update(snapshot)


def build_top_bar(
    *,
    snapshot_fn: Callable[[], Mapping[str, Any] | None],
    request_start_fn: Callable[[], Mapping[str, Any] | None],
    title: str = "SAM3 交互式视觉工作台",
    subtitle: str = "基于 SAM3 的 PCS 自动概念分割与 PVS 手动实例分割工作台",
) -> ComponentRefs:
    """Build the global model status bar and register its two events.

    ``snapshot_fn`` and ``request_start_fn`` are called only after a browser
    event.  The initial card is deliberately rendered as ``UNLOADED`` so UI
    construction remains side-effect free.
    """

    with gr.Row(equal_height=True, elem_id="sam3_model_top_bar"):
        with gr.Column(scale=1, min_width=280, elem_classes="sam3-model-controls"):
            with gr.Row(equal_height=True, elem_classes="sam3-model-controls-row"):
                model_status = gr.HTML(
                    value=_render_status({"state": "UNLOADED"}),
                    label="模型状态",
                    show_label=False,
                    elem_id="sam3_model_status",
                )
                model_start_btn = gr.Button(
                    "启动模型",
                    variant="secondary",
                    interactive=True,
                    elem_id="sam3_model_start",
                )
        with gr.Column(scale=2, min_width=480, elem_classes="sam3-model-heading"):
            gr.Markdown(f"# {title}")
            gr.Markdown(subtitle, elem_classes="description")
        with gr.Column(scale=1, min_width=280, elem_classes="sam3-model-spacer"):
            gr.Markdown("")

    # Timer is an invisible browser-side source, but it must be rendered into
    # the Blocks config for its periodic event to run in the browser.
    model_status_timer = gr.Timer(
        value=2,
        active=True,
        render=True,
    )
    model_status_timer.tick(
        fn=lambda: _model_status_view(snapshot_fn),
        inputs=None,
        outputs=[model_status, model_start_btn],
        queue=False,
        show_progress="hidden",
        concurrency_limit=1,
    )
    model_start_btn.click(
        fn=lambda: _request_model_start(request_start_fn),
        inputs=None,
        outputs=[model_status, model_start_btn],
        queue=False,
        show_progress="hidden",
        concurrency_limit=1,
    )
    return ComponentRefs(
        model_status=model_status,
        model_start_btn=model_start_btn,
        model_status_timer=model_status_timer,
    )


__all__ = [
    "build_top_bar",
    "_button_update",
    "_model_status_view",
    "_render_status",
    "_request_model_start",
    "_status_presentation",
]
