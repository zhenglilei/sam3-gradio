"""Session-safe callbacks for the hole-pattern template-stitching tab."""

from __future__ import annotations

from typing import Any, Callable

import gradio as gr
from PIL import Image

from sam3_demo.template_stitch_workflow import (
    load_template_images,
    stitch_template_group,
)


PublishMosaic = Callable[[Image.Image, str], str]


def _revision(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def new_template_stitch_state(
    session_id: str = "",
    owner_token: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": str(session_id or ""),
        "owner_token": str(owner_token or ""),
        "revision": 0,
        "generated_revision": None,
        "mosaic": None,
        "meta": {},
        "status": "请上传与网格数量一致的一组图片",
    }


def _fresh_owned_state(previous) -> dict[str, Any]:
    old = previous if isinstance(previous, dict) else {}
    state = new_template_stitch_state(
        str(old.get("session_id") or ""),
        str(old.get("owner_token") or ""),
    )
    state["revision"] = _revision(old.get("revision")) + 1
    return state


def _grid_size(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是 1–6 的整数")
    number = int(value)
    if float(value) != number or not 1 <= number <= 6:
        raise ValueError(f"{name}必须是 1–6 的整数")
    return number


def _failure_updates(state, status):
    state["mosaic"] = None
    state["meta"] = {}
    state["generated_revision"] = None
    state["status"] = status
    return (
        state,
        gr.update(value=None),
        {},
        gr.update(value=None, visible=False),
        status,
        gr.update(interactive=False),
        "",
    )


def run_template_stitch(
    files,
    rows,
    cols,
    template_state,
    *,
    publish_mosaic: PublishMosaic | None = None,
):
    state = _fresh_owned_state(template_state)
    try:
        grid_rows = _grid_size(rows, "行数")
        grid_cols = _grid_size(cols, "列数")
        images, names = load_template_images(files)
        required = grid_rows * grid_cols
        if len(images) != required:
            raise ValueError(
                f"当前网格 {grid_rows}×{grid_cols} 需要 {required} 张图，实际上传 {len(images)} 张"
            )
        mosaic, meta = stitch_template_group(
            images,
            names,
            grid_rows,
            grid_cols,
        )
        if not isinstance(mosaic, Image.Image) or mosaic.width <= 0 or mosaic.height <= 0:
            raise ValueError("模板拼接没有生成有效图像")
        state["mosaic"] = mosaic.convert("RGB")
        state["meta"] = dict(meta or {})
        state["generated_revision"] = _revision(state.get("revision"))
        output_path = None
        publish_warning = ""
        if publish_mosaic is not None:
            try:
                output_path = publish_mosaic(
                    state["mosaic"],
                    str(state.get("session_id") or ""),
                )
            except Exception as exc:
                publish_warning = f"；下载文件发布失败：{exc}"
        period = float(state["meta"].get("period") or 0.0)
        method = str(state["meta"].get("layout_method") or "unknown")
        status = (
            f"模板拼接完成 {mosaic.width}×{mosaic.height}；"
            f"孔周期约 {period:.2f}px；布局方法 {method}{publish_warning}"
        )
        state["status"] = status
        return (
            state,
            gr.update(value=state["mosaic"]),
            state["meta"],
            gr.update(value=output_path, visible=bool(output_path)),
            status,
            gr.update(interactive=True),
            "",
        )
    except Exception as exc:
        return _failure_updates(state, f"模板拼接失败：{exc}")


def mosaic_for_handoff(template_state) -> Image.Image:
    state = template_state if isinstance(template_state, dict) else {}
    generated = state.get("generated_revision")
    current = _revision(state.get("revision"))
    mosaic = state.get("mosaic")
    if generated is None or _revision(generated) != current or not isinstance(mosaic, Image.Image):
        raise ValueError("请先生成当前模板拼接结果")
    return mosaic.convert("RGB").copy()


def handoff_status(source_status: str) -> str:
    text = str(source_status or "")
    if text.startswith("已载入完整原图"):
        return "模板拼接结果已显示在智能图像分割的上传与裁剪区域。"
    return f"导入失败：{text or '未知错误'}"
