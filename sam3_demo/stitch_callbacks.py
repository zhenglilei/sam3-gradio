"""Callbacks for the stitch-and-layout-align tab."""

from __future__ import annotations

from typing import Any, Callable

import gradio as gr
import numpy as np
from PIL import Image

import layout_transform_utils as _layout_tx
from sam3_demo.stitch_workflow import (
    STITCH_LAYOUTS,
    auto_align_images,
    canvas_payload,
    default_shifts_for_layout,
    empty_canvas_payload,
    export_mosaic,
    image_to_data_url,
    load_images_from_files,
    normalize_layout,
    pil_rgb,
    select_worst_tile,
    selected_from_canvas_payload,
    shifts_from_canvas_payload,
)


CacheGet = Callable[[dict], dict]


def new_stitch_state() -> dict[str, Any]:
    return {
        "images": [],
        "shifts": [],
        "layout": "horizontal",
        "selected": 0,
        "mosaic": None,
        "logs": [],
        "warnings": [],
        "nudge_step": 1,
        "diff_mode": False,
        "show_loupe": True,
        "blend": True,
        "crop_periodic": False,
        "step": 1,
        "status": "请上传一组已去 overlay 的分块图",
    }


def _images(state: dict | None) -> list[Image.Image]:
    state = state if isinstance(state, dict) else {}
    out = []
    for item in state.get("images") or []:
        img = pil_rgb(item)
        if img is not None:
            out.append(img)
    return out


def _status_text(logs: list[str], extra: str = "") -> str:
    lines = [line for line in logs if line]
    if extra:
        lines.append(extra)
    return "\n".join(lines) if lines else "就绪"


def _payload(state: dict, extra_status: str = "") -> dict:
    images = _images(state)
    shifts = list(state.get("shifts") or [])
    if images and len(shifts) != len(images):
        shifts = default_shifts_for_layout(images, state.get("layout") or "horizontal")
        state["shifts"] = shifts
    return canvas_payload(
        images,
        shifts,
        selected=int(state.get("selected") or 0),
        nudge_step=int(state.get("nudge_step") or 1),
        diff_mode=bool(state.get("diff_mode")),
        show_loupe=state.get("show_loupe", True),
        status=extra_status or state.get("status") or "",
    )


def _selected_xy(state: dict) -> tuple[float, float]:
    shifts = list(state.get("shifts") or [])
    selected = int(state.get("selected") or 0)
    if not shifts:
        return 0.0, 0.0
    selected = max(0, min(selected, len(shifts) - 1))
    x, y = shifts[selected]
    return float(x), float(y)


def _step2_visible(state: dict) -> bool:
    return bool(isinstance(state, dict) and state.get("mosaic") is not None)


def load_tiles(files, layout, stitch_state, nudge_step, diff_mode, show_loupe, blend, crop_periodic):
    state = new_stitch_state()
    if isinstance(stitch_state, dict):
        state["step"] = stitch_state.get("step") or 1
    layout_key = normalize_layout(layout)
    images = load_images_from_files(files)
    if not images:
        status = "没有读到图片。请上传 png/jpg/bmp/tif。"
        state["status"] = status
        return (
            state,
            empty_canvas_payload(status),
            0.0,
            0.0,
            status,
            None,
            None,
            empty_layout_payload(status),
        )
    shifts = default_shifts_for_layout(images, layout_key)
    state.update(
        {
            "images": images,
            "shifts": shifts,
            "layout": layout_key,
            "selected": 0,
            "mosaic": None,
            "logs": [f"已加载 {len(images)} 张，布局 {layout_key}，尚未自动对齐"],
            "nudge_step": int(nudge_step or 1),
            "diff_mode": bool(diff_mode),
            "show_loupe": bool(show_loupe if show_loupe is not None else True),
            "blend": bool(blend),
            "crop_periodic": bool(crop_periodic),
            "step": 1,
        }
    )
    status = _status_text(state["logs"])
    state["status"] = status
    dx, dy = _selected_xy(state)
    return (
        state,
        _payload(state, status),
        dx,
        dy,
        status,
        None,
        None,
        empty_layout_payload("请先生成拼接结果"),
    )


def auto_align(stitch_state, layout, nudge_step, diff_mode, show_loupe):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    layout_key = normalize_layout(layout or state.get("layout"))
    state["layout"] = layout_key
    state["nudge_step"] = int(nudge_step or state.get("nudge_step") or 1)
    state["diff_mode"] = bool(diff_mode)
    state["show_loupe"] = bool(show_loupe if show_loupe is not None else True)
    if not images:
        status = "请先加载分块图"
        return state, empty_canvas_payload(status), 0.0, 0.0, status
    try:
        shifts, logs = auto_align_images(images, layout_key)
    except Exception as exc:
        status = f"自动对齐失败：{exc}"
        state["status"] = status
        return state, _payload(state, status), *_selected_xy(state), status
    baseline = default_shifts_for_layout(images, layout_key)
    selected = select_worst_tile(shifts, baseline)
    state["shifts"] = shifts
    state["logs"] = logs
    state["selected"] = selected
    state["mosaic"] = None
    status = _status_text(logs, f"已自动选中偏移最大的图{selected + 1}，可用方向键 1px 微调")
    state["status"] = status
    dx, dy = _selected_xy(state)
    return state, _payload(state, status), dx, dy, status


def canvas_changed(payload, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    if not images:
        return state, 0.0, 0.0, state.get("status") or "请先加载分块图"
    shifts = shifts_from_canvas_payload(payload, state.get("shifts") or [])
    if len(shifts) == len(images):
        state["shifts"] = shifts
    selected = selected_from_canvas_payload(payload, len(images))
    state["selected"] = selected
    if isinstance(payload, dict):
        if payload.get("nudge_step") in (1, 5, 10):
            state["nudge_step"] = int(payload["nudge_step"])
        if "diff_mode" in payload:
            state["diff_mode"] = bool(payload["diff_mode"])
        if "show_loupe" in payload:
            state["show_loupe"] = bool(payload["show_loupe"])
    dx, dy = _selected_xy(state)
    status = f"图{selected + 1} 位置=({int(dx)}, {int(dy)})"
    state["status"] = status
    return state, dx, dy, status


def apply_numeric_shift(dx, dy, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    shifts = list(state.get("shifts") or [])
    if not images or not shifts:
        status = "请先加载分块图"
        return state, empty_canvas_payload(status), 0.0, 0.0, status
    selected = max(0, min(int(state.get("selected") or 0), len(shifts) - 1))
    try:
        nx = int(round(float(dx)))
        ny = int(round(float(dy)))
    except (TypeError, ValueError):
        nx, ny = shifts[selected]
    shifts[selected] = (nx, ny)
    state["shifts"] = shifts
    status = f"已把图{selected + 1} 设为 ({nx}, {ny})"
    state["status"] = status
    return state, _payload(state, status), float(nx), float(ny), status


def apply_canvas_options(nudge_step, diff_mode, show_loupe, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    state["nudge_step"] = int(nudge_step or 1)
    state["diff_mode"] = bool(diff_mode)
    state["show_loupe"] = bool(show_loupe if show_loupe is not None else True)
    images = _images(state)
    if not images:
        return state, empty_canvas_payload(state.get("status") or "请先加载分块图"), state.get("status") or ""
    status = f"步长 {state['nudge_step']}px；差分={'开' if state['diff_mode'] else '关'}；放大镜={'开' if state['show_loupe'] else '关'}"
    state["status"] = status
    return state, _payload(state, status), status


def generate_mosaic(stitch_state, blend, crop_periodic, layout_state=None, cache_get=None):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    shifts = list(state.get("shifts") or [])
    if not images or len(shifts) != len(images):
        status = "请先加载并对齐分块图"
        state["status"] = status
        return state, None, None, status, empty_layout_payload(status)
    state["blend"] = bool(blend)
    state["crop_periodic"] = bool(crop_periodic)
    try:
        mosaic, warn = export_mosaic(
            images,
            shifts,
            layout=state.get("layout") or "horizontal",
            blend=bool(blend),
            crop_periodic=bool(crop_periodic),
        )
    except Exception as exc:
        status = f"生成拼接失败：{exc}"
        state["status"] = status
        return state, None, None, status, empty_layout_payload(status)
    state["mosaic"] = mosaic
    state["warnings"] = warn
    state["step"] = 2
    extra = "；".join(warn) if warn else "可用半透明版图对齐"
    status = f"拼接完成 {mosaic.size[0]}×{mosaic.size[1]}。{extra}"
    state["status"] = status
    mosaic_file = _mosaic_temp_file(mosaic)
    layout_payload, layout_status = stitch_layout_payload(state, layout_state, cache_get=cache_get)
    status = f"{status}\n{layout_status}"
    state["status"] = status
    return state, mosaic, mosaic_file, status, layout_payload


def empty_layout_payload(status: str) -> dict:
    return {
        "enabled": False,
        "base_image": "",
        "mask_image": "",
        "transform": None,
        "target_width": 0,
        "target_height": 0,
        "source_width": 0,
        "source_height": 0,
        "foreground_bbox_xyxy": None,
        "status": status,
    }


def _mask_to_editor_image(mask) -> Image.Image:
    preview = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def stitch_layout_payload(stitch_state, layout_state, cache_get: CacheGet | None = None):
    state = stitch_state if isinstance(stitch_state, dict) else {}
    mosaic = state.get("mosaic")
    if mosaic is None:
        return empty_layout_payload("请先生成拼接结果"), "请先生成拼接结果"
    mosaic = pil_rgb(mosaic)
    tw, th = mosaic.size
    base_url = image_to_data_url(mosaic, max_side=2048, quality=88)
    layout_state = layout_state if isinstance(layout_state, dict) else {}
    if not layout_state.get("layout_id") or cache_get is None:
        payload = {
            "enabled": False,
            "base_image": base_url,
            "mask_image": "",
            "transform": None,
            "target_width": tw,
            "target_height": th,
            "source_width": 0,
            "source_height": 0,
            "foreground_bbox_xyxy": None,
            "status": "拼接底图已载入。请点「使用当前已保存版图 mask」或先到版图 Tab 生成 mask。",
        }
        return payload, payload["status"]
    try:
        cached = cache_get(layout_state)
        source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
        if source_mask.ndim != 2:
            raise ValueError("source_mask is not 2D")
        mask_image = image_to_data_url(_mask_to_editor_image(source_mask), max_side=2048, quality=90)
        pivot = cached.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(
            cached.get("foreground_bbox_xyxy") or _layout_tx.foreground_bbox_xyxy(source_mask)
        )
        center_x = float(layout_state.get("center_x") if layout_state.get("center_x") is not None else tw / 2.0)
        center_y = float(layout_state.get("center_y") if layout_state.get("center_y") is not None else th / 2.0)
        transform = _layout_tx.make_layout_transform_v2(
            session_id=str(layout_state.get("session_id") or cached.get("session_id") or ""),
            layout_id=str(layout_state.get("layout_id")),
            image_id="stitch-mosaic",
            target_size=(tw, th),
            source_mask=source_mask,
            center_x=center_x,
            center_y=center_y,
            pivot_xy=pivot,
            scale=float(layout_state.get("scale") or 1.0),
            rotation_deg=float(layout_state.get("rotation_deg") or 0.0),
            preview_alpha=float(layout_state.get("preview_alpha") or 0.35),
            revision=int(layout_state.get("revision") or 0),
            source_mask_pixel_sha256=cached.get("source_mask_pixel_sha256"),
            target_image_sha256=_layout_tx.image_pixel_sha256(mosaic),
        )
        transform = _layout_tx.transform_with_derived_fields(transform, (tw, th))
        payload = {
            "enabled": True,
            "base_image": base_url,
            "mask_image": mask_image,
            "transform": transform,
            "target_width": tw,
            "target_height": th,
            "source_width": int(cached.get("source_width") or source_mask.shape[1]),
            "source_height": int(cached.get("source_height") or source_mask.shape[0]),
            "foreground_bbox_xyxy": list(
                cached.get("foreground_bbox_xyxy") or _layout_tx.foreground_bbox_xyxy(source_mask)
            ),
            "status": "在拼接底图上拖动版图；拖动中更透，方向键 1px。点更新预览才会走后端 warp。",
        }
        return payload, payload["status"]
    except Exception as exc:
        payload = {
            "enabled": False,
            "base_image": base_url,
            "mask_image": "",
            "transform": None,
            "target_width": tw,
            "target_height": th,
            "source_width": 0,
            "source_height": 0,
            "foreground_bbox_xyxy": None,
            "status": f"版图 mask 不可用：{exc}",
        }
        return payload, payload["status"]


def _mosaic_temp_file(mosaic: Image.Image) -> str:
    import os
    import tempfile

    from sam3_demo.config import runtime_export_dir

    runtime_export_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="stitch_mosaic_", suffix=".png", dir=str(runtime_export_dir))
    os.close(fd)
    mosaic.save(name, format="PNG")
    return name


LAYOUT_CHOICES = list(STITCH_LAYOUTS.items())
