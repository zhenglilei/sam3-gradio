"""Session-safe callbacks for the standalone tile-stitching tab."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import gradio as gr
from PIL import Image

import image_crop_utils as _image_crop
from sam3_demo.stitch_workflow import (
    STITCH_LAYOUTS,
    auto_align_images,
    canvas_payload,
    default_shifts_for_layout,
    empty_canvas_payload,
    export_mosaic,
    load_images_from_files,
    normalize_layout,
    pil_rgb,
    select_worst_tile,
    selected_from_canvas_payload,
    shifts_from_canvas_payload,
)


PublishMosaic = Callable[[Image.Image, str], str]


def _as_revision(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def new_stitch_state(session_id: str = "", owner_token: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": str(session_id or ""),
        "owner_token": str(owner_token or ""),
        "images": [],
        "shifts": [],
        "layout": "horizontal",
        "selected": 0,
        "mosaic": None,
        "mosaic_full": None,
        "mosaic_view_revision": 0,
        "mosaic_crop_bbox_xyxy": None,
        "revision": 0,
        "generated_revision": None,
        "logs": [],
        "warnings": [],
        "nudge_step": 1,
        "diff_mode": False,
        "show_loupe": True,
        "blend": True,
        "crop_periodic": False,
        "status": "请上传一组已去 overlay 的分块图",
    }


def _fresh_owned_state(stitch_state: dict | None) -> dict[str, Any]:
    previous = stitch_state if isinstance(stitch_state, dict) else {}
    state = new_stitch_state(
        str(previous.get("session_id") or ""),
        str(previous.get("owner_token") or ""),
    )
    state["revision"] = _as_revision(previous.get("revision")) + 1
    return state


def _images(state: dict | None) -> list[Image.Image]:
    state = state if isinstance(state, dict) else {}
    out = []
    for item in state.get("images") or []:
        if isinstance(item, Image.Image) and item.mode == "RGB":
            img = item
        else:
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


def _invalidate_mosaic(state: dict, *, bump_revision: bool = True) -> None:
    state["mosaic"] = None
    state["mosaic_full"] = None
    state["mosaic_crop_bbox_xyxy"] = None
    state["mosaic_view_revision"] = int(state.get("mosaic_view_revision") or 0) + 1
    state["generated_revision"] = None
    state["warnings"] = []
    if bump_revision:
        state["revision"] = _as_revision(state.get("revision")) + 1


def _cleared_result_updates():
    return (
        gr.update(value=None, visible=True),
        gr.update(value=None, visible=False),
        gr.update(interactive=False),
        "",
    )


def _unchanged_result_updates():
    return gr.update(), gr.update(), gr.update(), gr.update()


def _mosaic_pixel_sha256(image: Image.Image) -> str:
    rgb = pil_rgb(image)
    if rgb is None:
        return ""
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def empty_mosaic_crop_payload(status: str = "请先生成拼接结果") -> dict[str, Any]:
    return {
        "server_view": {
            "enabled": False,
            "natural_width": 1,
            "natural_height": 1,
            "image_id": "",
            "image_sha256": "",
            "revision": 0,
            "interaction": "disabled",
            "selection_state": "",
            "status": str(status or ""),
        },
        "client_intent": {},
    }


def mosaic_crop_payload(stitch_state, status: str = "") -> dict[str, Any]:
    state = stitch_state if isinstance(stitch_state, dict) else {}
    mosaic = pil_rgb(state.get("mosaic"))
    generated = state.get("generated_revision")
    current = _as_revision(state.get("revision"))
    if mosaic is None or generated is None or _as_revision(generated) != current:
        return empty_mosaic_crop_payload(status or "请先生成当前拼接结果")
    view_revision = _as_revision(state.get("mosaic_view_revision"))
    image_hash = _mosaic_pixel_sha256(mosaic)
    return {
        "server_view": {
            "enabled": True,
            "natural_width": mosaic.width,
            "natural_height": mosaic.height,
            "image_id": f"stitch-mosaic-{current}-{view_revision}",
            "image_sha256": image_hash,
            "revision": view_revision,
            "interaction": "crop",
            "selection_state": "",
            "status": str(status or "拖拽长方形截取导出区域"),
        },
        "client_intent": {},
    }


def _validate_mosaic_crop_intent(payload, state: dict, mosaic: Image.Image):
    if not isinstance(payload, dict) or str(payload.get("gesture") or "") != "drag":
        raise ValueError("请在导出预览上拖拽长方形截图")
    expected = mosaic_crop_payload(state).get("server_view") or {}
    if payload.get("expected_revision") != expected.get("revision"):
        raise ValueError("截图 revision 已过期，请重新拖拽")
    if str(payload.get("image_id") or "") != str(expected.get("image_id") or ""):
        raise ValueError("截图图像 identity 已过期，请重新拖拽")
    if str(payload.get("image_sha256") or "") != str(expected.get("image_sha256") or ""):
        raise ValueError("截图图像 hash 已过期，请重新拖拽")
    return _image_crop.normalize_crop_box(
        payload.get("start_xy") or [],
        payload.get("end_xy") or [],
        mosaic.width,
        mosaic.height,
    )


def _publish_current_mosaic(state: dict, publish_mosaic: PublishMosaic | None):
    mosaic = pil_rgb(state.get("mosaic"))
    if mosaic is None or publish_mosaic is None:
        return None
    return publish_mosaic(mosaic, str(state.get("session_id") or ""))


def crop_mosaic_preview(gesture_payload, stitch_state, *, publish_mosaic=None):
    state = dict(stitch_state or new_stitch_state())
    mosaic = pil_rgb(state.get("mosaic"))
    try:
        if mosaic is None:
            raise ValueError("请先生成拼接结果")
        box = _validate_mosaic_crop_intent(gesture_payload, state, mosaic)
        cropped = _image_crop.crop_pil_image(mosaic, box)
        state["mosaic"] = cropped
        state["mosaic_crop_bbox_xyxy"] = list(box)
        state["mosaic_view_revision"] = _as_revision(state.get("mosaic_view_revision")) + 1
        path = None
        publish_warning = ""
        try:
            path = _publish_current_mosaic(state, publish_mosaic)
        except Exception as exc:
            publish_warning = f"；下载文件发布失败：{exc}"
        status = (
            f"已截取导出区域 {list(box)}；当前结果 {cropped.width}×{cropped.height}"
            f"{publish_warning}"
        )
        state["status"] = status
        file_update = gr.update(value=path, visible=bool(path))
        return (
            state,
            gr.update(value=cropped, visible=True),
            file_update,
            status,
            gr.update(interactive=True),
            "",
            mosaic_crop_payload(state, status),
            gr.update(interactive=True),
        )
    except Exception as exc:
        status = f"截图失败：{exc}"
        state["status"] = status
        return (
            state,
            gr.update(),
            gr.update(),
            status,
            gr.update(),
            gr.update(),
            mosaic_crop_payload(state, status),
            gr.update(),
        )


def restore_full_mosaic(stitch_state, *, publish_mosaic=None):
    state = dict(stitch_state or new_stitch_state())
    full = pil_rgb(state.get("mosaic_full"))
    generated = state.get("generated_revision")
    current = _as_revision(state.get("revision"))
    if full is None or generated is None or _as_revision(generated) != current:
        status = "恢复失败：请先生成当前拼接结果"
        state["status"] = status
        return (
            state, gr.update(), gr.update(), status, gr.update(), gr.update(),
            empty_mosaic_crop_payload(status), gr.update(interactive=False),
        )
    state["mosaic"] = full.copy()
    state["mosaic_crop_bbox_xyxy"] = None
    state["mosaic_view_revision"] = _as_revision(state.get("mosaic_view_revision")) + 1
    path = None
    publish_warning = ""
    try:
        path = _publish_current_mosaic(state, publish_mosaic)
    except Exception as exc:
        publish_warning = f"；下载文件发布失败：{exc}"
    status = f"已恢复完整拼接图 {full.width}×{full.height}{publish_warning}"
    state["status"] = status
    return (
        state,
        gr.update(value=state["mosaic"], visible=True),
        gr.update(value=path, visible=bool(path)),
        status,
        gr.update(interactive=True),
        "",
        mosaic_crop_payload(state, status),
        gr.update(interactive=False),
    )


def load_tiles(files, layout, stitch_state, nudge_step, diff_mode, show_loupe, blend, crop_periodic):
    state = _fresh_owned_state(stitch_state)
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
            *_cleared_result_updates(),
        )
    state.update(
        {
            "images": images,
            "layout": layout_key,
            "selected": 0,
            "nudge_step": int(nudge_step or 1),
            "diff_mode": bool(diff_mode),
            "show_loupe": bool(show_loupe if show_loupe is not None else True),
            "blend": bool(blend),
            "crop_periodic": bool(crop_periodic),
        }
    )
    try:
        shifts = default_shifts_for_layout(images, layout_key)
    except ValueError as exc:
        status = f"排列方式不可用：{exc}。请更换排列方式。"
        state["logs"] = [status]
        state["status"] = status
        return (
            state,
            empty_canvas_payload(status),
            0.0,
            0.0,
            status,
            *_cleared_result_updates(),
        )
    state["shifts"] = shifts
    state["logs"] = [f"已加载 {len(images)} 张，布局 {layout_key}，尚未自动对齐"]
    status = _status_text(state["logs"])
    state["status"] = status
    dx, dy = _selected_xy(state)
    return state, _payload(state, status), dx, dy, status, *_cleared_result_updates()


def auto_align(stitch_state, layout, nudge_step, diff_mode, show_loupe):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    previous_layout = normalize_layout(state.get("layout"))
    layout_key = normalize_layout(layout or state.get("layout"))
    state["nudge_step"] = int(nudge_step or state.get("nudge_step") or 1)
    state["diff_mode"] = bool(diff_mode)
    state["show_loupe"] = bool(show_loupe if show_loupe is not None else True)
    if not images:
        status = "请先加载分块图"
        state["status"] = status
        _invalidate_mosaic(state, bump_revision=False)
        return (
            state,
            empty_canvas_payload(status),
            0.0,
            0.0,
            status,
            *_cleared_result_updates(),
        )
    try:
        shifts, logs = auto_align_images(images, layout_key)
    except Exception as exc:
        status = f"自动对齐失败：{exc}"
        if layout_key != previous_layout:
            _invalidate_mosaic(state)
            result_updates = _cleared_result_updates()
        else:
            result_updates = _unchanged_result_updates()
        state["status"] = status
        return (
            state,
            _payload(state, status),
            *_selected_xy(state),
            status,
            *result_updates,
        )
    state["layout"] = layout_key
    baseline = default_shifts_for_layout(images, layout_key)
    selected = select_worst_tile(shifts, baseline)
    state["shifts"] = shifts
    state["logs"] = logs
    state["selected"] = selected
    _invalidate_mosaic(state)
    status = _status_text(logs, f"已自动选中偏移最大的图{selected + 1}，可拖动或用方向键微调")
    state["status"] = status
    dx, dy = _selected_xy(state)
    return state, _payload(state, status), dx, dy, status, *_cleared_result_updates()


def apply_layout(layout, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    layout_key = normalize_layout(layout)
    previous_layout = normalize_layout(state.get("layout"))
    if not images:
        state["layout"] = layout_key
        status = "请先加载分块图"
        state["status"] = status
        return (
            state,
            gr.update(value=layout_key),
            empty_canvas_payload(status),
            0.0,
            0.0,
            status,
            *_cleared_result_updates(),
        )
    if layout_key == previous_layout:
        status = state.get("status") or "就绪"
        return (
            state,
            gr.update(value=previous_layout),
            _payload(state, status),
            *_selected_xy(state),
            status,
            *_unchanged_result_updates(),
        )
    try:
        shifts = default_shifts_for_layout(images, layout_key)
    except ValueError as exc:
        _invalidate_mosaic(state)
        status = f"排列方式不可用：{exc}"
        state["status"] = status
        return (
            state,
            gr.update(value=previous_layout),
            _payload(state, status),
            *_selected_xy(state),
            status,
            *_cleared_result_updates(),
        )
    state["layout"] = layout_key
    state["shifts"] = shifts
    state["selected"] = 0
    state["logs"] = ["排列方式已改变，位置已重置；请重新自动对齐"]
    _invalidate_mosaic(state)
    status = _status_text(state["logs"])
    state["status"] = status
    return (
        state,
        gr.update(value=layout_key),
        _payload(state, status),
        *_selected_xy(state),
        status,
        *_cleared_result_updates(),
    )


def canvas_changed(payload, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    if not images:
        status = state.get("status") or "请先加载分块图"
        return state, 0.0, 0.0, status, *_unchanged_result_updates()
    old_shifts = list(state.get("shifts") or [])
    shifts = shifts_from_canvas_payload(payload, old_shifts)
    geometry_changed = len(shifts) == len(images) and shifts != old_shifts
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
    if geometry_changed:
        _invalidate_mosaic(state)
        status += "；位置已改变，请重新生成拼接结果"
        result_updates = _cleared_result_updates()
    else:
        result_updates = _unchanged_result_updates()
    state["status"] = status
    return state, dx, dy, status, *result_updates


def apply_numeric_shift(dx, dy, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    shifts = list(state.get("shifts") or [])
    if not images or not shifts:
        status = "请先加载分块图"
        state["status"] = status
        return (
            state,
            empty_canvas_payload(status),
            0.0,
            0.0,
            status,
            *_cleared_result_updates(),
        )
    selected = max(0, min(int(state.get("selected") or 0), len(shifts) - 1))
    try:
        nx = int(round(float(dx)))
        ny = int(round(float(dy)))
    except (TypeError, ValueError, OverflowError):
        nx, ny = shifts[selected]
    changed = (nx, ny) != tuple(shifts[selected])
    shifts[selected] = (nx, ny)
    state["shifts"] = shifts
    status = f"已把图{selected + 1} 设为 ({nx}, {ny})"
    if changed:
        _invalidate_mosaic(state)
        status += "；请重新生成拼接结果"
        result_updates = _cleared_result_updates()
    else:
        result_updates = _unchanged_result_updates()
    state["status"] = status
    return state, _payload(state, status), float(nx), float(ny), status, *result_updates


def apply_canvas_options(nudge_step, diff_mode, show_loupe, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    state["nudge_step"] = int(nudge_step or 1)
    state["diff_mode"] = bool(diff_mode)
    state["show_loupe"] = bool(show_loupe if show_loupe is not None else True)
    images = _images(state)
    if not images:
        status = state.get("status") or "请先加载分块图"
        return state, empty_canvas_payload(status), status
    status = (
        f"步长 {state['nudge_step']}px；"
        f"差分={'开' if state['diff_mode'] else '关'}；"
        f"放大镜={'开' if state['show_loupe'] else '关'}"
    )
    state["status"] = status
    return state, _payload(state, status), status


def apply_export_options(blend, crop_periodic, stitch_state):
    state = dict(stitch_state or new_stitch_state())
    next_blend = bool(blend)
    next_crop = bool(crop_periodic)
    changed = (
        next_blend != bool(state.get("blend"))
        or next_crop != bool(state.get("crop_periodic"))
    )
    state["blend"] = next_blend
    state["crop_periodic"] = next_crop
    if changed and state.get("mosaic") is not None:
        _invalidate_mosaic(state)
        status = "拼接输出选项已改变，请重新生成拼接结果"
        result_updates = _cleared_result_updates()
    else:
        status = state.get("status") or "就绪"
        result_updates = _unchanged_result_updates()
    state["status"] = status
    return state, status, *result_updates


def generate_mosaic(
    stitch_state,
    blend,
    crop_periodic,
    *,
    publish_mosaic: PublishMosaic | None = None,
):
    state = dict(stitch_state or new_stitch_state())
    images = _images(state)
    shifts = list(state.get("shifts") or [])
    if not images or len(shifts) != len(images):
        status = "请先加载并对齐分块图"
        state["status"] = status
        _invalidate_mosaic(state, bump_revision=False)
        return (
            state,
            *_cleared_result_updates()[:2],
            status,
            gr.update(interactive=False),
            "",
            empty_mosaic_crop_payload(status),
        )

    next_blend = bool(blend)
    next_crop = bool(crop_periodic)
    if (
        next_blend != bool(state.get("blend"))
        or next_crop != bool(state.get("crop_periodic"))
    ):
        state["revision"] = _as_revision(state.get("revision")) + 1
    state["blend"] = next_blend
    state["crop_periodic"] = next_crop
    try:
        mosaic, warnings = export_mosaic(
            images,
            shifts,
            layout=state.get("layout") or "horizontal",
            blend=next_blend,
            crop_periodic=next_crop,
        )
    except Exception as exc:
        status = f"生成拼接失败：{exc}"
        state["status"] = status
        _invalidate_mosaic(state, bump_revision=False)
        return (
            state,
            *_cleared_result_updates()[:2],
            status,
            gr.update(interactive=False),
            "",
            empty_mosaic_crop_payload(status),
        )

    state["mosaic"] = mosaic
    state["mosaic_full"] = mosaic.copy()
    state["mosaic_crop_bbox_xyxy"] = None
    state["mosaic_view_revision"] = _as_revision(state.get("mosaic_view_revision")) + 1
    state["warnings"] = warnings
    state["generated_revision"] = _as_revision(state.get("revision"))
    note = "；".join(warnings) if warnings else "已生成可交接结果"
    status = f"拼接完成 {mosaic.size[0]}×{mosaic.size[1]}。{note}"
    mosaic_file = None
    if publish_mosaic is not None:
        try:
            mosaic_file = publish_mosaic(mosaic, str(state.get("session_id") or ""))
        except Exception as exc:
            status += f"；下载文件发布失败：{exc}"
    state["status"] = status
    file_update = (
        gr.update(value=mosaic_file, visible=True)
        if mosaic_file
        else gr.update(value=None, visible=False)
    )
    return (
        state,
        gr.update(value=mosaic, visible=True),
        file_update,
        status,
        gr.update(interactive=True),
        "",
        mosaic_crop_payload(state, status),
    )


def mosaic_for_handoff(stitch_state) -> Image.Image:
    if not isinstance(stitch_state, dict):
        raise ValueError("请先生成当前拼接结果")
    generated = stitch_state.get("generated_revision")
    current = _as_revision(stitch_state.get("revision"))
    if generated is None or _as_revision(generated) != current:
        raise ValueError("拼接结果已过期，请重新生成后再写入工作区")
    mosaic = pil_rgb(stitch_state.get("mosaic"))
    if mosaic is None:
        raise ValueError("请先生成当前拼接结果")
    return mosaic


def handoff_status(source_status: str) -> str:
    text = str(source_status or "")
    if text.startswith("已载入完整原图"):
        return "拼接整图已写入智能图像分割工作区。"
    if not text:
        return "写入工作区失败。"
    return f"写入工作区失败：{text}"


LAYOUT_CHOICES = list(STITCH_LAYOUTS.items())
