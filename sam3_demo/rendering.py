"""Pure display and image-rendering helpers."""

from __future__ import annotations

import gradio as gr
from PIL import Image


def _layout_preview_alpha(state):
    value = (state or {}).get("preview_alpha")
    return float(0.35 if value is None else value)


def _result_placeholder(image_state):
    if not image_state or not image_state.get("image_id"):
        return None
    width = max(1, int(image_state.get("width") or 1))
    height = max(1, int(image_state.get("height") or 1))
    return Image.new("RGB", (width, height), (248, 250, 252))


def _workspace_image_impl(_deps, image_state, pcs_state, pvs_state, mode, prompt_state, layout_state):
    _overlay = _deps['_overlay']
    if not image_state or not image_state.get("image_id"):
        return None
    return _overlay(
        image_state,
        pcs_state,
        pvs_state,
        mode,
        prompt_state,
        show_instances=False,
        show_interaction_prompts=True,
        show_layout_overlay=bool(layout_state and layout_state.get("enabled")),
        layout_state=layout_state,
    )


def _result_image_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _instances_for_mode = _deps['_instances_for_mode']
    _overlay = _deps['_overlay']
    _result_placeholder = _deps['_result_placeholder']
    if not image_state or not image_state.get("image_id"):
        return None
    if not _instances_for_mode(pcs_state, pvs_state, mode):
        return _result_placeholder(image_state)
    return _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True, show_interaction_prompts=False, show_layout_overlay=False)


def _pcs_choice_update_impl(_deps, pcs_state):
    _active_instances = _deps['_active_instances']
    choices = [(f"PCS #{i['id']} score={i['score']:.3f}", str(i["id"])) for i in _active_instances(pcs_state)]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _status_label_impl(_deps, status):
    return {"draft": "草稿", "accepted": "已确认", "deleted": "已删除"}.get(str(status or "draft"), str(status or "草稿"))


def _pvs_choice_update_impl(_deps, pvs_state):
    _active_instances = _deps['_active_instances']
    _status_label = _deps['_status_label']
    choices = [(f"PVS #{i['id']} {_status_label(i.get('status'))}", str(i["id"])) for i in _active_instances(pvs_state)]
    active = pvs_state.get("active_instance_id")
    value = str(active) if active is not None and any(c[1] == str(active) for c in choices) else None
    return gr.update(choices=choices, value=value)


def _pvs_pending_count_text_impl(_deps, pvs_state):
    _sync_pvs_pending_boxes_from_records = _deps['_sync_pvs_pending_boxes_from_records']
    _sync_pvs_pending_boxes_from_records(pvs_state)
    return f"\u5f85\u751f\u6210 bbox \u6570\u91cf: {len(pvs_state.get('pending_boxes', []))}"


def _pcs_summary_impl(_deps, pcs_state):
    _active_instances = _deps['_active_instances']
    _pcs_bbox_records = _deps['_pcs_bbox_records']
    _sync_pcs_boxes_from_records = _deps['_sync_pcs_boxes_from_records']
    _sync_pcs_boxes_from_records(pcs_state)
    lines = [f"正样本 bbox: {len(pcs_state.get('positive_boxes', []))}", f"负样本 bbox: {len(pcs_state.get('negative_boxes', []))}"]
    for rec in _pcs_bbox_records(pcs_state)[:40]:
        role = "负样本" if rec.get("key") == "negative_boxes" else "正样本"
        lines.append(f"ID {rec.get('id')} {role}: {[round(float(v), 1) for v in rec.get('box', [])]}")
    items = _active_instances(pcs_state)
    lines.append(f"PCS 实例: {len(items)}")
    for inst in items[:80]:
        lines.append(f"#{inst['id']} score={inst['score']:.3f} box={[round(v,1) for v in inst['box_xyxy_px']]}")
    return "\n".join(lines)


def _pvs_summary_impl(_deps, pvs_state):
    _active_instances = _deps['_active_instances']
    _status_label = _deps['_status_label']
    items = _active_instances(pvs_state)
    active = pvs_state.get("active_instance_id")
    pending = pvs_state.get("pending_boxes", [])
    lines = [
        f"PVS 实例: {len(items)}",
        f"待生成 bbox: {len(pending)}",
        f"当前实例: {active or '-'}",
        "说明: 草稿=draft，表示还未点击确认；score 是 SAM3 返回的候选 mask 质量/置信估计，不等同于人工质检分数。",
    ]
    for idx, box in enumerate(pending[:20], start=1):
        lines.append(f"pending#{idx} box={[round(v,1) for v in box]}")
    for inst in items[:80]:
        mark = "*" if str(inst["id"]) == str(active) else " "
        lines.append(f"{mark}#{inst['id']} {inst['source']} {_status_label(inst.get('status'))} score={inst['score']:.3f}")
    return "\n".join(lines)


def _analysis_report_impl(_deps, pcs_state, pvs_state, mode, info):
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _pcs_summary = _deps['_pcs_summary']
    _pvs_summary = _deps['_pvs_summary']
    sections = [str(info or "")]
    if _is_pcs_mode(mode):
        sections.extend(["", "PCS Auto 自动概念分割", _pcs_summary(pcs_state)])
    elif _is_layout_mask_mode(mode):
        sections.extend(["", "版图 mask 提示分割", _pvs_summary(pvs_state)])
    else:
        sections.extend(["", "PVS Manual 手动实例分割", _pvs_summary(pvs_state)])
    return "\n".join(part for part in sections if part is not None)


def _view_impl(_deps, image_state, pcs_state, pvs_state, mode, info, prompt_state, layout_state):
    _analysis_report = _deps['_analysis_report']
    _pcs_summary = _deps['_pcs_summary']
    _pvs_choice_update = _deps['_pvs_choice_update']
    _pvs_pending_count_text = _deps['_pvs_pending_count_text']
    _pvs_summary = _deps['_pvs_summary']
    _result_image = _deps['_result_image']
    _workspace_image = _deps['_workspace_image']
    return (
        _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state, layout_state),
        _result_image(image_state, pcs_state, pvs_state, mode),
        _analysis_report(pcs_state, pvs_state, mode, info),
        _pcs_summary(pcs_state),
        _pvs_summary(pvs_state),
        _pvs_choice_update(pvs_state),
        info,
        _pvs_pending_count_text(pvs_state),
    )
