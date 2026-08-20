"""PCS/PVS interaction and segmentation callbacks."""

from __future__ import annotations

import gradio as gr
import numpy as np


def _workspace_gesture_payload_impl(_deps, image_state, mode, click_tool, status):
    _click_tool_key = _deps['_click_tool_key']
    _image_gesture_payload = _deps['_image_gesture_payload']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _is_pcs_mode = _deps['_is_pcs_mode']
    state = image_state if isinstance(image_state, dict) else {}
    enabled = bool(state.get("image_id"))
    if _is_layout_mask_mode(mode):
        interaction = "click"
    else:
        tool = "bbox" if _is_pcs_mode(mode) else _click_tool_key(click_tool)
        interaction = tool if tool in {"bbox", "point", "polygon"} else "disabled"
    return _image_gesture_payload(
        enabled=enabled,
        width=state.get("width"),
        height=state.get("height"),
        image_id=state.get("image_id"),
        image_sha256=state.get("target_image_sha256"),
        revision=state.get("interaction_revision"),
        interaction=interaction if enabled else "disabled",
        status=status,
    )


def _workspace_select_impl(_deps, image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, evt):
    _append_pcs_bbox_sample = _deps['_append_pcs_bbox_sample']
    _append_pvs_pending_bbox = _deps['_append_pvs_pending_bbox']
    _click_tool_key = _deps['_click_tool_key']
    _event_point = _deps['_event_point']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _norm_box = _deps['_norm_box']
    _payload_json = _deps['_payload_json']
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _view = _deps['_view']
    prompt_state = prompt_state or _new_prompt_state()
    bbox_payload = gr.update()
    point_payload = gr.update()
    polygon_payload = gr.update()
    try:
        point = _event_point(evt, image_state)
        w, h = int(image_state.get("width") or 0), int(image_state.get("height") or 0)
        tool = _click_tool_key(click_tool)
        if _is_layout_mask_mode(mode):
            prompt_state["last_point"] = point
            point_payload = _payload_json({"type": "point", "point_xy_px": point, "image_width": w, "image_height": h})
            info = f"已记录待应用版图修缮点: {[round(v, 1) for v in point]}。请选择点类型并点击“应用点提示”。"
            return prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)
        if _is_pcs_mode(mode) and tool != "bbox":
            tool = "bbox"
        if tool == "point":
            prompt_state["last_point"] = point
            point_payload = _payload_json({"type": "positive_point", "point_xy_px": point, "image_width": w, "image_height": h})
            info = f"\u5df2\u6dfb\u52a0\u6b63\u5411\u70b9: {[round(v, 1) for v in point]}"
        elif tool == "bbox":
            bbox_role = "negative" if _is_pcs_mode(mode) and str(pcs_bbox_kind or "").startswith("Negative") else "positive"
            prompt_state["bbox_role"] = bbox_role
            if prompt_state.get("bbox_start") is None:
                prompt_state["bbox_start"] = point
                prompt_state["last_bbox"] = None
                info = f"\u5df2\u8bb0\u5f55 bbox \u8d77\u70b9: {[round(v, 1) for v in point]}\u3002\u8bf7\u70b9\u51fb\u5bf9\u89d2\u70b9\u5b8c\u6210\u6846\u9009\u3002"
            else:
                start = prompt_state.get("bbox_start")
                box = _norm_box([start[0], start[1], point[0], point[1]], w, h)
                prompt_state["bbox_start"] = None
                prompt_state["last_bbox"] = box
                bbox_payload = _payload_json({"type": "bbox", "box_xyxy_px": box, "image_width": w, "image_height": h})
                if _is_pcs_mode(mode):
                    key = _append_pcs_bbox_sample(pcs_state, box, bbox_role)
                    prompt_state["last_bbox"] = None
                    label = "\u8d1f\u6837\u672c" if key == "negative_boxes" else "\u6b63\u6837\u672c"
                    info = f"\u5df2\u81ea\u52a8\u6dfb\u52a0 PCS {label} bbox: {[round(v, 1) for v in box]}"
                else:
                    bbox_id = _append_pvs_pending_bbox(pvs_state, box)
                    prompt_state["last_bbox"] = None
                    info = f"\u5df2\u52a0\u5165 PVS \u5f85\u751f\u6210 bbox ID {bbox_id}: {[round(v, 1) for v in box]}\u3002\u7ee7\u7eed\u6846\u9009\u6216\u70b9\u51fb\u201c\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b\u201d\u3002"
        elif tool == "polygon":
            points = prompt_state.setdefault("polygon_points", [])
            points.append(point)
            info = f"\u591a\u8fb9\u5f62\u5df2\u6dfb\u52a0\u7b2c {len(points)} \u4e2a\u9876\u70b9\u3002\u5b8c\u6210\u540e\u70b9\u51fb\u201c\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61\u201d\u3002"
        else:
            info = f"\u672a\u77e5\u4ea4\u4e92\u5de5\u5177: {click_tool}"
    except Exception as exc:
        info = f"\u56fe\u50cf\u70b9\u51fb\u5931\u8d25: {exc}"
    return prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _workspace_gesture_input_impl(_deps, image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, gesture_payload):
    _GestureSelectEvent = _deps['_GestureSelectEvent']
    _append_pcs_bbox_sample = _deps['_append_pcs_bbox_sample']
    _append_pvs_pending_bbox = _deps['_append_pvs_pending_bbox']
    _click_tool_key = _deps['_click_tool_key']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _norm_box = _deps['_norm_box']
    _payload_json = _deps['_payload_json']
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _validate_gesture_intent = _deps['_validate_gesture_intent']
    _view = _deps['_view']
    _workspace_gesture_payload = _deps['_workspace_gesture_payload']
    _workspace_select = _deps['_workspace_select']
    prompt_state = prompt_state or _new_prompt_state()
    try:
        gesture, start, end = _validate_gesture_intent(
            gesture_payload,
            image_state or {},
            allowed_gestures={"click", "drag"},
        )
        tool = "bbox" if _is_pcs_mode(mode) else _click_tool_key(click_tool)
        if _is_layout_mask_mode(mode):
            tool = "layout"
        if gesture == "click":
            if tool == "bbox":
                raise ValueError("BBox 已改为拖拽操作，请按住左键拖出矩形")
            result = _workspace_select(
                image_state,
                pcs_state,
                pvs_state,
                mode,
                click_tool,
                pcs_bbox_kind,
                prompt_state,
                _GestureSelectEvent(start),
            )
            return (*result, _workspace_gesture_payload(image_state, mode, click_tool))
        if tool != "bbox" or _is_layout_mask_mode(mode):
            raise ValueError("当前工具不接受拖拽，请使用单击")
        width = int(image_state.get("width") or 0)
        height = int(image_state.get("height") or 0)
        box = _norm_box([start[0], start[1], end[0], end[1]], width, height)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            raise ValueError("bbox 太小")
        prompt_state = dict(prompt_state)
        prompt_state["bbox_start"] = None
        prompt_state["last_bbox"] = None
        bbox_role = "negative" if _is_pcs_mode(mode) and str(pcs_bbox_kind or "").startswith("Negative") else "positive"
        prompt_state["bbox_role"] = bbox_role
        bbox_payload = _payload_json(
            {
                "type": "bbox",
                "box_xyxy_px": box,
                "image_width": width,
                "image_height": height,
            }
        )
        if _is_pcs_mode(mode):
            key = _append_pcs_bbox_sample(pcs_state, box, bbox_role)
            label = "负样本" if key == "negative_boxes" else "正样本"
            info = f"已拖拽添加 PCS {label} bbox: {[round(v, 1) for v in box]}"
        else:
            bbox_id = _append_pvs_pending_bbox(pvs_state, box)
            info = f"已拖拽加入 PVS 待生成 bbox ID {bbox_id}: {[round(v, 1) for v in box]}"
        result = (
            prompt_state,
            bbox_payload,
            gr.update(),
            gr.update(),
            pcs_state,
            pvs_state,
            _pcs_bbox_choices(pcs_state),
            _pvs_pending_bbox_choices(pvs_state),
            *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state),
        )
    except Exception as exc:
        info = f"图像交互失败: {exc}"
        result = (
            prompt_state,
            gr.update(),
            gr.update(),
            gr.update(),
            pcs_state,
            pvs_state,
            _pcs_bbox_choices(pcs_state),
            _pvs_pending_bbox_choices(pvs_state),
            *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state),
        )
    return (*result, _workspace_gesture_payload(image_state, mode, click_tool))


def _apply_polygon_to_pvs_impl(_deps, image_state, pvs_state, polygon, polygon_action, combine_mode, progress):
    _append_prompt_history = _deps['_append_prompt_history']
    _best = _deps['_best']
    _combine_logits = _deps['_combine_logits']
    _fresh_state = _deps['_fresh_state']
    _history_snapshot = _deps['_history_snapshot']
    _make_inst = _deps['_make_inst']
    _mask_box = _deps['_mask_box']
    _polygon_action_key = _deps['_polygon_action_key']
    _polygon_combine_key = _deps['_polygon_combine_key']
    _polygon_lowres_logits = _deps['_polygon_lowres_logits']
    _predict_inst = _deps['_predict_inst']
    _pvs_progress = _deps['_pvs_progress']
    _workspace = _deps['_workspace']
    action = _polygon_action_key(polygon_action)
    combine = _polygon_combine_key(combine_mode)
    ws = _workspace(image_state)
    w, h = ws["image"].size
    _pvs_progress(progress, 0.18, "转换 polygon 为 PVS mask prompt")
    polygon_logits = _polygon_lowres_logits(polygon, w, h)

    if action == "create":
        _pvs_progress(progress, 0.42, "SAM3 正在根据 polygon 创建实例", delay=0.12)
        pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=polygon_logits)
        _pvs_progress(progress, 0.82, "整理 polygon 候选 mask")
        idx = _best(pred)
        mask = pred["masks"][idx]
        inst_id = int(pvs_state.get("next_instance_id", 1))
        pvs_state.setdefault("instances", {})[inst_id] = _make_inst(
            inst_id,
            "manual_pvs_polygon",
            mask,
            _mask_box(mask),
            pred["scores"][idx],
            pvs_logits=pred["lowres_logits"][idx],
            history=[{"op":"create_from_polygon","prompt":{"type":"positive_polygon","points":polygon},"candidate_scores":pred["scores"].astype(float).tolist()}],
        )
        pvs_state["active_instance_id"] = inst_id
        pvs_state["next_instance_id"] = inst_id + 1
        return f"\u5df2\u7528 polygon mask prompt \u521b\u5efa PVS #{inst_id}"

    active_id = pvs_state.get("active_instance_id")
    if active_id is None or int(active_id) not in pvs_state.get("instances", {}):
        raise ValueError("\u8bf7\u5148\u521b\u5efa\u6216\u9009\u62e9\u4e00\u4e2a PVS \u5b9e\u4f8b\uff0c\u6216\u5c06 polygon \u52a8\u4f5c\u6539\u4e3a\u201c\u521b\u5efa\u65b0\u5b9e\u4f8b\u201d")
    inst = pvs_state["instances"][int(active_id)]
    _pvs_progress(progress, 0.34, f"融合当前实例 logits: {combine}")
    combined = _combine_logits(inst.get("pvs_lowres_logits"), polygon_logits, mode=combine)
    before = _history_snapshot(inst)
    _pvs_progress(progress, 0.52, "SAM3 正在精修当前 PVS 实例", delay=0.12)
    pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=combined)
    _pvs_progress(progress, 0.84, "更新实例 mask 与 logits")
    idx = _best(pred)
    mask = pred["masks"][idx]
    inst["mask_fullres_bool"] = mask
    inst["box_xyxy_px"] = _mask_box(mask)
    inst["score"] = float(pred["scores"][idx])
    inst["pvs_lowres_logits"] = pred["lowres_logits"][idx]
    after = _history_snapshot(inst)
    _append_prompt_history(inst, {"op":"positive_polygon_refine","mode":combine,"prompt":{"type":"positive_polygon","points":polygon},"before":before,"after":after,"candidate_scores":pred["scores"].astype(float).tolist()})
    return f"\u5df2\u7528\u591a\u8fb9\u5f62\u7cbe\u4fee PVS #{active_id}\uff0c\u878d\u5408\u65b9\u5f0f: {combine}"


def _finish_native_polygon_impl(_deps, image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode, progress):
    _apply_polygon_to_pvs = _deps['_apply_polygon_to_pvs']
    _is_pvs_manual_mode = _deps['_is_pvs_manual_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _payload_json = _deps['_payload_json']
    _pvs_progress = _deps['_pvs_progress']
    _view = _deps['_view']
    prompt_state = prompt_state or _new_prompt_state()
    points = prompt_state.get("polygon_points") or []
    polygon_payload = gr.update()
    _pvs_progress(progress, 0.03, "准备 PVS 多边形操作")
    if len(points) < 3:
        info = "\u591a\u8fb9\u5f62\u81f3\u5c11\u9700\u8981 3 \u4e2a\u9876\u70b9"
        return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)

    w, h = int(image_state.get("width") or 0), int(image_state.get("height") or 0)
    polygon_payload = _payload_json({"type": "positive_polygon", "points": points, "image_width": w, "image_height": h})

    if not _is_pvs_manual_mode(mode):
        info = "多边形已完成。当前模式不使用 polygon prompt。"
        return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)

    try:
        info = _apply_polygon_to_pvs(image_state, pvs_state, points, polygon_action, polygon_combine_mode, progress)
        prompt_state["polygon_points"] = []
        _pvs_progress(progress, 0.96, "渲染 PVS 分割结果", delay=0.16)
    except Exception as exc:
        info = f"PVS \u591a\u8fb9\u5f62\u5904\u7406\u5931\u8d25: {exc}"
    return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _clear_prompt_selection_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _clear_pvs_pending_bboxes = _deps['_clear_pvs_pending_bboxes']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _is_pvs_manual_mode = _deps['_is_pvs_manual_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _view = _deps['_view']
    prompt_state = _new_prompt_state()
    if _is_pcs_mode(mode):
        pcs_state["text_prompt"] = ""
        pcs_state["positive_boxes"] = []
        pcs_state["negative_boxes"] = []
        pcs_state["bbox_history"] = []
        pcs_state["bbox_records"] = []
        pcs_state["next_bbox_id"] = 1
        text_prompt_update = ""
        info = "PCS prompt 已清空；已有 PCS 分割结果不会被删除"
    elif _is_pvs_manual_mode(mode):
        cleared = _clear_pvs_pending_bboxes(pvs_state)
        text_prompt_update = gr.update()
        info = f"临时提示已清空，包括 {cleared} 个待生成 PVS bbox；已生成实例不会被删除"
    else:
        text_prompt_update = gr.update()
        info = "版图 mask 提示分割的临时点击提示已清空；已生成实例和待生成 bbox 不会被删除"
    return prompt_state, "", "", "", pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), text_prompt_update, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _delete_selected_pcs_bbox_impl(_deps, image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pcs_bbox_records = _deps['_pcs_bbox_records']
    _reset_pcs_predictions = _deps['_reset_pcs_predictions']
    _sync_pcs_boxes_from_records = _deps['_sync_pcs_boxes_from_records']
    _view = _deps['_view']
    try:
        records = list(_pcs_bbox_records(pcs_state))
        if not selected_bbox_id:
            raise ValueError("请先在 PCS bbox 列表中选择一个 bbox")
        target_id = int(selected_bbox_id)
        target = next((rec for rec in records if int(rec.get("id", -1)) == target_id), None)
        if target is None:
            raise ValueError("选中的 PCS bbox 已不存在，请重新选择")
        pcs_state["bbox_records"] = [rec for rec in records if int(rec.get("id", -1)) != target_id]
        pcs_state["bbox_history"] = [item for item in pcs_state.get("bbox_history", []) if int(item.get("id", -1)) != target_id]
        _sync_pcs_boxes_from_records(pcs_state)
        _reset_pcs_predictions(pcs_state)
        label = "负样本" if target.get("key") == "negative_boxes" else "正样本"
        info = f"已删除 PCS {label} bbox: {[round(float(v), 1) for v in target.get('box', [])]}；请重新运行 PCS 分割"
    except Exception as exc:
        info = f"删除 PCS bbox 失败: {exc}"
    return pcs_state, _pcs_bbox_choices(pcs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _clear_pcs_instances_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _is_pcs_mode = _deps['_is_pcs_mode']
    _reset_pcs_predictions = _deps['_reset_pcs_predictions']
    _view = _deps['_view']
    try:
        if not _is_pcs_mode(mode):
            raise ValueError("\u4ec5 PCS Auto \u6a21\u5f0f\u53ef\u4ee5\u6e05\u7a7a PCS \u5b9e\u4f8b")
        cleared = len(pcs_state.get("instances") or {})
        _reset_pcs_predictions(pcs_state)
        info = f"\u5df2\u6e05\u7a7a {cleared} \u4e2a PCS \u5b9e\u4f8b\uff1b\u6587\u672c\u63d0\u793a\u548c bbox \u63d0\u793a\u5df2\u4fdd\u7559"
    except Exception as exc:
        info = f"\u6e05\u7a7a PCS \u5b9e\u4f8b\u5931\u8d25: {exc}"
    return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _run_pcs_impl(_deps, image_state, pcs_state, pvs_state, mode, text_prompt, threshold):
    _fresh_state = _deps['_fresh_state']
    _make_inst = _deps['_make_inst']
    _norm_box = _deps['_norm_box']
    _view = _deps['_view']
    _workspace = _deps['_workspace']
    _xyxy_to_cxcywh_norm = _deps['_xyxy_to_cxcywh_norm']
    image_predictor = _deps['image_predictor']
    try:
        ws = _workspace(image_state)
        w, h = ws["image"].size
        text_prompt = (text_prompt or "").strip()
        has_positive = bool(pcs_state.get("positive_boxes"))
        has_negative = bool(pcs_state.get("negative_boxes"))
        if not text_prompt and not has_positive and not has_negative:
            raise ValueError("PCS needs a text prompt or bbox exemplar")
        if has_negative and not text_prompt and not has_positive:
            raise ValueError("PCS \u4e0d\u652f\u6301\u53ea\u4f7f\u7528\u8d1f\u6837\u672c bbox\uff0c\u8bf7\u5148\u6dfb\u52a0\u6587\u672c\u63d0\u793a\u6216\u6b63\u6837\u672c bbox")
        state = _fresh_state(image_state)
        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)
        for box in pcs_state.get("positive_boxes", []):
            state = image_predictor.add_geometric_prompt(_xyxy_to_cxcywh_norm(box, w, h), True, state)
        for box in pcs_state.get("negative_boxes", []):
            state = image_predictor.add_geometric_prompt(_xyxy_to_cxcywh_norm(box, w, h), False, state)
        state = image_predictor.set_confidence_threshold(float(threshold), state)
        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            pcs_state["instances"] = {}
            return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, "PCS found no instances")
        masks_np = masks.detach().cpu().numpy().astype(bool)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        probs = state.get("masks_logits")
        probs_np = None if probs is None else probs.detach().cpu().numpy().astype(np.float32)
        if probs_np is not None and probs_np.ndim == 4:
            probs_np = probs_np[:, 0]
        boxes_np = state["boxes"].detach().cpu().numpy()
        scores_np = state["scores"].detach().cpu().numpy()
        instances = {}
        for idx, mask in enumerate(masks_np):
            inst_id = idx + 1
            instances[inst_id] = _make_inst(inst_id, "pcs", mask, _norm_box(boxes_np[idx].tolist(), w, h), float(scores_np[idx]), pcs_prob=None if probs_np is None else probs_np[idx], history=[{"op":"pcs_grounding","text_prompt":text_prompt or ""}])
        pcs_state["instances"] = instances
        pcs_state["next_instance_id"] = len(instances) + 1
        pcs_state["text_prompt"] = text_prompt or ""
        info = f"PCS found {len(instances)} instances"
    except Exception as exc:
        info = f"PCS failed: {exc}"
    return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _create_pvs_from_pending_boxes_impl(_deps, image_state, pcs_state, pvs_state, mode, progress):
    _best = _deps['_best']
    _fresh_state = _deps['_fresh_state']
    _make_inst = _deps['_make_inst']
    _mask_box = _deps['_mask_box']
    _predict_inst = _deps['_predict_inst']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _pvs_progress = _deps['_pvs_progress']
    _view = _deps['_view']
    try:
        _pvs_progress(progress, 0.03, "\u51c6\u5907\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b")
        records = pvs_state.get("pending_bbox_records") or []
        boxes = (
            [record.get("box") for record in records]
            if records
            else list(pvs_state.get("pending_boxes", []))
        )
        if not boxes:
            raise ValueError("\u6ca1\u6709\u5f85\u751f\u6210\u7684 PVS bbox\uff0c\u8bf7\u5148\u5728\u56fe\u50cf\u4e0a\u6846\u9009\u4e00\u4e2a\u6216\u591a\u4e2a\u76ee\u6807")

        created_ids = []
        staged_instances = {}
        next_instance_id = int(pvs_state.get("next_instance_id", 1))
        _pvs_progress(progress, 0.12, f"\u8bfb\u53d6\u56fe\u50cf\u7f13\u5b58\uff0c\u5171 {len(boxes)} \u4e2a bbox")
        base_state = _fresh_state(image_state)
        for box_idx, box in enumerate(boxes, start=1):
            start = 0.18 + 0.62 * (box_idx - 1) / max(1, len(boxes))
            _pvs_progress(progress, start, f"SAM3 \u6b63\u5728\u751f\u6210\u7b2c {box_idx}/{len(boxes)} \u4e2a PVS \u5b9e\u4f8b", delay=0.06)
            pred = _predict_inst(base_state, box_xyxy_px=box)
            idx = _best(pred)
            mask = pred["masks"][idx]
            inst_id = next_instance_id + len(staged_instances)
            staged_instances[inst_id] = _make_inst(
                inst_id,
                "manual_pvs_bbox_batch",
                mask,
                _mask_box(mask),
                pred["scores"][idx],
                pvs_logits=pred["lowres_logits"][idx],
                history=[{"op":"create_from_pending_bbox","box_xyxy_px":box,"candidate_scores":pred["scores"].astype(float).tolist()}],
            )
            created_ids.append(inst_id)

        _pvs_progress(progress, 0.86, "\u66f4\u65b0 PVS \u5b9e\u4f8b\u6c60")
        _pvs_progress(progress, 0.96, "\u6e32\u67d3 PVS \u5206\u5272\u7ed3\u679c", delay=0.16)
        committed_instances = dict(pvs_state.get("instances") or {})
        committed_instances.update(staged_instances)
        pvs_state["instances"] = committed_instances
        pvs_state["next_instance_id"] = next_instance_id + len(staged_instances)
        pvs_state["active_instance_id"] = created_ids[-1]
        pvs_state["pending_bbox_records"] = []
        pvs_state["pending_boxes"] = []
        info = f"\u5df2\u4ece {len(created_ids)} \u4e2a\u5f85\u751f\u6210 bbox \u521b\u5efa PVS \u5b9e\u4f8b: {created_ids}"
    except Exception as exc:
        info = f"PVS \u6279\u91cf bbox \u751f\u6210\u5931\u8d25: {exc}"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _delete_selected_pending_pvs_bbox_impl(_deps, image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _pvs_pending_bbox_records = _deps['_pvs_pending_bbox_records']
    _sync_pvs_pending_boxes_from_records = _deps['_sync_pvs_pending_boxes_from_records']
    _view = _deps['_view']
    try:
        records = list(_pvs_pending_bbox_records(pvs_state))
        if not selected_bbox_id:
            raise ValueError("\u8bf7\u5148\u5728 PVS \u5f85\u751f\u6210 bbox \u5217\u8868\u4e2d\u9009\u62e9\u4e00\u4e2a bbox")
        target_id = int(selected_bbox_id)
        target = next((rec for rec in records if int(rec.get("id", -1)) == target_id), None)
        if target is None:
            raise ValueError("\u9009\u4e2d\u7684 PVS \u5f85\u751f\u6210 bbox \u5df2\u4e0d\u5b58\u5728\uff0c\u8bf7\u91cd\u65b0\u9009\u62e9")
        pvs_state["pending_bbox_records"] = [rec for rec in records if int(rec.get("id", -1)) != target_id]
        _sync_pvs_pending_boxes_from_records(pvs_state)
        info = f"\u5df2\u5220\u9664 PVS \u5f85\u751f\u6210 bbox ID {target_id}: {[round(float(v), 1) for v in target.get('box', [])]}"
    except Exception as exc:
        info = f"\u5220\u9664 PVS \u5f85\u751f\u6210 bbox \u5931\u8d25: {exc}"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _clear_pending_pvs_boxes_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _clear_pvs_pending_bboxes = _deps['_clear_pvs_pending_bboxes']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _view = _deps['_view']
    count = _clear_pvs_pending_bboxes(pvs_state)
    info = f"\u5df2\u6e05\u7a7a {count} \u4e2a\u5f85\u751f\u6210 PVS bbox\uff1b\u5df2\u751f\u6210\u5b9e\u4f8b\u4e0d\u4f1a\u88ab\u5220\u9664"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _set_active_pvs_impl(_deps, image_state, pcs_state, pvs_state, mode, selected_id):
    _view = _deps['_view']
    if selected_id:
        pvs_state["active_instance_id"] = int(selected_id)
        info = f"Selected PVS #{selected_id}"
    else:
        pvs_state["active_instance_id"] = None
        info = "No PVS instance selected"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _refine_active_pvs_with_point_impl(_deps, image_state, pvs_state, point, point_label, progress):
    _append_prompt_history = _deps['_append_prompt_history']
    _fresh_state = _deps['_fresh_state']
    _history_snapshot = _deps['_history_snapshot']
    _mask_box = _deps['_mask_box']
    _predict_inst = _deps['_predict_inst']
    _prompt_mask_size = _deps['_prompt_mask_size']
    _pvs_progress = _deps['_pvs_progress']
    active_id = pvs_state.get("active_instance_id")
    if active_id is None:
        raise ValueError("请先创建或选择一个 active PVS instance")
    try:
        active_id = int(active_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("active PVS instance ID 无效") from exc

    instances = pvs_state.get("instances", {})
    active_inst = instances.get(active_id)
    if active_inst is None:
        raise ValueError("active PVS instance 不存在，请重新选择")
    if active_inst.get("status") == "deleted":
        raise ValueError("active PVS instance 已删除，请重新选择")

    point_array = np.asarray(point, dtype=np.float32).reshape(-1)
    if point_array.shape != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point coordinate 必须是两个有限数值")
    width = int(image_state.get("width") or 0)
    height = int(image_state.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions 无效，请重新加载图像")
    if not (0.0 <= point_array[0] < width and 0.0 <= point_array[1] < height):
        raise ValueError("point coordinate 超出当前图像边界")
    try:
        point_label_value = float(point_label)
    except (TypeError, ValueError) as exc:
        raise ValueError("point label 必须是 0 或 1") from exc
    if not np.isfinite(point_label_value) or point_label_value not in (0.0, 1.0):
        raise ValueError("point label 必须是 0 或 1")
    point_label = int(point_label_value)
    point_xy = [float(point_array[0]), float(point_array[1])]

    previous_value = active_inst.get("pvs_lowres_logits")
    previous_status = active_inst.get("status")
    if previous_value is None:
        raise ValueError("active PVS instance 缺少上一轮 low-res logits")
    previous_logits = np.asarray(previous_value)
    expected = _prompt_mask_size()
    valid_previous_shape = (
        previous_logits.ndim == 2
        or (previous_logits.ndim == 3 and previous_logits.shape[0] == 1)
    ) and tuple(previous_logits.shape[-2:]) == expected
    if previous_logits.dtype != np.float32 or not valid_previous_shape or not np.isfinite(previous_logits).all():
        raise ValueError(f"active PVS instance logits 必须是有限 float32，形状为 {expected} 或 1x{expected}")

    point_name = "负向点" if point_label == 0 else "正向点"
    _pvs_progress(progress, 0.32, f"SAM3 正在根据{point_name}修缮 active instance", delay=0.12)
    pred = _predict_inst(
        _fresh_state(image_state),
        mask_input_lowres_logits=previous_logits.copy(),
        point_coords_px=[point_xy],
        point_labels=[point_label],
    )

    scores = np.asarray(pred.get("scores"), dtype=np.float32).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("predict_inst 返回的候选分数无效")
    idx = int(np.argmax(scores))

    masks = np.asarray(pred.get("masks"))
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim not in (3, 4) or masks.shape[0] != scores.size:
        raise ValueError("predict_inst 返回的候选 mask 数量或形状无效")
    mask = np.asarray(masks[idx])
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("predict_inst 返回的候选 mask 必须是二维图像")
    if mask.shape != (height, width):
        raise ValueError("predict_inst 返回的候选 mask 尺寸与当前图像不一致")
    if np.issubdtype(mask.dtype, np.number) and not np.isfinite(mask).all():
        raise ValueError("predict_inst 返回的候选 mask 包含非有限值")
    mask = mask.astype(bool)

    candidate_logits = np.asarray(pred.get("lowres_logits"), dtype=np.float32)
    if candidate_logits.ndim == 2:
        if scores.size != 1:
            raise ValueError("predict_inst 返回的 low-res logits 缺少候选维度")
        selected_logits = candidate_logits
    elif candidate_logits.ndim in (3, 4) and candidate_logits.shape[0] == scores.size:
        selected_logits = candidate_logits[idx]
    else:
        raise ValueError("predict_inst 返回的 low-res logits 数量或形状无效")
    valid_selected_shape = (
        selected_logits.ndim == 2
        or (selected_logits.ndim == 3 and selected_logits.shape[0] == 1)
    ) and tuple(selected_logits.shape[-2:]) == expected
    if not valid_selected_shape or not np.isfinite(selected_logits).all():
        raise ValueError("predict_inst 返回的 low-res logits 无效")

    score = float(scores[idx])
    box = _mask_box(mask)
    before = _history_snapshot(active_inst)
    after = {
        "box_xyxy_px": list(box),
        "score": score,
        "status": active_inst.get("status", "draft"),
    }
    prompt_type = "negative_point" if point_label == 0 else "positive_point"
    op = "negative_point_refine" if point_label == 0 else "positive_point_refine"
    updated_inst = dict(active_inst)
    updated_inst["mask_fullres_bool"] = mask
    updated_inst["box_xyxy_px"] = box
    updated_inst["score"] = score
    updated_inst["pvs_lowres_logits"] = selected_logits.copy()
    updated_inst["prompt_history"] = list(active_inst.get("prompt_history") or [])
    _append_prompt_history(
        updated_inst,
        {
            "op": op,
            "prompt": {"type": prompt_type, "point_xy_px": point_xy},
            "before": before,
            "after": after,
            "candidate_scores": scores.astype(float).tolist(),
        },
    )
    try:
        current_active_id = int(pvs_state.get("active_instance_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("active PVS instance 在预测期间发生变化，请重试") from exc
    if (
        current_active_id != active_id
        or pvs_state.get("instances") is not instances
        or instances.get(active_id) is not active_inst
        or active_inst.get("pvs_lowres_logits") is not previous_value
        or active_inst.get("status") != previous_status
    ):
        raise ValueError("active PVS instance 在预测期间发生变化，请重试")
    instances[active_id] = updated_inst
    return active_id, prompt_type


def _pvs_point_prompt_impl(_deps, image_state, pcs_state, pvs_state, mode, point_payload, point_kind, progress):
    _best = _deps['_best']
    _fresh_state = _deps['_fresh_state']
    _make_inst = _deps['_make_inst']
    _mask_box = _deps['_mask_box']
    _point_from_payload = _deps['_point_from_payload']
    _predict_inst = _deps['_predict_inst']
    _pvs_progress = _deps['_pvs_progress']
    _refine_active_pvs_with_point = _deps['_refine_active_pvs_with_point']
    _view = _deps['_view']
    try:
        is_negative = str(point_kind or "positive") == "negative"
        point_label = 0 if is_negative else 1
        point_name = "负向点" if is_negative else "正向点"
        _pvs_progress(progress, 0.04, f"准备{point_name} PVS 操作")
        point = _point_from_payload(point_payload, image_state)
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            if is_negative:
                raise ValueError("负向点必须先选择一个 active PVS instance")
            _pvs_progress(progress, 0.32, f"SAM3 正在根据{point_name}预测 mask", delay=0.12)
            pred = _predict_inst(_fresh_state(image_state), point_coords_px=[point], point_labels=[point_label])
            _pvs_progress(progress, 0.78, f"整理{point_name}候选 mask")
            idx = _best(pred)
            mask = pred["masks"][idx]
            inst_id = int(pvs_state.get("next_instance_id", 1))
            pvs_state.setdefault("instances", {})[inst_id] = _make_inst(inst_id, "manual_pvs_point", mask, _mask_box(mask), pred["scores"][idx], pvs_logits=pred["lowres_logits"][idx], history=[{"op":"create_from_positive_point","point_xy_px":point,"candidate_scores":pred["scores"].astype(float).tolist()}])
            pvs_state["active_instance_id"] = inst_id
            pvs_state["next_instance_id"] = inst_id + 1
            info = f"Created PVS instance #{inst_id} from positive point"
        else:
            refined_id, prompt_type = _refine_active_pvs_with_point(
                image_state,
                pvs_state,
                point,
                point_label,
                progress=progress,
            )
            _pvs_progress(progress, 0.78, f"整理{point_name}候选 mask")
            info = f"PVS #{refined_id} refined with {prompt_type}"
        _pvs_progress(progress, 0.96, "渲染 PVS 分割结果", delay=0.16)
    except Exception as exc:
        info = f"PVS point prompt failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _undo_pvs_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _active_instances = _deps['_active_instances']
    _restore = _deps['_restore']
    _view = _deps['_view']
    try:
        active_id = pvs_state.get("active_instance_id")
        active_inst = pvs_state.get("instances", {}).get(int(active_id)) if active_id is not None else None
        if active_inst is not None:
            history = active_inst.get("prompt_history", [])
            if history and history[-1].get("op") == "refine_with_layout_mask" and history[-1].get("before"):
                _restore(active_inst, history[-1]["before"])
                history.pop()
                info = f"已撤销 PVS #{active_id} 的版图精修"
                return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)
        items = _active_instances(pvs_state)
        if not items:
            raise ValueError("没有可撤销的 PVS 实例")
        inst = max(items, key=lambda item: int(item["id"]))
        pvs_state["instances"].pop(int(inst["id"]), None)
        if str(pvs_state.get("active_instance_id")) == str(inst["id"]):
            remaining = _active_instances(pvs_state)
            pvs_state["active_instance_id"] = max(remaining, key=lambda item: int(item["id"]))["id"] if remaining else None
        info = f"已撤销上一个 PVS 实例 #{inst['id']}"
    except Exception as exc:
        info = f"撤销上一个实例失败: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _delete_pvs_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _active_instances = _deps['_active_instances']
    _view = _deps['_view']
    try:
        items = _active_instances(pvs_state)
        if not items:
            raise ValueError("没有可清空的 PVS 实例")
        pvs_state["instances"] = {}
        pvs_state["active_instance_id"] = None
        info = f"已清空 {len(items)} 个 PVS 实例；待生成 bbox 不受影响"
    except Exception as exc:
        info = f"清空实例失败: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _accept_pvs_impl(_deps, image_state, pcs_state, pvs_state, mode):
    _view = _deps['_view']
    try:
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("Select a PVS instance first")
        pvs_state["instances"][int(active_id)]["status"] = "accepted"
        info = f"PVS #{active_id} accepted"
    except Exception as exc:
        info = f"Accept failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)
