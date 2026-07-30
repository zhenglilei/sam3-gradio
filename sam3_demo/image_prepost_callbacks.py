"""Image upload, crop, workspace, and template-matching callbacks."""

from __future__ import annotations

import copy
import json
import time
import uuid

import gradio as gr
import numpy as np
import torch
from PIL import Image

import image_crop_utils as _image_crop
import layout_transform_utils as _layout_tx
import template_match_workflow as _template_matching


def _init_workspace_impl(_deps, input_image, mode, session_state):
    _WORKSPACE_CACHE = _deps['_WORKSPACE_CACHE']
    _WORKSPACE_CACHE_LOCK = _deps['_WORKSPACE_CACHE_LOCK']
    _clear_workspace_cache = _deps['_clear_workspace_cache']
    _new_pcs_state = _deps['_new_pcs_state']
    _new_prompt_state = _deps['_new_prompt_state']
    _new_pvs_state = _deps['_new_pvs_state']
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pil_image = _deps['_pil_image']
    _prune_workspace_cache = _deps['_prune_workspace_cache']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _release_workspace_memory = _deps['_release_workspace_memory']
    _session_id_from_state = _deps['_session_id_from_state']
    _view = _deps['_view']
    image_predictor = _deps['image_predictor']
    pcs_state, pvs_state = _new_pcs_state(), _new_pvs_state()
    prompt_state = _new_prompt_state()
    session_id = _session_id_from_state(session_state)
    image_state = {"image_id": None, "width": 0, "height": 0, "session_id": session_id, "target_image_sha256": None, "interaction_revision": 0}
    if input_image is None:
        _clear_workspace_cache(session_id)
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, "Upload an image first", prompt_state), None
    if image_predictor is None:
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, "SAM3 image predictor is not initialized", prompt_state), None
    image = _pil_image(input_image)
    image_id = uuid.uuid4().hex
    target_hash = _layout_tx.image_pixel_sha256(image)
    try:
        base_state = image_predictor.set_image(image)
    except torch.OutOfMemoryError:
        info = "\u56fe\u50cf\u52a0\u8f7d\u5931\u8d25\uff1aGPU \u663e\u5b58\u4e0d\u8db3\u3002\u5f53\u524d\u5de5\u4f5c\u533a\u4fdd\u6301\u4e0d\u53d8\uff0c\u8bf7\u5173\u95ed\u5176\u4ed6 GPU \u4efb\u52a1\u6216\u91cd\u542f demo \u540e\u91cd\u8bd5\u3002"
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state), None
    _clear_workspace_cache(session_id)
    now = time.monotonic()
    with _WORKSPACE_CACHE_LOCK:
        _WORKSPACE_CACHE[image_id] = {
            "image": image,
            "base_state": base_state,
            "session_id": session_id,
            "target_image_sha256": target_hash,
            "created_at": now,
            "last_accessed_at": now,
        }
        removed = _prune_workspace_cache(now, protected_image_id=image_id)
    if removed:
        _release_workspace_memory()
    image_state = {"image_id": image_id, "width": image.width, "height": image.height, "session_id": session_id, "target_image_sha256": target_hash, "interaction_revision": 1}
    return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, f"Image loaded: {image.width}x{image.height}", prompt_state), None


def _init_workspace_with_layout_editor_impl(_deps, input_image, mode, session_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _init_workspace = _deps['_init_workspace']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _advance_layout_prompt_epoch(
        layout_state=layout_state,
        session_state=session_state,
    )
    result = _init_workspace(input_image, mode, session_state)
    image_state = result[0]
    if isinstance(layout_state, dict) and layout_state.get("layout_id"):
        editor = _layout_editor_payload(image_state, layout_state, "目标图像已更新，版图编辑器 payload 已刷新。")
    else:
        editor = _layout_editor_empty(image_state, "Image loaded; load or generate a layout mask next.")
    return (*result, editor)


def _attach_source_provenance_impl(_deps, init_result, source_state):
    _WORKSPACE_CACHE = _deps['_WORKSPACE_CACHE']
    _WORKSPACE_CACHE_LOCK = _deps['_WORKSPACE_CACHE_LOCK']
    result = list(init_result)
    image_state = dict(result[0] or {})
    if not image_state.get("image_id"):
        detail = str(result[12] or "") if len(result) > 12 else ""
        raise RuntimeError(detail or "SAM3 工作区初始化失败")
    if image_state.get("image_id") and source_state.get("source_image_id"):
        provenance = {
            "source_image_id": str(source_state["source_image_id"]),
            "source_image_sha256": str(source_state["source_image_sha256"]),
            "source_width": int(source_state["source_width"]),
            "source_height": int(source_state["source_height"]),
            "source_revision": int(source_state["source_revision"]),
            "crop_bbox_xyxy": list(source_state["crop_bbox_xyxy"]),
        }
        source_state["workspace_image_id"] = str(image_state["image_id"])
        source_state["workspace_hash"] = str(image_state.get("target_image_sha256") or "")
        image_state.update(provenance)
        with _WORKSPACE_CACHE_LOCK:
            workspace = _WORKSPACE_CACHE.get(str(image_state["image_id"]))
            if workspace is not None:
                workspace.update(provenance)
    result[0] = image_state
    return tuple(result)


def _source_upload_workspace_impl(_deps, input_image, mode, session_state, layout_state):
    _attach_source_provenance = _deps['_attach_source_provenance']
    _clear_source_image_cache = _deps['_clear_source_image_cache']
    _init_workspace_with_layout_editor = _deps['_init_workspace_with_layout_editor']
    _new_source_image_state = _deps['_new_source_image_state']
    _session_id_from_state = _deps['_session_id_from_state']
    _source_gesture_payload = _deps['_source_gesture_payload']
    _source_image_cache_get = _deps['_source_image_cache_get']
    _source_image_cache_put = _deps['_source_image_cache_put']
    session_id = _session_id_from_state(session_state)
    _clear_source_image_cache(session_id)
    if input_image is None:
        source_state = _new_source_image_state(session_id)
        init_result = _init_workspace_with_layout_editor(None, mode, session_state, layout_state)
        return (
            source_state,
            _source_gesture_payload(source_state),
            "请先上传完整原图",
            *init_result,
        )
    try:
        source_state = _source_image_cache_put(session_id, input_image)
        source = _source_image_cache_get(source_state)
        init_result = _init_workspace_with_layout_editor(source, mode, session_state, layout_state)
        init_result = _attach_source_provenance(init_result, source_state)
        status = f"已载入完整原图 {source.width}x{source.height}；当前使用整图"
        return source_state, _source_gesture_payload(source_state, status), status, *init_result
    except Exception as exc:
        source_state = _new_source_image_state(session_id)
        init_result = _init_workspace_with_layout_editor(None, mode, session_state, layout_state)
        status = f"完整原图加载失败: {exc}"
        return source_state, _source_gesture_payload(source_state, status), status, *init_result


def _record_source_crop_gesture_impl(_deps, source_state, gesture_payload):
    _source_gesture_payload = _deps['_source_gesture_payload']
    _validate_gesture_intent = _deps['_validate_gesture_intent']
    state = dict(source_state or {})
    try:
        gesture, start, end = _validate_gesture_intent(
            gesture_payload,
            state,
            allowed_gestures={"drag"},
        )
        if gesture != "drag":
            raise ValueError("请拖拽矩形选择裁剪区域")
        box = _image_crop.normalize_crop_box(
            start,
            end,
            int(state.get("source_width") or 0),
            int(state.get("source_height") or 0),
        )
        state["pending_crop_bbox_xyxy"] = list(box)
        status = f"待应用裁剪区域: {list(box)}"
    except Exception as exc:
        state["pending_crop_bbox_xyxy"] = None
        status = f"裁剪框无效: {exc}"
    return (
        state,
        _source_gesture_payload(state, status, retain_selection=state.get("pending_crop_bbox_xyxy") is not None),
        status,
    )


def _crop_failure_outputs_impl(_deps, source_state, status):
    _source_gesture_payload = _deps['_source_gesture_payload']
    return (
        source_state,
        _source_gesture_payload(source_state, status),
        status,
        *([gr.update()] * 16),
    )


def _apply_source_crop_impl(_deps, source_state, mode, session_state, layout_state):
    _attach_source_provenance = _deps['_attach_source_provenance']
    _crop_failure_outputs = _deps['_crop_failure_outputs']
    _init_workspace_with_layout_editor = _deps['_init_workspace_with_layout_editor']
    _source_gesture_payload = _deps['_source_gesture_payload']
    _source_image_cache_get = _deps['_source_image_cache_get']
    state = dict(source_state or {})
    try:
        source = _source_image_cache_get(state)
        box = state.get("pending_crop_bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("请先在完整原图上拖拽矩形裁剪框")
        box = _image_crop.normalize_crop_box(
            box[:2],
            box[2:],
            source.width,
            source.height,
        )
        cropped = _image_crop.crop_pil_image(source, box)
        next_state = dict(state)
        next_state["crop_bbox_xyxy"] = list(box)
        next_state["pending_crop_bbox_xyxy"] = None
        init_result = _init_workspace_with_layout_editor(cropped, mode, session_state, layout_state)
        init_result = _attach_source_provenance(init_result, next_state)
        status = f"已应用裁剪 {list(box)}；工作图尺寸 {cropped.width}x{cropped.height}"
        return next_state, _source_gesture_payload(next_state, status), status, *init_result
    except Exception as exc:
        return _crop_failure_outputs(state, f"应用裁剪失败: {exc}")


def _use_full_source_image_impl(_deps, source_state, mode, session_state, layout_state):
    _attach_source_provenance = _deps['_attach_source_provenance']
    _crop_failure_outputs = _deps['_crop_failure_outputs']
    _init_workspace_with_layout_editor = _deps['_init_workspace_with_layout_editor']
    _source_gesture_payload = _deps['_source_gesture_payload']
    _source_image_cache_get = _deps['_source_image_cache_get']
    state = dict(source_state or {})
    try:
        source = _source_image_cache_get(state)
        whole = list(_image_crop.whole_image_crop_box(source.width, source.height))
        if (
            list(state.get("crop_bbox_xyxy") or []) == whole
            and not state.get("pending_crop_bbox_xyxy")
            and state.get("workspace_image_id")
        ):
            return _crop_failure_outputs(state, "当前已使用整图，无需重复应用")
        next_state = dict(state)
        next_state["crop_bbox_xyxy"] = whole
        next_state["pending_crop_bbox_xyxy"] = None
        init_result = _init_workspace_with_layout_editor(source, mode, session_state, layout_state)
        init_result = _attach_source_provenance(init_result, next_state)
        status = f"已恢复完整原图 {source.width}x{source.height}"
        return next_state, _source_gesture_payload(next_state, status, retain_selection=False), status, *init_result
    except Exception as exc:
        return _crop_failure_outputs(state, f"恢复整图失败: {exc}")


def _clear_template_match_outputs_impl(_deps, status):
    _new_template_match_state = _deps['_new_template_match_state']
    return _new_template_match_state(), None, None, str(status)


def _publish_template_match_export_impl(_deps, source_image, workflow, source_state, image_state):
    _publish_segmentation_zip = _deps['_publish_segmentation_zip']
    runtime_export_dir = _deps['runtime_export_dir']
    export_id = uuid.uuid4().hex
    export_dir = runtime_export_dir / f"template_match_{export_id}"
    masks_dir = export_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    source_image.convert("RGB").save(export_dir / "original_image.png")
    seed_mask = np.asarray(workflow["seed_mask_fullres_bool"], dtype=bool)
    Image.fromarray(seed_mask.astype(np.uint8) * 255, mode="L").save(export_dir / "seed_mask.png")
    Image.fromarray(np.asarray(workflow["overlay_rgb"], dtype=np.uint8), mode="RGB").save(
        export_dir / "template_match_overlay.png"
    )
    for index, mask in enumerate(workflow["match_masks_fullres_bool"], start=1):
        Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(
            masks_dir / f"match_{index:04d}.png"
        )
    manifest = copy.deepcopy(workflow["result"])
    manifest["source_image"] = {
        "image_id": str(source_state.get("source_image_id") or ""),
        "pixel_sha256": str(source_state.get("source_image_sha256") or ""),
        "width": int(source_state.get("source_width") or 0),
        "height": int(source_state.get("source_height") or 0),
    }
    manifest["workspace"] = {
        "image_id": str(image_state.get("image_id") or ""),
        "pixel_sha256": str(image_state.get("target_image_sha256") or ""),
        "crop_bbox_xyxy": list(image_state.get("crop_bbox_xyxy") or []),
    }
    for match in manifest.get("matches", []):
        match["mask_file"] = f"masks/match_{int(match['match_id']):04d}.png"
    with (export_dir / "matches.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return _publish_segmentation_zip(export_dir, f"template_match_{export_id}.zip"), manifest


def _run_template_matching_impl(_deps, source_state, image_state, pvs_state, mode, match_threshold, expand_threshold, nms_threshold):
    _clear_template_match_outputs = _deps['_clear_template_match_outputs']
    _is_pvs_pool_mode = _deps['_is_pvs_pool_mode']
    _publish_template_match_export = _deps['_publish_template_match_export']
    _source_image_cache_get = _deps['_source_image_cache_get']
    _workspace = _deps['_workspace']
    try:
        if not _is_pvs_pool_mode(mode):
            raise ValueError("模板匹配需要先在 PVS Manual 或 Layout Mask 模式获得 active PVS 实例")
        _workspace(image_state)
        source = _source_image_cache_get(source_state)
        if str(source_state.get("workspace_image_id") or "") != str(image_state.get("image_id") or ""):
            raise ValueError("当前工作图 provenance 已过期，请重新应用裁剪")
        if str(source_state.get("workspace_hash") or "") != str(
            image_state.get("target_image_sha256") or ""
        ):
            raise ValueError("当前工作图 hash 已过期，请重新应用裁剪")
        source_id = str(source_state.get("source_image_id") or "")
        if str(image_state.get("source_image_id") or "") != source_id:
            raise ValueError("当前分割工作区不属于这张完整原图，请重新应用裁剪")
        if str(image_state.get("source_image_sha256") or "") != str(source_state.get("source_image_sha256") or ""):
            raise ValueError("完整原图 provenance 已过期，请重新应用裁剪")
        crop_bbox = list(image_state.get("crop_bbox_xyxy") or [])
        if crop_bbox != list(source_state.get("crop_bbox_xyxy") or []):
            raise ValueError("当前裁剪 provenance 已过期，请重新应用裁剪")
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("请先完成智能分割并选择当前 PVS 实例")
        workflow = _template_matching.run_template_match_workflow(
            np.asarray(source.convert("RGB")),
            crop_bbox,
            pvs_state,
            active_instance_id=active_id,
            match_threshold=float(0.7 if match_threshold in (None, "") else match_threshold),
            expand_threshold=int(20 if expand_threshold in (None, "") else expand_threshold),
            nms_threshold=float(0.3 if nms_threshold in (None, "") else nms_threshold),
        )
        zip_path, manifest = _publish_template_match_export(
            source,
            workflow,
            source_state,
            image_state,
        )
        count = int(manifest.get("match_count") or 0)
        state = {
            "schema_version": 1,
            "source_image_id": source_id,
            "workspace_image_id": str(image_state.get("image_id") or ""),
            "active_instance_id": active_id,
            "result": manifest,
        }
        status = f"模板匹配完成：{count} matches；结果使用完整原图坐标，不写入 PVS 实例池"
        return (
            state,
            Image.fromarray(np.asarray(workflow["overlay_rgb"], dtype=np.uint8), mode="RGB"),
            str(zip_path),
            status,
        )
    except Exception as exc:
        return _clear_template_match_outputs(f"模板匹配失败: {exc}")
