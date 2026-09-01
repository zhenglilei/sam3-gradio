"""Image upload, crop, workspace, and template-matching callbacks."""

from __future__ import annotations

import copy
import json
import time
import uuid

import gradio as gr
import numpy as np
from PIL import Image

import image_crop_utils as _image_crop
import layout_transform_utils as _layout_tx
import template_match_workflow as _template_matching


def _template_state_session_id(state, label):
    """Return a required session id from one template callback state."""
    if not isinstance(state, dict):
        raise ValueError(f"{label} is missing")
    session_id = str(state.get("session_id") or "").strip()
    if not session_id:
        raise ValueError(f"{label} session_id is missing")
    if len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
        raise ValueError(f"{label} has invalid session_id")
    return session_id


def _validated_template_session_id(source_state, image_state, pvs_state):
    """Require all template inputs to belong to one non-empty session."""
    session_ids = (
        _template_state_session_id(source_state, "source_state"),
        _template_state_session_id(image_state, "image_state"),
        _template_state_session_id(pvs_state, "pvs_state"),
    )
    if len(set(session_ids)) != 1:
        raise ValueError("source_state, image_state and pvs_state session_id must match")
    return session_ids[0]


def _template_failure_session_id(*states):
    """Keep a usable owner id when an internal template operation fails."""
    for state in states:
        try:
            return _template_state_session_id(state, "template state")
        except ValueError:
            continue
    return None


def _safe_template_session_component(session_id):
    """Validate the server-issued session id before using it in a path."""
    return _template_state_session_id({"session_id": session_id}, "template export")


def _init_workspace_impl(_deps, input_image, mode, session_state):
    _WORKSPACE_CACHE = _deps['_WORKSPACE_CACHE']
    _WORKSPACE_CACHE_LOCK = _deps['_WORKSPACE_CACHE_LOCK']
    _clear_workspace_cache = _deps['_clear_workspace_cache']
    _evict_workspace_images = _deps['_evict_workspace_images']
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
    session_id = _session_id_from_state(session_state)
    pcs_state, pvs_state = _new_pcs_state(session_id), _new_pvs_state(session_id)
    prompt_state = _new_prompt_state(session_id)
    image_state = {"image_id": None, "width": 0, "height": 0, "session_id": session_id, "target_image_sha256": None, "interaction_revision": 0}
    if input_image is None:
        _clear_workspace_cache(session_id)
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, "Upload an image first", prompt_state), None
    image = _pil_image(input_image)
    image_id = uuid.uuid4().hex
    target_hash = _layout_tx.image_pixel_sha256(image)
    _clear_workspace_cache(session_id)
    now = time.monotonic()
    with _WORKSPACE_CACHE_LOCK:
        _WORKSPACE_CACHE[image_id] = {
            "image": image,
            "session_id": session_id,
            "target_image_sha256": target_hash,
            "created_at": now,
            "last_accessed_at": now,
        }
        removed = _prune_workspace_cache(now, protected_image_id=image_id)
    if removed:
        _release_workspace_memory()
        _evict_workspace_images(removed)
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


def _clear_template_match_outputs_impl(_deps, status, session_id=None):
    _new_template_match_state = _deps['_new_template_match_state']
    return _new_template_match_state(session_id), None, None, str(status)


def _template_pvs_instance(pvs_state, instance_id):
    instances = pvs_state.get("instances") if isinstance(pvs_state, dict) else None
    if not isinstance(instances, dict):
        raise ValueError("PVS 实例池不可用")
    candidates = (instance_id, str(instance_id))
    try:
        candidates = (*candidates, int(instance_id))
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        instance = instances.get(candidate)
        if isinstance(instance, dict):
            if instance.get("status") == "deleted":
                raise ValueError(f"PVS #{instance_id} 已删除")
            return instance
    raise ValueError(f"PVS #{instance_id} 不存在")


def _template_instance_choices_impl(_deps, pvs_state):
    _active_instances = _deps["_active_instances"]
    choices = [
        (f"PVS #{item['id']}", str(item["id"]))
        for item in _active_instances(pvs_state or {})
    ]
    selected = (pvs_state or {}).get("template_match_instance_id")
    if selected is None:
        selected = (pvs_state or {}).get("active_instance_id")
    value = str(selected) if selected is not None and any(choice[1] == str(selected) for choice in choices) else None
    return gr.update(choices=choices, value=value)


def _preview_template_instance_impl(
    _deps,
    source_state,
    image_state,
    pvs_state,
    selected_id,
):
    _clear_template_match_outputs = _deps["_clear_template_match_outputs"]
    _new_template_match_state = _deps["_new_template_match_state"]
    _source_image_cache_get = _deps["_source_image_cache_get"]
    if selected_id in (None, ""):
        pvs_state.pop("template_match_instance_id", None)
        session_id = _template_failure_session_id(source_state, image_state, pvs_state)
        return (
            pvs_state,
            *_clear_template_match_outputs(
                {"session_id": session_id} if session_id else None,
                "请选择用于模板匹配的 PVS 实例",
            ),
        )
    try:
        session_id = _validated_template_session_id(source_state, image_state, pvs_state)
        source = _source_image_cache_get(source_state)
        if str(source_state.get("workspace_image_id") or "") != str(image_state.get("image_id") or ""):
            raise ValueError("当前工作图 provenance 已过期，请重新应用裁剪")
        if str(source_state.get("workspace_hash") or "") != str(image_state.get("target_image_sha256") or ""):
            raise ValueError("当前工作图 hash 已过期，请重新应用裁剪")
        crop_bbox = list(image_state.get("crop_bbox_xyxy") or [])
        if crop_bbox != list(source_state.get("crop_bbox_xyxy") or []):
            raise ValueError("当前裁剪 provenance 已过期，请重新应用裁剪")
        instance = _template_pvs_instance(pvs_state, selected_id)
        seed_mask = _template_matching.map_crop_mask_to_source(
            instance.get("mask_fullres_bool"),
            (source.height, source.width),
            crop_bbox,
        )
        overlay = _template_matching.render_template_overlay(
            np.asarray(source.convert("RGB")),
            seed_mask,
            [],
        )
        selected_id = int(selected_id)
        pvs_state["template_match_instance_id"] = selected_id
        state = _new_template_match_state(session_id)
        state.update(
            {
                "source_image_id": str(source_state.get("source_image_id") or ""),
                "workspace_image_id": str(image_state.get("image_id") or ""),
                "active_instance_id": selected_id,
            }
        )
        return (
            pvs_state,
            state,
            Image.fromarray(overlay, mode="RGB"),
            None,
            f"已选择 PVS #{selected_id} 作为模板；黄色轮廓为当前模板",
        )
    except Exception as exc:
        session_id = _template_failure_session_id(source_state, image_state, pvs_state)
        return (
            pvs_state,
            *_clear_template_match_outputs(
                {"session_id": session_id} if session_id else None,
                f"模板预览失败: {exc}",
            ),
        )


def _template_match_selection_choices_impl(_deps, template_match_state):
    result = template_match_state.get("result") if isinstance(template_match_state, dict) else None
    matches = result.get("matches") if isinstance(result, dict) else None
    choices = []
    for match in matches or []:
        if not isinstance(match, dict):
            continue
        match_id = int(match.get("match_id"))
        score = float(match.get("score", 0.0))
        choices.append((f"M{match_id}  score={score:.3f}", str(match_id)))
    return gr.update(choices=choices, value=[value for _, value in choices])


def _export_template_match_selection_impl(
    _deps,
    source_state,
    image_state,
    pvs_state,
    template_match_state,
    export_scope,
    selected_match_ids,
):
    _publish_template_match_export = _deps["_publish_template_match_export"]
    _source_image_cache_get = _deps["_source_image_cache_get"]
    try:
        session_id = _validated_template_session_id(source_state, image_state, pvs_state)
        if _template_state_session_id(template_match_state, "template_match_state") != session_id:
            raise ValueError("模板匹配结果不属于当前会话")
        if str(template_match_state.get("source_image_id") or "") != str(source_state.get("source_image_id") or ""):
            raise ValueError("模板匹配结果已过期，请重新运行")
        if str(template_match_state.get("workspace_image_id") or "") != str(image_state.get("image_id") or ""):
            raise ValueError("模板匹配工作图已过期，请重新运行")
        result = template_match_state.get("result")
        matches = result.get("matches") if isinstance(result, dict) else None
        if not isinstance(matches, list) or not matches:
            raise ValueError("没有可导出的模板匹配结果")
        available = {int(match["match_id"]): match for match in matches}
        scope = str(export_scope or "all")
        if scope == "all":
            selected_ids = list(available)
        elif scope == "selected":
            selected_ids = []
            for value in selected_match_ids or []:
                match_id = int(value)
                if match_id not in available:
                    raise ValueError(f"匹配实例 M{match_id} 不存在")
                if match_id not in selected_ids:
                    selected_ids.append(match_id)
            if not selected_ids:
                raise ValueError("请选择至少一个匹配实例")
        else:
            raise ValueError("未知的保存范围")

        source = _source_image_cache_get(source_state)
        crop_bbox = list(image_state.get("crop_bbox_xyxy") or [])
        if crop_bbox != list(source_state.get("crop_bbox_xyxy") or []):
            raise ValueError("当前裁剪 provenance 已过期，请重新应用裁剪")
        seed_id = template_match_state.get("active_instance_id")
        instance = _template_pvs_instance(pvs_state, seed_id)
        seed_mask = _template_matching.map_crop_mask_to_source(
            instance.get("mask_fullres_bool"),
            (source.height, source.width),
            crop_bbox,
        )
        selected_matches = [copy.deepcopy(available[match_id]) for match_id in selected_ids]
        selected_masks = []
        for match in selected_matches:
            translation = match.get("translation_xy")
            if not isinstance(translation, (list, tuple)) or len(translation) != 2:
                raise ValueError(f"匹配实例 M{match['match_id']} 缺少合法位移")
            selected_masks.append(
                _template_matching.translate_source_mask(
                    seed_mask,
                    int(translation[0]),
                    int(translation[1]),
                )
            )
        filtered_result = copy.deepcopy(result)
        filtered_result["matches"] = selected_matches
        filtered_result["match_count"] = len(selected_matches)
        filtered_result["selection"] = {
            "scope": scope,
            "selected_match_ids": selected_ids,
        }
        source_rgb = np.asarray(source.convert("RGB"))
        workflow = {
            "result": filtered_result,
            "seed_mask_fullres_bool": seed_mask,
            "match_masks_fullres_bool": selected_masks,
            "overlay_rgb": _template_matching.render_template_overlay(
                source_rgb,
                seed_mask,
                selected_masks,
                selected_matches,
            ),
        }
        zip_path, _ = _publish_template_match_export(
            source,
            workflow,
            source_state,
            image_state,
        )
        scope_text = "全部" if scope == "all" else "所选"
        return str(zip_path), f"已生成{scope_text}结果下载包：{len(selected_ids)} 个匹配实例"
    except Exception as exc:
        return None, f"生成模板匹配下载包失败: {exc}"


def _publish_template_match_export_impl(_deps, source_image, workflow, source_state, image_state):
    _publish_segmentation_zip = _deps['_publish_segmentation_zip']
    runtime_export_dir = _deps['runtime_export_dir']
    source_session_id = _template_state_session_id(source_state, "source_state")
    image_session_id = _template_state_session_id(image_state, "image_state")
    if source_session_id != image_session_id:
        raise ValueError("source_state and image_state session_id must match")
    session_id = _safe_template_session_component(source_session_id)
    export_id = uuid.uuid4().hex
    export_dir = runtime_export_dir / session_id / f"template_match_{export_id}"
    masks_dir = export_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    source_image.convert("RGB").save(export_dir / "original_image.png")
    seed_mask = np.asarray(workflow["seed_mask_fullres_bool"], dtype=bool)
    Image.fromarray(seed_mask.astype(np.uint8) * 255, mode="L").save(export_dir / "seed_mask.png")
    Image.fromarray(np.asarray(workflow["overlay_rgb"], dtype=np.uint8), mode="RGB").save(
        export_dir / "template_match_overlay.png"
    )
    matches = list(workflow["result"].get("matches") or [])
    match_masks = list(workflow["match_masks_fullres_bool"])
    if len(matches) != len(match_masks):
        raise ValueError("template match metadata and masks must align")
    for index, (match, mask) in enumerate(zip(matches, match_masks), start=1):
        match_id = int(match.get("match_id", index))
        if match_id <= 0:
            raise ValueError("template match id must be positive")
        Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(
            masks_dir / f"match_{match_id:04d}.png"
        )
    manifest = copy.deepcopy(workflow["result"])
    manifest["session_id"] = session_id
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
    return (
        _publish_segmentation_zip(
            export_dir,
            f"template_match_{export_id}.zip",
            session_id,
        ),
        manifest,
    )


def _run_template_matching_impl(_deps, source_state, image_state, pvs_state, mode, match_threshold, expand_threshold, nms_threshold):
    _clear_template_match_outputs = _deps['_clear_template_match_outputs']
    _is_pvs_pool_mode = _deps['_is_pvs_pool_mode']
    _publish_template_match_export = _deps['_publish_template_match_export']
    _source_image_cache_get = _deps['_source_image_cache_get']
    _workspace = _deps['_workspace']
    try:
        session_id = _validated_template_session_id(source_state, image_state, pvs_state)
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
        seed_instance_id = pvs_state.get("template_match_instance_id")
        if seed_instance_id is None:
            seed_instance_id = pvs_state.get("active_instance_id")
        if seed_instance_id is None:
            raise ValueError("请先完成智能分割并选择当前 PVS 实例")
        workflow = _template_matching.run_template_match_workflow(
            np.asarray(source.convert("RGB")),
            crop_bbox,
            pvs_state,
            active_instance_id=seed_instance_id,
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
            "session_id": session_id,
            "schema_version": 1,
            "source_image_id": source_id,
            "workspace_image_id": str(image_state.get("image_id") or ""),
            "active_instance_id": seed_instance_id,
            "result": manifest,
        }
        status = (
            f"模板匹配完成：PVS #{seed_instance_id}，{count} matches；"
            "结果使用完整原图坐标，不写入 PVS 实例池"
        )
        return (
            state,
            Image.fromarray(np.asarray(workflow["overlay_rgb"], dtype=np.uint8), mode="RGB"),
            str(zip_path),
            status,
        )
    except Exception as exc:
        failure_session_id = _template_failure_session_id(source_state, image_state, pvs_state)
        return _clear_template_match_outputs(
            {"session_id": failure_session_id} if failure_session_id else None,
            f"模板匹配失败: {exc}",
        )
