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


def _template_available_instance_ids(pvs_state):
    instances = pvs_state.get("instances") if isinstance(pvs_state, dict) else None
    if not isinstance(instances, dict):
        return []
    values = []
    for key, instance in instances.items():
        if not isinstance(instance, dict) or instance.get("status") == "deleted":
            continue
        try:
            instance_id = int(instance.get("id", key))
        except (TypeError, ValueError):
            continue
        if instance_id not in values:
            values.append(instance_id)
    return sorted(values)


def _template_selected_instance_ids(pvs_state, selected_values=None, *, default_all=False):
    if selected_values is None:
        if isinstance(pvs_state, dict) and "template_match_instance_ids" in pvs_state:
            selected_values = pvs_state.get("template_match_instance_ids")
        elif isinstance(pvs_state, dict) and pvs_state.get("template_match_instance_id") is not None:
            selected_values = [pvs_state.get("template_match_instance_id")]
        elif default_all:
            selected_values = _template_available_instance_ids(pvs_state)
        else:
            selected_values = []
    if selected_values in (None, ""):
        selected_values = []
    elif not isinstance(selected_values, (list, tuple, set)):
        selected_values = [selected_values]
    selected_ids = []
    for value in selected_values:
        try:
            instance_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PVS 实例 {value!r} 非法") from exc
        _template_pvs_instance(pvs_state, instance_id)
        if instance_id not in selected_ids:
            selected_ids.append(instance_id)
    return selected_ids


def _template_group_alias(index):
    number = int(index) + 1
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}x"


def _template_group_id(instance_id):
    return f"PVS-{int(instance_id)}"


def _template_mask_iou(
    left,
    right,
    *,
    left_area=None,
    right_area=None,
    left_bbox=None,
    right_bbox=None,
):
    """Return pixel IoU, optionally using cached mask geometry."""
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("template match masks must be two-dimensional")
    if left_array.shape != right_array.shape:
        raise ValueError("template match masks must have the same shape")
    if left_area is None:
        left_area = int(np.count_nonzero(left_array))
    if right_area is None:
        right_area = int(np.count_nonzero(right_array))
    if not left_area or not right_area:
        return 0.0
    if left_bbox is None:
        ys, xs = np.nonzero(left_array)
        left_bbox = (
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if len(xs)
            else None
        )
    if right_bbox is None:
        ys, xs = np.nonzero(right_array)
        right_bbox = (
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if len(xs)
            else None
        )
    if left_bbox is None or right_bbox is None:
        return 0.0
    x1 = max(left_bbox[0], right_bbox[0])
    y1 = max(left_bbox[1], right_bbox[1])
    x2 = min(left_bbox[2], right_bbox[2])
    y2 = min(left_bbox[3], right_bbox[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = np.count_nonzero(
        left_array[y1:y2, x1:x2] & right_array[y1:y2, x1:x2]
    )
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def _deduplicate_template_groups(groups, group_masks, iou_threshold):
    """Remove only cross-group duplicate matches with score-ordered NMS.

    Candidates are compared only when their source groups differ.  A higher
    ``match['score']`` wins an overlap strictly above ``iou_threshold``;
    equal scores retain the candidate that appeared first in the original
    group/match order.  The returned groups, group masks, and flattened lists
    are rebuilt together so their positional correspondence is preserved.
    """
    if len(groups) != len(group_masks):
        raise ValueError("template groups and group masks must be aligned")
    threshold = float(iou_threshold)
    candidates = []
    for group_index, (group, group_mask) in enumerate(zip(groups, group_masks)):
        matches = list(group.get("matches") or [])
        masks = list(group_mask.get("match_masks_fullres_bool") or [])
        if len(matches) != len(masks):
            raise ValueError(
                f"template group {group.get('group_id', group_index)} matches and masks are misaligned"
            )
        for match_index, (match, mask) in enumerate(zip(matches, masks)):
            if not isinstance(match, dict):
                raise ValueError("template match metadata must be a mapping")
            try:
                score = float(match["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("template match metadata must contain a numeric score") from exc
            if not np.isfinite(score):
                raise ValueError("template match score must be finite")
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.ndim != 2:
                raise ValueError("template match masks must be two-dimensional")
            ys, xs = np.nonzero(mask_array)
            mask_bbox = (
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                if len(xs)
                else None
            )
            candidates.append(
                {
                    "group_index": group_index,
                    "match_index": match_index,
                    "original_order": len(candidates),
                    "score": score,
                    "match": match,
                    "mask": mask_array,
                    "area": int(len(xs)),
                    "bbox": mask_bbox,
                }
            )

    retained = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item["score"], item["original_order"]),
    ):
        if any(
            candidate["group_index"] != previous["group_index"]
            and _template_mask_iou(
                candidate["mask"],
                previous["mask"],
                left_area=candidate["area"],
                right_area=previous["area"],
                left_bbox=candidate["bbox"],
                right_bbox=previous["bbox"],
            ) > threshold
            for previous in retained
        ):
            continue
        retained.append(candidate)

    retained_indices = {}
    for candidate in retained:
        retained_indices.setdefault(candidate["group_index"], set()).add(
            candidate["match_index"]
        )

    filtered_groups = []
    filtered_group_masks = []
    flat_matches = []
    flat_masks = []
    for group_index, (group, group_mask) in enumerate(zip(groups, group_masks)):
        matches = list(group.get("matches") or [])
        masks = list(group_mask.get("match_masks_fullres_bool") or [])
        keep = sorted(retained_indices.get(group_index, set()))
        filtered_group = copy.deepcopy(group)
        filtered_group["matches"] = [copy.deepcopy(matches[index]) for index in keep]
        filtered_group["match_count"] = len(keep)
        filtered_group_mask = copy.deepcopy(group_mask)
        filtered_group_mask["match_masks_fullres_bool"] = [
            np.asarray(masks[index], dtype=bool).copy() for index in keep
        ]
        filtered_groups.append(filtered_group)
        filtered_group_masks.append(filtered_group_mask)
        flat_matches.extend(filtered_group["matches"])
        flat_masks.extend(filtered_group_mask["match_masks_fullres_bool"])
    return filtered_groups, filtered_group_masks, flat_matches, flat_masks


def _template_result_groups(result, fallback_instance_id=None):
    if not isinstance(result, dict):
        return []
    groups = result.get("groups")
    if isinstance(groups, list):
        return [group for group in groups if isinstance(group, dict)]
    matches = result.get("matches")
    if not isinstance(matches, list):
        return []
    seed = result.get("seed") if isinstance(result.get("seed"), dict) else {}
    instance_id = seed.get("instance_id", fallback_instance_id)
    if instance_id is None:
        return []
    return [
        {
            "group_id": _template_group_id(instance_id),
            "group_label": "Ax",
            "source_instance_id": int(instance_id),
            "seed": copy.deepcopy(seed),
            "blockers": copy.deepcopy(result.get("blockers") or {}),
            "match_count": len(matches),
            "matches": copy.deepcopy(matches),
        }
    ]


def _template_instance_choices_impl(_deps, pvs_state):
    _active_instances = _deps["_active_instances"]
    choices = [
        (f"PVS #{item['id']}", str(item["id"]))
        for item in _active_instances(pvs_state or {})
    ]
    available = {value for _, value in choices}
    if isinstance(pvs_state, dict) and "template_match_instance_ids" in pvs_state:
        selected = pvs_state.get("template_match_instance_ids") or []
    else:
        selected = [value for _, value in choices]
    value = [str(item) for item in selected if str(item) in available]
    return gr.update(choices=choices, value=value)


def _preview_template_instance_impl(
    _deps,
    source_state,
    image_state,
    pvs_state,
    selected_ids,
):
    _clear_template_match_outputs = _deps["_clear_template_match_outputs"]
    _new_template_match_state = _deps["_new_template_match_state"]
    _source_image_cache_get = _deps["_source_image_cache_get"]
    if not selected_ids:
        pvs_state["template_match_instance_ids"] = []
        pvs_state.pop("template_match_instance_id", None)
        session_id = _template_failure_session_id(source_state, image_state, pvs_state)
        return (
            pvs_state,
            *_clear_template_match_outputs(
                {"session_id": session_id} if session_id else None,
                "请选择至少一个用于模板匹配的 PVS 实例",
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
        selected_ids = _template_selected_instance_ids(pvs_state, selected_ids)
        seed_mask = np.zeros((source.height, source.width), dtype=bool)
        for selected_id in selected_ids:
            instance = _template_pvs_instance(pvs_state, selected_id)
            seed_mask |= _template_matching.map_crop_mask_to_source(
                instance.get("mask_fullres_bool"),
                (source.height, source.width),
                crop_bbox,
            )
        overlay = _template_matching.render_template_overlay(
            np.asarray(source.convert("RGB")),
            seed_mask,
            [],
        )
        pvs_state["template_match_instance_ids"] = selected_ids
        pvs_state.pop("template_match_instance_id", None)
        state = _new_template_match_state(session_id)
        state.update(
            {
                "source_image_id": str(source_state.get("source_image_id") or ""),
                "workspace_image_id": str(image_state.get("image_id") or ""),
                "selected_instance_ids": selected_ids,
            }
        )
        selected_text = "、".join(f"PVS #{instance_id}" for instance_id in selected_ids)
        return (
            pvs_state,
            state,
            Image.fromarray(overlay, mode="RGB"),
            None,
            f"已选择 {selected_text}；黄色轮廓为模板源，每个源将生成独立衍生组",
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
    choices = []
    fallback = template_match_state.get("active_instance_id") if isinstance(template_match_state, dict) else None
    for group in _template_result_groups(result, fallback):
        group_id = str(group.get("group_id") or "")
        if not group_id:
            continue
        group_label = str(group.get("group_label") or group_id)
        instance_id = int(group.get("source_instance_id"))
        count = int(group.get("match_count") or len(group.get("matches") or []))
        choices.append(
            (f"{group_label} · PVS #{instance_id} · {count} 个衍生实例", group_id)
        )
    return gr.update(choices=choices, value=[value for _, value in choices])


def _export_template_match_selection_impl(
    _deps,
    source_state,
    image_state,
    pvs_state,
    template_match_state,
    selected_group_ids,
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
        groups = _template_result_groups(
            result,
            template_match_state.get("active_instance_id"),
        )
        if not groups:
            raise ValueError("没有可导出的模板匹配衍生组")
        available = {
            str(group.get("group_id")): group
            for group in groups
            if str(group.get("group_id") or "")
        }
        selected_ids = []
        for value in selected_group_ids or []:
            group_id = str(value)
            if group_id not in available:
                raise ValueError(f"衍生结果组 {group_id} 不存在")
            if group_id not in selected_ids:
                selected_ids.append(group_id)
        if not selected_ids:
            raise ValueError("请选择至少一个衍生结果组")

        source = _source_image_cache_get(source_state)
        crop_bbox = list(image_state.get("crop_bbox_xyxy") or [])
        if crop_bbox != list(source_state.get("crop_bbox_xyxy") or []):
            raise ValueError("当前裁剪 provenance 已过期，请重新应用裁剪")
        selected_groups = []
        group_masks = []
        flat_matches = []
        flat_masks = []
        seed_union = np.zeros((source.height, source.width), dtype=bool)
        for group_id in selected_ids:
            group = copy.deepcopy(available[group_id])
            source_instance_id = int(group.get("source_instance_id"))
            instance = _template_pvs_instance(pvs_state, source_instance_id)
            seed_mask = _template_matching.map_crop_mask_to_source(
                instance.get("mask_fullres_bool"),
                (source.height, source.width),
                crop_bbox,
            )
            seed_union |= seed_mask
            matches = list(group.get("matches") or [])
            masks = []
            for match in matches:
                translation = match.get("translation_xy")
                if not isinstance(translation, (list, tuple)) or len(translation) != 2:
                    raise ValueError(
                        f"衍生组 {group_id} 的 M{match.get('match_id')} 缺少合法位移"
                    )
                mask = _template_matching.translate_source_mask(
                    seed_mask,
                    int(translation[0]),
                    int(translation[1]),
                )
                masks.append(mask)
                flat_masks.append(mask)
                flat_matches.append(match)
            group["match_count"] = len(matches)
            selected_groups.append(group)
            group_masks.append(
                {
                    "group_id": group_id,
                    "source_instance_id": source_instance_id,
                    "seed_mask_fullres_bool": seed_mask,
                    "match_masks_fullres_bool": masks,
                }
            )
        parameters = result.get("parameters")
        export_nms_threshold = (
            parameters.get("nms_threshold", 0.3)
            if isinstance(parameters, dict)
            else 0.3
        )
        (
            selected_groups,
            group_masks,
            flat_matches,
            flat_masks,
        ) = _deduplicate_template_groups(
            selected_groups,
            group_masks,
            export_nms_threshold,
        )
        filtered_result = copy.deepcopy(result)
        filtered_result["schema_version"] = 2
        filtered_result.pop("seed", None)
        filtered_result.pop("blockers", None)
        filtered_result.pop("matches", None)
        filtered_result["groups"] = selected_groups
        filtered_result["group_count"] = len(selected_groups)
        filtered_result["match_count"] = len(flat_matches)
        filtered_result["selection"] = {
            "selected_group_ids": selected_ids,
        }
        source_rgb = np.asarray(source.convert("RGB"))
        workflow = {
            "result": filtered_result,
            "seed_mask_fullres_bool": seed_union,
            "match_masks_fullres_bool": flat_masks,
            "group_masks": group_masks,
            "overlay_rgb": _template_matching.render_template_overlay(
                source_rgb,
                seed_union,
                flat_masks,
                flat_matches,
            ),
        }
        zip_path, _ = _publish_template_match_export(
            source,
            workflow,
            source_state,
            image_state,
        )
        return (
            str(zip_path),
            f"已生成所选衍生组下载包：{len(selected_groups)} 组，"
            f"{len(flat_matches)} 个衍生实例",
        )
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
    groups = manifest.get("groups")
    if isinstance(groups, list):
        runtime_groups = {
            str(group.get("group_id")): group
            for group in workflow.get("group_masks") or []
            if isinstance(group, dict)
        }
        for group in groups:
            group_id = str(group.get("group_id") or "")
            runtime_group = runtime_groups.get(group_id)
            if runtime_group is None:
                raise ValueError(f"template group masks missing for {group_id}")
            source_instance_id = int(group.get("source_instance_id"))
            seed_name = f"pvs_{source_instance_id:04d}_seed.png"
            Image.fromarray(
                np.asarray(
                    runtime_group.get("seed_mask_fullres_bool"),
                    dtype=bool,
                ).astype(np.uint8) * 255,
                mode="L",
            ).save(masks_dir / seed_name)
            group.setdefault("seed", {})["mask_file"] = f"masks/{seed_name}"
            matches = list(group.get("matches") or [])
            match_masks = list(runtime_group.get("match_masks_fullres_bool") or [])
            if len(matches) != len(match_masks):
                raise ValueError(
                    f"template group {group_id} metadata and masks must align"
                )
            for index, (match, mask) in enumerate(zip(matches, match_masks), start=1):
                match_id = int(match.get("match_id", index))
                if match_id <= 0:
                    raise ValueError("template match id must be positive")
                mask_name = (
                    f"pvs_{source_instance_id:04d}_match_{match_id:04d}.png"
                )
                Image.fromarray(
                    np.asarray(mask, dtype=bool).astype(np.uint8) * 255,
                    mode="L",
                ).save(masks_dir / mask_name)
                match["mask_file"] = f"masks/{mask_name}"
    else:
        matches = list(manifest.get("matches") or [])
        match_masks = list(workflow["match_masks_fullres_bool"])
        if len(matches) != len(match_masks):
            raise ValueError("template match metadata and masks must align")
        for index, (match, mask) in enumerate(zip(matches, match_masks), start=1):
            match_id = int(match.get("match_id", index))
            if match_id <= 0:
                raise ValueError("template match id must be positive")
            mask_name = f"match_{match_id:04d}.png"
            Image.fromarray(
                np.asarray(mask, dtype=bool).astype(np.uint8) * 255,
                mode="L",
            ).save(masks_dir / mask_name)
            match["mask_file"] = f"masks/{mask_name}"
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
        seed_instance_ids = _template_selected_instance_ids(
            pvs_state,
            default_all=True,
        )
        if not seed_instance_ids:
            raise ValueError("请先完成智能分割并选择至少一个 PVS 模板实例")
        pvs_state["template_match_instance_ids"] = seed_instance_ids
        pvs_state.pop("template_match_instance_id", None)
        source_rgb = np.asarray(source.convert("RGB"))
        resolved_match_threshold = float(
            0.7 if match_threshold in (None, "") else match_threshold
        )
        resolved_expand_threshold = int(
            20 if expand_threshold in (None, "") else expand_threshold
        )
        resolved_nms_threshold = float(
            0.3 if nms_threshold in (None, "") else nms_threshold
        )
        groups = []
        group_masks = []
        flat_matches = []
        flat_masks = []
        seed_union = np.zeros((source.height, source.width), dtype=bool)
        for group_index, seed_instance_id in enumerate(seed_instance_ids):
            group_workflow = _template_matching.run_template_match_workflow(
                source_rgb,
                crop_bbox,
                pvs_state,
                active_instance_id=seed_instance_id,
                match_threshold=resolved_match_threshold,
                expand_threshold=resolved_expand_threshold,
                nms_threshold=resolved_nms_threshold,
            )
            group_result = group_workflow["result"]
            group_id = _template_group_id(seed_instance_id)
            group_label = _template_group_alias(group_index)
            matches = copy.deepcopy(group_result.get("matches") or [])
            for match in matches:
                match["group_id"] = group_id
                match["group_label"] = group_label
                match["source_instance_id"] = seed_instance_id
                match["display_id"] = (
                    f"{group_label[:-1]}{int(match.get('match_id'))}"
                )
            group = {
                "group_id": group_id,
                "group_label": group_label,
                "source_instance_id": seed_instance_id,
                "seed": copy.deepcopy(group_result.get("seed") or {}),
                "blockers": copy.deepcopy(group_result.get("blockers") or {}),
                "match_count": len(matches),
                "matches": matches,
            }
            seed_mask = np.asarray(
                group_workflow["seed_mask_fullres_bool"],
                dtype=bool,
            )
            match_masks = [
                np.asarray(mask, dtype=bool)
                for mask in group_workflow["match_masks_fullres_bool"]
            ]
            seed_union |= seed_mask
            groups.append(group)
            group_masks.append(
                {
                    "group_id": group_id,
                    "source_instance_id": seed_instance_id,
                    "seed_mask_fullres_bool": seed_mask,
                    "match_masks_fullres_bool": match_masks,
                }
            )
            flat_matches.extend(matches)
            flat_masks.extend(match_masks)
        (
            groups,
            group_masks,
            flat_matches,
            flat_masks,
        ) = _deduplicate_template_groups(
            groups,
            group_masks,
            resolved_nms_threshold,
        )
        result = {
            "schema_version": 2,
            "source_size_wh": [source.width, source.height],
            "crop_bbox_xyxy": crop_bbox,
            "parameters": {
                "match_threshold": resolved_match_threshold,
                "expand_threshold": resolved_expand_threshold,
                "nms_threshold": resolved_nms_threshold,
            },
            "group_count": len(groups),
            "match_count": len(flat_matches),
            "groups": groups,
        }
        workflow = {
            "result": result,
            "seed_mask_fullres_bool": seed_union,
            "match_masks_fullres_bool": flat_masks,
            "group_masks": group_masks,
            "overlay_rgb": _template_matching.render_template_overlay(
                source_rgb,
                seed_union,
                flat_masks,
                flat_matches,
            ),
        }
        zip_path, manifest = _publish_template_match_export(
            source,
            workflow,
            source_state,
            image_state,
        )
        count = int(manifest.get("match_count") or 0)
        state = {
            "session_id": session_id,
            "schema_version": 2,
            "source_image_id": source_id,
            "workspace_image_id": str(image_state.get("image_id") or ""),
            "selected_instance_ids": seed_instance_ids,
            "result": manifest,
        }
        selected_text = "、".join(
            f"PVS #{instance_id}" for instance_id in seed_instance_ids
        )
        status = (
            f"模板匹配完成：{selected_text}，"
            f"{len(groups)} 个衍生组，{count} 个衍生实例；"
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
