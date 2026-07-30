"""Layout mask prompt selection and PVS creation callbacks."""

from __future__ import annotations


def _layout_point_refine_impl(_deps, image_state, pcs_state, pvs_state, mode, point_payload, point_kind, prompt_state, progress):
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _point_from_payload = _deps['_point_from_payload']
    _pvs_progress = _deps['_pvs_progress']
    _refine_active_pvs_with_point = _deps['_refine_active_pvs_with_point']
    _view = _deps['_view']
    gr = _deps['gr']
    np = _deps['np']
    prompt_state = prompt_state or _new_prompt_state()
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("点提示修缮只能在 Layout Mask 模式使用")
        pending_point = prompt_state.get("last_point")
        if not isinstance(pending_point, (list, tuple)) or len(pending_point) != 2:
            raise ValueError("请先点击左侧原图记录一个待应用点")
        point = _point_from_payload(point_payload, image_state)
        pending_array = np.asarray(pending_point, dtype=np.float32)
        if not np.isfinite(pending_array).all() or not np.allclose(pending_array, point, rtol=0.0, atol=1e-3):
            raise ValueError("待应用点状态已过期，请重新点击左侧原图")

        active_id = pvs_state.get("active_instance_id")
        try:
            active_id = int(active_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("请先创建或选择一个版图 PVS instance") from exc
        active_inst = pvs_state.get("instances", {}).get(active_id)
        if active_inst is None or active_inst.get("status") == "deleted":
            raise ValueError("当前 active PVS instance 不存在或已删除")
        creation_history = active_inst.get("prompt_history") or []
        created_from_layout = any(
            isinstance(event, dict) and event.get("op") == "create_from_layout_mask"
            for event in creation_history
        )
        if active_inst.get("source") != "manual_pvs_layout_mask" or not created_from_layout:
            raise ValueError("点提示修缮仅支持由版图 mask 创建的 PVS instance")

        point_kind = str(point_kind or "")
        if point_kind not in {"positive", "negative"}:
            raise ValueError("点类型必须是正向点或负向点")
        point_label = 0 if point_kind == "negative" else 1
        point_name = "负向点" if point_label == 0 else "正向点"
        _pvs_progress(progress, 0.04, f"准备版图实例{point_name}修缮")
        refined_id, prompt_type = _refine_active_pvs_with_point(
            image_state,
            pvs_state,
            point,
            point_label,
            progress=progress,
        )
        prompt_state = dict(prompt_state)
        prompt_state["last_point"] = None
        point_payload = ""
        info = f"版图 PVS #{refined_id} 已应用 {prompt_type}；可继续点击下一修缮点"
        try:
            _pvs_progress(progress, 0.78, f"整理{point_name}候选 mask")
            _pvs_progress(progress, 0.96, "渲染版图点提示修缮结果", delay=0.16)
        except Exception as progress_exc:
            info += f"；结果已保存，但进度提示更新失败: {progress_exc}"
    except Exception as exc:
        info = f"版图点提示修缮失败: {exc}"
    try:
        view = _view(image_state, pcs_state, pvs_state, mode, info, prompt_state)
    except Exception as view_exc:
        status = f"{info}；界面刷新失败: {view_exc}"
        active = pvs_state.get("active_instance_id")
        view = (
            gr.update(),
            gr.update(),
            gr.update(value=status),
            gr.update(),
            gr.update(),
            gr.update(value=str(active) if active is not None else None),
            status,
            gr.update(),
        )
    return prompt_state, point_payload, pvs_state, *view


def _switch_mode_impl(_deps, mode, image_state, pcs_state, pvs_state):
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _is_pvs_manual_mode = _deps['_is_pvs_manual_mode']
    _is_pvs_pool_mode = _deps['_is_pvs_pool_mode']
    _new_prompt_state = _deps['_new_prompt_state']
    _pcs_bbox_choices = _deps['_pcs_bbox_choices']
    _pvs_pending_bbox_choices = _deps['_pvs_pending_bbox_choices']
    _view = _deps['_view']
    gr = _deps['gr']
    prompt_state = _new_prompt_state()
    is_pcs = _is_pcs_mode(mode)
    is_pvs = _is_pvs_manual_mode(mode)
    is_layout = _is_layout_mask_mode(mode)
    if is_pcs:
        tool_update = gr.update(choices=[("框提示 (Box)", "bbox")], value="bbox")
        finish_update = gr.update(visible=False)
    elif is_pvs:
        tool_update = gr.update(
            choices=[("点提示 (Point)", "point"), ("框提示 (Box)", "bbox"), ("多边形Mask (Polygon)", "polygon")],
            value="bbox",
        )
        finish_update = gr.update(visible=True)
    else:
        tool_update = gr.update(choices=[("版图 mask 提示", "layout")], value="layout")
        finish_update = gr.update(visible=False)
    return (
        prompt_state,
        "",
        "",
        "",
        tool_update,
        finish_update,
        gr.update(visible=is_pcs),
        gr.update(visible=is_pcs),
        gr.update(visible=is_pvs),
        gr.update(visible=_is_pvs_pool_mode(mode)),
        gr.update(visible=not is_layout),
        gr.update(visible=is_layout),
        gr.update(visible=is_layout),
        gr.update(visible=is_pvs),
        gr.update(visible=False),
        gr.update(visible=False),
        _pcs_bbox_choices(pcs_state),
        _pvs_pending_bbox_choices(pvs_state),
        gr.update(visible=is_layout),
        *_view(image_state, pcs_state, pvs_state, mode, f"Mode: {mode}，交互提示已重置", prompt_state),
    )


def _switch_mode_with_layout_editor_impl(_deps, mode, image_state, pcs_state, pvs_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _switch_mode = _deps['_switch_mode']
    gr = _deps['gr']
    _advance_layout_prompt_epoch(image_state, layout_state)
    result = _switch_mode(mode, image_state, pcs_state, pvs_state)
    if _is_layout_mask_mode(mode):
        editor = _layout_editor_payload(image_state, layout_state, "已切换到版图 mask 提示分割，Canvas payload 已刷新。")
    else:
        editor = gr.update()
    return (*result, editor)


def _layout_prompt_epoch_key_impl(_deps, image_state, layout_state, session_state):
    for state in (layout_state, image_state, session_state):
        if isinstance(state, dict) and state.get("session_id"):
            return str(state["session_id"])
    return "default"


def _layout_prompt_epoch_snapshot_impl(_deps, image_state, layout_state, session_state):
    _LAYOUT_PROMPT_EPOCHS = _deps['_LAYOUT_PROMPT_EPOCHS']
    _LAYOUT_PROMPT_EPOCH_LOCK = _deps['_LAYOUT_PROMPT_EPOCH_LOCK']
    _layout_prompt_epoch_key = _deps['_layout_prompt_epoch_key']
    key = _layout_prompt_epoch_key(
        image_state,
        layout_state,
        session_state,
    )
    with _LAYOUT_PROMPT_EPOCH_LOCK:
        return key, int(_LAYOUT_PROMPT_EPOCHS.get(key, 0))


def _advance_layout_prompt_epoch_impl(_deps, image_state, layout_state, session_state):
    _LAYOUT_PROMPT_EPOCHS = _deps['_LAYOUT_PROMPT_EPOCHS']
    _LAYOUT_PROMPT_EPOCH_LOCK = _deps['_LAYOUT_PROMPT_EPOCH_LOCK']
    _layout_prompt_epoch_key = _deps['_layout_prompt_epoch_key']
    key = _layout_prompt_epoch_key(
        image_state,
        layout_state,
        session_state,
    )
    with _LAYOUT_PROMPT_EPOCH_LOCK:
        value = int(_LAYOUT_PROMPT_EPOCHS.get(key, 0)) + 1
        _LAYOUT_PROMPT_EPOCHS[key] = value
        return value


def _reset_layout_prompt_selection_state_impl(_deps, layout_state):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    state = dict(layout_state or {})
    state.update(
        {
            "prompt_mask_scope": _LAYOUT_PROMPT_SCOPE_FULL,
            "prompt_class_label": None,
            "prompt_labels": [],
            "prompt_group_transforms": {},
            "prompt_active_group_id": None,
            "prompt_selection_signature": None,
            "prompt_transform_set_revision": 0,
            "prompt_regions_revision": None,
            "prompt_region_ids": [],
        }
    )
    return state


def _layout_prompt_selection_token_impl(_deps, scope, region_id):
    _LAYOUT_PROMPT_LABEL_PREFIX = _deps['_LAYOUT_PROMPT_LABEL_PREFIX']
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_CLASS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_CLASS']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    if scope == _LAYOUT_PROMPT_SCOPE_FULL:
        return _LAYOUT_PROMPT_SCOPE_FULL
    if scope in {
        _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
        _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
    }:
        try:
            value = int(region_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("版图 Label ID 无效") from exc
        if value <= 0:
            raise ValueError("版图 Label ID 无效")
        return f"{_LAYOUT_PROMPT_LABEL_PREFIX}{value}"
    raise ValueError("版图 mask 选择无效")


def _parse_layout_prompt_selection_impl(_deps, value):
    _LAYOUT_PROMPT_LABEL_PREFIX = _deps['_LAYOUT_PROMPT_LABEL_PREFIX']
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item or "") for item in value]
    else:
        raise ValueError("版图 mask 选择无效")
    values = list(dict.fromkeys(values))
    if values == [_LAYOUT_PROMPT_SCOPE_FULL]:
        return _LAYOUT_PROMPT_SCOPE_FULL, []
    if not values or _LAYOUT_PROMPT_SCOPE_FULL in values:
        raise ValueError("全部版图 mask 与 label 不能同时选择")
    region_ids = []
    for token in values:
        if not token.startswith(_LAYOUT_PROMPT_LABEL_PREFIX):
            raise ValueError("版图 mask 选择无效")
        try:
            region_id = int(token[len(_LAYOUT_PROMPT_LABEL_PREFIX):])
        except (TypeError, ValueError) as exc:
            raise ValueError("版图 Label ID 无效") from exc
        if region_id <= 0 or region_id in region_ids:
            raise ValueError("版图 Label ID 重复或无效")
        region_ids.append(region_id)
    return _LAYOUT_PROMPT_SCOPE_REGION_LABELS, region_ids


def _normalize_layout_prompt_checkbox_selection_impl(_deps, value, layout_state):
    _LAYOUT_PROMPT_LABEL_PREFIX = _deps['_LAYOUT_PROMPT_LABEL_PREFIX']
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    if isinstance(value, str):
        incoming = [value]
    elif isinstance(value, (list, tuple)):
        incoming = list(dict.fromkeys(str(item or "") for item in value))
    else:
        incoming = []
    label_tokens = [
        token
        for token in incoming
        if token.startswith(_LAYOUT_PROMPT_LABEL_PREFIX)
    ]
    if not incoming:
        return [_LAYOUT_PROMPT_SCOPE_FULL]
    if _LAYOUT_PROMPT_SCOPE_FULL in incoming and label_tokens:
        if (layout_state or {}).get("prompt_mask_scope") == _LAYOUT_PROMPT_SCOPE_FULL:
            return label_tokens
        return [_LAYOUT_PROMPT_SCOPE_FULL]
    return incoming


def _layout_prompt_label_counts_impl(_deps, document):
    _layout_regions = _deps['_layout_regions']
    counts = {}
    for record in sorted(
        _layout_regions.active_regions(document),
        key=lambda item: int(item["region_id"]),
    ):
        label = _layout_regions.region_label(record)
        counts[label] = counts.get(label, 0) + 1
    return list(counts.items())


def _layout_label_choice_text_impl(_deps, record, label_counts):
    _layout_regions = _deps['_layout_regions']
    label = _layout_regions.region_label(record)
    if int(label_counts.get(label) or 0) > 1:
        return f"{label} (R{int(record['region_id'])})"
    return label


def _layout_prompt_class_counts_impl(_deps, document):
    _layout_prompt_label_counts = _deps['_layout_prompt_label_counts']
    return _layout_prompt_label_counts(document)


def _layout_prompt_choice_update_impl(_deps, document, selected_value):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _layout_label_choice_text = _deps['_layout_label_choice_text']
    _layout_prompt_label_counts = _deps['_layout_prompt_label_counts']
    _layout_prompt_selection_token = _deps['_layout_prompt_selection_token']
    _layout_regions = _deps['_layout_regions']
    gr = _deps['gr']
    choices = [("全部版图 mask", _LAYOUT_PROMPT_SCOPE_FULL)]
    label_counts = dict(_layout_prompt_label_counts(document or {}))
    for record in sorted(
        _layout_regions.active_regions(document or {}),
        key=lambda item: int(item["region_id"]),
    ):
        choices.append(
            (
                _layout_label_choice_text(record, label_counts),
                _layout_prompt_selection_token(
                    _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                    int(record["region_id"]),
                ),
            )
        )
    values = {value for _, value in choices}
    if isinstance(selected_value, str):
        requested = [selected_value]
    elif isinstance(selected_value, (list, tuple)):
        requested = list(dict.fromkeys(selected_value))
    else:
        requested = [_LAYOUT_PROMPT_SCOPE_FULL]
    selected = [value for value in requested if value in values]
    if not selected:
        selected = [_LAYOUT_PROMPT_SCOPE_FULL]
    return gr.update(
        choices=choices,
        value=selected,
        interactive=len(choices) > 1,
    )


def _load_layout_prompt_region_document_impl(_deps, layout_state, expected_revision):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_regions = _deps['_layout_regions']
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    document, source_mask = _LAYOUT_REGION_STORE.load_document(
        session_id,
        layout_id,
        source_mask_hash,
    )
    if expected_revision is not None:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision != int(document.get("regions_revision") or 0)
        ):
            raise _layout_regions.StaleRegionsRevisionError(
                "版图 Region 已变化；请重新加载 Label 后再创建 PVS 实例"
            )
    return document, source_mask


def _layout_prompt_label_records_impl(_deps, document, labels):
    _layout_regions = _deps['_layout_regions']
    records = _layout_regions.active_regions_for_labels(document, labels)
    if not records:
        raise _layout_regions.RegionValidationError(
            "所选 label 没有活动 Region"
        )
    return records


def _layout_prompt_class_records_impl(_deps, document, class_label):
    _layout_prompt_label_records = _deps['_layout_prompt_label_records']
    return _layout_prompt_label_records(document, [class_label])


def _layout_prompt_region_records_impl(_deps, document, region_ids):
    _layout_regions = _deps['_layout_regions']
    if not isinstance(region_ids, (list, tuple)) or not region_ids:
        raise _layout_regions.RegionValidationError("请至少选择一个 Label")
    normalized = []
    for value in region_ids:
        try:
            region_id = int(value)
        except (TypeError, ValueError) as exc:
            raise _layout_regions.RegionValidationError("版图 Label ID 无效") from exc
        if region_id <= 0 or region_id in normalized:
            raise _layout_regions.RegionValidationError("版图 Label ID 重复或无效")
        normalized.append(region_id)
    active_by_id = {
        int(record["region_id"]): record
        for record in _layout_regions.active_regions(document)
    }
    missing = [region_id for region_id in normalized if region_id not in active_by_id]
    if missing:
        raise _layout_regions.RegionValidationError(
            f"Label 对应的活动 Region R{missing[0]} 不存在"
        )
    return [active_by_id[region_id] for region_id in sorted(normalized)]


def _layout_prompt_display_mask_impl(_deps, layout_state, source_mask):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_CLASS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_CLASS']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _layout_prompt_region_records = _deps['_layout_prompt_region_records']
    _layout_regions = _deps['_layout_regions']
    _load_layout_prompt_region_document = _deps['_load_layout_prompt_region_document']
    np = _deps['np']
    source = np.asarray(source_mask, dtype=bool)
    scope = str(
        (layout_state or {}).get("prompt_mask_scope")
        or _LAYOUT_PROMPT_SCOPE_FULL
    )
    if scope == _LAYOUT_PROMPT_SCOPE_FULL:
        return source, None
    if scope not in {
        _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
        _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
    }:
        raise _layout_regions.RegionValidationError("未知的版图 prompt mask scope")
    expected_revision = (layout_state or {}).get("prompt_regions_revision")
    document, stored_source = _load_layout_prompt_region_document(
        layout_state,
        expected_revision=expected_revision,
    )
    if stored_source.shape != source.shape:
        raise _layout_regions.RegionValidationError(
            "Region source mask 尺寸与当前版图不一致"
        )
    records = _layout_prompt_region_records(
        document,
        (layout_state or {}).get("prompt_region_ids"),
    )
    region_ids = [int(record["region_id"]) for record in records]
    preview = np.zeros_like(source, dtype=bool)
    for _, mask in _layout_regions.decode_region_masks(records, source.shape):
        preview = np.logical_or(preview, mask)
    if not preview.any():
        raise _layout_regions.RegionValidationError(
            "所选 label 的 Region mask 为空"
        )
    return preview, (
        f"Canvas 正在预览 {len(region_ids)} 个 Label；"
        "每个 Label 可独立变换，并各自创建一个 PVS 实例。"
    )


def _layout_prompt_group_id_impl(_deps, region_id):
    return f"region_{int(region_id)}"


def _layout_prompt_group_data_impl(_deps, layout_state, source_mask):
    _layout_prompt_region_records = _deps['_layout_prompt_region_records']
    _layout_regions = _deps['_layout_regions']
    _load_layout_prompt_region_document = _deps['_load_layout_prompt_region_document']
    hashlib = _deps['hashlib']
    json = _deps['json']
    np = _deps['np']
    expected_revision = (layout_state or {}).get("prompt_regions_revision")
    document, stored_source = _load_layout_prompt_region_document(
        layout_state,
        expected_revision=expected_revision,
    )
    source = np.asarray(source_mask, dtype=bool)
    if stored_source.shape != source.shape:
        raise _layout_regions.RegionValidationError(
            "Region source mask 尺寸与当前版图不一致"
        )
    records = _layout_prompt_region_records(
        document,
        (layout_state or {}).get("prompt_region_ids"),
    )
    decoded = _layout_regions.decode_region_masks(records, source.shape)
    signature_payload = {
        "session_id": str((layout_state or {}).get("session_id") or ""),
        "layout_id": str((layout_state or {}).get("layout_id") or ""),
        "source_mask_hash": str(
            (layout_state or {}).get("source_mask_pixel_sha256") or ""
        ),
        "target_image_sha256": str(
            (layout_state or {}).get("target_image_sha256") or ""
        ),
        "regions_revision": int(document.get("regions_revision") or 0),
        "regions": [
            {
                "region_id": int(record["region_id"]),
                "label": _layout_regions.region_label(record),
                "mask_hash": _layout_regions.mask_pixel_sha256(
                    mask.astype(np.uint8)
                ),
            }
            for record, mask in decoded
        ],
    }
    encoded = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return document, decoded, hashlib.sha256(encoded).hexdigest()


def _layout_group_transform_impl(_deps, layout_state, base_transform, record, group_mask, target_size, values):
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_prompt_group_id = _deps['_layout_prompt_group_id']
    _layout_regions = _deps['_layout_regions']
    _layout_tx = _deps['_layout_tx']
    np = _deps['np']
    mask = np.asarray(group_mask, dtype=bool)
    region_id = int(record["region_id"])
    group_id = _layout_prompt_group_id(region_id)
    group_hash = _layout_regions.mask_pixel_sha256(mask.astype(np.uint8))
    bbox = _layout_tx.foreground_bbox_xyxy(mask)
    pivot = _layout_tx.pivot_from_bbox_xyxy(bbox)
    supplied = values if isinstance(values, dict) else None
    if supplied is None:
        base_matrix = _layout_tx.build_layout_affine_matrix(base_transform)
        center_x, center_y = _layout_tx.apply_affine_to_point(
            pivot,
            base_matrix,
        )
        scale = float(base_transform.get("scale") or 1.0)
        rotation = float(base_transform.get("rotation_deg") or 0.0)
        alpha = _layout_preview_alpha(base_transform)
        revision = int(base_transform.get("revision") or 0)
    else:
        for field, expected in {
            "session_id": layout_state.get("session_id"),
            "layout_id": layout_state.get("layout_id"),
            "image_id": base_transform.get("image_id"),
            "source_mask_pixel_sha256": layout_state.get(
                "source_mask_pixel_sha256"
            ),
            "target_image_sha256": base_transform.get(
                "target_image_sha256"
            ),
            "group_id": group_id,
            "group_mask_pixel_sha256": group_hash,
        }.items():
            actual = supplied.get(field)
            if actual is not None and str(actual) != str(expected):
                raise ValueError(f"Label transform {field} 不匹配")
        numeric = [
            supplied.get("center_x"),
            supplied.get("center_y"),
            supplied.get("scale"),
            supplied.get("rotation_deg", 0.0),
            supplied.get("preview_alpha", 0.35),
        ]
        try:
            numeric = [float(value) for value in numeric]
        except (TypeError, ValueError) as exc:
            raise ValueError("Label transform 数值无效") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("Label transform 包含非有限数值")
        center_x, center_y, scale, rotation, alpha = numeric
        scale = float(np.clip(scale, 0.01, 20.0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        revision_value = supplied.get("revision", 0)
        if (
            isinstance(revision_value, bool)
            or not isinstance(revision_value, (int, float))
            or int(revision_value) < 0
        ):
            raise ValueError("Label transform revision 无效")
        revision = int(revision_value)
    transform = _layout_tx.make_layout_transform_v2(
        session_id=str(layout_state.get("session_id") or ""),
        layout_id=str(layout_state.get("layout_id") or ""),
        image_id=str(base_transform.get("image_id") or ""),
        target_size=target_size,
        source_mask=mask,
        center_x=center_x,
        center_y=center_y,
        pivot_xy=pivot,
        scale=scale,
        rotation_deg=rotation,
        preview_alpha=alpha,
        revision=revision,
        source_mask_pixel_sha256=layout_state.get(
            "source_mask_pixel_sha256"
        ),
        target_image_sha256=base_transform.get("target_image_sha256"),
    )
    transform = _layout_tx.transform_with_derived_fields(
        transform,
        target_size,
    )
    transform.update(
        {
            "group_id": group_id,
            "region_id": region_id,
            "label": _layout_regions.region_label(record),
            "group_mask_pixel_sha256": group_hash,
        }
    )
    return transform, bbox


def _layout_prompt_group_payload_impl(_deps, layout_state, source_mask, base_transform, target_size):
    _data_url = _deps['_data_url']
    _layout_group_transform = _deps['_layout_group_transform']
    _layout_mask_to_editor_image = _deps['_layout_mask_to_editor_image']
    _layout_prompt_group_data = _deps['_layout_prompt_group_data']
    _layout_prompt_group_id = _deps['_layout_prompt_group_id']
    _layout_regions = _deps['_layout_regions']
    copy = _deps['copy']
    document, decoded, signature = _layout_prompt_group_data(
        layout_state,
        source_mask,
    )
    previous = (layout_state or {}).get("prompt_group_transforms")
    if not isinstance(previous, dict):
        previous = {}
    groups = []
    transforms = []
    for record, mask in decoded:
        group_id = _layout_prompt_group_id(record["region_id"])
        try:
            values = previous.get(group_id)
            transform, bbox = _layout_group_transform(
                layout_state,
                base_transform,
                record,
                mask,
                target_size,
                values=values,
            )
        except Exception:
            transform, bbox = _layout_group_transform(
                layout_state,
                base_transform,
                record,
                mask,
                target_size,
            )
        groups.append(
            {
                "group_id": group_id,
                "label": _layout_regions.region_label(record),
                "region_ids": [int(record["region_id"])],
                "mask_image": _data_url(_layout_mask_to_editor_image(mask)),
                "foreground_bbox_xyxy": [float(value) for value in bbox],
                "group_mask_pixel_sha256": transform[
                    "group_mask_pixel_sha256"
                ],
            }
        )
        transforms.append(
            {
                "group_id": group_id,
                "transform": copy.deepcopy(transform),
            }
        )
    group_ids = [group["group_id"] for group in groups]
    active_group_id = (layout_state or {}).get("prompt_active_group_id")
    if active_group_id not in group_ids:
        active_group_id = group_ids[0]
    return {
        "transform_mode": "label_groups",
        "group_view": {
            "selection_signature": signature,
            "regions_revision": int(document.get("regions_revision") or 0),
            "groups": groups,
        },
        "group_intent": {
            "selection_signature": signature,
            "transform_set_revision": int(
                (layout_state or {}).get("prompt_transform_set_revision")
                or 0
            ),
            "active_group_id": active_group_id,
            "transforms": transforms,
        },
    }


def _mask_to_lowres_logits_impl(_deps, mask):
    _prompt_mask_size = _deps['_prompt_mask_size']
    cv2 = _deps['cv2']
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("layout mask must be a 2D binary mask")
    target_h, target_w = _prompt_mask_size()
    lowres = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return ((lowres * 2.0 - 1.0) * 10.0).astype(np.float32)


def _validate_layout_prompt_mask_impl(_deps, mask):
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("layout transformed mask must be 2D")
    foreground = int(mask.sum())
    total = int(mask.size)
    if foreground == 0:
        raise ValueError("layout transformed mask is empty")
    if foreground < 8:
        raise ValueError("layout transformed mask is too small")
    if foreground >= int(total * 0.98):
        raise ValueError("layout transformed mask is almost all foreground; check invert or transform")
    return mask


def _layout_transformed_mask_for_image_impl(_deps, image_state, layout_state):
    _commit_layout_transform = _deps['_commit_layout_transform']
    _layout_cache_get = _deps['_layout_cache_get']
    _validate_layout_prompt_mask = _deps['_validate_layout_prompt_mask']
    _workspace = _deps['_workspace']
    np = _deps['np']
    ws = _workspace(image_state)
    image = ws["image"]
    target_shape = (int(image.height), int(image.width))
    cached = _layout_cache_get(layout_state)
    transformed = cached.get("transformed_mask")
    if transformed is None or np.asarray(transformed).shape != target_shape:
        state, transformed, _ = _commit_layout_transform(
            image_state,
            layout_state,
            layout_state.get("enabled", True),
            layout_state.get("tx", 0.0),
            layout_state.get("ty", 0.0),
            layout_state.get("scale", 1.0),
            layout_state.get("rotation_deg", 0.0),
            layout_state.get("preview_alpha", 0.35),
        )
        layout_state.update(state)
    transformed = np.asarray(transformed, dtype=bool)
    if transformed.shape != target_shape:
        raise ValueError(f"layout transformed mask shape {transformed.shape} does not match target {target_shape}")
    return _validate_layout_prompt_mask(transformed)


def _layout_prompt_metadata_impl(_deps, image_state, layout_state):
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    copy = _deps['copy']
    cached = _layout_cache_get(layout_state)
    transform = copy.deepcopy(cached.get("backend_transform") or layout_state)
    return {
        "type": "layout_mask",
        "session_id": layout_state.get("session_id"),
        "layout_id": layout_state.get("layout_id"),
        "region_mode": layout_state.get("region_mode"),
        "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        "target_image_sha256": image_state.get("target_image_sha256") if isinstance(image_state, dict) else cached.get("target_image_sha256"),
        "source_width": int(cached.get("source_width") or 0),
        "source_height": int(cached.get("source_height") or 0),
        "target_width": int(image_state.get("width") or 0) if isinstance(image_state, dict) else None,
        "target_height": int(image_state.get("height") or 0) if isinstance(image_state, dict) else None,
        "transform": transform,
        "matrix_2x3": copy.deepcopy(cached.get("matrix_2x3")),
        "revision": int(cached.get("committed_revision") or transform.get("revision") or 0),
        "preview_alpha": _layout_preview_alpha(layout_state),
        "binarize_params": copy.deepcopy(cached.get("binarize_params", {})),
    }


def _create_pvs_from_layout_mask_impl(_deps, image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload, progress):
    _best = _deps['_best']
    _commit_layout_transform = _deps['_commit_layout_transform']
    _fresh_state = _deps['_fresh_state']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_prompt_metadata = _deps['_layout_prompt_metadata']
    _make_inst = _deps['_make_inst']
    _mask_box = _deps['_mask_box']
    _mask_to_lowres_logits = _deps['_mask_to_lowres_logits']
    _new_layout_state = _deps['_new_layout_state']
    _predict_inst = _deps['_predict_inst']
    _pvs_progress = _deps['_pvs_progress']
    _validate_layout_prompt_mask = _deps['_validate_layout_prompt_mask']
    _view = _deps['_view']
    copy = _deps['copy']
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 mask prompt 只支持在版图 mask 提示分割模式使用")
        _pvs_progress(progress, 0.05, "Commit and validate layout transform")
        state, transformed, _ = _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=editor_payload)
        transformed = _validate_layout_prompt_mask(transformed)
        lowres_logits = _mask_to_lowres_logits(transformed)
        _pvs_progress(progress, 0.35, "SAM3 is creating PVS instance from layout mask_input", delay=0.08)
        pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=lowres_logits)
        idx = _best(pred)
        mask = pred["masks"][idx]
        inst_id = int(pvs_state.get("next_instance_id", 1))
        prompt = _layout_prompt_metadata(image_state, state)
        pvs_state.setdefault("instances", {})[inst_id] = _make_inst(
            inst_id,
            "manual_pvs_layout_mask",
            mask,
            _mask_box(mask),
            pred["scores"][idx],
            pvs_logits=pred["lowres_logits"][idx],
            history=[{"op": "create_from_layout_mask", "prompt": copy.deepcopy(prompt), "candidate_scores": pred["scores"].astype(float).tolist()}],
        )
        pvs_state["active_instance_id"] = inst_id
        pvs_state["next_instance_id"] = inst_id + 1
        _pvs_progress(progress, 0.96, "Render PVS layout result", delay=0.12)
        info = f"已用版图 mask prompt 创建 PVS #{inst_id}"
    except Exception as exc:
        state = layout_state or _new_layout_state(image_state.get("session_id") if isinstance(image_state, dict) else None)
        info = f"用版图 mask 创建 PVS 实例失败：{exc}"
    editor = _layout_editor_payload(image_state, state, info)
    return pvs_state, state, editor, info, *_view(image_state, pcs_state, pvs_state, mode, info, layout_state=state)


def _layout_state_summary_impl(_deps, layout_state):
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    if not layout_state or not layout_state.get("layout_id"):
        return "No layout mask selected"
    return (
        f"layout: {layout_state.get('layout_id')}\n"
        f"session: {layout_state.get('session_id')}\n"
        f"enabled: {bool(layout_state.get('enabled'))}\n"
        f"source: {int(layout_state.get('source_width') or 0)}x{int(layout_state.get('source_height') or 0)}\n"
        f"revision={int(layout_state.get('revision') or 0)}, tx={float(layout_state.get('tx') or 0):.1f}, ty={float(layout_state.get('ty') or 0):.1f}, "
        f"scale={float(layout_state.get('scale') or 1):.3f}, rotation={float(layout_state.get('rotation_deg') or 0):.1f}, "
        f"alpha={_layout_preview_alpha(layout_state):.2f}\n"
        f"source_mask_pixel_sha256: {layout_state.get('source_mask_pixel_sha256') or ''}\n"
        f"target_image_sha256: {layout_state.get('target_image_sha256') or ''}"
    )


def _load_layout_binary_mask_png_impl(_deps, session_state, image_state, input_image, region_mode):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _filter_layout_components = _deps['_filter_layout_components']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_mask_contours = _deps['_layout_mask_contours']
    _layout_state_summary = _deps['_layout_state_summary']
    _new_layout_state = _deps['_new_layout_state']
    _pil_image = _deps['_pil_image']
    _save_layout_mask_files = _deps['_save_layout_mask_files']
    _session_id_from_state = _deps['_session_id_from_state']
    cv2 = _deps['cv2']
    np = _deps['np']
    _advance_layout_prompt_epoch(
        image_state=image_state,
        session_state=session_state,
    )
    try:
        image = _pil_image(input_image)
        if image is None:
            raise ValueError("Upload a binary mask PNG first")
        gray = cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        white_fg = gray >= 128
        black_fg = gray < 128
        candidates = [mask for mask in (white_fg, black_fg) if mask.any()]
        if not candidates:
            raise ValueError("Uploaded binary mask has no foreground pixels")
        mask = min(candidates, key=lambda arr: float(arr.mean()))
        mask = _filter_layout_components(mask, 0, region_mode)
        if not mask.any():
            raise ValueError("Binary mask is empty after filtering")
        contours = _layout_mask_contours(mask)
        params = {"source": "uploaded_binary_mask_png", "region_mode": str(region_mode or "all"), "foreground_rule": "auto_smaller_nonzero"}
        state, _, _, _ = _save_layout_mask_files(session_state, image, mask, contours, params)
        info = f"二值 mask PNG 已载入。\n{_layout_state_summary(state)}"
        return state, _layout_editor_payload(image_state, state, "二值 mask PNG 已载入版图编辑器。"), info
    except Exception as exc:
        state = _new_layout_state(_session_id_from_state(session_state))
        return state, _layout_editor_empty(image_state, f"载入二值 mask PNG 失败：{exc}"), f"载入二值 mask PNG 失败：{exc}"


def _use_current_layout_mask_impl(_deps, image_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_state_summary = _deps['_layout_state_summary']
    _new_layout_state = _deps['_new_layout_state']
    _advance_layout_prompt_epoch(image_state, layout_state)
    try:
        _layout_cache_get(layout_state)
        info = "Using current saved layout mask.\n" + _layout_state_summary(layout_state)
        return layout_state, _layout_editor_payload(image_state, layout_state, "当前已保存版图 mask 已载入 Canvas。"), info
    except Exception as exc:
        return layout_state or _new_layout_state(), _layout_editor_empty(image_state, f"当前版图 mask 不可用：{exc}"), f"当前版图 mask 不可用：{exc}"


def _pvs_creation_commit_token_impl(_deps, pvs_state):
    instances = pvs_state.get("instances")
    if not isinstance(instances, dict):
        raise ValueError("PVS instance state 无效")
    instance_tokens = []
    for instance_id in sorted(instances, key=int):
        instance = instances[instance_id]
        instance_tokens.append(
            (
                int(instance_id),
                id(instance),
                instance.get("status"),
                id(instance.get("mask_fullres_bool")),
                id(instance.get("pvs_lowres_logits")),
                len(instance.get("prompt_history") or []),
            )
        )
    pending_records = pvs_state.get("pending_bbox_records")
    pending_boxes = pvs_state.get("pending_boxes")
    return (
        id(instances),
        tuple(instance_tokens),
        pvs_state.get("active_instance_id"),
        int(pvs_state.get("next_instance_id", 1)),
        id(pending_records),
        len(pending_records or []),
        id(pending_boxes),
        len(pending_boxes or []),
        int(pvs_state.get("next_pending_bbox_id", 1)),
    )


def _selected_pvs_candidate_impl(_deps, prediction, image_shape):
    _prompt_mask_size = _deps['_prompt_mask_size']
    np = _deps['np']
    scores = np.asarray(prediction.get("scores"), dtype=np.float32).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("predict_inst 返回的候选分数无效")
    index = int(np.argmax(scores))

    masks = np.asarray(prediction.get("masks"))
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] != scores.size:
        raise ValueError("predict_inst 返回的候选 mask 数量或形状无效")
    mask = np.asarray(masks[index])
    if mask.shape != tuple(image_shape):
        raise ValueError("predict_inst 返回的候选 mask 尺寸与当前图像不一致")
    if np.issubdtype(mask.dtype, np.number) and not np.isfinite(mask).all():
        raise ValueError("predict_inst 返回的候选 mask 包含非有限值")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("predict_inst 返回的最佳候选 mask 为空")

    logits = np.asarray(prediction.get("lowres_logits"), dtype=np.float32)
    if logits.ndim == 2:
        if scores.size != 1:
            raise ValueError("predict_inst 返回的 low-res logits 缺少候选维度")
        selected_logits = logits
    elif logits.ndim in (3, 4) and logits.shape[0] == scores.size:
        selected_logits = logits[index]
    else:
        raise ValueError("predict_inst 返回的 low-res logits 数量或形状无效")
    expected = _prompt_mask_size()
    valid_shape = (
        selected_logits.ndim == 2
        or (selected_logits.ndim == 3 and selected_logits.shape[0] == 1)
    ) and tuple(selected_logits.shape[-2:]) == expected
    if not valid_shape or not np.isfinite(selected_logits).all():
        raise ValueError("predict_inst 返回的 low-res logits 无效")
    return (
        mask,
        float(scores[index]),
        selected_logits.copy(),
        scores.astype(float).tolist(),
    )


def _layout_prompt_region_fingerprint_impl(_deps, decoded_records):
    _layout_regions = _deps['_layout_regions']
    np = _deps['np']
    return tuple(
        (
            int(record["region_id"]),
            _layout_regions.region_label(record),
            _layout_regions.mask_pixel_sha256(mask.astype(np.uint8)),
        )
        for record, mask in decoded_records
    )


def _validate_layout_transform_snapshot_impl(_deps, layout_state, transform):
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_get = _deps['_layout_cache_get']
    copy = _deps['copy']
    np = _deps['np']
    with _LAYOUT_CACHE_LOCK:
        cached = _layout_cache_get(layout_state)
        committed_revision = int(cached.get("committed_revision") or 0)
        source_hash = cached.get("source_mask_pixel_sha256")
        target_hash = cached.get("target_image_sha256")
        matrix = copy.deepcopy(cached.get("matrix_2x3"))
    expected_matrix = transform.get("matrix_2x3")
    if committed_revision != int(transform.get("revision") or 0):
        raise ValueError("版图 transform 在批量预测期间发生变化")
    if source_hash != transform.get("source_mask_pixel_sha256"):
        raise ValueError("版图 source mask 在批量预测期间发生变化")
    if target_hash != transform.get("target_image_sha256"):
        raise ValueError("目标图像在批量预测期间发生变化")
    if matrix is None or expected_matrix is None or not np.allclose(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(expected_matrix, dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("版图 affine matrix 在批量预测期间发生变化")


def _load_layout_prompt_choices_impl(_deps, image_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_prompt_choice_update = _deps['_layout_prompt_choice_update']
    _layout_regions = _deps['_layout_regions']
    _load_layout_prompt_region_document = _deps['_load_layout_prompt_region_document']
    _reset_layout_prompt_selection_state = _deps['_reset_layout_prompt_selection_state']
    _advance_layout_prompt_epoch(image_state, layout_state)
    state = _reset_layout_prompt_selection_state(layout_state)
    try:
        _layout_cache_get(state)
    except Exception as exc:
        info = f"当前版图 mask 不可用：{exc}"
        return (
            state,
            _layout_editor_empty(image_state, info),
            _layout_prompt_choice_update(),
            info,
        )
    try:
        document, _ = _load_layout_prompt_region_document(state)
        label_count = len(_layout_regions.active_regions(document))
        info = (
            f"当前已保存版图 mask 已加载；可选择完整 mask 或 "
            f"{label_count} 个独立 Label。"
        )
        return (
            state,
            _layout_editor_payload(image_state, state, info),
            _layout_prompt_choice_update(document),
            info,
        )
    except Exception as exc:
        info = (
            "当前已保存版图 mask 已加载；Region Label 不可用，"
            f"仍可使用完整 mask：{exc}"
        )
        return (
            state,
            _layout_editor_payload(image_state, state, info),
            _layout_prompt_choice_update(),
            info,
        )


def _select_layout_prompt_mask_impl(_deps, image_state, layout_state, selection):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _commit_layout_group_transforms = _deps['_commit_layout_group_transforms']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_group_control_values = _deps['_layout_group_control_values']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_prompt_region_records = _deps['_layout_prompt_region_records']
    _layout_prompt_selection_token = _deps['_layout_prompt_selection_token']
    _layout_regions = _deps['_layout_regions']
    _load_layout_prompt_region_document = _deps['_load_layout_prompt_region_document']
    _normalize_layout_prompt_checkbox_selection = _deps['_normalize_layout_prompt_checkbox_selection']
    _parse_layout_prompt_selection = _deps['_parse_layout_prompt_selection']
    _reset_layout_prompt_selection_state = _deps['_reset_layout_prompt_selection_state']
    gr = _deps['gr']
    np = _deps['np']
    _advance_layout_prompt_epoch(image_state, layout_state)
    state = dict(layout_state or {})
    try:
        normalized_selection = _normalize_layout_prompt_checkbox_selection(
            selection,
            state,
        )
        scope, selected_region_ids = _parse_layout_prompt_selection(
            normalized_selection
        )
        if scope == _LAYOUT_PROMPT_SCOPE_FULL:
            state = _reset_layout_prompt_selection_state(state)
            info = "已选择全部版图 mask；将保持原有单实例创建行为。"
            return (
                state,
                _layout_editor_payload(image_state, state, info),
                gr.update(value=[_LAYOUT_PROMPT_SCOPE_FULL]),
                info,
                bool(state.get("enabled", True)),
                float(state.get("tx") or 0.0),
                float(state.get("ty") or 0.0),
                float(state.get("scale") or 1.0),
                float(state.get("rotation_deg") or 0.0),
                _layout_preview_alpha(state),
            )

        document, source_mask = _load_layout_prompt_region_document(state)
        records = _layout_prompt_region_records(
            document,
            selected_region_ids,
        )
        region_ids = [int(record["region_id"]) for record in records]
        preview = np.zeros_like(source_mask, dtype=bool)
        for _, mask in _layout_regions.decode_region_masks(
            records,
            source_mask.shape,
        ):
            preview = np.logical_or(preview, mask)
        if not preview.any():
            raise _layout_regions.RegionValidationError(
                "所选 Label 的 Region mask 为空"
            )
        labels = [
            _layout_regions.region_label(record)
            for record in records
        ]
        state.update(
            {
                "prompt_mask_scope": _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                "prompt_class_label": None,
                "prompt_labels": labels,
                "prompt_regions_revision": int(
                    document.get("regions_revision") or 0
                ),
                "prompt_region_ids": region_ids,
                "image_id": image_state.get("image_id"),
                "target_image_sha256": image_state.get(
                    "target_image_sha256"
                ),
            }
        )
        editor = _layout_editor_payload(
            image_state,
            state,
            "正在初始化独立 Label 图层。",
        )
        state, _, _, _, active_transform = (
            _commit_layout_group_transforms(
                image_state,
                state,
                editor,
            )
        )
        editor = _layout_editor_payload(image_state, state)
        selected_tokens = [
            _layout_prompt_selection_token(
                _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                region_id,
            )
            for region_id in region_ids
        ]
        tx, ty, group_scale, group_rotation, group_alpha = (
            _layout_group_control_values(
                active_transform,
                (
                    int(image_state.get("width") or 0),
                    int(image_state.get("height") or 0),
                ),
            )
        )
        info = (
            f"已选择 {len(region_ids)} 个 Label；每个 Label 可独立拖动，"
            "并各自生成一个 PVS instance。"
        )
        editor["status"] = info
        return (
            state,
            editor,
            gr.update(value=selected_tokens),
            info,
            True,
            tx,
            ty,
            group_scale,
            group_rotation,
            group_alpha,
        )
    except Exception as exc:
        if state.get("prompt_mask_scope") == _LAYOUT_PROMPT_SCOPE_REGION_LABELS:
            current_value = [
                _layout_prompt_selection_token(
                    _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                    region_id,
                )
                for region_id in state.get("prompt_region_ids") or []
            ]
        else:
            current_value = [_LAYOUT_PROMPT_SCOPE_FULL]
        info = f"版图 mask Label 选择失败，已保留原选择：{exc}"
        return (
            state,
            _layout_editor_payload(image_state, state, info),
            gr.update(value=current_value),
            info,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _reset_layout_prompt_selection_impl(_deps, image_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_prompt_choice_update = _deps['_layout_prompt_choice_update']
    _reset_layout_prompt_selection_state = _deps['_reset_layout_prompt_selection_state']
    _advance_layout_prompt_epoch(image_state, layout_state)
    state = _reset_layout_prompt_selection_state(layout_state)
    info = "版图 mask 选择已重置；点击‘使用当前已保存版图 mask’加载 Label。"
    return (
        state,
        _layout_editor_payload(image_state, state, info),
        _layout_prompt_choice_update(),
    )


def _create_pvs_from_layout_selection_impl(_deps, image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload, prompt_selection, progress):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _LayoutPromptConflictError = _deps['_LayoutPromptConflictError']
    _commit_layout_group_transforms = _deps['_commit_layout_group_transforms']
    _create_pvs_from_layout_mask = _deps['_create_pvs_from_layout_mask']
    _fresh_state = _deps['_fresh_state']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_prompt_epoch_snapshot = _deps['_layout_prompt_epoch_snapshot']
    _layout_prompt_group_id = _deps['_layout_prompt_group_id']
    _layout_prompt_metadata = _deps['_layout_prompt_metadata']
    _layout_prompt_region_fingerprint = _deps['_layout_prompt_region_fingerprint']
    _layout_prompt_region_records = _deps['_layout_prompt_region_records']
    _layout_regions = _deps['_layout_regions']
    _layout_tx = _deps['_layout_tx']
    _load_layout_prompt_region_document = _deps['_load_layout_prompt_region_document']
    _make_inst = _deps['_make_inst']
    _mask_box = _deps['_mask_box']
    _mask_to_lowres_logits = _deps['_mask_to_lowres_logits']
    _new_layout_state = _deps['_new_layout_state']
    _parse_layout_prompt_selection = _deps['_parse_layout_prompt_selection']
    _predict_inst = _deps['_predict_inst']
    _pvs_creation_commit_token = _deps['_pvs_creation_commit_token']
    _pvs_progress = _deps['_pvs_progress']
    _reset_layout_prompt_selection_state = _deps['_reset_layout_prompt_selection_state']
    _selected_pvs_candidate = _deps['_selected_pvs_candidate']
    _validate_layout_group_transform_snapshot = _deps['_validate_layout_group_transform_snapshot']
    _validate_layout_prompt_mask = _deps['_validate_layout_prompt_mask']
    _view = _deps['_view']
    _workspace = _deps['_workspace']
    copy = _deps['copy']
    gr = _deps['gr']
    np = _deps['np']
    state = dict(layout_state or _new_layout_state())
    try:
        scope, selected_region_ids = _parse_layout_prompt_selection(prompt_selection)
        state_scope = str(
            state.get("prompt_mask_scope") or _LAYOUT_PROMPT_SCOPE_FULL
        )
        if scope != state_scope:
            raise ValueError("版图 mask 选择与服务端状态不一致，请重新选择")
    except Exception as exc:
        info = f"用版图 mask 创建 PVS 实例失败：{exc}"
        editor = _layout_editor_payload(image_state, state, info)
        return (
            pvs_state,
            state,
            editor,
            info,
            *_view(
                image_state,
                pcs_state,
                pvs_state,
                mode,
                info,
                layout_state=state,
            ),
        )

    if scope == _LAYOUT_PROMPT_SCOPE_FULL:
        state = _reset_layout_prompt_selection_state(state)
        return _create_pvs_from_layout_mask(
            image_state,
            pcs_state,
            pvs_state,
            mode,
            state,
            enabled,
            tx,
            ty,
            scale,
            rotation_deg,
            preview_alpha,
            editor_payload,
            progress=progress,
        )

    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 Region prompt 只支持在版图 mask 提示分割模式使用")
        if state.get("prompt_mask_scope") != _LAYOUT_PROMPT_SCOPE_REGION_LABELS:
            raise ValueError("请先在版图 mask 选择中确认至少一个 Label")
        if sorted(selected_region_ids) != [
            int(value) for value in state.get("prompt_region_ids") or []
        ]:
            raise ValueError("版图 Label 选择状态不一致，请重新选择")
        expected_revision = state.get("prompt_regions_revision")
        prompt_epoch_key, prompt_epoch = _layout_prompt_epoch_snapshot(
            image_state,
            state,
        )
        document, source_mask = _load_layout_prompt_region_document(
            state,
            expected_revision=expected_revision,
        )
        records = _layout_prompt_region_records(
            document,
            selected_region_ids,
        )
        region_ids = [int(record["region_id"]) for record in records]
        if region_ids != [int(value) for value in state.get("prompt_region_ids") or []]:
            raise _layout_regions.StaleRegionsRevisionError(
                "版图 Region 列表已变化，请重新加载 Label"
            )
        decoded_records = _layout_regions.decode_region_masks(
            records,
            source_mask.shape,
        )
        frozen_region_fingerprint = _layout_prompt_region_fingerprint(
            decoded_records
        )
        pvs_commit_token = _pvs_creation_commit_token(pvs_state)
        next_instance_id = int(pvs_state.get("next_instance_id", 1))
        existing_instance_ids = {
            int(instance_id)
            for instance_id in (pvs_state.get("instances") or {})
        }
        if next_instance_id <= 0 or (
            existing_instance_ids
            and next_instance_id <= max(existing_instance_ids)
        ):
            raise ValueError("next PVS instance ID 无效或会复用已有 ID")
        planned_instance_ids = set(
            range(next_instance_id, next_instance_id + len(decoded_records))
        )
        if planned_instance_ids.intersection(existing_instance_ids):
            raise ValueError("Region 批次计划的 PVS instance ID 已存在")

        _pvs_progress(progress, 0.04, "提交并冻结各 Label transform")
        (
            state,
            committed_decoded,
            frozen_group_snapshot,
            _,
            _,
        ) = _commit_layout_group_transforms(
            image_state,
            state,
            editor_payload,
            numeric_override=(
                tx,
                ty,
                scale,
                rotation_deg,
                preview_alpha,
            ),
        )
        if (
            [int(record["region_id"]) for record, _ in committed_decoded]
            != region_ids
            or _layout_prompt_region_fingerprint(committed_decoded)
            != frozen_region_fingerprint
        ):
            raise _layout_regions.StaleRegionsRevisionError(
                "版图 Region 在冻结 Label transform 前发生变化"
            )
        decoded_records = committed_decoded
        target_width = int(image_state.get("width") or 0)
        target_height = int(image_state.get("height") or 0)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("目标图像尺寸无效")
        target_shape = (target_height, target_width)
        base_prompt = _layout_prompt_metadata(image_state, state)
        base_prompt["mask_scope"] = "region"
        base_prompt["regions_revision"] = int(
            document.get("regions_revision") or 0
        )
        base_prompt["batch_region_ids"] = list(region_ids)
        base_prompt["batch_labels"] = list(
            frozen_group_snapshot["labels"]
        )
        base_prompt["selection_signature"] = (
            frozen_group_snapshot["selection_signature"]
        )
        base_prompt["transform_set_revision"] = int(
            frozen_group_snapshot["transform_set_revision"]
        )

        staged_instances = {}
        batch_size = len(decoded_records)
        for batch_index, (record, region_mask) in enumerate(
            decoded_records,
            start=1,
        ):
            label = _layout_regions.region_label(record)
            group_id = _layout_prompt_group_id(record["region_id"])
            group_transform = copy.deepcopy(
                frozen_group_snapshot["transforms"][group_id]
            )
            group_matrix = group_transform["matrix_2x3"]
            _pvs_progress(
                progress,
                0.14 + 0.68 * (batch_index - 1) / max(1, batch_size),
                (
                    f"SAM3 正在处理 Label {label} 的 "
                    f"R{record['region_id']}（{batch_index}/{batch_size}）"
                ),
                delay=0.0,
            )
            transformed_region = _layout_tx.warp_layout_mask(
                region_mask,
                group_matrix,
                (target_width, target_height),
            )
            transformed_region = _validate_layout_prompt_mask(
                transformed_region
            )
            lowres_logits = _mask_to_lowres_logits(transformed_region)
            if not np.any(lowres_logits > 0):
                raise ValueError(
                    f"R{record['region_id']} 在 low-res mask_input 中没有前景"
                )
            prediction = _predict_inst(
                _fresh_state(image_state),
                mask_input_lowres_logits=lowres_logits,
            )
            mask, score, selected_logits, candidate_scores = (
                _selected_pvs_candidate(prediction, target_shape)
            )
            instance_id = next_instance_id + batch_index - 1
            prompt = copy.deepcopy(base_prompt)
            prompt.update(
                {
                    "region_id": int(record["region_id"]),
                    "label": label,
                    "group_id": group_id,
                    "group_transform_revision": int(
                        group_transform.get("revision") or 0
                    ),
                    "revision": int(group_transform.get("revision") or 0),
                    "preview_alpha": float(
                        _layout_preview_alpha(group_transform)
                    ),
                    "transform": group_transform,
                    "matrix_2x3": copy.deepcopy(group_matrix),
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "region_mask_pixel_sha256": (
                        _layout_regions.mask_pixel_sha256(
                            region_mask.astype(np.uint8)
                        )
                    ),
                }
            )
            staged_instances[instance_id] = _make_inst(
                instance_id,
                "manual_pvs_layout_mask",
                mask,
                _mask_box(mask),
                score,
                pvs_logits=selected_logits,
                history=[
                    {
                        "op": "create_from_layout_mask",
                        "prompt": prompt,
                        "candidate_scores": candidate_scores,
                    }
                ],
            )

        candidate_state = dict(pvs_state)
        candidate_instances = dict(pvs_state.get("instances") or {})
        candidate_instances.update(staged_instances)
        candidate_state["instances"] = candidate_instances
        candidate_state["next_instance_id"] = next_instance_id + batch_size
        candidate_state["active_instance_id"] = next_instance_id + batch_size - 1
        created_ids = list(staged_instances)
        mapping = ", ".join(
            f"R{region_id}→PVS#{instance_id}"
            for region_id, instance_id in zip(region_ids, created_ids)
        )
        info = f"已按选中 Label 原子创建 {batch_size} 个 PVS 实例：{mapping}"
        editor = _layout_editor_payload(image_state, state, info)
        view = _view(
            image_state,
            pcs_state,
            candidate_state,
            mode,
            info,
            layout_state=state,
        )

        _pvs_progress(
            progress,
            0.96,
            "准备原子提交 Region PVS 批次",
            delay=0.12,
        )
        try:
            latest_document, latest_source = (
                _load_layout_prompt_region_document(
                    state,
                    expected_revision=int(
                        document.get("regions_revision") or 0
                    ),
                )
            )
            latest_records = _layout_prompt_region_records(
                latest_document,
                region_ids,
            )
            latest_decoded = _layout_regions.decode_region_masks(
                latest_records,
                latest_source.shape,
            )
            if (
                [int(record["region_id"]) for record, _ in latest_decoded]
                != region_ids
                or _layout_prompt_region_fingerprint(latest_decoded)
                != frozen_region_fingerprint
            ):
                raise _layout_regions.StaleRegionsRevisionError(
                    "版图 Region 在批量预测期间发生变化"
                )
            current_epoch_key, current_epoch = (
                _layout_prompt_epoch_snapshot(image_state, state)
            )
            if (
                current_epoch_key != prompt_epoch_key
                or current_epoch != prompt_epoch
            ):
                raise ValueError(
                    "版图 identity、Label 选择或模式在批量预测期间发生变化"
                )
            _validate_layout_group_transform_snapshot(
                state,
                frozen_group_snapshot,
            )
            workspace_image = _workspace(image_state)["image"]
            workspace_hash = _layout_tx.image_pixel_sha256(
                workspace_image
            )
            if workspace_hash != frozen_group_snapshot.get(
                "target_image_sha256"
            ):
                raise ValueError("目标图像在批量预测期间发生变化")
            if _pvs_creation_commit_token(pvs_state) != pvs_commit_token:
                raise ValueError(
                    "PVS state 在批量预测期间发生变化，整批结果未提交"
                )
        except Exception as conflict:
            raise _LayoutPromptConflictError(str(conflict)) from conflict

        return candidate_state, state, editor, info, *view
    except Exception as exc:
        info = f"按 Label 创建 PVS 失败，整批未提交：{exc}"
        if isinstance(exc, _LayoutPromptConflictError):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                info,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                info,
                gr.skip(),
            )
        editor = _layout_editor_payload(image_state, state, info)
        try:
            view = _view(
                image_state,
                pcs_state,
                pvs_state,
                mode,
                info,
                layout_state=state,
            )
        except Exception as view_exc:
            info = f"{info}；界面刷新失败：{view_exc}"
            view = (
                gr.update(),
                gr.update(),
                gr.update(value=info),
                gr.update(),
                gr.update(),
                gr.update(
                    value=(
                        str(pvs_state.get("active_instance_id"))
                        if pvs_state.get("active_instance_id") is not None
                        else None
                    )
                ),
                info,
                gr.update(),
            )
        return pvs_state, state, editor, info, *view
