"""Layout Region annotation and persistence callbacks."""

from __future__ import annotations


def _layout_region_png_data_url_impl(_deps, image):
    _sam3_base64 = _deps['_sam3_base64']
    io = _deps['io']
    if image is None:
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + _sam3_base64.b64encode(buf.getvalue()).decode("ascii")


def _new_layout_region_state_impl(_deps):
    return {
        "session_id": None,
        "layout_id": None,
        "source_mask_hash": None,
        "regions_revision": 0,
        "next_region_id": 1,
        "selected_region_id": None,
    }


def _layout_region_state_from_document_impl(_deps, document, selected_region_id):
    _layout_regions = _deps['_layout_regions']
    active_ids = {int(region["region_id"]) for region in _layout_regions.active_regions(document)}
    selected = int(selected_region_id) if selected_region_id not in (None, "") else None
    if selected not in active_ids:
        selected = None
    return {
        "session_id": document.get("session_id"),
        "layout_id": document.get("layout_id"),
        "source_mask_hash": document.get("source_mask_hash"),
        "regions_revision": int(document.get("regions_revision") or 0),
        "next_region_id": int(document.get("next_region_id") or 1),
        "selected_region_id": selected,
    }


def _layout_region_editor_empty_impl(_deps, status):
    return {
        "server_view": {
            "enabled": False,
            "source_image": "",
            "source_mask_image": "",
            "saved_region_overlay_image": "",
            "draft_region_overlay_image": "",
            "natural_width": 0,
            "natural_height": 0,
            "regions_revision": 0,
            "selected_region_id": None,
            "regions": [],
            "status": status,
        },
        "client_intent": {
            "tool_mode": "browse",
            "lasso_polygon": [],
            "expected_regions_revision": 0,
            "session_id": "",
            "layout_id": "",
            "source_mask_hash": "",
        },
    }


def _layout_region_identity_impl(_deps, layout_state):
    _layout_regions = _deps['_layout_regions']
    if not isinstance(layout_state, dict):
        raise _layout_regions.RegionValidationError("layout state is missing")
    session_id = _layout_regions.safe_path_component(layout_state.get("session_id"), "session_id")
    layout_id = _layout_regions.safe_path_component(layout_state.get("layout_id"), "layout_id")
    source_mask_hash = str(layout_state.get("source_mask_pixel_sha256") or "")
    if not source_mask_hash:
        raise _layout_regions.RegionValidationError("layout state source mask hash is missing")
    return session_id, layout_id, source_mask_hash


def _layout_region_state_matches_identity_impl(_deps, region_state, session_id, layout_id, source_mask_hash):
    if not isinstance(region_state, dict):
        return False
    return all(
        region_state.get(field) == expected
        for field, expected in {
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
        }.items()
    )


def _validate_layout_region_state_identity_impl(_deps, layout_state, region_state):
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_regions = _deps['_layout_regions']
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    if not isinstance(region_state, dict):
        raise _layout_regions.RegionValidationError("Region state is missing")
    for field, expected in {
        "session_id": session_id,
        "layout_id": layout_id,
        "source_mask_hash": source_mask_hash,
    }.items():
        if region_state.get(field) != expected:
            raise _layout_regions.RegionValidationError(
                f"Region state {field} does not match current layout"
            )
    return session_id, layout_id, source_mask_hash


def _layout_region_client_intent_impl(_deps, payload):
    raw = payload if isinstance(payload, dict) else {}
    intent = raw.get("client_intent") if isinstance(raw.get("client_intent"), dict) else raw
    tool_mode = intent.get("tool_mode") if intent.get("tool_mode") in {"browse", "lasso"} else "browse"
    polygon = intent.get("lasso_polygon") if isinstance(intent.get("lasso_polygon"), list) else []
    revision = intent.get("expected_regions_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        revision = None
    return {
        "tool_mode": tool_mode,
        "lasso_polygon": polygon,
        "expected_regions_revision": revision,
        "session_id": str(intent.get("session_id") or ""),
        "layout_id": str(intent.get("layout_id") or ""),
        "source_mask_hash": str(intent.get("source_mask_hash") or ""),
    }


def _validate_layout_region_intent_impl(_deps, layout_state, intent, region_state, require_lasso):
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_regions = _deps['_layout_regions']
    _validate_layout_region_state_identity = _deps['_validate_layout_region_state_identity']
    if region_state is None:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    else:
        session_id, layout_id, source_mask_hash = _validate_layout_region_state_identity(
            layout_state, region_state
        )
    for field, value in {
        "session_id": session_id,
        "layout_id": layout_id,
        "source_mask_hash": source_mask_hash,
    }.items():
        if intent.get(field) != value:
            raise _layout_regions.RegionValidationError(
                f"Region request {field} does not match current layout"
            )
    if intent.get("expected_regions_revision") is None:
        raise _layout_regions.RegionValidationError("Region request revision is missing")
    if require_lasso and intent.get("tool_mode") != "lasso":
        raise _layout_regions.RegionValidationError("请切换到套索选择工具")
    return session_id, layout_id, source_mask_hash


def _layout_region_source_image_impl(_deps, session_id, layout_id, source_mask):
    Image = _deps['Image']
    _layout_mask_to_preview = _deps['_layout_mask_to_preview']
    runtime_layout_dir = _deps['runtime_layout_dir']
    source_path = runtime_layout_dir / session_id / layout_id / "source_image.png"
    if source_path.exists():
        with Image.open(source_path) as image:
            return image.convert("RGB").copy()
    return _layout_mask_to_preview(source_mask)


def _layout_region_summaries_impl(_deps, document):
    _layout_regions = _deps['_layout_regions']
    return [
        {
            "region_id": int(region["region_id"]),
            "label": _layout_regions.region_label(region),
            "area": int(region.get("area") or 0),
        }
        for region in _layout_regions.active_regions(document)
    ]


def _layout_region_choice_update_impl(_deps, document, selected_region_id):
    _layout_label_choice_text = _deps['_layout_label_choice_text']
    _layout_prompt_label_counts = _deps['_layout_prompt_label_counts']
    _layout_regions = _deps['_layout_regions']
    gr = _deps['gr']
    active = list(_layout_regions.active_regions(document))
    label_counts = dict(_layout_prompt_label_counts(document))
    choices = []
    active_ids = set()
    for region in active:
        region_id = int(region["region_id"])
        active_ids.add(region_id)
        display = (
            f"{_layout_label_choice_text(region, label_counts)}"
            f" | area={int(region.get('area') or 0)}"
        )
        choices.append((display, region_id))
    selected = int(selected_region_id) if selected_region_id not in (None, "") else None
    if selected not in active_ids:
        selected = None
    return gr.update(choices=choices, value=selected)


def _layout_region_category_update_impl(_deps, value):
    _layout_region_label_update = _deps['_layout_region_label_update']
    return _layout_region_label_update(value=value)


def _layout_region_label_update_impl(_deps, document, value):
    gr = _deps['gr']
    del document
    selected = value.strip() if isinstance(value, str) else ""
    return gr.update(value=selected)


def _layout_region_editor_payload_impl(_deps, layout_state, document, source_mask, status, selected_region_id, lasso_polygon, draft_region_mask):
    _data_url = _deps['_data_url']
    _layout_mask_to_editor_image = _deps['_layout_mask_to_editor_image']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_region_png_data_url = _deps['_layout_region_png_data_url']
    _layout_region_source_image = _deps['_layout_region_source_image']
    _layout_region_summaries = _deps['_layout_region_summaries']
    _layout_regions = _deps['_layout_regions']
    copy = _deps['copy']
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    source_image = _layout_region_source_image(session_id, layout_id, source_mask)
    saved_overlay = _layout_regions.render_saved_region_overlay(
        document.get("regions") or [], source_mask.shape, selected_region_id=selected_region_id
    )
    draft_overlay = (
        _layout_regions.render_draft_region_overlay(draft_region_mask)
        if draft_region_mask is not None
        else None
    )
    return {
        "server_view": {
            "enabled": True,
            "source_image": _data_url(source_image),
            "source_mask_image": _data_url(_layout_mask_to_editor_image(source_mask)),
            "saved_region_overlay_image": _layout_region_png_data_url(saved_overlay),
            "draft_region_overlay_image": _layout_region_png_data_url(draft_overlay),
            "natural_width": int(source_mask.shape[1]),
            "natural_height": int(source_mask.shape[0]),
            "regions_revision": int(document.get("regions_revision") or 0),
            "selected_region_id": selected_region_id,
            "regions": _layout_region_summaries(document),
            "status": status,
        },
        "client_intent": {
            "tool_mode": "lasso",
            "lasso_polygon": copy.deepcopy(lasso_polygon or []),
            "expected_regions_revision": int(document.get("regions_revision") or 0),
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
        },
    }


def _load_layout_region_context_impl(_deps, layout_state):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_choice_update = _deps['_layout_region_choice_update']
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_region_label_update = _deps['_layout_region_label_update']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _layout_regions = _deps['_layout_regions']
    _new_layout_region_state = _deps['_new_layout_region_state']
    gr = _deps['gr']
    try:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
        document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        active = _layout_regions.active_regions(document)
        selected = int(active[0]["region_id"]) if active else None
        status = f"Label 标注器已加载：active={len(active)}, revision={document['regions_revision']}"
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            _layout_region_label_update(document),
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"Label 标注器加载失败：{exc}"
        try:
            label_update = _layout_region_label_update()
        except Exception:
            label_update = gr.update(value="")
        return (
            _new_layout_region_state(),
            _layout_region_editor_empty(status),
            label_update,
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            status,
        )


def _clear_layout_region_context_impl(_deps, _layout_state):
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_label_update = _deps['_layout_region_label_update']
    _new_layout_region_state = _deps['_new_layout_region_state']
    gr = _deps['gr']
    status = "当前版图 Label UI 已清空；磁盘 regions.json 未删除"
    try:
        label_update = _layout_region_label_update()
    except Exception:
        label_update = gr.update(value="")
    return (
        _new_layout_region_state(),
        _layout_region_editor_empty(status),
        label_update,
        gr.update(choices=[], value=None),
        gr.update(interactive=False),
        gr.update(interactive=False),
        status,
    )


def _preview_layout_region_impl(_deps, layout_state, region_state, editor_payload):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_client_intent = _deps['_layout_region_client_intent']
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _new_layout_region_state = _deps['_new_layout_region_state']
    _validate_layout_region_intent = _deps['_validate_layout_region_intent']
    gr = _deps['gr']
    intent = _layout_region_client_intent(editor_payload)
    selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(
            layout_state, intent, region_state=region_state, require_lasso=True
        )
        region_mask, document = _LAYOUT_REGION_STORE.preview_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            lasso_polygon=intent["lasso_polygon"],
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        status = (
            f"Draft 预览完成：area={int(region_mask.sum())}；"
            "Label 可选，留空将按序号自动生成"
        )
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state,
                document,
                source_mask,
                status=status,
                selected_region_id=selected,
                lasso_polygon=intent["lasso_polygon"],
                draft_region_mask=region_mask,
            ),
            gr.update(interactive=True),
            status,
        )
    except Exception as exc:
        status = f"Draft 预览失败：{exc}"
        try:
            session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
            document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
            state = _layout_region_state_from_document(document, selected)
            editor = _layout_region_editor_payload(
                layout_state,
                document,
                source_mask,
                status=status,
                selected_region_id=state.get("selected_region_id"),
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
        return state, editor, gr.update(interactive=False), status


def _select_layout_region_impl(_deps, layout_state, region_state, selected_region_id):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _layout_regions = _deps['_layout_regions']
    _new_layout_region_state = _deps['_new_layout_region_state']
    _validate_layout_region_state_identity = _deps['_validate_layout_region_state_identity']
    gr = _deps['gr']
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_state_identity(
            layout_state, region_state
        )
        document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        selected = int(selected_region_id) if selected_region_id not in (None, "") else None
        state = _layout_region_state_from_document(document, selected)
        selected = state.get("selected_region_id")
        selected_record = next(
            (
                record
                for record in _layout_regions.active_regions(document)
                if int(record["region_id"]) == int(selected or -1)
            ),
            None,
        )
        status = (
            f"已选择 {_layout_regions.region_label(selected_record)}"
            if selected_record is not None
            else "未选择活动 Label"
        )
        return (
            state,
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"选择 Label 失败：{exc}"
        return (
            region_state or _new_layout_region_state(),
            _layout_region_editor_empty(status),
            gr.update(interactive=False),
            gr.update(interactive=False),
            status,
        )


def _layout_region_latest_values_impl(_deps, layout_state, selected, status, intent, keep_draft, region_state):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_choice_update = _deps['_layout_region_choice_update']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _layout_region_state_matches_identity = _deps['_layout_region_state_matches_identity']
    _layout_regions = _deps['_layout_regions']
    gr = _deps['gr']
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
    state = _layout_region_state_from_document(document, selected)
    draft_mask = None
    polygon = None
    if (
        keep_draft
        and isinstance(intent, dict)
        and _layout_region_state_matches_identity(
            region_state, session_id, layout_id, source_mask_hash
        )
        and intent.get("session_id") == session_id
        and intent.get("layout_id") == layout_id
        and intent.get("source_mask_hash") == source_mask_hash
        and intent.get("expected_regions_revision") == document.get("regions_revision")
        and intent.get("lasso_polygon")
    ):
        draft_mask = _layout_regions.rasterize_uncovered_region_mask(
            source_mask,
            intent["lasso_polygon"],
            document,
        )
        _layout_regions.mask_metadata(draft_mask)
        polygon = intent["lasso_polygon"]
    editor = _layout_region_editor_payload(
        layout_state,
        document,
        source_mask,
        status=status,
        selected_region_id=state.get("selected_region_id"),
        lasso_polygon=polygon,
        draft_region_mask=draft_mask,
    )
    return (
        state,
        editor,
        _layout_region_choice_update(document, state.get("selected_region_id")),
        gr.update(interactive=draft_mask is not None),
        gr.update(interactive=state.get("selected_region_id") is not None),
    )


def _save_layout_region_impl(_deps, layout_state, region_state, editor_payload, label):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_choice_update = _deps['_layout_region_choice_update']
    _layout_region_client_intent = _deps['_layout_region_client_intent']
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_label_update = _deps['_layout_region_label_update']
    _layout_region_latest_values = _deps['_layout_region_latest_values']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _layout_regions = _deps['_layout_regions']
    _new_layout_region_state = _deps['_new_layout_region_state']
    _validate_layout_region_intent = _deps['_validate_layout_region_intent']
    gr = _deps['gr']
    intent = _layout_region_client_intent(editor_payload)
    previous_selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(
            layout_state, intent, region_state=region_state, require_lasso=True
        )
        document, record = _LAYOUT_REGION_STORE.save_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            lasso_polygon=intent["lasso_polygon"],
            label="" if label is None else label,
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        selected = int(record["region_id"])
        status = (
            f"已保存 Label {_layout_regions.region_label(record)}"
            f" | area={record['area']}"
        )
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            _layout_region_label_update(document),
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=True),
            status,
        )
    except Exception as exc:
        status = f"保存 Label 失败：{exc}"
        try:
            state, editor, choices, save_update, delete_update = _layout_region_latest_values(
                layout_state,
                previous_selected,
                status,
                intent=intent,
                keep_draft=True,
                region_state=region_state,
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
            choices = gr.update(choices=[], value=None)
            save_update = gr.update(interactive=False)
            delete_update = gr.update(interactive=False)
        try:
            label_update = _layout_region_label_update(value=label)
        except Exception:
            label_update = gr.update(value="")
        return (
            state,
            editor,
            label_update,
            choices,
            save_update,
            delete_update,
            status,
        )


def _delete_layout_region_impl(_deps, layout_state, region_state, editor_payload, selected_region_id):
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_choice_update = _deps['_layout_region_choice_update']
    _layout_region_client_intent = _deps['_layout_region_client_intent']
    _layout_region_editor_empty = _deps['_layout_region_editor_empty']
    _layout_region_editor_payload = _deps['_layout_region_editor_payload']
    _layout_region_label_update = _deps['_layout_region_label_update']
    _layout_region_latest_values = _deps['_layout_region_latest_values']
    _layout_region_state_from_document = _deps['_layout_region_state_from_document']
    _layout_regions = _deps['_layout_regions']
    _new_layout_region_state = _deps['_new_layout_region_state']
    _validate_layout_region_intent = _deps['_validate_layout_region_intent']
    gr = _deps['gr']
    intent = _layout_region_client_intent(editor_payload)
    previous_selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(
            layout_state, intent, region_state=region_state
        )
        if selected_region_id in (None, ""):
            raise _layout_regions.RegionValidationError("请先选择活动 Label")
        document, deleted = _LAYOUT_REGION_STORE.delete_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            region_id=int(selected_region_id),
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        active = _layout_regions.active_regions(document)
        selected = int(active[0]["region_id"]) if active else None
        status = (
            f"已软删除 Label {_layout_regions.region_label(deleted)}；"
            "binary mask 与历史 RLE 均保留"
        )
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            _layout_region_label_update(document),
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"软删除 Label 失败：{exc}"
        try:
            state, editor, choices, save_update, delete_update = _layout_region_latest_values(
                layout_state, previous_selected, status
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
            choices = gr.update(choices=[], value=None)
            save_update = gr.update(interactive=False)
            delete_update = gr.update(interactive=False)
        return (
            state,
            editor,
            gr.update(),
            choices,
            save_update,
            delete_update,
            status,
        )


def _export_layout_regions_impl(_deps, layout_state, region_state):
    Path = _deps['Path']
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    _layout_region_identity = _deps['_layout_region_identity']
    _layout_regions = _deps['_layout_regions']
    _prune_public_downloads = _deps['_prune_public_downloads']
    _public_downloads = _deps['_public_downloads']
    cv2 = _deps['cv2']
    json = _deps['json']
    np = _deps['np']
    public_download_dir = _deps['public_download_dir']
    runtime_export_dir = _deps['runtime_export_dir']
    tempfile = _deps['tempfile']
    try:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
        state = region_state if isinstance(region_state, dict) else {}
        expected_identity = {
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
        }
        for field, expected in expected_identity.items():
            if state.get(field) != expected:
                raise _layout_regions.RegionValidationError(
                    f"Region export {field} does not match current layout"
                )
        revision = state.get("regions_revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise _layout_regions.RegionValidationError("Region export revision is missing")

        document, source_mask = _LAYOUT_REGION_STORE.load_document(
            session_id,
            layout_id,
            source_mask_hash,
        )
        current_revision = document.get("regions_revision")
        if revision != current_revision:
            raise _layout_regions.StaleRegionsRevisionError(
                f"stale regions revision: expected {revision}, current {current_revision}"
            )

        records = document.get("regions") or []
        active_count = len(_layout_regions.active_regions(document))
        label_index, label_mapping = _layout_regions.region_label_index(
            document,
            source_mask.shape,
        )
        label_mask_payloads = []
        for entry in label_mapping:
            mask_file = (
                "label_masks/"
                f"label_{int(entry['index']):04d}_R{int(entry['region_id'])}.png"
            )
            entry["mask_file"] = mask_file
            label_mask_payloads.append(
                (
                    mask_file,
                    np.asarray(label_index == int(entry["index"]), dtype=np.uint8)
                    * 255,
                )
            )
        label_mask_files = [path for path, _ in label_mask_payloads]
        labels_payload = {
            "schema_version": 1,
            "background_or_unlabeled_value": 0,
            "labels": label_mapping,
        }
        manifest = {
            "schema_version": 1,
            "export_type": "layout_region_annotations",
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
            "regions_revision": revision,
            "region_count": len(records),
            "active_region_count": active_count,
            "deleted_region_count": len(records) - active_count,
            "label_mask_count": len(label_mask_files),
            "exported_at": _layout_regions.utc_now_iso(),
            "files": [
                "regions.json",
                "source_mask.png",
                "region_label_index.png",
                "labels.json",
                "manifest.json",
                *label_mask_files,
            ],
            "label_mask_files": label_mask_files,
            "label_index_encoding": (
                "uint16; 0 means background or unlabeled source-mask foreground"
            ),
            "label_mask_encoding": (
                "8-bit grayscale PNG; 0 means background and 255 means Label foreground"
            ),
        }

        with tempfile.TemporaryDirectory(
            prefix="layout_region_export_",
            dir=runtime_export_dir,
        ) as temporary:
            staging_dir = Path(temporary)
            with (staging_dir / "regions.json").open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            with (staging_dir / "labels.json").open("w", encoding="utf-8") as handle:
                json.dump(labels_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            with (staging_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if not cv2.imwrite(
                str(staging_dir / "source_mask.png"),
                np.asarray(source_mask, dtype=np.uint8) * 255,
            ):
                raise OSError("cannot write source_mask.png")
            if not cv2.imwrite(
                str(staging_dir / "region_label_index.png"),
                np.asarray(label_index, dtype=np.uint16),
            ):
                raise OSError("cannot write region_label_index.png")
            for mask_file, label_mask in label_mask_payloads:
                mask_path = staging_dir / mask_file
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), label_mask):
                    raise OSError(f"cannot write {mask_file}")
            _prune_public_downloads()
            archive_path = _public_downloads.publish_zip(
                public_download_dir,
                "region_annotation_exports",
                staging_dir,
                f"layout_regions_{layout_id}_r{revision}.zip",
            )

        status = (
            "已导出版图 mask 与 label 标注："
            f"active={active_count}, label_masks={len(label_mask_files)}, "
            f"revision={revision}"
        )
        return str(archive_path), status
    except Exception as exc:
        return None, f"导出 Label 标注失败：{exc}"
