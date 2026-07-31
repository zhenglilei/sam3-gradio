"""Layout mask generation, persistence, and transform callbacks."""

from __future__ import annotations


def _layout_cache_key_impl(_deps, session_id, layout_id):
    _layout_tx = _deps['_layout_tx']
    if not layout_id:
        raise ValueError("请先在‘版图截图转掩码’Tab 中生成并保存当前版图 mask")
    sid = _layout_tx.safe_id(session_id, "default")
    lid = _layout_tx.safe_id(layout_id, "layout")
    return f"{sid}:{lid}"


def _layout_disk_dir_impl(_deps, session_id, layout_id):
    _layout_tx = _deps['_layout_tx']
    runtime_layout_dir = _deps['runtime_layout_dir']
    return runtime_layout_dir / _layout_tx.safe_id(session_id, "default") / _layout_tx.safe_id(layout_id, "layout")


def _layout_cache_get_impl(_deps, layout_state_or_id, session_id):
    _LAYOUT_CACHE = _deps['_LAYOUT_CACHE']
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_key = _deps['_layout_cache_key']
    _restore_layout_cache_from_disk = _deps['_restore_layout_cache_from_disk']
    if isinstance(layout_state_or_id, dict):
        layout_id = layout_state_or_id.get("layout_id")
        session_id = session_id or layout_state_or_id.get("session_id")
    else:
        layout_id = layout_state_or_id
    if not layout_id:
        raise ValueError("请先在‘版图截图转掩码’Tab 中生成并保存当前版图 mask")
    session_id = session_id or "default"
    key = _layout_cache_key(session_id, layout_id)
    with _LAYOUT_CACHE_LOCK:
        cached = _LAYOUT_CACHE.get(key)
        if cached is not None:
            return cached
        cached = _restore_layout_cache_from_disk(session_id, layout_id)
        if cached is not None:
            _LAYOUT_CACHE[key] = cached
            return cached
    raise ValueError(f"版图缓存已失效或不存在: {layout_id}。请重新生成版图 mask。")


def _restore_layout_cache_from_disk_impl(_deps, session_id, layout_id):
    Image = _deps['Image']
    _layout_disk_dir = _deps['_layout_disk_dir']
    _layout_mask_to_preview = _deps['_layout_mask_to_preview']
    _layout_tx = _deps['_layout_tx']
    cv2 = _deps['cv2']
    json = _deps['json']
    np = _deps['np']
    out_dir = _layout_disk_dir(session_id, layout_id)
    meta_path = out_dir / "layout_meta.json"
    mask_path = out_dir / "source_mask.png"
    if not meta_path.exists() or not mask_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    file_hash = _layout_tx.file_sha256(mask_path)
    if meta.get("source_mask_file_sha256") and meta.get("source_mask_file_sha256") != file_hash:
        raise ValueError("版图 source_mask.png 文件 hash 不匹配，拒绝恢复缓存")
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("版图 source_mask.png 无法读取")
    source_mask = gray >= 128
    pixel_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    if meta.get("source_mask_pixel_sha256") and meta.get("source_mask_pixel_sha256") != pixel_hash:
        raise ValueError("版图 source mask 像素 hash 不匹配，拒绝恢复缓存")
    image_path = out_dir / "source_image.png"
    source_image = Image.open(image_path).convert("RGB") if image_path.exists() else _layout_mask_to_preview(source_mask)
    return {
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "source_image": source_image,
        "source_mask": source_mask,
        "source_mask_path": str(mask_path),
        "source_mask_pixel_sha256": pixel_hash,
        "source_mask_file_sha256": file_hash,
        "target_image_sha256": meta.get("target_image_sha256"),
        "layout_meta_path": str(meta_path),
        "foreground_bbox_xyxy": meta.get("foreground_bbox_xyxy") or _layout_tx.foreground_bbox_xyxy(source_mask),
        "pivot_xy": meta.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(_layout_tx.foreground_bbox_xyxy(source_mask)),
        "source_width": int(source_mask.shape[1]),
        "source_height": int(source_mask.shape[0]),
        "transformed_mask": None,
        "committed_revision": int(meta.get("committed_revision") or 0),
        "backend_transform": meta.get("backend_transform"),
        "matrix_2x3": meta.get("matrix_2x3"),
        "contours": meta.get("contours") or [],
        "binarize_params": meta.get("binarize_params") or {},
        "mask_path": str(mask_path),
        "contour_json_path": str(out_dir / "contours.json") if (out_dir / "contours.json").exists() else None,
        "overlay_path": str(out_dir / "contour_overlay.png") if (out_dir / "contour_overlay.png").exists() else None,
    }


def _write_layout_meta_impl(_deps, cached):
    Path = _deps['Path']
    json = _deps['json']
    meta_path = Path(cached["layout_meta_path"])
    payload = {
        "session_id": cached.get("session_id"),
        "layout_id": cached.get("layout_id"),
        "source_mask_path": cached.get("source_mask_path"),
        "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        "target_image_sha256": cached.get("target_image_sha256"),
        "foreground_bbox_xyxy": cached.get("foreground_bbox_xyxy"),
        "pivot_xy": cached.get("pivot_xy"),
        "source_width": cached.get("source_width"),
        "source_height": cached.get("source_height"),
        "committed_revision": cached.get("committed_revision"),
        "backend_transform": cached.get("backend_transform"),
        "matrix_2x3": cached.get("matrix_2x3"),
        "binarize_params": cached.get("binarize_params") or {},
        "contours": cached.get("contours") or [],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _layout_cache_put_impl(_deps, session_id, layout_id, source_image, source_mask, contours, binarize_params, mask_path, contour_json_path, overlay_path, layout_meta_path):
    _LAYOUT_CACHE = _deps['_LAYOUT_CACHE']
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_key = _deps['_layout_cache_key']
    _layout_tx = _deps['_layout_tx']
    _pil_image = _deps['_pil_image']
    _write_layout_meta = _deps['_write_layout_meta']
    np = _deps['np']
    source_mask = np.asarray(source_mask, dtype=bool)
    bbox = _layout_tx.foreground_bbox_xyxy(source_mask)
    pivot = _layout_tx.pivot_from_bbox_xyxy(bbox)
    mask_path = str(mask_path) if mask_path else None
    file_hash = _layout_tx.file_sha256(mask_path) if mask_path else None
    pixel_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    cached = {
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "source_image": _pil_image(source_image),
        "source_mask": source_mask,
        "source_mask_path": mask_path,
        "layout_meta_path": str(layout_meta_path) if layout_meta_path else None,
        "source_mask_pixel_sha256": pixel_hash,
        "source_mask_file_sha256": file_hash,
        "target_image_sha256": None,
        "foreground_bbox_xyxy": bbox,
        "pivot_xy": pivot,
        "source_width": int(source_mask.shape[1]),
        "source_height": int(source_mask.shape[0]),
        "transformed_mask": None,
        "committed_revision": 0,
        "backend_transform": None,
        "matrix_2x3": None,
        "contours": contours or [],
        "binarize_params": dict(binarize_params or {}),
        "mask_path": mask_path,
        "contour_json_path": str(contour_json_path) if contour_json_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
    }
    key = _layout_cache_key(session_id, layout_id)
    with _LAYOUT_CACHE_LOCK:
        _LAYOUT_CACHE[key] = cached
        if cached.get("layout_meta_path"):
            _write_layout_meta(cached)
    return cached


def _clear_layout_cache_impl(_deps, layout_state):
    _LAYOUT_CACHE = _deps['_LAYOUT_CACHE']
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_key = _deps['_layout_cache_key']
    with _LAYOUT_CACHE_LOCK:
        if layout_state and isinstance(layout_state, dict) and layout_state.get("layout_id"):
            _LAYOUT_CACHE.pop(_layout_cache_key(layout_state.get("session_id") or "default", layout_state.get("layout_id")), None)
        else:
            _LAYOUT_CACHE.clear()


def _normalize_layout_morph_pixels_impl(_deps, value):
    _LAYOUT_MASK_MORPH_LIMIT_PX = _deps['_LAYOUT_MASK_MORPH_LIMIT_PX']
    np = _deps['np']
    return int(
        np.clip(
            int(value or 0),
            -_LAYOUT_MASK_MORPH_LIMIT_PX,
            _LAYOUT_MASK_MORPH_LIMIT_PX,
        )
    )


def _apply_layout_mask_morphology_impl(_deps, mask, morph_pixels):
    _normalize_layout_morph_pixels = _deps['_normalize_layout_morph_pixels']
    cv2 = _deps['cv2']
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    pixels = _normalize_layout_morph_pixels(morph_pixels)
    if pixels == 0:
        return mask.copy()
    radius = abs(pixels)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    operation = cv2.dilate if pixels > 0 else cv2.erode
    result = operation(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return result.astype(bool)


def _binarize_layout_image_impl(_deps, input_image, threshold, invert, open_kernel, close_kernel, morph_pixels):
    _apply_layout_mask_morphology = _deps['_apply_layout_mask_morphology']
    _layout_extract_mask = _deps['_layout_extract_mask']
    _pil_image = _deps['_pil_image']
    cv2 = _deps['cv2']
    np = _deps['np']
    image = _pil_image(input_image)
    if image is None:
        raise ValueError("请先上传版图截图")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    threshold = int(np.clip(int(threshold), 0, 255))
    if _layout_extract_mask is not None:
        mask = _layout_extract_mask(rgb, saturation_min=max(1, threshold), value_min=1, chroma_min=0)
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = hsv[..., 1] >= max(1, threshold)
    if not np.asarray(mask).any():
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray <= max(1, 255 - threshold)
    mask = np.asarray(mask, dtype=bool)
    if invert:
        mask = ~mask
    open_kernel = int(max(0, open_kernel or 0))
    close_kernel = int(max(0, close_kernel or 0))
    work = mask.astype(np.uint8)
    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k, iterations=1)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)
    work = _apply_layout_mask_morphology(work, morph_pixels)
    return image, work.astype(bool)


def _filter_layout_components_impl(_deps, mask, min_component_area, region_mode):
    cv2 = _deps['cv2']
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    min_area = max(0, int(min_component_area or 0))
    region_mode = str(region_mode or "all")
    if not mask.any():
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(bool)
    component_ids = list(range(1, num_labels))
    if min_area > 0:
        component_ids = [idx for idx in component_ids if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area]
    if region_mode == "largest" and component_ids:
        component_ids = [max(component_ids, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))]
    filtered = np.isin(labels, component_ids)
    return filtered.astype(bool)


def _compute_layout_mask_draft_impl(
    _deps,
    input_image,
    threshold,
    invert,
    open_kernel,
    close_kernel,
    min_component_area,
    region_mode,
    morph_pixels,
):
    """Run the authoritative preprocessing pipeline without persisting output."""
    _binarize_layout_image = _deps['_binarize_layout_image']
    _filter_layout_components = _deps['_filter_layout_components']
    _layout_mask_contours = _deps['_layout_mask_contours']
    _normalize_layout_morph_pixels = _deps['_normalize_layout_morph_pixels']
    image, mask = _binarize_layout_image(
        input_image,
        threshold,
        invert,
        open_kernel,
        close_kernel,
        morph_pixels,
    )
    mask = _filter_layout_components(mask, min_component_area, region_mode)
    contours = _layout_mask_contours(mask)
    params = {
        "threshold": int(threshold),
        "invert": bool(invert),
        "open_kernel": int(open_kernel or 0),
        "close_kernel": int(close_kernel or 0),
        "morph_pixels": _normalize_layout_morph_pixels(morph_pixels),
        "min_component_area": int(min_component_area or 0),
        "region_mode": str(region_mode or "all"),
    }
    return image, mask, contours, params


def _layout_mask_contours_impl(_deps, mask):
    cv2 = _deps['cv2']
    np = _deps['np']
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy_rows = hierarchy[0] if hierarchy is not None else []
    rows = []
    for idx, contour in enumerate(contours):
        if contour.shape[0] < 3:
            continue
        points = contour.reshape(-1, 2).astype(float).tolist()
        x, y, w, h = cv2.boundingRect(contour)
        parent = int(hierarchy_rows[idx][3]) if len(hierarchy_rows) else -1
        rows.append({
            "id": idx + 1,
            "is_hole": parent >= 0,
            "area": float(cv2.contourArea(contour)),
            "bbox_xywh": [float(x), float(y), float(w), float(h)],
            "points": points,
        })
    return rows


def _layout_mask_to_preview_impl(_deps, mask):
    Image = _deps['Image']
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_contour_overlay_impl(_deps, image, mask, contours):
    Image = _deps['Image']
    _pil_image = _deps['_pil_image']
    cv2 = _deps['cv2']
    np = _deps['np']
    base = np.asarray(_pil_image(image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    fill = base.copy()
    fill[mask] = (40, 220, 80)
    vis = cv2.addWeighted(fill, 0.32, base, 0.68, 0)
    for item in contours:
        pts = np.asarray(item.get("points", []), dtype=np.int32).reshape((-1, 1, 2))
        if pts.shape[0] < 3:
            continue
        color = (255, 60, 60) if item.get("is_hole") else (0, 255, 80)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 0), thickness=4)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    return Image.fromarray(vis)


def _save_layout_mask_files_impl(_deps, session_state, source_image, mask, contours, params):
    _layout_cache_put = _deps['_layout_cache_put']
    _layout_contour_overlay = _deps['_layout_contour_overlay']
    _layout_disk_dir = _deps['_layout_disk_dir']
    _new_layout_state = _deps['_new_layout_state']
    _pil_image = _deps['_pil_image']
    _session_id_from_state = _deps['_session_id_from_state']
    cv2 = _deps['cv2']
    json = _deps['json']
    np = _deps['np']
    time = _deps['time']
    uuid = _deps['uuid']
    session_id = _session_id_from_state(session_state)
    layout_id = f"layout_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = _layout_disk_dir(session_id, layout_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    image = _pil_image(source_image)
    mask_bool = np.asarray(mask, dtype=bool)
    image_path = out_dir / "source_image.png"
    mask_path = out_dir / "source_mask.png"
    contour_path = out_dir / "contours.json"
    overlay_path = out_dir / "contour_overlay.png"
    meta_path = out_dir / "layout_meta.json"
    image.save(image_path)
    cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)
    overlay = _layout_contour_overlay(image, mask_bool, contours)
    overlay.save(overlay_path)
    payload = {
        "layout_id": layout_id,
        "session_id": session_id,
        "image_size": [int(image.width), int(image.height)],
        "mask_semantics": {"foreground": 1, "background": 0},
        "foreground_pixels": int(mask_bool.sum()),
        "foreground_ratio": float(mask_bool.mean()),
        "binarize_params": params,
        "contours": contours,
    }
    with contour_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cached = _layout_cache_put(
        session_id,
        layout_id,
        image,
        mask_bool,
        contours,
        params,
        mask_path=mask_path,
        contour_json_path=contour_path,
        overlay_path=overlay_path,
        layout_meta_path=meta_path,
    )
    state = _new_layout_state(session_id)
    state.update(
        {
            "layout_id": layout_id,
            "enabled": True,
            "region_mode": str(params.get("region_mode") or "all"),
            "source_width": int(image.width),
            "source_height": int(image.height),
            "pivot_x": float(cached["pivot_xy"][0]),
            "pivot_y": float(cached["pivot_xy"][1]),
            "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
            "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        }
    )
    return state, str(mask_path), str(contour_path), overlay


def _layout_mask_to_editor_image_impl(_deps, mask):
    Image = _deps['Image']
    np = _deps['np']
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_editor_empty_impl(_deps, image_state, status):
    _data_url = _deps['_data_url']
    _workspace = _deps['_workspace']
    base_url = ""
    target_width = 0
    target_height = 0
    if isinstance(image_state, dict) and image_state.get("image_id"):
        try:
            image = _workspace(image_state)["image"]
            base_url = _data_url(image)
            target_width, target_height = int(image.width), int(image.height)
        except Exception:
            pass
    return {
        "enabled": False,
        "base_image": base_url,
        "mask_image": "",
        "transform": None,
        "target_width": target_width,
        "target_height": target_height,
        "source_width": 0,
        "source_height": 0,
        "foreground_bbox_xyxy": None,
        "status": status,
    }


def _layout_editor_payload_impl(_deps, image_state, layout_state, status):
    _LAYOUT_PROMPT_SCOPE_REGION_CLASS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_CLASS']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _data_url = _deps['_data_url']
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_mask_to_editor_image = _deps['_layout_mask_to_editor_image']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_prompt_display_mask = _deps['_layout_prompt_display_mask']
    _layout_prompt_group_payload = _deps['_layout_prompt_group_payload']
    _layout_tx = _deps['_layout_tx']
    _workspace = _deps['_workspace']
    copy = _deps['copy']
    np = _deps['np']
    if not layout_state or not layout_state.get("layout_id"):
        return _layout_editor_empty(image_state, status or "请先加载或生成版图 mask")
    try:
        cached = _layout_cache_get(layout_state)
        source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
        if source_mask.ndim != 2:
            raise ValueError("source_mask is not 2D")
        base_url = ""
        target_width = int(cached.get("source_width") or source_mask.shape[1])
        target_height = int(cached.get("source_height") or source_mask.shape[0])
        image_id = layout_state.get("image_id")
        target_hash = cached.get("target_image_sha256")
        if isinstance(image_state, dict) and image_state.get("image_id"):
            image = _workspace(image_state)["image"]
            base_url = _data_url(image)
            target_width, target_height = int(image.width), int(image.height)
            image_id = image_state.get("image_id")
            target_hash = image_state.get("target_image_sha256") or _layout_tx.image_pixel_sha256(image)
        pivot = cached.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(cached.get("foreground_bbox_xyxy"))
        state = dict(layout_state or {})
        display_mask = source_mask
        prompt_display_status = None
        try:
            display_mask, prompt_display_status = _layout_prompt_display_mask(
                state,
                source_mask,
            )
        except Exception as exc:
            if state.get("prompt_mask_scope") in {
                _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
            }:
                display_mask = np.zeros_like(source_mask, dtype=bool)
                prompt_display_status = f"Label mask 预览不可用：{exc}"
        if all(k in state and state.get(k) is not None for k in ("center_x", "center_y", "pivot_x", "pivot_y")):
            center_x = float(state.get("center_x"))
            center_y = float(state.get("center_y"))
            pivot_xy = [float(state.get("pivot_x")), float(state.get("pivot_y"))]
        else:
            center_x = float(target_width) / 2.0 + float(state.get("tx") or 0.0)
            center_y = float(target_height) / 2.0 + float(state.get("ty") or 0.0)
            pivot_xy = pivot
        transform = _layout_tx.make_layout_transform_v2(
            session_id=str(state.get("session_id") or cached.get("session_id") or "default"),
            layout_id=str(state.get("layout_id")),
            image_id=str(image_id or ""),
            target_size=(target_width, target_height),
            source_mask=source_mask,
            center_x=center_x,
            center_y=center_y,
            pivot_xy=pivot_xy,
            scale=float(state.get("scale") or 1.0),
            rotation_deg=float(state.get("rotation_deg") or 0.0),
            preview_alpha=_layout_preview_alpha(state),
            revision=int(state.get("revision") or cached.get("committed_revision") or 0),
            source_mask_pixel_sha256=cached.get("source_mask_pixel_sha256"),
            target_image_sha256=target_hash,
        )
        transform = _layout_tx.transform_with_derived_fields(transform, (target_width, target_height))
        editor_status = status or "版图编辑器已加载：拖动 mask 平移，滚轮缩放，拖动圆形手柄旋转。"
        if prompt_display_status:
            editor_status = (
                f"{editor_status}\n{prompt_display_status}"
                if status else prompt_display_status
            )
        payload = {
            "enabled": bool(state.get("enabled", True)),
            "base_image": base_url,
            "mask_image": _data_url(_layout_mask_to_editor_image(display_mask)),
            "transform": copy.deepcopy(transform),
            "target_width": target_width,
            "target_height": target_height,
            "source_width": int(cached.get("source_width") or source_mask.shape[1]),
            "source_height": int(cached.get("source_height") or source_mask.shape[0]),
            "foreground_bbox_xyxy": copy.deepcopy(cached.get("foreground_bbox_xyxy")),
            "status": editor_status,
        }
        if state.get("prompt_mask_scope") == _LAYOUT_PROMPT_SCOPE_REGION_LABELS:
            group_payload = _layout_prompt_group_payload(
                state,
                source_mask,
                transform,
                (target_width, target_height),
            )
            payload.update(group_payload)
            active_group_id = group_payload["group_intent"][
                "active_group_id"
            ]
            active_item = next(
                item
                for item in group_payload["group_intent"]["transforms"]
                if item["group_id"] == active_group_id
            )
            payload["transform"] = copy.deepcopy(active_item["transform"])
        return payload
    except Exception as exc:
        return _layout_editor_empty(image_state, status or f"版图编辑器不可用：{exc}")


def _layout_editor_transform_impl(_deps, editor_payload):
    if not isinstance(editor_payload, dict):
        return None
    transform = editor_payload.get("transform")
    return transform if isinstance(transform, dict) else None


def _layout_group_control_values_impl(_deps, transform, target_size):
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_tx = _deps['_layout_tx']
    tx, ty = _layout_tx.derive_legacy_tx_ty(transform, target_size)
    return (
        float(tx),
        float(ty),
        float(transform.get("scale") or 1.0),
        float(transform.get("rotation_deg") or 0.0),
        _layout_preview_alpha(transform),
    )


def _commit_layout_group_transforms_impl(_deps, image_state, layout_state, editor_payload, numeric_override, reset_active):
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_editor_transform = _deps['_layout_editor_transform']
    _layout_group_transform = _deps['_layout_group_transform']
    _layout_prompt_group_data = _deps['_layout_prompt_group_data']
    _layout_prompt_group_id = _deps['_layout_prompt_group_id']
    _layout_regions = _deps['_layout_regions']
    _layout_tx = _deps['_layout_tx']
    copy = _deps['copy']
    np = _deps['np']
    state = dict(layout_state or {})
    if state.get("prompt_mask_scope") != _LAYOUT_PROMPT_SCOPE_REGION_LABELS:
        raise ValueError("当前不是 Label 独立变换模式")
    cached = _layout_cache_get(state)
    source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
    if source_mask.ndim != 2:
        raise ValueError("layout source_mask must be a 2D binary mask")
    base_state = dict(state)
    base_state["prompt_mask_scope"] = _LAYOUT_PROMPT_SCOPE_FULL
    base_payload = _layout_editor_payload(image_state, base_state)
    base_transform = _layout_editor_transform(base_payload)
    if not base_transform:
        raise ValueError("完整 mask transform 不可用")
    target_size = (
        int(base_payload.get("target_width") or 0),
        int(base_payload.get("target_height") or 0),
    )
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("目标图像尺寸无效")
    state["image_id"] = base_transform.get("image_id")
    state["target_image_sha256"] = base_transform.get(
        "target_image_sha256"
    )
    document, decoded, signature = _layout_prompt_group_data(
        state,
        source_mask,
    )
    intent = (
        editor_payload.get("group_intent")
        if isinstance(editor_payload, dict)
        else None
    )
    if not isinstance(intent, dict):
        raise ValueError("Canvas 缺少 Label group_intent")
    if intent.get("selection_signature") != signature:
        raise ValueError("Canvas Label 选择签名已过期")
    active_group_id = str(intent.get("active_group_id") or "")
    expected_group_ids = [
        _layout_prompt_group_id(record["region_id"])
        for record, _ in decoded
    ]
    if active_group_id not in expected_group_ids:
        raise ValueError("Canvas active Label 无效")
    changed_group_values = intent.get("changed_group_ids")
    if changed_group_values is None:
        changed_group_values = []
    if not isinstance(changed_group_values, list):
        raise ValueError("Canvas changed Label 集合无效")
    changed_group_ids = [str(value) for value in changed_group_values]
    if (
        len(set(changed_group_ids)) != len(changed_group_ids)
        or not set(changed_group_ids).issubset(expected_group_ids)
    ):
        raise ValueError("Canvas changed Label 集合重复或无效")
    incoming_items = intent.get("transforms")
    if not isinstance(incoming_items, list):
        raise ValueError("Canvas Label transforms 无效")
    incoming_by_id = {}
    for item in incoming_items:
        if not isinstance(item, dict):
            raise ValueError("Canvas Label transform item 无效")
        group_id = str(item.get("group_id") or "")
        transform = item.get("transform")
        if (
            group_id in incoming_by_id
            or not isinstance(transform, dict)
        ):
            raise ValueError("Canvas Label transform 重复或无效")
        incoming_by_id[group_id] = transform
    if set(incoming_by_id) != set(expected_group_ids):
        raise ValueError("Canvas Label transform 集合与服务端不一致")
    set_revision = intent.get("transform_set_revision")
    if (
        isinstance(set_revision, bool)
        or not isinstance(set_revision, (int, float))
        or int(set_revision) < int(
            state.get("prompt_transform_set_revision") or 0
        )
    ):
        raise ValueError("Canvas Label transform set revision 已过期")
    previous = state.get("prompt_group_transforms")
    if not isinstance(previous, dict):
        previous = {}
    backend_updates_active = numeric_override is not None or reset_active
    authoritative = {}
    transformed_groups = {}
    transformed_union = np.zeros(
        (target_size[1], target_size[0]),
        dtype=bool,
    )
    for record, group_mask in decoded:
        group_id = _layout_prompt_group_id(record["region_id"])
        incoming = copy.deepcopy(incoming_by_id[group_id])
        previous_transform = previous.get(group_id)
        previous_revision = (
            int(previous_transform.get("revision") or 0)
            if isinstance(previous_transform, dict)
            else 0
        )
        incoming_revision = incoming.get("revision", 0)
        if (
            isinstance(incoming_revision, bool)
            or not isinstance(incoming_revision, (int, float))
            or not float(incoming_revision).is_integer()
            or int(incoming_revision) < 0
        ):
            raise ValueError(f"{group_id} transform revision 无效")
        if int(incoming_revision) < previous_revision:
            if group_id != active_group_id and isinstance(previous_transform, dict):
                incoming = copy.deepcopy(previous_transform)
                incoming_revision = previous_revision
            else:
                raise ValueError(f"{group_id} transform revision 已过期")
        accepts_changed_non_active = (
            group_id != active_group_id
            and group_id in changed_group_ids
            and int(incoming_revision) > previous_revision
        )
        if (
            group_id != active_group_id
            and isinstance(previous_transform, dict)
            and not accepts_changed_non_active
        ):
            incoming = copy.deepcopy(previous_transform)
            incoming_revision = previous_revision
        if reset_active and group_id == active_group_id:
            incoming = None
        elif (
            numeric_override is not None
            and group_id == active_group_id
        ):
            tx, ty, scale, rotation, alpha = numeric_override
            incoming.update(
                {
                    "center_x": target_size[0] / 2.0 + float(tx),
                    "center_y": target_size[1] / 2.0 + float(ty),
                    "scale": float(scale),
                    "rotation_deg": float(rotation),
                    "preview_alpha": float(alpha),
                }
            )
        transform, _ = _layout_group_transform(
            state,
            base_transform,
            record,
            group_mask,
            target_size,
            values=incoming,
        )
        if group_id == active_group_id and backend_updates_active:
            transform["revision"] = max(
                int(transform.get("revision") or 0),
                previous_revision,
            ) + 1
        elif (
            group_id != active_group_id
            and isinstance(previous_transform, dict)
            and not accepts_changed_non_active
        ):
            transform["revision"] = previous_revision
        else:
            transform["revision"] = int(transform.get("revision") or 0)
        authoritative[group_id] = transform
        transformed = _layout_tx.warp_layout_mask(
            group_mask,
            transform["matrix_2x3"],
            target_size,
        )
        transformed_groups[group_id] = transformed
        transformed_union = np.logical_or(
            transformed_union,
            transformed,
        )
    next_set_revision = max(
        int(set_revision),
        int(state.get("prompt_transform_set_revision") or 0),
    ) + (1 if backend_updates_active else 0)
    labels = [
        _layout_regions.region_label(record)
        for record, _ in decoded
    ]
    region_ids = [
        int(record["region_id"])
        for record, _ in decoded
    ]
    snapshot = {
        "selection_signature": signature,
        "regions_revision": int(document.get("regions_revision") or 0),
        "transform_set_revision": next_set_revision,
        "active_group_id": active_group_id,
        "region_ids": region_ids,
        "labels": labels,
        "target_image_sha256": base_transform.get(
            "target_image_sha256"
        ),
        "transforms": copy.deepcopy(authoritative),
    }
    with _LAYOUT_CACHE_LOCK:
        cached["prompt_group_snapshot"] = copy.deepcopy(snapshot)
        cached["prompt_group_transformed_masks"] = {
            group_id: mask.copy()
            for group_id, mask in transformed_groups.items()
        }
        cached["prompt_group_transformed_union"] = (
            transformed_union.copy()
        )
    state.update(
        {
            "prompt_labels": labels,
            "prompt_region_ids": region_ids,
            "prompt_group_transforms": copy.deepcopy(authoritative),
            "prompt_active_group_id": active_group_id,
            "prompt_selection_signature": signature,
            "prompt_transform_set_revision": next_set_revision,
        }
    )
    return (
        state,
        decoded,
        snapshot,
        transformed_union,
        authoritative[active_group_id],
    )


def _validate_layout_group_transform_snapshot_impl(_deps, layout_state, snapshot):
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_get = _deps['_layout_cache_get']
    copy = _deps['copy']
    with _LAYOUT_CACHE_LOCK:
        cached = _layout_cache_get(layout_state)
        current = copy.deepcopy(cached.get("prompt_group_snapshot"))
    if not isinstance(current, dict) or current != snapshot:
        raise ValueError("Label transform 在批量预测期间发生变化")


def _sync_layout_controls_from_editor_impl(_deps, layout_state, editor_payload):
    _layout_editor_transform = _deps['_layout_editor_transform']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_tx = _deps['_layout_tx']
    gr = _deps['gr']
    np = _deps['np']
    state = dict(layout_state or {})
    transform = _layout_editor_transform(editor_payload)
    if not transform:
        return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "版图编辑器还没有 transform payload"
    try:
        if state.get("layout_id") and transform.get("layout_id") and str(state.get("layout_id")) != str(transform.get("layout_id")):
            raise ValueError("canvas transform belongs to a different layout mask")
        if state.get("session_id") and transform.get("session_id") and str(state.get("session_id")) != str(transform.get("session_id")):
            raise ValueError("canvas transform belongs to a different session")

        payload = editor_payload if isinstance(editor_payload, dict) else {}
        target_w = int(payload.get("target_width") or 0)
        target_h = int(payload.get("target_height") or 0)
        center_x = float(transform.get("center_x", state.get("center_x") or 0.0))
        center_y = float(transform.get("center_y", state.get("center_y") or 0.0))
        if target_w > 0 and target_h > 0:
            tx, ty = _layout_tx.derive_legacy_tx_ty({"center_x": center_x, "center_y": center_y}, (target_w, target_h))
        else:
            tx = float(transform.get("tx", state.get("tx") or 0.0))
            ty = float(transform.get("ty", state.get("ty") or 0.0))

        state.update({
            "enabled": bool(payload.get("enabled", True)),
            "transform_version": 2,
            "image_id": transform.get("image_id") or state.get("image_id"),
            "center_x": center_x,
            "center_y": center_y,
            "pivot_x": float(transform.get("pivot_x", state.get("pivot_x") or 0.0)),
            "pivot_y": float(transform.get("pivot_y", state.get("pivot_y") or 0.0)),
            "scale": float(np.clip(float(transform.get("scale", state.get("scale") or 1.0)), 0.01, 20.0)),
            "rotation_deg": float(_layout_tx.normalize_rotation_deg(float(transform.get("rotation_deg", state.get("rotation_deg") or 0.0)))),
            "preview_alpha": float(
                np.clip(
                    _layout_preview_alpha(
                        transform
                        if transform.get("preview_alpha") is not None
                        else state
                    ),
                    0.0,
                    1.0,
                )
            ),
            "revision": int(float(transform.get("revision", state.get("revision") or 0))),
            "source_mask_pixel_sha256": transform.get("source_mask_pixel_sha256") or state.get("source_mask_pixel_sha256"),
            "target_image_sha256": transform.get("target_image_sha256") or state.get("target_image_sha256"),
            "tx": float(tx),
            "ty": float(ty),
        })
        info = f"Canvas 变换已同步到数值控件：tx={tx:.1f}, ty={ty:.1f}, 缩放={state['scale']:.3f}, 旋转={state['rotation_deg']:.1f}"
        return state, bool(state.get("enabled", True)), float(tx), float(ty), float(state.get("scale") or 1.0), float(state.get("rotation_deg") or 0.0), _layout_preview_alpha(state), info
    except Exception as exc:
        return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Canvas 变换同步失败：{exc}"


def _sync_layout_controls_from_editor_with_prompt_epoch_impl(_deps, layout_state, editor_payload):
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _commit_layout_group_transforms = _deps['_commit_layout_group_transforms']
    _layout_editor_transform = _deps['_layout_editor_transform']
    _layout_group_control_values = _deps['_layout_group_control_values']
    _sync_layout_controls_from_editor = _deps['_sync_layout_controls_from_editor']
    gr = _deps['gr']
    _advance_layout_prompt_epoch(layout_state=layout_state)
    if (
        isinstance(layout_state, dict)
        and layout_state.get("prompt_mask_scope")
        == _LAYOUT_PROMPT_SCOPE_REGION_LABELS
    ):
        state = dict(layout_state)
        try:
            transform = _layout_editor_transform(editor_payload) or {}
            image_state = {
                "image_id": transform.get("image_id")
                or state.get("image_id"),
                "session_id": state.get("session_id"),
                "width": int((editor_payload or {}).get("target_width") or 0),
                "height": int((editor_payload or {}).get("target_height") or 0),
                "target_image_sha256": transform.get(
                    "target_image_sha256"
                ),
            }
            state, _, _, _, active_transform = (
                _commit_layout_group_transforms(
                    image_state,
                    state,
                    editor_payload,
                )
            )
            tx, ty, scale, rotation, alpha = (
                _layout_group_control_values(
                    active_transform,
                    (image_state["width"], image_state["height"]),
                )
            )
            label = active_transform.get("label") or "Label"
            info = f"已激活 {label}；下方数值控件仅修改该 Label"
            return state, True, tx, ty, scale, rotation, alpha, info
        except Exception as exc:
            return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Label Canvas 同步失败：{exc}"
    return _sync_layout_controls_from_editor(layout_state, editor_payload)


def _run_layout_mask_page_impl(_deps, session_state, image_state, input_image, threshold, invert, open_kernel, close_kernel, min_component_area, region_mode, morph_pixels):
    _layout_editor_empty = _deps['_layout_editor_empty']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_mask_to_preview = _deps['_layout_mask_to_preview']
    _new_layout_state = _deps['_new_layout_state']
    _save_layout_mask_files = _deps['_save_layout_mask_files']
    _session_id_from_state = _deps['_session_id_from_state']
    try:
        image, mask, contours, params = _compute_layout_mask_draft_impl(
            _deps,
            input_image,
            threshold,
            invert,
            open_kernel,
            close_kernel,
            min_component_area,
            region_mode,
            morph_pixels,
        )
        if not mask.any():
            raise ValueError("Binary mask is empty; lower threshold or check invert")
        state, mask_path, contour_path, overlay = _save_layout_mask_files(session_state, image, mask, contours, params)
        info = (
            f"版图 mask 已生成：{state['layout_id']}\n"
            f"session: {state.get('session_id')}\n"
            f"size: {image.width}x{image.height}\n"
            f"foreground pixels: {int(mask.sum())} ({mask.mean():.4f})\n"
            f"contours: {len(contours)}\n"
            f"source_mask_pixel_sha256: {state.get('source_mask_pixel_sha256')}\n"
            f"mask: {mask_path}\ncontours: {contour_path}"
        )
        editor_payload = _layout_editor_payload(image_state, state, "版图 mask 已生成；切换到版图 mask 提示分割后可拖动、缩放和旋转。")
        return state, editor_payload, image, _layout_mask_to_preview(mask), overlay, mask_path, contour_path, info
    except Exception as exc:
        info = f"版图 mask 生成失败：{exc}"
        state = _new_layout_state(_session_id_from_state(session_state))
        return state, _layout_editor_empty(image_state, info), None, None, None, None, None, info


def _run_layout_mask_page_with_downloads_impl(_deps, session_state, image_state, input_image, threshold, invert, open_kernel, close_kernel, min_component_area, region_mode, morph_pixels):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _publish_layout_downloads = _deps['_publish_layout_downloads']
    _run_layout_mask_page = _deps['_run_layout_mask_page']
    _advance_layout_prompt_epoch(
        image_state=image_state,
        session_state=session_state,
    )
    result = list(
        _run_layout_mask_page(
            session_state,
            image_state,
            input_image,
            threshold,
            invert,
            open_kernel,
            close_kernel,
            min_component_area,
            region_mode,
            morph_pixels,
        )
    )
    internal_mask_path, internal_contour_path = result[5], result[6]
    if not internal_mask_path or not internal_contour_path:
        return tuple(result)
    try:
        public_mask_path, public_contour_path = _publish_layout_downloads(
            internal_mask_path,
            internal_contour_path,
        )
    except Exception as exc:
        result[5] = None
        result[6] = None
        safe_info = str(result[7]).split("\nmask:", 1)[0]
        result[7] = f"{safe_info}\n\u4e0b\u8f7d\u526f\u672c\u751f\u6210\u5931\u8d25\uff1a{exc}"
        return tuple(result)
    result[5] = public_mask_path
    result[6] = public_contour_path
    result[7] = (
        str(result[7])
        .replace(str(internal_mask_path), public_mask_path)
        .replace(str(internal_contour_path), public_contour_path)
    )
    return tuple(result)


def _save_current_layout_mask_impl(_deps, layout_state):
    _layout_cache_get = _deps['_layout_cache_get']
    _publish_layout_downloads = _deps['_publish_layout_downloads']
    try:
        cached = _layout_cache_get(layout_state)
        mask_path, contour_path = _publish_layout_downloads(cached.get("mask_path"), cached.get("contour_json_path"))
        return (
            mask_path,
            contour_path,
            f"Saved current layout mask: {layout_state.get('layout_id')}",
        )
    except Exception as exc:
        return None, None, f"保存当前版图 mask 失败：{exc}"


def _clear_current_layout_mask_impl(_deps, image_state, layout_state):
    _clear_layout_cache = _deps['_clear_layout_cache']
    _layout_editor_empty = _deps['_layout_editor_empty']
    _new_layout_state = _deps['_new_layout_state']
    session_id = layout_state.get("session_id") if isinstance(layout_state, dict) else None
    _clear_layout_cache(layout_state)
    state = _new_layout_state(session_id)
    return state, _layout_editor_empty(image_state, "当前版图 mask 已清除"), None, None, None, None, None, "当前版图 mask 已清除"


def _clear_current_layout_mask_with_prompt_epoch_impl(_deps, image_state, layout_state):
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _clear_current_layout_mask = _deps['_clear_current_layout_mask']
    _advance_layout_prompt_epoch(image_state, layout_state)
    return _clear_current_layout_mask(image_state, layout_state)


def _layout_numeric_controls_changed_impl(_deps, layout_state, tx, ty, scale, rotation_deg, preview_alpha, tol):
    state = layout_state if isinstance(layout_state, dict) else {}

    def changed(field, value, default):
        if value is None:
            return False
        try:
            current = float(value)
            previous = float(state.get(field) if state.get(field) is not None else default)
        except (TypeError, ValueError):
            return False
        return abs(current - previous) > tol

    return any(
        [
            changed("tx", tx, 0.0),
            changed("ty", ty, 0.0),
            changed("scale", scale, 1.0),
            changed("rotation_deg", rotation_deg, 0.0),
            changed("preview_alpha", preview_alpha, 0.35),
        ]
    )


def _commit_layout_transform_impl(_deps, image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload, prefer_numeric):
    _LAYOUT_CACHE_LOCK = _deps['_LAYOUT_CACHE_LOCK']
    _layout_cache_get = _deps['_layout_cache_get']
    _layout_editor_transform = _deps['_layout_editor_transform']
    _layout_numeric_controls_changed = _deps['_layout_numeric_controls_changed']
    _layout_tx = _deps['_layout_tx']
    _new_layout_state = _deps['_new_layout_state']
    _workspace = _deps['_workspace']
    _write_layout_meta = _deps['_write_layout_meta']
    copy = _deps['copy']
    np = _deps['np']
    ws = _workspace(image_state)
    image = ws["image"]
    target_size = (int(image.width), int(image.height))
    target_hash = image_state.get("target_image_sha256") or ws.get("target_image_sha256") or _layout_tx.image_pixel_sha256(image)
    state = dict(layout_state or _new_layout_state(image_state.get("session_id")))
    if prefer_numeric is None:
        prefer_numeric = _layout_numeric_controls_changed(state, tx, ty, scale, rotation_deg, preview_alpha)
    incoming = None if prefer_numeric else _layout_editor_transform(transform_payload)
    if incoming:
        state.update({
            "center_x": incoming.get("center_x", state.get("center_x")),
            "center_y": incoming.get("center_y", state.get("center_y")),
            "pivot_x": incoming.get("pivot_x", state.get("pivot_x")),
            "pivot_y": incoming.get("pivot_y", state.get("pivot_y")),
            "scale": incoming.get("scale", state.get("scale")),
            "rotation_deg": incoming.get("rotation_deg", state.get("rotation_deg")),
            "preview_alpha": incoming.get("preview_alpha", state.get("preview_alpha")),
            "revision": incoming.get("revision", state.get("revision")),
        })
    if not state.get("layout_id"):
        raise ValueError("请先加载或生成版图 mask")
    if state.get("session_id") and image_state.get("session_id") and str(state.get("session_id")) != str(image_state.get("session_id")):
        raise ValueError("版图 mask 属于另一个浏览器会话")
    cached = _layout_cache_get(state)
    source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
    if source_mask.ndim != 2:
        raise ValueError("layout source_mask must be a 2D binary mask")
    source_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    if cached.get("source_mask_pixel_sha256") and cached.get("source_mask_pixel_sha256") != source_hash:
        raise ValueError("source mask pixel hash mismatch; refusing transform")
    if incoming:
        if incoming.get("session_id") and str(incoming.get("session_id")) != str(state.get("session_id")):
            raise ValueError("frontend transform session_id mismatch")
        if incoming.get("layout_id") and str(incoming.get("layout_id")) != str(state.get("layout_id")):
            raise ValueError("frontend transform layout_id mismatch")
        if incoming.get("image_id") and image_state.get("image_id") and str(incoming.get("image_id")) != str(image_state.get("image_id")):
            raise ValueError("frontend transform image_id mismatch")
        if incoming.get("source_mask_pixel_sha256") and incoming.get("source_mask_pixel_sha256") != source_hash:
            raise ValueError("frontend transform source mask hash mismatch")
        if incoming.get("target_image_sha256") and incoming.get("target_image_sha256") != target_hash:
            raise ValueError("frontend transform target image hash mismatch")
    payload_revision = int(float(state.get("revision") or 0))
    with _LAYOUT_CACHE_LOCK:
        committed = int(cached.get("committed_revision") or 0)
        if payload_revision < committed and cached.get("target_image_sha256") == target_hash:
            raise ValueError(f"layout transform revision is stale: payload={payload_revision}, committed={committed}")
        pivot = cached.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(cached.get("foreground_bbox_xyxy"))
        if incoming:
            pivot_xy = [float(state.get("pivot_x") if state.get("pivot_x") is not None else pivot[0]), float(state.get("pivot_y") if state.get("pivot_y") is not None else pivot[1])]
            center_x = float(state.get("center_x") if state.get("center_x") is not None else target_size[0] / 2.0)
            center_y = float(state.get("center_y") if state.get("center_y") is not None else target_size[1] / 2.0)
        else:
            pivot_xy = pivot
            center_x = target_size[0] / 2.0 + float(tx or 0.0)
            center_y = target_size[1] / 2.0 + float(ty or 0.0)
        if incoming:
            scale_source = state.get("scale") if state.get("scale") is not None else scale
            rotation_source = state.get("rotation_deg") if state.get("rotation_deg") is not None else rotation_deg
            alpha_source = state.get("preview_alpha") if state.get("preview_alpha") is not None else preview_alpha
        else:
            scale_source = scale
            rotation_source = rotation_deg
            alpha_source = preview_alpha
        scale_value = float(np.clip(float(scale_source if scale_source is not None else 1.0), 0.01, 20.0))
        rotation_value = float(rotation_source if rotation_source is not None else 0.0)
        alpha_value = float(np.clip(float(alpha_source if alpha_source is not None else 0.35), 0.0, 1.0))
        revision = max(payload_revision, committed) + 1
        transform = _layout_tx.make_layout_transform_v2(
            session_id=str(state.get("session_id") or cached.get("session_id") or image_state.get("session_id") or "default"),
            layout_id=str(state.get("layout_id")),
            image_id=str(image_state.get("image_id")),
            target_size=target_size,
            source_mask=source_mask,
            center_x=center_x,
            center_y=center_y,
            pivot_xy=pivot_xy,
            scale=scale_value,
            rotation_deg=rotation_value,
            preview_alpha=alpha_value,
            revision=revision,
            source_mask_pixel_sha256=source_hash,
            target_image_sha256=target_hash,
        )
        transform = _layout_tx.transform_with_derived_fields(transform, target_size)
        transformed = _layout_tx.warp_layout_mask(source_mask, transform["matrix_2x3"], target_size)
        cached["target_image_sha256"] = target_hash
        cached["committed_revision"] = revision
        cached["backend_transform"] = copy.deepcopy(transform)
        cached["matrix_2x3"] = copy.deepcopy(transform["matrix_2x3"])
        cached["transformed_mask"] = transformed
        _write_layout_meta(cached)
    state.update(copy.deepcopy(transform))
    state.update({
        "enabled": bool(enabled),
        "region_mode": state.get("region_mode") or cached.get("binarize_params", {}).get("region_mode") or "all",
        "source_width": int(cached.get("source_width") or source_mask.shape[1]),
        "source_height": int(cached.get("source_height") or source_mask.shape[0]),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
    })
    return state, transformed, copy.deepcopy(transform)


def _transform_layout_mask_impl(_deps, layout_state, target_width, target_height):
    raise RuntimeError("_transform_layout_mask is deprecated; use _commit_layout_transform(image_state, ...) so target image hash and revision are validated")


def _layout_mask_to_overlay_impl(_deps, base_image, mask, alpha):
    Image = _deps['Image']
    _pil_image = _deps['_pil_image']
    cv2 = _deps['cv2']
    np = _deps['np']
    base = np.asarray(_pil_image(base_image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != base.shape[:2]:
        raise ValueError("layout mask and target image sizes do not match")
    fill = base.copy()
    fill[mask] = (0, 255, 130)
    return Image.fromarray(cv2.addWeighted(fill, float(alpha), base, 1.0 - float(alpha), 0))


def _update_layout_preview_impl(_deps, image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload):
    _commit_layout_transform = _deps['_commit_layout_transform']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_preview_alpha = _deps['_layout_preview_alpha']
    _layout_state_summary = _deps['_layout_state_summary']
    _new_layout_state = _deps['_new_layout_state']
    _workspace_image = _deps['_workspace_image']
    gr = _deps['gr']
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 overlay 只在版图 mask 提示分割模式可用")
        state, transformed, _ = _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=editor_payload)
        info = (
            "后端 warpAffine 已更新版图预览。\n"
            + _layout_state_summary(state)
            + f"\ntransformed mask: {transformed.shape[1]}x{transformed.shape[0]}, foreground={int(transformed.sum())}"
        )
        workspace = _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None, layout_state=state)
        editor = _layout_editor_payload(image_state, state, "后端权威 overlay 已返回，Canvas 变换已校正。")
        return state, workspace, editor, bool(state.get("enabled")), float(state.get("tx") or 0.0), float(state.get("ty") or 0.0), float(state.get("scale") or 1.0), float(state.get("rotation_deg") or 0.0), _layout_preview_alpha(state), info
    except Exception as exc:
        state = layout_state or _new_layout_state(image_state.get("session_id") if isinstance(image_state, dict) else None)
        return state, gr.update(), _layout_editor_payload(image_state, state, f"版图预览更新失败：{exc}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"版图预览更新失败：{exc}"


def _update_layout_preview_with_groups_impl(_deps, image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload):
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _commit_layout_group_transforms = _deps['_commit_layout_group_transforms']
    _is_layout_mask_mode = _deps['_is_layout_mask_mode']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_group_control_values = _deps['_layout_group_control_values']
    _update_layout_preview = _deps['_update_layout_preview']
    _workspace_image = _deps['_workspace_image']
    gr = _deps['gr']
    if (
        not isinstance(layout_state, dict)
        or layout_state.get("prompt_mask_scope")
        != _LAYOUT_PROMPT_SCOPE_REGION_LABELS
    ):
        return _update_layout_preview(
            image_state,
            pcs_state,
            pvs_state,
            mode,
            layout_state,
            enabled,
            tx,
            ty,
            scale,
            rotation_deg,
            preview_alpha,
            editor_payload,
        )
    state = dict(layout_state)
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 overlay 只在版图 mask 提示分割模式可用")
        _advance_layout_prompt_epoch(image_state, state)
        state, _, _, transformed, active_transform = (
            _commit_layout_group_transforms(
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
        )
        state["enabled"] = bool(enabled)
        active_label = active_transform.get("label") or "Label"
        values = _layout_group_control_values(
            active_transform,
            (
                int(image_state.get("width") or 0),
                int(image_state.get("height") or 0),
            ),
        )
        info = (
            f"已更新 {active_label}；其他 Label transform 保持不变。\n"
            f"selected Label union foreground={int(transformed.sum())}"
        )
        workspace = _workspace_image(
            image_state,
            pcs_state,
            pvs_state,
            mode,
            prompt_state=None,
            layout_state=state,
        )
        editor = _layout_editor_payload(image_state, state, info)
        return (
            state,
            workspace,
            editor,
            bool(state.get("enabled")),
            *values,
            info,
        )
    except Exception as exc:
        info = f"Label 预览更新失败：{exc}"
        return (
            state,
            gr.update(),
            _layout_editor_payload(image_state, state, info),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            info,
        )


def _reset_layout_controls_impl(_deps, image_state, layout_state):
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_state_summary = _deps['_layout_state_summary']
    _new_layout_state = _deps['_new_layout_state']
    state = dict(layout_state or _new_layout_state())
    state.update({"enabled": bool(state.get("layout_id")), "tx": 0.0, "ty": 0.0, "scale": 1.0, "rotation_deg": 0.0, "preview_alpha": 0.35})
    if isinstance(image_state, dict) and image_state.get("width") and image_state.get("height"):
        state["center_x"] = float(image_state.get("width")) / 2.0
        state["center_y"] = float(image_state.get("height")) / 2.0
    info = "版图变换控件已重置。\n" + _layout_state_summary(state)
    return state, True if state.get("layout_id") else False, 0.0, 0.0, 1.0, 0.0, 0.35, _layout_editor_payload(image_state, state, "Canvas 变换已重置。"), info


def _reset_layout_controls_with_prompt_epoch_impl(_deps, image_state, layout_state):
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS = _deps['_LAYOUT_PROMPT_SCOPE_REGION_LABELS']
    _advance_layout_prompt_epoch = _deps['_advance_layout_prompt_epoch']
    _commit_layout_group_transforms = _deps['_commit_layout_group_transforms']
    _layout_editor_payload = _deps['_layout_editor_payload']
    _layout_group_control_values = _deps['_layout_group_control_values']
    _reset_layout_controls = _deps['_reset_layout_controls']
    gr = _deps['gr']
    _advance_layout_prompt_epoch(image_state, layout_state)
    if (
        isinstance(layout_state, dict)
        and layout_state.get("prompt_mask_scope")
        == _LAYOUT_PROMPT_SCOPE_REGION_LABELS
    ):
        state = dict(layout_state)
        try:
            editor = _layout_editor_payload(image_state, state)
            state, _, _, _, active_transform = (
                _commit_layout_group_transforms(
                    image_state,
                    state,
                    editor,
                    reset_active=True,
                )
            )
            values = _layout_group_control_values(
                active_transform,
                (
                    int(image_state.get("width") or 0),
                    int(image_state.get("height") or 0),
                ),
            )
            label = active_transform.get("label") or "Label"
            info = f"已重置 {label}；其他 Label transform 保持不变。"
            return (
                state,
                bool(state.get("enabled", True)),
                *values,
                _layout_editor_payload(image_state, state, info),
                info,
            )
        except Exception as exc:
            info = f"Label transform 重置失败：{exc}"
            return (
                state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                _layout_editor_payload(image_state, state, info),
                info,
            )
    return _reset_layout_controls(image_state, layout_state)
