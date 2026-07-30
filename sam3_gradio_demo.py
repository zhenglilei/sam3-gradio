#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
基于 SAM3 的交互式图像分割系统
"""

import time
import io
from pathlib import Path
import tempfile
import json
import uuid
import copy
import hashlib

from sam3_demo.config import (
    MODE_LAYOUT,
    MODE_PCS,
    MODE_PVS,
    _LAYOUT_MASK_MORPH_LIMIT_PX,
    _LAYOUT_PROMPT_CLASS_PREFIX,
    _LAYOUT_PROMPT_LABEL_PREFIX,
    _LAYOUT_PROMPT_SCOPE_FULL,
    _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
    _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
    _MAX_PROMPT_HISTORY_ENTRIES,
    _PUBLIC_DOWNLOAD_TTL_SECONDS,
    _SOURCE_IMAGE_CACHE_MAX_ENTRIES,
    _SOURCE_IMAGE_CACHE_TTL_SECONDS,
    _WORKSPACE_CACHE_MAX_ENTRIES,
    _WORKSPACE_CACHE_TTL_SECONDS,
    coco_dataset_choices,
    coco_dataset_configs,
    coco_eval_scope_full,
    coco_eval_scope_overlap,
    current_dir,
    default_coco_dataset,
    ge1_category_display_order,
    ge1_coco_dir,
    logger,
    o3_coco_dir,
    public_download_dir,
    qiyuan_cache_dir,
    runtime_dir,
    runtime_export_dir,
    runtime_feedback_dir,
    runtime_gradio_dir,
    runtime_layout_dir,
    runtime_layout_region_dir,
    runtime_log_dir,
    runtime_tmp_dir,
)

import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
import layout_transform_utils as _layout_tx
import layout_region_utils as _layout_regions
import image_crop_utils as _image_crop
import public_download_utils as _public_downloads
import template_match_workflow as _template_matching

try:
    from gradio_layout_transform_editor import LayoutTransformEditor
except Exception as exc:
    LayoutTransformEditor = None
    _layout_editor_import_error = exc
else:
    _layout_editor_import_error = None

try:
    from gradio_layout_region_annotator import LayoutRegionAnnotator
except Exception as exc:
    LayoutRegionAnnotator = None
    _layout_region_annotator_import_error = exc
else:
    _layout_region_annotator_import_error = None

try:
    from gradio_image_gesture_overlay import ImageGestureOverlay
except Exception as exc:
    ImageGestureOverlay = None
    _image_gesture_overlay_import_error = exc
else:
    _image_gesture_overlay_import_error = None

try:
    from scripts.layout_image_to_mask import extract_layout_mask as _layout_extract_mask
    _layout_extract_mask_import_error = None
except Exception as exc:
    _layout_extract_mask = None
    _layout_extract_mask_import_error = exc

def _gradio_allowed_paths():
    return [str(public_download_dir.resolve())]


def _gradio_blocked_paths():
    exempt_top_level = {
        public_download_dir.name,
        runtime_dir.name,
        ".gradio",
    }
    blocked = [
        str(path.resolve())
        for path in current_dir.iterdir()
        if path.name not in exempt_top_level
    ]
    exempt_runtime = {runtime_gradio_dir.resolve()}
    blocked.extend(
        str(path.resolve())
        for path in runtime_dir.iterdir()
        if path.resolve() not in exempt_runtime
    )
    return sorted(set(blocked))


def _prune_public_downloads():
    try:
        return _public_downloads.prune_public_downloads(
            public_download_dir,
            max_age_seconds=_PUBLIC_DOWNLOAD_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Cannot prune public downloads: %s", exc)
        return []


def _publish_layout_downloads(mask_path, contour_path):
    _prune_public_downloads()
    export_dir = _public_downloads.publish_files(
        public_download_dir,
        "layout_mask_exports",
        {
            "source_mask.png": mask_path,
            "contours.json": contour_path,
        },
    )
    return str(export_dir / "source_mask.png"), str(export_dir / "contours.json")


def _publish_segmentation_zip(export_dir, zip_name):
    _prune_public_downloads()
    return _public_downloads.publish_zip(
        public_download_dir,
        "pcs_pvs_exports",
        export_dir,
        zip_name,
    )


from sam3_demo.model_runtime import (
    DEVICE,
    FindStage,
    box_ops,
    image_predictor,
    initialize_models,
)

from sam3_demo.segmentation_evaluation import (
    parse_polygon_prompt,
    serialize_polygon_prompt,
    append_polygon_prompt,
    polygon_to_mask,
    draw_polygons,
    safe_stem,
    mask_to_polygons,
    annotation_to_mask,
    mask_bbox_xywh,
    create_prediction_coco_json,
    mask_boundary,
    boundary_band,
    boundary_iou,
    boundary_distances_px,
    hd95_and_chamfer,
    ap_from_scores,
    boundary_ap,
    encode_binary_mask,
    compute_coco_segm_metrics,
    get_coco_dataset_name,
    get_coco_dataset_config,
    load_coco_image_record,
    resolve_uploaded_json_path,
    label_text_variants,
    infer_labelme_category_ids,
    labelme_shape_segmentation,
    load_labelme_annotation_record,
    infer_prompt_category_ids,
    coco_category_display_name,
    sorted_coco_categories,
    rasterize_coco_annotations,
    filter_gt_pairs_by_eval_scope,
    select_predictions_for_gt_masks,
    evaluate_prediction_gt_metrics,
    compare_with_labelme_json,
    compare_with_coco,
    _create_segmentation_export_impl,
)


def create_segmentation_export(
    result_image,
    source_image,
    state,
    prompts,
    coco_dataset,
    coco_image_name,
    coco_split,
    coco_eval_scope,
    annotation_json_file=None,
):
    return _create_segmentation_export_impl(
        result_image,
        source_image,
        state,
        prompts,
        coco_dataset,
        coco_image_name,
        coco_split,
        coco_eval_scope,
        annotation_json_file,
        export_root=runtime_export_dir,
        compare_fn=compare_with_coco,
        publish_zip_fn=_publish_segmentation_zip,
    )



# Legacy mixed point/box/polygon image segmentation flow removed.

# --- PCS/PVS single-workspace override ---
import base64 as _sam3_base64
import threading as _sam3_threading

_PVS_PREDICT_LOCK = _sam3_threading.Lock()
_FEEDBACK_WRITE_LOCK = _sam3_threading.Lock()
_LAYOUT_CACHE = {}
_LAYOUT_CACHE_LOCK = _sam3_threading.RLock()
_LAYOUT_PROMPT_EPOCHS = {}
_LAYOUT_PROMPT_EPOCH_LOCK = _sam3_threading.RLock()
_LAYOUT_REGION_STORE = _layout_regions.LayoutRegionStore(
    layout_masks_root=runtime_layout_dir,
    layout_regions_root=runtime_layout_region_dir,
    categories_path=current_dir / "layout_categories.json",
)


from sam3_demo.workspace import (
    _WORKSPACE_CACHE,
    _WORKSPACE_CACHE_LOCK,
    _SOURCE_IMAGE_CACHE,
    _SOURCE_IMAGE_CACHE_LOCK,
    _release_workspace_memory,
    _prune_workspace_cache,
    _clear_workspace_cache,
    _prune_source_image_cache,
    _clear_source_image_cache,
    _source_image_cache_put,
    _source_image_cache_get,
    _image_gesture_payload,
    _source_gesture_payload,
    _validate_gesture_intent,
    _pil_image,
    _data_url,
    _workspace,
    _fresh_state,
)







from sam3_demo.state import (
    _new_source_image_state,
    _new_template_match_state,
    _new_pcs_state,
    _new_pvs_state,
    _new_session_state,
    _session_id_from_state,
    _new_layout_state,
    _norm_box,
    _bbox_from_payload,
    _xyxy_to_cxcywh_norm,
    _polygon_from_payload,
    _point_from_payload,
    _combine_logits,
    _polygon_action_key,
    _polygon_combine_key,
    _mask_box,
    _pvs_progress,
    _best,
    _make_inst,
    _snapshot,
    _restore,
    _history_snapshot,
    _append_prompt_history,
    _active_instances,
    _is_pcs_mode,
    _is_pvs_manual_mode,
    _is_layout_mask_mode,
    _is_pvs_pool_mode,
    _new_prompt_state,
)


# Kept as source-level compatibility shims for the overlay contract test.
def _active_instances(state):
    return [inst for inst in state.get("instances", {}).values() if inst.get("status") != "deleted"]


def _is_pcs_mode(mode):
    return str(mode or "") == MODE_PCS


def _is_pvs_manual_mode(mode):
    return str(mode or "") == MODE_PVS


def _is_layout_mask_mode(mode):
    return str(mode or "") == MODE_LAYOUT


def _is_pvs_pool_mode(mode):
    return _is_pvs_manual_mode(mode) or _is_layout_mask_mode(mode)


















def _workspace_gesture_payload(image_state, mode=None, click_tool=None, status=""):
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



















def _layout_cache_key(session_id, layout_id):
    if not layout_id:
        raise ValueError("请先在‘版图截图转掩码’Tab 中生成并保存当前版图 mask")
    sid = _layout_tx.safe_id(session_id, "default")
    lid = _layout_tx.safe_id(layout_id, "layout")
    return f"{sid}:{lid}"


def _layout_disk_dir(session_id, layout_id):
    return runtime_layout_dir / _layout_tx.safe_id(session_id, "default") / _layout_tx.safe_id(layout_id, "layout")


def _layout_cache_get(layout_state_or_id, session_id=None):
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


def _restore_layout_cache_from_disk(session_id, layout_id):
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


def _write_layout_meta(cached):
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


def _layout_cache_put(session_id, layout_id, source_image, source_mask, contours, binarize_params, mask_path=None, contour_json_path=None, overlay_path=None, layout_meta_path=None):
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


def _clear_layout_cache(layout_state=None):
    with _LAYOUT_CACHE_LOCK:
        if layout_state and isinstance(layout_state, dict) and layout_state.get("layout_id"):
            _LAYOUT_CACHE.pop(_layout_cache_key(layout_state.get("session_id") or "default", layout_state.get("layout_id")), None)
        else:
            _LAYOUT_CACHE.clear()















def _prompt_mask_size():
    return tuple(int(v) for v in image_predictor.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)


def _polygon_lowres_logits(polygon, width, height):
    target_h, target_w = _prompt_mask_size()
    mask = polygon_to_mask(polygon, height, width).astype(np.float32)
    lowres = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return ((np.clip(lowres, 0.0, 1.0) * 2.0 - 1.0) * 10.0).astype(np.float32)









def _predict_inst(base_state, box_xyxy_px=None, mask_input_lowres_logits=None, point_coords_px=None, point_labels=None):
    if image_predictor is None:
        raise RuntimeError("Image predictor is not initialized")
    kwargs = {"multimask_output": True, "return_logits": True}
    if box_xyxy_px is not None:
        kwargs["box"] = np.asarray(box_xyxy_px, dtype=np.float32)
    if point_coords_px is not None:
        coords = np.asarray(point_coords_px, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords[None, :]
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError("point_coords_px must have shape Nx2")
        labels = np.ones((coords.shape[0],), dtype=np.int64) if point_labels is None else np.asarray(point_labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != coords.shape[0]:
            raise ValueError("point_labels length must match point_coords_px")
        kwargs["point_coords"] = coords
        kwargs["point_labels"] = labels
    if mask_input_lowres_logits is not None:
        mask_input = np.asarray(mask_input_lowres_logits, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, :, :]
        expected = _prompt_mask_size()
        if mask_input.ndim != 3 or tuple(mask_input.shape[-2:]) != expected:
            raise ValueError(f"mask_input_lowres_logits must be 1x{expected[0]}x{expected[1]}")
        kwargs["mask_input"] = mask_input
    with _PVS_PREDICT_LOCK:
        masks, scores, lowres_logits = image_predictor.model.predict_inst(base_state, **kwargs)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return {
        "masks": masks > 0,
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
        "lowres_logits": np.asarray(lowres_logits, dtype=np.float32),
    }






























from sam3_demo.rendering import (
    _layout_preview_alpha,
    _result_placeholder,
)



def _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True, show_interaction_prompts=True, show_layout_overlay=False, layout_state=None):
    image = np.array(_workspace(image_state)["image"].convert("RGB"))
    overlay = image.copy()
    line = image.copy()
    box_draws = []
    label_draws = []
    polygon_draws = []

    def paint(mask, color, alpha):
        nonlocal overlay
        mask = np.asarray(mask).astype(bool)
        if mask.shape[:2] != overlay.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        c = np.array(color, dtype=np.uint8)
        overlay[mask] = (overlay[mask] * (1 - alpha) + c * alpha).astype(np.uint8)

    def queue_box(box, color, thickness=3):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        box_draws.append((x1, y1, x2, y2, color, thickness))

    def queue_label(text, x, y, color):
        label_draws.append((text, int(round(x)), int(round(y)), color))

    def queue_polygon(points, color):
        arr = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32).reshape((-1, 1, 2))
        polygon_draws.append((arr, color, len(points) >= 3))

    if show_layout_overlay and layout_state and layout_state.get("enabled"):
        layout_id = layout_state.get("layout_id")
        try:
            cached = _layout_cache_get(layout_state)
        except Exception:
            cached = None
        painted_groups = False
        if (
            cached is not None
            and layout_state.get("prompt_mask_scope")
            == _LAYOUT_PROMPT_SCOPE_REGION_LABELS
        ):
            masks = cached.get("prompt_group_transformed_masks")
            snapshot = cached.get("prompt_group_snapshot")
            if isinstance(masks, dict) and isinstance(snapshot, dict):
                active_group_id = snapshot.get("active_group_id")
                for group_id, transform in (snapshot.get("transforms") or {}).items():
                    layout_mask = np.asarray(masks.get(group_id), dtype=bool)
                    if layout_mask.shape != overlay.shape[:2]:
                        continue
                    color = (
                        (0, 255, 130)
                        if group_id == active_group_id
                        else (0, 205, 105)
                    )
                    paint(
                        layout_mask,
                        color,
                        _layout_preview_alpha(transform),
                    )
                    ys, xs = np.where(layout_mask)
                    if len(xs):
                        label_text = str(transform.get("label") or group_id)
                        if not label_text.isascii() or not label_text.isprintable():
                            label_text = f"L{int(transform['region_id'])}"
                        queue_label(
                            label_text,
                            int(xs.min()),
                            max(18, int(ys.min()) - 6),
                            color,
                        )
                    painted_groups = True
        if (
            not painted_groups
            and cached is not None
            and cached.get("transformed_mask") is not None
        ):
            layout_mask = np.asarray(cached["transformed_mask"], dtype=bool)
            if layout_mask.shape == overlay.shape[:2]:
                paint(layout_mask, (0, 255, 130), _layout_preview_alpha(layout_state))
                ys, xs = np.where(layout_mask)
                if len(xs):
                    queue_label(f"LAYOUT {layout_id}", int(xs.min()), max(18, int(ys.min()) - 6), (0, 255, 130))
    if show_instances and _is_pcs_mode(mode):
        for inst in _active_instances(pcs_state):
            color = (0, 255, 90)
            paint(inst["mask_fullres_bool"], color, 0.40)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 3)
            queue_label(f"PCS#{inst['id']}", x1, max(18, y1 - 6), color)
    if _is_pvs_manual_mode(mode) and not show_instances:
        for rec in _pvs_pending_bbox_records(pvs_state):
            box = rec.get("box", [])
            if len(box) != 4:
                continue
            color = (0, 255, 90)
            queue_box(box, color, 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            queue_label(f"B-ID{rec.get('id')}", x1, max(18, y1 - 6), color)
        for inst in _active_instances(pvs_state):
            for event in reversed(inst.get("prompt_history") or []):
                if event.get("op") != "create_from_pending_bbox":
                    continue
                box = event.get("box_xyxy_px") or []
                if len(box) != 4:
                    continue
                color = (0, 255, 90)
                queue_box(box, color, 3)
                x1, y1, _, _ = [int(round(v)) for v in box]
                queue_label(f"PVS-B#{inst['id']}", x1, max(18, y1 - 6), color)
                break
    if show_instances and _is_pvs_pool_mode(mode):
        active_id = pvs_state.get("active_instance_id")
        for inst in _active_instances(pvs_state):
            is_active = str(inst["id"]) == str(active_id)
            color = (255, 0, 220) if is_active else (0, 185, 255)
            paint(inst["mask_fullres_bool"], color, 0.32 if is_active else 0.22)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 4 if is_active else 3)
            queue_label(f"PVS#{inst['id']}", x1, max(18, y1 - 6), color)
    if _is_pcs_mode(mode) and not show_instances:
        for rec in _pcs_bbox_records(pcs_state):
            color = (255, 48, 48) if rec.get("key") == "negative_boxes" else (0, 255, 90)
            box = rec.get("box", [])
            if len(box) != 4:
                continue
            queue_box(box, color, 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            role = "N" if rec.get("key") == "negative_boxes" else "P"
            queue_label(f"{role}-ID{rec.get('id')}", x1, max(18, y1 - 6), color)
    if show_interaction_prompts and prompt_state:
        bbox_color = (255, 48, 48) if prompt_state.get("bbox_role") == "negative" else (0, 255, 90)
        if prompt_state.get("last_bbox"):
            queue_box(prompt_state["last_bbox"], bbox_color, 4)
        if prompt_state.get("bbox_start"):
            x, y = [int(round(v)) for v in prompt_state["bbox_start"]]
            cv2.circle(line, (x, y), 7, bbox_color, -1)
            cv2.putText(line, "bbox start", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bbox_color, 2)
        if prompt_state.get("last_point"):
            x, y = [int(round(v)) for v in prompt_state["last_point"]]
            cv2.circle(line, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(line, (x, y), 9, (255, 255, 255), 2)
        pts = prompt_state.get("polygon_points") or []
        if pts:
            queue_polygon(pts, (0, 255, 60))
    result = cv2.addWeighted(overlay, 0.72, line, 0.28, 0)
    for arr, color, is_closed in polygon_draws:
        if is_closed:
            filled = result.copy()
            cv2.fillPoly(filled, [arr], color)
            result = cv2.addWeighted(filled, 0.18, result, 0.82, 0)
        cv2.polylines(result, [arr], isClosed=is_closed, color=(0, 0, 0), thickness=5)
        cv2.polylines(result, [arr], isClosed=is_closed, color=color, thickness=3)
        for point in arr.reshape((-1, 2)):
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 6, (0, 0, 0), -1)
            cv2.circle(result, (x, y), 4, color, -1)
    for x1, y1, x2, y2, color, thickness in box_draws:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    for text, x, y, color in label_draws:
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return Image.fromarray(result)


def _instances_for_mode(pcs_state, pvs_state, mode):
    return _active_instances(pcs_state if _is_pcs_mode(mode) else pvs_state)


def _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None, layout_state=None):
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




def _result_image(image_state, pcs_state, pvs_state, mode):
    if not image_state or not image_state.get("image_id"):
        return None
    if not _instances_for_mode(pcs_state, pvs_state, mode):
        return _result_placeholder(image_state)
    return _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True, show_interaction_prompts=False, show_layout_overlay=False)



def _event_point(evt, image_state):
    index = getattr(evt, "index", None)
    if isinstance(index, dict):
        point = index.get("point") or index.get("index") or index.get("value")
    else:
        point = index
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"Unsupported Gradio select event index: {index!r}")
    width = int(image_state.get("width") or 0)
    height = int(image_state.get("height") or 0)
    return [max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))]


def _payload_json(value):
    return json.dumps(value, ensure_ascii=False)


def _reset_pcs_predictions(pcs_state):
    pcs_state["instances"] = {}
    pcs_state["next_instance_id"] = 1


def _pcs_bbox_records(pcs_state):
    records = pcs_state.setdefault("bbox_records", [])
    if records:
        return records

    next_id = int(pcs_state.get("next_bbox_id", 1) or 1)
    rebuilt = []
    for key in ("positive_boxes", "negative_boxes"):
        for box in pcs_state.get(key, []):
            rebuilt.append({"id": next_id, "key": key, "box": box})
            next_id += 1
    if rebuilt:
        pcs_state["bbox_records"] = rebuilt
        pcs_state["bbox_history"] = [dict(item) for item in rebuilt]
        pcs_state["next_bbox_id"] = next_id
    return pcs_state.setdefault("bbox_records", [])


def _sync_pcs_boxes_from_records(pcs_state):
    records = _pcs_bbox_records(pcs_state)
    pcs_state["positive_boxes"] = [rec.get("box") for rec in records if rec.get("key") == "positive_boxes"]
    pcs_state["negative_boxes"] = [rec.get("box") for rec in records if rec.get("key") == "negative_boxes"]


def _pcs_bbox_choices(pcs_state):
    choices = []
    for rec in _pcs_bbox_records(pcs_state):
        role = "负样本" if rec.get("key") == "negative_boxes" else "正样本"
        box = [round(float(v), 1) for v in rec.get("box", [])]
        choices.append((f"ID {rec.get('id')} {role} {box}", str(rec.get("id"))))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _append_pcs_bbox_sample(pcs_state, box, bbox_role):
    _pcs_bbox_records(pcs_state)
    key = "negative_boxes" if bbox_role == "negative" else "positive_boxes"
    bbox_id = int(pcs_state.get("next_bbox_id", 1) or 1)
    record = {"id": bbox_id, "key": key, "box": box}
    pcs_state.setdefault("bbox_records", []).append(record)
    pcs_state.setdefault("bbox_history", []).append(dict(record))
    pcs_state["next_bbox_id"] = bbox_id + 1
    _sync_pcs_boxes_from_records(pcs_state)
    _reset_pcs_predictions(pcs_state)
    return key



def _pvs_pending_bbox_records(pvs_state):
    records = pvs_state.setdefault("pending_bbox_records", [])
    if records:
        return records

    next_id = int(pvs_state.get("next_pending_bbox_id", 1) or 1)
    rebuilt = []
    for box in pvs_state.get("pending_boxes", []):
        rebuilt.append({"id": next_id, "box": box})
        next_id += 1
    if rebuilt:
        pvs_state["pending_bbox_records"] = rebuilt
        pvs_state["next_pending_bbox_id"] = next_id
    return pvs_state.setdefault("pending_bbox_records", [])


def _sync_pvs_pending_boxes_from_records(pvs_state):
    pvs_state["pending_boxes"] = [rec.get("box") for rec in _pvs_pending_bbox_records(pvs_state)]


def _pvs_pending_bbox_choices(pvs_state):
    choices = []
    for rec in _pvs_pending_bbox_records(pvs_state):
        box = [round(float(v), 1) for v in rec.get("box", [])]
        choices.append((f"ID {rec.get('id')} \u5f85\u751f\u6210 bbox {box}", str(rec.get("id"))))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _append_pvs_pending_bbox(pvs_state, box):
    _pvs_pending_bbox_records(pvs_state)
    bbox_id = int(pvs_state.get("next_pending_bbox_id", 1) or 1)
    pvs_state.setdefault("pending_bbox_records", []).append({"id": bbox_id, "box": box})
    pvs_state["next_pending_bbox_id"] = bbox_id + 1
    _sync_pvs_pending_boxes_from_records(pvs_state)
    return bbox_id


def _clear_pvs_pending_bboxes(pvs_state):
    count = len(_pvs_pending_bbox_records(pvs_state))
    pvs_state["pending_bbox_records"] = []
    pvs_state["pending_boxes"] = []
    return count

def _click_tool_key(click_tool):
    text = str(click_tool or "").strip()
    lower = text.lower()
    if lower in {"point", "bbox", "polygon", "layout"}:
        return lower
    if "point" in lower or "\u70b9" in text:
        return "point"
    if "bbox" in lower or "box" in lower or "\u6846" in text:
        return "bbox"
    if "polygon" in lower or "\u591a\u8fb9\u5f62" in text:
        return "polygon"
    return ""


def _workspace_select(image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, evt: gr.SelectData):
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


class _GestureSelectEvent:
    def __init__(self, point):
        self.index = list(point)


def _workspace_gesture_input(
    image_state,
    pcs_state,
    pvs_state,
    mode,
    click_tool,
    pcs_bbox_kind,
    prompt_state,
    gesture_payload,
):
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


def _apply_polygon_to_pvs(image_state, pvs_state, polygon, polygon_action="refine", combine_mode="replace", progress=None):
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


def _finish_native_polygon(image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action="create", polygon_combine_mode="replace", progress=gr.Progress(track_tqdm=False)):
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


def _clear_prompt_selection(image_state, pcs_state, pvs_state, mode):
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


def _clear_pending_point_payload():
    return ""


def _clear_bbox_polygon_payloads():
    return "", ""


def _pcs_choice_update(pcs_state):
    choices = [(f"PCS #{i['id']} score={i['score']:.3f}", str(i["id"])) for i in _active_instances(pcs_state)]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _status_label(status):
    return {"draft": "草稿", "accepted": "已确认", "deleted": "已删除"}.get(str(status or "draft"), str(status or "草稿"))


def _pvs_choice_update(pvs_state):
    choices = [(f"PVS #{i['id']} {_status_label(i.get('status'))}", str(i["id"])) for i in _active_instances(pvs_state)]
    active = pvs_state.get("active_instance_id")
    value = str(active) if active is not None and any(c[1] == str(active) for c in choices) else None
    return gr.update(choices=choices, value=value)


def _pvs_pending_count_text(pvs_state):
    _sync_pvs_pending_boxes_from_records(pvs_state)
    return f"\u5f85\u751f\u6210 bbox \u6570\u91cf: {len(pvs_state.get('pending_boxes', []))}"


def _pcs_summary(pcs_state):
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


def _pvs_summary(pvs_state):
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

def _analysis_report(pcs_state, pvs_state, mode, info):
    sections = [str(info or "")]
    if _is_pcs_mode(mode):
        sections.extend(["", "PCS Auto 自动概念分割", _pcs_summary(pcs_state)])
    elif _is_layout_mask_mode(mode):
        sections.extend(["", "版图 mask 提示分割", _pvs_summary(pvs_state)])
    else:
        sections.extend(["", "PVS Manual 手动实例分割", _pvs_summary(pvs_state)])
    return "\n".join(part for part in sections if part is not None)
def _view(image_state, pcs_state, pvs_state, mode, info, prompt_state=None, layout_state=None):
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


def _init_workspace(input_image, mode, session_state=None):
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


def _init_workspace_with_layout_editor(input_image, mode, session_state=None, layout_state=None):
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


def _attach_source_provenance(init_result, source_state):
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


def _source_upload_workspace(input_image, mode, session_state=None, layout_state=None):
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


def _record_source_crop_gesture(source_state, gesture_payload):
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


def _crop_failure_outputs(source_state, status):
    return (
        source_state,
        _source_gesture_payload(source_state, status),
        status,
        *([gr.update()] * 16),
    )


def _apply_source_crop(source_state, mode, session_state=None, layout_state=None):
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


def _use_full_source_image(source_state, mode, session_state=None, layout_state=None):
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


def _clear_template_match_outputs(status="请先完成智能分割并选择当前 PVS 实例"):
    return _new_template_match_state(), None, None, str(status)


def _publish_template_match_export(source_image, workflow, source_state, image_state):
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


def _run_template_matching(
    source_state,
    image_state,
    pvs_state,
    mode,
    match_threshold,
    expand_threshold,
    nms_threshold,
):
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


def _delete_selected_pcs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
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


def _run_pcs(image_state, pcs_state, pvs_state, mode, text_prompt, threshold):
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


def _create_pvs_from_pending_boxes(image_state, pcs_state, pvs_state, mode, progress=gr.Progress(track_tqdm=False)):
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


def _delete_selected_pending_pvs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
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


def _clear_pending_pvs_boxes(image_state, pcs_state, pvs_state, mode):
    count = _clear_pvs_pending_bboxes(pvs_state)
    info = f"\u5df2\u6e05\u7a7a {count} \u4e2a\u5f85\u751f\u6210 PVS bbox\uff1b\u5df2\u751f\u6210\u5b9e\u4f8b\u4e0d\u4f1a\u88ab\u5220\u9664"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _set_active_pvs(image_state, pcs_state, pvs_state, mode, selected_id):
    if selected_id:
        pvs_state["active_instance_id"] = int(selected_id)
        info = f"Selected PVS #{selected_id}"
    else:
        pvs_state["active_instance_id"] = None
        info = "No PVS instance selected"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)

def _refine_active_pvs_with_point(image_state, pvs_state, point, point_label, progress=None):
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

def _pvs_point_prompt(image_state, pcs_state, pvs_state, mode, point_payload, point_kind, progress=gr.Progress(track_tqdm=False)):
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


def _layout_point_refine(image_state, pcs_state, pvs_state, mode, point_payload, point_kind, prompt_state, progress=gr.Progress(track_tqdm=False)):
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

def _undo_pvs(image_state, pcs_state, pvs_state, mode):
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


def _delete_pvs(image_state, pcs_state, pvs_state, mode):
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


def _accept_pvs(image_state, pcs_state, pvs_state, mode):
    try:
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("Select a PVS instance first")
        pvs_state["instances"][int(active_id)]["status"] = "accepted"
        info = f"PVS #{active_id} accepted"
    except Exception as exc:
        info = f"Accept failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _history_json(history):
    rows = []
    for item in history:
        row = {"op": item.get("op"), "prompt": item.get("prompt"), "box_xyxy_px": item.get("box_xyxy_px"), "candidate_scores": item.get("candidate_scores")}
        if item.get("before"):
            row["before"] = {"box_xyxy_px": item["before"].get("box_xyxy_px"), "score": item["before"].get("score"), "status": item["before"].get("status")}
        if item.get("after"):
            row["after"] = {"box_xyxy_px": item["after"].get("box_xyxy_px"), "score": item["after"].get("score"), "status": item["after"].get("status")}
        rows.append({k: v for k, v in row.items() if v is not None})
    return rows



def _latest_layout_prompt_from_instances(instances):
    for item in reversed(list(instances or [])):
        for hist in reversed(item.get("prompt_history", []) or []):
            prompt = hist.get("prompt") or {}
            if hist.get("op") in {"create_from_layout_mask", "refine_with_layout_mask"} or prompt.get("type") == "layout_mask":
                return prompt
    return None


def _reconstruct_frozen_layout_prompt_mask(layout_prompt):
    if not isinstance(layout_prompt, dict):
        raise ValueError("layout prompt metadata 无效")
    session_id = layout_prompt.get("session_id")
    layout_id = layout_prompt.get("layout_id")
    source_mask_hash = layout_prompt.get("source_mask_pixel_sha256")
    target_width = int(layout_prompt.get("target_width") or 0)
    target_height = int(layout_prompt.get("target_height") or 0)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("layout prompt 目标尺寸无效")
    matrix = np.asarray(layout_prompt.get("matrix_2x3"), dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("layout prompt 缺少有效的 frozen affine matrix")

    scope = layout_prompt.get("mask_scope")
    details = {"reconstruction_status": "ok"}
    if not scope or scope == _LAYOUT_PROMPT_SCOPE_FULL:
        source_mask, _ = _LAYOUT_REGION_STORE.load_source_mask(
            session_id,
            layout_id,
            source_mask_hash,
        )
        prompt_mask = source_mask
        details["reconstruction_method"] = "frozen_full_mask"
    elif scope == "region":
        document, source_mask = _LAYOUT_REGION_STORE.load_document(
            session_id,
            layout_id,
            source_mask_hash,
        )
        details["current_regions_revision"] = int(
            document.get("regions_revision") or 0
        )
        try:
            region_id = int(layout_prompt.get("region_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("layout Region prompt 缺少有效 region_id") from exc
        record = next(
            (
                item
                for item in document.get("regions", [])
                if int(item.get("region_id") or 0) == region_id
            ),
            None,
        )
        if record is None:
            raise ValueError(f"layout Region R{region_id} 不存在")
        prompt_mask = _layout_regions.decode_binary_mask(
            record.get("mask_rle"),
            source_mask.shape,
        )
        actual_hash = _layout_regions.mask_pixel_sha256(
            prompt_mask.astype(np.uint8)
        )
        expected_hash = layout_prompt.get("region_mask_pixel_sha256")
        if not expected_hash or actual_hash != str(expected_hash):
            raise ValueError(f"layout Region R{region_id} mask hash 不匹配")
        prompt_label = layout_prompt.get("label")
        if prompt_label is not None:
            if _layout_regions.region_label(record) != str(prompt_label):
                raise ValueError(f"layout Region R{region_id} Label 与实例记录不一致")
        else:
            prompt_class = layout_prompt.get("class_label")
            if (
                prompt_class is not None
                and record.get("class_label") != prompt_class
            ):
                raise ValueError(f"layout Region R{region_id} 历史类别与实例记录不一致")
        details.update(
            {
                "reconstruction_method": "frozen_region_rle",
                "region_id": region_id,
                "region_deleted_at": record.get("deleted_at"),
            }
        )
    else:
        raise ValueError(f"未知 layout prompt mask_scope: {scope}")

    transformed = _layout_tx.warp_layout_mask(
        prompt_mask,
        matrix,
        (target_width, target_height),
    )
    return transformed, details


def _write_feedback_layout_artifacts(sample_dir, layout_prompt):
    if not layout_prompt:
        return {}
    transform_path = sample_dir / "layout_transform.json"
    layout_id = layout_prompt.get("layout_id")
    transformed_mask_path = None
    reconstruction = {}
    try:
        transformed_mask, reconstruction = (
            _reconstruct_frozen_layout_prompt_mask(layout_prompt)
        )
        transformed_mask_path = sample_dir / "layout_transformed_mask.png"
        written = cv2.imwrite(
            str(transformed_mask_path),
            np.asarray(transformed_mask, dtype=np.uint8) * 255,
        )
        if not written:
            raise OSError("cannot write reconstructed layout prompt mask")
    except Exception as exc:
        transformed_mask_path = None
        reconstruction = {
            "reconstruction_status": "unavailable",
            "reconstruction_error": str(exc),
        }

    payload = {
        "layout_prompt": layout_prompt,
        "layout_id": layout_id,
        "has_cached_transformed_mask": False,
        "has_reconstructed_transformed_mask": (
            transformed_mask_path is not None
        ),
        "layout_transformed_mask_file": (
            str(transformed_mask_path)
            if transformed_mask_path is not None
            else None
        ),
        **reconstruction,
    }
    with transform_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "layout_transform_file": str(transform_path),
        "layout_transformed_mask_file": (
            str(transformed_mask_path)
            if transformed_mask_path is not None
            else None
        ),
    }

def _submit_feedback(image_state, pcs_state, pvs_state, mode, rating, feedback_tags, feedback_comment):
    try:
        if _is_pvs_pool_mode(mode):
            active_id = pvs_state.get("active_instance_id")
            if active_id is None:
                raise ValueError("请先选择一个 active PVS instance")
            inst = pvs_state.get("instances", {}).get(int(active_id))
            if inst is None or inst.get("status") == "deleted":
                raise ValueError("当前 active PVS instance 不存在或已删除")
            feedback_instances = [inst]
            feedback_target = "active_pvs_instance"
        elif _is_pcs_mode(mode):
            feedback_instances = _active_instances(pcs_state)
            if not feedback_instances:
                raise ValueError("请先运行 PCS 并生成至少一个 PCS instance")
            inst = None
            feedback_target = "pcs_instance_pool"
        else:
            raise ValueError(f"不支持的 feedback 模式: {mode}")

        ws = _workspace(image_state)
        image = ws["image"]
        feedback_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sample_dir = runtime_feedback_dir / "samples" / feedback_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        image_path = sample_dir / "image.png"
        overlay_path = sample_dir / "overlay.png"
        mask_path = sample_dir / "mask.png"
        npz_path = sample_dir / "mask.npz"
        feedback_path = sample_dir / "feedback.json"

        image.save(image_path)
        overlay = _result_image(image_state, pcs_state, pvs_state, mode)
        if overlay is not None:
            overlay.save(overlay_path)

        masks = [np.asarray(item["mask_fullres_bool"]).astype(bool) for item in feedback_instances]
        mask_stack = np.stack([mask.astype(np.uint8) for mask in masks], axis=0)
        mask_preview = np.any(mask_stack.astype(bool), axis=0).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_preview * 255)
        pvs_logits_values = [item.get("pvs_lowres_logits") for item in feedback_instances if item.get("pvs_lowres_logits") is not None]
        pcs_prob_values = [item.get("pcs_fullres_prob") for item in feedback_instances if item.get("pcs_fullres_prob") is not None]
        np.savez_compressed(
            npz_path,
            mask_fullres_uint8=mask_stack,
            pvs_lowres_logits=np.stack([np.asarray(v, dtype=np.float32) for v in pvs_logits_values], axis=0) if pvs_logits_values else np.empty((0,), dtype=np.float32),
            pcs_fullres_prob=np.stack([np.asarray(v, dtype=np.float32) for v in pcs_prob_values], axis=0) if pcs_prob_values else np.empty((0,), dtype=np.float32),
        )
        instance_rows = [
            {
                "instance_id": int(item["id"]),
                "source": item.get("source"),
                "status": item.get("status"),
                "score": float(item.get("score", 0.0)),
                "bbox_xyxy_px": [float(v) for v in item.get("box_xyxy_px", [])],
                "prompt_history": _history_json(item.get("prompt_history", [])),
            }
            for item in feedback_instances
        ]
        layout_artifacts = _write_feedback_layout_artifacts(sample_dir, _latest_layout_prompt_from_instances(feedback_instances))

        payload = {
            "feedback_id": feedback_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "target": feedback_target,
            "rating": rating,
            "tags": feedback_tags or [],
            "comment": feedback_comment or "",
            "image_id": image_state.get("image_id"),
            "image_size": [int(image.width), int(image.height)],
            "instance_id": int(inst["id"]) if inst is not None else None,
            "source": inst.get("source") if inst is not None else "pcs",
            "status": inst.get("status") if inst is not None else None,
            "score": float(inst.get("score", 0.0)) if inst is not None else None,
            "bbox_xyxy_px": [float(v) for v in inst.get("box_xyxy_px", [])] if inst is not None else None,
            "prompt_history": _history_json(inst.get("prompt_history", [])) if inst is not None else [],
            "instance_count": len(feedback_instances),
            "instances": instance_rows,
            "image_file": str(image_path),
            "overlay_file": str(overlay_path) if overlay is not None else None,
            "mask_file": str(mask_path),
            "mask_npz_file": str(npz_path),
            "layout_transform_file": layout_artifacts.get("layout_transform_file"),
            "layout_transformed_mask_file": layout_artifacts.get("layout_transformed_mask_file"),
            "branch": "Zhengqiyuan/PVS-demo",
        }

        with feedback_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with _FEEDBACK_WRITE_LOCK:
            with (runtime_feedback_dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        info = f"反馈已保存: {feedback_id}"
    except Exception as exc:
        info = f"反馈保存失败: {exc}"
    return _view(image_state, pcs_state, pvs_state, mode, info)


def _export_pool(image_state, pcs_state, pvs_state, mode, pool_name, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    try:
        ws = _workspace(image_state)
        image = ws["image"]
        pool = pcs_state if pool_name == "pcs" else pvs_state
        instances = _active_instances(pool)
        if not instances:
            raise ValueError(f"No active {pool_name.upper()} instance")
        export_id = f"{pool_name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        export_dir = runtime_export_dir / export_id
        mask_dir = export_dir / "masks"
        export_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        _overlay(image_state, pcs_state, pvs_state, mode).save(export_dir / "overlay.png")
        masks, scores, predictions = [], [], []
        coco_annotation_extras = []
        for inst in instances:
            mask = np.asarray(inst["mask_fullres_bool"]).astype(bool)
            masks.append(mask)
            scores.append(float(inst.get("score", 1.0)))
            mask_path = mask_dir / f"{pool_name}_{inst['id']:03d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            mask_file = str(mask_path.relative_to(export_dir))
            bbox_xyxy = [float(v) for v in inst.get("box_xyxy_px", [])]
            predictions.append({"id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "score": float(inst.get("score", 0.0)), "bbox_xyxy": bbox_xyxy, "mask_file": mask_file, "final_contour_polygon": mask_to_polygons(mask), "prompt_history": _history_json(inst.get("prompt_history", []))})
            coco_annotation_extras.append({"instance_id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "mask_file": mask_file, "bbox_xyxy": bbox_xyxy})
        metrics = compare_with_coco(masks, scores, coco_dataset, coco_image_name.strip() if coco_image_name else "", coco_split, pcs_state.get("text_prompt", "") if pool_name == "pcs" else "", image.width, image.height, coco_eval_scope, annotation_json_file)
        with (export_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump({"export_id": export_id, "pool": pool_name, "image": {"width": image.width, "height": image.height}, "predictions": predictions, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        with (export_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        coco_payload = create_prediction_coco_json(
            masks,
            scores,
            image.width,
            image.height,
            image_file_name=coco_image_name.strip() if coco_image_name else "source_image",
            category_name=f"{pool_name}_object",
            export_id=export_id,
            annotation_extras=coco_annotation_extras,
        )
        with (export_dir / "coco_masks.json").open("w", encoding="utf-8") as f:
            json.dump(coco_payload, f, ensure_ascii=False, indent=2)
        zip_path = _publish_segmentation_zip(export_dir, f"{export_id}.zip")
        info = f"Exported {len(instances)} {pool_name.upper()} instances: {zip_path}"
        if metrics.get("summary_lines"):
            info += "\n" + "\n".join(metrics["summary_lines"])
        return str(zip_path), info
    except Exception as exc:
        return None, f"Export failed: {exc}"


def _export_pcs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pcs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)


def _export_pvs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pvs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)


def _switch_mode(mode, image_state, pcs_state, pvs_state):
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
def _switch_mode_with_layout_editor(mode, image_state, pcs_state, pvs_state, layout_state):
    _advance_layout_prompt_epoch(image_state, layout_state)
    result = _switch_mode(mode, image_state, pcs_state, pvs_state)
    if _is_layout_mask_mode(mode):
        editor = _layout_editor_payload(image_state, layout_state, "已切换到版图 mask 提示分割，Canvas payload 已刷新。")
    else:
        editor = gr.update()
    return (*result, editor)


def _switch_click_tool(click_tool, mode):
    tool = _click_tool_key(click_tool)
    is_pvs = _is_pvs_manual_mode(mode)
    return (
        gr.update(visible=is_pvs and tool == "bbox"),
        gr.update(visible=is_pvs and tool == "point"),
        gr.update(visible=is_pvs and tool == "polygon"),
    )


def _normalize_layout_morph_pixels(value):
    return int(
        np.clip(
            int(value or 0),
            -_LAYOUT_MASK_MORPH_LIMIT_PX,
            _LAYOUT_MASK_MORPH_LIMIT_PX,
        )
    )


def _apply_layout_mask_morphology(mask, morph_pixels=0):
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


def _binarize_layout_image(input_image, threshold=12, invert=False, open_kernel=0, close_kernel=0, morph_pixels=0):
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


def _filter_layout_components(mask, min_component_area=0, region_mode="all"):
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


def _layout_mask_contours(mask):
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


def _layout_mask_to_preview(mask):
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_contour_overlay(image, mask, contours):
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


def _save_layout_mask_files(session_state, source_image, mask, contours, params):
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




def _layout_mask_to_editor_image(mask):
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_region_png_data_url(image):
    if image is None:
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + _sam3_base64.b64encode(buf.getvalue()).decode("ascii")


def _new_layout_region_state():
    return {
        "session_id": None,
        "layout_id": None,
        "source_mask_hash": None,
        "regions_revision": 0,
        "next_region_id": 1,
        "selected_region_id": None,
    }


def _layout_region_state_from_document(document, selected_region_id=None):
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


def _layout_region_editor_empty(status="请先生成版图 binary mask"):
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


def _layout_region_identity(layout_state):
    if not isinstance(layout_state, dict):
        raise _layout_regions.RegionValidationError("layout state is missing")
    session_id = _layout_regions.safe_path_component(layout_state.get("session_id"), "session_id")
    layout_id = _layout_regions.safe_path_component(layout_state.get("layout_id"), "layout_id")
    source_mask_hash = str(layout_state.get("source_mask_pixel_sha256") or "")
    if not source_mask_hash:
        raise _layout_regions.RegionValidationError("layout state source mask hash is missing")
    return session_id, layout_id, source_mask_hash


def _layout_region_state_matches_identity(
    region_state, session_id, layout_id, source_mask_hash
):
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


def _validate_layout_region_state_identity(layout_state, region_state):
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


def _layout_region_client_intent(payload):
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


def _validate_layout_region_intent(
    layout_state,
    intent,
    *,
    region_state=None,
    require_lasso=False,
):
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


def _layout_region_source_image(session_id, layout_id, source_mask):
    source_path = runtime_layout_dir / session_id / layout_id / "source_image.png"
    if source_path.exists():
        with Image.open(source_path) as image:
            return image.convert("RGB").copy()
    return _layout_mask_to_preview(source_mask)


def _layout_region_summaries(document):
    return [
        {
            "region_id": int(region["region_id"]),
            "label": _layout_regions.region_label(region),
            "area": int(region.get("area") or 0),
        }
        for region in _layout_regions.active_regions(document)
    ]


class _LayoutPromptConflictError(RuntimeError):
    pass


def _layout_prompt_epoch_key(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    for state in (layout_state, image_state, session_state):
        if isinstance(state, dict) and state.get("session_id"):
            return str(state["session_id"])
    return "default"


def _layout_prompt_epoch_snapshot(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    key = _layout_prompt_epoch_key(
        image_state,
        layout_state,
        session_state,
    )
    with _LAYOUT_PROMPT_EPOCH_LOCK:
        return key, int(_LAYOUT_PROMPT_EPOCHS.get(key, 0))


def _advance_layout_prompt_epoch(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    key = _layout_prompt_epoch_key(
        image_state,
        layout_state,
        session_state,
    )
    with _LAYOUT_PROMPT_EPOCH_LOCK:
        value = int(_LAYOUT_PROMPT_EPOCHS.get(key, 0)) + 1
        _LAYOUT_PROMPT_EPOCHS[key] = value
        return value

def _reset_layout_prompt_selection_state(layout_state):
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


def _layout_prompt_selection_token(scope, region_id=None):
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


def _parse_layout_prompt_selection(value):
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


def _normalize_layout_prompt_checkbox_selection(value, layout_state):
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


def _layout_prompt_label_counts(document):
    counts = {}
    for record in sorted(
        _layout_regions.active_regions(document),
        key=lambda item: int(item["region_id"]),
    ):
        label = _layout_regions.region_label(record)
        counts[label] = counts.get(label, 0) + 1
    return list(counts.items())

def _layout_label_choice_text(record, label_counts):
    label = _layout_regions.region_label(record)
    if int(label_counts.get(label) or 0) > 1:
        return f"{label} (R{int(record['region_id'])})"
    return label



def _layout_prompt_class_counts(document):
    return _layout_prompt_label_counts(document)


def _layout_prompt_choice_update(document=None, selected_value=None):
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


def _load_layout_prompt_region_document(layout_state, expected_revision=None):
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


def _layout_prompt_label_records(document, labels):
    records = _layout_regions.active_regions_for_labels(document, labels)
    if not records:
        raise _layout_regions.RegionValidationError(
            "所选 label 没有活动 Region"
        )
    return records


def _layout_prompt_class_records(document, class_label):
    return _layout_prompt_label_records(document, [class_label])


def _layout_prompt_region_records(document, region_ids):
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


def _layout_prompt_display_mask(layout_state, source_mask):
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


def _layout_region_choice_update(document, selected_region_id=None):
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


def _layout_region_category_update(value=None):
    return _layout_region_label_update(value=value)


def _layout_region_label_update(document=None, value=None):
    del document
    selected = value.strip() if isinstance(value, str) else ""
    return gr.update(value=selected)


def _layout_region_editor_payload(
    layout_state,
    document,
    source_mask,
    *,
    status,
    selected_region_id=None,
    lasso_polygon=None,
    draft_region_mask=None,
):
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


def _load_layout_region_context(layout_state):
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


def _clear_layout_region_context(_layout_state):
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


def _preview_layout_region(layout_state, region_state, editor_payload):
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


def _select_layout_region(layout_state, region_state, selected_region_id):
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


def _layout_region_latest_values(
    layout_state,
    selected,
    status,
    intent=None,
    keep_draft=False,
    region_state=None,
):
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


def _save_layout_region(layout_state, region_state, editor_payload, label):
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


def _delete_layout_region(layout_state, region_state, editor_payload, selected_region_id):
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


def _export_layout_regions(layout_state, region_state):
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


def _layout_prompt_group_id(region_id):
    return f"region_{int(region_id)}"


def _layout_prompt_group_data(layout_state, source_mask):
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


def _layout_group_transform(
    layout_state,
    base_transform,
    record,
    group_mask,
    target_size,
    values=None,
):
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


def _layout_prompt_group_payload(
    layout_state,
    source_mask,
    base_transform,
    target_size,
):
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



def _layout_editor_empty(image_state=None, status="请先加载或生成版图 mask"):
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


def _layout_editor_payload(image_state, layout_state, status=None):
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


def _layout_editor_transform(editor_payload):
    if not isinstance(editor_payload, dict):
        return None
    transform = editor_payload.get("transform")
    return transform if isinstance(transform, dict) else None


def _layout_group_control_values(transform, target_size):
    tx, ty = _layout_tx.derive_legacy_tx_ty(transform, target_size)
    return (
        float(tx),
        float(ty),
        float(transform.get("scale") or 1.0),
        float(transform.get("rotation_deg") or 0.0),
        _layout_preview_alpha(transform),
    )


def _commit_layout_group_transforms(
    image_state,
    layout_state,
    editor_payload,
    *,
    numeric_override=None,
    reset_active=False,
):
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


def _validate_layout_group_transform_snapshot(layout_state, snapshot):
    with _LAYOUT_CACHE_LOCK:
        cached = _layout_cache_get(layout_state)
        current = copy.deepcopy(cached.get("prompt_group_snapshot"))
    if not isinstance(current, dict) or current != snapshot:
        raise ValueError("Label transform 在批量预测期间发生变化")



def _sync_layout_controls_from_editor(layout_state, editor_payload):
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


def _sync_layout_controls_from_editor_with_prompt_epoch(
    layout_state,
    editor_payload,
):
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


def _run_layout_mask_page(session_state, image_state, input_image, threshold, invert, open_kernel, close_kernel, min_component_area, region_mode, morph_pixels=0):
    try:
        image, mask = _binarize_layout_image(input_image, threshold, invert, open_kernel, close_kernel, morph_pixels)
        mask = _filter_layout_components(mask, min_component_area, region_mode)
        if not mask.any():
            raise ValueError("Binary mask is empty; lower threshold or check invert")
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


def _run_layout_mask_page_with_downloads(
    session_state,
    image_state,
    input_image,
    threshold,
    invert,
    open_kernel,
    close_kernel,
    min_component_area,
    region_mode,
    morph_pixels=0,
):
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


def _save_current_layout_mask(layout_state):
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


def _clear_current_layout_mask(image_state, layout_state):
    session_id = layout_state.get("session_id") if isinstance(layout_state, dict) else None
    _clear_layout_cache(layout_state)
    state = _new_layout_state(session_id)
    return state, _layout_editor_empty(image_state, "当前版图 mask 已清除"), None, None, None, None, None, "当前版图 mask 已清除"


def _clear_current_layout_mask_with_prompt_epoch(
    image_state,
    layout_state,
):
    _advance_layout_prompt_epoch(image_state, layout_state)
    return _clear_current_layout_mask(image_state, layout_state)


def _layout_numeric_controls_changed(layout_state, tx, ty, scale, rotation_deg, preview_alpha, tol=1e-6):
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


def _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=None, prefer_numeric=None):
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


def _transform_layout_mask(layout_state, target_width, target_height):
    raise RuntimeError("_transform_layout_mask is deprecated; use _commit_layout_transform(image_state, ...) so target image hash and revision are validated")


def _layout_mask_to_overlay(base_image, mask, alpha=0.35):
    base = np.asarray(_pil_image(base_image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != base.shape[:2]:
        raise ValueError("layout mask and target image sizes do not match")
    fill = base.copy()
    fill[mask] = (0, 255, 130)
    return Image.fromarray(cv2.addWeighted(fill, float(alpha), base, 1.0 - float(alpha), 0))


def _update_layout_preview(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload):
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


def _update_layout_preview_with_groups(
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
):
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


def _mask_to_lowres_logits(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("layout mask must be a 2D binary mask")
    target_h, target_w = _prompt_mask_size()
    lowres = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return ((lowres * 2.0 - 1.0) * 10.0).astype(np.float32)


def _validate_layout_prompt_mask(mask):
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


def _layout_transformed_mask_for_image(image_state, layout_state):
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


def _layout_prompt_metadata(image_state, layout_state):
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


def _create_pvs_from_layout_mask(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload, progress=gr.Progress(track_tqdm=False)):
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


def _layout_state_summary(layout_state):
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


def _load_layout_binary_mask_png(session_state, image_state, input_image, region_mode="all"):
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


def _use_current_layout_mask(image_state, layout_state):
    _advance_layout_prompt_epoch(image_state, layout_state)
    try:
        _layout_cache_get(layout_state)
        info = "Using current saved layout mask.\n" + _layout_state_summary(layout_state)
        return layout_state, _layout_editor_payload(image_state, layout_state, "当前已保存版图 mask 已载入 Canvas。"), info
    except Exception as exc:
        return layout_state or _new_layout_state(), _layout_editor_empty(image_state, f"当前版图 mask 不可用：{exc}"), f"当前版图 mask 不可用：{exc}"


def _pvs_creation_commit_token(pvs_state):
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


def _selected_pvs_candidate(prediction, image_shape):
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


def _layout_prompt_region_fingerprint(decoded_records):
    return tuple(
        (
            int(record["region_id"]),
            _layout_regions.region_label(record),
            _layout_regions.mask_pixel_sha256(mask.astype(np.uint8)),
        )
        for record, mask in decoded_records
    )


def _validate_layout_transform_snapshot(layout_state, transform):
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


def _load_layout_prompt_choices(image_state, layout_state):
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


def _select_layout_prompt_mask(image_state, layout_state, selection):
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


def _reset_layout_prompt_selection(image_state, layout_state):
    _advance_layout_prompt_epoch(image_state, layout_state)
    state = _reset_layout_prompt_selection_state(layout_state)
    info = "版图 mask 选择已重置；点击‘使用当前已保存版图 mask’加载 Label。"
    return (
        state,
        _layout_editor_payload(image_state, state, info),
        _layout_prompt_choice_update(),
    )


def _create_pvs_from_layout_selection(
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
    prompt_selection,
    progress=gr.Progress(track_tqdm=False),
):
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


def _reset_layout_controls(image_state, layout_state):
    state = dict(layout_state or _new_layout_state())
    state.update({"enabled": bool(state.get("layout_id")), "tx": 0.0, "ty": 0.0, "scale": 1.0, "rotation_deg": 0.0, "preview_alpha": 0.35})
    if isinstance(image_state, dict) and image_state.get("width") and image_state.get("height"):
        state["center_x"] = float(image_state.get("width")) / 2.0
        state["center_y"] = float(image_state.get("height")) / 2.0
    info = "版图变换控件已重置。\n" + _layout_state_summary(state)
    return state, True if state.get("layout_id") else False, 0.0, 0.0, 1.0, 0.0, 0.35, _layout_editor_payload(image_state, state, "Canvas 变换已重置。"), info

def _reset_layout_controls_with_prompt_epoch(image_state, layout_state):
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


def create_demo():
    """Create the PCS/PVS Gradio interface while preserving the original demo layout."""
    custom_css = """
    .container { max-width: 1200px; margin: auto; padding-top: 10px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin: 0 0 8px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin: 0 0 16px; }
    #main_tabs { margin-top: -32px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    .hidden-payload { display: none !important; }
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    .sam3-panel textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .gesture-overlay-anchor { min-height: 0 !important; height: 0 !important; overflow: visible !important; }
    .image-prepost-row { align-items: stretch !important; }
    .image-prepost-column { height: 100%; }
    .polygon-finish-btn button {
        width: 100%;
        min-height: 42px;
        font-weight: 700;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }
    .layout-preview-pager {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 12px;
        overflow-x: auto !important;
        overscroll-behavior-x: contain;
        scroll-behavior: smooth;
        scroll-snap-type: x mandatory;
        scrollbar-gutter: stable;
        padding-bottom: 8px;
    }
    .layout-preview-page {
        flex: 0 0 100% !important;
        min-width: 100% !important;
        scroll-snap-align: start;
        scroll-snap-stop: always;
    }
    """
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    with gr.Blocks(
        theme=theme,
        css=custom_css,
        title="SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0",
        delete_cache=(3600, _PUBLIC_DOWNLOAD_TTL_SECONDS),
    ) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0")
            gr.Markdown("\u57fa\u4e8e SAM3 \u7684 PCS \u81ea\u52a8\u6982\u5ff5\u5206\u5272\u4e0e PVS \u624b\u52a8\u5b9e\u4f8b\u5206\u5272\u5de5\u4f5c\u53f0", elem_classes="description")
            session_state = gr.State(_new_session_state())
            image_state = gr.State({"image_id": None, "width": 0, "height": 0})
            source_image_state = gr.State(_new_source_image_state())
            pcs_state = gr.State(_new_pcs_state())
            pvs_state = gr.State(_new_pvs_state())
            template_match_state = gr.State(_new_template_match_state())
            prompt_state = gr.State(_new_prompt_state())
            layout_state = gr.State(_new_layout_state())
            layout_region_state = gr.State(_new_layout_region_state())
            bbox_payload = gr.Textbox(label="bbox payload", elem_id="bbox_payload", elem_classes="hidden-payload")
            polygon_payload = gr.Textbox(label="polygon payload", elem_id="polygon_payload", elem_classes="hidden-payload")
            point_payload = gr.Textbox(label="point payload", elem_id="point_payload", elem_classes="hidden-payload")

            with gr.Tabs(elem_id="main_tabs"):
                with gr.TabItem("智能图像分割", id="tab_image"):
                    with gr.Row(equal_height=True, elem_id="image_prepost_row", elem_classes="image-prepost-row"):
                        with gr.Column(scale=1, elem_id="source_prepost_column", elem_classes="image-prepost-column"):
                            gr.Markdown("### 上传与裁剪")
                            source_image_upload = gr.Image(
                                type="pil",
                                label="完整原图",
                                show_label=False,
                                sources=["upload", "clipboard"],
                                elem_id="source_input_image",
                                elem_classes="aligned-prepost-preview",
                                height=320,
                            )
                            if ImageGestureOverlay is not None:
                                source_crop_overlay = ImageGestureOverlay(
                                    value=_source_gesture_payload(_new_source_image_state()),
                                    label="完整原图裁剪手势",
                                    show_label=False,
                                    target_elem_id="source_input_image",
                                    height=1,
                                    elem_classes="gesture-overlay-anchor",
                                )
                            else:
                                gr.Markdown(f"图像手势组件不可用。错误：{_image_gesture_overlay_import_error}")
                                source_crop_overlay = gr.JSON(
                                    value=_source_gesture_payload(_new_source_image_state()),
                                    visible=False,
                                )
                            with gr.Row():
                                apply_crop_btn = gr.Button("应用裁剪", variant="primary")
                                use_full_image_btn = gr.Button("使用整图", variant="secondary")
                            source_crop_status = gr.Markdown("请上传完整原图；默认直接使用整图。")
                            with gr.Accordion("模板匹配可选参数", open=False):
                                match_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="matchThreshold")
                                expand_threshold = gr.Number(value=20, minimum=0, precision=0, label="expandThreshold (px)")
                                nms_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="nmsThreshold")
                        with gr.Column(scale=1, elem_id="template_prepost_column", elem_classes="image-prepost-column"):
                            gr.Markdown("### 模板匹配")
                            template_match_preview = gr.Image(
                                type="pil",
                                label="完整原图模板匹配结果",
                                show_label=False,
                                interactive=False,
                                height=320,
                                elem_id="template_match_preview",
                                elem_classes="aligned-prepost-preview",
                            )
                            gr.Markdown("基于当前 active PVS 分割结果，在完整原图中寻找相似结构。")
                            run_template_match_btn = gr.Button("开始模板匹配", variant="primary")
                            template_match_status = gr.Markdown("请先完成智能分割并选择当前 PVS 实例")
                            template_match_file = gr.File(label="下载模板匹配结果包", interactive=False)
                    mode = gr.Radio(
                        choices=[("PCS Auto 自动概念分割", "PCS Auto"), ("PVS Manual 手动实例分割", "PVS Manual"), ("版图 mask 提示分割", "Layout Mask")],
                        value="PVS Manual",
                        label="功能模式",
                        elem_classes="mode-radio",
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 原始图像（点击进行交互）")
                            image_upload = gr.Image(type="numpy", label="原始图像", show_label=False, interactive=False, elem_id="input_image")
                            if ImageGestureOverlay is not None:
                                workspace_gesture_overlay = ImageGestureOverlay(
                                    value=_workspace_gesture_payload({}, "PVS Manual", "bbox"),
                                    label="分割交互手势",
                                    show_label=False,
                                    target_elem_id="input_image",
                                    height=1,
                                    elem_classes="gesture-overlay-anchor",
                                )
                            else:
                                workspace_gesture_overlay = gr.JSON(
                                    value=_workspace_gesture_payload({}, "PVS Manual", "bbox"),
                                    visible=False,
                                )
                            with gr.Group():
                                gr.Markdown("### \u4ea4\u4e92\u6a21\u5f0f")
                                click_tool = gr.Radio(
                                    choices=[("\u70b9\u63d0\u793a (Point)", "point"), ("\u6846\u63d0\u793a (Box)", "bbox"), ("\u591a\u8fb9\u5f62Mask (Polygon)", "polygon")],
                                    value="bbox",
                                    label="\u9009\u62e9\u6a21\u5f0f",
                                    show_label=False,
                                    elem_classes="mode-radio",
                                )
                                with gr.Group(visible=False) as pcs_bbox_tools:
                                    pcs_bbox_kind = gr.Radio(
                                        choices=[("\u6b63\u6837\u672c bbox", "Positive exemplar"), ("\u8d1f\u6837\u672c bbox", "Negative exemplar")],
                                        value="Positive exemplar",
                                        label="PCS bbox \u6837\u672c\u7c7b\u578b",
                                        elem_classes="mode-radio",
                                    )
                                    pcs_bbox_selector = gr.Dropdown(choices=[], label="PCS bbox \u5217\u8868", interactive=True)
                                    delete_selected_pcs_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d PCS bbox", size="sm", variant="secondary")
                                with gr.Row():
                                    clear_prompt_btn = gr.Button("\u6e05\u7a7a\u63d0\u793a (Clear Prompts)", size="sm", variant="secondary")
                                interaction_info = gr.Markdown("\u70b9\u51fb\u56fe\u50cf\u5f00\u59cb\u6dfb\u52a0\u63d0\u793a...", elem_id="interaction-info")

                            with gr.Accordion("点提示修缮", open=False, visible=False) as layout_point_refine_panel:
                                layout_point_kind = gr.Radio(
                                    choices=[("正向点", "positive"), ("负向点", "negative")],
                                    value="positive",
                                    label="点类型",
                                    elem_classes="mode-radio",
                                )
                                layout_point_btn = gr.Button("应用点提示", variant="primary")

                            with gr.Accordion("\u9ad8\u7ea7\u63d0\u793a\u9009\u9879", open=True):
                                with gr.Group(visible=False) as pcs_panel:
                                    gr.Markdown("### PCS Auto \u81ea\u52a8\u6982\u5ff5\u5206\u5272")
                                    text_prompt = gr.Textbox(label="\u6587\u672c\u63d0\u793a (Text Prompt)", placeholder="\u8f93\u5165\u7269\u4f53\u63cf\u8ff0\uff0c\u4f8b\u5982\uff1a'a red car' \u6216 '\u4e00\u53ea\u732b'", lines=1)
                                    confidence_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="\u7f6e\u4fe1\u5ea6\u9608\u503c (Confidence)")
                                    run_pcs_btn = gr.Button("\u5f00\u59cb PCS \u5206\u5272", variant="primary")
                                    export_pcs_btn = gr.Button("\u5bfc\u51fa PCS")
                                    pcs_summary = gr.Textbox(label="PCS \u5b9e\u4f8b", lines=6, interactive=False)

                                with gr.Group(visible=True) as pvs_panel:
                                    gr.Markdown("### PVS Manual \u624b\u52a8\u5b9e\u4f8b\u5206\u5272")
                                    with gr.Group(visible=True) as pvs_bbox_prompt_panel:
                                        gr.Markdown("#### BBox prompt")
                                        pvs_pending_count = gr.Markdown("\u5f85\u751f\u6210 bbox \u6570\u91cf: 0")
                                        pvs_pending_bbox_selector = gr.Dropdown(choices=[], label="PVS \u5f85\u751f\u6210 bbox \u5217\u8868", interactive=True)
                                        create_pvs_batch_btn = gr.Button("\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b", variant="primary")
                                        with gr.Row():
                                            delete_selected_pending_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
                                            clear_pending_bbox_btn = gr.Button("\u6e05\u7a7a\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
                                    with gr.Group(visible=False) as pvs_point_prompt_panel:
                                        gr.Markdown("#### Point prompt")
                                        pvs_point_kind = gr.Radio(
                                            choices=[("正向点", "positive"), ("负向点", "negative")],
                                            value="positive",
                                            label="点类型",
                                            elem_classes="mode-radio",
                                        )
                                        pvs_point_btn = gr.Button("应用点提示", variant="primary")
                                    with gr.Group(visible=False) as pvs_polygon_prompt_panel:
                                        gr.Markdown("#### Polygon prompt")
                                        polygon_action = gr.Radio(
                                            choices=[("\u521b\u5efa\u65b0 PVS \u5b9e\u4f8b", "create"), ("\u7cbe\u4fee\u5f53\u524d PVS \u5b9e\u4f8b", "refine")],
                                            value="create",
                                            label="\u591a\u8fb9\u5f62\u52a8\u4f5c",
                                            elem_classes="mode-radio",
                                        )
                                        finish_polygon_btn = gr.Button("\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61", variant="primary", elem_classes="polygon-finish-btn")
                                        with gr.Accordion("高级 Polygon 融合方式", open=False):
                                            polygon_combine_mode = gr.Radio(
                                                choices=[("Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b", "replace"), ("Blend \u4e0e\u65e7 mask \u878d\u5408", "blend"), ("Union \u8865\u5145\u533a\u57df", "union"), ("Intersect \u9650\u5236\u8303\u56f4", "intersect")],
                                                value="replace",
                                                label="\u591a\u8fb9\u5f62\u878d\u5408\u65b9\u5f0f",
                                                elem_classes="mode-radio",
                                            )
                                            gr.Markdown(
                                                "**\u4ee5\u4e0a\u56db\u79cd\u90fd\u662f Positive Polygon \u7684\u878d\u5408\u65b9\u5f0f\uff0c\u4e0d\u5305\u542b negative prompt\u3002**  \n"
                                                "- Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b\uff1a\u7528\u5f53\u524d polygon \u4f5c\u4e3a\u5b8c\u6574 mask prompt\u3002  \n"
                                                "- Blend \u4e0e\u65e7 mask \u878d\u5408\uff1a\u65e7 logits \u548c polygon logits \u5171\u540c\u5f71\u54cd\u7ed3\u679c\u3002  \n"
                                                "- Union \u8865\u5145\u533a\u57df\uff1a\u4fdd\u7559\u65e7 mask\uff0c\u5e76\u52a0\u5165 polygon \u533a\u57df\u3002  \n"
                                                "- Intersect \u9650\u5236\u8303\u56f4\uff1a\u5c06\u7ed3\u679c\u9650\u5236\u5728 polygon \u8303\u56f4\u5185\u3002"
                                            )
                                    pvs_summary = gr.Textbox(label="PVS 实例", lines=6, interactive=False, visible=False)

                                with gr.Group(visible=False) as pvs_layout_panel:
                                    gr.Markdown("### PVS 版图 mask 提示")
                                    gr.Markdown("上传或选择二值版图 mask；多选 Label 后，每个 Label 可在右侧独立拖动、缩放和旋转。")
                                    with gr.Row():
                                        use_current_layout_btn = gr.Button("使用当前已保存版图 mask", variant="secondary")
                                        load_layout_binary_btn = gr.Button("载入二值 mask PNG", variant="secondary")
                                    layout_prompt_mask_selector = gr.CheckboxGroup(
                                        choices=[("全部版图 mask", _LAYOUT_PROMPT_SCOPE_FULL)],
                                        value=[_LAYOUT_PROMPT_SCOPE_FULL],
                                        label="版图 mask 选择（可多选 Label）",
                                        interactive=False,
                                        elem_id="layout_prompt_mask_selector",
                                    )
                                    gr.Markdown("#### 直接上传二值 mask PNG")
                                    layout_binary_upload = gr.Image(type="numpy", label="直接上传二值 mask PNG", show_label=False, sources=["upload", "clipboard"])
                                with gr.Accordion("\u5bfc\u51fa\u4e0e COCO \u91cf\u5316", open=False):
                                    coco_dataset = gr.Dropdown(choices=coco_dataset_choices, value=default_coco_dataset, label="\u6307\u6807\u6570\u636e\u96c6")
                                    coco_image_name = gr.Textbox(label="COCO image file_name\uff08\u53ef\u9009\uff09", lines=1)
                                    coco_split = gr.Radio(choices=["auto", "val", "train", "test"], value="auto", label="\u6807\u6ce8 split")
                                    coco_eval_scope = gr.Radio(choices=[coco_eval_scope_overlap, coco_eval_scope_full], value=coco_eval_scope_overlap, label="\u8bc4\u4f30\u8303\u56f4")
                                    annotation_json_file = gr.File(label="\u4e0a\u4f20 O3/LabelMe-like JSON \u6807\u6ce8\uff08\u4f18\u5148\u4e8e COCO lookup\uff09", file_types=[".json"], type="filepath")

                        with gr.Column(scale=1):
                            gr.Markdown("### \u5206\u5272\u7ed3\u679c")
                            result_image = gr.Image(type="numpy", label="\u5206\u5272\u7ed3\u679c", show_label=False)
                            with gr.Group(visible=True) as analysis_report_panel:
                                analysis_report = gr.Textbox(label="分析报告", interactive=False, lines=18)
                            with gr.Group(visible=False) as layout_transform_panel:
                                gr.Markdown("### 修改变形版图")
                                if LayoutTransformEditor is not None:
                                    layout_editor = LayoutTransformEditor(value=_layout_editor_empty(), label="\u7248\u56fe\u4ea4\u4e92\u7f16\u8f91\u5668", show_label=False, height=520, elem_id="layout_transform_editor")
                                else:
                                    gr.Markdown(f"版图 Canvas 编辑器组件不可用；仍可使用数值控件。错误：{_layout_editor_import_error}")
                                    layout_editor = gr.JSON(value=_layout_editor_empty(), label="layout transform payload", visible=False)
                                layout_enabled = gr.Checkbox(value=False, label="显示/启用版图 overlay")
                                gr.Markdown("多选 Label 时，下方数值只对应 Canvas 中当前激活的 Label。")
                                with gr.Row():
                                    layout_tx = gr.Number(value=0.0, label="水平偏移 tx")
                                    layout_ty = gr.Number(value=0.0, label="垂直偏移 ty")
                                with gr.Row():
                                    layout_scale = gr.Slider(minimum=0.1, maximum=20.0, value=1.0, step=0.01, label="缩放 scale")
                                    layout_rotation = gr.Slider(minimum=-180.0, maximum=180.0, value=0.0, step=1.0, label="旋转 rotation")
                                layout_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.35, step=0.05, label="透明度 alpha")
                                with gr.Row():
                                    reset_layout_btn = gr.Button("重置", variant="secondary")
                                    update_layout_preview_btn = gr.Button("更新预览", variant="primary")
                                create_from_layout_btn = gr.Button("用所选版图 mask / Label 创建实例", variant="primary")
                                layout_pvs_info = gr.Textbox(label="版图提示状态", lines=5, interactive=False)
                            with gr.Group(visible=True) as pvs_action_panel:
                                gr.Markdown("### PVS 实例操作")
                                active_pvs = gr.Dropdown(choices=[], label="\u5f53\u524d PVS \u5b9e\u4f8b")
                                with gr.Row():
                                    undo_pvs_btn = gr.Button("撤销上一个实例")
                                    delete_pvs_btn = gr.Button("清空实例")
                                    accept_pvs_btn = gr.Button("确认", variant="primary")
                                export_pvs_btn = gr.Button("\u5bfc\u51fa PVS")
                            export_file = gr.File(label="\u4e0b\u8f7d\u7ed3\u679c\u5305\uff08PNG + masks + JSON\uff09", interactive=False)
                            with gr.Accordion("结果反馈（PCS 结果 / PVS 当前实例，用于 RL 数据收集）", open=False):
                                feedback_rating = gr.Radio(
                                    choices=[("好", "good"), ("及格", "pass"), ("差", "bad")],
                                    value="pass",
                                    label="结果质量",
                                    elem_classes="mode-radio",
                                )
                                feedback_tags = gr.CheckboxGroup(
                                    choices=["毛边", "空缺", "漏检", "误检", "边界偏移", "多分/粘连", "polygon 不贴合", "其他"],
                                    label="问题标签",
                                )
                                feedback_comment = gr.Textbox(label="备注", lines=3, placeholder="可选：描述这次生成的问题或可用性")
                                submit_feedback_btn = gr.Button("提交反馈", variant="primary")

                with gr.TabItem("版图截图转掩码", id="tab_layout_mask"):
                    gr.Markdown("### 版图截图转二值 mask")
                    gr.Markdown("binary mask 是唯一权威数据；contour 仅用于预览和导出。左右两侧预览使用相同高度。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 上传版图截图")
                            layout_input = gr.Image(type="numpy", label="上传版图截图", show_label=False, sources=["upload", "clipboard"], height=430)
                            layout_threshold = gr.Slider(minimum=0, maximum=255, value=12, step=1, label="threshold（色彩/饱和度阈值）")
                            layout_invert = gr.Checkbox(value=False, label="invert（反转前景/背景）")
                            with gr.Row():
                                layout_open_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="open kernel")
                                layout_close_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="close kernel")
                            layout_morph_pixels = gr.Slider(
                                minimum=-_LAYOUT_MASK_MORPH_LIMIT_PX,
                                maximum=_LAYOUT_MASK_MORPH_LIMIT_PX,
                                value=0,
                                step=1,
                                label="膨胀/腐蚀像素（正数膨胀，负数腐蚀）",
                            )
                            layout_min_area = gr.Number(value=0, precision=0, label="min component area")
                            layout_region_mode = gr.Radio(
                                choices=[("全部区域", "all"), ("最大连通区域", "largest")],
                                value="all",
                                label="区域模式",
                                elem_classes="mode-radio",
                            )
                            run_layout_mask_btn = gr.Button("生成并保存当前版图 mask", variant="primary")
                            save_layout_mask_btn = gr.Button("保存为当前版图 mask", variant="secondary")
                            clear_layout_mask_btn = gr.Button("清除当前版图", variant="secondary")
                            layout_info = gr.Textbox(label="处理信息", lines=8, interactive=False)
                            with gr.Row():
                                layout_mask_file = gr.File(label="下载 mask PNG", interactive=False)
                                layout_contour_file = gr.File(label="下载 contour JSON", interactive=False)
                        with gr.Column(scale=1):
                            layout_source_preview = gr.Image(type="pil", label="原图预览", show_label=False, visible=False)
                            gr.Markdown("#### mask 与 contour 预览")
                            with gr.Row(elem_classes="layout-preview-pager"):
                                with gr.Column(elem_classes="layout-preview-page"):
                                    layout_mask_preview = gr.Image(type="pil", label="binary mask 预览", show_label=False, height=430)
                                    gr.Markdown("**binary mask 预览**")
                                with gr.Column(elem_classes="layout-preview-page"):
                                    layout_overlay_preview = gr.Image(type="pil", label="contour overlay", show_label=False, height=430)
                                    gr.Markdown("**contour overlay**")
                            gr.Markdown("左右滑动或拖动下方滚动条切换预览。")
                            gr.Markdown("### Label Annotation Layer")
                            gr.Markdown("黄色表示未保存 Draft；绿色表示已保存 Label。一个套索对应一个独立 Label；Label 可选留空并自动按序号命名。")
                            if LayoutRegionAnnotator is not None:
                                layout_region_annotator = LayoutRegionAnnotator(
                                    value=_layout_region_editor_empty(),
                                    label="版图 Label 套索标注器",
                                    show_label=False,
                                    height=520,
                                    elem_id="layout_region_annotator",
                                )
                            else:
                                gr.Markdown(f"Label 套索组件不可用。错误：{_layout_region_annotator_import_error}")
                                layout_region_annotator = gr.JSON(
                                    value=_layout_region_editor_empty(),
                                    label="layout Region payload",
                                    visible=False,
                                )
                            layout_region_label = gr.Textbox(
                                label="Label（可选）",
                                value="",
                                placeholder="留空将自动命名为 Label 1、Label 2……",
                                max_lines=1,
                            )
                            save_layout_region_btn = gr.Button("保存当前 Draft Label", variant="primary", interactive=False)
                            layout_region_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="活动 Label",
                                interactive=True,
                            )
                            delete_layout_region_btn = gr.Button("软删除选中 Label", variant="secondary", interactive=False)
                            layout_region_status = gr.Textbox(label="Label 状态", lines=4, interactive=False)
                            export_layout_regions_btn = gr.Button(
                                "导出当前 Label 标注",
                                variant="secondary",
                            )
                            layout_region_export_file = gr.File(
                                label="下载 Label 标注包",
                                interactive=False,
                            )

            run_layout_mask_event = run_layout_mask_btn.click(
                fn=_run_layout_mask_page_with_downloads,
                inputs=[session_state, image_state, layout_input, layout_threshold, layout_invert, layout_open_kernel, layout_close_kernel, layout_min_area, layout_region_mode, layout_morph_pixels],
                outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
                api_name="_run_layout_mask_page",
            )
            run_layout_region_event = run_layout_mask_event.then(
                fn=_load_layout_region_context,
                inputs=[layout_state],
                outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            run_layout_region_event.then(
                fn=_reset_layout_prompt_selection,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            save_layout_mask_btn.click(
                fn=_save_current_layout_mask,
                inputs=[layout_state],
                outputs=[layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
            )
            clear_layout_mask_event = clear_layout_mask_btn.click(
                fn=_clear_current_layout_mask_with_prompt_epoch,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
                api_name="_clear_current_layout_mask",
            )
            clear_layout_region_event = clear_layout_mask_event.then(
                fn=_clear_layout_region_context,
                inputs=[layout_state],
                outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            clear_layout_region_event.then(
                fn=_reset_layout_prompt_selection,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            layout_region_annotator.input(
                fn=_preview_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator],
                outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            save_layout_region_event = save_layout_region_btn.click(
                fn=_save_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_label],
                outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            save_layout_region_event.then(
                fn=_reset_layout_prompt_selection,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            layout_region_selector.input(
                fn=_select_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_selector],
                outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            delete_layout_region_event = delete_layout_region_btn.click(
                fn=_delete_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_selector],
                outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            delete_layout_region_event.then(
                fn=_reset_layout_prompt_selection,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )

            export_layout_regions_btn.click(
                fn=_export_layout_regions,
                inputs=[layout_state, layout_region_state],
                outputs=[layout_region_export_file, layout_region_status],
                concurrency_limit=1,
            )

            common = [image_upload, result_image, analysis_report, pcs_summary, pvs_summary, active_pvs, interaction_info, pvs_pending_count]
            use_current_layout_event = use_current_layout_btn.click(
                fn=_use_current_layout_mask,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            use_current_layout_event.then(
                fn=_load_layout_prompt_choices,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            load_layout_binary_event = load_layout_binary_btn.click(
                fn=_load_layout_binary_mask_png,
                inputs=[session_state, image_state, layout_binary_upload, layout_region_mode],
                outputs=[layout_state, layout_editor, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            load_layout_binary_event.then(
                fn=_reset_layout_prompt_selection,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            layout_prompt_mask_selector.input(
                fn=_select_layout_prompt_mask,
                inputs=[image_state, layout_state, layout_prompt_mask_selector],
                outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            update_layout_preview_btn.click(
                fn=_update_layout_preview_with_groups,
                inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor],
                outputs=[layout_state, image_upload, layout_editor, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            layout_editor.change(
                fn=_sync_layout_controls_from_editor_with_prompt_epoch,
                inputs=[layout_state, layout_editor],
                outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            reset_layout_btn.click(
                fn=_reset_layout_controls_with_prompt_epoch,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_pvs_info],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            create_from_layout_event = create_from_layout_btn.click(
                fn=_create_pvs_from_layout_selection,
                inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_prompt_mask_selector],
                outputs=[pvs_state, layout_state, layout_editor, layout_pvs_info, *common],
                show_progress_on=[result_image],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            create_from_layout_event.then(
                fn=_clear_template_match_outputs,
                inputs=None,
                outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )

            workspace_init_outputs = [
                image_state,
                pcs_state,
                pvs_state,
                prompt_state,
                pcs_bbox_selector,
                pvs_pending_bbox_selector,
                *common,
                export_file,
                layout_editor,
            ]
            source_image_event = source_image_upload.upload(
                fn=_source_upload_workspace,
                inputs=[source_image_upload, mode, session_state, layout_state],
                outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            source_clear_event = source_image_upload.clear(
                fn=_source_upload_workspace,
                inputs=[source_image_upload, mode, session_state, layout_state],
                outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            source_crop_overlay.input(
                fn=_record_source_crop_gesture,
                inputs=[source_image_state, source_crop_overlay],
                outputs=[source_image_state, source_crop_overlay, source_crop_status],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            apply_crop_event = apply_crop_btn.click(
                fn=_apply_source_crop,
                inputs=[source_image_state, mode, session_state, layout_state],
                outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            use_full_image_event = use_full_image_btn.click(
                fn=_use_full_source_image,
                inputs=[source_image_state, mode, session_state, layout_state],
                outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            for workspace_event in (source_image_event, source_clear_event, apply_crop_event, use_full_image_event):
                workspace_event.then(fn=_clear_pending_point_payload, inputs=None, outputs=[point_payload], concurrency_limit=1, concurrency_id="image-prepost-state")
                workspace_event.then(fn=_clear_bbox_polygon_payloads, inputs=None, outputs=[bbox_payload, polygon_payload], concurrency_limit=1, concurrency_id="image-prepost-state")
                workspace_event.then(
                    fn=_reset_layout_prompt_selection,
                    inputs=[image_state, layout_state],
                    outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
                    concurrency_limit=1,
                    concurrency_id="image-prepost-state",
                )
                workspace_event.then(
                    fn=_workspace_gesture_payload,
                    inputs=[image_state, mode, click_tool],
                    outputs=[workspace_gesture_overlay],
                    concurrency_limit=1,
                    concurrency_id="image-prepost-state",
                )
                workspace_event.then(
                    fn=_clear_template_match_outputs,
                    inputs=None,
                    outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
                    concurrency_limit=1,
                    concurrency_id="image-prepost-state",
                )
            workspace_gesture_overlay.input(
                fn=_workspace_gesture_input,
                inputs=[image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, workspace_gesture_overlay],
                outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, *common, workspace_gesture_overlay],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            run_template_match_btn.click(
                fn=_run_template_matching,
                inputs=[source_image_state, image_state, pvs_state, mode, match_threshold, expand_threshold, nms_threshold],
                outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
                concurrency_limit=1,
                concurrency_id="image-prepost-state",
            )
            for template_parameter in (match_threshold, expand_threshold, nms_threshold):
                template_parameter.change(
                    fn=_clear_template_match_outputs,
                    inputs=None,
                    outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
                    concurrency_limit=1,
                    concurrency_id="image-prepost-state",
                )
            finish_polygon_event = finish_polygon_btn.click(fn=_finish_native_polygon, inputs=[image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode], outputs=[prompt_state, polygon_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
            clear_prompt_btn.click(fn=_clear_prompt_selection, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, text_prompt, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            mode_event = mode.change(fn=_switch_mode_with_layout_editor, inputs=[mode, image_state, pcs_state, pvs_state, layout_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, click_tool, finish_polygon_btn, pcs_bbox_tools, pcs_panel, pvs_panel, pvs_action_panel, analysis_report_panel, pvs_layout_panel, layout_transform_panel, pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel, pcs_bbox_selector, pvs_pending_bbox_selector, layout_point_refine_panel, *common, layout_editor], concurrency_limit=1, concurrency_id="image-prepost-state")
            mode_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=1, concurrency_id="image-prepost-state")
            click_tool_event = click_tool.change(fn=_switch_click_tool, inputs=[click_tool, mode], outputs=[pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel], concurrency_limit=1)
            click_tool_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=1, concurrency_id="image-prepost-state")
            delete_selected_pcs_bbox_btn.click(fn=_delete_selected_pcs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pcs_bbox_selector], outputs=[pcs_state, pcs_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            run_pcs_btn.click(fn=_run_pcs, inputs=[image_state, pcs_state, pvs_state, mode, text_prompt, confidence_threshold], outputs=[pcs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            create_pvs_batch_event = create_pvs_batch_btn.click(fn=_create_pvs_from_pending_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
            delete_selected_pending_bbox_btn.click(fn=_delete_selected_pending_pvs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pvs_pending_bbox_selector], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            clear_pending_bbox_btn.click(fn=_clear_pending_pvs_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            pvs_point_event = pvs_point_btn.click(fn=_pvs_point_prompt, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, pvs_point_kind], outputs=[pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
            layout_point_event = layout_point_btn.click(fn=_layout_point_refine, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, layout_point_kind, prompt_state], outputs=[prompt_state, point_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
            active_pvs_event = active_pvs.change(fn=_set_active_pvs, inputs=[image_state, pcs_state, pvs_state, mode, active_pvs], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            undo_pvs_event = undo_pvs_btn.click(fn=_undo_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            delete_pvs_event = delete_pvs_btn.click(fn=_delete_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            accept_pvs_event = accept_pvs_btn.click(fn=_accept_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            for invalidating_event in (
                mode_event,
                finish_polygon_event,
                create_pvs_batch_event,
                pvs_point_event,
                layout_point_event,
                active_pvs_event,
                undo_pvs_event,
                delete_pvs_event,
                accept_pvs_event,
            ):
                invalidating_event.then(
                    fn=_clear_template_match_outputs,
                    inputs=None,
                    outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
                    concurrency_limit=1,
                    concurrency_id="image-prepost-state",
                )
            export_pcs_btn.click(fn=_export_pcs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            export_pvs_btn.click(fn=_export_pvs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
            submit_feedback_btn.click(fn=_submit_feedback, inputs=[image_state, pcs_state, pvs_state, mode, feedback_rating, feedback_tags, feedback_comment], outputs=common, concurrency_limit=1, concurrency_id="image-prepost-state")
        gr.Markdown("---\n<div style='text-align:center;color:#718096;font-size:0.9em;'>Powered by SAM3</div>")
    return demo
# --- end PCS/PVS single-workspace override ---


def main():
    """主函数"""
    # 检查模型文件
    model_dir = current_dir / "models"
    if not model_dir.exists():
        print(f"创建模型目录: {model_dir}")
        model_dir.mkdir(exist_ok=True)

    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ 模型文件缺失")
        print(f"请确保以下文件存在:\n1. {checkpoint_path}\n2. {bpe_path}")

        response = input("是否尝试自动下载模型文件？(y/n): ").lower().strip()
        if response == "y":
            try:
                import download_models

                download_models.main()
            except Exception as e:
                print(f"自动下载失败: {e}")
                return
        else:
            return

    print("🚀 正在启动 SAM3 交互式视觉工作台...")
    _public_downloads.ensure_public_download_dirs(public_download_dir)
    _prune_public_downloads()
    demo = create_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=False,
        allowed_paths=_gradio_allowed_paths(),
        blocked_paths=_gradio_blocked_paths(),
    )


if __name__ == "__main__":
    main()
