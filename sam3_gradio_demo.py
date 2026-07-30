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
    _PVS_PREDICT_LOCK,
    _predict_inst,
    _prompt_mask_size,
    _polygon_lowres_logits,
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


















from sam3_demo.pcs_pvs_callbacks import (
    _workspace_gesture_payload_impl,
    _workspace_select_impl,
    _workspace_gesture_input_impl,
    _apply_polygon_to_pvs_impl,
    _finish_native_polygon_impl,
    _clear_prompt_selection_impl,
    _delete_selected_pcs_bbox_impl,
    _run_pcs_impl,
    _create_pvs_from_pending_boxes_impl,
    _delete_selected_pending_pvs_bbox_impl,
    _clear_pending_pvs_boxes_impl,
    _set_active_pvs_impl,
    _refine_active_pvs_with_point_impl,
    _pvs_point_prompt_impl,
    _undo_pvs_impl,
    _delete_pvs_impl,
    _accept_pvs_impl,
)

def _workspace_gesture_payload(image_state, mode=None, click_tool=None, status=""):
    return _workspace_gesture_payload_impl(
        {
            '_click_tool_key': _click_tool_key,
            '_image_gesture_payload': _image_gesture_payload,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_is_pcs_mode': _is_pcs_mode,
        },
        image_state,
        mode,
        click_tool,
        status,
    )



















from sam3_demo.layout.mask_callbacks import (
    _layout_cache_key_impl,
    _layout_disk_dir_impl,
    _layout_cache_get_impl,
    _restore_layout_cache_from_disk_impl,
    _write_layout_meta_impl,
    _layout_cache_put_impl,
    _clear_layout_cache_impl,
    _normalize_layout_morph_pixels_impl,
    _apply_layout_mask_morphology_impl,
    _binarize_layout_image_impl,
    _filter_layout_components_impl,
    _layout_mask_contours_impl,
    _layout_mask_to_preview_impl,
    _layout_contour_overlay_impl,
    _save_layout_mask_files_impl,
    _layout_mask_to_editor_image_impl,
    _layout_editor_empty_impl,
    _layout_editor_payload_impl,
    _layout_editor_transform_impl,
    _layout_group_control_values_impl,
    _commit_layout_group_transforms_impl,
    _validate_layout_group_transform_snapshot_impl,
    _sync_layout_controls_from_editor_impl,
    _sync_layout_controls_from_editor_with_prompt_epoch_impl,
    _run_layout_mask_page_impl,
    _run_layout_mask_page_with_downloads_impl,
    _save_current_layout_mask_impl,
    _clear_current_layout_mask_impl,
    _clear_current_layout_mask_with_prompt_epoch_impl,
    _layout_numeric_controls_changed_impl,
    _commit_layout_transform_impl,
    _transform_layout_mask_impl,
    _layout_mask_to_overlay_impl,
    _update_layout_preview_impl,
    _update_layout_preview_with_groups_impl,
    _reset_layout_controls_impl,
    _reset_layout_controls_with_prompt_epoch_impl,
)

def _layout_cache_key(session_id, layout_id):
    return _layout_cache_key_impl(
        {
            '_layout_tx': _layout_tx,
        },
        session_id,
        layout_id,
    )


def _layout_disk_dir(session_id, layout_id):
    return _layout_disk_dir_impl(
        {
            '_layout_tx': _layout_tx,
            'runtime_layout_dir': runtime_layout_dir,
        },
        session_id,
        layout_id,
    )


def _layout_cache_get(layout_state_or_id, session_id=None):
    return _layout_cache_get_impl(
        {
            '_LAYOUT_CACHE': _LAYOUT_CACHE,
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_key': _layout_cache_key,
            '_restore_layout_cache_from_disk': _restore_layout_cache_from_disk,
        },
        layout_state_or_id,
        session_id,
    )


def _restore_layout_cache_from_disk(session_id, layout_id):
    return _restore_layout_cache_from_disk_impl(
        {
            'Image': Image,
            '_layout_disk_dir': _layout_disk_dir,
            '_layout_mask_to_preview': _layout_mask_to_preview,
            '_layout_tx': _layout_tx,
            'cv2': cv2,
            'json': json,
            'np': np,
        },
        session_id,
        layout_id,
    )


def _write_layout_meta(cached):
    return _write_layout_meta_impl(
        {
            'Path': Path,
            'json': json,
        },
        cached,
    )


def _layout_cache_put(session_id, layout_id, source_image, source_mask, contours, binarize_params, mask_path=None, contour_json_path=None, overlay_path=None, layout_meta_path=None):
    return _layout_cache_put_impl(
        {
            '_LAYOUT_CACHE': _LAYOUT_CACHE,
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_key': _layout_cache_key,
            '_layout_tx': _layout_tx,
            '_pil_image': _pil_image,
            '_write_layout_meta': _write_layout_meta,
            'np': np,
        },
        session_id,
        layout_id,
        source_image,
        source_mask,
        contours,
        binarize_params,
        mask_path,
        contour_json_path,
        overlay_path,
        layout_meta_path,
    )


def _clear_layout_cache(layout_state=None):
    return _clear_layout_cache_impl(
        {
            '_LAYOUT_CACHE': _LAYOUT_CACHE,
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_key': _layout_cache_key,
        },
        layout_state,
    )
























































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


from sam3_demo.rendering import (
    _workspace_image_impl,
    _result_image_impl,
    _pcs_choice_update_impl,
    _status_label_impl,
    _pvs_choice_update_impl,
    _pvs_pending_count_text_impl,
    _pcs_summary_impl,
    _pvs_summary_impl,
    _analysis_report_impl,
    _view_impl,
)

def _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None, layout_state=None):
    return _workspace_image_impl(
        {
            '_overlay': _overlay,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        prompt_state,
        layout_state,
    )




def _result_image(image_state, pcs_state, pvs_state, mode):
    return _result_image_impl(
        {
            '_instances_for_mode': _instances_for_mode,
            '_overlay': _overlay,
            '_result_placeholder': _result_placeholder,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )



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
    return _workspace_select_impl(
        {
            '_append_pcs_bbox_sample': _append_pcs_bbox_sample,
            '_append_pvs_pending_bbox': _append_pvs_pending_bbox,
            '_click_tool_key': _click_tool_key,
            '_event_point': _event_point,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_is_pcs_mode': _is_pcs_mode,
            '_new_prompt_state': _new_prompt_state,
            '_norm_box': _norm_box,
            '_payload_json': _payload_json,
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        click_tool,
        pcs_bbox_kind,
        prompt_state,
        evt,
    )


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
    return _workspace_gesture_input_impl(
        {
            '_GestureSelectEvent': _GestureSelectEvent,
            '_append_pcs_bbox_sample': _append_pcs_bbox_sample,
            '_append_pvs_pending_bbox': _append_pvs_pending_bbox,
            '_click_tool_key': _click_tool_key,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_is_pcs_mode': _is_pcs_mode,
            '_new_prompt_state': _new_prompt_state,
            '_norm_box': _norm_box,
            '_payload_json': _payload_json,
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_validate_gesture_intent': _validate_gesture_intent,
            '_view': _view,
            '_workspace_gesture_payload': _workspace_gesture_payload,
            '_workspace_select': _workspace_select,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        click_tool,
        pcs_bbox_kind,
        prompt_state,
        gesture_payload,
    )


def _apply_polygon_to_pvs(image_state, pvs_state, polygon, polygon_action="refine", combine_mode="replace", progress=None):
    return _apply_polygon_to_pvs_impl(
        {
            '_append_prompt_history': _append_prompt_history,
            '_best': _best,
            '_combine_logits': _combine_logits,
            '_fresh_state': _fresh_state,
            '_history_snapshot': _history_snapshot,
            '_make_inst': _make_inst,
            '_mask_box': _mask_box,
            '_polygon_action_key': _polygon_action_key,
            '_polygon_combine_key': _polygon_combine_key,
            '_polygon_lowres_logits': _polygon_lowres_logits,
            '_predict_inst': _predict_inst,
            '_pvs_progress': _pvs_progress,
            '_workspace': _workspace,
        },
        image_state,
        pvs_state,
        polygon,
        polygon_action,
        combine_mode,
        progress,
    )


def _finish_native_polygon(image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action="create", polygon_combine_mode="replace", progress=gr.Progress(track_tqdm=False)):
    return _finish_native_polygon_impl(
        {
            '_apply_polygon_to_pvs': _apply_polygon_to_pvs,
            '_is_pvs_manual_mode': _is_pvs_manual_mode,
            '_new_prompt_state': _new_prompt_state,
            '_payload_json': _payload_json,
            '_pvs_progress': _pvs_progress,
            '_view': _view,
        },
        image_state,
        prompt_state,
        pcs_state,
        pvs_state,
        mode,
        polygon_action,
        polygon_combine_mode,
        progress,
    )


def _clear_prompt_selection(image_state, pcs_state, pvs_state, mode):
    return _clear_prompt_selection_impl(
        {
            '_clear_pvs_pending_bboxes': _clear_pvs_pending_bboxes,
            '_is_pcs_mode': _is_pcs_mode,
            '_is_pvs_manual_mode': _is_pvs_manual_mode,
            '_new_prompt_state': _new_prompt_state,
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )


def _clear_pending_point_payload():
    return ""


def _clear_bbox_polygon_payloads():
    return "", ""


def _pcs_choice_update(pcs_state):
    return _pcs_choice_update_impl(
        {
            '_active_instances': _active_instances,
        },
        pcs_state,
    )


def _status_label(status):
    return _status_label_impl(
        {
        },
        status,
    )


def _pvs_choice_update(pvs_state):
    return _pvs_choice_update_impl(
        {
            '_active_instances': _active_instances,
            '_status_label': _status_label,
        },
        pvs_state,
    )


def _pvs_pending_count_text(pvs_state):
    return _pvs_pending_count_text_impl(
        {
            '_sync_pvs_pending_boxes_from_records': _sync_pvs_pending_boxes_from_records,
        },
        pvs_state,
    )


def _pcs_summary(pcs_state):
    return _pcs_summary_impl(
        {
            '_active_instances': _active_instances,
            '_pcs_bbox_records': _pcs_bbox_records,
            '_sync_pcs_boxes_from_records': _sync_pcs_boxes_from_records,
        },
        pcs_state,
    )


def _pvs_summary(pvs_state):
    return _pvs_summary_impl(
        {
            '_active_instances': _active_instances,
            '_status_label': _status_label,
        },
        pvs_state,
    )

def _analysis_report(pcs_state, pvs_state, mode, info):
    return _analysis_report_impl(
        {
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_is_pcs_mode': _is_pcs_mode,
            '_pcs_summary': _pcs_summary,
            '_pvs_summary': _pvs_summary,
        },
        pcs_state,
        pvs_state,
        mode,
        info,
    )
def _view(image_state, pcs_state, pvs_state, mode, info, prompt_state=None, layout_state=None):
    return _view_impl(
        {
            '_analysis_report': _analysis_report,
            '_pcs_summary': _pcs_summary,
            '_pvs_choice_update': _pvs_choice_update,
            '_pvs_pending_count_text': _pvs_pending_count_text,
            '_pvs_summary': _pvs_summary,
            '_result_image': _result_image,
            '_workspace_image': _workspace_image,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        info,
        prompt_state,
        layout_state,
    )


from sam3_demo.image_prepost_callbacks import (
    _init_workspace_impl,
    _init_workspace_with_layout_editor_impl,
    _attach_source_provenance_impl,
    _source_upload_workspace_impl,
    _record_source_crop_gesture_impl,
    _crop_failure_outputs_impl,
    _apply_source_crop_impl,
    _use_full_source_image_impl,
    _clear_template_match_outputs_impl,
    _publish_template_match_export_impl,
    _run_template_matching_impl,
)

def _init_workspace(input_image, mode, session_state=None):
    return _init_workspace_impl(
        {
            '_WORKSPACE_CACHE': _WORKSPACE_CACHE,
            '_WORKSPACE_CACHE_LOCK': _WORKSPACE_CACHE_LOCK,
            '_clear_workspace_cache': _clear_workspace_cache,
            '_new_pcs_state': _new_pcs_state,
            '_new_prompt_state': _new_prompt_state,
            '_new_pvs_state': _new_pvs_state,
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pil_image': _pil_image,
            '_prune_workspace_cache': _prune_workspace_cache,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_release_workspace_memory': _release_workspace_memory,
            '_session_id_from_state': _session_id_from_state,
            '_view': _view,
            'image_predictor': image_predictor,
        },
        input_image,
        mode,
        session_state,
    )


def _init_workspace_with_layout_editor(input_image, mode, session_state=None, layout_state=None):
    return _init_workspace_with_layout_editor_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_init_workspace': _init_workspace,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_editor_payload': _layout_editor_payload,
        },
        input_image,
        mode,
        session_state,
        layout_state,
    )


def _attach_source_provenance(init_result, source_state):
    return _attach_source_provenance_impl(
        {
            '_WORKSPACE_CACHE': _WORKSPACE_CACHE,
            '_WORKSPACE_CACHE_LOCK': _WORKSPACE_CACHE_LOCK,
        },
        init_result,
        source_state,
    )


def _source_upload_workspace(input_image, mode, session_state=None, layout_state=None):
    return _source_upload_workspace_impl(
        {
            '_attach_source_provenance': _attach_source_provenance,
            '_clear_source_image_cache': _clear_source_image_cache,
            '_init_workspace_with_layout_editor': _init_workspace_with_layout_editor,
            '_new_source_image_state': _new_source_image_state,
            '_session_id_from_state': _session_id_from_state,
            '_source_gesture_payload': _source_gesture_payload,
            '_source_image_cache_get': _source_image_cache_get,
            '_source_image_cache_put': _source_image_cache_put,
        },
        input_image,
        mode,
        session_state,
        layout_state,
    )


def _record_source_crop_gesture(source_state, gesture_payload):
    return _record_source_crop_gesture_impl(
        {
            '_source_gesture_payload': _source_gesture_payload,
            '_validate_gesture_intent': _validate_gesture_intent,
        },
        source_state,
        gesture_payload,
    )


def _crop_failure_outputs(source_state, status):
    return _crop_failure_outputs_impl(
        {
            '_source_gesture_payload': _source_gesture_payload,
        },
        source_state,
        status,
    )


def _apply_source_crop(source_state, mode, session_state=None, layout_state=None):
    return _apply_source_crop_impl(
        {
            '_attach_source_provenance': _attach_source_provenance,
            '_crop_failure_outputs': _crop_failure_outputs,
            '_init_workspace_with_layout_editor': _init_workspace_with_layout_editor,
            '_source_gesture_payload': _source_gesture_payload,
            '_source_image_cache_get': _source_image_cache_get,
        },
        source_state,
        mode,
        session_state,
        layout_state,
    )


def _use_full_source_image(source_state, mode, session_state=None, layout_state=None):
    return _use_full_source_image_impl(
        {
            '_attach_source_provenance': _attach_source_provenance,
            '_crop_failure_outputs': _crop_failure_outputs,
            '_init_workspace_with_layout_editor': _init_workspace_with_layout_editor,
            '_source_gesture_payload': _source_gesture_payload,
            '_source_image_cache_get': _source_image_cache_get,
        },
        source_state,
        mode,
        session_state,
        layout_state,
    )


def _clear_template_match_outputs(status="请先完成智能分割并选择当前 PVS 实例"):
    return _clear_template_match_outputs_impl(
        {
            '_new_template_match_state': _new_template_match_state,
        },
        status,
    )


def _publish_template_match_export(source_image, workflow, source_state, image_state):
    return _publish_template_match_export_impl(
        {
            '_publish_segmentation_zip': _publish_segmentation_zip,
            'runtime_export_dir': runtime_export_dir,
        },
        source_image,
        workflow,
        source_state,
        image_state,
    )


def _run_template_matching(
    source_state,
    image_state,
    pvs_state,
    mode,
    match_threshold,
    expand_threshold,
    nms_threshold,
):
    return _run_template_matching_impl(
        {
            '_clear_template_match_outputs': _clear_template_match_outputs,
            '_is_pvs_pool_mode': _is_pvs_pool_mode,
            '_publish_template_match_export': _publish_template_match_export,
            '_source_image_cache_get': _source_image_cache_get,
            '_workspace': _workspace,
        },
        source_state,
        image_state,
        pvs_state,
        mode,
        match_threshold,
        expand_threshold,
        nms_threshold,
    )


def _delete_selected_pcs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    return _delete_selected_pcs_bbox_impl(
        {
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pcs_bbox_records': _pcs_bbox_records,
            '_reset_pcs_predictions': _reset_pcs_predictions,
            '_sync_pcs_boxes_from_records': _sync_pcs_boxes_from_records,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        selected_bbox_id,
    )


def _run_pcs(image_state, pcs_state, pvs_state, mode, text_prompt, threshold):
    return _run_pcs_impl(
        {
            '_fresh_state': _fresh_state,
            '_make_inst': _make_inst,
            '_norm_box': _norm_box,
            '_view': _view,
            '_workspace': _workspace,
            '_xyxy_to_cxcywh_norm': _xyxy_to_cxcywh_norm,
            'image_predictor': image_predictor,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        text_prompt,
        threshold,
    )


def _create_pvs_from_pending_boxes(image_state, pcs_state, pvs_state, mode, progress=gr.Progress(track_tqdm=False)):
    return _create_pvs_from_pending_boxes_impl(
        {
            '_best': _best,
            '_fresh_state': _fresh_state,
            '_make_inst': _make_inst,
            '_mask_box': _mask_box,
            '_predict_inst': _predict_inst,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_pvs_progress': _pvs_progress,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        progress,
    )


def _delete_selected_pending_pvs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    return _delete_selected_pending_pvs_bbox_impl(
        {
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_pvs_pending_bbox_records': _pvs_pending_bbox_records,
            '_sync_pvs_pending_boxes_from_records': _sync_pvs_pending_boxes_from_records,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        selected_bbox_id,
    )


def _clear_pending_pvs_boxes(image_state, pcs_state, pvs_state, mode):
    return _clear_pending_pvs_boxes_impl(
        {
            '_clear_pvs_pending_bboxes': _clear_pvs_pending_bboxes,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )


def _set_active_pvs(image_state, pcs_state, pvs_state, mode, selected_id):
    return _set_active_pvs_impl(
        {
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        selected_id,
    )

def _refine_active_pvs_with_point(image_state, pvs_state, point, point_label, progress=None):
    return _refine_active_pvs_with_point_impl(
        {
            '_append_prompt_history': _append_prompt_history,
            '_fresh_state': _fresh_state,
            '_history_snapshot': _history_snapshot,
            '_mask_box': _mask_box,
            '_predict_inst': _predict_inst,
            '_prompt_mask_size': _prompt_mask_size,
            '_pvs_progress': _pvs_progress,
        },
        image_state,
        pvs_state,
        point,
        point_label,
        progress,
    )

def _pvs_point_prompt(image_state, pcs_state, pvs_state, mode, point_payload, point_kind, progress=gr.Progress(track_tqdm=False)):
    return _pvs_point_prompt_impl(
        {
            '_best': _best,
            '_fresh_state': _fresh_state,
            '_make_inst': _make_inst,
            '_mask_box': _mask_box,
            '_point_from_payload': _point_from_payload,
            '_predict_inst': _predict_inst,
            '_pvs_progress': _pvs_progress,
            '_refine_active_pvs_with_point': _refine_active_pvs_with_point,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        point_payload,
        point_kind,
        progress,
    )


from sam3_demo.layout.prompt_callbacks import (
    _layout_point_refine_impl,
    _switch_mode_impl,
    _switch_mode_with_layout_editor_impl,
    _layout_prompt_epoch_key_impl,
    _layout_prompt_epoch_snapshot_impl,
    _advance_layout_prompt_epoch_impl,
    _reset_layout_prompt_selection_state_impl,
    _layout_prompt_selection_token_impl,
    _parse_layout_prompt_selection_impl,
    _normalize_layout_prompt_checkbox_selection_impl,
    _layout_prompt_label_counts_impl,
    _layout_label_choice_text_impl,
    _layout_prompt_class_counts_impl,
    _layout_prompt_choice_update_impl,
    _load_layout_prompt_region_document_impl,
    _layout_prompt_label_records_impl,
    _layout_prompt_class_records_impl,
    _layout_prompt_region_records_impl,
    _layout_prompt_display_mask_impl,
    _layout_prompt_group_id_impl,
    _layout_prompt_group_data_impl,
    _layout_group_transform_impl,
    _layout_prompt_group_payload_impl,
    _mask_to_lowres_logits_impl,
    _validate_layout_prompt_mask_impl,
    _layout_transformed_mask_for_image_impl,
    _layout_prompt_metadata_impl,
    _create_pvs_from_layout_mask_impl,
    _layout_state_summary_impl,
    _load_layout_binary_mask_png_impl,
    _use_current_layout_mask_impl,
    _pvs_creation_commit_token_impl,
    _selected_pvs_candidate_impl,
    _layout_prompt_region_fingerprint_impl,
    _validate_layout_transform_snapshot_impl,
    _load_layout_prompt_choices_impl,
    _select_layout_prompt_mask_impl,
    _reset_layout_prompt_selection_impl,
    _create_pvs_from_layout_selection_impl,
)

def _layout_point_refine(image_state, pcs_state, pvs_state, mode, point_payload, point_kind, prompt_state, progress=gr.Progress(track_tqdm=False)):
    return _layout_point_refine_impl(
        {
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_new_prompt_state': _new_prompt_state,
            '_point_from_payload': _point_from_payload,
            '_pvs_progress': _pvs_progress,
            '_refine_active_pvs_with_point': _refine_active_pvs_with_point,
            '_view': _view,
            'gr': gr,
            'np': np,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        point_payload,
        point_kind,
        prompt_state,
        progress,
    )

def _undo_pvs(image_state, pcs_state, pvs_state, mode):
    return _undo_pvs_impl(
        {
            '_active_instances': _active_instances,
            '_restore': _restore,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )


def _delete_pvs(image_state, pcs_state, pvs_state, mode):
    return _delete_pvs_impl(
        {
            '_active_instances': _active_instances,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )


def _accept_pvs(image_state, pcs_state, pvs_state, mode):
    return _accept_pvs_impl(
        {
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
    )


from sam3_demo.feedback_export import (
    _history_json_impl,
    _latest_layout_prompt_from_instances_impl,
    _reconstruct_frozen_layout_prompt_mask_impl,
    _write_feedback_layout_artifacts_impl,
    _submit_feedback_impl,
    _export_pool_impl,
    _export_pcs_impl,
    _export_pvs_impl,
)

def _history_json(history):
    return _history_json_impl(
        {
        },
        history,
    )



def _latest_layout_prompt_from_instances(instances):
    return _latest_layout_prompt_from_instances_impl(
        {
        },
        instances,
    )


def _reconstruct_frozen_layout_prompt_mask(layout_prompt):
    return _reconstruct_frozen_layout_prompt_mask_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
        },
        layout_prompt,
    )


def _write_feedback_layout_artifacts(sample_dir, layout_prompt):
    return _write_feedback_layout_artifacts_impl(
        {
            '_reconstruct_frozen_layout_prompt_mask': _reconstruct_frozen_layout_prompt_mask,
        },
        sample_dir,
        layout_prompt,
    )

def _submit_feedback(image_state, pcs_state, pvs_state, mode, rating, feedback_tags, feedback_comment):
    return _submit_feedback_impl(
        {
            '_FEEDBACK_WRITE_LOCK': _FEEDBACK_WRITE_LOCK,
            '_active_instances': _active_instances,
            '_history_json': _history_json,
            '_is_pcs_mode': _is_pcs_mode,
            '_is_pvs_pool_mode': _is_pvs_pool_mode,
            '_latest_layout_prompt_from_instances': _latest_layout_prompt_from_instances,
            '_result_image': _result_image,
            '_view': _view,
            '_workspace': _workspace,
            '_write_feedback_layout_artifacts': _write_feedback_layout_artifacts,
            'runtime_feedback_dir': runtime_feedback_dir,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        rating,
        feedback_tags,
        feedback_comment,
    )


def _export_pool(image_state, pcs_state, pvs_state, mode, pool_name, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    return _export_pool_impl(
        {
            '_active_instances': _active_instances,
            '_history_json': _history_json,
            '_overlay': _overlay,
            '_publish_segmentation_zip': _publish_segmentation_zip,
            '_workspace': _workspace,
            'compare_with_coco': compare_with_coco,
            'create_prediction_coco_json': create_prediction_coco_json,
            'mask_to_polygons': mask_to_polygons,
            'runtime_export_dir': runtime_export_dir,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        pool_name,
        coco_dataset,
        coco_image_name,
        coco_split,
        coco_eval_scope,
        annotation_json_file,
    )


def _export_pcs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    return _export_pcs_impl(
        {
            '_export_pool': _export_pool,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        coco_dataset,
        coco_image_name,
        coco_split,
        coco_eval_scope,
        annotation_json_file,
    )


def _export_pvs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    return _export_pvs_impl(
        {
            '_export_pool': _export_pool,
            '_view': _view,
        },
        image_state,
        pcs_state,
        pvs_state,
        mode,
        coco_dataset,
        coco_image_name,
        coco_split,
        coco_eval_scope,
        annotation_json_file,
    )


def _switch_mode(mode, image_state, pcs_state, pvs_state):
    return _switch_mode_impl(
        {
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_is_pcs_mode': _is_pcs_mode,
            '_is_pvs_manual_mode': _is_pvs_manual_mode,
            '_is_pvs_pool_mode': _is_pvs_pool_mode,
            '_new_prompt_state': _new_prompt_state,
            '_pcs_bbox_choices': _pcs_bbox_choices,
            '_pvs_pending_bbox_choices': _pvs_pending_bbox_choices,
            '_view': _view,
            'gr': gr,
        },
        mode,
        image_state,
        pcs_state,
        pvs_state,
    )
def _switch_mode_with_layout_editor(mode, image_state, pcs_state, pvs_state, layout_state):
    return _switch_mode_with_layout_editor_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_layout_editor_payload': _layout_editor_payload,
            '_switch_mode': _switch_mode,
            'gr': gr,
        },
        mode,
        image_state,
        pcs_state,
        pvs_state,
        layout_state,
    )


def _switch_click_tool(click_tool, mode):
    tool = _click_tool_key(click_tool)
    is_pvs = _is_pvs_manual_mode(mode)
    return (
        gr.update(visible=is_pvs and tool == "bbox"),
        gr.update(visible=is_pvs and tool == "point"),
        gr.update(visible=is_pvs and tool == "polygon"),
    )


def _normalize_layout_morph_pixels(value):
    return _normalize_layout_morph_pixels_impl(
        {
            '_LAYOUT_MASK_MORPH_LIMIT_PX': _LAYOUT_MASK_MORPH_LIMIT_PX,
            'np': np,
        },
        value,
    )


def _apply_layout_mask_morphology(mask, morph_pixels=0):
    return _apply_layout_mask_morphology_impl(
        {
            '_normalize_layout_morph_pixels': _normalize_layout_morph_pixels,
            'cv2': cv2,
            'np': np,
        },
        mask,
        morph_pixels,
    )


def _binarize_layout_image(input_image, threshold=12, invert=False, open_kernel=0, close_kernel=0, morph_pixels=0):
    return _binarize_layout_image_impl(
        {
            '_apply_layout_mask_morphology': _apply_layout_mask_morphology,
            '_layout_extract_mask': _layout_extract_mask,
            '_pil_image': _pil_image,
            'cv2': cv2,
            'np': np,
        },
        input_image,
        threshold,
        invert,
        open_kernel,
        close_kernel,
        morph_pixels,
    )


def _filter_layout_components(mask, min_component_area=0, region_mode="all"):
    return _filter_layout_components_impl(
        {
            'cv2': cv2,
            'np': np,
        },
        mask,
        min_component_area,
        region_mode,
    )


def _layout_mask_contours(mask):
    return _layout_mask_contours_impl(
        {
            'cv2': cv2,
            'np': np,
        },
        mask,
    )


def _layout_mask_to_preview(mask):
    return _layout_mask_to_preview_impl(
        {
            'Image': Image,
            'np': np,
        },
        mask,
    )


def _layout_contour_overlay(image, mask, contours):
    return _layout_contour_overlay_impl(
        {
            'Image': Image,
            '_pil_image': _pil_image,
            'cv2': cv2,
            'np': np,
        },
        image,
        mask,
        contours,
    )


def _save_layout_mask_files(session_state, source_image, mask, contours, params):
    return _save_layout_mask_files_impl(
        {
            '_layout_cache_put': _layout_cache_put,
            '_layout_contour_overlay': _layout_contour_overlay,
            '_layout_disk_dir': _layout_disk_dir,
            '_new_layout_state': _new_layout_state,
            '_pil_image': _pil_image,
            '_session_id_from_state': _session_id_from_state,
            'cv2': cv2,
            'json': json,
            'np': np,
            'time': time,
            'uuid': uuid,
        },
        session_state,
        source_image,
        mask,
        contours,
        params,
    )




def _layout_mask_to_editor_image(mask):
    return _layout_mask_to_editor_image_impl(
        {
            'Image': Image,
            'np': np,
        },
        mask,
    )


from sam3_demo.layout.region_callbacks import (
    _layout_region_png_data_url_impl,
    _new_layout_region_state_impl,
    _layout_region_state_from_document_impl,
    _layout_region_editor_empty_impl,
    _layout_region_identity_impl,
    _layout_region_state_matches_identity_impl,
    _validate_layout_region_state_identity_impl,
    _layout_region_client_intent_impl,
    _validate_layout_region_intent_impl,
    _layout_region_source_image_impl,
    _layout_region_summaries_impl,
    _layout_region_choice_update_impl,
    _layout_region_category_update_impl,
    _layout_region_label_update_impl,
    _layout_region_editor_payload_impl,
    _load_layout_region_context_impl,
    _clear_layout_region_context_impl,
    _preview_layout_region_impl,
    _select_layout_region_impl,
    _layout_region_latest_values_impl,
    _save_layout_region_impl,
    _delete_layout_region_impl,
    _export_layout_regions_impl,
)

def _layout_region_png_data_url(image):
    return _layout_region_png_data_url_impl(
        {
            '_sam3_base64': _sam3_base64,
            'io': io,
        },
        image,
    )


def _new_layout_region_state():
    return _new_layout_region_state_impl(
        {
        },
    )


def _layout_region_state_from_document(document, selected_region_id=None):
    return _layout_region_state_from_document_impl(
        {
            '_layout_regions': _layout_regions,
        },
        document,
        selected_region_id,
    )


def _layout_region_editor_empty(status="请先生成版图 binary mask"):
    return _layout_region_editor_empty_impl(
        {
        },
        status,
    )


def _layout_region_identity(layout_state):
    return _layout_region_identity_impl(
        {
            '_layout_regions': _layout_regions,
        },
        layout_state,
    )


def _layout_region_state_matches_identity(
    region_state, session_id, layout_id, source_mask_hash
):
    return _layout_region_state_matches_identity_impl(
        {
        },
        region_state,
        session_id,
        layout_id,
        source_mask_hash,
    )


def _validate_layout_region_state_identity(layout_state, region_state):
    return _validate_layout_region_state_identity_impl(
        {
            '_layout_region_identity': _layout_region_identity,
            '_layout_regions': _layout_regions,
        },
        layout_state,
        region_state,
    )


def _layout_region_client_intent(payload):
    return _layout_region_client_intent_impl(
        {
        },
        payload,
    )


def _validate_layout_region_intent(
    layout_state,
    intent,
    *,
    region_state=None,
    require_lasso=False,
):
    return _validate_layout_region_intent_impl(
        {
            '_layout_region_identity': _layout_region_identity,
            '_layout_regions': _layout_regions,
            '_validate_layout_region_state_identity': _validate_layout_region_state_identity,
        },
        layout_state,
        intent,
        region_state,
        require_lasso,
    )


def _layout_region_source_image(session_id, layout_id, source_mask):
    return _layout_region_source_image_impl(
        {
            'Image': Image,
            '_layout_mask_to_preview': _layout_mask_to_preview,
            'runtime_layout_dir': runtime_layout_dir,
        },
        session_id,
        layout_id,
        source_mask,
    )


def _layout_region_summaries(document):
    return _layout_region_summaries_impl(
        {
            '_layout_regions': _layout_regions,
        },
        document,
    )


class _LayoutPromptConflictError(RuntimeError):
    pass


def _layout_prompt_epoch_key(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    return _layout_prompt_epoch_key_impl(
        {
        },
        image_state,
        layout_state,
        session_state,
    )


def _layout_prompt_epoch_snapshot(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    return _layout_prompt_epoch_snapshot_impl(
        {
            '_LAYOUT_PROMPT_EPOCHS': _LAYOUT_PROMPT_EPOCHS,
            '_LAYOUT_PROMPT_EPOCH_LOCK': _LAYOUT_PROMPT_EPOCH_LOCK,
            '_layout_prompt_epoch_key': _layout_prompt_epoch_key,
        },
        image_state,
        layout_state,
        session_state,
    )


def _advance_layout_prompt_epoch(
    image_state=None,
    layout_state=None,
    session_state=None,
):
    return _advance_layout_prompt_epoch_impl(
        {
            '_LAYOUT_PROMPT_EPOCHS': _LAYOUT_PROMPT_EPOCHS,
            '_LAYOUT_PROMPT_EPOCH_LOCK': _LAYOUT_PROMPT_EPOCH_LOCK,
            '_layout_prompt_epoch_key': _layout_prompt_epoch_key,
        },
        image_state,
        layout_state,
        session_state,
    )

def _reset_layout_prompt_selection_state(layout_state):
    return _reset_layout_prompt_selection_state_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
        },
        layout_state,
    )


def _layout_prompt_selection_token(scope, region_id=None):
    return _layout_prompt_selection_token_impl(
        {
            '_LAYOUT_PROMPT_LABEL_PREFIX': _LAYOUT_PROMPT_LABEL_PREFIX,
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_CLASS': _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
        },
        scope,
        region_id,
    )


def _parse_layout_prompt_selection(value):
    return _parse_layout_prompt_selection_impl(
        {
            '_LAYOUT_PROMPT_LABEL_PREFIX': _LAYOUT_PROMPT_LABEL_PREFIX,
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
        },
        value,
    )


def _normalize_layout_prompt_checkbox_selection(value, layout_state):
    return _normalize_layout_prompt_checkbox_selection_impl(
        {
            '_LAYOUT_PROMPT_LABEL_PREFIX': _LAYOUT_PROMPT_LABEL_PREFIX,
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
        },
        value,
        layout_state,
    )


def _layout_prompt_label_counts(document):
    return _layout_prompt_label_counts_impl(
        {
            '_layout_regions': _layout_regions,
        },
        document,
    )

def _layout_label_choice_text(record, label_counts):
    return _layout_label_choice_text_impl(
        {
            '_layout_regions': _layout_regions,
        },
        record,
        label_counts,
    )



def _layout_prompt_class_counts(document):
    return _layout_prompt_class_counts_impl(
        {
            '_layout_prompt_label_counts': _layout_prompt_label_counts,
        },
        document,
    )


def _layout_prompt_choice_update(document=None, selected_value=None):
    return _layout_prompt_choice_update_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_layout_label_choice_text': _layout_label_choice_text,
            '_layout_prompt_label_counts': _layout_prompt_label_counts,
            '_layout_prompt_selection_token': _layout_prompt_selection_token,
            '_layout_regions': _layout_regions,
            'gr': gr,
        },
        document,
        selected_value,
    )


def _load_layout_prompt_region_document(layout_state, expected_revision=None):
    return _load_layout_prompt_region_document_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_identity': _layout_region_identity,
            '_layout_regions': _layout_regions,
        },
        layout_state,
        expected_revision,
    )


def _layout_prompt_label_records(document, labels):
    return _layout_prompt_label_records_impl(
        {
            '_layout_regions': _layout_regions,
        },
        document,
        labels,
    )


def _layout_prompt_class_records(document, class_label):
    return _layout_prompt_class_records_impl(
        {
            '_layout_prompt_label_records': _layout_prompt_label_records,
        },
        document,
        class_label,
    )


def _layout_prompt_region_records(document, region_ids):
    return _layout_prompt_region_records_impl(
        {
            '_layout_regions': _layout_regions,
        },
        document,
        region_ids,
    )


def _layout_prompt_display_mask(layout_state, source_mask):
    return _layout_prompt_display_mask_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_CLASS': _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_layout_prompt_region_records': _layout_prompt_region_records,
            '_layout_regions': _layout_regions,
            '_load_layout_prompt_region_document': _load_layout_prompt_region_document,
            'np': np,
        },
        layout_state,
        source_mask,
    )


def _layout_region_choice_update(document, selected_region_id=None):
    return _layout_region_choice_update_impl(
        {
            '_layout_label_choice_text': _layout_label_choice_text,
            '_layout_prompt_label_counts': _layout_prompt_label_counts,
            '_layout_regions': _layout_regions,
            'gr': gr,
        },
        document,
        selected_region_id,
    )


def _layout_region_category_update(value=None):
    return _layout_region_category_update_impl(
        {
            '_layout_region_label_update': _layout_region_label_update,
        },
        value,
    )


def _layout_region_label_update(document=None, value=None):
    return _layout_region_label_update_impl(
        {
            'gr': gr,
        },
        document,
        value,
    )


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
    return _layout_region_editor_payload_impl(
        {
            '_data_url': _data_url,
            '_layout_mask_to_editor_image': _layout_mask_to_editor_image,
            '_layout_region_identity': _layout_region_identity,
            '_layout_region_png_data_url': _layout_region_png_data_url,
            '_layout_region_source_image': _layout_region_source_image,
            '_layout_region_summaries': _layout_region_summaries,
            '_layout_regions': _layout_regions,
            'copy': copy,
        },
        layout_state,
        document,
        source_mask,
        status,
        selected_region_id,
        lasso_polygon,
        draft_region_mask,
    )


def _load_layout_region_context(layout_state):
    return _load_layout_region_context_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_choice_update': _layout_region_choice_update,
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_identity': _layout_region_identity,
            '_layout_region_label_update': _layout_region_label_update,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_layout_regions': _layout_regions,
            '_new_layout_region_state': _new_layout_region_state,
            'gr': gr,
        },
        layout_state,
    )


def _clear_layout_region_context(_layout_state):
    return _clear_layout_region_context_impl(
        {
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_label_update': _layout_region_label_update,
            '_new_layout_region_state': _new_layout_region_state,
            'gr': gr,
        },
        _layout_state,
    )


def _preview_layout_region(layout_state, region_state, editor_payload):
    return _preview_layout_region_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_client_intent': _layout_region_client_intent,
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_identity': _layout_region_identity,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_new_layout_region_state': _new_layout_region_state,
            '_validate_layout_region_intent': _validate_layout_region_intent,
            'gr': gr,
        },
        layout_state,
        region_state,
        editor_payload,
    )


def _select_layout_region(layout_state, region_state, selected_region_id):
    return _select_layout_region_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_layout_regions': _layout_regions,
            '_new_layout_region_state': _new_layout_region_state,
            '_validate_layout_region_state_identity': _validate_layout_region_state_identity,
            'gr': gr,
        },
        layout_state,
        region_state,
        selected_region_id,
    )


def _layout_region_latest_values(
    layout_state,
    selected,
    status,
    intent=None,
    keep_draft=False,
    region_state=None,
):
    return _layout_region_latest_values_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_choice_update': _layout_region_choice_update,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_identity': _layout_region_identity,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_layout_region_state_matches_identity': _layout_region_state_matches_identity,
            '_layout_regions': _layout_regions,
            'gr': gr,
        },
        layout_state,
        selected,
        status,
        intent,
        keep_draft,
        region_state,
    )


def _save_layout_region(layout_state, region_state, editor_payload, label):
    return _save_layout_region_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_choice_update': _layout_region_choice_update,
            '_layout_region_client_intent': _layout_region_client_intent,
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_label_update': _layout_region_label_update,
            '_layout_region_latest_values': _layout_region_latest_values,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_layout_regions': _layout_regions,
            '_new_layout_region_state': _new_layout_region_state,
            '_validate_layout_region_intent': _validate_layout_region_intent,
            'gr': gr,
        },
        layout_state,
        region_state,
        editor_payload,
        label,
    )


def _delete_layout_region(layout_state, region_state, editor_payload, selected_region_id):
    return _delete_layout_region_impl(
        {
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_choice_update': _layout_region_choice_update,
            '_layout_region_client_intent': _layout_region_client_intent,
            '_layout_region_editor_empty': _layout_region_editor_empty,
            '_layout_region_editor_payload': _layout_region_editor_payload,
            '_layout_region_label_update': _layout_region_label_update,
            '_layout_region_latest_values': _layout_region_latest_values,
            '_layout_region_state_from_document': _layout_region_state_from_document,
            '_layout_regions': _layout_regions,
            '_new_layout_region_state': _new_layout_region_state,
            '_validate_layout_region_intent': _validate_layout_region_intent,
            'gr': gr,
        },
        layout_state,
        region_state,
        editor_payload,
        selected_region_id,
    )


def _export_layout_regions(layout_state, region_state):
    return _export_layout_regions_impl(
        {
            'Path': Path,
            '_LAYOUT_REGION_STORE': _LAYOUT_REGION_STORE,
            '_layout_region_identity': _layout_region_identity,
            '_layout_regions': _layout_regions,
            '_prune_public_downloads': _prune_public_downloads,
            '_public_downloads': _public_downloads,
            'cv2': cv2,
            'json': json,
            'np': np,
            'public_download_dir': public_download_dir,
            'runtime_export_dir': runtime_export_dir,
            'tempfile': tempfile,
        },
        layout_state,
        region_state,
    )


def _layout_prompt_group_id(region_id):
    return _layout_prompt_group_id_impl(
        {
        },
        region_id,
    )


def _layout_prompt_group_data(layout_state, source_mask):
    return _layout_prompt_group_data_impl(
        {
            '_layout_prompt_region_records': _layout_prompt_region_records,
            '_layout_regions': _layout_regions,
            '_load_layout_prompt_region_document': _load_layout_prompt_region_document,
            'hashlib': hashlib,
            'json': json,
            'np': np,
        },
        layout_state,
        source_mask,
    )


def _layout_group_transform(
    layout_state,
    base_transform,
    record,
    group_mask,
    target_size,
    values=None,
):
    return _layout_group_transform_impl(
        {
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_prompt_group_id': _layout_prompt_group_id,
            '_layout_regions': _layout_regions,
            '_layout_tx': _layout_tx,
            'np': np,
        },
        layout_state,
        base_transform,
        record,
        group_mask,
        target_size,
        values,
    )


def _layout_prompt_group_payload(
    layout_state,
    source_mask,
    base_transform,
    target_size,
):
    return _layout_prompt_group_payload_impl(
        {
            '_data_url': _data_url,
            '_layout_group_transform': _layout_group_transform,
            '_layout_mask_to_editor_image': _layout_mask_to_editor_image,
            '_layout_prompt_group_data': _layout_prompt_group_data,
            '_layout_prompt_group_id': _layout_prompt_group_id,
            '_layout_regions': _layout_regions,
            'copy': copy,
        },
        layout_state,
        source_mask,
        base_transform,
        target_size,
    )



def _layout_editor_empty(image_state=None, status="请先加载或生成版图 mask"):
    return _layout_editor_empty_impl(
        {
            '_data_url': _data_url,
            '_workspace': _workspace,
        },
        image_state,
        status,
    )


def _layout_editor_payload(image_state, layout_state, status=None):
    return _layout_editor_payload_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_REGION_CLASS': _LAYOUT_PROMPT_SCOPE_REGION_CLASS,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_data_url': _data_url,
            '_layout_cache_get': _layout_cache_get,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_mask_to_editor_image': _layout_mask_to_editor_image,
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_prompt_display_mask': _layout_prompt_display_mask,
            '_layout_prompt_group_payload': _layout_prompt_group_payload,
            '_layout_tx': _layout_tx,
            '_workspace': _workspace,
            'copy': copy,
            'np': np,
        },
        image_state,
        layout_state,
        status,
    )


def _layout_editor_transform(editor_payload):
    return _layout_editor_transform_impl(
        {
        },
        editor_payload,
    )


def _layout_group_control_values(transform, target_size):
    return _layout_group_control_values_impl(
        {
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_tx': _layout_tx,
        },
        transform,
        target_size,
    )


def _commit_layout_group_transforms(
    image_state,
    layout_state,
    editor_payload,
    *,
    numeric_override=None,
    reset_active=False,
):
    return _commit_layout_group_transforms_impl(
        {
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_layout_cache_get': _layout_cache_get,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_editor_transform': _layout_editor_transform,
            '_layout_group_transform': _layout_group_transform,
            '_layout_prompt_group_data': _layout_prompt_group_data,
            '_layout_prompt_group_id': _layout_prompt_group_id,
            '_layout_regions': _layout_regions,
            '_layout_tx': _layout_tx,
            'copy': copy,
            'np': np,
        },
        image_state,
        layout_state,
        editor_payload,
        numeric_override,
        reset_active,
    )


def _validate_layout_group_transform_snapshot(layout_state, snapshot):
    return _validate_layout_group_transform_snapshot_impl(
        {
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_get': _layout_cache_get,
            'copy': copy,
        },
        layout_state,
        snapshot,
    )



def _sync_layout_controls_from_editor(layout_state, editor_payload):
    return _sync_layout_controls_from_editor_impl(
        {
            '_layout_editor_transform': _layout_editor_transform,
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_tx': _layout_tx,
            'gr': gr,
            'np': np,
        },
        layout_state,
        editor_payload,
    )


def _sync_layout_controls_from_editor_with_prompt_epoch(
    layout_state,
    editor_payload,
):
    return _sync_layout_controls_from_editor_with_prompt_epoch_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_commit_layout_group_transforms': _commit_layout_group_transforms,
            '_layout_editor_transform': _layout_editor_transform,
            '_layout_group_control_values': _layout_group_control_values,
            '_sync_layout_controls_from_editor': _sync_layout_controls_from_editor,
            'gr': gr,
        },
        layout_state,
        editor_payload,
    )


def _run_layout_mask_page(session_state, image_state, input_image, threshold, invert, open_kernel, close_kernel, min_component_area, region_mode, morph_pixels=0):
    return _run_layout_mask_page_impl(
        {
            '_binarize_layout_image': _binarize_layout_image,
            '_filter_layout_components': _filter_layout_components,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_mask_contours': _layout_mask_contours,
            '_layout_mask_to_preview': _layout_mask_to_preview,
            '_new_layout_state': _new_layout_state,
            '_normalize_layout_morph_pixels': _normalize_layout_morph_pixels,
            '_save_layout_mask_files': _save_layout_mask_files,
            '_session_id_from_state': _session_id_from_state,
        },
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
    return _run_layout_mask_page_with_downloads_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_publish_layout_downloads': _publish_layout_downloads,
            '_run_layout_mask_page': _run_layout_mask_page,
        },
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


def _save_current_layout_mask(layout_state):
    return _save_current_layout_mask_impl(
        {
            '_layout_cache_get': _layout_cache_get,
            '_publish_layout_downloads': _publish_layout_downloads,
        },
        layout_state,
    )


def _clear_current_layout_mask(image_state, layout_state):
    return _clear_current_layout_mask_impl(
        {
            '_clear_layout_cache': _clear_layout_cache,
            '_layout_editor_empty': _layout_editor_empty,
            '_new_layout_state': _new_layout_state,
        },
        image_state,
        layout_state,
    )


def _clear_current_layout_mask_with_prompt_epoch(
    image_state,
    layout_state,
):
    return _clear_current_layout_mask_with_prompt_epoch_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_clear_current_layout_mask': _clear_current_layout_mask,
        },
        image_state,
        layout_state,
    )


def _layout_numeric_controls_changed(layout_state, tx, ty, scale, rotation_deg, preview_alpha, tol=1e-6):
    return _layout_numeric_controls_changed_impl(
        {
        },
        layout_state,
        tx,
        ty,
        scale,
        rotation_deg,
        preview_alpha,
        tol,
    )


def _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=None, prefer_numeric=None):
    return _commit_layout_transform_impl(
        {
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_get': _layout_cache_get,
            '_layout_editor_transform': _layout_editor_transform,
            '_layout_numeric_controls_changed': _layout_numeric_controls_changed,
            '_layout_tx': _layout_tx,
            '_new_layout_state': _new_layout_state,
            '_workspace': _workspace,
            '_write_layout_meta': _write_layout_meta,
            'copy': copy,
            'np': np,
        },
        image_state,
        layout_state,
        enabled,
        tx,
        ty,
        scale,
        rotation_deg,
        preview_alpha,
        transform_payload,
        prefer_numeric,
    )


def _transform_layout_mask(layout_state, target_width, target_height):
    return _transform_layout_mask_impl(
        {
        },
        layout_state,
        target_width,
        target_height,
    )


def _layout_mask_to_overlay(base_image, mask, alpha=0.35):
    return _layout_mask_to_overlay_impl(
        {
            'Image': Image,
            '_pil_image': _pil_image,
            'cv2': cv2,
            'np': np,
        },
        base_image,
        mask,
        alpha,
    )


def _update_layout_preview(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload):
    return _update_layout_preview_impl(
        {
            '_commit_layout_transform': _commit_layout_transform,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_state_summary': _layout_state_summary,
            '_new_layout_state': _new_layout_state,
            '_workspace_image': _workspace_image,
            'gr': gr,
        },
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
    return _update_layout_preview_with_groups_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_commit_layout_group_transforms': _commit_layout_group_transforms,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_group_control_values': _layout_group_control_values,
            '_update_layout_preview': _update_layout_preview,
            '_workspace_image': _workspace_image,
            'gr': gr,
        },
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


def _mask_to_lowres_logits(mask):
    return _mask_to_lowres_logits_impl(
        {
            '_prompt_mask_size': _prompt_mask_size,
            'cv2': cv2,
            'np': np,
        },
        mask,
    )


def _validate_layout_prompt_mask(mask):
    return _validate_layout_prompt_mask_impl(
        {
            'np': np,
        },
        mask,
    )


def _layout_transformed_mask_for_image(image_state, layout_state):
    return _layout_transformed_mask_for_image_impl(
        {
            '_commit_layout_transform': _commit_layout_transform,
            '_layout_cache_get': _layout_cache_get,
            '_validate_layout_prompt_mask': _validate_layout_prompt_mask,
            '_workspace': _workspace,
            'np': np,
        },
        image_state,
        layout_state,
    )


def _layout_prompt_metadata(image_state, layout_state):
    return _layout_prompt_metadata_impl(
        {
            '_layout_cache_get': _layout_cache_get,
            '_layout_preview_alpha': _layout_preview_alpha,
            'copy': copy,
        },
        image_state,
        layout_state,
    )


def _create_pvs_from_layout_mask(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload, progress=gr.Progress(track_tqdm=False)):
    return _create_pvs_from_layout_mask_impl(
        {
            '_best': _best,
            '_commit_layout_transform': _commit_layout_transform,
            '_fresh_state': _fresh_state,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_prompt_metadata': _layout_prompt_metadata,
            '_make_inst': _make_inst,
            '_mask_box': _mask_box,
            '_mask_to_lowres_logits': _mask_to_lowres_logits,
            '_new_layout_state': _new_layout_state,
            '_predict_inst': _predict_inst,
            '_pvs_progress': _pvs_progress,
            '_validate_layout_prompt_mask': _validate_layout_prompt_mask,
            '_view': _view,
            'copy': copy,
        },
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
        progress,
    )


def _layout_state_summary(layout_state):
    return _layout_state_summary_impl(
        {
            '_layout_preview_alpha': _layout_preview_alpha,
        },
        layout_state,
    )


def _load_layout_binary_mask_png(session_state, image_state, input_image, region_mode="all"):
    return _load_layout_binary_mask_png_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_filter_layout_components': _filter_layout_components,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_mask_contours': _layout_mask_contours,
            '_layout_state_summary': _layout_state_summary,
            '_new_layout_state': _new_layout_state,
            '_pil_image': _pil_image,
            '_save_layout_mask_files': _save_layout_mask_files,
            '_session_id_from_state': _session_id_from_state,
            'cv2': cv2,
            'np': np,
        },
        session_state,
        image_state,
        input_image,
        region_mode,
    )


def _use_current_layout_mask(image_state, layout_state):
    return _use_current_layout_mask_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_layout_cache_get': _layout_cache_get,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_state_summary': _layout_state_summary,
            '_new_layout_state': _new_layout_state,
        },
        image_state,
        layout_state,
    )


def _pvs_creation_commit_token(pvs_state):
    return _pvs_creation_commit_token_impl(
        {
        },
        pvs_state,
    )


def _selected_pvs_candidate(prediction, image_shape):
    return _selected_pvs_candidate_impl(
        {
            '_prompt_mask_size': _prompt_mask_size,
            'np': np,
        },
        prediction,
        image_shape,
    )


def _layout_prompt_region_fingerprint(decoded_records):
    return _layout_prompt_region_fingerprint_impl(
        {
            '_layout_regions': _layout_regions,
            'np': np,
        },
        decoded_records,
    )


def _validate_layout_transform_snapshot(layout_state, transform):
    return _validate_layout_transform_snapshot_impl(
        {
            '_LAYOUT_CACHE_LOCK': _LAYOUT_CACHE_LOCK,
            '_layout_cache_get': _layout_cache_get,
            'copy': copy,
            'np': np,
        },
        layout_state,
        transform,
    )


def _load_layout_prompt_choices(image_state, layout_state):
    return _load_layout_prompt_choices_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_layout_cache_get': _layout_cache_get,
            '_layout_editor_empty': _layout_editor_empty,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_prompt_choice_update': _layout_prompt_choice_update,
            '_layout_regions': _layout_regions,
            '_load_layout_prompt_region_document': _load_layout_prompt_region_document,
            '_reset_layout_prompt_selection_state': _reset_layout_prompt_selection_state,
        },
        image_state,
        layout_state,
    )


def _select_layout_prompt_mask(image_state, layout_state, selection):
    return _select_layout_prompt_mask_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_commit_layout_group_transforms': _commit_layout_group_transforms,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_group_control_values': _layout_group_control_values,
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_prompt_region_records': _layout_prompt_region_records,
            '_layout_prompt_selection_token': _layout_prompt_selection_token,
            '_layout_regions': _layout_regions,
            '_load_layout_prompt_region_document': _load_layout_prompt_region_document,
            '_normalize_layout_prompt_checkbox_selection': _normalize_layout_prompt_checkbox_selection,
            '_parse_layout_prompt_selection': _parse_layout_prompt_selection,
            '_reset_layout_prompt_selection_state': _reset_layout_prompt_selection_state,
            'gr': gr,
            'np': np,
        },
        image_state,
        layout_state,
        selection,
    )


def _reset_layout_prompt_selection(image_state, layout_state):
    return _reset_layout_prompt_selection_impl(
        {
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_prompt_choice_update': _layout_prompt_choice_update,
            '_reset_layout_prompt_selection_state': _reset_layout_prompt_selection_state,
        },
        image_state,
        layout_state,
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
    return _create_pvs_from_layout_selection_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_FULL': _LAYOUT_PROMPT_SCOPE_FULL,
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_LayoutPromptConflictError': _LayoutPromptConflictError,
            '_commit_layout_group_transforms': _commit_layout_group_transforms,
            '_create_pvs_from_layout_mask': _create_pvs_from_layout_mask,
            '_fresh_state': _fresh_state,
            '_is_layout_mask_mode': _is_layout_mask_mode,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_preview_alpha': _layout_preview_alpha,
            '_layout_prompt_epoch_snapshot': _layout_prompt_epoch_snapshot,
            '_layout_prompt_group_id': _layout_prompt_group_id,
            '_layout_prompt_metadata': _layout_prompt_metadata,
            '_layout_prompt_region_fingerprint': _layout_prompt_region_fingerprint,
            '_layout_prompt_region_records': _layout_prompt_region_records,
            '_layout_regions': _layout_regions,
            '_layout_tx': _layout_tx,
            '_load_layout_prompt_region_document': _load_layout_prompt_region_document,
            '_make_inst': _make_inst,
            '_mask_box': _mask_box,
            '_mask_to_lowres_logits': _mask_to_lowres_logits,
            '_new_layout_state': _new_layout_state,
            '_parse_layout_prompt_selection': _parse_layout_prompt_selection,
            '_predict_inst': _predict_inst,
            '_pvs_creation_commit_token': _pvs_creation_commit_token,
            '_pvs_progress': _pvs_progress,
            '_reset_layout_prompt_selection_state': _reset_layout_prompt_selection_state,
            '_selected_pvs_candidate': _selected_pvs_candidate,
            '_validate_layout_group_transform_snapshot': _validate_layout_group_transform_snapshot,
            '_validate_layout_prompt_mask': _validate_layout_prompt_mask,
            '_view': _view,
            '_workspace': _workspace,
            'copy': copy,
            'gr': gr,
            'np': np,
        },
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
        progress,
    )


def _reset_layout_controls(image_state, layout_state):
    return _reset_layout_controls_impl(
        {
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_state_summary': _layout_state_summary,
            '_new_layout_state': _new_layout_state,
        },
        image_state,
        layout_state,
    )

def _reset_layout_controls_with_prompt_epoch(image_state, layout_state):
    return _reset_layout_controls_with_prompt_epoch_impl(
        {
            '_LAYOUT_PROMPT_SCOPE_REGION_LABELS': _LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            '_advance_layout_prompt_epoch': _advance_layout_prompt_epoch,
            '_commit_layout_group_transforms': _commit_layout_group_transforms,
            '_layout_editor_payload': _layout_editor_payload,
            '_layout_group_control_values': _layout_group_control_values,
            '_reset_layout_controls': _reset_layout_controls,
            'gr': gr,
        },
        image_state,
        layout_state,
    )


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
