"""State schemas and pure instance-state helpers."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np

from sam3_demo.config import (
    MODE_LAYOUT,
    MODE_PCS,
    MODE_PVS,
    _LAYOUT_PROMPT_SCOPE_FULL,
    _MAX_PROMPT_HISTORY_ENTRIES,
)


def _new_source_image_state(session_id=None):
    return {
        "session_id": str(session_id or ""),
        "source_image_id": None,
        "source_image_sha256": None,
        "source_width": 0,
        "source_height": 0,
        "source_revision": 0,
        "crop_bbox_xyxy": None,
        "pending_crop_bbox_xyxy": None,
        "workspace_image_id": None,
        "workspace_hash": None,
    }

def _new_template_match_state():
    return {
        "schema_version": 1,
        "source_image_id": None,
        "workspace_image_id": None,
        "active_instance_id": None,
        "result": None,
    }

def _new_pcs_state():
    return {
        "text_prompt": "",
        "positive_boxes": [],
        "negative_boxes": [],
        "bbox_history": [],
        "bbox_records": [],
        "next_bbox_id": 1,
        "instances": {},
        "next_instance_id": 1,
    }

def _new_pvs_state():
    return {
        "instances": {},
        "active_instance_id": None,
        "next_instance_id": 1,
        "pending_boxes": [],
        "pending_bbox_records": [],
        "next_pending_bbox_id": 1,
    }

def _new_session_state():
    return {"session_id": uuid.uuid4().hex}

def _session_id_from_state(session_state=None):
    if isinstance(session_state, dict) and session_state.get("session_id"):
        return str(session_state["session_id"])
    return uuid.uuid4().hex

def _new_layout_state(session_id=None):
    return {
        "transform_version": 2,
        "session_id": str(session_id or uuid.uuid4().hex),
        "layout_id": None,
        "image_id": None,
        "enabled": False,
        "region_mode": "all",
        "revision": 0,
        "center_x": None,
        "center_y": None,
        "pivot_x": None,
        "pivot_y": None,
        "tx": 0.0,
        "ty": 0.0,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "preview_alpha": 0.35,
        "source_width": 0,
        "source_height": 0,
        "source_mask_pixel_sha256": None,
        "source_mask_file_sha256": None,
        "target_image_sha256": None,
        "matrix_2x3": None,
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

def _norm_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((max(0.0, min(x1, width - 1)), max(0.0, min(x2, width - 1))))
    y1, y2 = sorted((max(0.0, min(y1, height - 1)), max(0.0, min(y2, height - 1))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]

def _bbox_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Draw a bbox first")
    data = json.loads(payload)
    box = data.get("box_xyxy_px")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox payload is missing box_xyxy_px")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    box = _norm_box(box, width, height)
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("bbox is too small")
    return box

def _xyxy_to_cxcywh_norm(box, width, height):
    x1, y1, x2, y2 = _norm_box(box, width, height)
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, max(1.0, x2 - x1) / width, max(1.0, y2 - y1) / height]

def _polygon_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Draw a positive polygon first")
    data = json.loads(payload)
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("polygon needs at least 3 points")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    parsed = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            parsed.append([max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))])
    if len(parsed) < 3:
        raise ValueError("polygon needs at least 3 valid points")
    return parsed

def _point_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Click a positive point first")
    data = json.loads(payload)
    point = data.get("point_xy_px")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("point payload is missing point_xy_px")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    return [max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))]

def _combine_logits(active_logits, polygon_logits, mode="replace", alpha=0.35, max_logit=10.0):
    polygon = np.asarray(polygon_logits, dtype=np.float32)
    if polygon.ndim == 3:
        polygon = polygon[0]
    if active_logits is None:
        return np.clip(polygon, -max_logit, max_logit).astype(np.float32)

    active = np.asarray(active_logits, dtype=np.float32)
    if active.ndim == 3:
        active = active[0]

    if mode == "replace":
        out = polygon
    elif mode == "blend":
        out = alpha * active + (1.0 - alpha) * polygon
    elif mode == "union":
        out = np.maximum(active, polygon)
    elif mode == "intersect":
        out = np.minimum(active, polygon)
    else:
        raise ValueError(f"Unknown polygon combine mode: {mode}")
    return np.clip(out, -max_logit, max_logit).astype(np.float32)

def _polygon_action_key(value):
    text = str(value or "")
    if text == "create" or "create" in text.lower() or "\u521b\u5efa" in text:
        return "create"
    return "refine"

def _polygon_combine_key(value):
    text = str(value or "replace")
    if text in {"replace", "blend", "union", "intersect"}:
        return text
    if "blend" in text.lower() or "\u878d\u5408" in text:
        return "blend"
    if "union" in text.lower() or "\u8865\u5145" in text:
        return "union"
    if "intersect" in text.lower() or "\u9650\u5236" in text:
        return "intersect"
    return "replace"

def _mask_box(mask):
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

def _pvs_progress(progress, value, desc, delay=0.08):
    if progress is None:
        return
    progress(float(value), desc=desc)
    if delay:
        time.sleep(delay)

def _best(pred):
    if len(pred["scores"]) == 0:
        raise ValueError("predict_inst returned no masks")
    return int(np.argmax(pred["scores"]))

def _make_inst(inst_id, source, mask, box, score, pvs_logits=None, pcs_prob=None, history=None):
    return {
        "id": int(inst_id),
        "source": source,
        "mask_fullres_bool": np.asarray(mask).astype(bool),
        "box_xyxy_px": [float(v) for v in box],
        "score": float(score),
        "pvs_lowres_logits": None if pvs_logits is None else np.asarray(pvs_logits, dtype=np.float32),
        "pcs_fullres_prob": None if pcs_prob is None else np.asarray(pcs_prob, dtype=np.float16),
        "status": "draft",
        "prompt_history": history or [],
    }

def _snapshot(inst):
    return {
        "mask_fullres_bool": np.asarray(inst["mask_fullres_bool"]).copy(),
        "pvs_lowres_logits": None if inst.get("pvs_lowres_logits") is None else np.asarray(inst["pvs_lowres_logits"]).copy(),
        "box_xyxy_px": list(inst.get("box_xyxy_px") or []),
        "score": float(inst.get("score", 0.0)),
        "status": inst.get("status", "draft"),
    }

def _restore(inst, snap):
    inst["mask_fullres_bool"] = np.asarray(snap["mask_fullres_bool"]).copy()
    inst["pvs_lowres_logits"] = None if snap.get("pvs_lowres_logits") is None else np.asarray(snap["pvs_lowres_logits"]).copy()
    inst["box_xyxy_px"] = list(snap.get("box_xyxy_px") or [])
    inst["score"] = float(snap.get("score", 0.0))
    inst["status"] = snap.get("status", inst.get("status", "draft"))

def _history_snapshot(inst):
    return {
        "box_xyxy_px": list(inst.get("box_xyxy_px") or []),
        "score": float(inst.get("score", 0.0)),
        "status": inst.get("status", "draft"),
    }

def _append_prompt_history(inst, event):
    history = inst.setdefault("prompt_history", [])
    history.append(event)
    if len(history) > _MAX_PROMPT_HISTORY_ENTRIES:
        history[:] = [
            history[0],
            *history[-(_MAX_PROMPT_HISTORY_ENTRIES - 1) :],
        ]

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

def _new_prompt_state():
    return {"bbox_start": None, "last_bbox": None, "last_point": None, "polygon_points": [], "bbox_role": "positive"}
