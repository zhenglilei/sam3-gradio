"""Controller-side façade for the isolated SAM3 model worker."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from sam3_demo.model_supervisor import SUPERVISOR
from sam3_demo.segmentation_evaluation import polygon_to_mask


# Compatibility symbols retained for callers and source-level contract tests. The
# Controller deliberately owns no SAM3 processor, CUDA model, or model tensors.
DEVICE = "worker-managed"
image_predictor = None
FindStage = None
box_ops = None
_PVS_PREDICT_LOCK = threading.RLock()


def initialize_models():
    """Start the isolated Worker asynchronously through the compatibility API."""
    SUPERVISOR.request_start()
    return SUPERVISOR


def _prompt_mask_size():
    return tuple(int(v) for v in SUPERVISOR.mask_input_size())


def _polygon_lowres_logits(polygon, width, height):
    target_h, target_w = _prompt_mask_size()
    mask = polygon_to_mask(polygon, height, width).astype(np.float32)
    lowres = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return ((np.clip(lowres, 0.0, 1.0) * 2.0 - 1.0) * 10.0).astype(np.float32)


def _image_supplier(handle):
    from sam3_demo.workspace import _workspace_image_for_handle

    return lambda: _workspace_image_for_handle(handle)


def _predict_inst(
    base_state,
    box_xyxy_px=None,
    mask_input_lowres_logits=None,
    point_coords_px=None,
    point_labels=None,
):
    handle = dict(base_state or {})
    kwargs = {}
    if box_xyxy_px is not None:
        box = np.asarray(box_xyxy_px, dtype=np.float32).reshape(-1)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("box_xyxy_px must contain four finite values")
        kwargs["box_xyxy_px"] = box
    if point_coords_px is not None:
        coords = np.asarray(point_coords_px, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords[None, :]
        if coords.ndim != 2 or coords.shape[-1] != 2 or not np.isfinite(coords).all():
            raise ValueError("point_coords_px must have shape Nx2")
        labels = (
            np.ones((coords.shape[0],), dtype=np.int64)
            if point_labels is None
            else np.asarray(point_labels, dtype=np.int64).reshape(-1)
        )
        if labels.shape[0] != coords.shape[0]:
            raise ValueError("point_labels length must match point_coords_px")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("point_labels values must be 0 or 1")
        kwargs["point_coords_px"] = coords
        kwargs["point_labels"] = labels
    if mask_input_lowres_logits is not None:
        mask_input = np.asarray(mask_input_lowres_logits, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, :, :]
        expected = _prompt_mask_size()
        if (
            mask_input.ndim != 3
            or tuple(mask_input.shape[-2:]) != expected
            or not np.isfinite(mask_input).all()
        ):
            raise ValueError(
                f"mask_input_lowres_logits must be finite with shape 1x{expected[0]}x{expected[1]}"
            )
        kwargs["mask_input_lowres_logits"] = mask_input
    return SUPERVISOR.predict_inst(handle, _image_supplier(handle), **kwargs)


def _predict_pcs(
    base_state,
    *,
    text,
    positive_boxes_cxcywh,
    negative_boxes_cxcywh,
    threshold,
):
    handle = dict(base_state or {})
    positive = np.asarray(positive_boxes_cxcywh or [], dtype=np.float32).reshape(-1, 4)
    negative = np.asarray(negative_boxes_cxcywh or [], dtype=np.float32).reshape(-1, 4)
    confidence = float(threshold)
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("PCS boxes must contain finite values")
    if not np.isfinite(confidence):
        raise ValueError("PCS threshold must be finite")
    return SUPERVISOR.predict_pcs(
        handle,
        _image_supplier(handle),
        text=str(text or ""),
        positive_boxes_cxcywh=positive,
        negative_boxes_cxcywh=negative,
        threshold=confidence,
    )
