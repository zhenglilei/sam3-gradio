"""SAM3 model initialization and shared predictor runtime."""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np

import torch

from sam3_demo.config import current_dir
from sam3_demo.segmentation_evaluation import polygon_to_mask


try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import FindStage
    from sam3.model import box_ops
except ImportError as exc:
    print(f"导入SAM3模块失败: {exc}")
    print("请确保已正确安装SAM3依赖")
    sys.exit(1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")


def initialize_models():
    """初始化 SAM3 图像预测器。"""
    try:
        model_dir = current_dir / "models"
        checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if not checkpoint_path.exists():
            print(f"模型文件不存在: {checkpoint_path}")
            print("请下载SAM3模型文件到目录")
            return None

        if not bpe_path.exists():
            print(f"BPE文件不存在: {bpe_path}")
            return None

        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE,
            enable_inst_interactivity=True,
        )
        predictor = Sam3Processor(image_model, device=DEVICE)
        print("模型初始化成功")
        return predictor
    except Exception as exc:
        print(f"模型初始化失败: {exc}")
        return None


image_predictor = initialize_models()

_PVS_PREDICT_LOCK = threading.Lock()


def _disable_legacy_predict_mask_prompt():
    if image_predictor is None or not hasattr(image_predictor, "predict_mask_prompt"):
        return

    def _deprecated_predict_mask_prompt(*args, **kwargs):
        raise RuntimeError(
            "predict_mask_prompt() is deprecated in this demo. "
            "Use _predict_inst(..., mask_input_lowres_logits=...) so multimask "
            "candidates and low-res logits are preserved."
        )

    image_predictor.predict_mask_prompt = _deprecated_predict_mask_prompt


_disable_legacy_predict_mask_prompt()


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
