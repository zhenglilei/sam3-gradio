"""SAM3 model initialization and shared predictor runtime."""

from __future__ import annotations

import sys

import torch

from sam3_demo.config import current_dir


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
