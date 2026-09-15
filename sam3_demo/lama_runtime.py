"""Minimal, local-only runtime for the trusted Big-LaMa TorchScript model."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _ceil_modulo(value: int, modulo: int) -> int:
    return value if value % modulo == 0 else (value // modulo + 1) * modulo


def _prepare_image_and_mask(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    image_array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    mask_array = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("LaMa input image must be RGB")
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    if mask_array.ndim != 2 or mask_array.shape != image_array.shape[:2]:
        raise ValueError("LaMa mask size must match the input image")

    height, width = image_array.shape[:2]
    padded_height = _ceil_modulo(height, 8)
    padded_width = _ceil_modulo(width, 8)
    image_chw = np.transpose(image_array.astype(np.float32) / 255.0, (2, 0, 1))
    mask_chw = (mask_array > 0).astype(np.float32)[None, ...]
    padding = ((0, 0), (0, padded_height - height), (0, padded_width - width))
    image_chw = np.pad(image_chw, padding, mode="symmetric")
    mask_chw = np.pad(mask_chw, padding, mode="symmetric")
    return (
        torch.from_numpy(image_chw).unsqueeze(0),
        torch.from_numpy(mask_chw).unsqueeze(0),
        (width, height),
    )


class LamaRuntime:
    """Load one CPU TorchScript model lazily and serialize inference calls."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self._model = None
        self._lock = threading.RLock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LaMa model not found: {self.model_path}")
        model = torch.jit.load(str(self.model_path), map_location="cpu")
        model.eval()
        self._model = model
        return model

    def inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
    ) -> Image.Image:
        image_tensor, mask_tensor, original_size = _prepare_image_and_mask(image, mask)
        if not bool(mask_tensor.any()):
            raise ValueError("修复区域为空，请先自动检测或手动标记")
        with self._lock, torch.inference_mode():
            output = self._load_model()(image_tensor, mask_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError("LaMa model returned an unsupported output")
        result = output[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        width, height = original_size
        return Image.fromarray(result[:height, :width], mode="RGB")


_RUNTIMES: dict[Path, LamaRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_lama_runtime(model_path: str | Path) -> LamaRuntime:
    resolved = Path(model_path).resolve()
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(resolved)
        if runtime is None:
            runtime = LamaRuntime(resolved)
            _RUNTIMES[resolved] = runtime
        return runtime


__all__ = ["LamaRuntime", "get_lama_runtime"]
