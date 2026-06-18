#!/usr/bin/env python3
"""Convert colored CAD/layout images into binary mask prompts.

The historical source script that produced the saved AST/BSM/GE1/GE2 masks is
not present in the worktree anymore; only a pyc cache remains. This script is a
small, UI-friendly reimplementation of the observed two-stage pipeline:

1. Extract colored layout pixels from a near-white background.
2. Optionally close tiny gaps and fill enclosed background components.

Masks returned by the Python API are boolean foreground masks. Files written by
the CLI use black foreground on white background to match the existing
`*_binary_black_on_white.png` and `*_ccfill_g2p5.png` assets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape for {path}: {image.shape}")


def extract_layout_mask(
    image_rgb: np.ndarray,
    saturation_min: int = 12,
    value_min: int = 90,
    chroma_min: int = 0,
) -> np.ndarray:
    """Extract colored layout linework as a foreground boolean mask."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = (saturation >= saturation_min) & (value >= value_min)
    if chroma_min > 0:
        rgb_i = image_rgb.astype(np.int16)
        chroma = rgb_i.max(axis=2) - rgb_i.min(axis=2)
        mask &= chroma >= chroma_min
    return mask.astype(bool)


def _kernel(shape: str, size: int) -> np.ndarray:
    size = max(1, int(size))
    if shape == "rect":
        kind = cv2.MORPH_RECT
    elif shape == "cross":
        kind = cv2.MORPH_CROSS
    elif shape == "ellipse":
        kind = cv2.MORPH_ELLIPSE
    else:
        raise ValueError(f"Unsupported kernel shape: {shape}")
    return cv2.getStructuringElement(kind, (size, size))


def gap_to_kernel_size(gap: float) -> int:
    """Map a human gap value like 2.5 to a small odd close kernel."""
    if gap <= 0:
        return 1
    # Keep 2.5, 3.0, and 3.5 in the same 3x3 bucket, matching the historical
    # runtime summaries where these gaps produced identical output.
    return max(1, 2 * int(float(gap) // 2.0) + 1)


def ccfill_mask(
    foreground: np.ndarray,
    gap: float = 2.5,
    method: str = "distance",
    close_kernel_size: Optional[int] = None,
    close_shape: str = "rect",
) -> np.ndarray:
    """Fill narrow enclosed background components."""
    mask = foreground.astype(bool)

    if method == "distance":
        working = mask.copy()
        background = (~working).astype(np.uint8)
        distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    elif method == "close_fill":
        kernel_size = close_kernel_size if close_kernel_size is not None else gap_to_kernel_size(gap)
        working_u8 = mask.astype(np.uint8)
        if kernel_size > 1:
            working_u8 = cv2.morphologyEx(
                working_u8,
                cv2.MORPH_CLOSE,
                _kernel(close_shape, kernel_size),
                iterations=1,
            )
        working = working_u8.astype(bool)
        background = (~working).astype(np.uint8)
        distance = None
    else:
        raise ValueError(f"Unsupported ccfill method: {method}")

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(background, 8)
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    filled = working.astype(bool)
    for label in range(1, num_labels):
        if label in border_labels:
            continue
        component = labels == label
        if method == "distance" and float(distance[component].max()) > gap:
            continue
        filled[component] = True
    return filled


def write_black_on_white(path: Path, foreground: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.where(foreground.astype(bool), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def load_black_foreground(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image < 128


def compare_masks(pred: np.ndarray, reference: np.ndarray) -> Dict[str, float | int]:
    pred = pred.astype(bool)
    reference = reference.astype(bool)
    diff = pred ^ reference
    union = pred | reference
    intersection = pred & reference
    return {
        "diff_pixels": int(diff.sum()),
        "iou": float(intersection.sum() / union.sum()) if union.any() else 1.0,
        "pred_foreground_ratio": float(pred.mean()),
        "reference_foreground_ratio": float(reference.mean()),
    }


def convert_image(
    input_path: Path,
    output_dir: Path,
    stem: Optional[str] = None,
    saturation_min: int = 12,
    value_min: int = 90,
    chroma_min: int = 0,
    gap: float = 2.5,
    ccfill_method: str = "distance",
    close_kernel_size: Optional[int] = None,
    close_shape: str = "rect",
    write_added: bool = True,
) -> Dict:
    image_rgb = read_rgb(input_path)
    name = stem or input_path.stem
    binary = extract_layout_mask(image_rgb, saturation_min, value_min, chroma_min)
    filled = ccfill_mask(binary, gap, ccfill_method, close_kernel_size, close_shape)

    binary_path = output_dir / f"{name}_binary_black_on_white.png"
    filled_path = output_dir / f"{name}_ccfill_g{str(gap).replace('.', 'p')}.png"
    write_black_on_white(binary_path, binary)
    write_black_on_white(filled_path, filled)

    added_path = ""
    if write_added:
        added = np.zeros((*binary.shape, 3), dtype=np.uint8)
        added[:] = 255
        added[binary] = (0, 0, 0)
        added[filled & ~binary] = (0, 0, 255)
        added_path = str(output_dir / f"{name}_ccfill_added_g{str(gap).replace('.', 'p')}.png")
        cv2.imwrite(added_path, added)

    return {
        "input_path": str(input_path),
        "binary_path": str(binary_path),
        "ccfill_path": str(filled_path),
        "added_path": added_path,
        "shape": list(binary.shape),
        "binary_foreground_ratio": float(binary.mean()),
        "ccfill_foreground_ratio": float(filled.mean()),
        "added_foreground_ratio": float((filled & ~binary).mean()),
        "parameters": {
            "saturation_min": saturation_min,
            "value_min": value_min,
            "chroma_min": chroma_min,
            "gap": gap,
            "ccfill_method": ccfill_method,
            "close_kernel_size": close_kernel_size if close_kernel_size is not None else gap_to_kernel_size(gap),
            "close_shape": close_shape,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="")
    parser.add_argument("--saturation-min", type=int, default=12)
    parser.add_argument("--value-min", type=int, default=90)
    parser.add_argument("--chroma-min", type=int, default=0)
    parser.add_argument("--gap", type=float, default=2.5)
    parser.add_argument("--ccfill-method", choices=["distance", "close_fill"], default="distance")
    parser.add_argument("--close-kernel-size", type=int, default=0)
    parser.add_argument("--close-shape", choices=["rect", "ellipse", "cross"], default="rect")
    parser.add_argument("--qa-reference-binary", type=Path, default=None)
    parser.add_argument("--qa-reference-ccfill", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    close_kernel_size = args.close_kernel_size or None
    summary = convert_image(
        input_path=args.input,
        output_dir=args.output_dir,
        stem=args.stem or None,
        saturation_min=args.saturation_min,
        value_min=args.value_min,
        chroma_min=args.chroma_min,
        gap=args.gap,
        ccfill_method=args.ccfill_method,
        close_kernel_size=close_kernel_size,
        close_shape=args.close_shape,
    )

    if args.qa_reference_binary:
        summary["binary_qa"] = compare_masks(
            load_black_foreground(Path(summary["binary_path"])),
            load_black_foreground(args.qa_reference_binary),
        )
    if args.qa_reference_ccfill:
        summary["ccfill_qa"] = compare_masks(
            load_black_foreground(Path(summary["ccfill_path"])),
            load_black_foreground(args.qa_reference_ccfill),
        )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
