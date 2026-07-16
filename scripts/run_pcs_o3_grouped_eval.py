#!/usr/bin/env python3
"""Grouped whole-image PCS bbox-exemplar visualization for O3 COCO labels.

For each O3 COCO category, this script picks three original images containing
that category. For each selected image/category pair, it runs PCS without a text
prompt using 1, 2, and 3 positive bbox exemplars at confidence 0.5 by default.

The output is whole-image visualization, not one image per bbox.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import offline_eval_utils as eval_utils

spec = importlib.util.spec_from_file_location("pvs_bbox_label_base", SCRIPT_DIR / "run_pvs_bbox_label_eval.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import run_pvs_bbox_label_eval.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


PALETTE = [
    (0, 255, 120),
    (255, 180, 0),
    (0, 180, 255),
    (255, 80, 200),
    (160, 255, 0),
    (180, 120, 255),
    (255, 80, 80),
    (80, 255, 255),
]


@dataclass
class PcsCategoryImage:
    group_id: str
    layer: str
    dataset: str
    split: str
    label: str
    image_path: str
    image_id: str
    image_file_name: str
    category_id: int
    boxes: list[list[float]]
    annotation_ids: list[str]
    annotation_relpath: str
    annotation_sha256: str
    image_sha256: str


def safe(value: Any) -> str:
    return base._safe_name(value)


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def text_box(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int], size: int) -> None:
    fnt = font(size)
    bbox = draw.textbbox(xy, text, font=fnt)
    pad = max(2, size // 6)
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=fnt)


def xyxy_to_cxcywh_norm(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = base._clip_box_xyxy(box, width, height)
    return [
        ((x1 + x2) / 2.0) / float(width),
        ((y1 + y2) / 2.0) / float(height),
        max(1.0, x2 - x1) / float(width),
        max(1.0, y2 - y1) / float(height),
    ]


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, chr(100)+chr(101)+chr(116)+chr(97)+chr(99)+chr(104)) and hasattr(value, chr(99)+chr(112)+chr(117)):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)

def collect_o3_category_images(
    o3_root: Path,
    images_per_category: int,
    split_order: list[str],
    min_boxes: int,
) -> list[PcsCategoryImage]:
    o3_root = base._require_dataset_root(o3_root, "O3")
    if images_per_category <= 0:
        raise ValueError("images_per_category must be positive")
    if min_boxes <= 0:
        raise ValueError("min_boxes must be positive")
    if not split_order or any(not split for split in split_order):
        raise ValueError("split_order must contain at least one split")
    if len(split_order) != len(set(split_order)):
        raise ValueError("split_order must not contain duplicate splits")
    dataset_dirs = sorted(
        path
        for path in o3_root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name.endswith("_coco")
    )
    if not dataset_dirs:
        raise ValueError(f"No O3 *_coco datasets found under {o3_root}")

    selected: list[PcsCategoryImage] = []
    image_hashes: dict[Path, str] = {}
    for dataset_dir in dataset_dirs:
        layer = dataset_dir.name.removesuffix("_coco")
        candidates_by_label: dict[str, list[PcsCategoryImage]] = {}
        for split in split_order:
            annotation_path = (
                dataset_dir / "annotations" / f"instances_{split}.json"
            )
            if not annotation_path.exists():
                continue
            categories, images, annotations = base._load_coco(annotation_path)
            annotation_relpath = annotation_path.relative_to(o3_root).as_posix()
            annotation_sha256 = eval_utils.file_sha256(annotation_path)
            annotations_by_image_category: dict[
                tuple[int, int], list[dict[str, Any]]
            ] = {}
            for annotation in annotations:
                key = (
                    int(annotation["image_id"]),
                    int(annotation["category_id"]),
                )
                annotations_by_image_category.setdefault(key, []).append(annotation)

            for (image_id, category_id), category_annotations in sorted(
                annotations_by_image_category.items()
            ):
                label = categories[category_id]
                image_record = images[image_id]
                image_path = base._resolve_coco_image(
                    dataset_dir,
                    image_record,
                    annotation_path,
                    split=split,
                )
                if image_path not in image_hashes:
                    image_hashes[image_path] = eval_utils.file_sha256(image_path)
                boxes: list[list[float]] = []
                annotation_ids: list[str] = []
                for annotation in category_annotations:
                    x, y, width, height = [
                        float(value) for value in annotation["bbox"]
                    ]
                    boxes.append(
                        base._clip_box_xyxy(
                            [x, y, x + width, y + height],
                            int(image_record["width"]),
                            int(image_record["height"]),
                        )
                    )
                    annotation_ids.append(str(annotation["id"]))
                group_id = (
                    f"pcs_o3_{safe(layer)}_{safe(label)}_"
                    f"{safe(Path(str(image_record['file_name'])).stem)[:70]}"
                )
                candidates_by_label.setdefault(label, []).append(
                    PcsCategoryImage(
                        group_id=group_id,
                        layer=layer,
                        dataset=dataset_dir.name,
                        split=split,
                        label=label,
                        image_path=str(image_path),
                        image_id=str(image_id),
                        image_file_name=str(image_record["file_name"]),
                        category_id=category_id,
                        boxes=boxes,
                        annotation_ids=annotation_ids,
                        annotation_relpath=annotation_relpath,
                        annotation_sha256=annotation_sha256,
                        image_sha256=image_hashes[image_path],
                    )
                )
        for label in sorted(candidates_by_label):
            candidates = candidates_by_label[label]
            preferred = [
                group for group in candidates if len(group.boxes) >= min_boxes
            ]
            fallback = [
                group for group in candidates if len(group.boxes) < min_boxes
            ]
            for index, group in enumerate(
                (preferred + fallback)[:images_per_category], start=1
            ):
                group.group_id = (
                    f"pcs_o3_{safe(group.layer)}_{safe(group.label)}_"
                    f"{index:02d}_"
                    f"{safe(Path(group.image_file_name).stem)[:70]}"
                )
                selected.append(group)
    if not selected:
        raise ValueError(f"No O3 category-image groups found under {o3_root}")
    return selected

def draw_pcs_input_overlay(image: Image.Image, group: PcsCategoryImage, prompt_boxes: list[list[float]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(group.boxes, start=1):
        color = (80, 180, 255)
        draw.rectangle(box, outline=color, width=2)
        if i <= 6:
            text_box(draw, (box[0], max(0, box[1] - 18)), f"GT {group.label}", color, 11)
    for i, box in enumerate(prompt_boxes, start=1):
        draw.rectangle(box, outline=(0, 255, 80), width=5)
        text_box(draw, (box[0], max(0, box[1] - 24)), f"POS#{i}", (0, 255, 80), 15)
    text_box(draw, (8, 8), f"{group.label} PCS prompts={len(prompt_boxes)} GT={len(group.boxes)}", (0, 255, 120), 16)
    return out


def draw_pcs_result_overlay(image: Image.Image, group: PcsCategoryImage, pcs: dict[str, Any]) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    masks = pcs.get("masks", [])
    for i, mask in enumerate(masks, start=1):
        mask_bool = np.asarray(mask, dtype=bool)
        color = np.asarray(PALETTE[(i - 1) % len(PALETTE)], dtype=np.float32)
        arr_f = arr.astype(np.float32)
        arr_f[mask_bool] = arr_f[mask_bool] * 0.62 + color * 0.38
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    for i, (box, score) in enumerate(zip(pcs.get("boxes", []), pcs.get("scores", [])), start=1):
        color = PALETTE[(i - 1) % len(PALETTE)]
        draw.rectangle(box, outline=color, width=4)
        text_box(draw, (box[0], max(0, box[1] - 22)), f"PCS#{i} {score:.2f}", color, 13)
    text_box(draw, (8, 8), f"{group.label} PCS outputs={len(pcs.get('boxes', []))}", (255, 220, 80), 16)
    return out


def run_pcs_from_boxes(predictor: Any, image: Image.Image, prompt_boxes: list[list[float]], threshold: float) -> dict[str, Any]:
    width, height = image.size
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    state = predictor.set_image(image)
    for box in prompt_boxes:
        state = predictor.add_geometric_prompt(xyxy_to_cxcywh_norm(box, width, height), True, state)
    state = predictor.set_confidence_threshold(float(threshold), state)

    masks_value = state.get("masks")
    if masks_value is None:
        masks_np = np.empty((0, height, width), dtype=bool)
    else:
        masks_np = to_numpy(masks_value).astype(bool)
    if masks_np.ndim == 4:
        if masks_np.shape[1] != 1:
            raise ValueError(f"PCS masks must have one channel, got shape {masks_np.shape}")
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim != 3 or tuple(masks_np.shape[1:]) != (height, width):
        raise ValueError(
            f"PCS mask shape {masks_np.shape} does not match image {(height, width)}"
        )
    instance_count = int(masks_np.shape[0])

    boxes_value = state.get("boxes")
    if boxes_value is None:
        boxes_np = np.empty((0, 4), dtype=np.float32)
    else:
        boxes_np = to_numpy(boxes_value).astype(np.float32)
        if boxes_np.size == 0:
            boxes_np = np.empty((0, 4), dtype=np.float32)
    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"PCS boxes must have shape Nx4, got {boxes_np.shape}")

    scores_value = state.get("scores")
    if scores_value is None:
        scores_np = np.empty((0,), dtype=np.float32)
    else:
        scores_np = to_numpy(scores_value).astype(np.float32).reshape(-1)
    if len(boxes_np) != instance_count or len(scores_np) != instance_count:
        raise ValueError(
            "PCS output count mismatch: "
            f"masks={instance_count}, boxes={len(boxes_np)}, scores={len(scores_np)}"
        )
    if not np.isfinite(boxes_np).all() or not np.isfinite(scores_np).all():
        raise ValueError("PCS boxes and scores must be finite")

    probs = state.get("masks_logits")
    probs_np = [] if probs is None else to_numpy(probs).astype(np.float32)
    if isinstance(probs_np, np.ndarray) and probs_np.ndim == 4:
        if probs_np.shape[1] != 1:
            raise ValueError(f"PCS mask logits must have one channel, got shape {probs_np.shape}")
        probs_np = probs_np[:, 0]
    elif isinstance(probs_np, np.ndarray) and probs_np.ndim == 2:
        probs_np = probs_np[None, ...]
    if isinstance(probs_np, np.ndarray) and len(probs_np) != instance_count:
        raise ValueError(
            f"PCS mask logits count {len(probs_np)} does not match masks {instance_count}"
        )
    return {
        "masks": [m for m in masks_np],
        "boxes": [base._clip_box_xyxy(b.tolist(), width, height) for b in boxes_np],
        "scores": [float(v) for v in scores_np],
        "pcs_fullres_prob": [] if isinstance(probs_np, list) else [p for p in probs_np],
    }


def review_row(
    group: PcsCategoryImage,
    prompt_count: int,
    confidence: float,
    side: str = "",
    input_overlay: str = "",
    pcs_overlay: str = "",
    pred_count: int | str = "",
) -> dict[str, Any]:
    return {
        "group_id": f"{group.group_id}_p{prompt_count}",
        "source": "O3",
        "layer": group.layer,
        "dataset": group.dataset,
        "split": group.split,
        "image_path": group.image_path,
        "image_id": group.image_id,
        "label": group.label,
        "annotation_relpath": group.annotation_relpath,
        "annotation_sha256": group.annotation_sha256,
        "image_sha256": group.image_sha256,
        "text_prompt": "",
        "confidence": float(confidence),
        "prompt_count": prompt_count,
        "gt_count": len(group.boxes),
        "pred_count": pred_count,
        "prompt_boxes_xyxy": json.dumps(group.boxes[:prompt_count], ensure_ascii=False),
        "all_gt_boxes_xyxy": json.dumps(group.boxes, ensure_ascii=False),
        "side_by_side": side,
        "input_overlay": input_overlay,
        "pcs_overlay": pcs_overlay,
        "success": "",
        "review_comment": "",
    }


def write_review(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "group_id",
        "source",
        "layer",
        "dataset",
        "split",
        "image_path",
        "image_id",
        "label",
        "annotation_relpath",
        "annotation_sha256",
        "image_sha256",
        "text_prompt",
        "confidence",
        "prompt_count",
        "gt_count",
        "pred_count",
        "prompt_boxes_xyxy",
        "all_gt_boxes_xyxy",
        "side_by_side",
        "input_overlay",
        "pcs_overlay",
        "success",
        "review_comment",
    ]
    path = out_dir / "review_sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_class_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["layer"], row["label"], str(row["prompt_count"]))
        item = summary.setdefault(
            key,
            {
                "layer": row["layer"],
                "label": row["label"],
                "prompt_count": row["prompt_count"],
                "total_groups": 0,
                "reviewed_count": 0,
                "success_count": 0,
                "fail_count": 0,
            },
        )
        item["total_groups"] += 1
        value = str(row.get("success", "")).strip().lower()
        if value in base.SUCCESS_VALUES:
            item["reviewed_count"] += 1
            item["success_count"] += 1
        elif value in base.FAIL_VALUES:
            item["reviewed_count"] += 1
            item["fail_count"] += 1
    with (out_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["layer", "label", "prompt_count", "total_groups", "reviewed_count", "success_count", "fail_count", "success_rate", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(summary):
            item = summary[key]
            reviewed = int(item["reviewed_count"])
            rate = "" if reviewed == 0 else f"{float(item['success_count']) / reviewed:.6f}"
            writer.writerow({**item, "success_rate": rate, "status": "pending" if reviewed == 0 else "reviewed"})


def run_groups(
    groups: list[PcsCategoryImage],
    out_dir: Path,
    prompt_counts: list[int],
    threshold: float,
    device: str,
    dry_run: bool,
) -> None:
    if not groups:
        raise ValueError("At least one PCS group is required")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("PCS group_id values must be unique")
    if (
        not prompt_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in prompt_counts
        )
        or len(set(prompt_counts)) != len(prompt_counts)
    ):
        raise ValueError("prompt_counts must contain unique positive integers")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")

    expected_run_ids = [
        f"{group.group_id}_p{prompt_count}"
        for group in groups
        for prompt_count in prompt_counts
        if len(group.boxes) >= prompt_count
    ]
    if not expected_run_ids:
        raise ValueError("No PCS runs can be generated from the selected groups")
    if len(set(expected_run_ids)) != len(expected_run_ids):
        raise ValueError("PCS run_id values must be unique")

    eval_utils.prepare_empty_output_dir(out_dir)
    (out_dir / "groups").mkdir(parents=True, exist_ok=True)
    eval_utils.write_json_atomic(
        out_dir / "selected_groups.json",
        [asdict(group) for group in groups],
    )

    rows: list[dict[str, Any]] = []
    if dry_run:
        for group in groups:
            for prompt_count in prompt_counts:
                if len(group.boxes) >= prompt_count:
                    rows.append(review_row(group, prompt_count, threshold))
        write_review(rows, out_dir)
        write_class_summary(rows, out_dir)
        return

    predictor = base.init_image_predictor(device)
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for group_index, group in enumerate(groups, start=1):
            with Image.open(group.image_path) as image_file:
                image = image_file.convert("RGB")
            for prompt_count in prompt_counts:
                if len(group.boxes) < prompt_count:
                    continue
                prompt_boxes = group.boxes[:prompt_count]
                run_id = f"{group.group_id}_p{prompt_count}"
                run_dir = out_dir / "groups" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[{group_index}/{len(groups)}] PCS {run_id} "
                    f"prompts={prompt_count} gt={len(group.boxes)}"
                )
                pcs = run_pcs_from_boxes(
                    predictor, image, prompt_boxes, float(threshold)
                )
                input_image = draw_pcs_input_overlay(image, group, prompt_boxes)
                pcs_image = draw_pcs_result_overlay(image, group, pcs)
                side = base.side_by_side(input_image, pcs_image)
                input_path = run_dir / "input_overlay.png"
                pcs_path = run_dir / "pcs_overlay.png"
                side_path = run_dir / "side_by_side.png"
                input_image.save(input_path)
                pcs_image.save(pcs_path)
                side.save(side_path)

                pred_instances = []
                for instance_index, (mask, box, score) in enumerate(
                    zip(pcs["masks"], pcs["boxes"], pcs["scores"])
                ):
                    mask_artifact = eval_utils.save_binary_mask(
                        out_dir,
                        f"groups/{run_id}/pred_masks/pred_{instance_index:04d}.png",
                        mask,
                    )
                    pred_instances.append(
                        {
                            "instance_index": instance_index,
                            "mask_artifact": mask_artifact,
                            "box_xyxy": box,
                            "score": float(score),
                        }
                    )

                pred_record = {
                    **asdict(group),
                    "source": "O3",
                    "run_id": run_id,
                    "text_prompt": "",
                    "confidence": float(threshold),
                    "prompt_count": prompt_count,
                    "prompt_boxes_xyxy": prompt_boxes,
                    "pred_count": len(pcs["boxes"]),
                    "pred_boxes_xyxy": pcs["boxes"],
                    "scores": pcs["scores"],
                    "pred_instances": pred_instances,
                    "input_overlay": base.relative(input_path, out_dir),
                    "pcs_overlay": base.relative(pcs_path, out_dir),
                    "side_by_side": base.relative(side_path, out_dir),
                }
                eval_utils.write_json_atomic(run_dir / "prediction.json", pred_record)
                stream.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                rows.append(
                    review_row(
                        group,
                        prompt_count,
                        float(threshold),
                        side=base.relative(side_path, out_dir),
                        input_overlay=base.relative(input_path, out_dir),
                        pcs_overlay=base.relative(pcs_path, out_dir),
                        pred_count=len(pcs["boxes"]),
                    )
                )
        stream.flush()
        os.fsync(stream.fileno())

    write_review(rows, out_dir)
    write_class_summary(rows, out_dir)
    eval_utils.write_complete_run_manifest(
        out_dir,
        run_kind="pcs_o3_grouped_eval",
        selected_manifest_path="selected_groups.json",
        predictions_jsonl_path="predictions.jsonl",
        expected_selected_ids=[group.group_id for group in groups],
        expected_prediction_ids=expected_run_ids,
        metadata={
            "source": "O3",
            "confidence": float(threshold),
            "prompt_counts": list(prompt_counts),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o3-root", default="/data/zhengqiyuan/ADC_contour/datasets/O3_coco")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--images-per-category", type=int, default=3)
    parser.add_argument("--prompt-counts", default="1,2,3")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--split-order", default="val,train,test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0, help="debug limit after selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT
        / ".runtime"
        / "eval"
        / "pcs_o3_grouped_eval"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    prompt_counts = [
        int(value.strip())
        for value in args.prompt_counts.split(",")
        if value.strip()
    ]
    if (
        not prompt_counts
        or any(value <= 0 for value in prompt_counts)
        or len(set(prompt_counts)) != len(prompt_counts)
    ):
        raise ValueError("--prompt-counts must contain unique positive integers")
    if not np.isfinite(args.confidence) or not 0.0 <= float(args.confidence) <= 1.0:
        raise ValueError("--confidence must be finite and within [0, 1]")
    split_order = [
        split.strip() for split in args.split_order.split(",") if split.strip()
    ]
    if not split_order:
        raise ValueError("--split-order must contain at least one split")

    groups = collect_o3_category_images(
        Path(args.o3_root),
        args.images_per_category,
        split_order,
        max(prompt_counts),
    )
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("No PCS groups selected")
    print(f"Selected {len(groups)} O3 category-image groups")
    print(
        "Runs to generate: "
        f"{sum(1 for group in groups for count in prompt_counts if len(group.boxes) >= count)}"
    )
    print(f"Output directory: {out_dir}")
    run_groups(
        groups,
        out_dir,
        prompt_counts,
        float(args.confidence),
        args.device,
        args.dry_run,
    )
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
