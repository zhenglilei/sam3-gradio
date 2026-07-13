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

def collect_o3_category_images(o3_root: Path, images_per_category: int, split_order: list[str], min_boxes: int) -> list[PcsCategoryImage]:
    selected: list[PcsCategoryImage] = []
    for dataset_dir in sorted(p for p in o3_root.iterdir() if p.is_dir() and p.name.endswith("_coco")):
        layer = dataset_dir.name.replace("_coco", "")
        candidates_by_label: dict[str, list[PcsCategoryImage]] = {}
        for split in split_order:
            ann_path = dataset_dir / "annotations" / f"instances_{split}.json"
            if not ann_path.exists():
                continue
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            categories = {int(c["id"]): str(c.get("name") or c["id"]) for c in data.get("categories", [])}
            images = {int(img["id"]): img for img in data.get("images", [])}
            anns_by_image_cat: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for ann in sorted(data.get("annotations", []), key=lambda a: int(a.get("id", 0))):
                anns_by_image_cat.setdefault((int(ann.get("image_id")), int(ann.get("category_id"))), []).append(ann)
            for (image_id, category_id), anns in sorted(anns_by_image_cat.items()):
                label = categories.get(category_id, str(category_id))
                image_record = images.get(image_id)
                if not image_record:
                    continue
                image_path = base._find_image_path(dataset_dir, split, str(image_record.get("file_name")))
                if image_path is None:
                    continue
                width = int(image_record.get("width") or 0)
                height = int(image_record.get("height") or 0)
                if width <= 0 or height <= 0:
                    with Image.open(image_path) as im:
                        width, height = im.size
                boxes = []
                ann_ids = []
                for ann in anns:
                    x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])]
                    boxes.append(base._clip_box_xyxy([x, y, x + w, y + h], width, height))
                    ann_ids.append(str(ann.get("id")))
                group_id = (
                    f"pcs_o3_{safe(layer)}_{safe(label)}_"
                    f"{safe(Path(str(image_record.get('file_name'))).stem)[:70]}"
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
                        image_file_name=str(image_record.get("file_name")),
                        category_id=category_id,
                        boxes=boxes,
                        annotation_ids=ann_ids,
                    )
                )
        for label in sorted(candidates_by_label):
            candidates = candidates_by_label[label]
            preferred = [g for g in candidates if len(g.boxes) >= min_boxes]
            fallback = [g for g in candidates if len(g.boxes) < min_boxes]
            for idx, group in enumerate((preferred + fallback)[:images_per_category], start=1):
                group.group_id = f"pcs_o3_{safe(group.layer)}_{safe(group.label)}_{idx:02d}_{safe(Path(group.image_file_name).stem)[:70]}"
                selected.append(group)
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
    state = predictor.set_image(image)
    for box in prompt_boxes:
        state = predictor.add_geometric_prompt(xyxy_to_cxcywh_norm(box, width, height), True, state)
    state = predictor.set_confidence_threshold(float(threshold), state)
    masks = state.get("masks")
    if masks is None or len(masks) == 0:
        return {"masks": [], "boxes": [], "scores": [], "pcs_fullres_prob": []}
    masks_np = to_numpy(masks).astype(bool)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    boxes_np = to_numpy(state.get("boxes")).astype(np.float32)
    scores_np = to_numpy(state.get("scores")).astype(np.float32).reshape(-1)
    probs = state.get("masks_logits")
    probs_np = [] if probs is None else to_numpy(probs).astype(np.float32)
    if isinstance(probs_np, np.ndarray) and probs_np.ndim == 4:
        probs_np = probs_np[:, 0]
    return {
        "masks": [m for m in masks_np],
        "boxes": [base._clip_box_xyxy(b.tolist(), width, height) for b in boxes_np],
        "scores": [float(v) for v in scores_np],
        "pcs_fullres_prob": [] if isinstance(probs_np, list) else [p for p in probs_np],
    }


def review_row(
    group: PcsCategoryImage,
    prompt_count: int,
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
        "text_prompt": "",
        "confidence": "0.5",
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


def run_groups(groups: list[PcsCategoryImage], out_dir: Path, prompt_counts: list[int], threshold: float, device: str, dry_run: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "groups").mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_groups.json").write_text(json.dumps([asdict(g) for g in groups], ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    if dry_run:
        for group in groups:
            for prompt_count in prompt_counts:
                if len(group.boxes) >= prompt_count:
                    rows.append(review_row(group, prompt_count))
        write_review(rows, out_dir)
        write_class_summary(rows, out_dir)
        return

    predictor = base.init_image_predictor(device)
    pred_jsonl = out_dir / "predictions.jsonl"
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for gi, group in enumerate(groups, start=1):
            with Image.open(group.image_path) as im:
                image = im.convert("RGB")
            for prompt_count in prompt_counts:
                if len(group.boxes) < prompt_count:
                    continue
                prompt_boxes = group.boxes[:prompt_count]
                run_id = f"{group.group_id}_p{prompt_count}"
                run_dir = out_dir / "groups" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{gi}/{len(groups)}] PCS {run_id} prompts={prompt_count} gt={len(group.boxes)}")
                pcs = run_pcs_from_boxes(predictor, image, prompt_boxes, threshold)
                input_img = draw_pcs_input_overlay(image, group, prompt_boxes)
                pcs_img = draw_pcs_result_overlay(image, group, pcs)
                side = base.side_by_side(input_img, pcs_img)
                input_path = run_dir / "input_overlay.png"
                pcs_path = run_dir / "pcs_overlay.png"
                side_path = run_dir / "side_by_side.png"
                input_img.save(input_path)
                pcs_img.save(pcs_path)
                side.save(side_path)
                pred_record = {
                    **asdict(group),
                    "run_id": run_id,
                    "text_prompt": "",
                    "confidence": threshold,
                    "prompt_count": prompt_count,
                    "prompt_boxes_xyxy": prompt_boxes,
                    "pred_count": len(pcs["boxes"]),
                    "pred_boxes_xyxy": pcs["boxes"],
                    "scores": pcs["scores"],
                    "input_overlay": base.relative(input_path, out_dir),
                    "pcs_overlay": base.relative(pcs_path, out_dir),
                    "side_by_side": base.relative(side_path, out_dir),
                }
                (run_dir / "prediction.json").write_text(json.dumps(pred_record, ensure_ascii=False, indent=2), encoding="utf-8")
                f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                rows.append(
                    review_row(
                        group,
                        prompt_count,
                        side=base.relative(side_path, out_dir),
                        input_overlay=base.relative(input_path, out_dir),
                        pcs_overlay=base.relative(pcs_path, out_dir),
                        pred_count=len(pcs["boxes"]),
                    )
                )
    write_review(rows, out_dir)
    write_class_summary(rows, out_dir)


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
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / ".runtime" / "eval" / "pcs_o3_grouped_eval" / time.strftime("%Y%m%d_%H%M%S")
    prompt_counts = [int(x.strip()) for x in args.prompt_counts.split(",") if x.strip()]
    min_boxes = max(prompt_counts) if prompt_counts else 1
    groups = collect_o3_category_images(Path(args.o3_root), args.images_per_category, [s.strip() for s in args.split_order.split(",") if s.strip()], min_boxes)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
    print(f"Selected {len(groups)} O3 category-image groups")
    print(f"Runs to generate: {sum(1 for g in groups for n in prompt_counts if len(g.boxes) >= n)}")
    print(f"Output directory: {out_dir}")
    run_groups(groups, out_dir, prompt_counts, args.confidence, args.device, args.dry_run)
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
