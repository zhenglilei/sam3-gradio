#!/usr/bin/env python3
"""Grouped whole-image PVS bbox visualization for O3 and T4 labels.

O3 output: 3 original images per layer dataset (ACT/BSM/GE1/GE2), 12 grouped
visualizations total by default.
T4 output: one visualization per existing large-image/layer JSON. The current
remote dataset has ACT, GE1, and GE2 directories; if a large image is missing a
layer JSON, that group is naturally skipped.
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
from PIL import Image, ImageDraw

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
    (0, 255, 120), (255, 180, 0), (0, 180, 255), (255, 80, 200),
    (160, 255, 0), (180, 120, 255), (255, 80, 80), (80, 255, 255),
]


@dataclass
class ImageGroup:
    group_id: str
    source: str
    layer: str
    dataset: str
    split: str
    image_path: str
    image_id: str
    items: list[Any]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe(value: Any) -> str:
    return base._safe_name(value)


def collect_o3_groups(o3_root: Path, images_per_layer: int, split_order: list[str]) -> list[ImageGroup]:
    groups: list[ImageGroup] = []
    for dataset_dir in sorted(p for p in o3_root.iterdir() if p.is_dir() and p.name.endswith("_coco")):
        layer = dataset_dir.name.replace("_coco", "")
        picked = 0
        for split in split_order:
            ann_path = dataset_dir / "annotations" / f"instances_{split}.json"
            if not ann_path.exists():
                continue
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            cats = {int(c["id"]): str(c.get("name") or c["id"]) for c in data.get("categories", [])}
            images = {int(img["id"]): img for img in data.get("images", [])}
            anns_by_image: dict[int, list[dict[str, Any]]] = {}
            for ann in sorted(data.get("annotations", []), key=lambda a: int(a.get("id", 0))):
                anns_by_image.setdefault(int(ann.get("image_id")), []).append(ann)
            for image_id in sorted(images):
                if picked >= images_per_layer:
                    break
                anns = anns_by_image.get(image_id, [])
                if not anns:
                    continue
                img_rec = images[image_id]
                image_path = base._find_image_path(dataset_dir, split, str(img_rec.get("file_name")))
                if image_path is None:
                    continue
                width = int(img_rec.get("width") or 0)
                height = int(img_rec.get("height") or 0)
                if width <= 0 or height <= 0:
                    with Image.open(image_path) as im:
                        width, height = im.size
                picked += 1
                group_id = f"o3_{safe(layer)}_{picked:02d}_{safe(Path(str(img_rec.get('file_name'))).stem)[:70]}"
                items = []
                for idx, ann in enumerate(anns, start=1):
                    x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])]
                    label = cats.get(int(ann.get("category_id")), str(ann.get("category_id")))
                    items.append(base.EvalSample(
                        sample_id=f"{group_id}__ann{idx:04d}", source="O3", dataset=dataset_dir.name,
                        split=split, image_path=str(image_path), image_id=str(image_id),
                        annotation_id=str(ann.get("id")), label=label,
                        bbox_xyxy=base._clip_box_xyxy([x, y, x + w, y + h], width, height)
                    ))
                groups.append(ImageGroup(group_id, "O3", layer, dataset_dir.name, split, str(image_path), str(image_id), items))
                if picked >= images_per_layer:
                    break
    return groups


def collect_t4_groups(t4_root: Path, scope: str) -> list[ImageGroup]:
    if scope == "none":
        return []
    groups: list[ImageGroup] = []
    for json_path in sorted(t4_root.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = base._labelme_image_path(json_path, data)
        if image_path is None:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        rel = json_path.relative_to(t4_root)
        layer = rel.parts[0]
        group_id = f"t4_{safe(layer)}_{safe(json_path.stem)}"
        items = []
        for idx, shape in enumerate(data.get("shapes", []), start=1):
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            label = str(shape.get("label") or "unknown")
            items.append(base.EvalSample(
                sample_id=f"{group_id}__shape{idx:04d}", source="T4", dataset=str(rel.parent),
                split="original_size", image_path=str(image_path), image_id=str(rel),
                annotation_id=f"{json_path.stem}:shape{idx}", label=label,
                bbox_xyxy=base._clip_box_xyxy([x1, y1, x2, y2], width, height),
                label_shape_type=str(shape.get("shape_type") or ""),
            ))
        if items:
            groups.append(ImageGroup(group_id, "T4", layer, str(rel.parent), "original_size", str(image_path), str(rel), items))
    return groups



def font(size: int):
    try:
        from PIL import ImageFont
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def text_box(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int], size: int) -> None:
    fnt = font(size)
    bbox = draw.textbbox(xy, text, font=fnt)
    pad = max(2, size // 6)
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=fnt)


def draw_group_input(image: Image.Image, group: ImageGroup) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    text_size = 11 if len(group.items) > 120 else 14
    for i, item in enumerate(group.items, start=1):
        color = PALETTE[(i - 1) % len(PALETTE)]
        text_box(draw, (item.bbox_xyxy[0], max(0, item.bbox_xyxy[1] - text_size - 6)), item.label, color, text_size)
        draw.rectangle(item.bbox_xyxy, outline=color, width=2)
    text_box(draw, (8, 8), f"{group.group_id} input labels={len(group.items)}", (0, 255, 120), 16)
    return out


def draw_group_pvs(image: Image.Image, group: ImageGroup, preds: list[dict[str, Any]]) -> tuple[Image.Image, np.ndarray]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    union = np.zeros(arr.shape[:2], dtype=bool)
    for i, pred in enumerate(preds, start=1):
        mask = np.asarray(pred["mask"], dtype=bool)
        union |= mask
        color = np.asarray(PALETTE[(i - 1) % len(PALETTE)], dtype=np.float32)
        arr_f = arr.astype(np.float32)
        arr_f[mask] = arr_f[mask] * 0.62 + color * 0.38
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    text_size = 10 if len(preds) > 120 else 13
    for i, pred in enumerate(preds, start=1):
        item = pred["item"]
        color = PALETTE[(i - 1) % len(PALETTE)]
        draw.rectangle(pred["pred_bbox_xyxy"], outline=color, width=2)
        x1, y1, _, _ = pred["pred_bbox_xyxy"]
        text_box(draw, (x1, max(0, y1 - text_size - 6)), f"{item.label} {pred['score']:.2f}", color, text_size)
    text_box(draw, (8, 8), f"{group.group_id} PVS outputs={len(preds)}", (255, 220, 80), 16)
    return out, union


def review_row(group: ImageGroup, item: Any, side: str = "", overlay: str = "", mask: str = "", pred_box: Any = "", score: Any = "") -> dict[str, Any]:
    return {
        "group_id": group.group_id, "sample_id": item.sample_id, "source": item.source, "layer": group.layer,
        "dataset": item.dataset, "split": item.split, "image_path": item.image_path,
        "annotation_id": item.annotation_id, "label": item.label, "output_label": item.label,
        "input_bbox_xyxy": json.dumps(item.bbox_xyxy), "pred_bbox_xyxy": json.dumps(pred_box) if pred_box else "",
        "score": f"{float(score):.6f}" if score != "" else "", "side_by_side": side, "pvs_overlay": overlay,
        "mask": mask, "success": "", "review_comment": "",
    }


def write_review(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = ["group_id", "sample_id", "source", "layer", "dataset", "split", "image_path", "annotation_id", "label", "output_label", "input_bbox_xyxy", "pred_bbox_xyxy", "score", "side_by_side", "pvs_overlay", "mask", "success", "review_comment"]
    path = out_dir / "review_sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    base.summarize_review_sheet(path, out_dir / "class_summary.csv")


def run_groups(groups: list[ImageGroup], out_dir: Path, device: str, dry_run: bool, max_items_per_group: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "groups").mkdir(parents=True, exist_ok=True)
    manifest = []
    dry_rows = []
    for group in groups:
        items = group.items[:max_items_per_group] if max_items_per_group > 0 else group.items
        manifest.append({**asdict(group), "items": [asdict(item) for item in items]})
        dry_rows.extend(review_row(group, item) for item in items)
    (out_dir / "selected_groups.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if dry_run:
        write_review(dry_rows, out_dir)
        return

    predictor = base.init_image_predictor(device)
    rows = []
    pred_jsonl = out_dir / "predictions.jsonl"
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for gi, group in enumerate(groups, start=1):
            items = group.items[:max_items_per_group] if max_items_per_group > 0 else group.items
            with Image.open(group.image_path) as im:
                image = im.convert("RGB")
            print(f"[{gi}/{len(groups)}] set_image {group.group_id} items={len(items)}")
            state = predictor.set_image(image)
            preds = []
            for item in items:
                pred = base.predict_from_box(predictor, state, item.bbox_xyxy)
                preds.append({
                    "item": item, "mask": pred["mask"], "score": float(pred["score"]),
                    "pred_bbox_xyxy": base._mask_box(pred["mask"]), "best_index": int(pred["best_index"]),
                    "candidate_scores": pred["candidate_scores"],
                })
            group_dir = out_dir / "groups" / group.group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            input_overlay = draw_group_input(image, ImageGroup(**{**asdict(group), "items": items}))
            pvs_overlay, union = draw_group_pvs(image, ImageGroup(**{**asdict(group), "items": items}), preds)
            side = base.side_by_side(input_overlay, pvs_overlay)
            input_path = group_dir / "input_overlay.png"
            pvs_path = group_dir / "pvs_overlay.png"
            side_path = group_dir / "side_by_side.png"
            mask_path = group_dir / "combined_mask.png"
            pred_path = group_dir / "prediction.json"
            input_overlay.save(input_path)
            pvs_overlay.save(pvs_path)
            side.save(side_path)
            Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(mask_path)
            pred_items = []
            for pred in preds:
                item = pred["item"]
                rows.append(review_row(group, item, base.relative(side_path, out_dir), base.relative(pvs_path, out_dir), base.relative(mask_path, out_dir), pred["pred_bbox_xyxy"], pred["score"]))
                pred_items.append({**asdict(item), "output_label": item.label, "score": pred["score"], "pred_bbox_xyxy": pred["pred_bbox_xyxy"], "best_index": pred["best_index"], "candidate_scores": pred["candidate_scores"]})
            payload = {**asdict(group), "items": pred_items, "input_overlay": base.relative(input_path, out_dir), "pvs_overlay": base.relative(pvs_path, out_dir), "side_by_side": base.relative(side_path, out_dir), "combined_mask": base.relative(mask_path, out_dir)}
            pred_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_review(rows, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped whole-image PVS bbox visualization for O3/T4 labels.")
    parser.add_argument("--o3-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco"))
    parser.add_argument("--t4-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/T4/labelme_pairs/original_size"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", choices=["both", "o3", "t4"], default="both")
    parser.add_argument("--o3-images-per-layer", type=int, default=3)
    parser.add_argument("--o3-split-order", default="val,train")
    parser.add_argument("--t4-scope", choices=["all", "none"], default="all")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-items-per-group", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        out = args.summary_out or (args.summary_only.parent / "class_summary.csv")
        base.summarize_review_sheet(args.summary_only, out)
        print(f"Wrote summary: {out}")
        return
    out_dir = args.out_dir or (base.RUNTIME_DIR / "eval" / "pvs_bbox_grouped_eval" / time.strftime("%Y%m%d_%H%M%S"))
    groups: list[ImageGroup] = []
    if args.datasets in {"both", "o3"}:
        split_order = [s.strip() for s in args.o3_split_order.split(",") if s.strip()]
        groups.extend(collect_o3_groups(args.o3_root, max(1, args.o3_images_per_layer), split_order))
    if args.datasets in {"both", "t4"}:
        groups.extend(collect_t4_groups(args.t4_root, args.t4_scope))
    if args.max_groups > 0:
        groups = groups[:args.max_groups]
    if not groups:
        raise SystemExit("No image groups found")
    total_items = sum(len(g.items[:args.max_items_per_group] if args.max_items_per_group > 0 else g.items) for g in groups)
    print(f"Selected {len(groups)} image groups and {total_items} label bboxes")
    print(f"Output directory: {out_dir}")
    run_groups(groups, out_dir, args.device, bool(args.dry_run), int(args.max_items_per_group or 0))
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
