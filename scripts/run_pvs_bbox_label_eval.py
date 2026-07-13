#!/usr/bin/env python3
"""Batch PVS bbox prompting from labeled O3 COCO and T4 LabelMe annotations.

This script uses the same SAM3 PVS path as the demo: set_image once per image,
then predict_inst(box=...) for each labeled bbox. PVS is class agnostic, so the
output label is inherited from the input annotation label/category.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QY_CACHE_DIR = Path("/data/zhengqiyuan/.cache")
RUNTIME_DIR = REPO_ROOT / ".runtime"
for path in (
    RUNTIME_DIR,
    RUNTIME_DIR / "tmp",
    RUNTIME_DIR / "gradio",
    QY_CACHE_DIR,
    QY_CACHE_DIR / "huggingface",
    QY_CACHE_DIR / "huggingface" / "hub",
    QY_CACHE_DIR / "modelscope",
):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(RUNTIME_DIR / "tmp"))
os.environ.setdefault("TEMP", str(RUNTIME_DIR / "tmp"))
os.environ.setdefault("TMP", str(RUNTIME_DIR / "tmp"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(RUNTIME_DIR / "gradio"))
os.environ.setdefault("XDG_CACHE_HOME", str(QY_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(QY_CACHE_DIR / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(QY_CACHE_DIR / "huggingface" / "hub"))
os.environ.setdefault("MODELSCOPE_CACHE", str(QY_CACHE_DIR / "modelscope"))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUCCESS_VALUES = {"1", "true", "yes", "y", "success", "pass", "成功", "通过", "好", "及格"}
FAIL_VALUES = {"0", "false", "no", "n", "fail", "failed", "失败", "不通过", "差"}


@dataclass
class EvalSample:
    sample_id: str
    source: str
    dataset: str
    split: str
    image_path: str
    image_id: str
    annotation_id: str
    label: str
    bbox_xyxy: list[float]
    label_shape_type: str = ""


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_name(value: Any) -> str:
    text = str(value or "unknown").strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out).strip("._") or "unknown"


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _clip_box_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def _mask_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _find_image_path(base_dir: Path, split: str, file_name: str) -> Path | None:
    candidates = [
        base_dir / split / file_name,
        base_dir / file_name,
        base_dir / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    name = Path(file_name).name
    matches = list(base_dir.rglob(name))
    return matches[0] if matches else None


def _labelme_image_path(json_path: Path, data: dict[str, Any]) -> Path | None:
    image_path = data.get("imagePath")
    if image_path:
        candidate = json_path.parent / image_path
        if candidate.exists():
            return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    stem = json_path.stem
    for p in json_path.parent.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES:
            return p
    return None


def collect_o3_samples(o3_root: Path, per_class: int, splits: set[str]) -> list[EvalSample]:
    buckets: dict[str, list[EvalSample]] = defaultdict(list)
    for ann_path in sorted(o3_root.rglob("annotations/instances_*.json")):
        split = ann_path.stem.replace("instances_", "")
        if split not in splits:
            continue
        dataset_dir = ann_path.parent.parent
        dataset_name = dataset_dir.name
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c.get("name") or c["id"]) for c in data.get("categories", [])}
        images = {int(img["id"]): img for img in data.get("images", [])}
        anns = sorted(data.get("annotations", []), key=lambda a: (int(a.get("image_id", 0)), int(a.get("id", 0))))
        for ann in anns:
            cat_id = int(ann.get("category_id"))
            label = categories.get(cat_id, str(cat_id))
            if len(buckets[label]) >= per_class:
                continue
            image_record = images.get(int(ann.get("image_id")))
            if not image_record:
                continue
            image_path = _find_image_path(dataset_dir, split, str(image_record.get("file_name")))
            if image_path is None:
                continue
            x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])]
            width = int(image_record.get("width") or 0)
            height = int(image_record.get("height") or 0)
            if width <= 0 or height <= 0:
                with Image.open(image_path) as img:
                    width, height = img.size
            sample_index = len(buckets[label]) + 1
            buckets[label].append(
                EvalSample(
                    sample_id=f"o3_{_safe_name(label)}_{sample_index:03d}",
                    source="O3",
                    dataset=dataset_name,
                    split=split,
                    image_path=str(image_path),
                    image_id=str(image_record.get("id")),
                    annotation_id=str(ann.get("id")),
                    label=label,
                    bbox_xyxy=_clip_box_xyxy([x, y, x + w, y + h], width, height),
                )
            )
    samples: list[EvalSample] = []
    for label in sorted(buckets):
        samples.extend(buckets[label][:per_class])
    return samples


def collect_t4_samples(t4_root: Path, scope: str) -> list[EvalSample]:
    if scope == "none":
        return []
    samples: list[EvalSample] = []
    counters: dict[str, int] = defaultdict(int)
    for json_path in sorted(t4_root.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = _labelme_image_path(json_path, data)
        if image_path is None:
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        dataset = str(json_path.relative_to(t4_root).parent)
        for idx, shape in enumerate(data.get("shapes", []), start=1):
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            label = str(shape.get("label") or "unknown")
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            counters[label] += 1
            samples.append(
                EvalSample(
                    sample_id=f"t4_{_safe_name(label)}_{counters[label]:04d}",
                    source="T4",
                    dataset=dataset,
                    split="original_size",
                    image_path=str(image_path),
                    image_id=str(json_path.relative_to(t4_root)),
                    annotation_id=f"{json_path.stem}:shape{idx}",
                    label=label,
                    bbox_xyxy=_clip_box_xyxy([x1, y1, x2, y2], width, height),
                    label_shape_type=str(shape.get("shape_type") or ""),
                )
            )
    return samples


def init_image_predictor(device: str) -> Any:
    checkpoint_path = REPO_ROOT / "models" / "sam3.pt"
    bpe_path = REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SAM3 checkpoint: {checkpoint_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"Missing BPE vocabulary: {bpe_path}")
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        device=device,
        enable_inst_interactivity=True,
    )
    return Sam3Processor(model, device=device)


def predict_from_box(predictor: Any, base_state: dict[str, Any], box_xyxy: list[float]) -> dict[str, Any]:
    kwargs = {
        "box": np.asarray(box_xyxy, dtype=np.float32),
        "multimask_output": True,
        "return_logits": True,
    }
    import torch

    with torch.inference_mode():
        masks, scores, lowres_logits = predictor.model.predict_inst(base_state, **kwargs)
    masks_np = _to_numpy(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    scores_np = _to_numpy(scores).astype(np.float32).reshape(-1)
    lowres_np = _to_numpy(lowres_logits).astype(np.float32)
    best = int(np.argmax(scores_np)) if scores_np.size else 0
    return {
        "mask": masks_np[best] > 0,
        "score": float(scores_np[best]) if scores_np.size else 0.0,
        "lowres_logits": lowres_np[best] if lowres_np.ndim >= 3 else lowres_np,
        "candidate_scores": scores_np.astype(float).tolist(),
        "best_index": best,
    }


def _draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text(xy, text, fill=fill, font=font)


def draw_input_overlay(image: Image.Image, sample: EvalSample) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = sample.bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 80), width=4)
    _draw_text_box(draw, (x1, max(0.0, y1 - 24)), f"{sample.label} input bbox", (0, 255, 80))
    return out


def draw_pvs_overlay(image: Image.Image, sample: EvalSample, mask: np.ndarray, pred_box: list[float], score: float) -> Image.Image:
    base = image.convert("RGB")
    arr = np.asarray(base, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    tint = arr.copy()
    tint[mask_bool] = (0, 255, 120)
    arr = cv2.addWeighted(tint, 0.35, arr, 0.65, 0)
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = pred_box
    draw.rectangle([x1, y1, x2, y2], outline=(255, 170, 0), width=4)
    _draw_text_box(draw, (x1, max(0.0, y1 - 24)), f"{sample.label} score={score:.3f}", (255, 220, 80))
    return out


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    w = left.width + right.width
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(left.convert("RGB"), (0, 0))
    out.paste(right.convert("RGB"), (left.width, 0))
    return out


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_review_and_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    review_path = out_dir / "review_sheet.csv"
    fieldnames = [
        "sample_id",
        "source",
        "dataset",
        "split",
        "image_path",
        "annotation_id",
        "label",
        "output_label",
        "input_bbox_xyxy",
        "pred_bbox_xyxy",
        "score",
        "side_by_side",
        "pvs_overlay",
        "mask",
        "success",
        "review_comment",
    ]
    with review_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    summarize_review_sheet(review_path, out_dir / "class_summary.csv")


def summarize_review_sheet(review_sheet: Path, summary_out: Path) -> None:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    with review_sheet.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("source", ""), row.get("label", ""))
            group = groups.setdefault(
                key,
                {
                    "source": key[0],
                    "label": key[1],
                    "total_samples": 0,
                    "reviewed_count": 0,
                    "success_count": 0,
                    "fail_count": 0,
                },
            )
            group["total_samples"] += 1
            value = str(row.get("success", "")).strip().lower()
            if value in SUCCESS_VALUES:
                group["reviewed_count"] += 1
                group["success_count"] += 1
            elif value in FAIL_VALUES:
                group["reviewed_count"] += 1
                group["fail_count"] += 1
    fieldnames = ["source", "label", "total_samples", "reviewed_count", "success_count", "fail_count", "success_rate", "status"]
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            group = groups[key]
            reviewed = int(group["reviewed_count"])
            rate = "" if reviewed == 0 else f"{float(group['success_count']) / reviewed:.6f}"
            status = "pending" if reviewed == 0 else "reviewed"
            writer.writerow({**group, "success_rate": rate, "status": status})


def run_samples(samples: list[EvalSample], out_dir: Path, device: str, dry_run: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "selected_samples.json"
    selected_path.write_text(json.dumps([asdict(s) for s in samples], ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        rows = [
            {
                "sample_id": s.sample_id,
                "source": s.source,
                "dataset": s.dataset,
                "split": s.split,
                "image_path": s.image_path,
                "annotation_id": s.annotation_id,
                "label": s.label,
                "output_label": s.label,
                "input_bbox_xyxy": json.dumps(s.bbox_xyxy),
            }
            for s in samples
        ]
        write_review_and_summary(rows, out_dir)
        return

    predictor = init_image_predictor(device)
    state_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as jsonl:
        for index, sample in enumerate(samples, start=1):
            image_path = Path(sample.image_path)
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            box = _clip_box_xyxy(sample.bbox_xyxy, image.width, image.height)
            cache_key = str(image_path)
            if cache_key not in state_cache:
                print(f"[{index}/{len(samples)}] set_image {image_path}")
                state_cache[cache_key] = predictor.set_image(image)
            else:
                print(f"[{index}/{len(samples)}] reuse_image {image_path}")

            pred = predict_from_box(predictor, state_cache[cache_key], box)
            mask = pred["mask"]
            score = float(pred["score"])
            pred_box = _mask_box(mask)

            sample_dir = out_dir / "samples" / sample.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            input_overlay = draw_input_overlay(image, sample)
            pvs_overlay = draw_pvs_overlay(image, sample, mask, pred_box, score)
            side = side_by_side(input_overlay, pvs_overlay)
            input_overlay_path = sample_dir / "input_overlay.png"
            pvs_overlay_path = sample_dir / "pvs_overlay.png"
            side_path = sample_dir / "side_by_side.png"
            mask_path = sample_dir / "mask.png"
            pred_path = sample_dir / "prediction.json"
            input_overlay.save(input_overlay_path)
            pvs_overlay.save(pvs_overlay_path)
            side.save(side_path)
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

            prediction = {
                **asdict(sample),
                "output_label": sample.label,
                "input_bbox_xyxy": box,
                "score": score,
                "pred_bbox_xyxy": pred_box,
                "best_index": int(pred["best_index"]),
                "candidate_scores": pred["candidate_scores"],
                "input_overlay": relative(input_overlay_path, out_dir),
                "pvs_overlay": relative(pvs_overlay_path, out_dir),
                "side_by_side": relative(side_path, out_dir),
                "mask": relative(mask_path, out_dir),
            }
            pred_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
            jsonl.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "dataset": sample.dataset,
                    "split": sample.split,
                    "image_path": sample.image_path,
                    "annotation_id": sample.annotation_id,
                    "label": sample.label,
                    "output_label": sample.label,
                    "input_bbox_xyxy": json.dumps(box),
                    "pred_bbox_xyxy": json.dumps(pred_box),
                    "score": f"{score:.6f}",
                    "side_by_side": relative(side_path, out_dir),
                    "pvs_overlay": relative(pvs_overlay_path, out_dir),
                    "mask": relative(mask_path, out_dir),
                }
            )
    write_review_and_summary(rows, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 PVS bbox prompting from O3 COCO and T4 LabelMe labels.")
    parser.add_argument("--o3-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco"))
    parser.add_argument("--t4-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/T4/labelme_pairs/original_size"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", choices=["both", "o3", "t4"], default="both")
    parser.add_argument("--o3-per-class", type=int, default=3)
    parser.add_argument("--o3-splits", default="train,val", help="Comma-separated COCO splits to sample.")
    parser.add_argument("--t4-scope", choices=["all", "none"], default="all")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional total sample cap for smoke tests.")
    parser.add_argument("--device", default="auto", help="Device for SAM3 inference: auto, cuda, or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Only parse annotations and write review/sample manifests; do not load SAM3.")
    parser.add_argument("--summary-only", type=Path, default=None, help="Recompute class_summary.csv from an existing review_sheet.csv.")
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        summary_out = args.summary_out or (args.summary_only.parent / "class_summary.csv")
        summarize_review_sheet(args.summary_only, summary_out)
        print(f"Wrote summary: {summary_out}")
        return

    out_dir = args.out_dir or (RUNTIME_DIR / "eval" / "pvs_bbox_label_eval" / _now_stamp())
    splits = {s.strip() for s in args.o3_splits.split(",") if s.strip()}
    samples: list[EvalSample] = []
    if args.datasets in {"both", "o3"}:
        samples.extend(collect_o3_samples(args.o3_root, max(1, int(args.o3_per_class)), splits))
    if args.datasets in {"both", "t4"}:
        samples.extend(collect_t4_samples(args.t4_root, args.t4_scope))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit("No evaluation samples found.")
    print(f"Selected {len(samples)} samples")
    print(f"Output directory: {out_dir}")
    run_samples(samples, out_dir, args.device, dry_run=bool(args.dry_run))
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
