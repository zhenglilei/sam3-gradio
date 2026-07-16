#!/usr/bin/env python3
"""Batch PVS bbox prompting from labeled O3 and T4 COCO annotations.

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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import offline_eval_utils as eval_utils


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
    category_label: str
    annotation_relpath: str
    annotation_sha256: str
    image_sha256: str
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


def _require_dataset_root(root: Path, source: str) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{source} dataset root does not exist: {root}")
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{source} dataset root must be a non-symlink directory: {root}")
    return root


def _coco_integer(value: Any, *, field: str, annotation_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"COCO {field} must be an integer in {annotation_path}")
    return value


def _coco_relative_path(value: Any, *, field: str, annotation_path: Path) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or ":" in text
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"COCO {field} is not a safe relative path in {annotation_path}: {text!r}"
        )
    return path


def _load_coco(
    annotation_path: Path,
) -> tuple[dict[int, str], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read COCO annotation document {annotation_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"COCO document must be an object: {annotation_path}")
    for field in ("categories", "images", "annotations"):
        if not isinstance(document.get(field), list):
            raise ValueError(f"COCO schema requires a {field} list: {annotation_path}")

    categories: dict[int, str] = {}
    for record in document["categories"]:
        if not isinstance(record, dict):
            raise ValueError(f"COCO category must be an object: {annotation_path}")
        category_id = _coco_integer(
            record.get("id"), field="category id", annotation_path=annotation_path
        )
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError(f"COCO category {category_id} has no name: {annotation_path}")
        if category_id in categories:
            raise ValueError(f"Duplicate COCO category id {category_id}: {annotation_path}")
        categories[category_id] = name

    images: dict[int, dict[str, Any]] = {}
    for record in document["images"]:
        if not isinstance(record, dict):
            raise ValueError(f"COCO image must be an object: {annotation_path}")
        image_id = _coco_integer(
            record.get("id"), field="image id", annotation_path=annotation_path
        )
        if image_id in images:
            raise ValueError(f"Duplicate COCO image id {image_id}: {annotation_path}")
        _coco_relative_path(
            record.get("file_name"), field="image file_name", annotation_path=annotation_path
        )
        for field in ("width", "height"):
            value = _coco_integer(
                record.get(field), field=f"image {field}", annotation_path=annotation_path
            )
            if value <= 0:
                raise ValueError(f"COCO image {field} must be positive: {annotation_path}")
        images[image_id] = record

    annotations: list[dict[str, Any]] = []
    annotation_ids: set[int] = set()
    for record in document["annotations"]:
        if not isinstance(record, dict):
            raise ValueError(f"COCO annotation must be an object: {annotation_path}")
        annotation_id = _coco_integer(
            record.get("id"), field="annotation id", annotation_path=annotation_path
        )
        image_id = _coco_integer(
            record.get("image_id"), field="annotation image_id", annotation_path=annotation_path
        )
        category_id = _coco_integer(
            record.get("category_id"),
            field="annotation category_id",
            annotation_path=annotation_path,
        )
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate COCO annotation id {annotation_id}: {annotation_path}")
        if image_id not in images:
            raise ValueError(
                f"COCO annotation {annotation_id} references unknown image_id "
                f"{image_id}: {annotation_path}"
            )
        if category_id not in categories:
            raise ValueError(
                f"COCO annotation {annotation_id} references unknown category_id "
                f"{category_id}: {annotation_path}"
            )
        bbox = record.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"COCO annotation {annotation_id} bbox must be [x,y,w,h]: {annotation_path}"
            )
        try:
            bbox_array = np.asarray(bbox, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"COCO annotation {annotation_id} bbox is not numeric: {annotation_path}"
            ) from exc
        if (
            not np.isfinite(bbox_array).all()
            or bbox_array[2] <= 0
            or bbox_array[3] <= 0
        ):
            raise ValueError(f"COCO annotation {annotation_id} bbox is invalid: {annotation_path}")
        if "segmentation" not in record:
            raise ValueError(
                f"COCO annotation {annotation_id} has no segmentation: {annotation_path}"
            )
        annotation_ids.add(annotation_id)
        annotations.append(record)
    return categories, images, sorted(
        annotations, key=lambda record: (int(record["image_id"]), int(record["id"]))
    )


def _annotation_label(
    annotation: dict[str, Any],
    category_label: str,
    annotation_path: Path,
) -> str:
    if "original_label" not in annotation:
        return category_label
    value = annotation["original_label"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"COCO annotation {annotation.get('id')} original_label must be a "
            f"non-empty string: {annotation_path}"
        )
    return value.strip()


def _resolve_coco_image(
    dataset_dir: Path,
    image_record: dict[str, Any],
    annotation_path: Path,
    *,
    split: str | None,
) -> Path:
    relative_name = _coco_relative_path(
        image_record.get("file_name"),
        field="image file_name",
        annotation_path=annotation_path,
    )
    if split is not None and relative_name.parts[0] != split:
        image_path = dataset_dir.joinpath(split, *relative_name.parts)
    else:
        image_path = dataset_dir.joinpath(*relative_name.parts)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"COCO image does not exist: {image_path}")
    if image_path.is_symlink():
        raise ValueError(f"COCO image must not be a symlink: {image_path}")
    with Image.open(image_path) as image:
        actual_size = image.size
    declared_size = (int(image_record["width"]), int(image_record["height"]))
    if actual_size != declared_size:
        raise ValueError(
            f"COCO image dimensions {declared_size} do not match {actual_size}: {image_path}"
        )
    return image_path


def collect_o3_samples(o3_root: Path, per_class: int, splits: set[str]) -> list[EvalSample]:
    o3_root = _require_dataset_root(o3_root, "O3")
    if not splits:
        raise ValueError("O3 splits must not be empty")
    buckets: dict[str, list[EvalSample]] = defaultdict(list)
    image_hashes: dict[Path, str] = {}
    annotation_paths = sorted(o3_root.glob("*_coco/annotations/instances_*.json"))
    for annotation_path in annotation_paths:
        split = annotation_path.stem.removeprefix("instances_")
        if split not in splits:
            continue
        dataset_dir = annotation_path.parent.parent
        dataset_name = dataset_dir.name
        categories, images, annotations = _load_coco(annotation_path)
        annotation_relpath = annotation_path.relative_to(o3_root).as_posix()
        annotation_sha256 = eval_utils.file_sha256(annotation_path)
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            category_label = categories[category_id]
            if len(buckets[category_label]) >= per_class:
                continue
            image_record = images[int(annotation["image_id"])]
            image_path = _resolve_coco_image(
                dataset_dir,
                image_record,
                annotation_path,
                split=split,
            )
            if image_path not in image_hashes:
                image_hashes[image_path] = eval_utils.file_sha256(image_path)
            image_sha256 = image_hashes[image_path]
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            sample_index = len(buckets[category_label]) + 1
            buckets[category_label].append(
                EvalSample(
                    sample_id=f"o3_{_safe_name(category_label)}_{sample_index:03d}",
                    source="O3",
                    dataset=dataset_name,
                    split=split,
                    image_path=str(image_path),
                    image_id=str(image_record["id"]),
                    annotation_id=str(annotation["id"]),
                    label=category_label,
                    bbox_xyxy=_clip_box_xyxy(
                        [x, y, x + width, y + height],
                        int(image_record["width"]),
                        int(image_record["height"]),
                    ),
                    category_label=category_label,
                    annotation_relpath=annotation_relpath,
                    annotation_sha256=annotation_sha256,
                    image_sha256=image_sha256,
                )
            )
    samples: list[EvalSample] = []
    for label in sorted(buckets):
        samples.extend(buckets[label][:per_class])
    return samples


def collect_t4_samples(t4_root: Path, scope: str) -> list[EvalSample]:
    if scope == "none":
        return []
    t4_root = _require_dataset_root(t4_root, "T4")
    samples: list[EvalSample] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    image_hashes: dict[Path, str] = {}
    annotation_paths = sorted(t4_root.glob("*/annotations/instances_all.json"))
    for annotation_path in annotation_paths:
        layer_dir = annotation_path.parent.parent
        layer = layer_dir.name
        categories, images, annotations = _load_coco(annotation_path)
        annotation_relpath = annotation_path.relative_to(t4_root).as_posix()
        annotation_sha256 = eval_utils.file_sha256(annotation_path)
        for annotation in annotations:
            category_label = categories[int(annotation["category_id"])]
            label = _annotation_label(
                annotation, category_label, annotation_path
            )
            image_record = images[int(annotation["image_id"])]
            image_path = _resolve_coco_image(
                layer_dir,
                image_record,
                annotation_path,
                split=None,
            )
            if image_path not in image_hashes:
                image_hashes[image_path] = eval_utils.file_sha256(image_path)
            image_sha256 = image_hashes[image_path]
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            counter_key = (layer, label)
            counters[counter_key] += 1
            samples.append(
                EvalSample(
                    sample_id=(
                        f"t4_{_safe_name(layer)}_{_safe_name(label)}_"
                        f"{counters[counter_key]:04d}"
                    ),
                    source="T4",
                    dataset=f"{t4_root.name}/{layer}",
                    split="original_size",
                    image_path=str(image_path),
                    image_id=str(image_record["id"]),
                    annotation_id=str(annotation["id"]),
                    label=label,
                    bbox_xyxy=_clip_box_xyxy(
                        [x, y, x + width, y + height],
                        int(image_record["width"]),
                        int(image_record["height"]),
                    ),
                    category_label=category_label,
                    annotation_relpath=annotation_relpath,
                    annotation_sha256=annotation_sha256,
                    image_sha256=image_sha256,
                    label_shape_type=str(
                        annotation.get("source_shape_type")
                        or annotation.get("shape_type")
                        or ""
                    ),
                )
            )
    return samples


def require_requested_sources(
    requested_sources: set[str],
    counts: dict[str, int],
) -> None:
    missing = sorted(
        source
        for source in requested_sources
        if int(counts.get(source.lower(), counts.get(source, 0))) <= 0
    )
    if missing:
        raise ValueError(
            "Requested evaluation source(s) produced zero samples: "
            + ", ".join(missing)
        )


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


def run_samples(
    samples: list[EvalSample],
    out_dir: Path,
    device: str,
    dry_run: bool = False,
) -> None:
    if not samples:
        raise ValueError("At least one evaluation sample is required")
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Evaluation sample_id values must be unique")

    eval_utils.prepare_empty_output_dir(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=False)
    selected_path = out_dir / "selected_samples.json"
    eval_utils.write_json_atomic(selected_path, [asdict(sample) for sample in samples])

    if dry_run:
        rows = [
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "dataset": sample.dataset,
                "split": sample.split,
                "image_path": sample.image_path,
                "annotation_id": sample.annotation_id,
                "label": sample.label,
                "output_label": sample.label,
                "input_bbox_xyxy": json.dumps(sample.bbox_xyxy),
            }
            for sample in samples
        ]
        write_review_and_summary(rows, out_dir)
        return

    predictor = init_image_predictor(device)
    state_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as jsonl:
        for index, sample in enumerate(samples, start=1):
            image_path = Path(sample.image_path)
            if eval_utils.file_sha256(image_path) != sample.image_sha256:
                raise ValueError(f"Source image changed after sample selection: {image_path}")
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
            mask = np.asarray(pred["mask"], dtype=bool)
            if mask.shape != (image.height, image.width):
                raise ValueError(
                    f"Prediction mask shape {mask.shape} does not match "
                    f"image shape {(image.height, image.width)} for {sample.sample_id}"
                )
            score = float(pred["score"])
            pred_box = _mask_box(mask)

            sample_dir = out_dir / "samples" / sample.sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            input_overlay = draw_input_overlay(image, sample)
            pvs_overlay = draw_pvs_overlay(image, sample, mask, pred_box, score)
            side = side_by_side(input_overlay, pvs_overlay)
            input_overlay_path = sample_dir / "input_overlay.png"
            pvs_overlay_path = sample_dir / "pvs_overlay.png"
            side_path = sample_dir / "side_by_side.png"
            pred_path = sample_dir / "prediction.json"
            input_overlay.save(input_overlay_path)
            pvs_overlay.save(pvs_overlay_path)
            side.save(side_path)
            mask_artifact = eval_utils.save_binary_mask(
                out_dir,
                f"samples/{sample.sample_id}/mask.png",
                mask,
            )

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
                "mask": mask_artifact["path"],
                "mask_artifact": mask_artifact,
            }
            eval_utils.write_json_atomic(pred_path, prediction)
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
                    "mask": mask_artifact["path"],
                }
            )
        jsonl.flush()
        os.fsync(jsonl.fileno())
    write_review_and_summary(rows, out_dir)
    eval_utils.write_complete_run_manifest(
        out_dir,
        run_kind="pvs_bbox_label_eval",
        selected_manifest_path="selected_samples.json",
        predictions_jsonl_path="predictions.jsonl",
        expected_selected_ids=sample_ids,
        expected_prediction_ids=sample_ids,
        metadata={
            "sources": sorted({sample.source for sample in samples}),
            "device": str(device),
            "sample_count": len(samples),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 PVS bbox prompting from strict O3 and T4 COCO labels.")
    parser.add_argument("--o3-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco"))
    parser.add_argument("--t4-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/T4/original_size"))
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
    requested_sources: set[str] = set()
    source_counts = {"o3": 0, "t4": 0}
    if args.datasets in {"both", "o3"}:
        requested_sources.add("o3")
        o3_samples = collect_o3_samples(
            args.o3_root,
            max(1, int(args.o3_per_class)),
            splits,
        )
        source_counts["o3"] = len(o3_samples)
        samples.extend(o3_samples)
    if args.datasets in {"both", "t4"} and args.t4_scope != "none":
        requested_sources.add("t4")
        t4_samples = collect_t4_samples(args.t4_root, args.t4_scope)
        source_counts["t4"] = len(t4_samples)
        samples.extend(t4_samples)
    require_requested_sources(requested_sources, source_counts)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
        capped_counts = {
            "o3": sum(sample.source == "O3" for sample in samples),
            "t4": sum(sample.source == "T4" for sample in samples),
        }
        require_requested_sources(requested_sources, capped_counts)
    if not samples:
        raise SystemExit("No evaluation samples found.")
    print(f"Selected {len(samples)} samples")
    print(f"Output directory: {out_dir}")
    run_samples(samples, out_dir, args.device, dry_run=bool(args.dry_run))
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
