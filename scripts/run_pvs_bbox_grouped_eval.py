#!/usr/bin/env python3
"""Grouped whole-image PVS bbox evaluation artifacts for O3 and T4 COCO data.

O3 output: 3 original images per layer dataset (ACT/BSM/GE1/GE2), 12 grouped
visualizations total by default.
T4 output: one group per image in each layer's annotations/instances_all.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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

    annotation_relpath: str
    annotation_sha256: str
    image_sha256: str
    items: list[Any]
    category_label: str
    category_labels: list[str]



def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe(value: Any) -> str:
    return base._safe_name(value)


def _annotation_box(record: dict[str, Any], width: int, height: int) -> list[float]:
    x, y, w, h = [float(value) for value in record["bbox"]]
    return base._clip_box_xyxy([x, y, x + w, y + h], width, height)


def collect_o3_groups(
    o3_root: Path,
    images_per_layer: int,
    split_order: list[str],
) -> list[ImageGroup]:
    o3_root = base._require_dataset_root(o3_root, "O3")
    if images_per_layer <= 0:
        raise ValueError("images_per_layer must be positive")
    if not split_order or any(not split for split in split_order):
        raise ValueError("At least one non-empty O3 split is required")

    groups: list[ImageGroup] = []
    dataset_dirs = sorted(
        path for path in o3_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.endswith("_coco")
    )
    for dataset_dir in dataset_dirs:
        layer = dataset_dir.name.removesuffix("_coco")
        picked = 0
        for split in split_order:
            annotation_path = dataset_dir / "annotations" / f"instances_{split}.json"
            if not annotation_path.exists():
                continue
            categories, images, annotations = base._load_coco(annotation_path)
            annotations_by_image: dict[int, list[dict[str, Any]]] = {}
            for annotation in annotations:
                annotations_by_image.setdefault(
                    int(annotation["image_id"]), []
                ).append(annotation)
            annotation_sha256 = eval_utils.file_sha256(annotation_path)
            annotation_relpath = annotation_path.relative_to(o3_root).as_posix()
            for image_id in sorted(images):
                if picked >= images_per_layer:
                    break
                annotations = annotations_by_image.get(image_id, [])
                if not annotations:
                    continue
                image_record = images[image_id]
                width = int(image_record["width"])
                height = int(image_record["height"])
                image_path = base._resolve_coco_image(
                    dataset_dir,
                    image_record,
                    annotation_path,
                    split=split,
                )
                image_sha256 = eval_utils.file_sha256(image_path)
                picked += 1
                group_id = (
                    f"o3_{safe(layer)}_{safe(split)}_{safe(image_id)}_"
                    f"{safe(Path(str(image_record['file_name'])).stem)[:70]}"
                )
                items: list[Any] = []
                for annotation in annotations:
                    category_label = categories[int(annotation["category_id"])]
                    annotation_id = int(annotation["id"])
                    items.append(
                        base.EvalSample(
                            sample_id=f"{group_id}__ann{annotation_id}",
                            source="O3",
                            dataset=dataset_dir.name,
                            split=split,
                            image_path=str(image_path),
                            image_id=str(image_id),
                            annotation_id=str(annotation_id),
                            label=category_label,
                            category_label=category_label,
                            bbox_xyxy=_annotation_box(annotation, width, height),
                            annotation_relpath=annotation_relpath,
                            annotation_sha256=annotation_sha256,
                            image_sha256=image_sha256,
                        )
                    )
                groups.append(
                    ImageGroup(
                        group_id=group_id,
                        source="O3",
                        layer=layer,
                        dataset=dataset_dir.name,
                        split=split,
                        image_path=str(image_path),
                        image_id=str(image_id),
                        annotation_relpath=annotation_relpath,
                        annotation_sha256=annotation_sha256,
                        image_sha256=image_sha256,
                        category_label=layer,
                        category_labels=sorted({item.category_label for item in items}),
                        items=items,
                    )
                )
            if picked >= images_per_layer:
                break
    return groups


def collect_t4_groups(t4_root: Path, scope: str) -> list[ImageGroup]:
    if scope == "none":
        return []
    if scope != "all":
        raise ValueError(f"Unsupported T4 scope: {scope}")
    t4_root = base._require_dataset_root(t4_root, "T4")

    groups: list[ImageGroup] = []
    annotation_paths = sorted(t4_root.glob("*/annotations/instances_all.json"))
    for annotation_path in annotation_paths:
        layer_dir = annotation_path.parent.parent
        layer = layer_dir.name
        categories, images, annotations = base._load_coco(annotation_path)
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            annotations_by_image.setdefault(
                int(annotation["image_id"]), []
            ).append(annotation)
        annotation_sha256 = eval_utils.file_sha256(annotation_path)
        annotation_relpath = annotation_path.relative_to(t4_root).as_posix()
        dataset = f"{t4_root.name}/{layer}"
        for image_id in sorted(images):
            annotations = annotations_by_image.get(image_id, [])
            if not annotations:
                continue
            image_record = images[image_id]
            width = int(image_record["width"])
            height = int(image_record["height"])
            image_path = base._resolve_coco_image(
                layer_dir,
                image_record,
                annotation_path,
                split=None,
            )
            image_sha256 = eval_utils.file_sha256(image_path)
            group_id = (
                f"t4_{safe(layer)}_{safe(image_id)}_"
                f"{safe(Path(str(image_record['file_name'])).stem)[:70]}"
            )
            items: list[Any] = []
            for annotation in annotations:
                category_label = categories[int(annotation["category_id"])]
                label = base._annotation_label(
                    annotation, category_label, annotation_path
                )
                annotation_id = int(annotation["id"])
                items.append(
                    base.EvalSample(
                        sample_id=f"{group_id}__ann{annotation_id}",
                        source="T4",
                        dataset=dataset,
                        split="original_size",
                        image_path=str(image_path),
                        image_id=str(image_id),
                        annotation_id=str(annotation_id),
                        label=label,
                        category_label=category_label,
                        bbox_xyxy=_annotation_box(annotation, width, height),
                        annotation_relpath=annotation_relpath,
                        annotation_sha256=annotation_sha256,
                        image_sha256=image_sha256,
                    )
                )
            groups.append(
                ImageGroup(
                    group_id=group_id,
                    source="T4",
                    layer=layer,
                    dataset=dataset,
                    split="original_size",
                    image_path=str(image_path),
                    image_id=str(image_id),
                    annotation_relpath=annotation_relpath,
                    annotation_sha256=annotation_sha256,
                    image_sha256=image_sha256,
                    category_label=layer,
                    category_labels=sorted({item.category_label for item in items}),
                    items=items,
                )
            )
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


def review_row(
    group: ImageGroup,
    item: Any,
    side: str = "",
    overlay: str = "",
    mask: str = "",
    pred_box: Any = "",
    score: Any = "",
) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "sample_id": item.sample_id,
        "source": item.source,
        "layer": group.layer,
        "dataset": item.dataset,
        "split": item.split,
        "image_path": item.image_path,
        "image_id": item.image_id,
        "image_sha256": item.image_sha256,
        "annotation_relpath": item.annotation_relpath,
        "annotation_sha256": item.annotation_sha256,
        "annotation_id": item.annotation_id,
        "category_label": item.category_label,
        "label": item.label,
        "output_label": item.label,
        "input_bbox_xyxy": json.dumps(item.bbox_xyxy),
        "pred_bbox_xyxy": json.dumps(pred_box) if pred_box else "",
        "score": f"{float(score):.6f}" if score != "" else "",
        "side_by_side": side,
        "pvs_overlay": overlay,
        "mask": mask,
        "success": "",
        "review_comment": "",
    }


def write_review(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "group_id",
        "sample_id",
        "source",
        "layer",
        "dataset",
        "split",
        "image_path",
        "image_id",
        "image_sha256",
        "annotation_relpath",
        "annotation_sha256",
        "annotation_id",
        "category_label",
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
    path = out_dir / "review_sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
        stream.flush()
        os.fsync(stream.fileno())
    base.summarize_review_sheet(path, out_dir / "class_summary.csv")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def run_groups(
    groups: list[ImageGroup],
    out_dir: Path,
    device: str,
    dry_run: bool,
    max_items_per_group: int,
) -> None:
    if not groups:
        raise ValueError("At least one image group is required")
    if max_items_per_group < 0:
        raise ValueError("max_items_per_group must be non-negative")
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Image group IDs must be unique")

    out_dir = eval_utils.prepare_empty_output_dir(out_dir)
    (out_dir / "groups").mkdir(parents=True, exist_ok=False)
    selected_groups: list[dict[str, Any]] = []
    selected_items: list[list[Any]] = []
    dry_rows: list[dict[str, Any]] = []
    for group in groups:
        items = (
            group.items[:max_items_per_group]
            if max_items_per_group > 0
            else list(group.items)
        )
        if not items:
            raise ValueError(f"Image group has no selected items: {group.group_id}")
        sample_ids = [item.sample_id for item in items]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample IDs in image group: {group.group_id}")
        selected_items.append(items)
        selected_groups.append(
            {**asdict(group), "items": [asdict(item) for item in items]}
        )
        dry_rows.extend(review_row(group, item) for item in items)
    eval_utils.write_json_atomic(out_dir / "selected_groups.json", selected_groups)
    if dry_run:
        write_review(dry_rows, out_dir)
        return

    predictor = base.init_image_predictor(device)
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for group_index, (group, items) in enumerate(
        zip(groups, selected_items), start=1
    ):
        image_path = Path(group.image_path)
        if eval_utils.file_sha256(image_path) != group.image_sha256:
            raise ValueError(f"Source image changed before inference: {image_path}")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        print(
            f"[{group_index}/{len(groups)}] set_image "
            f"{group.group_id} items={len(items)}"
        )
        state = predictor.set_image(image)
        predictions: list[dict[str, Any]] = []
        for item in items:
            prediction = base.predict_from_box(predictor, state, item.bbox_xyxy)
            mask = np.asarray(prediction["mask"], dtype=bool)
            if mask.shape != (image.height, image.width):
                raise ValueError(
                    f"Prediction mask shape {mask.shape} does not match "
                    f"image {(image.height, image.width)}"
                )
            predictions.append(
                {
                    "item": item,
                    "mask": mask,
                    "score": float(prediction["score"]),
                    "pred_bbox_xyxy": base._mask_box(mask),
                    "best_index": int(prediction["best_index"]),
                    "candidate_scores": prediction["candidate_scores"],
                }
            )

        group_dir = out_dir / "groups" / group.group_id
        group_dir.mkdir(parents=True, exist_ok=False)
        group_view = replace(group, items=items)
        input_overlay = draw_group_input(image, group_view)
        pvs_overlay, union = draw_group_pvs(image, group_view, predictions)
        side = base.side_by_side(input_overlay, pvs_overlay)
        input_path = group_dir / "input_overlay.png"
        pvs_path = group_dir / "pvs_overlay.png"
        side_path = group_dir / "side_by_side.png"
        prediction_path = group_dir / "prediction.json"
        input_overlay.save(input_path)
        pvs_overlay.save(pvs_path)
        side.save(side_path)
        combined_mask_artifact = eval_utils.save_binary_mask(
            out_dir,
            f"groups/{group.group_id}/combined_mask.png",
            union,
        )

        prediction_items: list[dict[str, Any]] = []
        for item_index, prediction in enumerate(predictions, start=1):
            item = prediction["item"]
            mask_artifact = eval_utils.save_binary_mask(
                out_dir,
                (
                    f"groups/{group.group_id}/masks/"
                    f"{item_index:04d}_{safe(item.sample_id)}.png"
                ),
                prediction["mask"],
            )
            rows.append(
                review_row(
                    group,
                    item,
                    base.relative(side_path, out_dir),
                    base.relative(pvs_path, out_dir),
                    str(mask_artifact["path"]),
                    prediction["pred_bbox_xyxy"],
                    prediction["score"],
                )
            )
            prediction_items.append(
                {
                    **asdict(item),
                    "output_label": item.label,
                    "score": prediction["score"],
                    "pred_bbox_xyxy": prediction["pred_bbox_xyxy"],
                    "best_index": prediction["best_index"],
                    "candidate_scores": prediction["candidate_scores"],
                    "mask_artifact": mask_artifact,
                }
            )
        payload = {
            **asdict(group),
            "items": prediction_items,
            "input_overlay": base.relative(input_path, out_dir),
            "pvs_overlay": base.relative(pvs_path, out_dir),
            "side_by_side": base.relative(side_path, out_dir),
            "combined_mask": combined_mask_artifact["path"],
            "combined_mask_artifact": combined_mask_artifact,
        }
        eval_utils.write_json_atomic(prediction_path, payload)
        prediction_records.append(payload)

    _write_jsonl_atomic(out_dir / "predictions.jsonl", prediction_records)
    write_review(rows, out_dir)
    source_group_counts = {
        source: sum(group.source == source for group in groups)
        for source in sorted({group.source for group in groups})
    }
    eval_utils.write_complete_run_manifest(
        out_dir,
        run_kind="pvs_bbox_grouped_eval",
        selected_manifest_path="selected_groups.json",
        expected_selected_ids=group_ids,
        expected_prediction_ids=group_ids,
        metadata={
            "source_group_counts": source_group_counts,
            "group_count": len(groups),
            "item_count": sum(len(items) for items in selected_items),
            "max_items_per_group": max_items_per_group,
        },
    )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped whole-image PVS bbox visualization for O3/T4 labels.")
    parser.add_argument("--o3-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco"))
    parser.add_argument("--t4-root", type=Path, default=Path("/data/zhengqiyuan/ADC_contour/datasets/T4/original_size"))
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


def _require_requested_sources(
    requested_sources: set[str],
    groups: list[ImageGroup],
) -> None:
    counts = {
        source: sum(group.source == source for group in groups)
        for source in requested_sources
    }
    missing = sorted(source for source, count in counts.items() if count == 0)
    if missing:
        raise ValueError(
            "Requested evaluation source produced zero groups: "
            + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    if args.summary_only:
        out = args.summary_out or (args.summary_only.parent / "class_summary.csv")
        base.summarize_review_sheet(args.summary_only, out)
        print(f"Wrote summary: {out}")
        return
    if args.o3_images_per_layer <= 0:
        raise ValueError("--o3-images-per-layer must be positive")
    if args.max_groups < 0 or args.max_items_per_group < 0:
        raise ValueError("--max-groups and --max-items-per-group must be non-negative")
    if args.datasets == "t4" and args.t4_scope == "none":
        raise ValueError("--datasets t4 cannot be combined with --t4-scope none")

    out_dir = args.out_dir or (
        base.RUNTIME_DIR
        / "eval"
        / "pvs_bbox_grouped_eval"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    groups: list[ImageGroup] = []
    requested_sources: set[str] = set()
    if args.datasets in {"both", "o3"}:
        split_order = [
            split.strip()
            for split in args.o3_split_order.split(",")
            if split.strip()
        ]
        if len(split_order) != len(set(split_order)):
            raise ValueError("--o3-split-order must not contain duplicate splits")
        requested_sources.add("O3")
        groups.extend(
            collect_o3_groups(
                args.o3_root,
                int(args.o3_images_per_layer),
                split_order,
            )
        )
    if args.datasets in {"both", "t4"} and args.t4_scope != "none":
        requested_sources.add("T4")
        groups.extend(collect_t4_groups(args.t4_root, args.t4_scope))
    _require_requested_sources(requested_sources, groups)

    if args.max_groups > 0:
        groups = groups[: args.max_groups]
        _require_requested_sources(requested_sources, groups)
    if not groups:
        raise ValueError("No image groups found")
    total_items = sum(
        len(
            group.items[: args.max_items_per_group]
            if args.max_items_per_group > 0
            else group.items
        )
        for group in groups
    )
    print(f"Selected {len(groups)} image groups and {total_items} label bboxes")
    print(f"Output directory: {out_dir}")
    run_groups(
        groups,
        out_dir,
        args.device,
        bool(args.dry_run),
        int(args.max_items_per_group),
    )
    print(f"Done. Review sheet: {out_dir / 'review_sheet.csv'}")
    print(f"Class summary: {out_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()
