#!/usr/bin/env python3
"""Re-score persisted PCS/PVS masks against strict COCO ground truth.

This evaluator is intentionally model-free. It accepts only completed producer
runs with verified manifests and never scans output directories for extra files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import offline_eval_utils as eval_utils


DEFAULT_O3_ROOT = Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco")
DEFAULT_T4_ROOT = Path("/data/zhengqiyuan/ADC_contour/datasets/T4/original_size")


def _require_string(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _require_list(record: Mapping[str, Any], field: str, context: str) -> list[Any]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{field} must be a list")
    return value


def _safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ).strip("._")
    return result[:120] or "unknown"


def _validate_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("IoU threshold must be a finite number in [0, 1]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be a finite number in [0, 1]")
    return threshold


def mask_iou(left: Any, right: Any) -> float:
    left_mask = np.asarray(left)
    right_mask = np.asarray(right)
    if (
        left_mask.ndim != 2
        or right_mask.ndim != 2
        or left_mask.shape != right_mask.shape
    ):
        raise ValueError(
            f"Mask shapes must be equal non-empty 2D arrays, got "
            f"{left_mask.shape} and {right_mask.shape}"
        )
    if left_mask.size == 0:
        raise ValueError("Masks must not be empty arrays")
    left_mask = left_mask.astype(bool)
    right_mask = right_mask.astype(bool)
    intersection = np.logical_and(left_mask, right_mask).sum(dtype=np.float64)
    union = np.logical_or(left_mask, right_mask).sum(dtype=np.float64)
    return 0.0 if union == 0 else float(intersection / union)


def _assert_equal_fields(
    selected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    fields: tuple[str, ...],
    context: str,
) -> None:
    for field in fields:
        if field not in selected or field not in prediction:
            raise ValueError(f"{context} is missing required field {field!r}")
        if selected[field] != prediction[field]:
            raise ValueError(f"{context}.{field} changed between selection and prediction")


def _load_producer_run(
    run_dir: Path,
    expected_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    try:
        manifest, selected, predictions = eval_utils.read_complete_run_manifest(run_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"{run_dir} is not a valid completed {expected_kind} run. "
            "Legacy or incomplete artifacts must be regenerated with the current "
            f"producer: {exc}"
        ) from exc
    if manifest["run_kind"] != expected_kind:
        raise ValueError(
            f"Producer run kind {manifest['run_kind']!r} is not {expected_kind!r}"
        )
    return manifest, selected, predictions


def _ground_truth(
    item: Mapping[str, Any],
    *,
    o3_root: Path,
    t4_root: Path,
    cache: dict[tuple[Any, ...], np.ndarray],
) -> np.ndarray:
    context = f"sample {_require_string(item, 'sample_id', 'item')}"
    source = _require_string(item, "source", context)
    if source == "O3":
        root = Path(o3_root)
        expected_dataset = _require_string(item, "dataset", context)
        expected_split: str | None = _require_string(item, "split", context)
        expected_annotation_label = None
    elif source == "T4":
        root = Path(t4_root)
        dataset = PurePosixPath(_require_string(item, "dataset", context))
        if dataset.is_absolute() or not dataset.parts:
            raise ValueError(f"{context}.dataset is invalid")
        expected_dataset = dataset.name
        expected_split = None
        expected_annotation_label = _require_string(item, "label", context)
    else:
        raise ValueError(f"{context}.source must be O3 or T4")

    annotation_relpath = _require_string(item, "annotation_relpath", context)
    annotation_path = eval_utils.resolve_relative_artifact(root, annotation_relpath)
    image_path = Path(_require_string(item, "image_path", context))
    key = (
        source,
        str(annotation_path),
        _require_string(item, "annotation_id", context),
        expected_dataset,
        expected_split,
        _require_string(item, "image_id", context),
        str(image_path),
        _require_string(item, "annotation_sha256", context),
        _require_string(item, "image_sha256", context),
        _require_string(item, "category_label", context),
        expected_annotation_label,
    )
    if key not in cache:
        mask, _, _ = eval_utils.load_validated_coco_ground_truth(
            annotation_path,
            key[2],
            expected_dataset=expected_dataset,
            expected_split=expected_split,
            expected_image_id=key[5],
            image_path=image_path,
            expected_annotation_sha256=key[7],
            expected_image_sha256=key[8],
            expected_category_label=key[9],
            expected_annotation_label=key[10],
        )
        cache[key] = mask
    return cache[key]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty evaluation records: {path}")
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_pvs_summary(rows: list[dict[str, Any]], path: Path) -> None:
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"total": 0, "success": 0, "iou_sum": 0.0}
    )
    for row in rows:
        keys = (
            ("ALL", "ALL"),
            (row["source"], "ALL"),
            (row["source"], row["layer"]),
            (row["source"], row["label"]),
        )
        for key in keys:
            group = groups[key]
            group["total"] += 1
            group["success"] += int(row["success"])
            group["iou_sum"] += float(row["mask_iou"])
    output = []
    for key, value in sorted(groups.items()):
        total = int(value["total"])
        output.append(
            {
                "group_a": key[0],
                "group_b": key[1],
                "total": total,
                "success": int(value["success"]),
                "success_rate": float(value["success"]) / total,
                "mean_iou": float(value["iou_sum"]) / total,
            }
        )
    _write_csv(
        path,
        ["group_a", "group_b", "total", "success", "success_rate", "mean_iou"],
        output,
    )

def evaluate_pvs(
    run_dir: Path,
    selected_groups: list[dict[str, Any]],
    prediction_groups: list[dict[str, Any]],
    out_dir: Path,
    *,
    o3_root: Path,
    t4_root: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    selected_ids = [
        _require_string(group, "group_id", "selected PVS group")
        for group in selected_groups
    ]
    prediction_ids = [
        _require_string(group, "group_id", "PVS prediction group")
        for group in prediction_groups
    ]
    if selected_ids != prediction_ids:
        raise ValueError("PVS selected and prediction group identities differ")

    output_dir = out_dir / "pvs"
    output_dir.mkdir(parents=True, exist_ok=False)
    ground_truth_cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    group_fields = (
        "group_id",
        "source",
        "layer",
        "dataset",
        "split",
        "image_path",
        "image_id",
        "annotation_relpath",
        "annotation_sha256",
        "image_sha256",
        "category_label",
        "category_labels",
    )
    item_fields = (
        "sample_id",
        "source",
        "dataset",
        "split",
        "image_path",
        "image_id",
        "annotation_id",
        "label",
        "bbox_xyxy",
        "category_label",
        "annotation_relpath",
        "annotation_sha256",
        "image_sha256",
        "label_shape_type",
    )
    shared_fields = (
        "source",
        "dataset",
        "split",
        "image_path",
        "image_id",
        "annotation_relpath",
        "annotation_sha256",
        "image_sha256",
    )

    for group_index, (selected_group, prediction_group) in enumerate(
        zip(selected_groups, prediction_groups), start=1
    ):
        group_id = selected_ids[group_index - 1]
        context = f"PVS group {group_id}"
        _assert_equal_fields(selected_group, prediction_group, group_fields, context)
        selected_items = _require_list(selected_group, "items", context)
        prediction_items = _require_list(prediction_group, "items", context)
        if not selected_items:
            raise ValueError(f"{context} has no selected items")
        selected_sample_ids = [
            _require_string(
                _require_mapping(item, f"{context}.items[{index}]"),
                "sample_id",
                f"{context}.items[{index}]",
            )
            for index, item in enumerate(selected_items)
        ]
        prediction_sample_ids = [
            _require_string(
                _require_mapping(item, f"{context}.prediction_items[{index}]"),
                "sample_id",
                f"{context}.prediction_items[{index}]",
            )
            for index, item in enumerate(prediction_items)
        ]
        if selected_sample_ids != prediction_sample_ids:
            raise ValueError(f"{context} selected and prediction item identities differ")

        for item_index, (selected_value, prediction_value) in enumerate(
            zip(selected_items, prediction_items), start=1
        ):
            selected_item = _require_mapping(
                selected_value, f"{context}.items[{item_index - 1}]"
            )
            prediction_item = _require_mapping(
                prediction_value, f"{context}.prediction_items[{item_index - 1}]"
            )
            sample_id = selected_sample_ids[item_index - 1]
            item_context = f"PVS sample {sample_id}"
            _assert_equal_fields(
                selected_item, prediction_item, item_fields, item_context
            )
            for field in shared_fields:
                if selected_item.get(field) != selected_group.get(field):
                    raise ValueError(
                        f"{item_context}.{field} is inconsistent with its group"
                    )

            artifact = _require_mapping(
                prediction_item.get("mask_artifact"),
                f"{item_context}.mask_artifact",
            )
            predicted_mask = eval_utils.load_binary_mask(run_dir, artifact)
            ground_truth = _ground_truth(
                selected_item,
                o3_root=o3_root,
                t4_root=t4_root,
                cache=ground_truth_cache,
            )
            value = mask_iou(ground_truth, predicted_mask)
            score = prediction_item.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError(f"{item_context}.score must be finite")

            base_path = (
                f"pvs/groups/{group_index:04d}_{_safe_name(group_id)}/"
                f"{item_index:04d}_{_safe_name(sample_id)}"
            )
            gt_artifact = eval_utils.save_binary_mask(
                out_dir, f"{base_path}_gt.png", ground_truth
            )
            pred_artifact = eval_utils.save_binary_mask(
                out_dir, f"{base_path}_pred.png", predicted_mask
            )
            rows.append(
                {
                    "source": selected_group["source"],
                    "layer": selected_group["layer"],
                    "dataset": selected_group["dataset"],
                    "label": selected_item["label"],
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "mask_iou": value,
                    "success": int(value >= threshold),
                    "score": float(score),
                    "gt_mask_artifact": gt_artifact,
                    "pred_mask_artifact": pred_artifact,
                }
            )

    if not rows:
        raise ValueError("Completed PVS run contains no evaluation items")
    _write_jsonl(output_dir / "pvs_mask_iou95_predictions.jsonl", rows)
    _write_csv(
        output_dir / "pvs_mask_iou95_items.csv",
        [
            "source",
            "layer",
            "dataset",
            "label",
            "group_id",
            "sample_id",
            "mask_iou",
            "success",
            "score",
            "gt_mask",
            "pred_mask",
        ],
        [
            {
                **row,
                "gt_mask": row["gt_mask_artifact"]["path"],
                "pred_mask": row["pred_mask_artifact"]["path"],
            }
            for row in rows
        ],
    )
    _write_pvs_summary(rows, output_dir / "pvs_mask_iou95_summary.csv")
    return rows


def _write_pcs_summary(rows: list[dict[str, Any]], path: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "runs": 0,
            "gt": 0,
            "pred": 0,
            "matched": 0,
            "strict": 0,
        }
    )
    for row in rows:
        prompt_count = str(row["prompt_count"])
        keys = (
            ("ALL", "ALL", "ALL"),
            (row["layer"], "ALL", "ALL"),
            (row["layer"], row["label"], "ALL"),
            (row["layer"], row["label"], prompt_count),
            ("ALL", "ALL", prompt_count),
        )
        for key in keys:
            group = groups[key]
            group["runs"] += 1
            group["gt"] += int(row["gt_count"])
            group["pred"] += int(row["pred_count"])
            group["matched"] += int(row["matched_count"])
            group["strict"] += int(row["strict_all_gt_success"])
    output = []
    for key, value in sorted(groups.items()):
        runs = int(value["runs"])
        gt_count = int(value["gt"])
        pred_count = int(value["pred"])
        output.append(
            {
                "layer": key[0],
                "label": key[1],
                "prompt_count": key[2],
                "runs": runs,
                "gt": gt_count,
                "pred": pred_count,
                "matched": int(value["matched"]),
                "recall": float(value["matched"]) / gt_count if gt_count else 0.0,
                "precision": (
                    float(value["matched"]) / pred_count if pred_count else 0.0
                ),
                "strict_all_gt_rate": float(value["strict"]) / runs,
            }
        )
    _write_csv(
        path,
        [
            "layer",
            "label",
            "prompt_count",
            "runs",
            "gt",
            "pred",
            "matched",
            "recall",
            "precision",
            "strict_all_gt_rate",
        ],
        output,
    )


def evaluate_pcs(
    run_dir: Path,
    selected_groups: list[dict[str, Any]],
    prediction_runs: list[dict[str, Any]],
    out_dir: Path,
    *,
    o3_root: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    selected_by_id = {
        _require_string(group, "group_id", "selected PCS group"): group
        for group in selected_groups
    }
    if len(selected_by_id) != len(selected_groups):
        raise ValueError("PCS selected group identities are not unique")

    output_dir = out_dir / "pcs"
    output_dir.mkdir(parents=True, exist_ok=False)
    ground_truth_cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    group_fields = (
        "group_id",
        "layer",
        "dataset",
        "split",
        "label",
        "image_path",
        "image_id",
        "image_file_name",
        "category_id",
        "boxes",
        "annotation_ids",
        "annotation_relpath",
        "annotation_sha256",
        "image_sha256",
    )

    for run_index, prediction in enumerate(prediction_runs, start=1):
        run_id = _require_string(prediction, "run_id", "PCS prediction")
        group_id = _require_string(prediction, "group_id", f"PCS run {run_id}")
        if group_id not in selected_by_id:
            raise ValueError(f"PCS run {run_id} references unknown group {group_id}")
        selected = selected_by_id[group_id]
        seen_group_ids.add(group_id)
        context = f"PCS run {run_id}"
        _assert_equal_fields(selected, prediction, group_fields, context)
        if prediction.get("source") != "O3":
            raise ValueError(f"{context}.source must be O3")

        annotation_ids = _require_list(selected, "annotation_ids", context)
        boxes = _require_list(selected, "boxes", context)
        if not annotation_ids or len(annotation_ids) != len(boxes):
            raise ValueError(f"{context} annotation_ids and boxes must be non-empty and aligned")
        if len({str(value) for value in annotation_ids}) != len(annotation_ids):
            raise ValueError(f"{context} annotation_ids must be unique")

        prompt_count = prediction.get("prompt_count")
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count <= 0
            or prompt_count > len(boxes)
        ):
            raise ValueError(f"{context}.prompt_count is invalid")
        if run_id != f"{group_id}_p{prompt_count}":
            raise ValueError(f"{context}.run_id is inconsistent with prompt_count")
        if prediction.get("prompt_boxes_xyxy") != boxes[:prompt_count]:
            raise ValueError(f"{context}.prompt_boxes_xyxy changed after selection")
        _validate_threshold(prediction.get("confidence"))

        instances = _require_list(prediction, "pred_instances", context)
        pred_count = prediction.get("pred_count")
        if (
            isinstance(pred_count, bool)
            or not isinstance(pred_count, int)
            or pred_count < 0
            or pred_count != len(instances)
        ):
            raise ValueError(f"{context}.pred_count does not match pred_instances")
        pred_boxes = _require_list(prediction, "pred_boxes_xyxy", context)
        scores = _require_list(prediction, "scores", context)
        if len(pred_boxes) != pred_count or len(scores) != pred_count:
            raise ValueError(f"{context} prediction arrays have inconsistent lengths")

        predicted_masks: list[np.ndarray] = []
        for instance_index, value in enumerate(instances):
            instance = _require_mapping(
                value, f"{context}.pred_instances[{instance_index}]"
            )
            if instance.get("instance_index") != instance_index:
                raise ValueError(f"{context} has a non-sequential instance_index")
            if instance.get("box_xyxy") != pred_boxes[instance_index]:
                raise ValueError(f"{context} instance box differs from pred_boxes_xyxy")
            score = instance.get("score")
            recorded_score = scores[instance_index]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or isinstance(recorded_score, bool)
                or not isinstance(recorded_score, (int, float))
                or not math.isfinite(float(recorded_score))
                or float(score) != float(recorded_score)
            ):
                raise ValueError(f"{context} instance score is invalid or inconsistent")
            artifact = _require_mapping(
                instance.get("mask_artifact"),
                f"{context}.pred_instances[{instance_index}].mask_artifact",
            )
            predicted_masks.append(eval_utils.load_binary_mask(run_dir, artifact))

        gt_masks: list[np.ndarray] = []
        label = _require_string(selected, "label", context)
        for annotation_index, annotation_id in enumerate(annotation_ids):
            item = {
                "sample_id": f"{run_id}__gt{annotation_index}",
                "source": "O3",
                "dataset": selected["dataset"],
                "split": selected["split"],
                "image_path": selected["image_path"],
                "image_id": selected["image_id"],
                "annotation_id": str(annotation_id),
                "label": label,
                "category_label": label,
                "annotation_relpath": selected["annotation_relpath"],
                "annotation_sha256": selected["annotation_sha256"],
                "image_sha256": selected["image_sha256"],
            }
            gt_masks.append(
                _ground_truth(
                    item,
                    o3_root=o3_root,
                    t4_root=Path("."),
                    cache=ground_truth_cache,
                )
            )

        iou_matrix = np.zeros(
            (len(gt_masks), len(predicted_masks)), dtype=np.float64
        )
        for gt_index, ground_truth in enumerate(gt_masks):
            for pred_index, predicted_mask in enumerate(predicted_masks):
                iou_matrix[gt_index, pred_index] = mask_iou(
                    ground_truth, predicted_mask
                )
        matches = eval_utils.thresholded_hungarian_matches(iou_matrix, threshold)

        run_base = f"pcs/runs/{run_index:04d}_{_safe_name(run_id)}"
        gt_artifacts = [
            eval_utils.save_binary_mask(
                out_dir, f"{run_base}/gt_{index:04d}.png", mask
            )
            for index, mask in enumerate(gt_masks)
        ]
        pred_artifacts = [
            eval_utils.save_binary_mask(
                out_dir, f"{run_base}/pred_{index:04d}.png", mask
            )
            for index, mask in enumerate(predicted_masks)
        ]
        match_records = [
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": value,
            }
            for gt_index, pred_index, value in matches
        ]
        eval_utils.write_json_atomic(
            out_dir / run_base / "mask_iou_matches.json",
            {
                "run_id": run_id,
                "threshold": threshold,
                "matches": match_records,
                "gt_mask_artifacts": gt_artifacts,
                "pred_mask_artifacts": pred_artifacts,
            },
        )
        matched_count = len(matches)
        gt_count = len(gt_masks)
        mean_iou = (
            sum(value for _, _, value in matches) / matched_count
            if matched_count
            else 0.0
        )
        rows.append(
            {
                "layer": selected["layer"],
                "dataset": selected["dataset"],
                "label": label,
                "run_id": run_id,
                "prompt_count": prompt_count,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_count": matched_count,
                "recall": matched_count / gt_count,
                "precision": matched_count / pred_count if pred_count else 0.0,
                "strict_all_gt_success": int(matched_count == gt_count),
                "mean_matched_iou": mean_iou,
            }
        )

    if seen_group_ids != set(selected_by_id):
        missing = sorted(set(selected_by_id) - seen_group_ids)
        raise ValueError(f"PCS selected groups have no prediction runs: {missing}")

    if not rows:
        raise ValueError("Completed PCS run contains no prediction records")
    _write_jsonl(output_dir / "pcs_mask_iou95_runs.jsonl", rows)
    fields = [
        "layer",
        "dataset",
        "label",
        "run_id",
        "prompt_count",
        "gt_count",
        "pred_count",
        "matched_count",
        "recall",
        "precision",
        "strict_all_gt_success",
        "mean_matched_iou",
    ]
    _write_csv(output_dir / "pcs_mask_iou95_runs.csv", fields, rows)
    _write_pcs_summary(rows, output_dir / "pcs_mask_iou95_summary.csv")
    return rows


def _file_descriptor(path: Path, out_dir: Path, row_count: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": path.relative_to(out_dir).as_posix(),
        "sha256": eval_utils.file_sha256(path),
    }
    if row_count is not None:
        descriptor["row_count"] = int(row_count)
    return descriptor


def _input_descriptor(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_dir": str(Path(run_dir).resolve(strict=True)),
        "run_kind": manifest["run_kind"],
        "run_manifest_sha256": eval_utils.file_sha256(
            Path(run_dir) / eval_utils.RUN_MANIFEST_NAME
        ),
        "selected_manifest_sha256": manifest["selected_manifest"]["sha256"],
        "predictions_jsonl_sha256": manifest["predictions_jsonl"]["sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvs-dir", type=Path, default=None)
    parser.add_argument("--pcs-dir", type=Path, default=None)
    parser.add_argument("--o3-root", type=Path, default=DEFAULT_O3_ROOT)
    parser.add_argument("--t4-root", type=Path, default=DEFAULT_T4_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="auto",
        help="Deprecated compatibility option; no model inference is performed.",
    )
    parser.add_argument("--only", choices=["all", "pvs", "pcs"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = _validate_threshold(args.threshold)
    required_pvs = args.only in {"all", "pvs"}
    required_pcs = args.only in {"all", "pcs"}
    if required_pvs and args.pvs_dir is None:
        raise ValueError("--pvs-dir is required for --only all or pvs")
    if required_pcs and args.pcs_dir is None:
        raise ValueError("--pcs-dir is required for --only all or pcs")

    loaded_pvs = (
        _load_producer_run(args.pvs_dir, "pvs_bbox_grouped_eval")
        if required_pvs
        else None
    )
    loaded_pcs = (
        _load_producer_run(args.pcs_dir, "pcs_o3_grouped_eval")
        if required_pcs
        else None
    )
    out_dir = args.out_dir or (
        REPO_ROOT
        / ".runtime"
        / "eval"
        / f"mask_iou95_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    eval_utils.prepare_empty_output_dir(out_dir)

    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    pvs_rows: list[dict[str, Any]] = []
    pcs_rows: list[dict[str, Any]] = []
    if loaded_pvs is not None:
        pvs_manifest, pvs_selected, pvs_predictions = loaded_pvs
        pvs_rows = evaluate_pvs(
            args.pvs_dir,
            pvs_selected,
            pvs_predictions,
            out_dir,
            o3_root=args.o3_root,
            t4_root=args.t4_root,
            threshold=threshold,
        )
        inputs["pvs"] = _input_descriptor(args.pvs_dir, pvs_manifest)
        outputs["pvs_items"] = _file_descriptor(
            out_dir / "pvs" / "pvs_mask_iou95_predictions.jsonl",
            out_dir,
            len(pvs_rows),
        )
        outputs["pvs_summary"] = _file_descriptor(
            out_dir / "pvs" / "pvs_mask_iou95_summary.csv", out_dir
        )
    if loaded_pcs is not None:
        pcs_manifest, pcs_selected, pcs_predictions = loaded_pcs
        pcs_rows = evaluate_pcs(
            args.pcs_dir,
            pcs_selected,
            pcs_predictions,
            out_dir,
            o3_root=args.o3_root,
            threshold=threshold,
        )
        inputs["pcs"] = _input_descriptor(args.pcs_dir, pcs_manifest)
        outputs["pcs_runs"] = _file_descriptor(
            out_dir / "pcs" / "pcs_mask_iou95_runs.jsonl",
            out_dir,
            len(pcs_rows),
        )
        outputs["pcs_summary"] = _file_descriptor(
            out_dir / "pcs" / "pcs_mask_iou95_summary.csv", out_dir
        )

    evaluation_manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "mode": args.only,
        "model_inference_performed": False,
        "deprecated_device_argument": str(args.device),
        "inputs": inputs,
        "outputs": outputs,
    }
    eval_utils.write_json_atomic(
        out_dir / "evaluation_manifest.json", evaluation_manifest
    )
    print(f"OUTPUT_DIR {out_dir}")
    if pvs_rows:
        successes = sum(int(row["success"]) for row in pvs_rows)
        print(f"PVS {successes}/{len(pvs_rows)} success at IoU >= {threshold}")
    if pcs_rows:
        gt_count = sum(int(row["gt_count"]) for row in pcs_rows)
        matched = sum(int(row["matched_count"]) for row in pcs_rows)
        pred_count = sum(int(row["pred_count"]) for row in pcs_rows)
        precision = matched / pred_count if pred_count else 0.0
        print(
            f"PCS matched/gt {matched}/{gt_count} "
            f"recall={matched / gt_count if gt_count else 0.0:.6f} "
            f"precision={precision:.6f}"
        )


if __name__ == "__main__":
    main()
