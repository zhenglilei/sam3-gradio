"""Pure segmentation geometry, evaluation, and export helpers."""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import public_download_utils as _public_downloads


from sam3_demo.config import (
    _PUBLIC_DOWNLOAD_TTL_SECONDS,
    coco_dataset_configs,
    coco_eval_scope_overlap,
    default_coco_dataset,
    public_download_dir,
    runtime_export_dir,
)


def _publish_segmentation_zip(export_dir, zip_name):
    _public_downloads.prune_public_downloads(
        public_download_dir,
        max_age_seconds=_PUBLIC_DOWNLOAD_TTL_SECONDS,
    )
    return _public_downloads.publish_zip(
        public_download_dir,
        "pcs_pvs_exports",
        export_dir,
        zip_name,
    )


def parse_polygon_prompt(polygons_str):
    """Parse polygon JSON stored by the Gradio UI."""
    if not polygons_str:
        return []
    try:
        polygons = json.loads(polygons_str)
    except json.JSONDecodeError:
        return []

    parsed = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            continue
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                points.append([int(round(float(point[0]))), int(round(float(point[1])))])
            except (TypeError, ValueError):
                continue
        if len(points) >= 3:
            parsed.append(points)
    return parsed


def serialize_polygon_prompt(polygons):
    return json.dumps(polygons, ensure_ascii=False)


def append_polygon_prompt(polygons_str, polygon):
    polygons = parse_polygon_prompt(polygons_str)
    polygons.append(polygon)
    return serialize_polygon_prompt(polygons)


def polygon_to_mask(polygon, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], 1)
    return mask


def draw_polygons(vis_img, polygons, color):
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = vis_img.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.18, vis_img, 0.82, 0, dst=vis_img)
        cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=3)


def safe_stem(name):
    stem = Path(name or "sam3_export").stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)


def mask_to_polygons(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if len(contour) >= 3:
            polygons.append(contour.astype(float).reshape(-1).tolist())
    return polygons


def annotation_to_mask(annotation, source_width, source_height, target_width, target_height):
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    segmentation = annotation.get("segmentation")

    if isinstance(segmentation, list):
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        try:
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(segmentation).astype(np.uint8)
            if decoded.shape[:2] != (target_height, target_width):
                decoded = cv2.resize(
                    decoded,
                    (target_width, target_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask |= decoded
        except Exception:
            pass
    return mask.astype(bool)


def mask_bbox_xywh(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def create_prediction_coco_json(
    masks,
    scores,
    width,
    height,
    image_file_name="source_image",
    category_name="object",
    export_id="",
    annotation_extras=None,
):
    annotations = []
    if annotation_extras is None:
        annotation_extras = []
    if scores is None:
        scores = []
    for idx, mask in enumerate(masks):
        mask_bool = np.asarray(mask).astype(bool)
        annotation = {
            "id": idx + 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": encode_binary_mask(mask_bool),
            "area": int(mask_bool.sum()),
            "bbox": mask_bbox_xywh(mask_bool),
            "iscrowd": 1,
            "segmentation_format": "coco_rle",
        }
        if idx < len(scores):
            annotation["score"] = float(scores[idx])
        if idx < len(annotation_extras):
            annotation.update(annotation_extras[idx])
        annotations.append(annotation)

    return {
        "info": {
            "description": "SAM3 predicted instance masks exported in COCO format",
            "version": "1.0",
            "export_id": export_id,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": image_file_name or "source_image",
                "width": int(width),
                "height": int(height),
            }
        ],
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name or "object",
                "supercategory": "sam3_prediction",
            }
        ],
    }


def mask_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    if not mask_u8.any():
        return np.zeros_like(mask_u8, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 ^ eroded).astype(bool)


def boundary_band(mask, radius):
    boundary = mask_boundary(mask).astype(np.uint8)
    if not boundary.any():
        return boundary.astype(bool)
    kernel_size = max(1, int(radius) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(boundary, kernel, iterations=1).astype(bool)


def boundary_iou(pred_mask, gt_mask, dilation_ratio=0.02):
    diag = (pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2) ** 0.5
    radius = max(1, int(round(dilation_ratio * diag)))
    pred_boundary = boundary_band(pred_mask, radius)
    gt_boundary = boundary_band(gt_mask, radius)
    union = np.logical_or(pred_boundary, gt_boundary).sum()
    if union == 0:
        return 1.0 if pred_mask.sum() == gt_mask.sum() == 0 else 0.0
    return float(np.logical_and(pred_boundary, gt_boundary).sum() / union)


def boundary_distances_px(source_boundary, target_boundary):
    if not source_boundary.any() or not target_boundary.any():
        return np.array([], dtype=np.float32)
    target_inverse = (~target_boundary).astype(np.uint8)
    distance_map = cv2.distanceTransform(target_inverse, cv2.DIST_L2, 5)
    return distance_map[source_boundary].astype(np.float32)


def hd95_and_chamfer(pred_mask, gt_mask):
    pred_boundary = mask_boundary(pred_mask)
    gt_boundary = mask_boundary(gt_mask)
    pred_to_gt = boundary_distances_px(pred_boundary, gt_boundary)
    gt_to_pred = boundary_distances_px(gt_boundary, pred_boundary)
    if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
        return None, None
    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    hd95 = float(np.percentile(all_distances, 95))
    chamfer = float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)
    return hd95, chamfer


def ap_from_scores(scores, matches, num_gt):
    if num_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    tp = np.asarray(matches, dtype=np.float32)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(num_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    recall_changes = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[recall_changes + 1] - recalls[recall_changes]) * precisions[recall_changes + 1]))


def boundary_ap(pred_masks, gt_masks, scores, thresholds):
    if not pred_masks or not gt_masks:
        return {threshold: 0.0 for threshold in thresholds}
    pair_scores = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            pair_scores[pred_idx, gt_idx] = boundary_iou(pred_mask, gt_mask)

    ap_values = {}
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    for threshold in thresholds:
        used_gts = set()
        matches = np.zeros((len(pred_masks),), dtype=bool)
        for pred_idx in order:
            gt_idx = int(np.argmax(pair_scores[pred_idx]))
            best_score = float(pair_scores[pred_idx, gt_idx])
            if best_score >= threshold and gt_idx not in used_gts:
                matches[pred_idx] = True
                used_gts.add(gt_idx)
        ap_values[threshold] = ap_from_scores(scores, matches, len(gt_masks))
    return ap_values


def encode_binary_mask(mask):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def compute_coco_segm_metrics(pred_masks, gt_masks, scores, width, height):
    if not pred_masks or not gt_masks:
        return {
            "status": "empty",
            "metric": "segm",
            "ap_50_95_all": 0.0,
            "ap_50_all": 0.0,
            "ap_75_all": 0.0,
            "ar_50_95_all_max_dets_100": 0.0,
        }
    try:
        import contextlib
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        image_id = 1
        gt_payload = {
            "images": [{"id": image_id, "width": width, "height": height}],
            "categories": [{"id": 1, "name": "object"}],
            "annotations": [],
            "info": {},
            "licenses": [],
        }
        for idx, gt_mask in enumerate(gt_masks, start=1):
            gt_payload["annotations"].append(
                {
                    "id": idx,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": encode_binary_mask(gt_mask),
                    "bbox": mask_bbox_xywh(gt_mask),
                    "area": int(gt_mask.sum()),
                    "iscrowd": 0,
                }
            )

        detections = []
        for pred_mask, score in zip(pred_masks, scores):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": encode_binary_mask(pred_mask),
                    "bbox": mask_bbox_xywh(pred_mask),
                    "score": float(score),
                }
            )

        coco_gt = COCO()
        coco_gt.dataset = gt_payload
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.imgIds = [image_id]
        coco_eval.params.catIds = [1]
        coco_eval.params.maxDets = [1, 10, 100]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        def clean_stat(value):
            value = float(value)
            return 0.0 if value < 0 else value

        return {
            "status": "ok",
            "metric": "segm",
            "ap_50_95_all": clean_stat(coco_eval.stats[0]),
            "ap_50_all": clean_stat(coco_eval.stats[1]),
            "ap_75_all": clean_stat(coco_eval.stats[2]),
            "ar_50_95_all_max_dets_100": clean_stat(coco_eval.stats[8]),
        }
    except Exception as exc:
        return {"status": "error", "metric": "segm", "reason": str(exc)}


def get_coco_dataset_name(dataset_name):
    return dataset_name if dataset_name in coco_dataset_configs else default_coco_dataset


def get_coco_dataset_config(dataset_name):
    return coco_dataset_configs[get_coco_dataset_name(dataset_name)]


def load_coco_image_record(image_name, split, dataset_name):
    if not image_name:
        return None, None, None

    dataset_name = get_coco_dataset_name(dataset_name)
    dataset_config = get_coco_dataset_config(dataset_name)
    dataset_dir = dataset_config["path"]
    candidate_splits = ["val", "train", "test"] if split == "auto" else [split]
    for split_name in candidate_splits:
        ann_path = dataset_dir / "annotations" / f"instances_{split_name}.json"
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        image_record = next(
            (img for img in data.get("images", []) if img.get("file_name") == image_name),
            None,
        )
        if image_record is not None:
            return data, image_record, str(ann_path)
    return None, None, None


def resolve_uploaded_json_path(uploaded_file):
    if not uploaded_file:
        return None
    if isinstance(uploaded_file, (list, tuple)):
        uploaded_file = uploaded_file[0] if uploaded_file else None
    if isinstance(uploaded_file, (str, os.PathLike)):
        raw_path = uploaded_file
    elif isinstance(uploaded_file, dict):
        raw_path = uploaded_file.get("path") or uploaded_file.get("name")
    else:
        raw_path = (
            getattr(uploaded_file, "path", None)
            or getattr(uploaded_file, "name", None)
        )
    if not raw_path:
        return None
    json_path = Path(raw_path)
    return json_path if json_path.exists() else None


def label_text_variants(label):
    label_text = str(label or "").lower().strip()
    variants = {
        label_text,
        label_text.replace("-", " "),
        label_text.replace("_", " "),
    }
    if "-" in label_text:
        variants.add(label_text.split("-", 1)[0])
    return {variant for variant in variants if variant}


def infer_labelme_category_ids(text_prompt, categories):
    text = (text_prompt or "").lower().strip()
    if not text:
        return []
    matched = []
    for category in categories:
        if any(
            variant and (variant in text or text in variant)
            for variant in label_text_variants(category.get("name", ""))
        ):
            matched.append(category["id"])
    return matched


def labelme_shape_segmentation(shape):
    shape_type = shape.get("shape_type") or "polygon"
    points = shape.get("points") or []
    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif shape_type not in {"polygon", "linestrip"}:
        return None
    if shape_type == "linestrip" and len(points) >= 3 and points[0] != points[-1]:
        points = list(points) + [points[0]]

    segmentation = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        segmentation.extend([x, y])
    return segmentation if len(segmentation) >= 6 else None


def load_labelme_annotation_record(annotation_json_file, target_width, target_height):
    json_path = resolve_uploaded_json_path(annotation_json_file)
    if json_path is None:
        return None, "上传 JSON 文件不可读"

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"上传 JSON 解析失败: {exc}"

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return None, "上传 JSON 缺少 shapes 字段"

    source_width = int(data.get("imageWidth") or target_width)
    source_height = int(data.get("imageHeight") or target_height)
    warnings = []
    if (source_width, source_height) != (target_width, target_height):
        warnings.append(
            f"JSON image size {source_width}x{source_height} differs from current image {target_width}x{target_height}; shapes were scaled for evaluation"
        )
    image_record = {
        "id": 0,
        "file_name": data.get("imagePath") or json_path.name,
        "width": source_width,
        "height": source_height,
    }

    label_to_id = {}
    annotations = []
    gt_masks = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        shape_type = shape.get("shape_type") or "polygon"
        if shape_type == "linestrip":
            warnings.append(f"linestrip shape for label {shape.get('label') or 'object'} was auto-closed as a mask polygon")
        segmentation = labelme_shape_segmentation(shape)
        if segmentation is None:
            warnings.append(f"skipped invalid {shape_type} shape for label {shape.get('label') or 'object'}")
            continue
        label = str(shape.get("label") or "object")
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id) + 1
        annotation = {
            "id": len(annotations) + 1,
            "category_id": label_to_id[label],
            "category_name": label,
            "segmentation": [segmentation],
            "iscrowd": 0,
        }
        mask = annotation_to_mask(
            annotation,
            source_width,
            source_height,
            target_width,
            target_height,
        )
        if not mask.any():
            continue
        annotation["bbox"] = mask_bbox_xywh(mask)
        annotation["area"] = int(mask.sum())
        annotations.append(annotation)
        gt_masks.append(mask)

    categories = [
        {"id": category_id, "name": label}
        for label, category_id in label_to_id.items()
    ]
    return {
        "path": str(json_path),
        "data": data,
        "image_record": image_record,
        "categories": categories,
        "annotations": annotations,
        "gt_masks": gt_masks,
        "warnings": warnings,
    }, None


def infer_prompt_category_ids(text_prompt, categories, dataset_name):
    text = (text_prompt or "").lower()
    if not text:
        return []
    matched = []
    for category in categories:
        name = category.get("name", "")
        display_name = coco_category_display_name(name, dataset_name)
        variants = {
            name.lower(),
            display_name.lower(),
            name.lower().replace("ge1-", ""),
            display_name.lower().replace("ge1-", ""),
            name.lower().replace("-", " "),
            display_name.lower().replace("-", " "),
            name.lower().replace("_", " "),
            display_name.lower().replace("_", " "),
            name.lower().replace("ge1-", "").replace("-", " "),
            display_name.lower().replace("ge1-", "").replace("-", " "),
        }
        if any(variant and variant in text for variant in variants):
            matched.append(category["id"])
    return matched


def coco_category_display_name(category_name, dataset_name=default_coco_dataset):
    display_name = category_name
    for prefix in get_coco_dataset_config(dataset_name).get("strip_prefixes", []):
        if display_name.startswith(prefix):
            display_name = display_name[len(prefix) :]
    return display_name


def sorted_coco_categories(categories, dataset_name):
    display_order = get_coco_dataset_config(dataset_name).get("category_display_order") or []

    def sort_key(category):
        display_name = coco_category_display_name(category.get("name", ""), dataset_name)
        if display_name in display_order:
            return (0, display_order.index(display_name))
        return (1, int(category.get("id", 0)))

    return sorted(categories, key=sort_key)


def rasterize_coco_annotations(annotations, image_record, width, height):
    return [
        annotation_to_mask(
            ann,
            image_record["width"],
            image_record["height"],
            width,
            height,
        )
        for ann in annotations
    ]


def filter_gt_pairs_by_eval_scope(pred_masks, annotations, gt_masks, eval_scope):
    gt_pairs = [
        (ann, np.asarray(mask).astype(bool))
        for ann, mask in zip(annotations, gt_masks)
        if np.asarray(mask).any()
    ]
    if eval_scope == coco_eval_scope_overlap:
        gt_pairs = [
            (ann, mask)
            for ann, mask in gt_pairs
            if any(np.logical_and(pred_mask, mask).any() for pred_mask in pred_masks)
        ]
    return [ann for ann, _ in gt_pairs], [mask for _, mask in gt_pairs]


def select_predictions_for_gt_masks(pred_masks, pred_scores, gt_masks):
    if not gt_masks:
        return [], [], []
    selected_indices = [
        idx
        for idx, pred_mask in enumerate(pred_masks)
        if any(np.logical_and(pred_mask, gt_mask).any() for gt_mask in gt_masks)
    ]
    return (
        selected_indices,
        [pred_masks[idx] for idx in selected_indices],
        [pred_scores[idx] for idx in selected_indices],
    )


def evaluate_prediction_gt_metrics(pred_masks, pred_scores, annotations, gt_masks, width, height):
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou_matrix[pred_idx, gt_idx] = float(intersection / union) if union else 0.0

    original_iou_matrix = iou_matrix.copy()
    matched_pairs = []
    used_preds = set()
    used_gts = set()
    while iou_matrix.size:
        pred_idx, gt_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = float(iou_matrix[pred_idx, gt_idx])
        if best_iou <= 0:
            break
        if pred_idx in used_preds or gt_idx in used_gts:
            iou_matrix[pred_idx, gt_idx] = -1
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matched_pairs.append(
            {
                "prediction_index": int(pred_idx),
                "ground_truth_index": int(gt_idx),
                "annotation_id": int(annotations[gt_idx]["id"]),
                "category_id": int(annotations[gt_idx]["category_id"]),
                "iou": best_iou,
            }
        )
        iou_matrix[pred_idx, :] = -1
        iou_matrix[:, gt_idx] = -1

    boundary_iou_values = []
    hd95_values = []
    chamfer_values = []
    for pair in matched_pairs:
        pred_mask = pred_masks[pair["prediction_index"]]
        gt_mask = gt_masks[pair["ground_truth_index"]]
        b_iou = boundary_iou(pred_mask, gt_mask)
        hd95, chamfer = hd95_and_chamfer(pred_mask, gt_mask)
        boundary_iou_values.append(b_iou)
        pair["boundary_iou"] = b_iou
        if hd95 is not None:
            hd95_values.append(hd95)
            pair["hd95_px"] = hd95
        if chamfer is not None:
            chamfer_values.append(chamfer)
            pair["chamfer_px"] = chamfer

    boundary_thresholds = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]
    boundary_ap_values = boundary_ap(
        pred_masks,
        gt_masks,
        pred_scores,
        boundary_thresholds,
    )
    boundary_ap_50_95 = float(np.mean(list(boundary_ap_values.values()))) if boundary_ap_values else 0.0
    coco_segm = compute_coco_segm_metrics(pred_masks, gt_masks, pred_scores, width, height)

    matched_at_50 = [pair for pair in matched_pairs if pair["iou"] >= 0.5]
    precision_at_50 = len(matched_at_50) / len(pred_masks) if pred_masks else 0.0
    recall_at_50 = len(matched_at_50) / len(gt_masks) if gt_masks else 0.0
    f1_at_50 = (
        2 * precision_at_50 * recall_at_50 / (precision_at_50 + recall_at_50)
        if precision_at_50 + recall_at_50
        else 0.0
    )
    match_recall = len(matched_pairs) / len(gt_masks) if gt_masks else 0.0
    mean_boundary_iou = float(np.mean(boundary_iou_values)) if boundary_iou_values else 0.0
    mean_hd95_px = float(np.mean(hd95_values)) if hd95_values else 0.0
    mean_chamfer_px = float(np.mean(chamfer_values)) if chamfer_values else 0.0

    return {
        "num_predictions": len(pred_masks),
        "num_ground_truth": len(gt_masks),
        "gt_instances": len(gt_masks),
        "matched_instances": len(matched_pairs),
        "match_recall": match_recall,
        "mean_best_prediction_iou": float(
            np.max(original_iou_matrix, axis=1).mean()
        )
        if len(pred_masks) and len(gt_masks)
        else 0.0,
        "mean_boundary_iou": mean_boundary_iou,
        "boundary_ap50": float(boundary_ap_values.get(0.50, 0.0)),
        "boundary_ap75": float(boundary_ap_values.get(0.75, 0.0)),
        "boundary_ap50_95": boundary_ap_50_95,
        "mean_hd95_px": mean_hd95_px,
        "mean_chamfer_px": mean_chamfer_px,
        "boundary_ap_by_threshold": {
            f"{threshold:.2f}": float(ap_value)
            for threshold, ap_value in boundary_ap_values.items()
        },
        "coco_segm": coco_segm,
        "matched_pairs": matched_pairs,
        "precision_at_iou_0_50": precision_at_50,
        "recall_at_iou_0_50": recall_at_50,
        "f1_at_iou_0_50": f1_at_50,
    }


def compare_with_labelme_json(
    pred_masks,
    pred_scores,
    annotation_json_file,
    text_prompt,
    width,
    height,
    eval_scope,
):
    record, error = load_labelme_annotation_record(annotation_json_file, width, height)
    if record is None:
        return {"status": "not_found", "reason": error or "上传 JSON 文件不可读"}
    if not record["annotations"]:
        return {
            "status": "not_found",
            "reason": f"上传 JSON 中没有可用 polygon/rectangle 标注: {record['path']}",
            "annotation_format": "labelme",
            "annotation_file": record["path"],
        }

    pred_masks = [np.asarray(mask).astype(bool) for mask in pred_masks]
    if len(pred_scores) != len(pred_masks):
        pred_scores = [1.0] * len(pred_masks)
    else:
        pred_scores = [float(score) for score in pred_scores]

    categories = record["categories"]
    category_ids = infer_labelme_category_ids(text_prompt, categories)
    all_annotations = record["annotations"]
    all_gt_masks = record["gt_masks"]
    annotation_pairs = [
        (ann, mask)
        for ann, mask in zip(all_annotations, all_gt_masks)
        if not category_ids or ann.get("category_id") in category_ids
    ]
    annotations = [ann for ann, _ in annotation_pairs]
    gt_masks = [mask for _, mask in annotation_pairs]
    annotations, gt_masks = filter_gt_pairs_by_eval_scope(
        pred_masks,
        annotations,
        gt_masks,
        eval_scope,
    )

    result = {
        "status": "ok",
        "annotation_format": "labelme",
        "annotation_file": record["path"],
        "image_file_name": record["image_record"]["file_name"],
        "category_filter_ids": category_ids,
        "eval_scope": eval_scope,
        "warnings": record.get("warnings", []),
        **evaluate_prediction_gt_metrics(
            pred_masks,
            pred_scores,
            annotations,
            gt_masks,
            width,
            height,
        ),
    }

    per_category = {}
    for category in categories:
        label = category.get("name", "")
        category_pairs = [
            (ann, mask)
            for ann, mask in zip(all_annotations, all_gt_masks)
            if ann.get("category_id") == category.get("id")
        ]
        category_annotations = [ann for ann, _ in category_pairs]
        category_gt_masks = [mask for _, mask in category_pairs]
        category_annotations, category_gt_masks = filter_gt_pairs_by_eval_scope(
            pred_masks,
            category_annotations,
            category_gt_masks,
            eval_scope,
        )
        pred_indices, category_pred_masks, category_pred_scores = select_predictions_for_gt_masks(
            pred_masks,
            pred_scores,
            category_gt_masks,
        )
        category_metrics = evaluate_prediction_gt_metrics(
            category_pred_masks,
            category_pred_scores,
            category_annotations,
            category_gt_masks,
            width,
            height,
        )
        for pair in category_metrics["matched_pairs"]:
            pair["original_prediction_index"] = int(pred_indices[pair["prediction_index"]])

        category_result = {
            "category_id": int(category.get("id")),
            "category_name": label,
            "display_name": label,
            "prediction_indices": [int(idx) for idx in pred_indices],
            **category_metrics,
        }
        category_result["summary_line"] = (
            f"{label}: GT {category_result['gt_instances']}, "
            f"Match {category_result['matched_instances']}, "
            f"BIoU {category_result['mean_boundary_iou']:.3f}, "
            f"BAP50 {category_result['boundary_ap50']:.3f}, "
            f"segm AP50 {category_result['coco_segm'].get('ap_50_all', 0.0):.3f}"
        )
        per_category[label] = category_result

    result["per_category"] = per_category
    result["summary_lines"] = [
        f"Annotation JSON: {Path(record['path']).name}",
        *[f"Warning: {warning}" for warning in record.get("warnings", [])],
        f"- GT instances: {result['gt_instances']}",
        f"- Matched instances: {result['matched_instances']}",
        f"- Match recall: {result['match_recall']:.6f}",
        f"- Mean Boundary IoU: {result['mean_boundary_iou']:.6f}",
        f"- Boundary AP50: {result['boundary_ap50']:.6f}",
        f"- Boundary AP75: {result['boundary_ap75']:.6f}",
        f"- Boundary AP50-95: {result['boundary_ap50_95']:.6f}",
        f"- Mean HD95 px: {result['mean_hd95_px']:.6f}",
        f"- Mean Chamfer px: {result['mean_chamfer_px']:.6f}",
        f"IoU metric: {result['coco_segm'].get('metric', 'segm')}",
        f" AP 0.50:0.95 all = {result['coco_segm'].get('ap_50_95_all', 0.0):.3f}",
        f" AP 0.50 all      = {result['coco_segm'].get('ap_50_all', 0.0):.3f}",
        f" AP 0.75 all      = {result['coco_segm'].get('ap_75_all', 0.0):.3f}",
        (
            " AR 0.50:0.95 all maxDets=100 = "
            f"{result['coco_segm'].get('ar_50_95_all_max_dets_100', 0.0):.3f}"
        ),
        "Per-label metrics:",
    ]
    for category in categories:
        label = category.get("name", "")
        if label in per_category:
            result["summary_lines"].append(f" {per_category[label]['summary_line']}")
    return result


def compare_with_coco(
    pred_masks,
    pred_scores,
    dataset_name,
    image_name,
    split,
    text_prompt,
    width,
    height,
    eval_scope,
    annotation_json_file=None,
):
    if annotation_json_file:
        annotation_json_path = resolve_uploaded_json_path(annotation_json_file)
        if annotation_json_path is None:
            return {"status": "not_found", "reason": "上传 JSON 文件不可读"}
        return compare_with_labelme_json(
            pred_masks,
            pred_scores,
            str(annotation_json_path),
            text_prompt,
            width,
            height,
            eval_scope,
        )

    if not image_name:
        return {
            "status": "skipped",
            "reason": "未填写 COCO image file_name，导出包仅保存预测结果",
        }

    dataset_name = get_coco_dataset_name(dataset_name)
    dataset_config = get_coco_dataset_config(dataset_name)
    data, image_record, ann_path = load_coco_image_record(image_name, split, dataset_name)
    if image_record is None:
        return {
            "status": "not_found",
            "reason": f"未在 {dataset_name} annotations 中找到 {image_name}",
            "dataset": dataset_name,
            "dataset_dir": str(dataset_config["path"]),
            "requested_split": split,
        }

    pred_masks = [np.asarray(mask).astype(bool) for mask in pred_masks]
    if len(pred_scores) != len(pred_masks):
        pred_scores = [1.0] * len(pred_masks)
    else:
        pred_scores = [float(score) for score in pred_scores]

    categories = data.get("categories", [])
    category_ids = infer_prompt_category_ids(text_prompt, categories, dataset_name)
    all_annotations = [
        ann for ann in data.get("annotations", []) if ann.get("image_id") == image_record["id"]
    ]
    annotations = [
        ann
        for ann in all_annotations
        if not category_ids or ann.get("category_id") in category_ids
    ]
    gt_masks = rasterize_coco_annotations(annotations, image_record, width, height)
    annotations, gt_masks = filter_gt_pairs_by_eval_scope(
        pred_masks,
        annotations,
        gt_masks,
        eval_scope,
    )

    result = {
        "status": "ok",
        "dataset": dataset_name,
        "dataset_dir": str(dataset_config["path"]),
        "annotation_file": ann_path,
        "image_id": image_record["id"],
        "image_file_name": image_record["file_name"],
        "category_filter_ids": category_ids,
        "eval_scope": eval_scope,
        "warnings": [],
        **evaluate_prediction_gt_metrics(
            pred_masks,
            pred_scores,
            annotations,
            gt_masks,
            width,
            height,
        ),
    }

    ordered_categories = sorted_coco_categories(categories, dataset_name)
    per_category = {}
    for category in ordered_categories:
        display_name = coco_category_display_name(category.get("name", ""), dataset_name)
        category_annotations = [
            ann for ann in all_annotations if ann.get("category_id") == category.get("id")
        ]
        category_gt_masks = rasterize_coco_annotations(
            category_annotations,
            image_record,
            width,
            height,
        )
        category_annotations, category_gt_masks = filter_gt_pairs_by_eval_scope(
            pred_masks,
            category_annotations,
            category_gt_masks,
            eval_scope,
        )
        pred_indices, category_pred_masks, category_pred_scores = select_predictions_for_gt_masks(
            pred_masks,
            pred_scores,
            category_gt_masks,
        )
        category_metrics = evaluate_prediction_gt_metrics(
            category_pred_masks,
            category_pred_scores,
            category_annotations,
            category_gt_masks,
            width,
            height,
        )
        for pair in category_metrics["matched_pairs"]:
            pair["original_prediction_index"] = int(pred_indices[pair["prediction_index"]])

        category_result = {
            "category_id": int(category.get("id")),
            "category_name": category.get("name", ""),
            "display_name": display_name,
            "prediction_indices": [int(idx) for idx in pred_indices],
            **category_metrics,
        }
        category_result["summary_line"] = (
            f"{display_name}: GT {category_result['gt_instances']}, "
            f"Match {category_result['matched_instances']}, "
            f"BIoU {category_result['mean_boundary_iou']:.3f}, "
            f"BAP50 {category_result['boundary_ap50']:.3f}, "
            f"segm AP50 {category_result['coco_segm'].get('ap_50_all', 0.0):.3f}"
        )
        per_category[display_name] = category_result

    result["per_category"] = per_category

    result["summary_lines"] = [
        f"Dataset: {dataset_name}",
        f"- GT instances: {result['gt_instances']}",
        f"- Matched instances: {result['matched_instances']}",
        f"- Match recall: {result['match_recall']:.6f}",
        f"- Mean Boundary IoU: {result['mean_boundary_iou']:.6f}",
        f"- Boundary AP50: {result['boundary_ap50']:.6f}",
        f"- Boundary AP75: {result['boundary_ap75']:.6f}",
        f"- Boundary AP50-95: {result['boundary_ap50_95']:.6f}",
        f"- Mean HD95 px: {result['mean_hd95_px']:.6f}",
        f"- Mean Chamfer px: {result['mean_chamfer_px']:.6f}",
        f"IoU metric: {result['coco_segm'].get('metric', 'segm')}",
        f" AP 0.50:0.95 all = {result['coco_segm'].get('ap_50_95_all', 0.0):.3f}",
        f" AP 0.50 all      = {result['coco_segm'].get('ap_50_all', 0.0):.3f}",
        f" AP 0.75 all      = {result['coco_segm'].get('ap_75_all', 0.0):.3f}",
        (
            " AR 0.50:0.95 all maxDets=100 = "
            f"{result['coco_segm'].get('ar_50_95_all_max_dets_100', 0.0):.3f}"
        ),
        "Per-label metrics:",
    ]
    for category in ordered_categories:
        label = coco_category_display_name(category.get("name", ""), dataset_name)
        if label in per_category:
            result["summary_lines"].append(f" {per_category[label]['summary_line']}")
    return result


def _create_segmentation_export_impl(
    result_image,
    source_image,
    state,
    prompts,
    coco_dataset,
    coco_image_name,
    coco_split,
    coco_eval_scope,
    annotation_json_file=None,
    *,
    export_root,
    compare_fn,
    publish_zip_fn,
):
    width, height = source_image.size
    masks = state["masks"].detach().cpu().numpy().astype(bool)
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().cpu().numpy()

    export_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    export_dir = export_root / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = export_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    result_path = export_dir / "segmentation_overlay.png"
    result_image.save(result_path)
    np.savez_compressed(export_dir / "masks.npz", masks=masks.astype(np.uint8))

    predictions = []
    coco_annotation_extras = []
    for idx, mask in enumerate(masks):
        mask_path = mask_dir / f"mask_{idx:03d}.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        x1, y1, x2, y2 = boxes[idx].tolist()
        mask_file = str(mask_path.relative_to(export_dir))
        predictions.append(
            {
                "id": idx,
                "score": float(scores[idx]),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": int(mask.sum()),
                "mask_file": mask_file,
                "segmentation": mask_to_polygons(mask),
            }
        )
        coco_annotation_extras.append(
            {
                "prediction_id": int(idx),
                "mask_file": mask_file,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    metrics = compare_fn(
        list(masks),
        scores.tolist(),
        coco_dataset,
        coco_image_name.strip() if coco_image_name else "",
        coco_split,
        prompts.get("text_prompt", ""),
        width,
        height,
        coco_eval_scope,
        annotation_json_file,
    )
    payload = {
        "export_id": export_id,
        "image": {
            "width": width,
            "height": height,
            "coco_dataset": coco_dataset,
            "coco_file_name": coco_image_name.strip() if coco_image_name else "",
            "coco_eval_scope": coco_eval_scope,
            "annotation_json_file": str(resolve_uploaded_json_path(annotation_json_file) or ""),
        },
        "prompts": prompts,
        "predictions": predictions,
        "coco_comparison": metrics,
    }

    with (export_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (export_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    coco_payload = create_prediction_coco_json(
        masks,
        scores.tolist(),
        width,
        height,
        image_file_name=coco_image_name.strip() if coco_image_name else "source_image",
        category_name="object",
        export_id=export_id,
        annotation_extras=coco_annotation_extras,
    )
    with (export_dir / "coco_masks.json").open("w", encoding="utf-8") as f:
        json.dump(coco_payload, f, ensure_ascii=False, indent=2)

    annotation_json_path = resolve_uploaded_json_path(annotation_json_file)
    zip_stem = coco_image_name or (annotation_json_path.stem if annotation_json_path else "")
    zip_path = publish_zip_fn(export_dir, f"{safe_stem(zip_stem)}_{export_id}.zip")
    return str(zip_path), metrics

def create_segmentation_export(
    result_image,
    source_image,
    state,
    prompts,
    coco_dataset,
    coco_image_name,
    coco_split,
    coco_eval_scope,
    annotation_json_file=None,
):
    return _create_segmentation_export_impl(
        result_image,
        source_image,
        state,
        prompts,
        coco_dataset,
        coco_image_name,
        coco_split,
        coco_eval_scope,
        annotation_json_file,
        export_root=runtime_export_dir,
        compare_fn=compare_with_coco,
        publish_zip_fn=_publish_segmentation_zip,
    )
