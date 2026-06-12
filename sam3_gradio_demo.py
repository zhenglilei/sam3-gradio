#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
基于 SAM3 的交互式图像分割与视频跟踪系统
"""

import os
import sys
import time
import io
import gc
import logging
from pathlib import Path
import tempfile
import json
import uuid
import zipfile

# 所有运行时文件固定在 /data/zhengqiyuan，避免 Gradio 默认写入 /tmp/gradio。
current_dir = Path(__file__).resolve().parent
runtime_dir = current_dir / ".runtime"
runtime_tmp_dir = runtime_dir / "tmp"
runtime_gradio_dir = runtime_dir / "gradio"
runtime_video_dir = runtime_dir / "videos"
runtime_export_dir = runtime_dir / "exports"
runtime_feedback_dir = runtime_dir / "feedback"
runtime_log_dir = runtime_dir / "logs"
qiyuan_cache_dir = Path("/data/zhengqiyuan/.cache")
ge1_coco_dir = Path("/data/zhengqiyuan/ADC_contour/datasets/GE1_coco")
o3_coco_dir = Path("/data/zhengqiyuan/ADC_contour/datasets/O3_coco")
coco_eval_scope_overlap = "只评估与预测相交的GT"
coco_eval_scope_full = "评估整图全部GT"
ge1_category_display_order = ["Block", "MainLine1", "MainLine2", "MainLine3"]
default_coco_dataset = "GE1_coco"
coco_dataset_configs = {
    "GE1_coco": {
        "path": ge1_coco_dir,
        "category_display_order": ge1_category_display_order,
        "strip_prefixes": ["GE1-"],
    },
    "O3_coco/GE1_coco": {"path": o3_coco_dir / "GE1_coco"},
    "O3_coco/GE2_coco": {"path": o3_coco_dir / "GE2_coco"},
    "O3_coco/ACT_coco": {"path": o3_coco_dir / "ACT_coco"},
    "O3_coco/BSM_coco": {"path": o3_coco_dir / "BSM_coco"},
}
coco_dataset_choices = list(coco_dataset_configs.keys())

for path in (
    runtime_tmp_dir,
    runtime_gradio_dir,
    runtime_video_dir,
    runtime_export_dir,
    runtime_feedback_dir,
    runtime_feedback_dir / "samples",
    runtime_log_dir,
    current_dir / ".gradio",
    qiyuan_cache_dir,
    qiyuan_cache_dir / "huggingface",
    qiyuan_cache_dir / "huggingface" / "hub",
    qiyuan_cache_dir / "modelscope",
):
    path.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(runtime_tmp_dir)
os.environ["TEMP"] = str(runtime_tmp_dir)
os.environ["TMP"] = str(runtime_tmp_dir)
os.environ["GRADIO_TEMP_DIR"] = str(runtime_gradio_dir)
os.environ["XDG_CACHE_HOME"] = str(qiyuan_cache_dir)
os.environ["HF_HOME"] = str(qiyuan_cache_dir / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(qiyuan_cache_dir / "huggingface" / "hub")
os.environ["MODELSCOPE_CACHE"] = str(qiyuan_cache_dir / "modelscope")
sys.path.insert(0, str(current_dir))

import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sam3_gradio_demo")

# 导入SAM3相关模块
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    from sam3.model.data_misc import FindStage
    from sam3.visualization_utils import (
        plot_results,
        visualize_formatted_frame_output,
        render_masklet_frame,
    )
    from sam3.model import box_ops
except ImportError as e:
    print(f"导入SAM3模块失败: {e}")
    print("请确保已正确安装SAM3依赖")
    sys.exit(1)

# 全局变量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")


# 初始化模型
def initialize_models():
    """初始化SAM3图像和视频预测器"""
    try:
        # 检查模型文件是否存在
        model_dir = current_dir / "models"
        checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if not checkpoint_path.exists():
            print(f"模型文件不存在: {checkpoint_path}")
            print("请下载SAM3模型文件到目录")
            return None, None

        if not bpe_path.exists():
            print(f"BPE文件不存在: {bpe_path}")
            return None, None

        # 初始化图像模型
        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE,
            enable_inst_interactivity=True,
        )

        # 创建图像处理器
        image_predictor = Sam3Processor(image_model, device=DEVICE)

        # 初始化视频预测器
        video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint_path), bpe_path=str(bpe_path)
        )

        print("模型初始化成功")
        return image_predictor, video_predictor

    except Exception as e:
        print(f"模型初始化失败: {e}")
        return None, None


# 全局预测器实例
image_predictor, video_predictor = initialize_models()


def _disable_legacy_predict_mask_prompt():
    if image_predictor is None or not hasattr(image_predictor, "predict_mask_prompt"):
        return

    def _deprecated_predict_mask_prompt(*args, **kwargs):
        raise RuntimeError(
            "predict_mask_prompt() is deprecated in this demo. "
            "Use _predict_inst(..., mask_input_lowres_logits=...) so multimask "
            "candidates and low-res logits are preserved."
        )

    image_predictor.predict_mask_prompt = _deprecated_predict_mask_prompt


_disable_legacy_predict_mask_prompt()


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
    width, height = source_image.size
    masks = state["masks"].detach().cpu().numpy().astype(bool)
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().cpu().numpy()

    export_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    export_dir = runtime_export_dir / export_id
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

    metrics = compare_with_coco(
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
    zip_path = runtime_export_dir / f"{safe_stem(zip_stem)}_{export_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in export_dir.rglob("*"):
            zf.write(file_path, arcname=file_path.relative_to(export_dir))
    return str(zip_path), metrics



# Legacy mixed point/box/polygon image segmentation flow removed.

def convert_output_format(outputs):
    """转换模型输出格式以适配可视化函数"""
    if not outputs:
        return {}

    # 简化版的转换逻辑，复用之前的核心逻辑
    if "out_binary_masks" in outputs:
        formatted_outputs = {
            "out_boxes_xywh": [],
            "out_probs": [],
            "out_obj_ids": [],
            "out_binary_masks": [],
        }

        masks = outputs["out_binary_masks"]
        if not isinstance(masks, (list, np.ndarray)):
            masks = [masks]
        formatted_outputs["out_binary_masks"] = list(masks)

        if "out_obj_ids" in outputs:
            formatted_outputs["out_obj_ids"] = list(outputs["out_obj_ids"])
        else:
            formatted_outputs["out_obj_ids"] = list(range(len(masks)))

        if "out_probs" in outputs:
            formatted_outputs["out_probs"] = list(outputs["out_probs"])
        else:
            formatted_outputs["out_probs"] = [1.0] * len(masks)

        if "out_boxes_xywh" in outputs:
            formatted_outputs["out_boxes_xywh"] = list(outputs["out_boxes_xywh"])
        else:
            # 计算边界框
            for mask in formatted_outputs["out_binary_masks"]:
                if isinstance(mask, np.ndarray) and mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        h, w = mask.shape
                        formatted_outputs["out_boxes_xywh"].append(
                            [
                                x_min / w,
                                y_min / h,
                                (x_max - x_min) / w,
                                (y_max - y_min) / h,
                            ]
                        )
                    else:
                        formatted_outputs["out_boxes_xywh"].append([0, 0, 0, 0])
                else:
                    formatted_outputs["out_boxes_xywh"].append([0, 0, 0, 0])
        return formatted_outputs

    # Fallback logic omitted for brevity as it mirrors previous implementation
    # ... (保持之前的辅助逻辑)
    # 这里为了节省空间，我们假设主要路径走通，如果需要完整fallback逻辑可以参考上一版代码
    # 但为了稳健性，这里保留基本的掩码处理
    elif "masks" in outputs:
        formatted_outputs = {
            "out_boxes_xywh": [],
            "out_probs": [],
            "out_obj_ids": [],
            "out_binary_masks": [],
        }
        masks = outputs["masks"]
        # Handle list or tensor
        if not isinstance(masks, list) and hasattr(masks, "shape"):
            if len(masks.shape) == 4:
                masks = [m[0] for m in masks.cpu().numpy()]
            elif len(masks.shape) == 3:
                masks = [m for m in masks.cpu().numpy()]

        for i, mask in enumerate(masks):
            if hasattr(mask, "shape") and len(mask.shape) > 2:
                mask = mask.squeeze()
            formatted_outputs["out_binary_masks"].append(mask)
            formatted_outputs["out_obj_ids"].append(i)
            formatted_outputs["out_probs"].append(1.0)
            # 简单box计算
            if isinstance(mask, np.ndarray) and mask.any():
                h, w = mask.shape
                y, x = np.where(mask)
                formatted_outputs["out_boxes_xywh"].append(
                    [
                        x.min() / w,
                        y.min() / h,
                        (x.max() - x.min()) / w,
                        (y.max() - y.min()) / h,
                    ]
                )
            else:
                formatted_outputs["out_boxes_xywh"].append([0, 0, 0, 0])
        return formatted_outputs

    return {}


def process_video(
    input_video, text_prompt, confidence_threshold, progress=gr.Progress()
):
    """视频处理功能"""
    if input_video is None:
        return None, "请上传视频"

    if not text_prompt:
        return None, "请提供文本提示"

    try:
        if video_predictor is None:
            return None, "模型未初始化，请检查模型文件"

        start_time = time.time()
        progress(0.1, desc="正在解析视频...")

        logger.info(
            json.dumps(
                {
                    "type": "video_tracking",
                    "device": DEVICE,
                    "confidence_threshold": confidence_threshold,
                    "text_prompt": text_prompt or "",
                },
                ensure_ascii=False,
            )
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fd, output_path = tempfile.mkstemp(suffix=".mp4", dir=runtime_video_dir)
            os.close(fd)

            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                return None, "无法打开视频文件"

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress(0.2, desc="初始化跟踪会话...")
            session_response = video_predictor.start_session(resource_path=input_video)
            session_id = session_response["session_id"]

            progress(0.3, desc="应用提示...")
            video_predictor.add_prompt(
                session_id=session_id, frame_idx=0, text=text_prompt
            )

            progress(0.4, desc="正在跟踪目标...")
            outputs_per_frame = {}
            for response in video_predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session_id}
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]

            for frame_idx in range(frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_idx in outputs_per_frame:
                    formatted_outputs = convert_output_format(
                        outputs_per_frame[frame_idx]
                    )
                    if formatted_outputs.get("out_binary_masks"):
                        vis_frame = render_masklet_frame(
                            img=frame_rgb,
                            outputs=formatted_outputs,
                            frame_idx=frame_idx,
                            alpha=0.5,
                        )
                    else:
                        vis_frame = frame_rgb
                else:
                    vis_frame = frame_rgb

                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                out.write(vis_frame_bgr)

                progress_value = 0.4 + 0.5 * (frame_idx / frame_count)
                progress(progress_value, desc=f"渲染帧 {frame_idx+1}/{frame_count}")

            cap.release()
            out.release()
            video_predictor.close_session(session_id)

            processing_time = time.time() - start_time
            info = f"✨ 处理完成 | 耗时: {processing_time:.2f}s | 总帧数: {frame_count}"

            return str(output_path), info

    except Exception as e:
        return None, f"❌ 处理失败: {str(e)}"


# --- PCS/PVS single-workspace override ---
import base64 as _sam3_base64
import threading as _sam3_threading

_PVS_PREDICT_LOCK = _sam3_threading.Lock()
_FEEDBACK_WRITE_LOCK = _sam3_threading.Lock()
_WORKSPACE_CACHE = {}


def _clear_workspace_cache():
    _WORKSPACE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pil_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


def _data_url(image):
    image = _pil_image(image)
    if image is None:
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + _sam3_base64.b64encode(buf.getvalue()).decode("ascii")


def _new_pcs_state():
    return {"text_prompt": "", "positive_boxes": [], "negative_boxes": [], "bbox_history": [], "instances": {}, "next_instance_id": 1}


def _new_pvs_state():
    return {"instances": {}, "active_instance_id": None, "next_instance_id": 1, "pending_boxes": []}


def _workspace(image_state):
    if not image_state or not image_state.get("image_id"):
        raise ValueError("Load an image first")
    ws = _WORKSPACE_CACHE.get(image_state["image_id"])
    if ws is None:
        raise ValueError("Image state expired; reload the image")
    return ws


def _fresh_state(image_state):
    base = _workspace(image_state)["base_state"]
    return {"original_height": base["original_height"], "original_width": base["original_width"], "backbone_out": dict(base["backbone_out"])}


def _norm_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((max(0.0, min(x1, width - 1)), max(0.0, min(x2, width - 1))))
    y1, y2 = sorted((max(0.0, min(y1, height - 1)), max(0.0, min(y2, height - 1))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def _bbox_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Draw a bbox first")
    data = json.loads(payload)
    box = data.get("box_xyxy_px")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox payload is missing box_xyxy_px")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    box = _norm_box(box, width, height)
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("bbox is too small")
    return box


def _xyxy_to_cxcywh_norm(box, width, height):
    x1, y1, x2, y2 = _norm_box(box, width, height)
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, max(1.0, x2 - x1) / width, max(1.0, y2 - y1) / height]


def _polygon_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Draw a positive polygon first")
    data = json.loads(payload)
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("polygon needs at least 3 points")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    parsed = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            parsed.append([max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))])
    if len(parsed) < 3:
        raise ValueError("polygon needs at least 3 valid points")
    return parsed


def _point_from_payload(payload, image_state):
    if not payload:
        raise ValueError("Click a positive point first")
    data = json.loads(payload)
    point = data.get("point_xy_px")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("point payload is missing point_xy_px")
    width = int(image_state.get("width") or data.get("image_width") or 0)
    height = int(image_state.get("height") or data.get("image_height") or 0)
    return [max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))]


def _prompt_mask_size():
    return tuple(int(v) for v in image_predictor.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)


def _polygon_lowres_logits(polygon, width, height):
    target_h, target_w = _prompt_mask_size()
    mask = polygon_to_mask(polygon, height, width).astype(np.float32)
    lowres = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return ((np.clip(lowres, 0.0, 1.0) * 2.0 - 1.0) * 10.0).astype(np.float32)


def _combine_logits(active_logits, polygon_logits, mode="replace", alpha=0.35, max_logit=10.0):
    polygon = np.asarray(polygon_logits, dtype=np.float32)
    if polygon.ndim == 3:
        polygon = polygon[0]
    if active_logits is None:
        return np.clip(polygon, -max_logit, max_logit).astype(np.float32)

    active = np.asarray(active_logits, dtype=np.float32)
    if active.ndim == 3:
        active = active[0]

    if mode == "replace":
        out = polygon
    elif mode == "blend":
        out = alpha * active + (1.0 - alpha) * polygon
    elif mode == "union":
        out = np.maximum(active, polygon)
    elif mode == "intersect":
        out = np.minimum(active, polygon)
    else:
        raise ValueError(f"Unknown polygon combine mode: {mode}")
    return np.clip(out, -max_logit, max_logit).astype(np.float32)


def _polygon_action_key(value):
    text = str(value or "")
    if text == "create" or "create" in text.lower() or "\u521b\u5efa" in text:
        return "create"
    return "refine"


def _polygon_combine_key(value):
    text = str(value or "replace")
    if text in {"replace", "blend", "union", "intersect"}:
        return text
    if "blend" in text.lower() or "\u878d\u5408" in text:
        return "blend"
    if "union" in text.lower() or "\u8865\u5145" in text:
        return "union"
    if "intersect" in text.lower() or "\u9650\u5236" in text:
        return "intersect"
    return "replace"


def _mask_box(mask):
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

def _predict_inst(base_state, box_xyxy_px=None, mask_input_lowres_logits=None, point_coords_px=None, point_labels=None):
    if image_predictor is None:
        raise RuntimeError("Image predictor is not initialized")
    kwargs = {"multimask_output": True, "return_logits": True}
    if box_xyxy_px is not None:
        kwargs["box"] = np.asarray(box_xyxy_px, dtype=np.float32)
    if point_coords_px is not None:
        coords = np.asarray(point_coords_px, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords[None, :]
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError("point_coords_px must have shape Nx2")
        labels = np.ones((coords.shape[0],), dtype=np.int64) if point_labels is None else np.asarray(point_labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != coords.shape[0]:
            raise ValueError("point_labels length must match point_coords_px")
        kwargs["point_coords"] = coords
        kwargs["point_labels"] = labels
    if mask_input_lowres_logits is not None:
        mask_input = np.asarray(mask_input_lowres_logits, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, :, :]
        expected = _prompt_mask_size()
        if mask_input.ndim != 3 or tuple(mask_input.shape[-2:]) != expected:
            raise ValueError(f"mask_input_lowres_logits must be 1x{expected[0]}x{expected[1]}")
        kwargs["mask_input"] = mask_input
    with _PVS_PREDICT_LOCK:
        masks, scores, lowres_logits = image_predictor.model.predict_inst(base_state, **kwargs)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return {
        "masks": masks > 0,
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
        "lowres_logits": np.asarray(lowres_logits, dtype=np.float32),
    }


def _pvs_progress(progress, value, desc, delay=0.08):
    if progress is None:
        return
    progress(float(value), desc=desc)
    if delay:
        time.sleep(delay)


def _best(pred):
    if len(pred["scores"]) == 0:
        raise ValueError("predict_inst returned no masks")
    return int(np.argmax(pred["scores"]))


def _make_inst(inst_id, source, mask, box, score, pvs_logits=None, pcs_prob=None, history=None):
    return {
        "id": int(inst_id),
        "source": source,
        "mask_fullres_bool": np.asarray(mask).astype(bool),
        "box_xyxy_px": [float(v) for v in box],
        "score": float(score),
        "pvs_lowres_logits": None if pvs_logits is None else np.asarray(pvs_logits, dtype=np.float32),
        "pcs_fullres_prob": None if pcs_prob is None else np.asarray(pcs_prob, dtype=np.float32),
        "status": "draft",
        "prompt_history": history or [],
    }


def _snapshot(inst):
    return {
        "mask_fullres_bool": np.asarray(inst["mask_fullres_bool"]).copy(),
        "pvs_lowres_logits": None if inst.get("pvs_lowres_logits") is None else np.asarray(inst["pvs_lowres_logits"]).copy(),
        "box_xyxy_px": list(inst.get("box_xyxy_px") or []),
        "score": float(inst.get("score", 0.0)),
        "status": inst.get("status", "draft"),
    }


def _restore(inst, snap):
    inst["mask_fullres_bool"] = np.asarray(snap["mask_fullres_bool"]).copy()
    inst["pvs_lowres_logits"] = None if snap.get("pvs_lowres_logits") is None else np.asarray(snap["pvs_lowres_logits"]).copy()
    inst["box_xyxy_px"] = list(snap.get("box_xyxy_px") or [])
    inst["score"] = float(snap.get("score", 0.0))
    inst["status"] = snap.get("status", inst.get("status", "draft"))


def _active_instances(state):
    return [inst for inst in state.get("instances", {}).values() if inst.get("status") != "deleted"]


def _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True):
    image = np.array(_workspace(image_state)["image"].convert("RGB"))
    overlay = image.copy()
    line = image.copy()
    box_draws = []
    label_draws = []
    polygon_draws = []

    def paint(mask, color, alpha):
        nonlocal overlay
        mask = np.asarray(mask).astype(bool)
        if mask.shape[:2] != overlay.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        c = np.array(color, dtype=np.uint8)
        overlay[mask] = (overlay[mask] * (1 - alpha) + c * alpha).astype(np.uint8)

    def queue_box(box, color, thickness=3):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        box_draws.append((x1, y1, x2, y2, color, thickness))

    def queue_label(text, x, y, color):
        label_draws.append((text, int(round(x)), int(round(y)), color))

    def queue_polygon(points, color):
        arr = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32).reshape((-1, 1, 2))
        polygon_draws.append((arr, color, len(points) >= 3))

    if show_instances and mode == "PCS Auto":
        for inst in _active_instances(pcs_state):
            color = (0, 255, 90)
            paint(inst["mask_fullres_bool"], color, 0.24)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 3)
            queue_label(f"PCS#{inst['id']}", x1, max(18, y1 - 6), color)
    if mode == "PVS Manual" and prompt_state is not None:
        for idx, box in enumerate(pvs_state.get("pending_boxes", []), start=1):
            color = (0, 255, 90)
            queue_box(box, color, 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            queue_label(f"pending#{idx}", x1, max(18, y1 - 6), color)
    if show_instances and mode == "PVS Manual":
        active_id = pvs_state.get("active_instance_id")
        for inst in _active_instances(pvs_state):
            is_active = str(inst["id"]) == str(active_id)
            color = (255, 0, 220) if is_active else (0, 185, 255)
            paint(inst["mask_fullres_bool"], color, 0.32 if is_active else 0.22)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 4 if is_active else 3)
            queue_label(f"PVS#{inst['id']}", x1, max(18, y1 - 6), color)
    if mode == "PCS Auto" and prompt_state is not None:
        for box in pcs_state.get("positive_boxes", []):
            queue_box(box, (0, 255, 90), 3)
        for box in pcs_state.get("negative_boxes", []):
            queue_box(box, (255, 48, 48), 3)
    if prompt_state:
        bbox_color = (255, 48, 48) if prompt_state.get("bbox_role") == "negative" else (0, 255, 90)
        if prompt_state.get("last_bbox"):
            queue_box(prompt_state["last_bbox"], bbox_color, 4)
        if prompt_state.get("bbox_start"):
            x, y = [int(round(v)) for v in prompt_state["bbox_start"]]
            cv2.circle(line, (x, y), 7, bbox_color, -1)
            cv2.putText(line, "bbox start", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bbox_color, 2)
        if prompt_state.get("last_point"):
            x, y = [int(round(v)) for v in prompt_state["last_point"]]
            cv2.circle(line, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(line, (x, y), 9, (255, 255, 255), 2)
        pts = prompt_state.get("polygon_points") or []
        if pts:
            queue_polygon(pts, (0, 255, 60))
    result = cv2.addWeighted(overlay, 0.72, line, 0.28, 0)
    for arr, color, is_closed in polygon_draws:
        if is_closed:
            filled = result.copy()
            cv2.fillPoly(filled, [arr], color)
            result = cv2.addWeighted(filled, 0.18, result, 0.82, 0)
        cv2.polylines(result, [arr], isClosed=is_closed, color=(0, 0, 0), thickness=5)
        cv2.polylines(result, [arr], isClosed=is_closed, color=color, thickness=3)
        for point in arr.reshape((-1, 2)):
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 6, (0, 0, 0), -1)
            cv2.circle(result, (x, y), 4, color, -1)
    for x1, y1, x2, y2, color, thickness in box_draws:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    for text, x, y, color in label_draws:
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return Image.fromarray(result)


def _instances_for_mode(pcs_state, pvs_state, mode):
    return _active_instances(pcs_state if mode == "PCS Auto" else pvs_state)


def _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None):
    if not image_state or not image_state.get("image_id"):
        return None
    return _overlay(image_state, pcs_state, pvs_state, mode, prompt_state, show_instances=False)


def _result_placeholder(image_state):
    if not image_state or not image_state.get("image_id"):
        return None
    width = max(1, int(image_state.get("width") or 1))
    height = max(1, int(image_state.get("height") or 1))
    return Image.new("RGB", (width, height), (248, 250, 252))


def _result_image(image_state, pcs_state, pvs_state, mode):
    if not image_state or not image_state.get("image_id"):
        return None
    if not _instances_for_mode(pcs_state, pvs_state, mode):
        return _result_placeholder(image_state)
    return _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True)

def _new_prompt_state():
    return {"bbox_start": None, "last_bbox": None, "last_point": None, "polygon_points": [], "bbox_role": "positive"}


def _event_point(evt, image_state):
    index = getattr(evt, "index", None)
    if isinstance(index, dict):
        point = index.get("point") or index.get("index") or index.get("value")
    else:
        point = index
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"Unsupported Gradio select event index: {index!r}")
    width = int(image_state.get("width") or 0)
    height = int(image_state.get("height") or 0)
    return [max(0.0, min(float(point[0]), width - 1)), max(0.0, min(float(point[1]), height - 1))]


def _payload_json(value):
    return json.dumps(value, ensure_ascii=False)


def _append_pcs_bbox_sample(pcs_state, box, bbox_role):
    key = "negative_boxes" if bbox_role == "negative" else "positive_boxes"
    pcs_state.setdefault(key, []).append(box)
    pcs_state.setdefault("bbox_history", []).append({"key": key, "box": box})
    pcs_state["instances"] = {}
    pcs_state["next_instance_id"] = 1
    return key


def _click_tool_key(click_tool):
    text = str(click_tool or "").strip()
    lower = text.lower()
    if lower in {"point", "bbox", "polygon"}:
        return lower
    if "point" in lower or "\u70b9" in text:
        return "point"
    if "bbox" in lower or "box" in lower or "\u6846" in text:
        return "bbox"
    if "polygon" in lower or "\u591a\u8fb9\u5f62" in text:
        return "polygon"
    return ""


def _workspace_select(image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, evt: gr.SelectData):
    prompt_state = prompt_state or _new_prompt_state()
    bbox_payload = gr.update()
    point_payload = gr.update()
    polygon_payload = gr.update()
    try:
        point = _event_point(evt, image_state)
        w, h = int(image_state.get("width") or 0), int(image_state.get("height") or 0)
        tool = _click_tool_key(click_tool)
        if mode == "PCS Auto" and tool != "bbox":
            tool = "bbox"
        if tool == "point":
            prompt_state["last_point"] = point
            point_payload = _payload_json({"type": "positive_point", "point_xy_px": point, "image_width": w, "image_height": h})
            info = f"\u5df2\u6dfb\u52a0\u6b63\u5411\u70b9: {[round(v, 1) for v in point]}"
        elif tool == "bbox":
            bbox_role = "negative" if mode == "PCS Auto" and str(pcs_bbox_kind or "").startswith("Negative") else "positive"
            prompt_state["bbox_role"] = bbox_role
            if prompt_state.get("bbox_start") is None:
                prompt_state["bbox_start"] = point
                prompt_state["last_bbox"] = None
                info = f"\u5df2\u8bb0\u5f55 bbox \u8d77\u70b9: {[round(v, 1) for v in point]}\u3002\u8bf7\u70b9\u51fb\u5bf9\u89d2\u70b9\u5b8c\u6210\u6846\u9009\u3002"
            else:
                start = prompt_state.get("bbox_start")
                box = _norm_box([start[0], start[1], point[0], point[1]], w, h)
                prompt_state["bbox_start"] = None
                prompt_state["last_bbox"] = box
                bbox_payload = _payload_json({"type": "bbox", "box_xyxy_px": box, "image_width": w, "image_height": h})
                if mode == "PCS Auto":
                    key = _append_pcs_bbox_sample(pcs_state, box, bbox_role)
                    prompt_state["last_bbox"] = None
                    label = "\u8d1f\u6837\u672c" if key == "negative_boxes" else "\u6b63\u6837\u672c"
                    info = f"\u5df2\u81ea\u52a8\u6dfb\u52a0 PCS {label} bbox: {[round(v, 1) for v in box]}"
                else:
                    pending = pvs_state.setdefault("pending_boxes", [])
                    pending.append(box)
                    prompt_state["last_bbox"] = None
                    info = f"\u5df2\u52a0\u5165 PVS \u5f85\u751f\u6210 bbox #{len(pending)}: {[round(v, 1) for v in box]}\u3002\u7ee7\u7eed\u6846\u9009\u6216\u70b9\u51fb\u201c\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b\u201d\u3002"
        elif tool == "polygon":
            points = prompt_state.setdefault("polygon_points", [])
            points.append(point)
            info = f"\u591a\u8fb9\u5f62\u5df2\u6dfb\u52a0\u7b2c {len(points)} \u4e2a\u9876\u70b9\u3002\u5b8c\u6210\u540e\u70b9\u51fb\u201c\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61\u201d\u3002"
        else:
            info = f"\u672a\u77e5\u4ea4\u4e92\u5de5\u5177: {click_tool}"
    except Exception as exc:
        info = f"\u56fe\u50cf\u70b9\u51fb\u5931\u8d25: {exc}"
    return prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _apply_polygon_to_pvs(image_state, pvs_state, polygon, polygon_action="refine", combine_mode="replace", progress=None):
    action = _polygon_action_key(polygon_action)
    combine = _polygon_combine_key(combine_mode)
    ws = _workspace(image_state)
    w, h = ws["image"].size
    _pvs_progress(progress, 0.18, "转换 polygon 为 PVS mask prompt")
    polygon_logits = _polygon_lowres_logits(polygon, w, h)

    if action == "create":
        _pvs_progress(progress, 0.42, "SAM3 正在根据 polygon 创建实例", delay=0.12)
        pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=polygon_logits)
        _pvs_progress(progress, 0.82, "整理 polygon 候选 mask")
        idx = _best(pred)
        mask = pred["masks"][idx]
        inst_id = int(pvs_state.get("next_instance_id", 1))
        pvs_state.setdefault("instances", {})[inst_id] = _make_inst(
            inst_id,
            "manual_pvs_polygon",
            mask,
            _mask_box(mask),
            pred["scores"][idx],
            pvs_logits=pred["lowres_logits"][idx],
            history=[{"op":"create_from_polygon","prompt":{"type":"positive_polygon","points":polygon},"candidate_scores":pred["scores"].astype(float).tolist()}],
        )
        pvs_state["active_instance_id"] = inst_id
        pvs_state["next_instance_id"] = inst_id + 1
        return f"\u5df2\u7528 polygon mask prompt \u521b\u5efa PVS #{inst_id}"

    active_id = pvs_state.get("active_instance_id")
    if active_id is None or int(active_id) not in pvs_state.get("instances", {}):
        raise ValueError("\u8bf7\u5148\u521b\u5efa\u6216\u9009\u62e9\u4e00\u4e2a PVS \u5b9e\u4f8b\uff0c\u6216\u5c06 polygon \u52a8\u4f5c\u6539\u4e3a\u201c\u521b\u5efa\u65b0\u5b9e\u4f8b\u201d")
    inst = pvs_state["instances"][int(active_id)]
    _pvs_progress(progress, 0.34, f"融合当前实例 logits: {combine}")
    combined = _combine_logits(inst.get("pvs_lowres_logits"), polygon_logits, mode=combine)
    before = _snapshot(inst)
    _pvs_progress(progress, 0.52, "SAM3 正在精修当前 PVS 实例", delay=0.12)
    pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=combined)
    _pvs_progress(progress, 0.84, "更新实例 mask 与 logits")
    idx = _best(pred)
    mask = pred["masks"][idx]
    inst["mask_fullres_bool"] = mask
    inst["box_xyxy_px"] = _mask_box(mask)
    inst["score"] = float(pred["scores"][idx])
    inst["pvs_lowres_logits"] = pred["lowres_logits"][idx]
    after = _snapshot(inst)
    inst.setdefault("prompt_history", []).append({"op":"positive_polygon_refine","mode":combine,"prompt":{"type":"positive_polygon","points":polygon},"before":before,"after":after,"candidate_scores":pred["scores"].astype(float).tolist()})
    return f"\u5df2\u7528\u591a\u8fb9\u5f62\u7cbe\u4fee PVS #{active_id}\uff0c\u878d\u5408\u65b9\u5f0f: {combine}"


def _finish_native_polygon(image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action="create", polygon_combine_mode="replace", progress=gr.Progress(track_tqdm=False)):
    prompt_state = prompt_state or _new_prompt_state()
    points = prompt_state.get("polygon_points") or []
    polygon_payload = gr.update()
    _pvs_progress(progress, 0.03, "准备 PVS 多边形操作")
    if len(points) < 3:
        info = "\u591a\u8fb9\u5f62\u81f3\u5c11\u9700\u8981 3 \u4e2a\u9876\u70b9"
        return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)

    w, h = int(image_state.get("width") or 0), int(image_state.get("height") or 0)
    polygon_payload = _payload_json({"type": "positive_polygon", "points": points, "image_width": w, "image_height": h})

    if mode != "PVS Manual":
        info = "\u591a\u8fb9\u5f62\u5df2\u5b8c\u6210\u3002PCS Auto \u4e0d\u4f7f\u7528 polygon prompt\u3002"
        return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)

    try:
        info = _apply_polygon_to_pvs(image_state, pvs_state, points, polygon_action, polygon_combine_mode, progress)
        prompt_state["polygon_points"] = []
        _pvs_progress(progress, 0.96, "渲染 PVS 分割结果", delay=0.16)
    except Exception as exc:
        info = f"PVS \u591a\u8fb9\u5f62\u5904\u7406\u5931\u8d25: {exc}"
    return prompt_state, polygon_payload, pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _clear_prompt_selection(image_state, pcs_state, pvs_state, mode):
    prompt_state = _new_prompt_state()
    if mode == "PCS Auto":
        pcs_state = _new_pcs_state()
        info = "PCS \u63d0\u793a\u5df2\u6e05\u7a7a"
    else:
        info = "\u63d0\u793a\u5df2\u6e05\u7a7a"
    return prompt_state, "", "", "", pcs_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


def _pcs_choice_update(pcs_state):
    choices = [(f"PCS #{i['id']} score={i['score']:.3f}", str(i["id"])) for i in _active_instances(pcs_state)]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _status_label(status):
    return {"draft": "草稿", "accepted": "已确认", "deleted": "已删除"}.get(str(status or "draft"), str(status or "草稿"))


def _pvs_choice_update(pvs_state):
    choices = [(f"PVS #{i['id']} {_status_label(i.get('status'))} score={i['score']:.3f}", str(i["id"])) for i in _active_instances(pvs_state)]
    active = pvs_state.get("active_instance_id")
    value = str(active) if active is not None and any(c[1] == str(active) for c in choices) else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def _pvs_pending_count_text(pvs_state):
    return f"待生成 bbox 数量: {len(pvs_state.get('pending_boxes', []))}"


def _pcs_summary(pcs_state):
    lines = [f"\u6b63\u6837\u672c bbox: {len(pcs_state.get('positive_boxes', []))}", f"\u8d1f\u6837\u672c bbox: {len(pcs_state.get('negative_boxes', []))}"]
    items = _active_instances(pcs_state)
    lines.append(f"PCS \u5b9e\u4f8b: {len(items)}")
    for inst in items[:80]:
        lines.append(f"#{inst['id']} score={inst['score']:.3f} box={[round(v,1) for v in inst['box_xyxy_px']]}")
    return "\n".join(lines)


def _pvs_summary(pvs_state):
    items = _active_instances(pvs_state)
    active = pvs_state.get("active_instance_id")
    pending = pvs_state.get("pending_boxes", [])
    lines = [
        f"PVS 实例: {len(items)}",
        f"待生成 bbox: {len(pending)}",
        f"当前实例: {active or '-'}",
        "说明: 草稿=draft，表示还未点击确认；score 是 SAM3 返回的候选 mask 质量/置信估计，不等同于人工质检分数。",
    ]
    for idx, box in enumerate(pending[:20], start=1):
        lines.append(f"pending#{idx} box={[round(v,1) for v in box]}")
    for inst in items[:80]:
        mark = "*" if str(inst["id"]) == str(active) else " "
        lines.append(f"{mark}#{inst['id']} {inst['source']} {_status_label(inst.get('status'))} score={inst['score']:.3f}")
    return "\n".join(lines)

def _analysis_report(pcs_state, pvs_state, mode, info):
    sections = [str(info or "")]
    if mode == "PCS Auto":
        sections.extend(["", "PCS Auto \u81ea\u52a8\u6982\u5ff5\u5206\u5272", _pcs_summary(pcs_state)])
    else:
        sections.extend(["", "PVS Manual \u624b\u52a8\u5b9e\u4f8b\u5206\u5272", _pvs_summary(pvs_state)])
    return "\n".join(part for part in sections if part is not None)


def _view(image_state, pcs_state, pvs_state, mode, info, prompt_state=None):
    return (
        _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state),
        _result_image(image_state, pcs_state, pvs_state, mode),
        _analysis_report(pcs_state, pvs_state, mode, info),
        _pcs_summary(pcs_state),
        _pvs_summary(pvs_state),
        _pvs_choice_update(pvs_state),
        info,
        _pvs_pending_count_text(pvs_state),
    )


def _init_workspace(input_image, mode):
    pcs_state, pvs_state = _new_pcs_state(), _new_pvs_state()
    prompt_state = _new_prompt_state()
    image_state = {"image_id": None, "width": 0, "height": 0}
    if input_image is None:
        return image_state, pcs_state, pvs_state, prompt_state, *_view(image_state, pcs_state, pvs_state, mode, "Upload an image first", prompt_state), None
    if image_predictor is None:
        return image_state, pcs_state, pvs_state, prompt_state, *_view(image_state, pcs_state, pvs_state, mode, "SAM3 image predictor is not initialized", prompt_state), None
    image = _pil_image(input_image)
    image_id = uuid.uuid4().hex
    _clear_workspace_cache()
    try:
        base_state = image_predictor.set_image(image)
    except torch.OutOfMemoryError:
        _clear_workspace_cache()
        info = "\u56fe\u50cf\u52a0\u8f7d\u5931\u8d25\uff1aGPU \u663e\u5b58\u4e0d\u8db3\u3002\u5df2\u6e05\u7406\u5f53\u524d\u5de5\u4f5c\u53f0\u7f13\u5b58\uff0c\u8bf7\u5173\u95ed\u5176\u4ed6 GPU \u4efb\u52a1\u6216\u91cd\u542f demo \u540e\u91cd\u8bd5\u3002"
        return image_state, pcs_state, pvs_state, prompt_state, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state), None
    _WORKSPACE_CACHE[image_id] = {"image": image, "base_state": base_state}
    image_state = {"image_id": image_id, "width": image.width, "height": image.height}
    return image_state, pcs_state, pvs_state, prompt_state, *_view(image_state, pcs_state, pvs_state, mode, f"Image loaded: {image.width}x{image.height}", prompt_state), None


def _undo_pcs_bbox(image_state, pcs_state, pvs_state, mode):
    try:
        history = pcs_state.setdefault("bbox_history", [])
        if not history:
            raise ValueError("\u6ca1\u6709\u53ef\u64a4\u9500\u7684 PCS bbox \u6837\u672c")
        item = history.pop()
        key = item.get("key")
        boxes = pcs_state.setdefault(key, [])
        if boxes:
            boxes.pop()
        pcs_state["instances"] = {}
        pcs_state["next_instance_id"] = 1
        label = "\u8d1f\u6837\u672c" if key == "negative_boxes" else "\u6b63\u6837\u672c"
        info = f"\u5df2\u64a4\u9500\u6700\u8fd1\u4e00\u4e2a PCS {label} bbox"
    except Exception as exc:
        info = f"PCS bbox \u64a4\u9500\u5931\u8d25: {exc}"
    return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _run_pcs(image_state, pcs_state, pvs_state, mode, text_prompt, threshold):
    try:
        ws = _workspace(image_state)
        w, h = ws["image"].size
        text_prompt = (text_prompt or "").strip()
        has_positive = bool(pcs_state.get("positive_boxes"))
        has_negative = bool(pcs_state.get("negative_boxes"))
        if not text_prompt and not has_positive and not has_negative:
            raise ValueError("PCS needs a text prompt or bbox exemplar")
        if has_negative and not text_prompt and not has_positive:
            raise ValueError("PCS \u4e0d\u652f\u6301\u53ea\u4f7f\u7528\u8d1f\u6837\u672c bbox\uff0c\u8bf7\u5148\u6dfb\u52a0\u6587\u672c\u63d0\u793a\u6216\u6b63\u6837\u672c bbox")
        state = _fresh_state(image_state)
        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)
        for box in pcs_state.get("positive_boxes", []):
            state = image_predictor.add_geometric_prompt(_xyxy_to_cxcywh_norm(box, w, h), True, state)
        for box in pcs_state.get("negative_boxes", []):
            state = image_predictor.add_geometric_prompt(_xyxy_to_cxcywh_norm(box, w, h), False, state)
        state = image_predictor.set_confidence_threshold(float(threshold), state)
        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            pcs_state["instances"] = {}
            return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, "PCS found no instances")
        masks_np = masks.detach().cpu().numpy().astype(bool)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        probs = state.get("masks_logits")
        probs_np = None if probs is None else probs.detach().cpu().numpy().astype(np.float32)
        if probs_np is not None and probs_np.ndim == 4:
            probs_np = probs_np[:, 0]
        boxes_np = state["boxes"].detach().cpu().numpy()
        scores_np = state["scores"].detach().cpu().numpy()
        instances = {}
        for idx, mask in enumerate(masks_np):
            inst_id = idx + 1
            instances[inst_id] = _make_inst(inst_id, "pcs", mask, _norm_box(boxes_np[idx].tolist(), w, h), float(scores_np[idx]), pcs_prob=None if probs_np is None else probs_np[idx], history=[{"op":"pcs_grounding","text_prompt":text_prompt or ""}])
        pcs_state["instances"] = instances
        pcs_state["next_instance_id"] = len(instances) + 1
        pcs_state["text_prompt"] = text_prompt or ""
        info = f"PCS found {len(instances)} instances"
    except Exception as exc:
        info = f"PCS failed: {exc}"
    return pcs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _create_pvs_from_pending_boxes(image_state, pcs_state, pvs_state, mode, progress=gr.Progress(track_tqdm=False)):
    try:
        _pvs_progress(progress, 0.03, "准备批量生成 PVS 实例")
        boxes = list(pvs_state.get("pending_boxes", []))
        if not boxes:
            raise ValueError("\u6ca1\u6709\u5f85\u751f\u6210\u7684 PVS bbox\uff0c\u8bf7\u5148\u5728\u56fe\u50cf\u4e0a\u6846\u9009\u4e00\u4e2a\u6216\u591a\u4e2a\u76ee\u6807")

        created_ids = []
        _pvs_progress(progress, 0.12, f"读取图像缓存，共 {len(boxes)} 个 bbox")
        base_state = _fresh_state(image_state)
        for box_idx, box in enumerate(boxes, start=1):
            start = 0.18 + 0.62 * (box_idx - 1) / max(1, len(boxes))
            _pvs_progress(progress, start, f"SAM3 正在生成第 {box_idx}/{len(boxes)} 个 PVS 实例", delay=0.06)
            pred = _predict_inst(base_state, box_xyxy_px=box)
            idx = _best(pred)
            mask = pred["masks"][idx]
            inst_id = int(pvs_state.get("next_instance_id", 1))
            pvs_state.setdefault("instances", {})[inst_id] = _make_inst(
                inst_id,
                "manual_pvs_bbox_batch",
                mask,
                _mask_box(mask),
                pred["scores"][idx],
                pvs_logits=pred["lowres_logits"][idx],
                history=[{"op":"create_from_pending_bbox","box_xyxy_px":box,"candidate_scores":pred["scores"].astype(float).tolist()}],
            )
            pvs_state["next_instance_id"] = inst_id + 1
            created_ids.append(inst_id)

        _pvs_progress(progress, 0.86, "更新 PVS 实例池")
        pvs_state["active_instance_id"] = created_ids[-1]
        pvs_state["pending_boxes"] = []
        info = f"\u5df2\u4ece {len(created_ids)} \u4e2a\u5f85\u751f\u6210 bbox \u521b\u5efa PVS \u5b9e\u4f8b: {created_ids}"
        _pvs_progress(progress, 0.96, "渲染 PVS 分割结果", delay=0.16)
    except Exception as exc:
        info = f"PVS \u6279\u91cf bbox \u751f\u6210\u5931\u8d25: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _undo_pending_pvs_bbox(image_state, pcs_state, pvs_state, mode):
    try:
        pending = pvs_state.setdefault("pending_boxes", [])
        if not pending:
            raise ValueError("\u6ca1\u6709\u53ef\u64a4\u9500\u7684\u5f85\u751f\u6210 bbox")
        removed = pending.pop()
        info = f"\u5df2\u64a4\u9500\u6700\u8fd1\u4e00\u4e2a\u5f85\u751f\u6210 bbox: {[round(v, 1) for v in removed]}"
    except Exception as exc:
        info = f"\u64a4\u9500\u5f85\u751f\u6210 bbox \u5931\u8d25: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _clear_pending_pvs_boxes(image_state, pcs_state, pvs_state, mode):
    count = len(pvs_state.get("pending_boxes", []))
    pvs_state["pending_boxes"] = []
    info = f"已清空 {count} 个待生成 PVS bbox；已生成实例不会被删除"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _clear_draft_pvs_instances(image_state, pcs_state, pvs_state, mode):
    instances = pvs_state.setdefault("instances", {})
    draft_ids = [inst_id for inst_id, inst in list(instances.items()) if inst.get("status", "draft") == "draft"]
    for inst_id in draft_ids:
        instances[inst_id]["status"] = "deleted"
    if pvs_state.get("active_instance_id") in draft_ids:
        remaining = _active_instances(pvs_state)
        pvs_state["active_instance_id"] = remaining[0]["id"] if remaining else None
    info = f"已清空 {len(draft_ids)} 个草稿 PVS 实例；已确认实例保留"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _set_active_pvs(image_state, pcs_state, pvs_state, mode, selected_id):
    if selected_id:
        pvs_state["active_instance_id"] = int(selected_id)
        info = f"Selected PVS #{selected_id}"
    else:
        info = "No PVS instance selected"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)

def _pvs_point_prompt(image_state, pcs_state, pvs_state, mode, point_payload, point_kind, progress=gr.Progress(track_tqdm=False)):
    try:
        is_negative = str(point_kind or "positive") == "negative"
        point_label = 0 if is_negative else 1
        point_name = "负向点" if is_negative else "正向点"
        _pvs_progress(progress, 0.04, f"准备{point_name} PVS 操作")
        point = _point_from_payload(point_payload, image_state)
        active_id = pvs_state.get("active_instance_id")
        active_inst = None
        mask_input = None
        if active_id is not None and int(active_id) in pvs_state.get("instances", {}):
            active_inst = pvs_state["instances"][int(active_id)]
            mask_input = active_inst.get("pvs_lowres_logits")
        if is_negative and active_inst is None:
            raise ValueError("负向点必须先选择一个 active PVS instance")
        _pvs_progress(progress, 0.32, f"SAM3 正在根据{point_name}预测 mask", delay=0.12)
        pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=mask_input, point_coords_px=[point], point_labels=[point_label])
        _pvs_progress(progress, 0.78, f"整理{point_name}候选 mask")
        idx = _best(pred)
        mask = pred["masks"][idx]
        if active_inst is None:
            inst_id = int(pvs_state.get("next_instance_id", 1))
            pvs_state.setdefault("instances", {})[inst_id] = _make_inst(inst_id, "manual_pvs_point", mask, _mask_box(mask), pred["scores"][idx], pvs_logits=pred["lowres_logits"][idx], history=[{"op":"create_from_positive_point","point_xy_px":point,"candidate_scores":pred["scores"].astype(float).tolist()}])
            pvs_state["active_instance_id"] = inst_id
            pvs_state["next_instance_id"] = inst_id + 1
            info = f"Created PVS instance #{inst_id} from positive point"
        else:
            before = _snapshot(active_inst)
            active_inst["mask_fullres_bool"] = mask
            active_inst["box_xyxy_px"] = _mask_box(mask)
            active_inst["score"] = float(pred["scores"][idx])
            active_inst["pvs_lowres_logits"] = pred["lowres_logits"][idx]
            after = _snapshot(active_inst)
            op = "negative_point_refine" if is_negative else "positive_point_refine"
            prompt_type = "negative_point" if is_negative else "positive_point"
            active_inst.setdefault("prompt_history", []).append({"op":op,"prompt":{"type":prompt_type,"point_xy_px":point},"before":before,"after":after,"candidate_scores":pred["scores"].astype(float).tolist()})
            info = f"PVS #{active_id} refined with {prompt_type}"
        _pvs_progress(progress, 0.96, "渲染 PVS 分割结果", delay=0.16)
    except Exception as exc:
        info = f"PVS point prompt failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)

def _undo_pvs(image_state, pcs_state, pvs_state, mode):
    try:
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("Select a PVS instance first")
        inst = pvs_state["instances"][int(active_id)]
        history = inst.setdefault("prompt_history", [])
        while history:
            item = history.pop()
            if item.get("before") is not None:
                _restore(inst, item["before"])
                info = f"Restored PVS #{active_id} to the previous refine state"
                break
        else:
            info = "No refine step to undo"
    except Exception as exc:
        info = f"Undo failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _delete_pvs(image_state, pcs_state, pvs_state, mode):
    try:
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("Select a PVS instance first")
        pvs_state["instances"][int(active_id)]["status"] = "deleted"
        pvs_state["active_instance_id"] = None
        info = f"Deleted PVS #{active_id}"
    except Exception as exc:
        info = f"Delete failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _accept_pvs(image_state, pcs_state, pvs_state, mode):
    try:
        active_id = pvs_state.get("active_instance_id")
        if active_id is None:
            raise ValueError("Select a PVS instance first")
        pvs_state["instances"][int(active_id)]["status"] = "accepted"
        info = f"PVS #{active_id} accepted"
    except Exception as exc:
        info = f"Accept failed: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _history_json(history):
    rows = []
    for item in history:
        row = {"op": item.get("op"), "prompt": item.get("prompt"), "box_xyxy_px": item.get("box_xyxy_px"), "candidate_scores": item.get("candidate_scores")}
        if item.get("before"):
            row["before"] = {"box_xyxy_px": item["before"].get("box_xyxy_px"), "score": item["before"].get("score"), "status": item["before"].get("status")}
        if item.get("after"):
            row["after"] = {"box_xyxy_px": item["after"].get("box_xyxy_px"), "score": item["after"].get("score"), "status": item["after"].get("status")}
        rows.append({k: v for k, v in row.items() if v is not None})
    return rows


def _submit_feedback(image_state, pcs_state, pvs_state, mode, rating, feedback_tags, feedback_comment):
    try:
        if mode == "PVS Manual":
            active_id = pvs_state.get("active_instance_id")
            if active_id is None:
                raise ValueError("请先选择一个 active PVS instance")
            inst = pvs_state.get("instances", {}).get(int(active_id))
            if inst is None or inst.get("status") == "deleted":
                raise ValueError("当前 active PVS instance 不存在或已删除")
            feedback_instances = [inst]
            feedback_target = "active_pvs_instance"
        elif mode == "PCS Auto":
            feedback_instances = _active_instances(pcs_state)
            if not feedback_instances:
                raise ValueError("请先运行 PCS 并生成至少一个 PCS instance")
            inst = None
            feedback_target = "pcs_instance_pool"
        else:
            raise ValueError(f"不支持的 feedback 模式: {mode}")

        ws = _workspace(image_state)
        image = ws["image"]
        feedback_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sample_dir = runtime_feedback_dir / "samples" / feedback_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        image_path = sample_dir / "image.png"
        overlay_path = sample_dir / "overlay.png"
        mask_path = sample_dir / "mask.png"
        npz_path = sample_dir / "mask.npz"
        feedback_path = sample_dir / "feedback.json"

        image.save(image_path)
        overlay = _result_image(image_state, pcs_state, pvs_state, mode)
        if overlay is not None:
            overlay.save(overlay_path)

        masks = [np.asarray(item["mask_fullres_bool"]).astype(bool) for item in feedback_instances]
        mask_stack = np.stack([mask.astype(np.uint8) for mask in masks], axis=0)
        mask_preview = np.any(mask_stack.astype(bool), axis=0).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_preview * 255)
        pvs_logits_values = [item.get("pvs_lowres_logits") for item in feedback_instances if item.get("pvs_lowres_logits") is not None]
        pcs_prob_values = [item.get("pcs_fullres_prob") for item in feedback_instances if item.get("pcs_fullres_prob") is not None]
        np.savez_compressed(
            npz_path,
            mask_fullres_uint8=mask_stack,
            pvs_lowres_logits=np.stack([np.asarray(v, dtype=np.float32) for v in pvs_logits_values], axis=0) if pvs_logits_values else np.empty((0,), dtype=np.float32),
            pcs_fullres_prob=np.stack([np.asarray(v, dtype=np.float32) for v in pcs_prob_values], axis=0) if pcs_prob_values else np.empty((0,), dtype=np.float32),
        )
        instance_rows = [
            {
                "instance_id": int(item["id"]),
                "source": item.get("source"),
                "status": item.get("status"),
                "score": float(item.get("score", 0.0)),
                "bbox_xyxy_px": [float(v) for v in item.get("box_xyxy_px", [])],
                "prompt_history": _history_json(item.get("prompt_history", [])),
            }
            for item in feedback_instances
        ]

        payload = {
            "feedback_id": feedback_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "target": feedback_target,
            "rating": rating,
            "tags": feedback_tags or [],
            "comment": feedback_comment or "",
            "image_id": image_state.get("image_id"),
            "image_size": [int(image.width), int(image.height)],
            "instance_id": int(inst["id"]) if inst is not None else None,
            "source": inst.get("source") if inst is not None else "pcs",
            "status": inst.get("status") if inst is not None else None,
            "score": float(inst.get("score", 0.0)) if inst is not None else None,
            "bbox_xyxy_px": [float(v) for v in inst.get("box_xyxy_px", [])] if inst is not None else None,
            "prompt_history": _history_json(inst.get("prompt_history", [])) if inst is not None else [],
            "instance_count": len(feedback_instances),
            "instances": instance_rows,
            "image_file": str(image_path),
            "overlay_file": str(overlay_path) if overlay is not None else None,
            "mask_file": str(mask_path),
            "mask_npz_file": str(npz_path),
            "branch": "Zhengqiyuan/PVS-demo",
        }

        with feedback_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with _FEEDBACK_WRITE_LOCK:
            with (runtime_feedback_dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        info = f"反馈已保存: {feedback_id}"
    except Exception as exc:
        info = f"反馈保存失败: {exc}"
    return _view(image_state, pcs_state, pvs_state, mode, info)


def _export_pool(image_state, pcs_state, pvs_state, mode, pool_name, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    try:
        ws = _workspace(image_state)
        image = ws["image"]
        pool = pcs_state if pool_name == "pcs" else pvs_state
        instances = _active_instances(pool)
        if not instances:
            raise ValueError(f"No active {pool_name.upper()} instance")
        export_id = f"{pool_name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        export_dir = runtime_export_dir / export_id
        mask_dir = export_dir / "masks"
        export_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        _overlay(image_state, pcs_state, pvs_state, mode).save(export_dir / "overlay.png")
        masks, scores, predictions = [], [], []
        coco_annotation_extras = []
        for inst in instances:
            mask = np.asarray(inst["mask_fullres_bool"]).astype(bool)
            masks.append(mask)
            scores.append(float(inst.get("score", 1.0)))
            mask_path = mask_dir / f"{pool_name}_{inst['id']:03d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            mask_file = str(mask_path.relative_to(export_dir))
            bbox_xyxy = [float(v) for v in inst.get("box_xyxy_px", [])]
            predictions.append({"id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "score": float(inst.get("score", 0.0)), "bbox_xyxy": bbox_xyxy, "mask_file": mask_file, "final_contour_polygon": mask_to_polygons(mask), "prompt_history": _history_json(inst.get("prompt_history", []))})
            coco_annotation_extras.append({"instance_id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "mask_file": mask_file, "bbox_xyxy": bbox_xyxy})
        metrics = compare_with_coco(masks, scores, coco_dataset, coco_image_name.strip() if coco_image_name else "", coco_split, pcs_state.get("text_prompt", "") if pool_name == "pcs" else "", image.width, image.height, coco_eval_scope, annotation_json_file)
        with (export_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump({"export_id": export_id, "pool": pool_name, "image": {"width": image.width, "height": image.height}, "predictions": predictions, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        with (export_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        coco_payload = create_prediction_coco_json(
            masks,
            scores,
            image.width,
            image.height,
            image_file_name=coco_image_name.strip() if coco_image_name else "source_image",
            category_name=f"{pool_name}_object",
            export_id=export_id,
            annotation_extras=coco_annotation_extras,
        )
        with (export_dir / "coco_masks.json").open("w", encoding="utf-8") as f:
            json.dump(coco_payload, f, ensure_ascii=False, indent=2)
        zip_path = runtime_export_dir / f"{export_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in export_dir.rglob("*"):
                zf.write(file_path, arcname=file_path.relative_to(export_dir))
        info = f"Exported {len(instances)} {pool_name.upper()} instances: {zip_path}"
        if metrics.get("summary_lines"):
            info += "\n" + "\n".join(metrics["summary_lines"])
        return str(zip_path), info
    except Exception as exc:
        return None, f"Export failed: {exc}"


def _export_pcs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pcs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)


def _export_pvs(image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pvs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)


def _switch_mode(mode, image_state, pcs_state, pvs_state):
    prompt_state = _new_prompt_state()
    is_pcs = mode == "PCS Auto"
    is_pvs = mode == "PVS Manual"
    if is_pcs:
        tool_update = gr.update(choices=[("框提示 (Box)", "bbox")], value="bbox")
        finish_update = gr.update(visible=False)
    else:
        tool_update = gr.update(
            choices=[("点提示 (Point)", "point"), ("框提示 (Box)", "bbox"), ("多边形Mask (Polygon)", "polygon")],
            value="bbox",
        )
        finish_update = gr.update(visible=True)
    return (
        prompt_state,
        "",
        "",
        "",
        tool_update,
        finish_update,
        gr.update(visible=is_pcs),
        gr.update(visible=is_pcs),
        gr.update(visible=is_pvs),
        gr.update(visible=is_pvs),
        gr.update(visible=is_pvs),
        gr.update(visible=False),
        gr.update(visible=False),
        *_view(image_state, pcs_state, pvs_state, mode, f"Mode: {mode}，交互提示已重置", prompt_state),
    )


def _switch_click_tool(click_tool, mode):
    tool = _click_tool_key(click_tool)
    is_pvs = mode == "PVS Manual"
    return (
        gr.update(visible=is_pvs and tool == "bbox"),
        gr.update(visible=is_pvs and tool == "point"),
        gr.update(visible=is_pvs and tool == "polygon"),
    )


def create_demo():
    """Create the PCS/PVS Gradio interface while preserving the original demo layout."""
    custom_css = """
    .container { max-width: 1200px; margin: auto; padding-top: 20px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    .hidden-payload { display: none !important; }
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    .sam3-panel textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    """
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0") as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0")
            gr.Markdown("\u57fa\u4e8e SAM3 \u7684 PCS \u81ea\u52a8\u6982\u5ff5\u5206\u5272\u4e0e PVS \u624b\u52a8\u5b9e\u4f8b\u5206\u5272\u5de5\u4f5c\u53f0", elem_classes="description")
            image_state = gr.State({"image_id": None, "width": 0, "height": 0})
            pcs_state = gr.State(_new_pcs_state())
            pvs_state = gr.State(_new_pvs_state())
            prompt_state = gr.State(_new_prompt_state())
            bbox_payload = gr.Textbox(label="bbox payload", elem_id="bbox_payload", elem_classes="hidden-payload")
            polygon_payload = gr.Textbox(label="polygon payload", elem_id="polygon_payload", elem_classes="hidden-payload")
            point_payload = gr.Textbox(label="point payload", elem_id="point_payload", elem_classes="hidden-payload")

            with gr.Tabs():
                with gr.TabItem("智能图像分割", id="tab_image"):
                    mode = gr.Radio(
                        choices=[("PCS Auto 自动概念分割", "PCS Auto"), ("PVS Manual 手动实例分割", "PVS Manual")],
                        value="PVS Manual",
                        label="功能模式",
                        elem_classes="mode-radio",
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 原始图像（点击进行交互）")
                            image_upload = gr.Image(type="numpy", label="原始图像", show_label=False, sources=["upload", "clipboard"], elem_id="input_image")
                            with gr.Group():
                                gr.Markdown("### \u4ea4\u4e92\u6a21\u5f0f")
                                click_tool = gr.Radio(
                                    choices=[("\u70b9\u63d0\u793a (Point)", "point"), ("\u6846\u63d0\u793a (Box)", "bbox"), ("\u591a\u8fb9\u5f62Mask (Polygon)", "polygon")],
                                    value="bbox",
                                    label="\u9009\u62e9\u6a21\u5f0f",
                                    show_label=False,
                                    elem_classes="mode-radio",
                                )
                                with gr.Group(visible=False) as pcs_bbox_tools:
                                    pcs_bbox_kind = gr.Radio(
                                        choices=[("\u6b63\u6837\u672c bbox", "Positive exemplar"), ("\u8d1f\u6837\u672c bbox", "Negative exemplar")],
                                        value="Positive exemplar",
                                        label="PCS bbox \u6837\u672c\u7c7b\u578b",
                                        elem_classes="mode-radio",
                                    )
                                    undo_pcs_bbox_btn = gr.Button("\u64a4\u9500\u6700\u8fd1 PCS bbox", size="sm", variant="secondary")
                                with gr.Row():
                                    clear_prompt_btn = gr.Button("\u6e05\u7a7a\u63d0\u793a (Clear Prompts)", size="sm", variant="secondary")
                                interaction_info = gr.Markdown("\u70b9\u51fb\u56fe\u50cf\u5f00\u59cb\u6dfb\u52a0\u63d0\u793a...", elem_id="interaction-info")

                            with gr.Accordion("\u9ad8\u7ea7\u63d0\u793a\u9009\u9879", open=True):
                                with gr.Group(visible=False) as pcs_panel:
                                    gr.Markdown("### PCS Auto \u81ea\u52a8\u6982\u5ff5\u5206\u5272")
                                    text_prompt = gr.Textbox(label="\u6587\u672c\u63d0\u793a (Text Prompt)", placeholder="\u8f93\u5165\u7269\u4f53\u63cf\u8ff0\uff0c\u4f8b\u5982\uff1a'a red car' \u6216 '\u4e00\u53ea\u732b'", lines=1)
                                    confidence_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="\u7f6e\u4fe1\u5ea6\u9608\u503c (Confidence)")
                                    run_pcs_btn = gr.Button("\u5f00\u59cb PCS \u5206\u5272", variant="primary")
                                    export_pcs_btn = gr.Button("\u5bfc\u51fa PCS")
                                    pcs_summary = gr.Textbox(label="PCS \u5b9e\u4f8b", lines=6, interactive=False)

                                with gr.Group(visible=True) as pvs_panel:
                                    gr.Markdown("### PVS Manual \u624b\u52a8\u5b9e\u4f8b\u5206\u5272")
                                    with gr.Group(visible=True) as pvs_bbox_prompt_panel:
                                        gr.Markdown("#### BBox prompt")
                                        pvs_pending_count = gr.Markdown("待生成 bbox 数量: 0")
                                        create_pvs_batch_btn = gr.Button("\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b", variant="primary")
                                        with gr.Row():
                                            undo_pending_bbox_btn = gr.Button("移除上一个待生成 bbox", size="sm", variant="secondary")
                                            clear_pending_bbox_btn = gr.Button("清空待生成 bbox", size="sm", variant="secondary")
                                    with gr.Group(visible=False) as pvs_point_prompt_panel:
                                        gr.Markdown("#### Point prompt")
                                        pvs_point_kind = gr.Radio(
                                            choices=[("正向点", "positive"), ("负向点", "negative")],
                                            value="positive",
                                            label="点类型",
                                            elem_classes="mode-radio",
                                        )
                                        pvs_point_btn = gr.Button("应用点提示", variant="primary")
                                    with gr.Group(visible=False) as pvs_polygon_prompt_panel:
                                        gr.Markdown("#### Polygon prompt")
                                        polygon_action = gr.Radio(
                                            choices=[("\u521b\u5efa\u65b0 PVS \u5b9e\u4f8b", "create"), ("\u7cbe\u4fee\u5f53\u524d PVS \u5b9e\u4f8b", "refine")],
                                            value="create",
                                            label="\u591a\u8fb9\u5f62\u52a8\u4f5c",
                                            elem_classes="mode-radio",
                                        )
                                        finish_polygon_btn = gr.Button("\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61", size="sm", variant="primary")
                                    clear_draft_pvs_btn = gr.Button("清空草稿 PVS 实例", size="sm", variant="secondary")
                                    active_pvs = gr.Dropdown(choices=[], label="\u5f53\u524d PVS \u5b9e\u4f8b")
                                    analysis_report = gr.Textbox(label="分析报告", interactive=False, lines=18)
                                    pvs_summary = gr.Textbox(label="PVS 实例", lines=6, interactive=False, visible=False)

                                with gr.Accordion("\u5bfc\u51fa\u4e0e COCO \u91cf\u5316", open=False):
                                    coco_dataset = gr.Dropdown(choices=coco_dataset_choices, value=default_coco_dataset, label="\u6307\u6807\u6570\u636e\u96c6")
                                    coco_image_name = gr.Textbox(label="COCO image file_name\uff08\u53ef\u9009\uff09", lines=1)
                                    coco_split = gr.Radio(choices=["auto", "val", "train", "test"], value="auto", label="\u6807\u6ce8 split")
                                    coco_eval_scope = gr.Radio(choices=[coco_eval_scope_overlap, coco_eval_scope_full], value=coco_eval_scope_overlap, label="\u8bc4\u4f30\u8303\u56f4")
                                    annotation_json_file = gr.File(label="\u4e0a\u4f20 O3/LabelMe-like JSON \u6807\u6ce8\uff08\u4f18\u5148\u4e8e COCO lookup\uff09", file_types=[".json"], type="filepath")

                        with gr.Column(scale=1):
                            result_image = gr.Image(type="numpy", label="\u5206\u5272\u7ed3\u679c")
                            with gr.Group(visible=True) as pvs_action_panel:
                                gr.Markdown("### PVS 实例操作")
                                polygon_combine_mode = gr.Radio(
                                    choices=[("Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b", "replace"), ("Blend \u4e0e\u65e7 mask \u878d\u5408", "blend"), ("Union \u8865\u5145\u533a\u57df", "union"), ("Intersect \u9650\u5236\u8303\u56f4", "intersect")],
                                    value="replace",
                                    label="\u591a\u8fb9\u5f62\u878d\u5408\u65b9\u5f0f",
                                    elem_classes="mode-radio",
                                )
                                gr.Markdown(
                                    "**\u591a\u8fb9\u5f62\u878d\u5408\u65b9\u5f0f\u8bf4\u660e**  \n"
                                    "- Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b\uff1a\u7528\u5f53\u524d polygon \u4f5c\u4e3a\u5b8c\u6574 mask prompt\u3002  \n"
                                    "- Blend \u4e0e\u65e7 mask \u878d\u5408\uff1a\u65e7 logits \u548c polygon logits \u5171\u540c\u5f71\u54cd\u7ed3\u679c\u3002  \n"
                                    "- Union \u8865\u5145\u533a\u57df\uff1a\u4fdd\u7559\u65e7 mask\uff0c\u5e76\u52a0\u5165 polygon \u533a\u57df\u3002  \n"
                                    "- Intersect \u9650\u5236\u8303\u56f4\uff1a\u5c06\u7ed3\u679c\u9650\u5236\u5728 polygon \u8303\u56f4\u5185\u3002"
                                )
                                with gr.Row():
                                    undo_pvs_btn = gr.Button("\u64a4\u9500")
                                    delete_pvs_btn = gr.Button("\u5220\u9664")
                                    accept_pvs_btn = gr.Button("确认", variant="primary")
                                export_pvs_btn = gr.Button("\u5bfc\u51fa PVS")
                            export_file = gr.File(label="\u4e0b\u8f7d\u7ed3\u679c\u5305\uff08PNG + masks + JSON\uff09", interactive=False)
                            with gr.Accordion("结果反馈（PCS 结果 / PVS 当前实例，用于 RL 数据收集）", open=False):
                                feedback_rating = gr.Radio(
                                    choices=[("好", "good"), ("及格", "pass"), ("差", "bad")],
                                    value="pass",
                                    label="结果质量",
                                    elem_classes="mode-radio",
                                )
                                feedback_tags = gr.CheckboxGroup(
                                    choices=["毛边", "空缺", "漏检", "误检", "边界偏移", "多分/粘连", "polygon 不贴合", "其他"],
                                    label="问题标签",
                                )
                                feedback_comment = gr.Textbox(label="备注", lines=3, placeholder="可选：描述这次生成的问题或可用性")
                                submit_feedback_btn = gr.Button("提交反馈", variant="primary")

                with gr.TabItem("\u89c6\u9891\u76ee\u6807\u8ddf\u8e2a", id="tab_video"):
                    gr.Markdown("\u5f53\u524d PVS demo \u5206\u652f\u805a\u7126\u56fe\u50cf\u5206\u5272\uff1b\u89c6\u9891\u76ee\u6807\u8ddf\u8e2a\u8bf7\u4f7f\u7528\u539f\u59cb demo \u5206\u652f\u3002")

            common = [image_upload, result_image, analysis_report, pcs_summary, pvs_summary, active_pvs, interaction_info, pvs_pending_count]
            image_upload.upload(fn=_init_workspace, inputs=[image_upload, mode], outputs=[image_state, pcs_state, pvs_state, prompt_state, *common, export_file], concurrency_limit=1)
            image_upload.select(fn=_workspace_select, inputs=[image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, *common], concurrency_limit=1)
            finish_polygon_btn.click(fn=_finish_native_polygon, inputs=[image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode], outputs=[prompt_state, polygon_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1)
            clear_prompt_btn.click(fn=_clear_prompt_selection, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, *common], concurrency_limit=1)
            mode.change(fn=_switch_mode, inputs=[mode, image_state, pcs_state, pvs_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, click_tool, finish_polygon_btn, pcs_bbox_tools, pcs_panel, pvs_panel, pvs_action_panel, pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel, *common], concurrency_limit=1)
            click_tool.change(fn=_switch_click_tool, inputs=[click_tool, mode], outputs=[pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel], concurrency_limit=1)
            undo_pcs_bbox_btn.click(fn=_undo_pcs_bbox, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pcs_state, *common], concurrency_limit=1)
            run_pcs_btn.click(fn=_run_pcs, inputs=[image_state, pcs_state, pvs_state, mode, text_prompt, confidence_threshold], outputs=[pcs_state, *common], concurrency_limit=1)
            create_pvs_batch_btn.click(fn=_create_pvs_from_pending_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1)
            undo_pending_bbox_btn.click(fn=_undo_pending_pvs_bbox, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            clear_pending_bbox_btn.click(fn=_clear_pending_pvs_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            clear_draft_pvs_btn.click(fn=_clear_draft_pvs_instances, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            pvs_point_btn.click(fn=_pvs_point_prompt, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, pvs_point_kind], outputs=[pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1)
            active_pvs.change(fn=_set_active_pvs, inputs=[image_state, pcs_state, pvs_state, mode, active_pvs], outputs=[pvs_state, *common], concurrency_limit=1)
            undo_pvs_btn.click(fn=_undo_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            delete_pvs_btn.click(fn=_delete_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            accept_pvs_btn.click(fn=_accept_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1)
            export_pcs_btn.click(fn=_export_pcs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1)
            export_pvs_btn.click(fn=_export_pvs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1)
            submit_feedback_btn.click(fn=_submit_feedback, inputs=[image_state, pcs_state, pvs_state, mode, feedback_rating, feedback_tags, feedback_comment], outputs=common, concurrency_limit=1)
        gr.Markdown("---\n<div style='text-align:center;color:#718096;font-size:0.9em;'>Powered by SAM3</div>")
    return demo
# --- end PCS/PVS single-workspace override ---


def main():
    """主函数"""
    # 检查模型文件
    model_dir = current_dir / "models"
    if not model_dir.exists():
        print(f"创建模型目录: {model_dir}")
        model_dir.mkdir(exist_ok=True)

    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ 模型文件缺失")
        print(f"请确保以下文件存在:\n1. {checkpoint_path}\n2. {bpe_path}")

        response = input("是否尝试自动下载模型文件？(y/n): ").lower().strip()
        if response == "y":
            try:
                import download_models

                download_models.main()
            except Exception as e:
                print(f"自动下载失败: {e}")
                return
        else:
            return

    print("🚀 正在启动 SAM3 交互式视觉工作台...")
    demo = create_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=True,
        allowed_paths=[str(current_dir)],
    )


if __name__ == "__main__":
    main()
