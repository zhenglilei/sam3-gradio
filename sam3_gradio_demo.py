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
import copy

# 所有运行时文件固定在 /data/zhengqiyuan，避免 Gradio 默认写入 /tmp/gradio。
current_dir = Path(__file__).resolve().parent
runtime_dir = current_dir / ".runtime"
runtime_tmp_dir = runtime_dir / "tmp"
runtime_gradio_dir = runtime_dir / "gradio"
runtime_video_dir = runtime_dir / "videos"
runtime_export_dir = runtime_dir / "exports"
runtime_feedback_dir = runtime_dir / "feedback"
runtime_layout_dir = runtime_dir / "layout_masks"
runtime_layout_region_dir = runtime_dir / "layout_regions"
runtime_log_dir = runtime_dir / "logs"
public_download_dir = current_dir / "public_downloads"
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
    runtime_layout_dir,
    runtime_layout_region_dir,
    runtime_log_dir,
    public_download_dir,
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
component_backend_dir = current_dir / 'layout_transform_editor' / 'backend'
if component_backend_dir.exists():
    sys.path.insert(0, str(component_backend_dir))
region_component_backend_dir = current_dir / "layout_region_annotator" / "backend"
if region_component_backend_dir.exists():
    sys.path.insert(0, str(region_component_backend_dir))

import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
import layout_transform_utils as _layout_tx
import layout_region_utils as _layout_regions
import public_download_utils as _public_downloads

try:
    from gradio_layout_transform_editor import LayoutTransformEditor
except Exception as exc:
    LayoutTransformEditor = None
    _layout_editor_import_error = exc
else:
    _layout_editor_import_error = None

try:
    from gradio_layout_region_annotator import LayoutRegionAnnotator
except Exception as exc:
    LayoutRegionAnnotator = None
    _layout_region_annotator_import_error = exc
else:
    _layout_region_annotator_import_error = None

try:
    from scripts.layout_image_to_mask import extract_layout_mask as _layout_extract_mask
    _layout_extract_mask_import_error = None
except Exception as exc:
    _layout_extract_mask = None
    _layout_extract_mask_import_error = exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sam3_gradio_demo")

_PUBLIC_DOWNLOAD_TTL_SECONDS = 24 * 60 * 60


def _gradio_allowed_paths():
    return [str(public_download_dir.resolve())]


def _gradio_blocked_paths():
    exempt_top_level = {
        public_download_dir.name,
        runtime_dir.name,
        ".gradio",
    }
    blocked = [
        str(path.resolve())
        for path in current_dir.iterdir()
        if path.name not in exempt_top_level
    ]
    exempt_runtime = {runtime_gradio_dir.resolve(), runtime_video_dir.resolve()}
    blocked.extend(
        str(path.resolve())
        for path in runtime_dir.iterdir()
        if path.resolve() not in exempt_runtime
    )
    return sorted(set(blocked))


def _prune_public_downloads():
    try:
        return _public_downloads.prune_public_downloads(
            public_download_dir,
            max_age_seconds=_PUBLIC_DOWNLOAD_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Cannot prune public downloads: %s", exc)
        return []


def _publish_layout_downloads(mask_path, contour_path):
    _prune_public_downloads()
    export_dir = _public_downloads.publish_files(
        public_download_dir,
        "layout_mask_exports",
        {
            "source_mask.png": mask_path,
            "contours.json": contour_path,
        },
    )
    return str(export_dir / "source_mask.png"), str(export_dir / "contours.json")


def _publish_segmentation_zip(export_dir, zip_name):
    _prune_public_downloads()
    return _public_downloads.publish_zip(
        public_download_dir,
        "pcs_pvs_exports",
        export_dir,
        zip_name,
    )


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
    zip_path = _publish_segmentation_zip(export_dir, f"{safe_stem(zip_stem)}_{export_id}.zip")
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
_LAYOUT_CACHE = {}
_LAYOUT_CACHE_LOCK = _sam3_threading.RLock()
_LAYOUT_REGION_STORE = _layout_regions.LayoutRegionStore(
    layout_masks_root=runtime_layout_dir,
    layout_regions_root=runtime_layout_region_dir,
    categories_path=current_dir / "layout_categories.json",
)


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
    return {
        "text_prompt": "",
        "positive_boxes": [],
        "negative_boxes": [],
        "bbox_history": [],
        "bbox_records": [],
        "next_bbox_id": 1,
        "instances": {},
        "next_instance_id": 1,
    }


def _new_pvs_state():
    return {
        "instances": {},
        "active_instance_id": None,
        "next_instance_id": 1,
        "pending_boxes": [],
        "pending_bbox_records": [],
        "next_pending_bbox_id": 1,
    }



def _new_session_state():
    return {"session_id": uuid.uuid4().hex}


def _session_id_from_state(session_state=None):
    if isinstance(session_state, dict) and session_state.get("session_id"):
        return str(session_state["session_id"])
    return uuid.uuid4().hex


def _new_layout_state(session_id=None):
    return {
        "transform_version": 2,
        "session_id": str(session_id or uuid.uuid4().hex),
        "layout_id": None,
        "image_id": None,
        "enabled": False,
        "region_mode": "all",
        "revision": 0,
        "center_x": None,
        "center_y": None,
        "pivot_x": None,
        "pivot_y": None,
        "tx": 0.0,
        "ty": 0.0,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "preview_alpha": 0.35,
        "source_width": 0,
        "source_height": 0,
        "source_mask_pixel_sha256": None,
        "source_mask_file_sha256": None,
        "target_image_sha256": None,
        "matrix_2x3": None,
    }


def _layout_cache_key(session_id, layout_id):
    if not layout_id:
        raise ValueError("请先在‘版图截图转掩码’Tab 中生成并保存当前版图 mask")
    sid = _layout_tx.safe_id(session_id, "default")
    lid = _layout_tx.safe_id(layout_id, "layout")
    return f"{sid}:{lid}"


def _layout_disk_dir(session_id, layout_id):
    return runtime_layout_dir / _layout_tx.safe_id(session_id, "default") / _layout_tx.safe_id(layout_id, "layout")


def _layout_cache_get(layout_state_or_id, session_id=None):
    if isinstance(layout_state_or_id, dict):
        layout_id = layout_state_or_id.get("layout_id")
        session_id = session_id or layout_state_or_id.get("session_id")
    else:
        layout_id = layout_state_or_id
    if not layout_id:
        raise ValueError("请先在‘版图截图转掩码’Tab 中生成并保存当前版图 mask")
    session_id = session_id or "default"
    key = _layout_cache_key(session_id, layout_id)
    with _LAYOUT_CACHE_LOCK:
        cached = _LAYOUT_CACHE.get(key)
        if cached is not None:
            return cached
        cached = _restore_layout_cache_from_disk(session_id, layout_id)
        if cached is not None:
            _LAYOUT_CACHE[key] = cached
            return cached
    raise ValueError(f"版图缓存已失效或不存在: {layout_id}。请重新生成版图 mask。")


def _restore_layout_cache_from_disk(session_id, layout_id):
    out_dir = _layout_disk_dir(session_id, layout_id)
    meta_path = out_dir / "layout_meta.json"
    mask_path = out_dir / "source_mask.png"
    if not meta_path.exists() or not mask_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    file_hash = _layout_tx.file_sha256(mask_path)
    if meta.get("source_mask_file_sha256") and meta.get("source_mask_file_sha256") != file_hash:
        raise ValueError("版图 source_mask.png 文件 hash 不匹配，拒绝恢复缓存")
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("版图 source_mask.png 无法读取")
    source_mask = gray >= 128
    pixel_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    if meta.get("source_mask_pixel_sha256") and meta.get("source_mask_pixel_sha256") != pixel_hash:
        raise ValueError("版图 source mask 像素 hash 不匹配，拒绝恢复缓存")
    image_path = out_dir / "source_image.png"
    source_image = Image.open(image_path).convert("RGB") if image_path.exists() else _layout_mask_to_preview(source_mask)
    return {
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "source_image": source_image,
        "source_mask": source_mask,
        "source_mask_path": str(mask_path),
        "source_mask_pixel_sha256": pixel_hash,
        "source_mask_file_sha256": file_hash,
        "target_image_sha256": meta.get("target_image_sha256"),
        "layout_meta_path": str(meta_path),
        "foreground_bbox_xyxy": meta.get("foreground_bbox_xyxy") or _layout_tx.foreground_bbox_xyxy(source_mask),
        "pivot_xy": meta.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(_layout_tx.foreground_bbox_xyxy(source_mask)),
        "source_width": int(source_mask.shape[1]),
        "source_height": int(source_mask.shape[0]),
        "transformed_mask": None,
        "committed_revision": int(meta.get("committed_revision") or 0),
        "backend_transform": meta.get("backend_transform"),
        "matrix_2x3": meta.get("matrix_2x3"),
        "contours": meta.get("contours") or [],
        "binarize_params": meta.get("binarize_params") or {},
        "mask_path": str(mask_path),
        "contour_json_path": str(out_dir / "contours.json") if (out_dir / "contours.json").exists() else None,
        "overlay_path": str(out_dir / "contour_overlay.png") if (out_dir / "contour_overlay.png").exists() else None,
    }


def _write_layout_meta(cached):
    meta_path = Path(cached["layout_meta_path"])
    payload = {
        "session_id": cached.get("session_id"),
        "layout_id": cached.get("layout_id"),
        "source_mask_path": cached.get("source_mask_path"),
        "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        "target_image_sha256": cached.get("target_image_sha256"),
        "foreground_bbox_xyxy": cached.get("foreground_bbox_xyxy"),
        "pivot_xy": cached.get("pivot_xy"),
        "source_width": cached.get("source_width"),
        "source_height": cached.get("source_height"),
        "committed_revision": cached.get("committed_revision"),
        "backend_transform": cached.get("backend_transform"),
        "matrix_2x3": cached.get("matrix_2x3"),
        "binarize_params": cached.get("binarize_params") or {},
        "contours": cached.get("contours") or [],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _layout_cache_put(session_id, layout_id, source_image, source_mask, contours, binarize_params, mask_path=None, contour_json_path=None, overlay_path=None, layout_meta_path=None):
    source_mask = np.asarray(source_mask, dtype=bool)
    bbox = _layout_tx.foreground_bbox_xyxy(source_mask)
    pivot = _layout_tx.pivot_from_bbox_xyxy(bbox)
    mask_path = str(mask_path) if mask_path else None
    file_hash = _layout_tx.file_sha256(mask_path) if mask_path else None
    pixel_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    cached = {
        "session_id": str(session_id),
        "layout_id": str(layout_id),
        "source_image": _pil_image(source_image),
        "source_mask": source_mask,
        "source_mask_path": mask_path,
        "layout_meta_path": str(layout_meta_path) if layout_meta_path else None,
        "source_mask_pixel_sha256": pixel_hash,
        "source_mask_file_sha256": file_hash,
        "target_image_sha256": None,
        "foreground_bbox_xyxy": bbox,
        "pivot_xy": pivot,
        "source_width": int(source_mask.shape[1]),
        "source_height": int(source_mask.shape[0]),
        "transformed_mask": None,
        "committed_revision": 0,
        "backend_transform": None,
        "matrix_2x3": None,
        "contours": contours or [],
        "binarize_params": dict(binarize_params or {}),
        "mask_path": mask_path,
        "contour_json_path": str(contour_json_path) if contour_json_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
    }
    key = _layout_cache_key(session_id, layout_id)
    with _LAYOUT_CACHE_LOCK:
        _LAYOUT_CACHE[key] = cached
        if cached.get("layout_meta_path"):
            _write_layout_meta(cached)
    return cached


def _clear_layout_cache(layout_state=None):
    with _LAYOUT_CACHE_LOCK:
        if layout_state and isinstance(layout_state, dict) and layout_state.get("layout_id"):
            _LAYOUT_CACHE.pop(_layout_cache_key(layout_state.get("session_id") or "default", layout_state.get("layout_id")), None)
        else:
            _LAYOUT_CACHE.clear()

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


MODE_PCS = "PCS Auto"
MODE_PVS = "PVS Manual"
MODE_LAYOUT = "Layout Mask"


def _is_pcs_mode(mode):
    return str(mode or "") == MODE_PCS


def _is_pvs_manual_mode(mode):
    return str(mode or "") == MODE_PVS


def _is_layout_mask_mode(mode):
    return str(mode or "") == MODE_LAYOUT


def _is_pvs_pool_mode(mode):
    return _is_pvs_manual_mode(mode) or _is_layout_mask_mode(mode)


def _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True, show_interaction_prompts=True, show_layout_overlay=False, layout_state=None):
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

    if show_layout_overlay and layout_state and layout_state.get("enabled"):
        layout_id = layout_state.get("layout_id")
        try:
            cached = _layout_cache_get(layout_state)
        except Exception:
            cached = None
        if cached is not None and cached.get("transformed_mask") is not None:
            layout_mask = np.asarray(cached["transformed_mask"], dtype=bool)
            if layout_mask.shape == overlay.shape[:2]:
                paint(layout_mask, (0, 255, 130), float(layout_state.get("preview_alpha") or 0.35))
                ys, xs = np.where(layout_mask)
                if len(xs):
                    queue_label(f"LAYOUT {layout_id}", int(xs.min()), max(18, int(ys.min()) - 6), (0, 255, 130))
    if show_instances and _is_pcs_mode(mode):
        for inst in _active_instances(pcs_state):
            color = (0, 255, 90)
            paint(inst["mask_fullres_bool"], color, 0.24)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 3)
            queue_label(f"PCS#{inst['id']}", x1, max(18, y1 - 6), color)
    if _is_pvs_manual_mode(mode) and not show_instances:
        for rec in _pvs_pending_bbox_records(pvs_state):
            box = rec.get("box", [])
            if len(box) != 4:
                continue
            color = (0, 255, 90)
            queue_box(box, color, 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            queue_label(f"B-ID{rec.get('id')}", x1, max(18, y1 - 6), color)
    if show_instances and _is_pvs_pool_mode(mode):
        active_id = pvs_state.get("active_instance_id")
        for inst in _active_instances(pvs_state):
            is_active = str(inst["id"]) == str(active_id)
            color = (255, 0, 220) if is_active else (0, 185, 255)
            paint(inst["mask_fullres_bool"], color, 0.32 if is_active else 0.22)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["box_xyxy_px"]]
            queue_box((x1, y1, x2, y2), color, 4 if is_active else 3)
            queue_label(f"PVS#{inst['id']}", x1, max(18, y1 - 6), color)
    if _is_pcs_mode(mode) and not show_instances:
        for rec in _pcs_bbox_records(pcs_state):
            color = (255, 48, 48) if rec.get("key") == "negative_boxes" else (0, 255, 90)
            box = rec.get("box", [])
            if len(box) != 4:
                continue
            queue_box(box, color, 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            role = "N" if rec.get("key") == "negative_boxes" else "P"
            queue_label(f"{role}-ID{rec.get('id')}", x1, max(18, y1 - 6), color)
    if show_interaction_prompts and prompt_state:
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
    return _active_instances(pcs_state if _is_pcs_mode(mode) else pvs_state)


def _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None, layout_state=None):
    if not image_state or not image_state.get("image_id"):
        return None
    return _overlay(
        image_state,
        pcs_state,
        pvs_state,
        mode,
        prompt_state,
        show_instances=False,
        show_interaction_prompts=True,
        show_layout_overlay=bool(layout_state and layout_state.get("enabled")),
        layout_state=layout_state,
    )


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
    return _overlay(image_state, pcs_state, pvs_state, mode, prompt_state=None, show_instances=True, show_interaction_prompts=False, show_layout_overlay=False)

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


def _reset_pcs_predictions(pcs_state):
    pcs_state["instances"] = {}
    pcs_state["next_instance_id"] = 1


def _pcs_bbox_records(pcs_state):
    records = pcs_state.setdefault("bbox_records", [])
    if records:
        return records

    next_id = int(pcs_state.get("next_bbox_id", 1) or 1)
    rebuilt = []
    for key in ("positive_boxes", "negative_boxes"):
        for box in pcs_state.get(key, []):
            rebuilt.append({"id": next_id, "key": key, "box": box})
            next_id += 1
    if rebuilt:
        pcs_state["bbox_records"] = rebuilt
        pcs_state["bbox_history"] = [dict(item) for item in rebuilt]
        pcs_state["next_bbox_id"] = next_id
    return pcs_state.setdefault("bbox_records", [])


def _sync_pcs_boxes_from_records(pcs_state):
    records = _pcs_bbox_records(pcs_state)
    pcs_state["positive_boxes"] = [rec.get("box") for rec in records if rec.get("key") == "positive_boxes"]
    pcs_state["negative_boxes"] = [rec.get("box") for rec in records if rec.get("key") == "negative_boxes"]


def _pcs_bbox_choices(pcs_state):
    choices = []
    for rec in _pcs_bbox_records(pcs_state):
        role = "负样本" if rec.get("key") == "negative_boxes" else "正样本"
        box = [round(float(v), 1) for v in rec.get("box", [])]
        choices.append((f"ID {rec.get('id')} {role} {box}", str(rec.get("id"))))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _append_pcs_bbox_sample(pcs_state, box, bbox_role):
    _pcs_bbox_records(pcs_state)
    key = "negative_boxes" if bbox_role == "negative" else "positive_boxes"
    bbox_id = int(pcs_state.get("next_bbox_id", 1) or 1)
    record = {"id": bbox_id, "key": key, "box": box}
    pcs_state.setdefault("bbox_records", []).append(record)
    pcs_state.setdefault("bbox_history", []).append(dict(record))
    pcs_state["next_bbox_id"] = bbox_id + 1
    _sync_pcs_boxes_from_records(pcs_state)
    _reset_pcs_predictions(pcs_state)
    return key



def _pvs_pending_bbox_records(pvs_state):
    records = pvs_state.setdefault("pending_bbox_records", [])
    if records:
        return records

    next_id = int(pvs_state.get("next_pending_bbox_id", 1) or 1)
    rebuilt = []
    for box in pvs_state.get("pending_boxes", []):
        rebuilt.append({"id": next_id, "box": box})
        next_id += 1
    if rebuilt:
        pvs_state["pending_bbox_records"] = rebuilt
        pvs_state["next_pending_bbox_id"] = next_id
    return pvs_state.setdefault("pending_bbox_records", [])


def _sync_pvs_pending_boxes_from_records(pvs_state):
    pvs_state["pending_boxes"] = [rec.get("box") for rec in _pvs_pending_bbox_records(pvs_state)]


def _pvs_pending_bbox_choices(pvs_state):
    choices = []
    for rec in _pvs_pending_bbox_records(pvs_state):
        box = [round(float(v), 1) for v in rec.get("box", [])]
        choices.append((f"ID {rec.get('id')} \u5f85\u751f\u6210 bbox {box}", str(rec.get("id"))))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _append_pvs_pending_bbox(pvs_state, box):
    _pvs_pending_bbox_records(pvs_state)
    bbox_id = int(pvs_state.get("next_pending_bbox_id", 1) or 1)
    pvs_state.setdefault("pending_bbox_records", []).append({"id": bbox_id, "box": box})
    pvs_state["next_pending_bbox_id"] = bbox_id + 1
    _sync_pvs_pending_boxes_from_records(pvs_state)
    return bbox_id


def _clear_pvs_pending_bboxes(pvs_state):
    count = len(_pvs_pending_bbox_records(pvs_state))
    pvs_state["pending_bbox_records"] = []
    pvs_state["pending_boxes"] = []
    return count

def _click_tool_key(click_tool):
    text = str(click_tool or "").strip()
    lower = text.lower()
    if lower in {"point", "bbox", "polygon", "layout"}:
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
        if _is_layout_mask_mode(mode):
            info = "版图 mask 提示分割不使用左侧点击交互；请在版图面板中加载 mask 并更新预览。"
            return prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)
        if _is_pcs_mode(mode) and tool != "bbox":
            tool = "bbox"
        if tool == "point":
            prompt_state["last_point"] = point
            point_payload = _payload_json({"type": "positive_point", "point_xy_px": point, "image_width": w, "image_height": h})
            info = f"\u5df2\u6dfb\u52a0\u6b63\u5411\u70b9: {[round(v, 1) for v in point]}"
        elif tool == "bbox":
            bbox_role = "negative" if _is_pcs_mode(mode) and str(pcs_bbox_kind or "").startswith("Negative") else "positive"
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
                if _is_pcs_mode(mode):
                    key = _append_pcs_bbox_sample(pcs_state, box, bbox_role)
                    prompt_state["last_bbox"] = None
                    label = "\u8d1f\u6837\u672c" if key == "negative_boxes" else "\u6b63\u6837\u672c"
                    info = f"\u5df2\u81ea\u52a8\u6dfb\u52a0 PCS {label} bbox: {[round(v, 1) for v in box]}"
                else:
                    bbox_id = _append_pvs_pending_bbox(pvs_state, box)
                    prompt_state["last_bbox"] = None
                    info = f"\u5df2\u52a0\u5165 PVS \u5f85\u751f\u6210 bbox ID {bbox_id}: {[round(v, 1) for v in box]}\u3002\u7ee7\u7eed\u6846\u9009\u6216\u70b9\u51fb\u201c\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b\u201d\u3002"
        elif tool == "polygon":
            points = prompt_state.setdefault("polygon_points", [])
            points.append(point)
            info = f"\u591a\u8fb9\u5f62\u5df2\u6dfb\u52a0\u7b2c {len(points)} \u4e2a\u9876\u70b9\u3002\u5b8c\u6210\u540e\u70b9\u51fb\u201c\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61\u201d\u3002"
        else:
            info = f"\u672a\u77e5\u4ea4\u4e92\u5de5\u5177: {click_tool}"
    except Exception as exc:
        info = f"\u56fe\u50cf\u70b9\u51fb\u5931\u8d25: {exc}"
    return prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)


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

    if not _is_pvs_manual_mode(mode):
        info = "多边形已完成。当前模式不使用 polygon prompt。"
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
    if _is_pcs_mode(mode):
        pcs_state["text_prompt"] = ""
        pcs_state["positive_boxes"] = []
        pcs_state["negative_boxes"] = []
        pcs_state["bbox_history"] = []
        pcs_state["bbox_records"] = []
        pcs_state["next_bbox_id"] = 1
        text_prompt_update = ""
        info = "PCS prompt 已清空；已有 PCS 分割结果不会被删除"
    elif _is_pvs_manual_mode(mode):
        cleared = _clear_pvs_pending_bboxes(pvs_state)
        text_prompt_update = gr.update()
        info = f"临时提示已清空，包括 {cleared} 个待生成 PVS bbox；已生成实例不会被删除"
    else:
        text_prompt_update = gr.update()
        info = "版图 mask 提示分割的临时点击提示已清空；已生成实例和待生成 bbox 不会被删除"
    return prompt_state, "", "", "", pcs_state, pvs_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), text_prompt_update, *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state)
def _pcs_choice_update(pcs_state):
    choices = [(f"PCS #{i['id']} score={i['score']:.3f}", str(i["id"])) for i in _active_instances(pcs_state)]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _status_label(status):
    return {"draft": "草稿", "accepted": "已确认", "deleted": "已删除"}.get(str(status or "draft"), str(status or "草稿"))


def _pvs_choice_update(pvs_state):
    choices = [(f"PVS #{i['id']} {_status_label(i.get('status'))}", str(i["id"])) for i in _active_instances(pvs_state)]
    active = pvs_state.get("active_instance_id")
    value = str(active) if active is not None and any(c[1] == str(active) for c in choices) else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def _pvs_pending_count_text(pvs_state):
    _sync_pvs_pending_boxes_from_records(pvs_state)
    return f"\u5f85\u751f\u6210 bbox \u6570\u91cf: {len(pvs_state.get('pending_boxes', []))}"


def _pcs_summary(pcs_state):
    _sync_pcs_boxes_from_records(pcs_state)
    lines = [f"正样本 bbox: {len(pcs_state.get('positive_boxes', []))}", f"负样本 bbox: {len(pcs_state.get('negative_boxes', []))}"]
    for rec in _pcs_bbox_records(pcs_state)[:40]:
        role = "负样本" if rec.get("key") == "negative_boxes" else "正样本"
        lines.append(f"ID {rec.get('id')} {role}: {[round(float(v), 1) for v in rec.get('box', [])]}")
    items = _active_instances(pcs_state)
    lines.append(f"PCS 实例: {len(items)}")
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
    if _is_pcs_mode(mode):
        sections.extend(["", "PCS Auto 自动概念分割", _pcs_summary(pcs_state)])
    elif _is_layout_mask_mode(mode):
        sections.extend(["", "版图 mask 提示分割", _pvs_summary(pvs_state)])
    else:
        sections.extend(["", "PVS Manual 手动实例分割", _pvs_summary(pvs_state)])
    return "\n".join(part for part in sections if part is not None)
def _view(image_state, pcs_state, pvs_state, mode, info, prompt_state=None, layout_state=None):
    return (
        _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state, layout_state),
        _result_image(image_state, pcs_state, pvs_state, mode),
        _analysis_report(pcs_state, pvs_state, mode, info),
        _pcs_summary(pcs_state),
        _pvs_summary(pvs_state),
        _pvs_choice_update(pvs_state),
        info,
        _pvs_pending_count_text(pvs_state),
    )


def _init_workspace(input_image, mode, session_state=None):
    pcs_state, pvs_state = _new_pcs_state(), _new_pvs_state()
    prompt_state = _new_prompt_state()
    session_id = _session_id_from_state(session_state)
    image_state = {"image_id": None, "width": 0, "height": 0, "session_id": session_id, "target_image_sha256": None}
    if input_image is None:
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, "Upload an image first", prompt_state), None
    if image_predictor is None:
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, "SAM3 image predictor is not initialized", prompt_state), None
    image = _pil_image(input_image)
    image_id = uuid.uuid4().hex
    target_hash = _layout_tx.image_pixel_sha256(image)
    _clear_workspace_cache()
    try:
        base_state = image_predictor.set_image(image)
    except torch.OutOfMemoryError:
        _clear_workspace_cache()
        info = "\u56fe\u50cf\u52a0\u8f7d\u5931\u8d25\uff1aGPU \u663e\u5b58\u4e0d\u8db3\u3002\u5df2\u6e05\u7406\u5f53\u524d\u5de5\u4f5c\u53f0\u7f13\u5b58\uff0c\u8bf7\u5173\u95ed\u5176\u4ed6 GPU \u4efb\u52a1\u6216\u91cd\u542f demo \u540e\u91cd\u8bd5\u3002"
        return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info, prompt_state), None
    _WORKSPACE_CACHE[image_id] = {"image": image, "base_state": base_state, "target_image_sha256": target_hash}
    image_state = {"image_id": image_id, "width": image.width, "height": image.height, "session_id": session_id, "target_image_sha256": target_hash}
    return image_state, pcs_state, pvs_state, prompt_state, _pcs_bbox_choices(pcs_state), _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, f"Image loaded: {image.width}x{image.height}", prompt_state), None


def _init_workspace_with_layout_editor(input_image, mode, session_state=None, layout_state=None):
    result = _init_workspace(input_image, mode, session_state)
    image_state = result[0]
    if isinstance(layout_state, dict) and layout_state.get("layout_id"):
        editor = _layout_editor_payload(image_state, layout_state, "目标图像已更新，版图编辑器 payload 已刷新。")
    else:
        editor = _layout_editor_empty(image_state, "Image loaded; load or generate a layout mask next.")
    return (*result, editor)


def _delete_selected_pcs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    try:
        records = list(_pcs_bbox_records(pcs_state))
        if not selected_bbox_id:
            raise ValueError("请先在 PCS bbox 列表中选择一个 bbox")
        target_id = int(selected_bbox_id)
        target = next((rec for rec in records if int(rec.get("id", -1)) == target_id), None)
        if target is None:
            raise ValueError("选中的 PCS bbox 已不存在，请重新选择")
        pcs_state["bbox_records"] = [rec for rec in records if int(rec.get("id", -1)) != target_id]
        pcs_state["bbox_history"] = [item for item in pcs_state.get("bbox_history", []) if int(item.get("id", -1)) != target_id]
        _sync_pcs_boxes_from_records(pcs_state)
        _reset_pcs_predictions(pcs_state)
        label = "负样本" if target.get("key") == "negative_boxes" else "正样本"
        info = f"已删除 PCS {label} bbox: {[round(float(v), 1) for v in target.get('box', [])]}；请重新运行 PCS 分割"
    except Exception as exc:
        info = f"删除 PCS bbox 失败: {exc}"
    return pcs_state, _pcs_bbox_choices(pcs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


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
        _pvs_progress(progress, 0.03, "\u51c6\u5907\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b")
        _sync_pvs_pending_boxes_from_records(pvs_state)
        boxes = list(pvs_state.get("pending_boxes", []))
        if not boxes:
            raise ValueError("\u6ca1\u6709\u5f85\u751f\u6210\u7684 PVS bbox\uff0c\u8bf7\u5148\u5728\u56fe\u50cf\u4e0a\u6846\u9009\u4e00\u4e2a\u6216\u591a\u4e2a\u76ee\u6807")

        created_ids = []
        _pvs_progress(progress, 0.12, f"\u8bfb\u53d6\u56fe\u50cf\u7f13\u5b58\uff0c\u5171 {len(boxes)} \u4e2a bbox")
        base_state = _fresh_state(image_state)
        for box_idx, box in enumerate(boxes, start=1):
            start = 0.18 + 0.62 * (box_idx - 1) / max(1, len(boxes))
            _pvs_progress(progress, start, f"SAM3 \u6b63\u5728\u751f\u6210\u7b2c {box_idx}/{len(boxes)} \u4e2a PVS \u5b9e\u4f8b", delay=0.06)
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

        _pvs_progress(progress, 0.86, "\u66f4\u65b0 PVS \u5b9e\u4f8b\u6c60")
        pvs_state["active_instance_id"] = created_ids[-1]
        _clear_pvs_pending_bboxes(pvs_state)
        info = f"\u5df2\u4ece {len(created_ids)} \u4e2a\u5f85\u751f\u6210 bbox \u521b\u5efa PVS \u5b9e\u4f8b: {created_ids}"
        _pvs_progress(progress, 0.96, "\u6e32\u67d3 PVS \u5206\u5272\u7ed3\u679c", delay=0.16)
    except Exception as exc:
        info = f"PVS \u6279\u91cf bbox \u751f\u6210\u5931\u8d25: {exc}"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _delete_selected_pending_pvs_bbox(image_state, pcs_state, pvs_state, mode, selected_bbox_id):
    try:
        records = list(_pvs_pending_bbox_records(pvs_state))
        if not selected_bbox_id:
            raise ValueError("\u8bf7\u5148\u5728 PVS \u5f85\u751f\u6210 bbox \u5217\u8868\u4e2d\u9009\u62e9\u4e00\u4e2a bbox")
        target_id = int(selected_bbox_id)
        target = next((rec for rec in records if int(rec.get("id", -1)) == target_id), None)
        if target is None:
            raise ValueError("\u9009\u4e2d\u7684 PVS \u5f85\u751f\u6210 bbox \u5df2\u4e0d\u5b58\u5728\uff0c\u8bf7\u91cd\u65b0\u9009\u62e9")
        pvs_state["pending_bbox_records"] = [rec for rec in records if int(rec.get("id", -1)) != target_id]
        _sync_pvs_pending_boxes_from_records(pvs_state)
        info = f"\u5df2\u5220\u9664 PVS \u5f85\u751f\u6210 bbox ID {target_id}: {[round(float(v), 1) for v in target.get('box', [])]}"
    except Exception as exc:
        info = f"\u5220\u9664 PVS \u5f85\u751f\u6210 bbox \u5931\u8d25: {exc}"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


def _clear_pending_pvs_boxes(image_state, pcs_state, pvs_state, mode):
    count = _clear_pvs_pending_bboxes(pvs_state)
    info = f"\u5df2\u6e05\u7a7a {count} \u4e2a\u5f85\u751f\u6210 PVS bbox\uff1b\u5df2\u751f\u6210\u5b9e\u4f8b\u4e0d\u4f1a\u88ab\u5220\u9664"
    return pvs_state, _pvs_pending_bbox_choices(pvs_state), *_view(image_state, pcs_state, pvs_state, mode, info)


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
        active_inst = pvs_state.get("instances", {}).get(int(active_id)) if active_id is not None else None
        if active_inst is not None:
            history = active_inst.get("prompt_history", [])
            if history and history[-1].get("op") == "refine_with_layout_mask" and history[-1].get("before"):
                _restore(active_inst, history[-1]["before"])
                history.pop()
                info = f"已撤销 PVS #{active_id} 的版图精修"
                return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)
        items = _active_instances(pvs_state)
        if not items:
            raise ValueError("没有可撤销的 PVS 实例")
        inst = max(items, key=lambda item: int(item["id"]))
        inst["status"] = "deleted"
        if str(pvs_state.get("active_instance_id")) == str(inst["id"]):
            remaining = _active_instances(pvs_state)
            pvs_state["active_instance_id"] = max(remaining, key=lambda item: int(item["id"]))["id"] if remaining else None
        info = f"已撤销上一个 PVS 实例 #{inst['id']}"
    except Exception as exc:
        info = f"撤销上一个实例失败: {exc}"
    return pvs_state, *_view(image_state, pcs_state, pvs_state, mode, info)


def _delete_pvs(image_state, pcs_state, pvs_state, mode):
    try:
        items = _active_instances(pvs_state)
        if not items:
            raise ValueError("没有可清空的 PVS 实例")
        for inst in items:
            pvs_state["instances"][int(inst["id"])]["status"] = "deleted"
        pvs_state["active_instance_id"] = None
        info = f"已清空 {len(items)} 个 PVS 实例；待生成 bbox 不受影响"
    except Exception as exc:
        info = f"清空实例失败: {exc}"
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



def _latest_layout_prompt_from_instances(instances):
    for item in reversed(list(instances or [])):
        for hist in reversed(item.get("prompt_history", []) or []):
            prompt = hist.get("prompt") or {}
            if hist.get("op") in {"create_from_layout_mask", "refine_with_layout_mask"} or prompt.get("type") == "layout_mask":
                return prompt
    return None


def _write_feedback_layout_artifacts(sample_dir, layout_prompt):
    if not layout_prompt:
        return {}
    transform_path = sample_dir / "layout_transform.json"
    layout_id = layout_prompt.get("layout_id")
    cached = None
    if layout_id:
        try:
            cached = _layout_cache_get({"layout_id": layout_id, "session_id": layout_prompt.get("session_id")})
        except Exception:
            cached = None
    transformed_mask_path = None
    if cached is not None and cached.get("transformed_mask") is not None:
        transformed_mask_path = sample_dir / "layout_transformed_mask.png"
        cv2.imwrite(str(transformed_mask_path), np.asarray(cached["transformed_mask"], dtype=np.uint8) * 255)
    payload = {
        "layout_prompt": layout_prompt,
        "layout_id": layout_id,
        "has_cached_transformed_mask": transformed_mask_path is not None,
        "layout_transformed_mask_file": str(transformed_mask_path) if transformed_mask_path is not None else None,
    }
    with transform_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "layout_transform_file": str(transform_path),
        "layout_transformed_mask_file": str(transformed_mask_path) if transformed_mask_path is not None else None,
    }

def _submit_feedback(image_state, pcs_state, pvs_state, mode, rating, feedback_tags, feedback_comment):
    try:
        if _is_pvs_pool_mode(mode):
            active_id = pvs_state.get("active_instance_id")
            if active_id is None:
                raise ValueError("请先选择一个 active PVS instance")
            inst = pvs_state.get("instances", {}).get(int(active_id))
            if inst is None or inst.get("status") == "deleted":
                raise ValueError("当前 active PVS instance 不存在或已删除")
            feedback_instances = [inst]
            feedback_target = "active_pvs_instance"
        elif _is_pcs_mode(mode):
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
        layout_artifacts = _write_feedback_layout_artifacts(sample_dir, _latest_layout_prompt_from_instances(feedback_instances))

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
            "layout_transform_file": layout_artifacts.get("layout_transform_file"),
            "layout_transformed_mask_file": layout_artifacts.get("layout_transformed_mask_file"),
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
        zip_path = _publish_segmentation_zip(export_dir, f"{export_id}.zip")
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
    is_pcs = _is_pcs_mode(mode)
    is_pvs = _is_pvs_manual_mode(mode)
    is_layout = _is_layout_mask_mode(mode)
    if is_pcs:
        tool_update = gr.update(choices=[("框提示 (Box)", "bbox")], value="bbox")
        finish_update = gr.update(visible=False)
    elif is_pvs:
        tool_update = gr.update(
            choices=[("点提示 (Point)", "point"), ("框提示 (Box)", "bbox"), ("多边形Mask (Polygon)", "polygon")],
            value="bbox",
        )
        finish_update = gr.update(visible=True)
    else:
        tool_update = gr.update(choices=[("版图 mask 提示", "layout")], value="layout")
        finish_update = gr.update(visible=False)
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
        gr.update(visible=_is_pvs_pool_mode(mode)),
        gr.update(visible=not is_layout),
        gr.update(visible=is_layout),
        gr.update(visible=is_layout),
        gr.update(visible=is_pvs),
        gr.update(visible=False),
        gr.update(visible=False),
        _pcs_bbox_choices(pcs_state),
        _pvs_pending_bbox_choices(pvs_state),
        *_view(image_state, pcs_state, pvs_state, mode, f"Mode: {mode}，交互提示已重置", prompt_state),
    )
def _switch_mode_with_layout_editor(mode, image_state, pcs_state, pvs_state, layout_state):
    result = _switch_mode(mode, image_state, pcs_state, pvs_state)
    if _is_layout_mask_mode(mode):
        editor = _layout_editor_payload(image_state, layout_state, "已切换到版图 mask 提示分割，Canvas payload 已刷新。")
    else:
        editor = gr.update()
    return (*result, editor)


def _switch_click_tool(click_tool, mode):
    tool = _click_tool_key(click_tool)
    is_pvs = _is_pvs_manual_mode(mode)
    return (
        gr.update(visible=is_pvs and tool == "bbox"),
        gr.update(visible=is_pvs and tool == "point"),
        gr.update(visible=is_pvs and tool == "polygon"),
    )
def _binarize_layout_image(input_image, threshold=12, invert=False, open_kernel=0, close_kernel=0):
    image = _pil_image(input_image)
    if image is None:
        raise ValueError("请先上传版图截图")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    threshold = int(np.clip(int(threshold), 0, 255))
    if _layout_extract_mask is not None:
        mask = _layout_extract_mask(rgb, saturation_min=max(1, threshold), value_min=1, chroma_min=0)
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = hsv[..., 1] >= max(1, threshold)
    if not np.asarray(mask).any():
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray <= max(1, 255 - threshold)
    mask = np.asarray(mask, dtype=bool)
    if invert:
        mask = ~mask
    open_kernel = int(max(0, open_kernel or 0))
    close_kernel = int(max(0, close_kernel or 0))
    work = mask.astype(np.uint8)
    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k, iterations=1)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)
    return image, work.astype(bool)


def _filter_layout_components(mask, min_component_area=0, region_mode="all"):
    mask = np.asarray(mask, dtype=bool)
    min_area = max(0, int(min_component_area or 0))
    region_mode = str(region_mode or "all")
    if not mask.any():
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(bool)
    component_ids = list(range(1, num_labels))
    if min_area > 0:
        component_ids = [idx for idx in component_ids if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area]
    if region_mode == "largest" and component_ids:
        component_ids = [max(component_ids, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))]
    filtered = np.isin(labels, component_ids)
    return filtered.astype(bool)


def _layout_mask_contours(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy_rows = hierarchy[0] if hierarchy is not None else []
    rows = []
    for idx, contour in enumerate(contours):
        if contour.shape[0] < 3:
            continue
        points = contour.reshape(-1, 2).astype(float).tolist()
        x, y, w, h = cv2.boundingRect(contour)
        parent = int(hierarchy_rows[idx][3]) if len(hierarchy_rows) else -1
        rows.append({
            "id": idx + 1,
            "is_hole": parent >= 0,
            "area": float(cv2.contourArea(contour)),
            "bbox_xywh": [float(x), float(y), float(w), float(h)],
            "points": points,
        })
    return rows


def _layout_mask_to_preview(mask):
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_contour_overlay(image, mask, contours):
    base = np.asarray(_pil_image(image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    fill = base.copy()
    fill[mask] = (40, 220, 80)
    vis = cv2.addWeighted(fill, 0.32, base, 0.68, 0)
    for item in contours:
        pts = np.asarray(item.get("points", []), dtype=np.int32).reshape((-1, 1, 2))
        if pts.shape[0] < 3:
            continue
        color = (255, 60, 60) if item.get("is_hole") else (0, 255, 80)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 0), thickness=4)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    return Image.fromarray(vis)


def _save_layout_mask_files(session_state, source_image, mask, contours, params):
    session_id = _session_id_from_state(session_state)
    layout_id = f"layout_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = _layout_disk_dir(session_id, layout_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    image = _pil_image(source_image)
    mask_bool = np.asarray(mask, dtype=bool)
    image_path = out_dir / "source_image.png"
    mask_path = out_dir / "source_mask.png"
    contour_path = out_dir / "contours.json"
    overlay_path = out_dir / "contour_overlay.png"
    meta_path = out_dir / "layout_meta.json"
    image.save(image_path)
    cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)
    overlay = _layout_contour_overlay(image, mask_bool, contours)
    overlay.save(overlay_path)
    payload = {
        "layout_id": layout_id,
        "session_id": session_id,
        "image_size": [int(image.width), int(image.height)],
        "mask_semantics": {"foreground": 1, "background": 0},
        "foreground_pixels": int(mask_bool.sum()),
        "foreground_ratio": float(mask_bool.mean()),
        "binarize_params": params,
        "contours": contours,
    }
    with contour_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cached = _layout_cache_put(
        session_id,
        layout_id,
        image,
        mask_bool,
        contours,
        params,
        mask_path=mask_path,
        contour_json_path=contour_path,
        overlay_path=overlay_path,
        layout_meta_path=meta_path,
    )
    state = _new_layout_state(session_id)
    state.update(
        {
            "layout_id": layout_id,
            "enabled": True,
            "region_mode": str(params.get("region_mode") or "all"),
            "source_width": int(image.width),
            "source_height": int(image.height),
            "pivot_x": float(cached["pivot_xy"][0]),
            "pivot_y": float(cached["pivot_xy"][1]),
            "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
            "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        }
    )
    return state, str(mask_path), str(contour_path), overlay




def _layout_mask_to_editor_image(mask):
    mask = np.asarray(mask, dtype=bool)
    preview = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(preview, mode="L").convert("RGB")


def _layout_region_png_data_url(image):
    if image is None:
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + _sam3_base64.b64encode(buf.getvalue()).decode("ascii")


def _new_layout_region_state():
    return {
        "session_id": None,
        "layout_id": None,
        "source_mask_hash": None,
        "regions_revision": 0,
        "next_region_id": 1,
        "selected_region_id": None,
    }


def _layout_region_state_from_document(document, selected_region_id=None):
    active_ids = {int(region["region_id"]) for region in _layout_regions.active_regions(document)}
    selected = int(selected_region_id) if selected_region_id not in (None, "") else None
    if selected not in active_ids:
        selected = None
    return {
        "session_id": document.get("session_id"),
        "layout_id": document.get("layout_id"),
        "source_mask_hash": document.get("source_mask_hash"),
        "regions_revision": int(document.get("regions_revision") or 0),
        "next_region_id": int(document.get("next_region_id") or 1),
        "selected_region_id": selected,
    }


def _layout_region_editor_empty(status="请先生成版图 binary mask"):
    return {
        "server_view": {
            "enabled": False,
            "source_image": "",
            "source_mask_image": "",
            "saved_region_overlay_image": "",
            "draft_region_overlay_image": "",
            "natural_width": 0,
            "natural_height": 0,
            "regions_revision": 0,
            "selected_region_id": None,
            "regions": [],
            "status": status,
        },
        "client_intent": {
            "tool_mode": "browse",
            "lasso_polygon": [],
            "expected_regions_revision": 0,
            "session_id": "",
            "layout_id": "",
            "source_mask_hash": "",
        },
    }


def _layout_region_identity(layout_state):
    if not isinstance(layout_state, dict):
        raise _layout_regions.RegionValidationError("layout state is missing")
    session_id = _layout_regions.safe_path_component(layout_state.get("session_id"), "session_id")
    layout_id = _layout_regions.safe_path_component(layout_state.get("layout_id"), "layout_id")
    source_mask_hash = str(layout_state.get("source_mask_pixel_sha256") or "")
    if not source_mask_hash:
        raise _layout_regions.RegionValidationError("layout state source mask hash is missing")
    return session_id, layout_id, source_mask_hash


def _layout_region_client_intent(payload):
    raw = payload if isinstance(payload, dict) else {}
    intent = raw.get("client_intent") if isinstance(raw.get("client_intent"), dict) else raw
    tool_mode = intent.get("tool_mode") if intent.get("tool_mode") in {"browse", "lasso"} else "browse"
    polygon = intent.get("lasso_polygon") if isinstance(intent.get("lasso_polygon"), list) else []
    revision = intent.get("expected_regions_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        revision = None
    return {
        "tool_mode": tool_mode,
        "lasso_polygon": polygon,
        "expected_regions_revision": revision,
        "session_id": str(intent.get("session_id") or ""),
        "layout_id": str(intent.get("layout_id") or ""),
        "source_mask_hash": str(intent.get("source_mask_hash") or ""),
    }


def _validate_layout_region_intent(layout_state, intent, *, require_lasso=False):
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    for field, value in {
        "session_id": session_id,
        "layout_id": layout_id,
        "source_mask_hash": source_mask_hash,
    }.items():
        if intent.get(field) != value:
            raise _layout_regions.RegionValidationError(f"Region request {field} does not match current layout")
    if intent.get("expected_regions_revision") is None:
        raise _layout_regions.RegionValidationError("Region request revision is missing")
    if require_lasso and intent.get("tool_mode") != "lasso":
        raise _layout_regions.RegionValidationError("请切换到套索选择工具")
    return session_id, layout_id, source_mask_hash


def _layout_region_source_image(session_id, layout_id, source_mask):
    source_path = runtime_layout_dir / session_id / layout_id / "source_image.png"
    if source_path.exists():
        with Image.open(source_path) as image:
            return image.convert("RGB").copy()
    return _layout_mask_to_preview(source_mask)


def _layout_region_summaries(document):
    return [
        {
            "region_id": int(region["region_id"]),
            "class_label": str(region.get("class_label") or ""),
            "name": str(region.get("name") or ""),
            "area": int(region.get("area") or 0),
        }
        for region in _layout_regions.active_regions(document)
    ]


def _layout_region_choice_update(document, selected_region_id=None):
    choices = []
    active_ids = set()
    for region in _layout_regions.active_regions(document):
        region_id = int(region["region_id"])
        active_ids.add(region_id)
        label = f"R{region_id} {region.get('class_label') or ''}"
        if region.get("name"):
            label += f" | {region['name']}"
        label += f" | area={int(region.get('area') or 0)}"
        choices.append((label, region_id))
    selected = int(selected_region_id) if selected_region_id not in (None, "") else None
    if selected not in active_ids:
        selected = None
    return gr.update(choices=choices, value=selected)


def _layout_region_category_update(value=None):
    categories = _layout_regions.load_layout_categories(current_dir / "layout_categories.json")
    return gr.update(choices=categories, value=value if value in categories else categories[0])


def _layout_region_editor_payload(
    layout_state,
    document,
    source_mask,
    *,
    status,
    selected_region_id=None,
    lasso_polygon=None,
    draft_region_mask=None,
):
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    source_image = _layout_region_source_image(session_id, layout_id, source_mask)
    saved_overlay = _layout_regions.render_saved_region_overlay(
        document.get("regions") or [], source_mask.shape, selected_region_id=selected_region_id
    )
    draft_overlay = (
        _layout_regions.render_draft_region_overlay(draft_region_mask)
        if draft_region_mask is not None
        else None
    )
    return {
        "server_view": {
            "enabled": True,
            "source_image": _data_url(source_image),
            "source_mask_image": _data_url(_layout_mask_to_editor_image(source_mask)),
            "saved_region_overlay_image": _layout_region_png_data_url(saved_overlay),
            "draft_region_overlay_image": _layout_region_png_data_url(draft_overlay),
            "natural_width": int(source_mask.shape[1]),
            "natural_height": int(source_mask.shape[0]),
            "regions_revision": int(document.get("regions_revision") or 0),
            "selected_region_id": selected_region_id,
            "regions": _layout_region_summaries(document),
            "status": status,
        },
        "client_intent": {
            "tool_mode": "lasso",
            "lasso_polygon": copy.deepcopy(lasso_polygon or []),
            "expected_regions_revision": int(document.get("regions_revision") or 0),
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
        },
    }


def _load_layout_region_context(layout_state):
    try:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
        document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        active = _layout_regions.active_regions(document)
        selected = int(active[0]["region_id"]) if active else None
        status = f"Region 标注器已加载：active={len(active)}, revision={document['regions_revision']}"
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            _layout_region_category_update(),
            "",
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"Region 标注器加载失败：{exc}"
        try:
            category_update = _layout_region_category_update()
        except Exception:
            category_update = gr.update(choices=[], value=None)
        return (
            _new_layout_region_state(),
            _layout_region_editor_empty(status),
            category_update,
            "",
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            status,
        )


def _clear_layout_region_context(_layout_state):
    status = "当前版图 Region UI 已清空；磁盘 regions.json 未删除"
    try:
        category_update = _layout_region_category_update()
    except Exception:
        category_update = gr.update(choices=[], value=None)
    return (
        _new_layout_region_state(),
        _layout_region_editor_empty(status),
        category_update,
        "",
        gr.update(choices=[], value=None),
        gr.update(interactive=False),
        gr.update(interactive=False),
        status,
    )


def _preview_layout_region(layout_state, region_state, editor_payload):
    intent = _layout_region_client_intent(editor_payload)
    selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(
            layout_state, intent, require_lasso=True
        )
        region_mask, document = _LAYOUT_REGION_STORE.preview_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            lasso_polygon=intent["lasso_polygon"],
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        status = f"Draft 预览完成：area={int(region_mask.sum())}；选择类别后点击保存区域"
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state,
                document,
                source_mask,
                status=status,
                selected_region_id=selected,
                lasso_polygon=intent["lasso_polygon"],
                draft_region_mask=region_mask,
            ),
            gr.update(interactive=True),
            status,
        )
    except Exception as exc:
        status = f"Draft 预览失败：{exc}"
        try:
            session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
            document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
            state = _layout_region_state_from_document(document, selected)
            editor = _layout_region_editor_payload(
                layout_state,
                document,
                source_mask,
                status=status,
                selected_region_id=state.get("selected_region_id"),
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
        return state, editor, gr.update(interactive=False), status


def _select_layout_region(layout_state, region_state, selected_region_id):
    try:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
        document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        selected = int(selected_region_id) if selected_region_id not in (None, "") else None
        state = _layout_region_state_from_document(document, selected)
        selected = state.get("selected_region_id")
        status = f"已选择 R{selected}" if selected is not None else "未选择活动 Region"
        return (
            state,
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"选择 Region 失败：{exc}"
        return (
            region_state or _new_layout_region_state(),
            _layout_region_editor_empty(status),
            gr.update(interactive=False),
            gr.update(interactive=False),
            status,
        )


def _layout_region_latest_values(layout_state, selected, status, intent=None, keep_draft=False):
    session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
    document, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
    state = _layout_region_state_from_document(document, selected)
    draft_mask = None
    polygon = None
    if (
        keep_draft
        and isinstance(intent, dict)
        and intent.get("expected_regions_revision") == document.get("regions_revision")
        and intent.get("lasso_polygon")
    ):
        draft_mask = _layout_regions.rasterize_uncovered_region_mask(
            source_mask,
            intent["lasso_polygon"],
            document,
        )
        _layout_regions.mask_metadata(draft_mask)
        polygon = intent["lasso_polygon"]
    editor = _layout_region_editor_payload(
        layout_state,
        document,
        source_mask,
        status=status,
        selected_region_id=state.get("selected_region_id"),
        lasso_polygon=polygon,
        draft_region_mask=draft_mask,
    )
    return (
        state,
        editor,
        _layout_region_choice_update(document, state.get("selected_region_id")),
        gr.update(interactive=draft_mask is not None),
        gr.update(interactive=state.get("selected_region_id") is not None),
    )


def _save_layout_region(layout_state, region_state, editor_payload, class_label, name):
    intent = _layout_region_client_intent(editor_payload)
    previous_selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(
            layout_state, intent, require_lasso=True
        )
        document, record = _LAYOUT_REGION_STORE.save_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            lasso_polygon=intent["lasso_polygon"],
            class_label=class_label,
            name=name,
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        selected = int(record["region_id"])
        status = f"已保存 R{selected} {record['class_label']}"
        if record.get("name"):
            status += f" | {record['name']}"
        status += f" | area={record['area']}"
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            _layout_region_category_update(record["class_label"]),
            "",
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=True),
            status,
        )
    except Exception as exc:
        status = f"保存 Region 失败：{exc}"
        try:
            state, editor, choices, save_update, delete_update = _layout_region_latest_values(
                layout_state, previous_selected, status, intent=intent, keep_draft=True
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
            choices = gr.update(choices=[], value=None)
            save_update = gr.update(interactive=False)
            delete_update = gr.update(interactive=False)
        try:
            category_update = _layout_region_category_update(class_label)
        except Exception:
            category_update = gr.update(choices=[], value=None)
        return (
            state,
            editor,
            category_update,
            gr.update(),
            choices,
            save_update,
            delete_update,
            status,
        )


def _delete_layout_region(layout_state, region_state, editor_payload, selected_region_id):
    intent = _layout_region_client_intent(editor_payload)
    previous_selected = (region_state or {}).get("selected_region_id") if isinstance(region_state, dict) else None
    try:
        session_id, layout_id, source_mask_hash = _validate_layout_region_intent(layout_state, intent)
        if selected_region_id in (None, ""):
            raise _layout_regions.RegionValidationError("请先选择活动 Region")
        document, deleted = _LAYOUT_REGION_STORE.delete_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_mask_hash,
            expected_revision=int(intent["expected_regions_revision"]),
            region_id=int(selected_region_id),
        )
        _, source_mask = _LAYOUT_REGION_STORE.load_document(session_id, layout_id, source_mask_hash)
        active = _layout_regions.active_regions(document)
        selected = int(active[0]["region_id"]) if active else None
        status = f"已软删除 R{deleted['region_id']}；binary mask 与 Region RLE 历史均保留"
        return (
            _layout_region_state_from_document(document, selected),
            _layout_region_editor_payload(
                layout_state, document, source_mask, status=status, selected_region_id=selected
            ),
            gr.update(),
            gr.update(),
            _layout_region_choice_update(document, selected),
            gr.update(interactive=False),
            gr.update(interactive=selected is not None),
            status,
        )
    except Exception as exc:
        status = f"软删除 Region 失败：{exc}"
        try:
            state, editor, choices, save_update, delete_update = _layout_region_latest_values(
                layout_state, previous_selected, status
            )
        except Exception:
            state = _new_layout_region_state()
            editor = _layout_region_editor_empty(status)
            choices = gr.update(choices=[], value=None)
            save_update = gr.update(interactive=False)
            delete_update = gr.update(interactive=False)
        return (
            state,
            editor,
            gr.update(),
            gr.update(),
            choices,
            save_update,
            delete_update,
            status,
        )


def _export_layout_regions(layout_state, region_state):
    try:
        session_id, layout_id, source_mask_hash = _layout_region_identity(layout_state)
        state = region_state if isinstance(region_state, dict) else {}
        expected_identity = {
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
        }
        for field, expected in expected_identity.items():
            if state.get(field) != expected:
                raise _layout_regions.RegionValidationError(
                    f"Region export {field} does not match current layout"
                )
        revision = state.get("regions_revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise _layout_regions.RegionValidationError("Region export revision is missing")

        document, source_mask = _LAYOUT_REGION_STORE.load_document(
            session_id,
            layout_id,
            source_mask_hash,
        )
        current_revision = document.get("regions_revision")
        if revision != current_revision:
            raise _layout_regions.StaleRegionsRevisionError(
                f"stale regions revision: expected {revision}, current {current_revision}"
            )

        records = document.get("regions") or []
        active_count = len(_layout_regions.active_regions(document))
        manifest = {
            "schema_version": 1,
            "export_type": "layout_region_annotations",
            "session_id": session_id,
            "layout_id": layout_id,
            "source_mask_hash": source_mask_hash,
            "regions_revision": revision,
            "region_count": len(records),
            "active_region_count": active_count,
            "deleted_region_count": len(records) - active_count,
            "exported_at": _layout_regions.utc_now_iso(),
            "files": ["regions.json", "source_mask.png", "manifest.json"],
        }

        with tempfile.TemporaryDirectory(
            prefix="layout_region_export_",
            dir=runtime_export_dir,
        ) as temporary:
            staging_dir = Path(temporary)
            with (staging_dir / "regions.json").open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            with (staging_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if not cv2.imwrite(
                str(staging_dir / "source_mask.png"),
                np.asarray(source_mask, dtype=np.uint8) * 255,
            ):
                raise OSError("cannot write source_mask.png")
            _prune_public_downloads()
            archive_path = _public_downloads.publish_zip(
                public_download_dir,
                "region_annotation_exports",
                staging_dir,
                f"layout_regions_{layout_id}_r{revision}.zip",
            )

        status = (
            f"\u5df2\u5bfc\u51fa Region \u6807\u6ce8\uff1a"
            f"active={active_count}, revision={revision}"
        )
        return str(archive_path), status
    except Exception as exc:
        return None, f"\u5bfc\u51fa Region \u5931\u8d25\uff1a{exc}"


def _layout_editor_empty(image_state=None, status="请先加载或生成版图 mask"):
    base_url = ""
    target_width = 0
    target_height = 0
    if isinstance(image_state, dict) and image_state.get("image_id"):
        try:
            image = _workspace(image_state)["image"]
            base_url = _data_url(image)
            target_width, target_height = int(image.width), int(image.height)
        except Exception:
            pass
    return {
        "enabled": False,
        "base_image": base_url,
        "mask_image": "",
        "transform": None,
        "target_width": target_width,
        "target_height": target_height,
        "source_width": 0,
        "source_height": 0,
        "foreground_bbox_xyxy": None,
        "status": status,
    }


def _layout_editor_payload(image_state, layout_state, status=None):
    if not layout_state or not layout_state.get("layout_id"):
        return _layout_editor_empty(image_state, status or "请先加载或生成版图 mask")
    try:
        cached = _layout_cache_get(layout_state)
        source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
        if source_mask.ndim != 2:
            raise ValueError("source_mask is not 2D")
        base_url = ""
        target_width = int(cached.get("source_width") or source_mask.shape[1])
        target_height = int(cached.get("source_height") or source_mask.shape[0])
        image_id = layout_state.get("image_id")
        target_hash = cached.get("target_image_sha256")
        if isinstance(image_state, dict) and image_state.get("image_id"):
            image = _workspace(image_state)["image"]
            base_url = _data_url(image)
            target_width, target_height = int(image.width), int(image.height)
            image_id = image_state.get("image_id")
            target_hash = image_state.get("target_image_sha256") or _layout_tx.image_pixel_sha256(image)
        pivot = cached.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(cached.get("foreground_bbox_xyxy"))
        state = dict(layout_state or {})
        if all(k in state and state.get(k) is not None for k in ("center_x", "center_y", "pivot_x", "pivot_y")):
            center_x = float(state.get("center_x"))
            center_y = float(state.get("center_y"))
            pivot_xy = [float(state.get("pivot_x")), float(state.get("pivot_y"))]
        else:
            center_x = float(target_width) / 2.0 + float(state.get("tx") or 0.0)
            center_y = float(target_height) / 2.0 + float(state.get("ty") or 0.0)
            pivot_xy = pivot
        transform = _layout_tx.make_layout_transform_v2(
            session_id=str(state.get("session_id") or cached.get("session_id") or "default"),
            layout_id=str(state.get("layout_id")),
            image_id=str(image_id or ""),
            target_size=(target_width, target_height),
            source_mask=source_mask,
            center_x=center_x,
            center_y=center_y,
            pivot_xy=pivot_xy,
            scale=float(state.get("scale") or 1.0),
            rotation_deg=float(state.get("rotation_deg") or 0.0),
            preview_alpha=float(state.get("preview_alpha") or 0.35),
            revision=int(state.get("revision") or cached.get("committed_revision") or 0),
            source_mask_pixel_sha256=cached.get("source_mask_pixel_sha256"),
            target_image_sha256=target_hash,
        )
        transform = _layout_tx.transform_with_derived_fields(transform, (target_width, target_height))
        return {
            "enabled": bool(state.get("enabled", True)),
            "base_image": base_url,
            "mask_image": _data_url(_layout_mask_to_editor_image(source_mask)),
            "transform": copy.deepcopy(transform),
            "target_width": target_width,
            "target_height": target_height,
            "source_width": int(cached.get("source_width") or source_mask.shape[1]),
            "source_height": int(cached.get("source_height") or source_mask.shape[0]),
            "foreground_bbox_xyxy": copy.deepcopy(cached.get("foreground_bbox_xyxy")),
            "status": status or "版图编辑器已加载：拖动 mask 平移，滚轮缩放，拖动圆形手柄旋转。",
        }
    except Exception as exc:
        return _layout_editor_empty(image_state, status or f"版图编辑器不可用：{exc}")


def _layout_editor_transform(editor_payload):
    if not isinstance(editor_payload, dict):
        return None
    transform = editor_payload.get("transform")
    return transform if isinstance(transform, dict) else None


def _sync_layout_controls_from_editor(layout_state, editor_payload):
    state = dict(layout_state or {})
    transform = _layout_editor_transform(editor_payload)
    if not transform:
        return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "版图编辑器还没有 transform payload"
    try:
        if state.get("layout_id") and transform.get("layout_id") and str(state.get("layout_id")) != str(transform.get("layout_id")):
            raise ValueError("canvas transform belongs to a different layout mask")
        if state.get("session_id") and transform.get("session_id") and str(state.get("session_id")) != str(transform.get("session_id")):
            raise ValueError("canvas transform belongs to a different session")

        payload = editor_payload if isinstance(editor_payload, dict) else {}
        target_w = int(payload.get("target_width") or 0)
        target_h = int(payload.get("target_height") or 0)
        center_x = float(transform.get("center_x", state.get("center_x") or 0.0))
        center_y = float(transform.get("center_y", state.get("center_y") or 0.0))
        if target_w > 0 and target_h > 0:
            tx, ty = _layout_tx.derive_legacy_tx_ty({"center_x": center_x, "center_y": center_y}, (target_w, target_h))
        else:
            tx = float(transform.get("tx", state.get("tx") or 0.0))
            ty = float(transform.get("ty", state.get("ty") or 0.0))

        state.update({
            "enabled": bool(payload.get("enabled", True)),
            "transform_version": 2,
            "image_id": transform.get("image_id") or state.get("image_id"),
            "center_x": center_x,
            "center_y": center_y,
            "pivot_x": float(transform.get("pivot_x", state.get("pivot_x") or 0.0)),
            "pivot_y": float(transform.get("pivot_y", state.get("pivot_y") or 0.0)),
            "scale": float(np.clip(float(transform.get("scale", state.get("scale") or 1.0)), 0.01, 20.0)),
            "rotation_deg": float(_layout_tx.normalize_rotation_deg(float(transform.get("rotation_deg", state.get("rotation_deg") or 0.0)))),
            "preview_alpha": float(np.clip(float(transform.get("preview_alpha", state.get("preview_alpha") or 0.35)), 0.0, 1.0)),
            "revision": int(float(transform.get("revision", state.get("revision") or 0))),
            "source_mask_pixel_sha256": transform.get("source_mask_pixel_sha256") or state.get("source_mask_pixel_sha256"),
            "target_image_sha256": transform.get("target_image_sha256") or state.get("target_image_sha256"),
            "tx": float(tx),
            "ty": float(ty),
        })
        info = f"Canvas 变换已同步到数值控件：tx={tx:.1f}, ty={ty:.1f}, 缩放={state['scale']:.3f}, 旋转={state['rotation_deg']:.1f}"
        return state, bool(state.get("enabled", True)), float(tx), float(ty), float(state.get("scale") or 1.0), float(state.get("rotation_deg") or 0.0), float(state.get("preview_alpha") or 0.35), info
    except Exception as exc:
        return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Canvas 变换同步失败：{exc}"


def _run_layout_mask_page(session_state, image_state, input_image, threshold, invert, open_kernel, close_kernel, min_component_area, region_mode):
    try:
        image, mask = _binarize_layout_image(input_image, threshold, invert, open_kernel, close_kernel)
        mask = _filter_layout_components(mask, min_component_area, region_mode)
        if not mask.any():
            raise ValueError("Binary mask is empty; lower threshold or check invert")
        contours = _layout_mask_contours(mask)
        params = {
            "threshold": int(threshold),
            "invert": bool(invert),
            "open_kernel": int(open_kernel or 0),
            "close_kernel": int(close_kernel or 0),
            "min_component_area": int(min_component_area or 0),
            "region_mode": str(region_mode or "all"),
        }
        state, mask_path, contour_path, overlay = _save_layout_mask_files(session_state, image, mask, contours, params)
        info = (
            f"版图 mask 已生成：{state['layout_id']}\n"
            f"session: {state.get('session_id')}\n"
            f"size: {image.width}x{image.height}\n"
            f"foreground pixels: {int(mask.sum())} ({mask.mean():.4f})\n"
            f"contours: {len(contours)}\n"
            f"source_mask_pixel_sha256: {state.get('source_mask_pixel_sha256')}\n"
            f"mask: {mask_path}\ncontours: {contour_path}"
        )
        editor_payload = _layout_editor_payload(image_state, state, "版图 mask 已生成；切换到版图 mask 提示分割后可拖动、缩放和旋转。")
        return state, editor_payload, image, _layout_mask_to_preview(mask), overlay, mask_path, contour_path, info
    except Exception as exc:
        info = f"版图 mask 生成失败：{exc}"
        state = _new_layout_state(_session_id_from_state(session_state))
        return state, _layout_editor_empty(image_state, info), None, None, None, None, None, info


def _run_layout_mask_page_with_downloads(
    session_state,
    image_state,
    input_image,
    threshold,
    invert,
    open_kernel,
    close_kernel,
    min_component_area,
    region_mode,
):
    result = list(
        _run_layout_mask_page(
            session_state,
            image_state,
            input_image,
            threshold,
            invert,
            open_kernel,
            close_kernel,
            min_component_area,
            region_mode,
        )
    )
    internal_mask_path, internal_contour_path = result[5], result[6]
    if not internal_mask_path or not internal_contour_path:
        return tuple(result)
    try:
        public_mask_path, public_contour_path = _publish_layout_downloads(
            internal_mask_path,
            internal_contour_path,
        )
    except Exception as exc:
        result[5] = None
        result[6] = None
        safe_info = str(result[7]).split("\nmask:", 1)[0]
        result[7] = f"{safe_info}\n\u4e0b\u8f7d\u526f\u672c\u751f\u6210\u5931\u8d25\uff1a{exc}"
        return tuple(result)
    result[5] = public_mask_path
    result[6] = public_contour_path
    result[7] = (
        str(result[7])
        .replace(str(internal_mask_path), public_mask_path)
        .replace(str(internal_contour_path), public_contour_path)
    )
    return tuple(result)


def _save_current_layout_mask(layout_state):
    try:
        cached = _layout_cache_get(layout_state)
        mask_path, contour_path = _publish_layout_downloads(cached.get("mask_path"), cached.get("contour_json_path"))
        return (
            mask_path,
            contour_path,
            f"Saved current layout mask: {layout_state.get('layout_id')}",
        )
    except Exception as exc:
        return None, None, f"保存当前版图 mask 失败：{exc}"


def _clear_current_layout_mask(image_state, layout_state):
    session_id = layout_state.get("session_id") if isinstance(layout_state, dict) else None
    _clear_layout_cache(layout_state)
    state = _new_layout_state(session_id)
    return state, _layout_editor_empty(image_state, "当前版图 mask 已清除"), None, None, None, None, None, "当前版图 mask 已清除"


def _layout_numeric_controls_changed(layout_state, tx, ty, scale, rotation_deg, preview_alpha, tol=1e-6):
    state = layout_state if isinstance(layout_state, dict) else {}

    def changed(field, value, default):
        if value is None:
            return False
        try:
            current = float(value)
            previous = float(state.get(field) if state.get(field) is not None else default)
        except (TypeError, ValueError):
            return False
        return abs(current - previous) > tol

    return any(
        [
            changed("tx", tx, 0.0),
            changed("ty", ty, 0.0),
            changed("scale", scale, 1.0),
            changed("rotation_deg", rotation_deg, 0.0),
            changed("preview_alpha", preview_alpha, 0.35),
        ]
    )


def _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=None, prefer_numeric=None):
    ws = _workspace(image_state)
    image = ws["image"]
    target_size = (int(image.width), int(image.height))
    target_hash = image_state.get("target_image_sha256") or ws.get("target_image_sha256") or _layout_tx.image_pixel_sha256(image)
    state = dict(layout_state or _new_layout_state(image_state.get("session_id")))
    if prefer_numeric is None:
        prefer_numeric = _layout_numeric_controls_changed(state, tx, ty, scale, rotation_deg, preview_alpha)
    incoming = None if prefer_numeric else _layout_editor_transform(transform_payload)
    if incoming:
        state.update({
            "center_x": incoming.get("center_x", state.get("center_x")),
            "center_y": incoming.get("center_y", state.get("center_y")),
            "pivot_x": incoming.get("pivot_x", state.get("pivot_x")),
            "pivot_y": incoming.get("pivot_y", state.get("pivot_y")),
            "scale": incoming.get("scale", state.get("scale")),
            "rotation_deg": incoming.get("rotation_deg", state.get("rotation_deg")),
            "preview_alpha": incoming.get("preview_alpha", state.get("preview_alpha")),
            "revision": incoming.get("revision", state.get("revision")),
        })
    if not state.get("layout_id"):
        raise ValueError("请先加载或生成版图 mask")
    if state.get("session_id") and image_state.get("session_id") and str(state.get("session_id")) != str(image_state.get("session_id")):
        raise ValueError("版图 mask 属于另一个浏览器会话")
    cached = _layout_cache_get(state)
    source_mask = np.asarray(cached.get("source_mask"), dtype=bool)
    if source_mask.ndim != 2:
        raise ValueError("layout source_mask must be a 2D binary mask")
    source_hash = _layout_tx.mask_pixel_sha256(source_mask.astype(np.uint8))
    if cached.get("source_mask_pixel_sha256") and cached.get("source_mask_pixel_sha256") != source_hash:
        raise ValueError("source mask pixel hash mismatch; refusing transform")
    if incoming:
        if incoming.get("session_id") and str(incoming.get("session_id")) != str(state.get("session_id")):
            raise ValueError("frontend transform session_id mismatch")
        if incoming.get("layout_id") and str(incoming.get("layout_id")) != str(state.get("layout_id")):
            raise ValueError("frontend transform layout_id mismatch")
        if incoming.get("image_id") and image_state.get("image_id") and str(incoming.get("image_id")) != str(image_state.get("image_id")):
            raise ValueError("frontend transform image_id mismatch")
        if incoming.get("source_mask_pixel_sha256") and incoming.get("source_mask_pixel_sha256") != source_hash:
            raise ValueError("frontend transform source mask hash mismatch")
        if incoming.get("target_image_sha256") and incoming.get("target_image_sha256") != target_hash:
            raise ValueError("frontend transform target image hash mismatch")
    payload_revision = int(float(state.get("revision") or 0))
    with _LAYOUT_CACHE_LOCK:
        committed = int(cached.get("committed_revision") or 0)
        if payload_revision < committed and cached.get("target_image_sha256") == target_hash:
            raise ValueError(f"layout transform revision is stale: payload={payload_revision}, committed={committed}")
        pivot = cached.get("pivot_xy") or _layout_tx.pivot_from_bbox_xyxy(cached.get("foreground_bbox_xyxy"))
        if incoming:
            pivot_xy = [float(state.get("pivot_x") if state.get("pivot_x") is not None else pivot[0]), float(state.get("pivot_y") if state.get("pivot_y") is not None else pivot[1])]
            center_x = float(state.get("center_x") if state.get("center_x") is not None else target_size[0] / 2.0)
            center_y = float(state.get("center_y") if state.get("center_y") is not None else target_size[1] / 2.0)
        else:
            pivot_xy = pivot
            center_x = target_size[0] / 2.0 + float(tx or 0.0)
            center_y = target_size[1] / 2.0 + float(ty or 0.0)
        if incoming:
            scale_source = state.get("scale") if state.get("scale") is not None else scale
            rotation_source = state.get("rotation_deg") if state.get("rotation_deg") is not None else rotation_deg
            alpha_source = state.get("preview_alpha") if state.get("preview_alpha") is not None else preview_alpha
        else:
            scale_source = scale
            rotation_source = rotation_deg
            alpha_source = preview_alpha
        scale_value = float(np.clip(float(scale_source if scale_source is not None else 1.0), 0.01, 20.0))
        rotation_value = float(rotation_source if rotation_source is not None else 0.0)
        alpha_value = float(np.clip(float(alpha_source if alpha_source is not None else 0.35), 0.0, 1.0))
        revision = max(payload_revision, committed) + 1
        transform = _layout_tx.make_layout_transform_v2(
            session_id=str(state.get("session_id") or cached.get("session_id") or image_state.get("session_id") or "default"),
            layout_id=str(state.get("layout_id")),
            image_id=str(image_state.get("image_id")),
            target_size=target_size,
            source_mask=source_mask,
            center_x=center_x,
            center_y=center_y,
            pivot_xy=pivot_xy,
            scale=scale_value,
            rotation_deg=rotation_value,
            preview_alpha=alpha_value,
            revision=revision,
            source_mask_pixel_sha256=source_hash,
            target_image_sha256=target_hash,
        )
        transform = _layout_tx.transform_with_derived_fields(transform, target_size)
        transformed = _layout_tx.warp_layout_mask(source_mask, transform["matrix_2x3"], target_size)
        cached["target_image_sha256"] = target_hash
        cached["committed_revision"] = revision
        cached["backend_transform"] = copy.deepcopy(transform)
        cached["matrix_2x3"] = copy.deepcopy(transform["matrix_2x3"])
        cached["transformed_mask"] = transformed
        _write_layout_meta(cached)
    state.update(copy.deepcopy(transform))
    state.update({
        "enabled": bool(enabled),
        "region_mode": state.get("region_mode") or cached.get("binarize_params", {}).get("region_mode") or "all",
        "source_width": int(cached.get("source_width") or source_mask.shape[1]),
        "source_height": int(cached.get("source_height") or source_mask.shape[0]),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
    })
    return state, transformed, copy.deepcopy(transform)


def _transform_layout_mask(layout_state, target_width, target_height):
    raise RuntimeError("_transform_layout_mask is deprecated; use _commit_layout_transform(image_state, ...) so target image hash and revision are validated")


def _layout_mask_to_overlay(base_image, mask, alpha=0.35):
    base = np.asarray(_pil_image(base_image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != base.shape[:2]:
        raise ValueError("layout mask and target image sizes do not match")
    fill = base.copy()
    fill[mask] = (0, 255, 130)
    return Image.fromarray(cv2.addWeighted(fill, float(alpha), base, 1.0 - float(alpha), 0))


def _update_layout_preview(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload):
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 overlay 只在版图 mask 提示分割模式可用")
        state, transformed, _ = _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=editor_payload)
        info = (
            "后端 warpAffine 已更新版图预览。\n"
            + _layout_state_summary(state)
            + f"\ntransformed mask: {transformed.shape[1]}x{transformed.shape[0]}, foreground={int(transformed.sum())}"
        )
        workspace = _workspace_image(image_state, pcs_state, pvs_state, mode, prompt_state=None, layout_state=state)
        editor = _layout_editor_payload(image_state, state, "后端权威 overlay 已返回，Canvas 变换已校正。")
        return state, workspace, editor, bool(state.get("enabled")), float(state.get("tx") or 0.0), float(state.get("ty") or 0.0), float(state.get("scale") or 1.0), float(state.get("rotation_deg") or 0.0), float(state.get("preview_alpha") or 0.35), info
    except Exception as exc:
        state = layout_state or _new_layout_state(image_state.get("session_id") if isinstance(image_state, dict) else None)
        return state, gr.update(), _layout_editor_payload(image_state, state, f"版图预览更新失败：{exc}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"版图预览更新失败：{exc}"


def _mask_to_lowres_logits(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("layout mask must be a 2D binary mask")
    target_h, target_w = _prompt_mask_size()
    lowres = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return ((lowres * 2.0 - 1.0) * 10.0).astype(np.float32)


def _validate_layout_prompt_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("layout transformed mask must be 2D")
    foreground = int(mask.sum())
    total = int(mask.size)
    if foreground == 0:
        raise ValueError("layout transformed mask is empty")
    if foreground < 8:
        raise ValueError("layout transformed mask is too small")
    if foreground >= int(total * 0.98):
        raise ValueError("layout transformed mask is almost all foreground; check invert or transform")
    return mask


def _layout_transformed_mask_for_image(image_state, layout_state):
    ws = _workspace(image_state)
    image = ws["image"]
    target_shape = (int(image.height), int(image.width))
    cached = _layout_cache_get(layout_state)
    transformed = cached.get("transformed_mask")
    if transformed is None or np.asarray(transformed).shape != target_shape:
        state, transformed, _ = _commit_layout_transform(
            image_state,
            layout_state,
            layout_state.get("enabled", True),
            layout_state.get("tx", 0.0),
            layout_state.get("ty", 0.0),
            layout_state.get("scale", 1.0),
            layout_state.get("rotation_deg", 0.0),
            layout_state.get("preview_alpha", 0.35),
        )
        layout_state.update(state)
    transformed = np.asarray(transformed, dtype=bool)
    if transformed.shape != target_shape:
        raise ValueError(f"layout transformed mask shape {transformed.shape} does not match target {target_shape}")
    return _validate_layout_prompt_mask(transformed)


def _layout_prompt_metadata(image_state, layout_state):
    cached = _layout_cache_get(layout_state)
    transform = copy.deepcopy(cached.get("backend_transform") or layout_state)
    return {
        "type": "layout_mask",
        "session_id": layout_state.get("session_id"),
        "layout_id": layout_state.get("layout_id"),
        "region_mode": layout_state.get("region_mode"),
        "source_mask_pixel_sha256": cached.get("source_mask_pixel_sha256"),
        "source_mask_file_sha256": cached.get("source_mask_file_sha256"),
        "target_image_sha256": image_state.get("target_image_sha256") if isinstance(image_state, dict) else cached.get("target_image_sha256"),
        "source_width": int(cached.get("source_width") or 0),
        "source_height": int(cached.get("source_height") or 0),
        "target_width": int(image_state.get("width") or 0) if isinstance(image_state, dict) else None,
        "target_height": int(image_state.get("height") or 0) if isinstance(image_state, dict) else None,
        "transform": transform,
        "matrix_2x3": copy.deepcopy(cached.get("matrix_2x3")),
        "revision": int(cached.get("committed_revision") or transform.get("revision") or 0),
        "preview_alpha": float(layout_state.get("preview_alpha") or 0.35),
        "binarize_params": copy.deepcopy(cached.get("binarize_params", {})),
    }


def _create_pvs_from_layout_mask(image_state, pcs_state, pvs_state, mode, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, editor_payload, progress=gr.Progress(track_tqdm=False)):
    try:
        if not _is_layout_mask_mode(mode):
            raise ValueError("版图 mask prompt 只支持在版图 mask 提示分割模式使用")
        _pvs_progress(progress, 0.05, "Commit and validate layout transform")
        state, transformed, _ = _commit_layout_transform(image_state, layout_state, enabled, tx, ty, scale, rotation_deg, preview_alpha, transform_payload=editor_payload)
        transformed = _validate_layout_prompt_mask(transformed)
        lowres_logits = _mask_to_lowres_logits(transformed)
        _pvs_progress(progress, 0.35, "SAM3 is creating PVS instance from layout mask_input", delay=0.08)
        pred = _predict_inst(_fresh_state(image_state), mask_input_lowres_logits=lowres_logits)
        idx = _best(pred)
        mask = pred["masks"][idx]
        inst_id = int(pvs_state.get("next_instance_id", 1))
        prompt = _layout_prompt_metadata(image_state, state)
        pvs_state.setdefault("instances", {})[inst_id] = _make_inst(
            inst_id,
            "manual_pvs_layout_mask",
            mask,
            _mask_box(mask),
            pred["scores"][idx],
            pvs_logits=pred["lowres_logits"][idx],
            history=[{"op": "create_from_layout_mask", "prompt": copy.deepcopy(prompt), "candidate_scores": pred["scores"].astype(float).tolist()}],
        )
        pvs_state["active_instance_id"] = inst_id
        pvs_state["next_instance_id"] = inst_id + 1
        _pvs_progress(progress, 0.96, "Render PVS layout result", delay=0.12)
        info = f"已用版图 mask prompt 创建 PVS #{inst_id}"
    except Exception as exc:
        state = layout_state or _new_layout_state(image_state.get("session_id") if isinstance(image_state, dict) else None)
        info = f"用版图 mask 创建 PVS 实例失败：{exc}"
    editor = _layout_editor_payload(image_state, state, info)
    return pvs_state, state, editor, info, *_view(image_state, pcs_state, pvs_state, mode, info, layout_state=state)


def _layout_state_summary(layout_state):
    if not layout_state or not layout_state.get("layout_id"):
        return "No layout mask selected"
    return (
        f"layout: {layout_state.get('layout_id')}\n"
        f"session: {layout_state.get('session_id')}\n"
        f"enabled: {bool(layout_state.get('enabled'))}\n"
        f"source: {int(layout_state.get('source_width') or 0)}x{int(layout_state.get('source_height') or 0)}\n"
        f"revision={int(layout_state.get('revision') or 0)}, tx={float(layout_state.get('tx') or 0):.1f}, ty={float(layout_state.get('ty') or 0):.1f}, "
        f"scale={float(layout_state.get('scale') or 1):.3f}, rotation={float(layout_state.get('rotation_deg') or 0):.1f}, "
        f"alpha={float(layout_state.get('preview_alpha') or 0.35):.2f}\n"
        f"source_mask_pixel_sha256: {layout_state.get('source_mask_pixel_sha256') or ''}\n"
        f"target_image_sha256: {layout_state.get('target_image_sha256') or ''}"
    )


def _load_layout_binary_mask_png(session_state, image_state, input_image, region_mode="all"):
    try:
        image = _pil_image(input_image)
        if image is None:
            raise ValueError("Upload a binary mask PNG first")
        gray = cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        white_fg = gray >= 128
        black_fg = gray < 128
        candidates = [mask for mask in (white_fg, black_fg) if mask.any()]
        if not candidates:
            raise ValueError("Uploaded binary mask has no foreground pixels")
        mask = min(candidates, key=lambda arr: float(arr.mean()))
        mask = _filter_layout_components(mask, 0, region_mode)
        if not mask.any():
            raise ValueError("Binary mask is empty after filtering")
        contours = _layout_mask_contours(mask)
        params = {"source": "uploaded_binary_mask_png", "region_mode": str(region_mode or "all"), "foreground_rule": "auto_smaller_nonzero"}
        state, _, _, _ = _save_layout_mask_files(session_state, image, mask, contours, params)
        info = f"二值 mask PNG 已载入。\n{_layout_state_summary(state)}"
        return state, _layout_editor_payload(image_state, state, "二值 mask PNG 已载入版图编辑器。"), info
    except Exception as exc:
        state = _new_layout_state(_session_id_from_state(session_state))
        return state, _layout_editor_empty(image_state, f"载入二值 mask PNG 失败：{exc}"), f"载入二值 mask PNG 失败：{exc}"


def _use_current_layout_mask(image_state, layout_state):
    try:
        _layout_cache_get(layout_state)
        info = "Using current saved layout mask.\n" + _layout_state_summary(layout_state)
        return layout_state, _layout_editor_payload(image_state, layout_state, "当前已保存版图 mask 已载入 Canvas。"), info
    except Exception as exc:
        return layout_state or _new_layout_state(), _layout_editor_empty(image_state, f"当前版图 mask 不可用：{exc}"), f"当前版图 mask 不可用：{exc}"


def _reset_layout_controls(image_state, layout_state):
    state = dict(layout_state or _new_layout_state())
    state.update({"enabled": bool(state.get("layout_id")), "tx": 0.0, "ty": 0.0, "scale": 1.0, "rotation_deg": 0.0, "preview_alpha": 0.35})
    if isinstance(image_state, dict) and image_state.get("width") and image_state.get("height"):
        state["center_x"] = float(image_state.get("width")) / 2.0
        state["center_y"] = float(image_state.get("height")) / 2.0
    info = "版图变换控件已重置。\n" + _layout_state_summary(state)
    return state, True if state.get("layout_id") else False, 0.0, 0.0, 1.0, 0.0, 0.35, _layout_editor_payload(image_state, state, "Canvas 变换已重置。"), info

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
    .polygon-finish-btn button {
        width: 100%;
        min-height: 42px;
        font-weight: 700;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }
    """
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    with gr.Blocks(
        theme=theme,
        css=custom_css,
        title="SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0",
        delete_cache=(3600, _PUBLIC_DOWNLOAD_TTL_SECONDS),
    ) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# SAM3 \u4ea4\u4e92\u5f0f\u89c6\u89c9\u5de5\u4f5c\u53f0")
            gr.Markdown("\u57fa\u4e8e SAM3 \u7684 PCS \u81ea\u52a8\u6982\u5ff5\u5206\u5272\u4e0e PVS \u624b\u52a8\u5b9e\u4f8b\u5206\u5272\u5de5\u4f5c\u53f0", elem_classes="description")
            session_state = gr.State(_new_session_state())
            image_state = gr.State({"image_id": None, "width": 0, "height": 0})
            pcs_state = gr.State(_new_pcs_state())
            pvs_state = gr.State(_new_pvs_state())
            prompt_state = gr.State(_new_prompt_state())
            layout_state = gr.State(_new_layout_state())
            layout_region_state = gr.State(_new_layout_region_state())
            bbox_payload = gr.Textbox(label="bbox payload", elem_id="bbox_payload", elem_classes="hidden-payload")
            polygon_payload = gr.Textbox(label="polygon payload", elem_id="polygon_payload", elem_classes="hidden-payload")
            point_payload = gr.Textbox(label="point payload", elem_id="point_payload", elem_classes="hidden-payload")

            with gr.Tabs():
                with gr.TabItem("智能图像分割", id="tab_image"):
                    mode = gr.Radio(
                        choices=[("PCS Auto 自动概念分割", "PCS Auto"), ("PVS Manual 手动实例分割", "PVS Manual"), ("版图 mask 提示分割", "Layout Mask")],
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
                                    pcs_bbox_selector = gr.Dropdown(choices=[], label="PCS bbox \u5217\u8868", interactive=True)
                                    delete_selected_pcs_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d PCS bbox", size="sm", variant="secondary")
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
                                        pvs_pending_count = gr.Markdown("\u5f85\u751f\u6210 bbox \u6570\u91cf: 0")
                                        pvs_pending_bbox_selector = gr.Dropdown(choices=[], label="PVS \u5f85\u751f\u6210 bbox \u5217\u8868", interactive=True)
                                        create_pvs_batch_btn = gr.Button("\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b", variant="primary")
                                        with gr.Row():
                                            delete_selected_pending_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
                                            clear_pending_bbox_btn = gr.Button("\u6e05\u7a7a\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
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
                                        finish_polygon_btn = gr.Button("\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61", variant="primary", elem_classes="polygon-finish-btn")
                                        with gr.Accordion("高级 Polygon 融合方式", open=False):
                                            polygon_combine_mode = gr.Radio(
                                                choices=[("Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b", "replace"), ("Blend \u4e0e\u65e7 mask \u878d\u5408", "blend"), ("Union \u8865\u5145\u533a\u57df", "union"), ("Intersect \u9650\u5236\u8303\u56f4", "intersect")],
                                                value="replace",
                                                label="\u591a\u8fb9\u5f62\u878d\u5408\u65b9\u5f0f",
                                                elem_classes="mode-radio",
                                            )
                                            gr.Markdown(
                                                "**\u4ee5\u4e0a\u56db\u79cd\u90fd\u662f Positive Polygon \u7684\u878d\u5408\u65b9\u5f0f\uff0c\u4e0d\u5305\u542b negative prompt\u3002**  \n"
                                                "- Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b\uff1a\u7528\u5f53\u524d polygon \u4f5c\u4e3a\u5b8c\u6574 mask prompt\u3002  \n"
                                                "- Blend \u4e0e\u65e7 mask \u878d\u5408\uff1a\u65e7 logits \u548c polygon logits \u5171\u540c\u5f71\u54cd\u7ed3\u679c\u3002  \n"
                                                "- Union \u8865\u5145\u533a\u57df\uff1a\u4fdd\u7559\u65e7 mask\uff0c\u5e76\u52a0\u5165 polygon \u533a\u57df\u3002  \n"
                                                "- Intersect \u9650\u5236\u8303\u56f4\uff1a\u5c06\u7ed3\u679c\u9650\u5236\u5728 polygon \u8303\u56f4\u5185\u3002"
                                            )
                                    pvs_summary = gr.Textbox(label="PVS 实例", lines=6, interactive=False, visible=False)

                                with gr.Group(visible=False) as pvs_layout_panel:
                                    gr.Markdown("### PVS 版图 mask 提示")
                                    gr.Markdown("上传或选择二值版图 mask；右侧编辑器支持拖动、滚轮缩放、旋转手柄，并由后端生成最终权威 mask。")
                                    with gr.Row():
                                        use_current_layout_btn = gr.Button("使用当前已保存版图 mask", variant="secondary")
                                        load_layout_binary_btn = gr.Button("载入二值 mask PNG", variant="secondary")
                                    gr.Markdown("#### 直接上传二值 mask PNG")
                                    layout_binary_upload = gr.Image(type="numpy", label="直接上传二值 mask PNG", show_label=False, sources=["upload", "clipboard"])
                                with gr.Accordion("\u5bfc\u51fa\u4e0e COCO \u91cf\u5316", open=False):
                                    coco_dataset = gr.Dropdown(choices=coco_dataset_choices, value=default_coco_dataset, label="\u6307\u6807\u6570\u636e\u96c6")
                                    coco_image_name = gr.Textbox(label="COCO image file_name\uff08\u53ef\u9009\uff09", lines=1)
                                    coco_split = gr.Radio(choices=["auto", "val", "train", "test"], value="auto", label="\u6807\u6ce8 split")
                                    coco_eval_scope = gr.Radio(choices=[coco_eval_scope_overlap, coco_eval_scope_full], value=coco_eval_scope_overlap, label="\u8bc4\u4f30\u8303\u56f4")
                                    annotation_json_file = gr.File(label="\u4e0a\u4f20 O3/LabelMe-like JSON \u6807\u6ce8\uff08\u4f18\u5148\u4e8e COCO lookup\uff09", file_types=[".json"], type="filepath")

                        with gr.Column(scale=1):
                            gr.Markdown("### \u5206\u5272\u7ed3\u679c")
                            result_image = gr.Image(type="numpy", label="\u5206\u5272\u7ed3\u679c", show_label=False)
                            with gr.Group(visible=True) as analysis_report_panel:
                                analysis_report = gr.Textbox(label="分析报告", interactive=False, lines=18)
                            with gr.Group(visible=False) as layout_transform_panel:
                                gr.Markdown("### 修改变形版图")
                                if LayoutTransformEditor is not None:
                                    layout_editor = LayoutTransformEditor(value=_layout_editor_empty(), label="\u7248\u56fe\u4ea4\u4e92\u7f16\u8f91\u5668", show_label=False, height=520, elem_id="layout_transform_editor")
                                else:
                                    gr.Markdown(f"版图 Canvas 编辑器组件不可用；仍可使用数值控件。错误：{_layout_editor_import_error}")
                                    layout_editor = gr.JSON(value=_layout_editor_empty(), label="layout transform payload", visible=False)
                                layout_enabled = gr.Checkbox(value=False, label="显示/启用版图 overlay")
                                with gr.Row():
                                    layout_tx = gr.Number(value=0.0, label="水平偏移 tx")
                                    layout_ty = gr.Number(value=0.0, label="垂直偏移 ty")
                                with gr.Row():
                                    layout_scale = gr.Slider(minimum=0.1, maximum=20.0, value=1.0, step=0.01, label="缩放 scale")
                                    layout_rotation = gr.Slider(minimum=-180.0, maximum=180.0, value=0.0, step=1.0, label="旋转 rotation")
                                layout_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.35, step=0.05, label="透明度 alpha")
                                with gr.Row():
                                    reset_layout_btn = gr.Button("重置", variant="secondary")
                                    update_layout_preview_btn = gr.Button("更新预览", variant="primary")
                                create_from_layout_btn = gr.Button("用版图创建实例", variant="primary")
                                layout_pvs_info = gr.Textbox(label="版图提示状态", lines=5, interactive=False)
                            with gr.Group(visible=True) as pvs_action_panel:
                                gr.Markdown("### PVS 实例操作")
                                active_pvs = gr.Dropdown(choices=[], label="\u5f53\u524d PVS \u5b9e\u4f8b")
                                with gr.Row():
                                    undo_pvs_btn = gr.Button("撤销上一个实例")
                                    delete_pvs_btn = gr.Button("清空实例")
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

                with gr.TabItem("版图截图转掩码", id="tab_layout_mask"):
                    gr.Markdown("### 版图截图转二值 mask")
                    gr.Markdown("binary mask 是唯一权威数据；contour 仅用于预览和导出。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 上传版图截图")
                            layout_input = gr.Image(type="numpy", label="上传版图截图", show_label=False, sources=["upload", "clipboard"])
                            layout_threshold = gr.Slider(minimum=0, maximum=255, value=12, step=1, label="threshold（色彩/饱和度阈值）")
                            layout_invert = gr.Checkbox(value=False, label="invert（反转前景/背景）")
                            with gr.Row():
                                layout_open_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="open kernel")
                                layout_close_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="close kernel")
                            layout_min_area = gr.Number(value=0, precision=0, label="min component area")
                            layout_region_mode = gr.Radio(
                                choices=[("全部区域", "all"), ("最大连通区域", "largest")],
                                value="all",
                                label="区域模式",
                                elem_classes="mode-radio",
                            )
                            run_layout_mask_btn = gr.Button("生成并保存当前版图 mask", variant="primary")
                            save_layout_mask_btn = gr.Button("保存为当前版图 mask", variant="secondary")
                            clear_layout_mask_btn = gr.Button("清除当前版图", variant="secondary")
                            layout_info = gr.Textbox(label="处理信息", lines=8, interactive=False)
                            with gr.Row():
                                layout_mask_file = gr.File(label="下载 mask PNG", interactive=False)
                                layout_contour_file = gr.File(label="下载 contour JSON", interactive=False)
                        with gr.Column(scale=1):
                            layout_source_preview = gr.Image(type="pil", label="原图预览", show_label=False, visible=False)
                            gr.Markdown("#### binary mask 预览")
                            layout_mask_preview = gr.Image(type="pil", label="binary mask 预览", show_label=False)
                            gr.Markdown("#### contour overlay")
                            layout_overlay_preview = gr.Image(type="pil", label="contour overlay", show_label=False)
                            gr.Markdown("### Region Annotation Layer")
                            gr.Markdown("黄色表示未保存 Draft；绿色表示已保存 Region。Region 标注不会进入 PCS/PVS prompt。")
                            if LayoutRegionAnnotator is not None:
                                layout_region_annotator = LayoutRegionAnnotator(
                                    value=_layout_region_editor_empty(),
                                    label="版图 Region 套索标注器",
                                    show_label=False,
                                    height=520,
                                    elem_id="layout_region_annotator",
                                )
                            else:
                                gr.Markdown(f"Region 套索组件不可用。错误：{_layout_region_annotator_import_error}")
                                layout_region_annotator = gr.JSON(
                                    value=_layout_region_editor_empty(),
                                    label="layout Region payload",
                                    visible=False,
                                )
                            with gr.Row():
                                layout_region_category = gr.Dropdown(
                                    choices=_layout_regions.load_layout_categories(current_dir / "layout_categories.json"),
                                    value=_layout_regions.load_layout_categories(current_dir / "layout_categories.json")[0],
                                    label="类别（必填）",
                                    allow_custom_value=False,
                                    interactive=True,
                                )
                                layout_region_name = gr.Textbox(
                                    label="区域/模块名称（可选）",
                                    placeholder="例如：M1_power",
                                    max_lines=1,
                                )
                            save_layout_region_btn = gr.Button("保存当前 Draft Region", variant="primary", interactive=False)
                            layout_region_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="活动 Region",
                                interactive=True,
                            )
                            delete_layout_region_btn = gr.Button("软删除选中 Region", variant="secondary", interactive=False)
                            layout_region_status = gr.Textbox(label="Region 状态", lines=4, interactive=False)
                            export_layout_regions_btn = gr.Button(
                                "\u5bfc\u51fa\u5f53\u524d Region \u6807\u6ce8",
                                variant="secondary",
                            )
                            layout_region_export_file = gr.File(
                                label="\u4e0b\u8f7d Region \u6807\u6ce8\u5305",
                                interactive=False,
                            )
                with gr.TabItem("\u89c6\u9891\u76ee\u6807\u8ddf\u8e2a", id="tab_video"):
                    gr.Markdown("\u5f53\u524d PVS demo \u5206\u652f\u805a\u7126\u56fe\u50cf\u5206\u5272\uff1b\u89c6\u9891\u76ee\u6807\u8ddf\u8e2a\u8bf7\u4f7f\u7528\u539f\u59cb demo \u5206\u652f\u3002")

            run_layout_mask_event = run_layout_mask_btn.click(
                fn=_run_layout_mask_page_with_downloads,
                inputs=[session_state, image_state, layout_input, layout_threshold, layout_invert, layout_open_kernel, layout_close_kernel, layout_min_area, layout_region_mode],
                outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
                api_name="_run_layout_mask_page",
            )
            run_layout_mask_event.then(
                fn=_load_layout_region_context,
                inputs=[layout_state],
                outputs=[layout_region_state, layout_region_annotator, layout_region_category, layout_region_name, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            save_layout_mask_btn.click(
                fn=_save_current_layout_mask,
                inputs=[layout_state],
                outputs=[layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
            )
            clear_layout_mask_event = clear_layout_mask_btn.click(
                fn=_clear_current_layout_mask,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
                concurrency_limit=1,
            )
            clear_layout_mask_event.then(
                fn=_clear_layout_region_context,
                inputs=[layout_state],
                outputs=[layout_region_state, layout_region_annotator, layout_region_category, layout_region_name, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            layout_region_annotator.input(
                fn=_preview_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator],
                outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            save_layout_region_btn.click(
                fn=_save_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_category, layout_region_name],
                outputs=[layout_region_state, layout_region_annotator, layout_region_category, layout_region_name, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            layout_region_selector.input(
                fn=_select_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_selector],
                outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )
            delete_layout_region_btn.click(
                fn=_delete_layout_region,
                inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_selector],
                outputs=[layout_region_state, layout_region_annotator, layout_region_category, layout_region_name, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
                concurrency_limit=1,
            )

            export_layout_regions_btn.click(
                fn=_export_layout_regions,
                inputs=[layout_state, layout_region_state],
                outputs=[layout_region_export_file, layout_region_status],
                concurrency_limit=1,
            )

            common = [image_upload, result_image, analysis_report, pcs_summary, pvs_summary, active_pvs, interaction_info, pvs_pending_count]
            use_current_layout_btn.click(
                fn=_use_current_layout_mask,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_editor, layout_pvs_info],
                concurrency_limit=1,
            )
            load_layout_binary_btn.click(
                fn=_load_layout_binary_mask_png,
                inputs=[session_state, image_state, layout_binary_upload, layout_region_mode],
                outputs=[layout_state, layout_editor, layout_pvs_info],
                concurrency_limit=1,
            )
            update_layout_preview_btn.click(
                fn=_update_layout_preview,
                inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor],
                outputs=[layout_state, image_upload, layout_editor, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
                concurrency_limit=1,
            )
            layout_editor.change(
                fn=_sync_layout_controls_from_editor,
                inputs=[layout_state, layout_editor],
                outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
                concurrency_limit=1,
            )
            reset_layout_btn.click(
                fn=_reset_layout_controls,
                inputs=[image_state, layout_state],
                outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_pvs_info],
                concurrency_limit=1,
            )
            create_from_layout_btn.click(
                fn=_create_pvs_from_layout_mask,
                inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor],
                outputs=[pvs_state, layout_state, layout_editor, layout_pvs_info, *common],
                show_progress_on=[result_image],
                concurrency_limit=1,
            )

            image_upload.upload(fn=_init_workspace_with_layout_editor, inputs=[image_upload, mode, session_state, layout_state], outputs=[image_state, pcs_state, pvs_state, prompt_state, pcs_bbox_selector, pvs_pending_bbox_selector, *common, export_file, layout_editor], concurrency_limit=1)
            image_upload.select(fn=_workspace_select, inputs=[image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, *common], concurrency_limit=1)
            finish_polygon_btn.click(fn=_finish_native_polygon, inputs=[image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode], outputs=[prompt_state, polygon_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1)
            clear_prompt_btn.click(fn=_clear_prompt_selection, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, text_prompt, *common], concurrency_limit=1)
            mode.change(fn=_switch_mode_with_layout_editor, inputs=[mode, image_state, pcs_state, pvs_state, layout_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, click_tool, finish_polygon_btn, pcs_bbox_tools, pcs_panel, pvs_panel, pvs_action_panel, analysis_report_panel, pvs_layout_panel, layout_transform_panel, pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel, pcs_bbox_selector, pvs_pending_bbox_selector, *common, layout_editor], concurrency_limit=1)
            click_tool.change(fn=_switch_click_tool, inputs=[click_tool, mode], outputs=[pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel], concurrency_limit=1)
            delete_selected_pcs_bbox_btn.click(fn=_delete_selected_pcs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pcs_bbox_selector], outputs=[pcs_state, pcs_bbox_selector, *common], concurrency_limit=1)
            run_pcs_btn.click(fn=_run_pcs, inputs=[image_state, pcs_state, pvs_state, mode, text_prompt, confidence_threshold], outputs=[pcs_state, *common], concurrency_limit=1)
            create_pvs_batch_btn.click(fn=_create_pvs_from_pending_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1)
            delete_selected_pending_bbox_btn.click(fn=_delete_selected_pending_pvs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pvs_pending_bbox_selector], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1)
            clear_pending_bbox_btn.click(fn=_clear_pending_pvs_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1)
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
    _public_downloads.ensure_public_download_dirs(public_download_dir)
    _prune_public_downloads()
    demo = create_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=False,
        allowed_paths=_gradio_allowed_paths(),
        blocked_paths=_gradio_blocked_paths(),
    )


if __name__ == "__main__":
    main()
