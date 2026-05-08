#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
基于 SAM3 的交互式图像分割与视频跟踪系统
"""

import os
import sys
import time
import io
import logging
from pathlib import Path
import tempfile
import json

# 所有运行时文件固定在 /data/zhengqiyuan，避免 Gradio 默认写入 /tmp/gradio。
current_dir = Path(__file__).resolve().parent
runtime_dir = current_dir / ".runtime"
runtime_tmp_dir = runtime_dir / "tmp"
runtime_gradio_dir = runtime_dir / "gradio"
runtime_video_dir = runtime_dir / "videos"
runtime_log_dir = runtime_dir / "logs"
qiyuan_cache_dir = Path("/data/zhengqiyuan/.cache")

for path in (
    runtime_tmp_dir,
    runtime_gradio_dir,
    runtime_video_dir,
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


def handle_image_click(
    img,
    original_img,
    evt: gr.SelectData,
    mode,
    prompt_polarity,
    pos_points,
    neg_points,
    pos_boxes,
    neg_boxes,
    pos_polygons,
    neg_polygons,
    click_state,
):
    """处理图像点击事件，提供实时视觉反馈"""
    if img is None:
        return (
            img,
            pos_points,
            neg_points,
            pos_boxes,
            neg_boxes,
            pos_polygons,
            neg_polygons,
            click_state,
            "请先上传图像",
        )

    # 如果没有原始图像，就使用当前图像作为原始图像
    if original_img is None:
        original_img = img.copy()

    # 基于当前显示的图像进行绘制
    vis_img = img.copy()

    x, y = evt.index
    x, y = int(x), int(y)

    info_msg = ""
    is_positive = prompt_polarity.startswith("✅")

    if mode == "📍 点提示 (Point)":
        new_point = f"{x},{y}"
        if is_positive:
            if pos_points:
                pos_points += f";{new_point}"
            else:
                pos_points = new_point
        else:
            if neg_points:
                neg_points += f";{new_point}"
            else:
                neg_points = new_point

        if is_positive:
            cv2.circle(vis_img, (x, y), 6, (0, 255, 0), -1)
        else:
            cv2.circle(vis_img, (x, y), 6, (255, 0, 0), -1)
        cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1)

        info_msg = f"{prompt_polarity} 已添加点: {new_point}"
        return (
            vis_img,
            pos_points,
            neg_points,
            pos_boxes,
            neg_boxes,
            pos_polygons,
            neg_polygons,
            None,
            info_msg,
        )

    elif mode == "🔲 框提示 (Box)":
        if click_state is None:
            click_state = {"start": [x, y], "label": is_positive}
            cv2.circle(vis_img, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1)
            info_msg = f"{prompt_polarity} 已记录起点: {x},{y}，请点击对角点完成框选"
            return (
                vis_img,
                pos_points,
                neg_points,
                pos_boxes,
                neg_boxes,
                pos_polygons,
                neg_polygons,
                click_state,
                info_msg,
            )
        else:
            if isinstance(click_state, dict):
                x1, y1 = click_state.get("start", [x, y])
                box_is_positive = bool(click_state.get("label", is_positive))
            else:
                x1, y1 = click_state
                box_is_positive = is_positive
            x2, y2 = x, y

            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)

            # 确保框有大小
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

            new_box = f"{xmin},{ymin},{xmax},{ymax}"
            if box_is_positive:
                if pos_boxes:
                    pos_boxes += f";{new_box}"
                else:
                    pos_boxes = new_box
            else:
                if neg_boxes:
                    neg_boxes += f";{new_box}"
                else:
                    neg_boxes = new_box

            if box_is_positive:
                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            else:
                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

            info_msg = f"{'✅ Positive' if box_is_positive else '❌ Negative'} 已添加框: {new_box}"
            return (
                vis_img,
                pos_points,
                neg_points,
                pos_boxes,
                neg_boxes,
                pos_polygons,
                neg_polygons,
                None,
                info_msg,
            )

    elif mode == "✏️ 多边形Mask (Polygon)":
        if not isinstance(click_state, dict) or click_state.get("type") != "polygon":
            click_state = {"type": "polygon", "points": [], "label": is_positive}

        polygon_is_positive = bool(click_state.get("label", is_positive))
        points = click_state.setdefault("points", [])
        points.append([x, y])

        point_color = (0, 255, 0) if polygon_is_positive else (255, 0, 0)
        for px, py in points:
            cv2.circle(vis_img, (int(px), int(py)), 5, point_color, -1)
            cv2.circle(vis_img, (int(px), int(py)), 5, (255, 255, 255), 1)
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=False, color=point_color, thickness=2)

        info_msg = (
            f"{'✅ Positive' if polygon_is_positive else '❌ Negative'} "
            f"多边形已添加 {len(points)} 个顶点，点击“完成多边形对象”闭合"
        )
        return (
            vis_img,
            pos_points,
            neg_points,
            pos_boxes,
            neg_boxes,
            pos_polygons,
            neg_polygons,
            click_state,
            info_msg,
        )

    return (
        vis_img,
        pos_points,
        neg_points,
        pos_boxes,
        neg_boxes,
        pos_polygons,
        neg_polygons,
        click_state,
        info_msg,
    )


def finish_polygon(img, pos_polygons, neg_polygons, click_state):
    """Close the in-progress polygon and store it as an object mask prompt."""
    if img is None:
        return img, pos_polygons, neg_polygons, click_state, "请先上传图像"
    if not isinstance(click_state, dict) or click_state.get("type") != "polygon":
        return img, pos_polygons, neg_polygons, click_state, "当前没有正在绘制的多边形"

    points = click_state.get("points", [])
    if len(points) < 3:
        return img, pos_polygons, neg_polygons, click_state, "多边形至少需要 3 个顶点"

    polygon_is_positive = bool(click_state.get("label", True))
    if polygon_is_positive:
        pos_polygons = append_polygon_prompt(pos_polygons, points)
    else:
        neg_polygons = append_polygon_prompt(neg_polygons, points)

    vis_img = img.copy()
    color = (0, 255, 0) if polygon_is_positive else (255, 0, 0)
    draw_polygons(vis_img, [points], color)
    info_msg = (
        f"{'✅ Positive' if polygon_is_positive else '❌ Negative'} "
        f"已完成多边形对象，共 {len(points)} 个顶点"
    )
    return vis_img, pos_polygons, neg_polygons, None, info_msg


def segment_image(
    input_image,
    text_prompt,
    confidence_threshold,
    pos_point_prompt,
    neg_point_prompt,
    pos_box_prompt,
    neg_box_prompt,
    pos_polygon_prompt,
    neg_polygon_prompt,
    original_image=None,
    progress=gr.Progress(),
):
    """图像分割功能"""
    # 优先使用原始图像，如果不存在则使用输入图像
    image_to_process = original_image if original_image is not None else input_image

    if image_to_process is None:
        return None, "请上传图像"

    if (
        not text_prompt
        and not pos_point_prompt
        and not neg_point_prompt
        and not pos_box_prompt
        and not neg_box_prompt
        and not pos_polygon_prompt
        and not neg_polygon_prompt
    ):
        return None, "请提供至少一种提示（文本、点、框或多边形）"

    try:
        if image_predictor is None:
            return None, "模型未初始化，请检查模型文件"

        start_time = time.time()
        progress(0.1, desc="正在加载图像...")

        # 转换图像格式
        if isinstance(image_to_process, np.ndarray):
            image = Image.fromarray(image_to_process)
        else:
            image = image_to_process

        # 设置图像
        state = image_predictor.set_image(image)
        progress(0.3, desc="解析提示信息...")

        # 处理文本提示
        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)

        width, height = image.size

        def parse_points(points_str):
            parsed = []
            if not points_str:
                return parsed
            for point_str in points_str.split(";"):
                if point_str:
                    try:
                        x, y = map(float, point_str.split(","))
                        parsed.append([x, y])
                    except ValueError:
                        continue
            return parsed

        def parse_boxes(boxes_str):
            parsed = []
            if not boxes_str:
                return parsed
            for box_str in boxes_str.split(";"):
                if box_str:
                    try:
                        x1, y1, x2, y2 = map(float, box_str.split(","))
                        parsed.append([x1, y1, x2, y2])
                    except ValueError:
                        continue
            return parsed

        def apply_points(points, label, current_state):
            if not points:
                return current_state
            box_size = min(width, height) * 0.05
            box_width = box_size / width
            box_height = box_size / height
            for x, y in points:
                cx = x / width
                cy = y / height
                box = [cx, cy, box_width, box_height]
                current_state = image_predictor.add_geometric_prompt(
                    box, label, current_state
                )
            return current_state

        def apply_boxes(boxes, label, current_state):
            if not boxes:
                return current_state
            for x1, y1, x2, y2 in boxes:
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                box = [center_x, center_y, box_width, box_height]
                current_state = image_predictor.add_geometric_prompt(
                    box, label, current_state
                )
            return current_state

        pos_points = parse_points(pos_point_prompt)
        neg_points = parse_points(neg_point_prompt)
        pos_boxes = parse_boxes(pos_box_prompt)
        neg_boxes = parse_boxes(neg_box_prompt)
        pos_polygons = parse_polygon_prompt(pos_polygon_prompt)
        neg_polygons = parse_polygon_prompt(neg_polygon_prompt)

        logger.info(
            json.dumps(
                {
                    "type": "image_segmentation",
                    "device": DEVICE,
                    "image_size": [width, height],
                    "confidence_threshold": confidence_threshold,
                    "text_prompt": text_prompt or "",
                    "pos_points_count": len(pos_points),
                    "neg_points_count": len(neg_points),
                    "pos_boxes_count": len(pos_boxes),
                    "neg_boxes_count": len(neg_boxes),
                    "pos_polygons_count": len(pos_polygons),
                    "neg_polygons_count": len(neg_polygons),
                    "pos_points": pos_points,
                    "neg_points": neg_points,
                    "pos_boxes_xyxy": pos_boxes,
                    "neg_boxes_xyxy": neg_boxes,
                    "pos_polygons": pos_polygons,
                    "neg_polygons": neg_polygons,
                },
                ensure_ascii=False,
            )
        )

        result_states = []
        has_classic_prompts = bool(
            text_prompt or pos_points or neg_points or pos_boxes or neg_boxes
        )

        state = apply_points(pos_points, True, state)
        state = apply_points(neg_points, False, state)

        state = apply_boxes(pos_boxes, True, state)
        state = apply_boxes(neg_boxes, False, state)

        # 设置置信度阈值
        state = image_predictor.set_confidence_threshold(confidence_threshold, state)
        if has_classic_prompts and "boxes" in state and len(state["boxes"]) > 0:
            result_states.append(state)

        if pos_polygons:
            progress(0.6, desc="处理多边形 mask prompt...")
            negative_mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in neg_polygons:
                negative_mask |= polygon_to_mask(polygon, height, width)

            for polygon in pos_polygons:
                polygon_mask = polygon_to_mask(polygon, height, width)
                if negative_mask.any():
                    polygon_mask[negative_mask > 0] = 0
                if not polygon_mask.any():
                    continue
                polygon_state = image_predictor.predict_mask_prompt(
                    polygon_mask, state.copy()
                )
                if "masks" in polygon_state and len(polygon_state["masks"]) > 0:
                    if negative_mask.any():
                        polygon_state["masks"] = polygon_state["masks"].clone()
                        polygon_state["masks"][:, :, negative_mask > 0] = False
                        polygon_state["masks_logits"] = polygon_state["masks"].float()
                    result_states.append(polygon_state)

        if result_states:
            state["boxes"] = torch.cat([s["boxes"] for s in result_states], dim=0)
            state["masks"] = torch.cat([s["masks"] for s in result_states], dim=0)
            state["masks_logits"] = torch.cat(
                [s["masks_logits"] for s in result_states], dim=0
            )
            state["scores"] = torch.cat([s["scores"] for s in result_states], dim=0)

        progress(0.7, desc="模型推理中...")

        # 获取结果
        if "boxes" in state and len(state["boxes"]) > 0:
            # 可视化结果
            import matplotlib.pyplot as plt

            # 使用官方的 plot_results 接口进行绘制
            # plot_results 内部会创建 figure 并绘制 masks, boxes, scores
            # 注意：它会打印找到的对象数量，但这不影响 Gradio 显示
            plot_results(image, state)

            # 获取当前的 figure (由 plot_results 创建) 并转换为 PIL 图像
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            result_image = Image.open(buf)
            plt.close()  # 关闭 figure 释放内存

            processing_time = time.time() - start_time
            info = f"✨ 处理完成 | 耗时: {processing_time:.2f}s | 检测到 {len(state['boxes'])} 个目标"

            return result_image, info
        else:
            return image, "⚠️ 未检测到任何对象，请尝试调整提示或降低置信度阈值"

    except Exception as e:
        return None, f"❌ 处理失败: {str(e)}"


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


def create_demo():
    """创建美化后的Gradio演示界面"""

    # 自定义CSS
    custom_css = """
    .container { max-width: 1200px; margin: auto; padding-top: 20px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    
    /* 让Radio按钮组水平撑满 */
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    )

    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 交互式视觉工作台") as demo:

        with gr.Column(elem_classes="container"):
            gr.Markdown("# 👁️ SAM3 交互式视觉工作台")
            gr.Markdown(
                "基于 SAM3 的下一代图像分割与视频跟踪系统", elem_classes="description"
            )

            with gr.Tabs():
                # ================= 图像分割标签页 =================
                with gr.TabItem("🖼️ 智能图像分割", id="tab_image"):
                    with gr.Row():
                        # 左侧控制栏
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="numpy",
                                label="原始图像 (点击进行交互)",
                                elem_id="input_image",
                            )

                            # 存储原始图像状态
                            original_image_state = gr.State(None)
                            click_state = gr.State(None)

                            with gr.Group():
                                gr.Markdown("### 🎮 交互模式")
                                # 第一行：模式选择
                                interaction_mode = gr.Radio(
                                    choices=[
                                        "📍 点提示 (Point)",
                                        "🔲 框提示 (Box)",
                                        "✏️ 多边形Mask (Polygon)",
                                    ],
                                    value="📍 点提示 (Point)",
                                    label="选择模式",
                                    show_label=False,
                                    elem_classes="mode-radio",
                                )
                                prompt_polarity = gr.Radio(
                                    choices=["✅ Positive", "❌ Negative"],
                                    value="✅ Positive",
                                    label="提示极性",
                                    show_label=False,
                                    elem_classes="mode-radio",
                                )
                                # 第二行：清空按钮（全宽）
                                with gr.Row():
                                    finish_polygon_btn = gr.Button(
                                        "✅ 完成多边形对象",
                                        size="sm",
                                        variant="secondary",
                                    )
                                    clear_prompts_btn = gr.Button(
                                        "🗑️ 清空提示 (Clear Prompts)",
                                        size="sm",
                                        variant="secondary",
                                    )

                                interaction_info = gr.Markdown(
                                    "👆 点击图像开始添加提示...",
                                    elem_id="interaction-info",
                                )

                            with gr.Accordion("📝 高级提示选项", open=True):
                                text_prompt = gr.Textbox(
                                    label="文本提示 (Text Prompt)",
                                    placeholder="输入物体描述，例如：'a red car' 或 '一只猫'",
                                    lines=1,
                                )

                                with gr.Row():
                                    gr.Markdown("示例快速填充：")
                                    example_text_btn = gr.Button("🐱 猫", size="sm")
                                    example_point_btn = gr.Button(
                                        "📍 示例点", size="sm"
                                    )

                                with gr.Row(
                                    visible=False
                                ):  # 隐藏原始坐标输入框，保持后端逻辑但减少界面干扰
                                    pos_point_prompt = gr.Textbox(label="正点坐标")
                                    neg_point_prompt = gr.Textbox(label="负点坐标")
                                    pos_box_prompt = gr.Textbox(label="正框坐标")
                                    neg_box_prompt = gr.Textbox(label="负框坐标")
                                    pos_polygon_prompt = gr.Textbox(label="正多边形坐标")
                                    neg_polygon_prompt = gr.Textbox(label="负多边形坐标")

                            confidence_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.05,
                                label="🎯 置信度阈值 (Confidence)",
                            )

                            segment_button = gr.Button(
                                "🚀 开始分割 (Segment)", variant="primary", size="lg"
                            )

                        # 右侧结果栏
                        with gr.Column(scale=1):
                            image_output = gr.Image(type="numpy", label="✨ 分割结果")
                            image_info = gr.Textbox(
                                label="📊 分析报告", interactive=False, lines=2
                            )

                    # 事件绑定

                    # 1. 上传图片时保存原图
                    def store_original_image(img):
                        return img, None, "", "", "", "", "", "", "👆 点击图像开始添加提示..."

                    image_input.upload(
                        fn=store_original_image,
                        inputs=[image_input],
                        outputs=[
                            original_image_state,
                            click_state,
                            pos_point_prompt,
                            neg_point_prompt,
                            pos_box_prompt,
                            neg_box_prompt,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            interaction_info,
                        ],
                    )

                    # 2. 点击图片处理
                    image_input.select(
                        fn=handle_image_click,
                        inputs=[
                            image_input,
                            original_image_state,
                            interaction_mode,
                            prompt_polarity,
                            pos_point_prompt,
                            neg_point_prompt,
                            pos_box_prompt,
                            neg_box_prompt,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            click_state,
                        ],
                        outputs=[
                            image_input,
                            pos_point_prompt,
                            neg_point_prompt,
                            pos_box_prompt,
                            neg_box_prompt,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            click_state,
                            interaction_info,
                        ],
                    )

                    # 3. 完成多边形对象
                    finish_polygon_btn.click(
                        fn=finish_polygon,
                        inputs=[
                            image_input,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            click_state,
                        ],
                        outputs=[
                            image_input,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            click_state,
                            interaction_info,
                        ],
                    )

                    # 4. 清空提示
                    def clear_prompts(orig_img):
                        if orig_img is None:
                            return None, "", "", "", "", "", "", None, "请先上传图像"
                        return (
                            orig_img,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            None,
                            "♻️ 提示已清空，图像已重置",
                        )

                    clear_prompts_btn.click(
                        fn=clear_prompts,
                        inputs=[original_image_state],
                        outputs=[
                            image_input,
                            pos_point_prompt,
                            neg_point_prompt,
                            pos_box_prompt,
                            neg_box_prompt,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            click_state,
                            interaction_info,
                        ],
                    )

                    # 5. 分割按钮
                    segment_button.click(
                        fn=segment_image,
                        inputs=[
                            image_input,
                            text_prompt,
                            confidence_threshold,
                            pos_point_prompt,
                            neg_point_prompt,
                            pos_box_prompt,
                            neg_box_prompt,
                            pos_polygon_prompt,
                            neg_polygon_prompt,
                            original_image_state,
                        ],
                        outputs=[image_output, image_info],
                    )

                    # 6. 示例按钮
                    example_text_btn.click(fn=lambda: "a cat", outputs=[text_prompt])
                    example_point_btn.click(
                        fn=lambda: "100,100", outputs=[pos_point_prompt]
                    )

                # ================= 视频跟踪标签页 =================
                with gr.TabItem("🎬 视频目标跟踪", id="tab_video"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="📂 上传视频文件")

                            with gr.Group():
                                video_text_prompt = gr.Textbox(
                                    label="📝 跟踪目标描述",
                                    placeholder="例如：'a person running' (目前仅支持第一帧文本提示)",
                                    lines=2,
                                )
                                video_confidence_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.05,
                                    label="🎯 跟踪置信度",
                                )

                            process_button = gr.Button(
                                "▶️ 开始跟踪处理", variant="primary", size="lg"
                            )

                        with gr.Column(scale=1):
                            video_output = gr.Video(label="✨ 跟踪结果")
                            video_info = gr.Textbox(
                                label="📊 处理报告", interactive=False
                            )

                    process_button.click(
                        fn=process_video,
                        inputs=[
                            video_input,
                            video_text_prompt,
                            video_confidence_threshold,
                        ],
                        outputs=[video_output, video_info],
                    )

        # 页脚
        gr.Markdown(
            """
        ---
        <div style="text-align: center; color: #718096; font-size: 0.9em;">
            Powered by <strong>SAM3</strong> | 2025 SAM3 Interactive Studio
        </div>
        """
        )

    return demo


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
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=True,
        allowed_paths=[str(current_dir)],
    )


if __name__ == "__main__":
    main()
