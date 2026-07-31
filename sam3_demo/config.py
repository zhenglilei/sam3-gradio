"""Runtime paths and immutable application configuration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parents[1]
runtime_dir = current_dir / ".runtime"
runtime_tmp_dir = runtime_dir / "tmp"
runtime_gradio_dir = runtime_dir / "gradio"
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

MODE_PCS = "PCS Auto"
MODE_PVS = "PVS Manual"
MODE_LAYOUT = "Layout Mask"

_LAYOUT_PROMPT_SCOPE_FULL = "full"
_LAYOUT_PROMPT_SCOPE_REGION_CLASS = "region_class"
_LAYOUT_PROMPT_CLASS_PREFIX = "region_class:"
_LAYOUT_PROMPT_SCOPE_REGION_LABELS = "region_labels"
_LAYOUT_PROMPT_LABEL_PREFIX = "region_label:"
_LAYOUT_MASK_MORPH_LIMIT_PX = 31
_LAYOUT_MASK_VLM_BASE_URL = os.environ.get(
    "LAYOUT_MASK_VLM_BASE_URL", "http://10.101.100.20:15000/v1"
).rstrip("/")
_LAYOUT_MASK_VLM_MODEL = os.environ.get("LAYOUT_MASK_VLM_MODEL", "qwen3.5-122b")
_LAYOUT_MASK_VLM_TIMEOUT_SECONDS = float(os.environ.get("LAYOUT_MASK_VLM_TIMEOUT_SECONDS", "180"))
_LAYOUT_MASK_VLM_MAX_TOKENS = int(os.environ.get("LAYOUT_MASK_VLM_MAX_TOKENS", "1024"))
_LAYOUT_MASK_VLM_TEMPERATURE = float(os.environ.get("LAYOUT_MASK_VLM_TEMPERATURE", "0.1"))
_LAYOUT_MASK_VLM_API_KEY = os.environ.get("LAYOUT_MASK_VLM_API_KEY", "").strip()
layout_mask_agent_skill_dir = current_dir / "sam3_demo" / "layout" / "skills" / "layout-mask-preprocess"
_PUBLIC_DOWNLOAD_TTL_SECONDS = 24 * 60 * 60
_MAX_PROMPT_HISTORY_ENTRIES = 64
_WORKSPACE_CACHE_MAX_ENTRIES = 4
_WORKSPACE_CACHE_TTL_SECONDS = 3600.0
_SOURCE_IMAGE_CACHE_MAX_ENTRIES = 4
_SOURCE_IMAGE_CACHE_TTL_SECONDS = 3600.0


for path in (
    runtime_tmp_dir,
    runtime_gradio_dir,
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
for component_backend_dir in (
    current_dir / "layout_transform_editor" / "backend",
    current_dir / "layout_region_annotator" / "backend",
    current_dir / "image_gesture_overlay" / "backend",
):
    if component_backend_dir.exists():
        sys.path.insert(0, str(component_backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sam3_gradio_demo")
