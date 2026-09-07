"""Session-draft logic for optional manual overlap-layout authoring."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from layout_transform_utils import (
    foreground_bbox_xyxy,
    make_layout_transform_v2,
    mask_pixel_sha256,
)
from sam3_demo.mask_boundary_alignment import align_mask_boundary
from sam3_demo.overlap_geometry import calculate_next_layer_overlap
from sam3_demo.overlap_store import OverlapLayoutStore, ValidationError


AUTHORING_SCHEMA_VERSION = 1
LAYER_TYPES = ("BSM", "ACT", "GE1", "GE2")
MAX_LAYERS = 32
MAX_COCO_JSON_BYTES = 64 * 1024 * 1024
MAX_COCO_IMAGES = 10_000
MAX_COCO_ANNOTATIONS = 200_000
MAX_COCO_DIMENSION = 32_768
MAX_COCO_PIXELS = 100_000_000


class OverlapAuthoringError(ValueError):
    """Raised when a manual-authoring command violates the draft contract."""


def _new_id() -> str:
    return str(uuid.uuid4())


def _clone(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)


def _require_draft(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OverlapAuthoringError("制作草稿无效")
    if value.get("schema_version") != AUTHORING_SCHEMA_VERSION:
        raise OverlapAuthoringError("制作草稿版本无效")
    if not isinstance(value.get("draft_id"), str) or not value["draft_id"]:
        raise OverlapAuthoringError("制作草稿缺少 draft_id")
    if not isinstance(value.get("layers"), list):
        raise OverlapAuthoringError("制作草稿 layers 无效")
    return _clone(value)


def _load_image(value: Any, *, mode: str, field: str) -> Image.Image:
    try:
        if isinstance(value, Image.Image):
            image = value.copy()
        elif isinstance(value, np.ndarray):
            image = Image.fromarray(np.asarray(value))
        elif isinstance(value, (str, Path)):
            with Image.open(value) as source:
                source.load()
                image = source.copy()
        else:
            raise OverlapAuthoringError(f"{field}必须是图像")
        return image.convert(mode)
    except (OSError, UnidentifiedImageError, ValueError, TypeError) as exc:
        if isinstance(exc, OverlapAuthoringError):
            raise
        raise OverlapAuthoringError(f"{field}无法读取") from exc


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_png(payload: bytes, *, mode: str, field: str) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format != "PNG":
                raise OverlapAuthoringError(f"{field}不是 PNG")
            source.load()
            return source.convert(mode)
    except (OSError, UnidentifiedImageError) as exc:
        raise OverlapAuthoringError(f"{field}无法解码") from exc


def _normalise_mask(value: Any) -> tuple[Image.Image, np.ndarray]:
    image = _load_image(value, mode="RGBA", field="层标注")
    array = np.asarray(image, dtype=np.uint8)
    foreground = np.max(array[:, :, :3], axis=2) >= 128
    foreground &= array[:, :, 3] > 0
    if not foreground.any():
        raise OverlapAuthoringError("层标注没有前景")
    mask_image = Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")
    return mask_image, foreground


def _coco_id_key(value: Any, *, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise OverlapAuthoringError(f"COCO {field}必须是字符串或整数")
    if isinstance(value, str) and not value.strip():
        raise OverlapAuthoringError(f"COCO {field}不能为空")
    return f"{type(value).__name__}:{value}"


def _coco_dimension(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OverlapAuthoringError(f"COCO {field}必须是正整数")
    if value > MAX_COCO_DIMENSION:
        raise OverlapAuthoringError(
            f"COCO {field}超过上限 {MAX_COCO_DIMENSION}"
        )
    return value


def _coco_segmentation_shape_is_valid(segmentation: Any) -> None:
    if isinstance(segmentation, list):
        if not segmentation:
            raise OverlapAuthoringError("COCO annotation segmentation 不能为空")
        return
    if isinstance(segmentation, dict):
        if "size" not in segmentation or "counts" not in segmentation:
            raise OverlapAuthoringError("COCO RLE segmentation 缺少 size 或 counts")
        if not isinstance(segmentation["size"], (list, tuple)):
            raise OverlapAuthoringError("COCO RLE size 必须是 [height, width]")
        if len(segmentation["size"]) != 2:
            raise OverlapAuthoringError("COCO RLE size 必须是 [height, width]")
        if not isinstance(segmentation["counts"], (list, str, bytes)):
            raise OverlapAuthoringError(
                "COCO RLE counts 必须是列表、字符串或 bytes"
            )
        return
    raise OverlapAuthoringError(
        "COCO segmentation 必须是 polygon 列表或 RLE 对象"
    )


def _load_coco_dataset(value: Any) -> dict[str, Any]:
    candidate = (
        value
        if isinstance(value, (str, Path))
        else getattr(value, "name", None)
    )
    if not isinstance(candidate, (str, Path)) or not str(candidate).strip():
        raise OverlapAuthoringError("请先上传 COCO JSON 标注文件")
    path = Path(candidate)
    try:
        size = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise OverlapAuthoringError("COCO JSON 文件无法读取") from exc
    if size > MAX_COCO_JSON_BYTES:
        raise OverlapAuthoringError(
            f"COCO JSON 文件超过大小上限 {MAX_COCO_JSON_BYTES // (1024 * 1024)} MiB"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OverlapAuthoringError("COCO JSON 文件无法读取") from exc

    def reject_constant(_value: str) -> None:
        raise ValueError("non-standard JSON number")

    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except (MemoryError, RecursionError, TypeError, ValueError) as exc:
        raise OverlapAuthoringError("COCO JSON 无法解析") from exc
    if not isinstance(payload, dict):
        raise OverlapAuthoringError("COCO JSON 顶层必须是对象")

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise OverlapAuthoringError("COCO JSON 缺少 images 或 images 为空")
    if len(images) > MAX_COCO_IMAGES:
        raise OverlapAuthoringError(
            f"COCO JSON images 超过上限 {MAX_COCO_IMAGES}"
        )
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise OverlapAuthoringError("COCO JSON 缺少 annotation 或 annotation 为空")
    if len(annotations) > MAX_COCO_ANNOTATIONS:
        raise OverlapAuthoringError(
            f"COCO JSON annotations 超过上限 {MAX_COCO_ANNOTATIONS}"
        )

    records: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise OverlapAuthoringError(f"COCO image {index} 数据无效")
        image_key = _coco_id_key(image.get("id"), field="image id")
        if image_key in by_key:
            raise OverlapAuthoringError(
                f"COCO image id 重复：{image.get('id')!r}"
            )
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise OverlapAuthoringError(
                f"COCO image {image.get('id')!r} 缺少有效 file_name"
            )
        width = _coco_dimension(image.get("width"), field="image width")
        height = _coco_dimension(image.get("height"), field="image height")
        if width * height > MAX_COCO_PIXELS:
            raise OverlapAuthoringError(
                f"COCO image {image.get('id')!r} 像素数量超过上限"
            )
        record = {
            "id": image["id"],
            "key": image_key,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        records.append(record)
        by_key[image_key] = record

    annotations_by_image = {record["key"]: [] for record in records}
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise OverlapAuthoringError(f"COCO annotation {index} 数据无效")
        if "image_id" not in annotation:
            raise OverlapAuthoringError(
                f"COCO annotation {index} 缺少 image_id"
            )
        annotation_key = _coco_id_key(
            annotation["image_id"], field="annotation image_id"
        )
        if annotation_key not in by_key:
            raise OverlapAuthoringError(
                f"COCO annotation {index} 的 image_id 不存在"
            )
        if "segmentation" not in annotation:
            raise OverlapAuthoringError(
                f"COCO annotation {index} 缺少 segmentation"
            )
        _coco_segmentation_shape_is_valid(annotation["segmentation"])
        annotations_by_image[annotation_key].append(annotation)

    return {
        "images": records,
        "by_key": by_key,
        "annotations_by_image": annotations_by_image,
    }


def _decode_coco_segmentation(
    segmentation: Any,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except (ImportError, ModuleNotFoundError) as exc:
        raise OverlapAuthoringError(
            "COCO 标注解码依赖 pycocotools，不可用"
        ) from exc

    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise OverlapAuthoringError(
                    "COCO polygon segmentation 不能为空"
                )
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in segmentation
            ):
                polygons = [segmentation]
            else:
                polygons = segmentation
            normalized_polygons: list[list[float]] = []
            for index, polygon in enumerate(polygons):
                if not isinstance(polygon, (list, tuple)):
                    raise OverlapAuthoringError(
                        f"COCO polygon {index} 必须是坐标列表"
                    )
                if len(polygon) < 6 or len(polygon) % 2:
                    raise OverlapAuthoringError(
                        f"COCO polygon {index} 至少需要 3 个 xy 点"
                    )
                coordinates: list[float] = []
                for coordinate in polygon:
                    if (
                        isinstance(coordinate, bool)
                        or not isinstance(coordinate, (int, float))
                    ):
                        raise OverlapAuthoringError(
                            f"COCO polygon {index} 坐标无效"
                        )
                    coordinate = float(coordinate)
                    if not math.isfinite(coordinate):
                        raise OverlapAuthoringError(
                            f"COCO polygon {index} 坐标不是有限数"
                        )
                    coordinates.append(coordinate)
                normalized_polygons.append(coordinates)
            rle = mask_utils.merge(
                mask_utils.frPyObjects(normalized_polygons, height, width)
            )
        elif isinstance(segmentation, dict):
            size = segmentation.get("size")
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise OverlapAuthoringError(
                    "COCO RLE size 必须是 [height, width]"
                )
            rle_height = _coco_dimension(size[0], field="RLE height")
            rle_width = _coco_dimension(size[1], field="RLE width")
            if (rle_height, rle_width) != (height, width):
                raise OverlapAuthoringError(
                    f"COCO RLE size {(rle_height, rle_width)} 与图片尺寸 "
                    f"{(height, width)} 不一致"
                )
            counts = segmentation.get("counts")
            if isinstance(counts, list):
                if not counts:
                    raise OverlapAuthoringError(
                        "COCO uncompressed RLE counts 不能为空"
                    )
                normalized_counts: list[int] = []
                for count in counts:
                    if (
                        isinstance(count, bool)
                        or not isinstance(count, int)
                        or count < 0
                    ):
                        raise OverlapAuthoringError(
                            "COCO RLE counts 必须是非负整数"
                        )
                    normalized_counts.append(count)
                if sum(normalized_counts) != height * width:
                    raise OverlapAuthoringError(
                        "COCO uncompressed RLE counts 无法覆盖图片"
                    )
                rle = mask_utils.frPyObjects(
                    {"size": [height, width], "counts": normalized_counts},
                    height,
                    width,
                )
            elif isinstance(counts, str):
                try:
                    counts = counts.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise OverlapAuthoringError(
                        "COCO compressed RLE counts 必须是 ASCII"
                    ) from exc
                if not counts:
                    raise OverlapAuthoringError(
                        "COCO compressed RLE counts 不能为空"
                    )
                rle = {"size": [height, width], "counts": counts}
            elif isinstance(counts, bytes):
                if not counts:
                    raise OverlapAuthoringError(
                        "COCO compressed RLE counts 不能为空"
                    )
                rle = {"size": [height, width], "counts": counts}
            else:
                raise OverlapAuthoringError(
                    "COCO RLE counts 必须是列表、字符串或 bytes"
                )
        else:
            raise OverlapAuthoringError(
                "COCO segmentation 必须是 polygon 列表或 RLE 对象"
            )
        decoded = np.asarray(mask_utils.decode(rle))
    except OverlapAuthoringError:
        raise
    except Exception as exc:
        raise OverlapAuthoringError(
            f"COCO segmentation 无法解码：{exc}"
        ) from exc
    if decoded.ndim != 2 or decoded.shape != (height, width):
        raise OverlapAuthoringError(
            f"COCO segmentation mask 尺寸 {decoded.shape} 与图片尺寸 "
            f"{(height, width)} 不一致"
        )
    if not np.isin(decoded, (0, 1)).all():
        raise OverlapAuthoringError("COCO segmentation mask 不是二值")
    return np.ascontiguousarray(decoded.astype(bool))


def _coco_image_mask(
    dataset: dict[str, Any],
    selected_image_id: Any = None,
) -> tuple[str, Image.Image]:
    records = dataset["images"]
    by_key = dataset["by_key"]
    if selected_image_id is None:
        record = records[0]
    elif isinstance(selected_image_id, str):
        record = by_key.get(selected_image_id)
        if record is None:
            record = by_key.get(
                _coco_id_key(selected_image_id, field="selected image id")
            )
        if record is None:
            raise OverlapAuthoringError("选择的 COCO 图片不存在")
    else:
        selected_key = _coco_id_key(selected_image_id, field="selected image id")
        record = by_key.get(selected_key)
        if record is None:
            raise OverlapAuthoringError("选择的 COCO 图片不存在")
    annotations = dataset["annotations_by_image"][record["key"]]
    if not annotations:
        raise OverlapAuthoringError(
            f"COCO 图片 {record['file_name']} 没有 annotation"
        )
    union = np.zeros((record["height"], record["width"]), dtype=bool)
    for index, annotation in enumerate(annotations):
        mask = _decode_coco_segmentation(
            annotation["segmentation"],
            height=record["height"],
            width=record["width"],
        )
        union |= mask
    if not union.any():
        raise OverlapAuthoringError(
            f"COCO 图片 {record['file_name']} 的 union mask 为空"
        )
    return record["key"], Image.fromarray(
        union.astype(np.uint8) * 255, mode="L"
    )

def _load_annotation_document(value: Any) -> dict[str, Any]:
    """Parse one uploaded JSON whose complete contents define one layer."""
    candidate = (
        value if isinstance(value, (str, Path)) else getattr(value, "name", None)
    )
    if not isinstance(candidate, (str, Path)) or not str(candidate).strip():
        raise OverlapAuthoringError("请先上传层标注 JSON 文件")
    path = Path(candidate)
    try:
        size = path.stat().st_size
        if size > MAX_COCO_JSON_BYTES:
            raise OverlapAuthoringError(
                f"JSON 文件超过大小上限 {MAX_COCO_JSON_BYTES // (1024 * 1024)} MiB"
            )
        text = path.read_text(encoding="utf-8")
    except OverlapAuthoringError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise OverlapAuthoringError("层标注 JSON 文件无法读取") from exc

    def reject_constant(_value: str) -> None:
        raise ValueError("non-standard JSON number")

    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except (MemoryError, RecursionError, TypeError, ValueError) as exc:
        raise OverlapAuthoringError("层标注 JSON 无法解析") from exc
    if not isinstance(payload, dict):
        raise OverlapAuthoringError("层标注 JSON 顶层必须是对象")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    if "shapes" in payload:
        width = _coco_dimension(payload.get("imageWidth"), field="imageWidth")
        height = _coco_dimension(payload.get("imageHeight"), field="imageHeight")
        if width * height > MAX_COCO_PIXELS:
            raise OverlapAuthoringError("LabelMe 图片像素数量超过上限")
        shapes = payload.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            raise OverlapAuthoringError("LabelMe JSON 缺少 shapes 或 shapes 为空")
        if len(shapes) > MAX_COCO_ANNOTATIONS:
            raise OverlapAuthoringError("LabelMe shapes 数量超过上限")
        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        for index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                raise OverlapAuthoringError(f"LabelMe shape {index} 数据无效")
            shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
            if shape_type != "polygon":
                raise OverlapAuthoringError(
                    f"LabelMe shape {index} 类型 {shape_type!r} 暂不支持"
                )
            points = shape.get("points")
            if not isinstance(points, list) or len(points) < 3:
                raise OverlapAuthoringError(
                    f"LabelMe shape {index} 至少需要 3 个点"
                )
            normalized_points: list[tuple[float, float]] = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise OverlapAuthoringError(
                        f"LabelMe shape {index} 点坐标无效"
                    )
                x, y = point
                if (
                    isinstance(x, bool)
                    or isinstance(y, bool)
                    or not isinstance(x, (int, float))
                    or not isinstance(y, (int, float))
                ):
                    raise OverlapAuthoringError(
                        f"LabelMe shape {index} 点坐标无效"
                    )
                x, y = float(x), float(y)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise OverlapAuthoringError(
                        f"LabelMe shape {index} 点坐标不是有限数"
                    )
                normalized_points.append((x, y))
            drawer.polygon(normalized_points, fill=255)
        if not np.asarray(mask_image, dtype=np.uint8).any():
            raise OverlapAuthoringError("LabelMe 标注合并后没有前景")
        return {
            "format": "LabelMe",
            "mask": mask_image,
            "annotation_json": canonical,
            "summary": f"LabelMe · {len(shapes)} 个标注 · {width}×{height}",
        }

    if "images" in payload or "annotations" in payload:
        dataset = _load_coco_dataset(path)
        if len(dataset["images"]) != 1:
            raise OverlapAuthoringError(
                "每个 JSON 必须只对应一层；COCO 文件只能包含一张图片"
            )
        record = dataset["images"][0]
        _selected_key, mask = _coco_image_mask(dataset)
        annotation_count = len(dataset["annotations_by_image"][record["key"]])
        return {
            "format": "COCO",
            "mask": mask,
            "annotation_json": canonical,
            "summary": (
                f"COCO · {annotation_count} 个标注 · "
                f"{record['width']}×{record['height']}"
            ),
        }
    raise OverlapAuthoringError("仅支持 LabelMe 或单图 COCO JSON 标注")


def _uncompressed_rle_counts(mask: np.ndarray) -> list[int]:
    counts: list[int] = []
    previous = 0
    run = 0
    for value in np.asarray(mask, dtype=np.uint8).ravel(order="F"):
        pixel = int(value)
        if pixel == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = pixel
    counts.append(run)
    return counts


def _overlap_annotation_document(
    layout_name: str,
    current: dict[str, Any],
    following: dict[str, Any],
    intersection: np.ndarray,
) -> dict[str, Any]:
    height, width = intersection.shape
    area = int(intersection.sum())
    annotations: list[dict[str, Any]] = []
    if area:
        ys, xs = np.where(intersection)
        annotations.append(
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": {
                    "size": [height, width],
                    "counts": _uncompressed_rle_counts(intersection),
                },
                "area": area,
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ],
                "iscrowd": 1,
            }
        )
    category = f"{current['layer_type']}∩{following['layer_type']}"
    return {
        "images": [
            {
                "id": 1,
                "file_name": f"{layout_name}_{category}_overlap.png",
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": category}],
        "metadata": {
            "layout_name": layout_name,
            "current_layer_id": current["id"],
            "current_layer_type": current["layer_type"],
            "next_layer_id": following["id"],
            "next_layer_type": following["layer_type"],
        },
    }


def _data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _document_epoch(draft: dict[str, Any]) -> str:
    """Identify assets whose replacement requires a full browser rehydrate."""

    base_png = draft.get("base_image_png")
    identity = {
        "draft_id": draft["draft_id"],
        "base_image_sha256": (
            hashlib.sha256(base_png).hexdigest() if base_png else None
        ),
        "layers": sorted(
            [
                {
                    "id": layer["id"],
                    "mask_sha256": layer["mask_sha256"],
                }
                for layer in draft["layers"]
            ],
            key=lambda item: item["id"],
        ),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_signature(draft: dict[str, Any]) -> str:
    return _document_epoch(draft)


def _layer_transform(
    draft: dict[str, Any],
    layer: dict[str, Any],
    mask: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    scale: float,
    rotation_deg: float,
    preview_alpha: float,
    revision: int,
) -> dict[str, Any]:
    current = layer.get("transform") or {}
    pivot = [
        float(current.get("pivot_x")),
        float(current.get("pivot_y")),
    ] if current.get("pivot_x") is not None and current.get("pivot_y") is not None else None
    return make_layout_transform_v2(
        session_id="overlap-authoring",

        layout_id=draft["draft_id"],
        image_id=None,
        target_size=(int(draft["width"]), int(draft["height"])),
        source_mask=mask,
        center_x=center_x,
        center_y=center_y,
        pivot_xy=pivot,
        scale=scale,
        rotation_deg=rotation_deg,
        preview_alpha=preview_alpha,
        revision=revision,
        source_mask_pixel_sha256=mask_pixel_sha256(mask),
        target_image_sha256=hashlib.sha256(draft["base_image_png"]).hexdigest(),
    )


class OverlapAuthoringService:
    """Build immutable per-session drafts and persist complete snapshots."""

    def __init__(self, store: OverlapLayoutStore) -> None:
        self._store = store

    def new_draft(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORING_SCHEMA_VERSION,
            "draft_id": _new_id(),
            "layout_id": None,
            "store_revision": None,
            "draft_revision": 0,
            "name": "",
            "base_image_png": None,
            "width": 0,
            "height": 0,
            "layers": [],
            "active_layer_id": None,
            "editor_sync_revision": 0,
            "editor_sync_hash": "",
        }

    def set_name(self, draft: Any, name: Any) -> dict[str, Any]:
        out = _require_draft(draft)
        if not isinstance(name, str):
            raise OverlapAuthoringError("版图名称必须是字符串")
        name = name.strip()
        if len(name) > 128:
            raise OverlapAuthoringError("版图名称最多 128 个字符")
        if out["name"] != name:
            out["name"] = name
            out["draft_revision"] += 1
        return out

    def set_base_image(self, draft: Any, image: Any) -> dict[str, Any]:
        out = _require_draft(draft)
        base = _load_image(image, mode="RGB", field="整体版图")
        out["base_image_png"] = _encode_png(base)
        out["width"], out["height"] = base.size
        for layer in out["layers"]:
            mask_image = _decode_png(
                layer["mask_png"], mode="L", field=f"{layer['name']} 标注"
            )
            mask = np.asarray(mask_image, dtype=np.uint8) > 0
            current = layer["transform"]
            layer["transform"] = _layer_transform(
                out,
                layer,
                mask,
                center_x=out["width"] / 2.0,
                center_y=out["height"] / 2.0,
                scale=float(current["scale"]),
                rotation_deg=float(current["rotation_deg"]),
                preview_alpha=float(current.get("preview_alpha", 0.35)),
                revision=int(current.get("revision") or 0) + 1,
            )
        out["draft_revision"] += 1
        return out

    def add_layer(
        self,
        draft: Any,
        mask: Any,
        layer_type: Any,
        name: Any = None,
        annotation_json: Any = None,
    ) -> dict[str, Any]:
        out = _require_draft(draft)
        if not out.get("base_image_png"):
            raise OverlapAuthoringError("请先上传整体版图")
        if len(out["layers"]) >= MAX_LAYERS:
            raise OverlapAuthoringError(f"最多支持 {MAX_LAYERS} 层")
        layer_type = str(layer_type or "").strip()
        if not layer_type:
            raise OverlapAuthoringError("请输入层分类")
        if len(layer_type) > 128:
            raise OverlapAuthoringError("层分类最多 128 个字符")
        layer_name = str(name or "").strip() or layer_type
        if len(layer_name) > 128:
            raise OverlapAuthoringError("层名称最多 128 个字符")
        mask_image, mask_array = _normalise_mask(mask)
        mask_png = _encode_png(mask_image)
        layer = {
            "id": _new_id(),
            "ordinal": len(out["layers"]),
            "layer_type": layer_type,
            "name": layer_name,
            "mask_png": mask_png,
            "mask_width": mask_image.width,
            "mask_height": mask_image.height,
            "mask_sha256": hashlib.sha256(mask_png).hexdigest(),
            "transform": {},
            "annotation_json": annotation_json,
        }
        if mask_image.size == (out["width"], out["height"]):
            x1, y1, x2, y2 = foreground_bbox_xyxy(mask_array)
            initial_center_x = (x1 + x2) / 2.0
            initial_center_y = (y1 + y2) / 2.0
        else:
            initial_center_x = out["width"] / 2.0
            initial_center_y = out["height"] / 2.0
        layer["transform"] = _layer_transform(
            out,
            layer,
            mask_array,
            center_x=initial_center_x,
            center_y=initial_center_y,
            scale=1.0,
            rotation_deg=0.0,
            preview_alpha=0.35,
            revision=0,
        )
        out["layers"].append(layer)
        out["active_layer_id"] = layer["id"]
        out["draft_revision"] += 1
        return out

    def coco_image_choices(self, json_file: Any) -> list[tuple[str, str]]:
        """Return collision-safe selector values for every JSON image."""

        dataset = _load_coco_dataset(json_file)
        return [
            (
                f"{record['file_name']} ({record['width']}×{record['height']})",
                record["key"],
            )
            for record in dataset["images"]
        ]

    def coco_image_mask(
        self,
        json_file: Any,
        selected_image_id: Any = None,
    ) -> Image.Image:
        """Rasterize all annotations for one COCO image into a binary mask."""

        dataset = _load_coco_dataset(json_file)
        _selected_key, mask = _coco_image_mask(dataset, selected_image_id)
        return mask

    def inspect_coco_json(
        self,
        json_file: Any,
        selected_image_id: Any = None,
    ) -> tuple[list[tuple[str, str]], str, Image.Image]:
        """Return selector choices, the selected value, and its union mask."""

        dataset = _load_coco_dataset(json_file)
        choices = [
            (
                f"{record['file_name']} ({record['width']}×{record['height']})",
                record["key"],
            )
            for record in dataset["images"]
        ]
        selected_key, mask = _coco_image_mask(dataset, selected_image_id)
        return choices, selected_key, mask

    def add_layer_from_coco(
        self,
        draft: Any,
        json_file: Any,
        selected_image_id: Any,
        layer_type: Any,
        name: Any = None,
    ) -> dict[str, Any]:
        """Re-read JSON and rasterize the selected image before adding a layer."""

        mask = self.coco_image_mask(json_file, selected_image_id)
        return self.add_layer(draft, mask, layer_type, name)

    def inspect_annotation_json(
        self,
        json_file: Any,
    ) -> tuple[Image.Image, str]:
        """Preview a LabelMe or single-image COCO document as one layer."""
        document = _load_annotation_document(json_file)
        return document["mask"], document["summary"]

    def add_layer_from_annotation(
        self,
        draft: Any,
        json_file: Any,
        layer_type: Any,
    ) -> dict[str, Any]:
        """Re-read one JSON and add its complete annotation as one layer."""
        document = _load_annotation_document(json_file)
        return self.add_layer(
            draft,
            document["mask"],
            layer_type,
            annotation_json=document["annotation_json"],
        )

    def remove_layer(self, draft: Any, layer_id: Any) -> dict[str, Any]:
        out = _require_draft(draft)
        layer_id = str(layer_id or "")
        original = list(out["layers"])
        out["layers"] = [layer for layer in original if layer["id"] != layer_id]
        if len(out["layers"]) == len(original):
            raise OverlapAuthoringError("待删除层不存在")
        for ordinal, layer in enumerate(out["layers"]):
            layer["ordinal"] = ordinal
        valid_ids = [layer["id"] for layer in out["layers"]]
        if out.get("active_layer_id") not in valid_ids:
            out["active_layer_id"] = valid_ids[0] if valid_ids else None
        out["draft_revision"] += 1
        return out

    def reorder_layers(
        self, draft: Any, ordered_layer_ids: Any
    ) -> dict[str, Any]:
        out = _require_draft(draft)
        if not isinstance(ordered_layer_ids, list):
            raise OverlapAuthoringError("叠层顺序必须是列表")
        ids = [str(value) for value in ordered_layer_ids]
        current_ids = [layer["id"] for layer in out["layers"]]
        if len(ids) != len(set(ids)) or set(ids) != set(current_ids):
            raise OverlapAuthoringError("叠层顺序必须精确包含全部层")
        by_id = {layer["id"]: layer for layer in out["layers"]}
        out["layers"] = [by_id[layer_id] for layer_id in ids]
        for ordinal, layer in enumerate(out["layers"]):
            layer["ordinal"] = ordinal
        out["draft_revision"] += 1
        return out

    def set_active_layer(self, draft: Any, layer_id: Any) -> dict[str, Any]:
        out = _require_draft(draft)
        layer_id = str(layer_id or "")
        if layer_id not in {layer["id"] for layer in out["layers"]}:
            raise OverlapAuthoringError("当前层不存在")
        if out.get("active_layer_id") != layer_id:
            out["active_layer_id"] = layer_id
            out["draft_revision"] += 1
        return out

    def editor_payload(self, draft: Any) -> dict[str, Any]:
        out = _require_draft(draft)
        epoch = _document_epoch(out)
        snapshot_meta = {
            "payload_kind": "snapshot",
            "editor_session_id": out["draft_id"],
            "document_epoch": epoch,
        }
        if not out.get("base_image_png"):
            return {
                **snapshot_meta,
                "enabled": False,
                "status": "请先上传整体版图",
            }
        if not out["layers"]:
            return {
                **snapshot_meta,
                "enabled": False,
                "base_image": _data_url(out["base_image_png"]),
                "target_width": out["width"],
                "target_height": out["height"],
                "status": "请上传至少一层标注",
            }
        groups = []
        transforms = []
        for layer in out["layers"]:
            mask = np.asarray(
                _decode_png(layer["mask_png"], mode="L", field=layer["name"]),
                dtype=np.uint8,
            ) > 0
            transform = copy.deepcopy(layer["transform"])
            transform.update(
                {
                    "group_id": layer["id"],
                    "label": f"{layer['ordinal'] + 1}. {layer['layer_type']}",
                    "group_mask_pixel_sha256": mask_pixel_sha256(mask),
                }
            )
            groups.append(
                {
                    "group_id": layer["id"],
                    "label": transform["label"],
                    "region_ids": [],
                    "mask_image": _data_url(layer["mask_png"]),
                    "foreground_bbox_xyxy": foreground_bbox_xyxy(mask),
                    "group_mask_pixel_sha256": transform[
                        "group_mask_pixel_sha256"
                    ],
                }
            )
            transforms.append(
                {"group_id": layer["id"], "transform": transform}
            )
        active_id = out.get("active_layer_id")
        if active_id not in {layer["id"] for layer in out["layers"]}:
            active_id = out["layers"][0]["id"]
        active_transform = next(
            item["transform"]
            for item in transforms
            if item["group_id"] == active_id
        )
        sync_revision = int(out.get("editor_sync_revision") or 0)
        sync_hash = str(out.get("editor_sync_hash") or "")
        return {
            **snapshot_meta,
            "enabled": True,
            "base_image": _data_url(out["base_image_png"]),
            "transform": copy.deepcopy(active_transform),
            "transform_mode": "label_groups",
            "group_view": {
                "selection_signature": epoch,
                "regions_revision": int(out["draft_revision"]),
                "allow_reorder": True,
                "groups": groups,
            },
            "group_intent": {
                "selection_signature": epoch,
                "transform_set_revision": sync_revision,
                "editor_session_id": out["draft_id"],
                "document_epoch": epoch,
                "client_revision": sync_revision,
                "payload_hash": sync_hash,
                "active_group_id": active_id,
                "changed_group_ids": [],
                "ordered_group_ids": [
                    layer["id"] for layer in out["layers"]
                ],
                "transforms": transforms,
            },
            "target_width": out["width"],
            "target_height": out["height"],
            "status": "拖动层标注平移；滚轮缩放；拖动右上圆点旋转；上方列表可调整顺序。",
        }

    def editor_ack(
        self,
        draft: Any,
        *,
        status: str = "叠层变换已同步",
        error: str | None = None,
    ) -> dict[str, Any]:
        current = _require_draft(draft)
        payload = {
            "payload_kind": "ack_error" if error else "ack",
            "editor_session_id": current["draft_id"],
            "document_epoch": _document_epoch(current),
            "server_ack_revision": int(
                current.get("editor_sync_revision") or 0
            ),
            "payload_hash": str(current.get("editor_sync_hash") or ""),
            "status": str(error or status),
        }
        return payload

    def apply_editor_intent(
        self, draft: Any, editor_payload: Any
    ) -> dict[str, Any]:
        current = _require_draft(draft)
        if not isinstance(editor_payload, dict):
            raise OverlapAuthoringError("编辑器返回值无效")
        intent = editor_payload.get("group_intent")
        if not isinstance(intent, dict):
            raise OverlapAuthoringError("编辑器缺少 group_intent")

        epoch = _document_epoch(current)
        if intent.get("selection_signature") != epoch:
            raise OverlapAuthoringError("编辑器数据已过期，请重试")
        if intent.get("editor_session_id") != current["draft_id"]:
            raise OverlapAuthoringError("编辑器会话已过期，请重试")
        if intent.get("document_epoch") != epoch:
            raise OverlapAuthoringError("编辑器版图已更新，请重试")

        client_revision = intent.get("client_revision")
        legacy_revision = intent.get("transform_set_revision")
        if (
            not isinstance(client_revision, int)
            or isinstance(client_revision, bool)
            or client_revision < 0
            or legacy_revision != client_revision
        ):
            raise OverlapAuthoringError("编辑器 revision 无效")
        payload_hash = intent.get("payload_hash")
        if (
            not isinstance(payload_hash, str)
            or len(payload_hash) > 128
        ):
            raise OverlapAuthoringError("编辑器 payload hash 无效")

        stored_revision = int(current.get("editor_sync_revision") or 0)
        stored_hash = str(current.get("editor_sync_hash") or "")
        if client_revision < stored_revision:
            raise OverlapAuthoringError("编辑器 revision 已过期，请重试")
        if client_revision == stored_revision:
            if payload_hash == stored_hash:
                return current
            raise OverlapAuthoringError("编辑器 revision 内容冲突")

        current_ids = [layer["id"] for layer in current["layers"]]
        active_id = str(intent.get("active_group_id") or "")
        if active_id not in set(current_ids):
            raise OverlapAuthoringError("编辑器当前层无效")
        changed = intent.get("changed_group_ids") or []
        if not isinstance(changed, list):
            raise OverlapAuthoringError("编辑器 changed_group_ids 无效")
        changed_ids = [str(value) for value in changed]
        if len(changed_ids) != len(set(changed_ids)) or not set(
            changed_ids
        ).issubset(current_ids):
            raise OverlapAuthoringError("编辑器 changed_group_ids 重复或无效")

        items = intent.get("transforms")
        if not isinstance(items, list):
            raise OverlapAuthoringError("编辑器 transforms 无效")
        incoming: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                raise OverlapAuthoringError("编辑器 transform item 无效")
            group_id = str(item.get("group_id") or "")
            transform = item.get("transform")
            if group_id in incoming or not isinstance(transform, dict):
                raise OverlapAuthoringError("编辑器 transform 重复或无效")
            incoming[group_id] = transform
        if set(incoming) != set(current_ids):
            raise OverlapAuthoringError("编辑器 transform 集合与服务端不一致")

        ordered_ids = intent.get("ordered_group_ids")
        if ordered_ids is None:
            ordered_ids = current_ids
        if not isinstance(ordered_ids, list):
            raise OverlapAuthoringError("编辑器叠层顺序无效")
        ordered_ids = [str(value) for value in ordered_ids]
        if len(ordered_ids) != len(set(ordered_ids)) or set(
            ordered_ids
        ) != set(current_ids):
            raise OverlapAuthoringError("编辑器叠层顺序与服务端不一致")

        by_id = {layer["id"]: layer for layer in current["layers"]}
        for layer_id in changed_ids:
            layer = by_id[layer_id]
            values = incoming[layer_id]
            numeric = {}
            for field in (
                "center_x",
                "center_y",
                "scale",
                "rotation_deg",
                "preview_alpha",
            ):
                value = values.get(field)
                if isinstance(value, bool):
                    raise OverlapAuthoringError(f"编辑器 {field} 无效")
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise OverlapAuthoringError(
                        f"编辑器 {field} 无效"
                    ) from exc
                if not math.isfinite(value):
                    raise OverlapAuthoringError(f"编辑器 {field} 无效")
                numeric[field] = value
            if numeric["scale"] <= 0:
                raise OverlapAuthoringError("编辑器 scale 必须大于零")
            mask = np.asarray(
                _decode_png(layer["mask_png"], mode="L", field=layer["name"]),
                dtype=np.uint8,
            ) > 0
            old = layer["transform"]
            layer["transform"] = _layer_transform(
                current,
                layer,
                mask,
                center_x=numeric["center_x"],
                center_y=numeric["center_y"],
                scale=numeric["scale"],
                rotation_deg=numeric["rotation_deg"],
                preview_alpha=numeric["preview_alpha"],
                revision=int(old.get("revision") or 0) + 1,
            )
        current["layers"] = [by_id[layer_id] for layer_id in ordered_ids]
        for ordinal, layer in enumerate(current["layers"]):
            layer["ordinal"] = ordinal
        current["active_layer_id"] = active_id
        current["editor_sync_revision"] = client_revision
        current["editor_sync_hash"] = payload_hash
        current["draft_revision"] = int(current["draft_revision"]) + 1
        return current

    def load_layout(self, layout_id: str) -> dict[str, Any]:
        snapshot = self._store.get_layout(layout_id)
        return {
            "schema_version": AUTHORING_SCHEMA_VERSION,
            "draft_id": _new_id(),
            "layout_id": snapshot["id"],
            "store_revision": snapshot["revision"],
            "draft_revision": 0,
            "name": snapshot["name"],
            "base_image_png": snapshot["base_image_png"],
            "width": snapshot["width"],
            "height": snapshot["height"],
            "layers": copy.deepcopy(snapshot["layers"]),
            "active_layer_id": (
                snapshot["layers"][0]["id"] if snapshot["layers"] else None
            ),
            "editor_sync_revision": 0,
            "editor_sync_hash": "",
        }

    def alignment_proposal(
        self,
        draft: Any,
        layer_id: str | None = None,
    ) -> dict[str, Any]:
        """Calculate proposals for one layer without mutating its draft."""

        current = _require_draft(draft)
        if not current.get("base_image_png"):
            raise OverlapAuthoringError("请先上传整体版图")
        target_id = str(layer_id or current.get("active_layer_id") or "")
        layer = next(
            (item for item in current["layers"] if item["id"] == target_id),
            None,
        )
        if layer is None:
            raise OverlapAuthoringError("待对齐层不存在")
        base = np.asarray(
            _decode_png(current["base_image_png"], mode="RGB", field="整体版图"),
            dtype=np.uint8,
        )
        mask = np.asarray(
            _decode_png(layer["mask_png"], mode="L", field=f"{layer['name']} 标注"),
            dtype=np.uint8,
        ) > 0
        transform = copy.deepcopy(layer["transform"])
        result = align_mask_boundary(base, mask, transform)
        candidates = result.get("candidates") or []
        if not candidates:
            status = f"{layer['layer_type']} 未找到可用粗对齐候选，请手动调整。"
        elif result.get("auto_apply"):
            status = (
                f"{layer['layer_type']} 粗对齐高置信，将自动应用最佳候选；"
                f"耗时 {result['elapsed_seconds']:.2f}s。"
            )
        elif result.get("ambiguous"):
            status = (
                f"{layer['layer_type']} 存在周期歧义，未自动应用；"
                "请选择候选或手动调整。"
            )
        else:
            status = (
                f"{layer['layer_type']} 置信度未达自动应用门槛，"
                "请选择候选或手动调整。"
            )
        return {
            "payload_kind": "alignment_proposal",
            "editor_session_id": current["draft_id"],
            "document_epoch": _document_epoch(current),
            "target_layer_id": layer["id"],
            "target_label": layer["layer_type"],
            "start_transform_revision": int(transform.get("revision") or 0),
            "candidates": candidates[:3],
            "auto_apply": bool(result.get("auto_apply")),
            "ambiguous": bool(result.get("ambiguous")),
            "distinct_margin": float(result.get("distinct_margin") or 0.0),
            "elapsed_seconds": float(result.get("elapsed_seconds") or 0.0),
            "status": status,
        }

    def save(self, draft: Any) -> dict[str, Any]:
        current = _require_draft(draft)
        if not current.get("name", "").strip():
            raise OverlapAuthoringError("请输入版图名称")
        if not current.get("base_image_png"):
            raise OverlapAuthoringError("请上传整体版图")
        if len(current["layers"]) < 2:
            raise OverlapAuthoringError("至少需要两层标注")
        geometry_layers: list[dict[str, Any]] = []
        for ordinal, layer in enumerate(current["layers"]):
            mask = np.asarray(
                _decode_png(
                    layer["mask_png"],
                    mode="L",
                    field=f"{layer['layer_type']} 标注",
                ),
                dtype=np.uint8,
            ) > 0
            geometry_layers.append(
                {
                    "id": layer["id"],
                    "ordinal": ordinal,
                    "layer_type": layer["layer_type"],
                    "name": layer["name"],
                    "transform": layer["transform"],
                    "mask": mask,
                }
            )
        overlap_documents: dict[str, dict[str, Any]] = {}
        for layer in geometry_layers[:-1]:
            result = calculate_next_layer_overlap(
                geometry_layers,
                layer["id"],
                (current["width"], current["height"]),
            )
            overlap_documents[layer["id"]] = _overlap_annotation_document(
                current["name"],
                result["current_layer"],
                result["next_layer"],
                result["intersection_mask"],
            )
        payload_layers = [
            {
                "id": layer["id"],
                "ordinal": ordinal,
                "layer_type": layer["layer_type"],
                "name": layer["name"],
                "annotation_json": layer.get("annotation_json"),
                "next_overlap_json": overlap_documents.get(layer["id"]),
                "mask_png": layer["mask_png"],
                "mask_width": layer["mask_width"],
                "mask_height": layer["mask_height"],
                "mask_sha256": layer["mask_sha256"],
                "transform": layer["transform"],
            }
            for ordinal, layer in enumerate(current["layers"])
        ]
        try:
            if current.get("layout_id"):
                snapshot = self._store.update_layout(
                    current["layout_id"],
                    current["store_revision"],
                    name=current["name"],
                    base_image_png=current["base_image_png"],
                    width=current["width"],
                    height=current["height"],
                    layers=payload_layers,
                )
            else:
                snapshot = self._store.create_layout(
                    current["name"],
                    current["base_image_png"],
                    current["width"],
                    current["height"],
                    payload_layers,
                )
        except ValidationError as exc:
            raise OverlapAuthoringError(str(exc)) from exc
        return self.load_layout(snapshot["id"])

    def delete(self, draft: Any) -> dict[str, Any]:
        current = _require_draft(draft)
        if not current.get("layout_id"):
            raise OverlapAuthoringError("当前草稿尚未保存")
        self._store.delete_layout(
            current["layout_id"], current["store_revision"]
        )
        return self.new_draft()


__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "LAYER_TYPES",
    "MAX_COCO_ANNOTATIONS",
    "MAX_COCO_DIMENSION",
    "MAX_COCO_IMAGES",
    "MAX_COCO_JSON_BYTES",
    "MAX_COCO_PIXELS",
    "OverlapAuthoringError",
    "OverlapAuthoringService",
]
