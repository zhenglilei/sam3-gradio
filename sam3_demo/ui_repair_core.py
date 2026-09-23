"""Session-owned data and image operations for the EL UI-repair workspace."""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .session_cleanup import validate_server_session_id


MAX_IMAGES = 32
MAX_PIXELS = 32_000_000
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TOOLS = {"brush", "eraser", "rect_add", "rect_erase"}
COLOR_RANGES = {
    "red": (((0, 80, 80), (12, 255, 255)), ((160, 80, 80), (180, 255, 255))),
    "yellow": (((18, 80, 80), (38, 255, 255)),),
}


def new_repair_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": validate_server_session_id(session_id),
        "items": [],
        "active_id": None,
        "selected_ids": [],
    }


def owned_repair_state(state: Any, session_state: dict[str, Any]) -> dict[str, Any]:
    session_id = validate_server_session_id(session_state["session_id"])
    if not isinstance(state, dict) or state.get("session_id") != session_id:
        state = new_repair_state(session_id)
        for field in ("owner_token", "resume_id"):
            if session_state.get(field):
                state[field] = session_state[field]
        return state
    state.setdefault("items", [])
    state.setdefault("active_id", None)
    state.setdefault("selected_ids", [])
    return state


def _upload_path(value: Any) -> Path:
    raw = getattr(value, "name", value)
    path = Path(str(raw or ""))
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片文件：{path.name or raw}")
    return path


def _state_root(runtime_root: str | Path, state: dict[str, Any]) -> Path:
    root = Path(runtime_root) / validate_server_session_id(state["session_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_uploaded_images(
    state: dict[str, Any],
    files: Iterable[Any] | None,
    runtime_root: str | Path,
) -> list[dict[str, Any]]:
    paths = [_upload_path(value) for value in (files or [])]
    if not paths:
        raise ValueError("请先添加图片")
    if len(state["items"]) + len(paths) > MAX_IMAGES:
        raise ValueError(f"每个会话最多保留 {MAX_IMAGES} 张待修复图片")

    current_pixels = sum(int(item["width"]) * int(item["height"]) for item in state["items"])
    prepared: list[tuple[Path, Image.Image]] = []
    for path in paths:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB").copy()
        current_pixels += image.width * image.height
        if current_pixels > MAX_PIXELS:
            raise ValueError(f"每个会话图片总量不能超过 {MAX_PIXELS // 1_000_000} 百万像素")
        prepared.append((path, image))

    root = _state_root(runtime_root, state)
    created: list[Path] = []
    items: list[dict[str, Any]] = []
    try:
        for source_path, image in prepared:
            image_id = uuid.uuid4().hex
            folder = root / image_id
            folder.mkdir(parents=True, exist_ok=False)
            created.append(folder)
            stored_source = folder / "source.png"
            stored_mask = folder / "mask.png"
            image.save(stored_source)
            Image.new("L", image.size, 0).save(stored_mask)
            item = {
                "id": image_id,
                "name": source_path.name,
                "width": image.width,
                "height": image.height,
                "source_path": str(stored_source),
                "source_sha256": _sha256(stored_source),
                "mask_path": str(stored_mask),
                "result_path": None,
                "revision": 0,
                "status": "待标记",
                "error": "",
            }
            state["items"].append(item)
            items.append(item)
    except Exception:
        for folder in created:
            shutil.rmtree(folder, ignore_errors=True)
        for item in items:
            if item in state["items"]:
                state["items"].remove(item)
        raise
    if items:
        state["active_id"] = items[0]["id"]
        state["selected_ids"] = list(
            dict.fromkeys([*state["selected_ids"], *(item["id"] for item in items)])
        )
    return items


def find_item(state: dict[str, Any], image_id: str | None) -> dict[str, Any] | None:
    return next(
        (item for item in state.get("items", []) if item.get("id") == image_id),
        None,
    )


def active_item(state: dict[str, Any]) -> dict[str, Any] | None:
    return find_item(state, state.get("active_id"))


def select_item(state: dict[str, Any], index: int) -> dict[str, Any]:
    items = state.get("items", [])
    if not items:
        raise ValueError("当前没有待修复图片")
    index = max(0, min(int(index), len(items) - 1))
    state["active_id"] = items[index]["id"]
    return items[index]


def move_active(state: dict[str, Any], delta: int) -> dict[str, Any]:
    items = state.get("items", [])
    current = next(
        (index for index, item in enumerate(items) if item["id"] == state.get("active_id")),
        0,
    )
    return select_item(state, current + int(delta))


def remove_repair_items(
    state: dict[str, Any],
    image_ids: Iterable[str] | None,
) -> list[str]:
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    requested = {
        str(image_id)
        for image_id in (image_ids or ())
        if image_id is not None
    }
    if not requested:
        return []

    items = state.get("items", [])
    active_id = state.get("active_id")
    active_index = next(
        (
            index
            for index, item in enumerate(items)
            if str(item.get("id") or "") == str(active_id or "")
        ),
        None,
    )
    removed_ids = [
        str(item.get("id") or "")
        for item in items
        if str(item.get("id") or "") in requested
    ]
    if not removed_ids:
        return []

    removed = set(removed_ids)
    remaining = [
        item
        for item in items
        if str(item.get("id") or "") not in removed
    ]
    state["items"] = remaining
    remaining_ids = {
        str(item.get("id") or "")
        for item in remaining
    }
    state["selected_ids"] = [
        image_id
        for image_id in state.get("selected_ids", [])
        if str(image_id) in remaining_ids
    ]
    if str(active_id or "") in removed:
        if remaining:
            state["active_id"] = remaining[
                min(active_index or 0, len(remaining) - 1)
            ]["id"]
        else:
            state["active_id"] = None
            state["selected_ids"] = []
    return removed_ids


def queue_view(
    state: dict[str, Any],
) -> tuple[list[tuple[Image.Image, str]], list[tuple[str, str]]]:
    gallery: list[tuple[Image.Image, str]] = []
    choices: list[tuple[str, str]] = []
    for index, item in enumerate(state.get("items", []), 1):
        with Image.open(item["source_path"]) as opened:
            thumb = opened.convert("RGB").copy()
        thumb.thumbnail((180, 120))
        title = f"{index}. {item['status']} | {item['name']}"
        gallery.append((thumb, title))
        choices.append((title, item["id"]))
    return gallery, choices


def _image_data_url(path: str | Path) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def empty_editor_payload() -> dict[str, Any]:
    return {
        "image_id": None,
        "revision": 0,
        "source_width": 0,
        "source_height": 0,
        "base_image": None,
        "mask_png": None,
        "preview_alpha": 0.45,
        "tool": "brush",
        "brush_size": 20,
    }


def editor_payload(
    item: dict[str, Any] | None,
    *,
    tool: str = "brush",
    brush_size: int = 20,
    overlay_alpha: float = 0.45,
) -> dict[str, Any]:
    if item is None:
        return empty_editor_payload()
    return {
        "image_id": item["id"],
        "revision": int(item["revision"]),
        "source_width": int(item["width"]),
        "source_height": int(item["height"]),
        "base_image": _image_data_url(item["source_path"]),
        "mask_png": _image_data_url(item["mask_path"]),
        "preview_alpha": max(0.05, min(float(overlay_alpha), 0.9)),
        "tool": tool if tool in TOOLS else "brush",
        "brush_size": max(1, min(int(brush_size), 120)),
    }


def _decode_mask_data_url(value: Any, expected_size: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
        raise ValueError("画布没有返回有效的 PNG mask")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError("画布 mask 无法解码") from exc
    width, height = expected_size
    if mask.shape != (height, width):
        raise ValueError("画布 mask 尺寸与原图不一致")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def commit_editor_value(state: dict[str, Any], payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("画布数据格式无效")
    item = find_item(state, str(payload.get("image_id") or ""))
    if item is None or item["id"] != state.get("active_id"):
        raise ValueError("画布对应的图片已切换，请在当前图片上重试")
    if (
        int(payload.get("source_width") or 0),
        int(payload.get("source_height") or 0),
    ) != (item["width"], item["height"]):
        raise ValueError("画布尺寸已过期，请重新选择当前图片")
    client_revision = int(payload.get("revision") or 0)
    if client_revision < int(item["revision"]):
        raise ValueError("画布版本已过期，请重新选择当前图片")
    mask = _decode_mask_data_url(
        payload.get("mask_png"),
        (item["width"], item["height"]),
    )
    with Image.open(item["mask_path"]) as current_image:
        current = np.asarray(current_image.convert("L"), dtype=np.uint8)
    if not np.array_equal(current > 0, mask > 0):
        Image.fromarray(mask, mode="L").save(item["mask_path"])
        item["revision"] = max(int(item["revision"]) + 1, client_revision)
        item["result_path"] = None
        item["status"] = "已标记" if bool(mask.any()) else "待标记"
        item["error"] = ""
    return item


def color_bbox_mask(
    image: Image.Image,
    color: str,
    *,
    saturation: int = 80,
    value: int = 80,
    min_area: int = 20,
    padding: int = 5,
    merge_distance: int = 10,
) -> tuple[np.ndarray, int]:
    if color not in COLOR_RANGES:
        raise ValueError(f"不支持的颜色：{color}")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = np.zeros(hsv.shape[:2], dtype=np.uint8)
    saturation = max(0, min(int(saturation), 255))
    value = max(0, min(int(value), 255))
    for lower, upper in COLOR_RANGES[color]:
        lo = np.asarray((lower[0], saturation, value), dtype=np.uint8)
        hi = np.asarray((upper[0], upper[1], upper[2]), dtype=np.uint8)
        pixels = cv2.bitwise_or(pixels, cv2.inRange(hsv, lo, hi))
    if not bool(pixels.any()):
        return pixels, 0
    merge_distance = max(0, min(int(merge_distance), 100))
    if merge_distance:
        kernel = np.ones((merge_distance, merge_distance), dtype=np.uint8)
        pixels = cv2.dilate(pixels, kernel, iterations=1)
    labels, _components, stats, _centroids = cv2.connectedComponentsWithStats(
        pixels,
        connectivity=8,
    )
    result = np.zeros_like(pixels)
    height, width = result.shape
    count = 0
    for component in range(1, labels):
        x, y, box_width, box_height, area = stats[component]
        if int(area) < max(1, int(min_area)):
            continue
        x1 = max(0, int(x) - int(padding))
        y1 = max(0, int(y) - int(padding))
        x2 = min(width, int(x + box_width) + int(padding))
        y2 = min(height, int(y + box_height) + int(padding))
        result[y1:y2, x1:x2] = 255
        count += 1
    return result, count


def apply_color_detection(
    state: dict[str, Any],
    image_ids: Iterable[str],
    colors: Iterable[str],
    **parameters: Any,
) -> tuple[int, int]:
    affected = 0
    region_count = 0
    for image_id in image_ids:
        item = find_item(state, image_id)
        if item is None:
            continue
        with Image.open(item["source_path"]) as opened:
            image = opened.convert("RGB").copy()
        with Image.open(item["mask_path"]) as opened:
            merged = np.asarray(opened.convert("L"), dtype=np.uint8).copy()
        item_regions = 0
        for color in colors:
            detected, count = color_bbox_mask(image, color, **parameters)
            merged = cv2.bitwise_or(merged, detected)
            item_regions += count
        if item_regions:
            Image.fromarray(merged, mode="L").save(item["mask_path"])
            item["revision"] = int(item["revision"]) + 1
            item["result_path"] = None
            item["status"] = "已标记"
            item["error"] = ""
            affected += 1
            region_count += item_regions
    return affected, region_count


def clear_active_mask(state: dict[str, Any]) -> dict[str, Any]:
    item = active_item(state)
    if item is None:
        raise ValueError("当前没有待修复图片")
    Image.new("L", (item["width"], item["height"]), 0).save(item["mask_path"])
    item["revision"] = int(item["revision"]) + 1
    item["result_path"] = None
    item["status"] = "待标记"
    item["error"] = ""
    return item


def repairable_ids(state: dict[str, Any], image_ids: Iterable[str]) -> list[str]:
    ready = []
    for image_id in image_ids:
        item = find_item(state, image_id)
        if item is None:
            continue
        with Image.open(item["mask_path"]) as mask:
            if mask.getbbox() is not None:
                ready.append(image_id)
    return ready


def repair_items(
    state: dict[str, Any],
    image_ids: Iterable[str],
    lama_runtime: Any,
) -> tuple[int, list[str]]:
    completed = 0
    failures: list[str] = []
    for image_id in image_ids:
        item = find_item(state, image_id)
        if item is None:
            continue
        try:
            with Image.open(item["source_path"]) as opened:
                source = opened.convert("RGB").copy()
            with Image.open(item["mask_path"]) as opened:
                mask = opened.convert("L").copy()
            result = lama_runtime.inpaint(source, mask)
            result_name = Path(item["name"]).stem + "_inpaint.png"
            result_path = Path(item["source_path"]).with_name(result_name)
            result.save(result_path)
            item["result_path"] = str(result_path)
            item["status"] = "已修复"
            item["error"] = ""
            completed += 1
        except Exception as exc:
            item["status"] = "失败"
            item["error"] = str(exc)
            failures.append(f"{item['name']}: {exc}")
    return completed, failures


def result_image(item: dict[str, Any] | None) -> Image.Image | None:
    path = Path(item["result_path"]) if item and item.get("result_path") else None
    if path is None or not path.is_file():
        return None
    with Image.open(path) as opened:
        return opened.convert("RGB").copy()


def repaired_paths(state: dict[str, Any], image_ids: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for image_id in image_ids:
        item = find_item(state, image_id)
        path = Path(item["result_path"]) if item and item.get("result_path") else None
        if path is not None and path.is_file():
            paths.append(str(path))
    if not paths:
        raise ValueError("所选图片尚未生成修复结果")
    return paths


__all__ = [
    "active_item",
    "add_uploaded_images",
    "apply_color_detection",
    "clear_active_mask",
    "color_bbox_mask",
    "commit_editor_value",
    "editor_payload",
    "empty_editor_payload",
    "move_active",
    "new_repair_state",
    "owned_repair_state",
    "queue_view",
    "repairable_ids",
    "remove_repair_items",
    "repair_items",
    "repaired_paths",
    "result_image",
    "select_item",
]
