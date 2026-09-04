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
from PIL import Image, UnidentifiedImageError

from layout_transform_utils import (
    foreground_bbox_xyxy,
    make_layout_transform_v2,
    mask_pixel_sha256,
)
from sam3_demo.overlap_store import OverlapLayoutStore, ValidationError


AUTHORING_SCHEMA_VERSION = 1
LAYER_TYPES = ("BSM", "ACT", "GE1", "GE2")
MAX_LAYERS = 32


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


def _data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _selection_signature(draft: dict[str, Any]) -> str:
    identity = {
        "draft_id": draft["draft_id"],
        "draft_revision": int(draft["draft_revision"]),
        "layers": [
            {
                "id": layer["id"],
                "ordinal": int(layer["ordinal"]),
                "mask_sha256": layer["mask_sha256"],
            }
            for layer in draft["layers"]
        ],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    ) -> dict[str, Any]:
        out = _require_draft(draft)
        if not out.get("base_image_png"):
            raise OverlapAuthoringError("请先上传整体版图")
        if len(out["layers"]) >= MAX_LAYERS:
            raise OverlapAuthoringError(f"最多支持 {MAX_LAYERS} 层")
        layer_type = str(layer_type or "").strip().upper()
        if layer_type not in LAYER_TYPES:
            raise OverlapAuthoringError("层分类必须是 BSM、ACT、GE1 或 GE2")
        layer_name = str(name or "").strip() or f"{layer_type} {len(out['layers']) + 1}"
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
        if not out.get("base_image_png"):
            return {"enabled": False, "status": "请先上传整体版图"}
        if not out["layers"]:
            return {
                "enabled": False,
                "base_image": _data_url(out["base_image_png"]),
                "target_width": out["width"],
                "target_height": out["height"],
                "status": "请上传至少一层标注",
            }
        signature = _selection_signature(out)
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
                    "label": f"{layer['ordinal'] + 1}. {layer['name']} ({layer['layer_type']})",
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
        return {
            "enabled": True,
            "base_image": _data_url(out["base_image_png"]),
            "transform": copy.deepcopy(active_transform),
            "transform_mode": "label_groups",
            "group_view": {
                "selection_signature": signature,
                "regions_revision": int(out["draft_revision"]),
                "allow_reorder": True,
                "groups": groups,
            },
            "group_intent": {
                "selection_signature": signature,
                "transform_set_revision": int(out["draft_revision"]),
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

    def apply_editor_intent(
        self, draft: Any, editor_payload: Any
    ) -> dict[str, Any]:
        current = _require_draft(draft)
        if not isinstance(editor_payload, dict):
            raise OverlapAuthoringError("编辑器返回值无效")
        intent = editor_payload.get("group_intent")
        if not isinstance(intent, dict):
            raise OverlapAuthoringError("编辑器缺少 group_intent")
        if intent.get("selection_signature") != _selection_signature(current):
            raise OverlapAuthoringError("编辑器数据已过期，请重试")
        if intent.get("transform_set_revision") != current["draft_revision"] + 1:
            raise OverlapAuthoringError("编辑器 revision 已过期，请重试")

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
        current["draft_revision"] += 1
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
        }

    def save(self, draft: Any) -> dict[str, Any]:
        current = _require_draft(draft)
        if not current.get("name", "").strip():
            raise OverlapAuthoringError("请输入版图名称")
        if not current.get("base_image_png"):
            raise OverlapAuthoringError("请上传整体版图")
        if len(current["layers"]) < 2:
            raise OverlapAuthoringError("至少需要两层标注")
        payload_layers = [
            {
                "id": layer["id"],
                "ordinal": ordinal,
                "layer_type": layer["layer_type"],
                "name": layer["name"],
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
    "OverlapAuthoringError",
    "OverlapAuthoringService",
]
