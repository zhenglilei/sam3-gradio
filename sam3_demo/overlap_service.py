"""Application service for read-only overlap prediction and visualisation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from sam3_demo.overlap_geometry import (
    OverlapGeometryError,
    calculate_next_layer_overlap,
    ordered_layers,
    render_overlap_preview,
)
from sam3_demo.overlap_store import NotFoundError, OverlapLayoutStore


class OverlapServiceError(RuntimeError):
    """Raised when persisted overlap data cannot be rendered safely."""


def _decode_png(
    payload: bytes,
    *,
    expected_size: tuple[int, int],
    mode: str,
    field: str,
) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format != "PNG":
                raise OverlapServiceError(f"{field} is not a PNG image")
            source.load()
            image = source.convert(mode)
    except (OSError, UnidentifiedImageError) as exc:
        raise OverlapServiceError(f"{field} cannot be decoded") from exc
    if image.size != expected_size:
        raise OverlapServiceError(
            f"{field} size {image.size} does not match stored size {expected_size}"
        )
    return image


class OverlapPredictionService:
    """Read persisted layouts and calculate current-to-next layer overlap."""

    def __init__(self, store: OverlapLayoutStore) -> None:
        self._store = store

    def layout_choices(self) -> list[tuple[str, str]]:
        """Return Gradio-compatible labels and stable layout ids."""
        return [
            (
                f"{item['name']} · O3 · {item['width']}×{item['height']}",
                item["id"],
            )
            for item in self._store.list_layouts()
        ]

    def layer_choices(self, layout_id: str) -> list[tuple[str, str]]:
        """Return only layers which have a following layer to predict."""
        snapshot = self._store.get_layout(layout_id)
        layers = ordered_layers(snapshot["layers"])
        return [
            (
                f"{layer['ordinal'] + 1}. {layer['name']} ({layer['layer_type']})",
                layer["id"],
            )
            for layer in layers[:-1]
        ]

    def render(
        self,
        layout_id: str,
        current_layer_id: str,
        *,
        observed_current_transform: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render a stored current layer and its predicted successor."""
        try:
            snapshot = self._store.get_layout(layout_id)
        except NotFoundError as exc:
            raise OverlapServiceError("选择的叠层版图不存在或已删除") from exc

        base = _decode_png(
            snapshot["base_image_png"],
            expected_size=(snapshot["width"], snapshot["height"]),
            mode="RGB",
            field="base image",
        )
        geometry_layers: list[dict[str, Any]] = []
        for layer in snapshot["layers"]:
            mask_image = _decode_png(
                layer["mask_png"],
                expected_size=(layer["mask_width"], layer["mask_height"]),
                mode="L",
                field=f"layer {layer['name']} mask",
            )
            geometry_layers.append(
                {
                    "id": layer["id"],
                    "ordinal": layer["ordinal"],
                    "layer_type": layer["layer_type"],
                    "name": layer["name"],
                    "transform": layer["transform"],
                    "mask": np.asarray(mask_image, dtype=np.uint8) > 0,
                }
            )
        try:
            result = calculate_next_layer_overlap(
                geometry_layers,
                current_layer_id,
                (snapshot["width"], snapshot["height"]),
                observed_current_transform=observed_current_transform,
            )
        except OverlapGeometryError as exc:
            raise OverlapServiceError(str(exc)) from exc
        preview = render_overlap_preview(
            base,
            result["current_mask"],
            result["next_mask"],
        )
        metrics = result["metrics"]
        current = result["current_layer"]
        following = result["next_layer"]
        bbox = metrics["intersection_bbox_xyxy"]
        bbox_text = "无" if bbox is None else ", ".join(str(value) for value in bbox)
        report = (
            f"当前层：{current['name']} ({current['layer_type']})\n"
            f"下一层：{following['name']} ({following['layer_type']})\n"
            f"交叠面积：{metrics['intersection_area']:,} px\n"
            f"IoU：{metrics['iou']:.4f}\n"
            f"当前层覆盖率：{metrics['current_coverage']:.2%}\n"
            f"下一层覆盖率：{metrics['next_coverage']:.2%}\n"
            f"交叠框 [x1, y1, x2, y2]：{bbox_text}"
        )
        return {
            "layout": {
                key: value
                for key, value in snapshot.items()
                if key not in {"base_image_png", "layers"}
            },
            "current_layer": current,
            "next_layer": following,
            "preview": preview,
            "report": report,
            "metrics": metrics,
            "current_transform": result["current_transform"],
            "predicted_next_transform": result["predicted_next_transform"],
        }


__all__ = [
    "OverlapPredictionService",
    "OverlapServiceError",
]
