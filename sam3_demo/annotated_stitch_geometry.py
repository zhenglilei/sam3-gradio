"""Geometry-preserving instance-mask composition for stitched annotations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from .stitch_workflow import (
    crop_to_complete_periods_bounds,
    detect_period,
    export_mosaic,
    normalize_rotations,
    phase_offset,
    stitch_canvas_geometry,
)


def _drop(instance: dict, reason: str) -> dict:
    return {
        "source_tile_index": instance.get("source_tile_index"),
        "source_instance_id": instance.get("source_instance_id"),
        "reason": reason,
    }


def crop_annotations(
    instances: Sequence[dict], bbox_xyxy: Sequence[int],
) -> tuple[list[dict], list[dict]]:
    """Crop mosaic-sized instances and record those left without foreground."""

    left, top, right, bottom = (int(value) for value in bbox_xyxy)
    kept: list[dict] = []
    dropped: list[dict] = []
    for instance in instances:
        cropped = np.asarray(instance["mask"], dtype=bool)[top:bottom, left:right].copy()
        if not cropped.any():
            dropped.append(_drop(instance, "crop_empty"))
            continue
        item = dict(instance)
        item["mask"] = cropped
        kept.append(item)
    return kept, dropped


def _source_mask(instance: dict, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(instance.get("mask"), dtype=bool)
    expected = (size[1], size[0])
    if mask.shape != expected:
        raise ValueError(f"实例 mask 尺寸 {mask.shape} 与图块尺寸 {expected} 不一致")
    return mask


def _instance_with_source(instance: dict, tile_index: int, mask: np.ndarray) -> dict:
    item = dict(instance)
    provenance = dict(instance.get("provenance") or {})
    source_id = instance.get("id")
    provenance["source_tile_index"] = tile_index
    provenance["source_instance_id"] = source_id
    item["provenance"] = provenance
    item["source_tile_index"] = tile_index
    item["source_instance_id"] = source_id
    item["mask"] = mask.astype(bool, copy=False)
    return item


def _rotate_mask(mask: np.ndarray, angle: float, expanded_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if angle:
        image = image.rotate(
            -angle,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
    if image.size != tuple(expanded_size):
        raise RuntimeError("mask 旋转扩展尺寸与图片拼接几何不一致")
    return np.asarray(image, dtype=np.uint8) > 0


def _rotated_coverage(size: tuple[int, int], angle: float, expanded_size: tuple[int, int]) -> np.ndarray:
    image = Image.new("L", size, 255)
    if angle:
        image = image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
        )
    if image.size != tuple(expanded_size):
        raise RuntimeError("图片透明覆盖尺寸与拼接几何不一致")
    return np.asarray(image, dtype=np.uint8) > 0


def compose_annotated_mosaic(
    images: Sequence[Image.Image],
    annotations: Sequence[Sequence[dict]],
    shifts: Sequence[tuple[int, int]],
    layout: str = "horizontal",
    blend: bool = True,
    crop_periodic: bool = False,
    rotations: Sequence[float] | None = None,
) -> tuple[Image.Image, list[dict], dict]:
    """Export a mosaic and independently transform each source instance mask."""

    if len(annotations) != len(images):
        raise ValueError("annotations 数量与图片不一致")
    normalized_rotations = normalize_rotations(rotations, len(images))
    geometry = stitch_canvas_geometry(
        images,
        shifts,
        layout=layout,
        blend=blend,
        rotations=normalized_rotations,
    )
    mosaic, _warnings = export_mosaic(
        images,
        shifts,
        layout=layout,
        blend=blend,
        crop_periodic=crop_periodic,
        rotations=normalized_rotations,
    )
    pieces_by_tile = {piece["tile_index"]: piece for piece in geometry["pieces"]}
    instances: list[dict] = []
    dropped: list[dict] = []
    for tile_index, tile_instances in enumerate(annotations):
        piece = pieces_by_tile.get(tile_index)
        for instance in tile_instances:
            source = _source_mask(instance, images[tile_index].size)
            if piece is None:
                dropped.append(
                    {
                        "source_tile_index": tile_index,
                        "source_instance_id": instance.get("id"),
                        "reason": "layout_empty",
                    }
                )
                continue
            if geometry["rotated"]:
                transformed = _rotate_mask(
                    source,
                    piece["rotation_deg"],
                    piece["expanded_size"],
                )
            else:
                left, top, right, bottom = piece["source_bbox_xyxy"]
                transformed = source[top:bottom, left:right]
            canvas = np.zeros((geometry["canvas_size"][1], geometry["canvas_size"][0]), dtype=bool)
            ox, oy = piece["origin_xy"]
            height, width = transformed.shape
            canvas[oy:oy + height, ox:ox + width] = transformed
            item = _instance_with_source(instance, tile_index, canvas)
            if item["mask"].any():
                instances.append(item)
            else:
                dropped.append(_drop(item, "mask_empty"))

    if geometry["rotated"] and not blend:
        covered = np.zeros((geometry["canvas_size"][1], geometry["canvas_size"][0]), dtype=bool)
        for tile_index in range(len(images) - 1, -1, -1):
            for item in instances:
                if item["source_tile_index"] == tile_index:
                    item["mask"] &= ~covered
            piece = pieces_by_tile[tile_index]
            coverage = _rotated_coverage(
                images[tile_index].size,
                piece["rotation_deg"],
                piece["expanded_size"],
            )
            ox, oy = piece["origin_xy"]
            height, width = coverage.shape
            covered[oy:oy + height, ox:ox + width] |= coverage
        retained: list[dict] = []
        for item in instances:
            if item["mask"].any():
                retained.append(item)
            else:
                dropped.append(_drop(item, "covered_by_later_tile"))
        instances = retained

    crop_bbox = (0, 0, geometry["canvas_size"][0], geometry["canvas_size"][1])
    if crop_periodic:
        gray = np.array(images[0].convert("L"), dtype=np.float32)
        px, py, phase_x, phase_y = detect_period(gray)
        crop_bbox = crop_to_complete_periods_bounds(
            geometry["canvas_size"],
            px,
            py,
            phase_offset(phase_x, px),
            phase_offset(phase_y, py),
        )
        instances, crop_dropped = crop_annotations(instances, crop_bbox)
        dropped.extend(crop_dropped)

    tiles = []
    for piece in geometry["pieces"]:
        tiles.append(
            {
                "source_tile_index": piece["tile_index"],
                "source_bbox_xyxy": list(piece["source_bbox_xyxy"]),
                "origin_xy": list(piece["origin_xy"]),
                "raw_origin_xy": list(piece["raw_origin_xy"]),
                "rotation_deg": piece["rotation_deg"],
                "expanded_size": list(piece["expanded_size"]),
                "canvas_normalization_xy": list(geometry["canvas_normalization_xy"]),
            }
        )
    return mosaic, instances, {
        "tiles": tiles,
        "canvas_size": list(geometry["canvas_size"]),
        "crop_bbox_xyxy": list(crop_bbox),
        "dropped_instances": dropped,
    }
