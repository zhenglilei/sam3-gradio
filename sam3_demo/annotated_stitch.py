"""Annotated tile queue operations; masks never use preview pixels."""
from __future__ import annotations
import copy
import uuid
import gradio as gr
import numpy as np
from PIL import Image


def queue_view(state, selected=None):
    tiles = state.get("saved_tiles") or []
    choices = [(f"{i + 1}. {tile['name']} · {len(tile['instances'])} 个实例", tile["tile_id"])
               for i, tile in enumerate(tiles)]
    keys = [value for _label, value in choices]
    selected = selected if selected in keys else (keys[0] if keys else None)
    gallery = []
    for tile in tiles:
        preview = tile["image"].copy()
        preview.thumbnail((160, 120))
        gallery.append((preview, f"{tile['name']} · {len(tile['instances'])}"))
    return gallery, gr.update(choices=choices, value=selected)


def change_queue(state, tiles):
    from .stitch_callbacks import _invalidate_mosaic
    updated = dict(state)
    updated["saved_tiles"] = tiles
    updated["queue_dirty"] = bool(updated.get("annotated_mode"))
    _invalidate_mosaic(updated)
    return updated


def append_tiles(state, tiles):
    incoming = [dict(tile, tile_id=uuid.uuid4().hex) for tile in tiles]
    return change_queue(state, list(state.get("saved_tiles") or []) + incoming)


def edit_queue(state, selected, action):
    tiles = list(state.get("saved_tiles") or [])
    index = next((i for i, tile in enumerate(tiles) if tile["tile_id"] == selected), None)
    if index is None:
        raise ValueError("请先选择已保存的小图")
    if action == "remove":
        tiles.pop(index)
    else:
        other = index + (-1 if action == "up" else 1)
        if 0 <= other < len(tiles):
            tiles[index], tiles[other] = tiles[other], tiles[index]
    return change_queue(state, tiles)


def prepare_annotations(tiles, margins, border_records):
    annotations, records = [], []
    top, bottom, left, right = (int(margins[key]) for key in ("top", "bottom", "left", "right"))
    for index, tile in enumerate(tiles):
        width, height = tile["image"].size
        trim = border_records[index]["trim"] if border_records else {}
        x0, y0 = left + trim.get("left", 0), top + trim.get("top", 0)
        x1, y1 = width - right - trim.get("right", 0), height - bottom - trim.get("bottom", 0)
        current = []
        for original in tile["instances"]:
            inst = copy.deepcopy(original)
            mask = np.asarray(inst["mask"], dtype=bool)
            if mask.shape != (height, width):
                raise ValueError(f"{tile['name']} 的 mask 尺寸与图片不一致")
            inst["mask"] = mask[y0:y1, x0:x1].copy()
            inst["provenance"] = {**(inst.get("provenance") or {}),
                "source_tile_id": tile["tile_id"], "source_image_name": tile["name"],
                "source_instance_id": original["id"]}
            current.append(inst)
        annotations.append(current)
        records.append({"tile_id": tile["tile_id"], "name": tile["name"],
                        "source_size": [width, height], "source_crop_xyxy": [x0, y0, x1, y1],
                        "provenance": tile.get("provenance") or {}})
    return annotations, records


def preview_images(state, images):
    if not state.get("annotation_visible", True):
        return images
    annotations = state.get("annotations") or []
    alpha = max(0.0, min(1.0, float(state.get("annotation_alpha", .35))))
    result = []
    colors = ((255, 83, 83), (55, 207, 143), (255, 193, 58), (82, 151, 255))
    for index, image in enumerate(images):
        pixels = np.asarray(image).copy()
        for number, inst in enumerate(annotations[index] if index < len(annotations) else []):
            mask = np.asarray(inst["mask"], dtype=bool)
            pixels[mask] = np.rint(pixels[mask] * (1 - alpha) + np.asarray(colors[number % 4]) * alpha).astype(np.uint8)
        result.append(Image.fromarray(pixels))
    return result


def queue_result(state, selected, status, package=None):
    from .stitch_callbacks import _cleared_result_updates
    gallery, choice = queue_view(state, selected)
    state["status"] = status
    return state, gallery, choice, status, package, *_cleared_result_updates(), status
