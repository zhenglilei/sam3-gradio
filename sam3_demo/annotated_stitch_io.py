"""Standalone IO for annotated stitch tiles."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import stat
import uuid
import zipfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


_MAX_ZIP_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
_PLACEHOLDER_IMAGE_NAMES = {"image", "mosaic", "source_image"}


def _rgb_copy(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    try:
        return image.convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"could not copy image as RGB: {exc}") from exc


def _instance_id(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise ValueError("instance id must be an int or str")
    return int(value) if isinstance(value, Integral) else value


def _finite_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("score must be a finite float")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("score must be a finite float")
    return score


def _mask_copy(mask: Any, expected_shape: Tuple[int, int]) -> np.ndarray:
    try:
        array = np.asarray(mask)
    except Exception as exc:
        raise ValueError("mask must be a two-dimensional array") from exc
    if array.ndim != 2 or tuple(array.shape) != expected_shape:
        raise ValueError(
            f"mask shape {tuple(array.shape)} does not match image shape "
            f"{expected_shape}"
        )
    try:
        return np.ascontiguousarray(array.astype(bool, copy=True), dtype=bool)
    except Exception as exc:
        raise ValueError("mask must be convertible to boolean values") from exc


def _dict_copy(value: Any, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a dict")
    return copy.deepcopy(dict(value))


def _normalize_instance(instance: Any, expected_shape: Tuple[int, int]) -> Dict[str, Any]:
    if not isinstance(instance, Mapping):
        raise ValueError("each instance must be a dict")
    for key in ("id", "category_name", "mask"):
        if key not in instance:
            raise ValueError(f"instance is missing {key}")
    if not isinstance(instance["category_name"], str):
        raise ValueError("category_name must be a str")
    normalized = {
        "id": _instance_id(instance["id"]),
        "category_name": instance["category_name"],
        "mask": _mask_copy(instance["mask"], expected_shape),
        "provenance": _dict_copy(
            instance.get("provenance"), "instance provenance"
        ),
    }
    if "score" in instance:
        normalized["score"] = _finite_score(instance["score"])
    return normalized


def make_tile(
    image: Image.Image,
    name: str,
    instances: list,
    tile_id: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Validate and detach a tile record."""
    if not isinstance(name, str):
        raise ValueError("name must be a str")
    if not isinstance(instances, list):
        raise ValueError("instances must be a list")
    rgb = _rgb_copy(image)
    shape = (rgb.height, rgb.width)
    normalized = [_normalize_instance(item, shape) for item in instances]
    if tile_id is None:
        normalized_id = uuid.uuid4().hex
    elif isinstance(tile_id, uuid.UUID):
        normalized_id = tile_id.hex
    elif isinstance(tile_id, str) and tile_id:
        normalized_id = tile_id
    else:
        raise ValueError("tile_id must be a non-empty str or UUID")
    return {
        "tile_id": normalized_id,
        "name": name,
        "image": rgb,
        "instances": normalized,
        "provenance": _dict_copy(provenance, "tile provenance"),
    }


def _coco_mask_utils():
    try:
        from pycocotools import mask as mask_utils
    except Exception as exc:
        raise RuntimeError("pycocotools is required for COCO mask IO") from exc
    return mask_utils


def _positive_dimension(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _decode_coco_segmentation(
    segmentation: Any, height: int, width: int
) -> np.ndarray:
    mask_utils = _coco_mask_utils()
    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("COCO polygon segmentation must not be empty")
            polygons = segmentation
            if all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in polygons
            ):
                polygons = [polygons]
            normalized = []
            for index, polygon in enumerate(polygons):
                if not isinstance(polygon, (list, tuple)):
                    raise ValueError(
                        f"COCO polygon {index} must be a coordinate list"
                    )
                if len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError(
                        f"COCO polygon {index} must contain at least 3 xy points"
                    )
                coordinates = np.asarray(polygon, dtype=np.float64)
                if not np.isfinite(coordinates).all():
                    raise ValueError(
                        f"COCO polygon {index} contains a non-finite coordinate"
                    )
                normalized.append(coordinates.tolist())
            encoded = mask_utils.frPyObjects(normalized, height, width)
            rle = mask_utils.merge(encoded)
        elif isinstance(segmentation, Mapping):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise ValueError("COCO RLE size must be [height, width]")
            rle_shape = (
                _positive_dimension(size[0], "RLE height"),
                _positive_dimension(size[1], "RLE width"),
            )
            if rle_shape != (height, width):
                raise ValueError(
                    f"COCO RLE size {rle_shape} does not match "
                    f"{(height, width)}"
                )
            if isinstance(counts, list):
                if not counts:
                    raise ValueError(
                        "uncompressed COCO RLE counts must not be empty"
                    )
                counts = [
                    int(count)
                    if not isinstance(count, bool)
                    and isinstance(count, Integral)
                    and int(count) >= 0
                    else None
                    for count in counts
                ]
                if any(count is None for count in counts):
                    raise ValueError(
                        "COCO RLE counts must be non-negative integers"
                    )
                if sum(counts) != height * width:
                    raise ValueError(
                        "uncompressed COCO RLE counts do not cover the image"
                    )
                rle = mask_utils.frPyObjects(
                    {"size": [height, width], "counts": counts},
                    height,
                    width,
                )
            elif isinstance(counts, str):
                try:
                    counts = counts.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "compressed COCO RLE counts must be ASCII"
                    ) from exc
                if not counts:
                    raise ValueError(
                        "compressed COCO RLE counts must not be empty"
                    )
                rle = {"size": [height, width], "counts": counts}
            elif isinstance(counts, bytes) and counts:
                rle = {"size": [height, width], "counts": counts}
            else:
                raise ValueError(
                    "COCO RLE counts must be a list, string, or bytes"
                )
        else:
            raise ValueError(
                "COCO segmentation must be a polygon list or RLE object"
            )
        decoded = np.asarray(mask_utils.decode(rle))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid COCO segmentation: {exc}") from exc
    if decoded.ndim != 2 or decoded.shape != (height, width):
        raise ValueError(
            f"decoded COCO mask shape {decoded.shape} does not match "
            f"{(height, width)}"
        )
    if not np.isin(decoded, (0, 1)).all():
        raise ValueError("decoded COCO mask is not binary")
    return np.ascontiguousarray(decoded.astype(bool))


def _load_image_file(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            opened.load()
            return opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not read image {path}: {exc}") from exc


def _load_image_bytes(data: bytes, source: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            return opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not read image {source}: {exc}") from exc


def _json_object(source: str, data: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON {source} must contain an object")
    return payload


def _normalized_reference(value: str) -> str:
    return value.replace("\\", "/")


def _path_identity(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(str(path.absolute()))


def _resolve_external_image(
    reference: str,
    candidates: Sequence[Path],
    json_path: Path,
    *,
    allow_placeholder: bool = False,
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise FileNotFoundError(f"JSON {json_path} does not name an image")
    normalized = _normalized_reference(reference)
    reference_path = Path(normalized)
    matches: List[Path] = []
    if reference_path.is_absolute() or PureWindowsPath(normalized).drive:
        target = _path_identity(reference_path)
        matches = [
            item for item in candidates if _path_identity(item) == target
        ]
    else:
        target = _path_identity(json_path.parent / normalized)
        matches = [
            item for item in candidates if _path_identity(item) == target
        ]
        if not matches and ".." not in PurePosixPath(normalized).parts:
            suffix = tuple(
                part.casefold()
                for part in PurePosixPath(normalized).parts
                if part not in ("", ".")
            )
            if suffix:
                matches = [
                    item
                    for item in candidates
                    if tuple(
                        part.casefold()
                        for part in item.parts[-len(suffix) :]
                    )
                    == suffix
                ]
        if not matches and ".." not in PurePosixPath(normalized).parts:
            basename = PurePosixPath(normalized).name.casefold()
            matches = [
                item
                for item in candidates
                if item.name.casefold() == basename
            ]
    if len(matches) > 1:
        raise ValueError(
            f"image reference {reference!r} in {json_path} is ambiguous"
        )
    if matches:
        return matches[0]
    if allow_placeholder and len(candidates) == 1:
        if PurePosixPath(normalized).stem.casefold() in _PLACEHOLDER_IMAGE_NAMES:
            return candidates[0]
    raise FileNotFoundError(
        f"image referenced by {json_path} is missing: {reference}"
    )


def _validate_image_dimensions(
    image: Image.Image, width: Any, height: Any, source: str
) -> None:
    expected_width = _positive_dimension(width, f"{source} width")
    expected_height = _positive_dimension(height, f"{source} height")
    actual = (image.width, image.height)
    if actual != (expected_width, expected_height):
        raise ValueError(
            f"image {source} has size {actual}, expected "
            f"{(expected_width, expected_height)}"
        )


def _tile_metadata(
    manifest: Optional[dict], file_name: str
) -> Tuple[str, Optional[str], dict]:
    default_name = PurePosixPath(_normalized_reference(file_name)).name
    if not default_name:
        default_name = "mosaic.png"
    if not manifest:
        return default_name, None, {}
    source_name = manifest.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        source_name = manifest.get("name")
    if not isinstance(source_name, str) or not source_name:
        source_name = default_name
    tile_id = manifest.get("tile_id")
    if not isinstance(tile_id, str) or not tile_id:
        tile_id = None
    provenance = manifest.get("provenance")
    if isinstance(provenance, Mapping):
        provenance = copy.deepcopy(dict(provenance))
    else:
        provenance = copy.deepcopy(manifest)
    return source_name, tile_id, provenance


def _parse_coco_document(
    data: Mapping[str, Any],
    source: str,
    image_loader: Callable[[str], Tuple[Image.Image, str]],
    *,
    manifest: Optional[dict] = None,
) -> List[dict]:
    image_records = data.get("images")
    if not isinstance(image_records, list):
        raise ValueError(f"COCO JSON {source} is missing an images list")
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"COCO JSON {source} annotations must be a list")
    categories = data.get("categories", [])
    if not isinstance(categories, list):
        raise ValueError(f"COCO JSON {source} categories must be a list")

    category_names: Dict[Any, str] = {}
    for category in categories:
        if not isinstance(category, Mapping) or "id" not in category:
            raise ValueError(f"COCO JSON {source} contains an invalid category")
        category_id = _instance_id(category["id"])
        if category_id in category_names:
            raise ValueError(f"duplicate COCO category id {category_id!r}")
        if not isinstance(category.get("name"), str):
            raise ValueError("COCO category name must be a str")
        category_names[category_id] = category["name"]

    image_ids: Dict[Any, Mapping[str, Any]] = {}
    for record in image_records:
        if not isinstance(record, Mapping):
            raise ValueError(f"COCO JSON {source} contains an invalid image record")
        if "id" not in record or "file_name" not in record:
            raise ValueError(
                f"COCO JSON {source} image record is missing id or file_name"
            )
        image_id = _instance_id(record["id"])
        if image_id in image_ids:
            raise ValueError(f"duplicate COCO image id {image_id!r}")
        if not isinstance(record["file_name"], str) or not record["file_name"]:
            raise ValueError(f"COCO image file_name must be a non-empty str in {source}")
        _positive_dimension(record.get("width"), "COCO image width")
        _positive_dimension(record.get("height"), "COCO image height")
        image_ids[image_id] = record

    by_image: Dict[Any, List[Tuple[int, Mapping[str, Any]]]] = {}
    seen_annotation_ids = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"COCO JSON {source} contains an invalid annotation")
        if "image_id" not in annotation or "segmentation" not in annotation:
            raise ValueError(
                f"COCO annotation in {source} is missing image_id or segmentation"
            )
        image_id = _instance_id(annotation["image_id"])
        if image_id not in image_ids:
            raise ValueError(
                f"COCO annotation references missing image_id {image_id!r}"
            )
        annotation_id = _instance_id(annotation.get("id", index + 1))
        if annotation_id in seen_annotation_ids:
            raise ValueError(f"duplicate COCO annotation id {annotation_id!r}")
        seen_annotation_ids.add(annotation_id)
        by_image.setdefault(image_id, []).append((index, annotation))

    result = []
    for record in image_records:
        image_id = _instance_id(record["id"])
        image, image_source = image_loader(record["file_name"])
        _validate_image_dimensions(
            image, record["width"], record["height"], image_source
        )
        instances = []
        for index, annotation in by_image.get(image_id, []):
            category_id = annotation.get("category_id")
            if category_id is not None:
                category_id = _instance_id(category_id)
            category_name = category_names.get(category_id)
            if category_name is None and isinstance(
                annotation.get("category_name"), str
            ):
                category_name = annotation["category_name"]
            if category_name is None:
                raise ValueError(
                    f"COCO annotation references unknown category_id {category_id!r}"
                )
            mask = _decode_coco_segmentation(
                annotation["segmentation"], image.height, image.width
            )
            annotation_id = _instance_id(annotation.get("id", index + 1))
            instance_id = _instance_id(
                annotation.get("instance_id", annotation_id)
            )
            provenance = {
                "source_format": "coco",
                "source_json": source,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "category_id": category_id,
            }
            if isinstance(annotation.get("provenance"), Mapping):
                provenance.update(
                    copy.deepcopy(dict(annotation["provenance"]))
                )
            instance = {
                "id": instance_id,
                "category_name": category_name,
                "mask": mask,
                "provenance": provenance,
            }
            if "score" in annotation:
                instance["score"] = _finite_score(annotation["score"])
            instances.append(instance)
        name, tile_id, provenance = _tile_metadata(
            manifest, record["file_name"]
        )
        if not manifest:
            provenance = {
                "source_format": "coco",
                "source_json": source,
                "image_id": image_id,
                "file_name": record["file_name"],
            }
        result.append(
            make_tile(
                image,
                name,
                instances,
                tile_id=tile_id,
                provenance=provenance,
            )
        )
    return result


def _decode_labelme_image_data(
    data: Mapping[str, Any], source: str
) -> Optional[Image.Image]:
    encoded = data.get("imageData")
    if not encoded:
        return None
    if not isinstance(encoded, str):
        raise ValueError(f"LabelMe imageData must be a base64 str in {source}")
    if "," in encoded and encoded.lstrip().lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid LabelMe imageData in {source}") from exc
    return _load_image_bytes(raw, source)


def _labelme_polygon(shape: Mapping[str, Any], source: str) -> List[float]:
    shape_type = shape.get("shape_type") or "polygon"
    points = shape.get("points")
    if shape_type == "rectangle":
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError(
                f"LabelMe rectangle in {source} must have exactly two points"
            )
        try:
            x1, y1 = float(points[0][0]), float(points[0][1])
            x2, y2 = float(points[1][0]), float(points[1][1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid LabelMe rectangle in {source}") from exc
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif shape_type != "polygon":
        raise ValueError(
            f"unsupported LabelMe shape_type {shape_type!r} in {source}"
        )
    elif not isinstance(points, list) or len(points) < 3:
        raise ValueError(
            f"LabelMe polygon in {source} must have at least three points"
        )
    flattened = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"invalid LabelMe point in {source}")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid LabelMe point in {source}") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"non-finite LabelMe point in {source}")
        flattened.extend((x, y))
    return flattened


def _rasterize_polygon(
    points: List[float], height: int, width: int
) -> np.ndarray:
    mask_utils = _coco_mask_utils()
    try:
        rle = mask_utils.merge(mask_utils.frPyObjects([points], height, width))
        decoded = np.asarray(mask_utils.decode(rle))
    except Exception as exc:
        raise ValueError(f"could not rasterize LabelMe polygon: {exc}") from exc
    if decoded.ndim != 2 or decoded.shape != (height, width):
        raise ValueError("LabelMe polygon produced a mask with wrong dimensions")
    return np.ascontiguousarray(decoded.astype(bool))


def _parse_labelme_document(
    data: Mapping[str, Any],
    source: str,
    image_loader: Callable[[Optional[str]], Tuple[Image.Image, str]],
) -> dict:
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"LabelMe JSON {source} is missing a shapes list")
    image_path = data.get("imagePath")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError(f"LabelMe imagePath must be a str in {source}")
    embedded = _decode_labelme_image_data(data, source)
    if embedded is not None:
        image, image_source = embedded, source + "#imageData"
    else:
        image, image_source = image_loader(image_path)
    if data.get("imageWidth") is not None:
        _validate_image_dimensions(image, data["imageWidth"], image.height, source)
    if data.get("imageHeight") is not None:
        _validate_image_dimensions(image, image.width, data["imageHeight"], source)

    groups: Dict[Any, dict] = {}
    for index, shape in enumerate(shapes):
        if not isinstance(shape, Mapping):
            raise ValueError(f"LabelMe shape {index} in {source} must be an object")
        label = shape.get("label", "object")
        if label is None:
            label = "object"
        if not isinstance(label, str):
            raise ValueError(f"LabelMe shape label {index} must be a str")
        mask = _rasterize_polygon(
            _labelme_polygon(shape, source), image.height, image.width
        )
        group_id = shape.get("group_id")
        if group_id is None:
            key = ("shape", index)
        else:
            try:
                group_key = json.dumps(
                    group_id, ensure_ascii=True, sort_keys=True
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid LabelMe group_id in {source}") from exc
            key = ("group", group_key, label)
        if key not in groups:
            groups[key] = {
                "id": index + 1,
                "category_name": label,
                "mask": mask,
                "provenance": {
                    "source_format": "labelme",
                    "source_json": source,
                    "shape_indices": [index],
                },
            }
            if group_id is not None:
                groups[key]["provenance"]["group_id"] = copy.deepcopy(group_id)
        else:
            groups[key]["mask"] |= mask
            groups[key]["provenance"]["shape_indices"].append(index)
    name = (
        PurePosixPath(_normalized_reference(image_path)).name
        if image_path
        else Path(source).stem
    )
    return make_tile(
        image,
        name or "labelme",
        list(groups.values()),
        provenance={
            "source_format": "labelme",
            "source_json": source,
            "image_path": image_path,
            "image_source": image_source,
        },
    )


def _archive_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if "\x00" in normalized or normalized.startswith("/"):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    windows_path = PureWindowsPath(normalized)
    if windows_path.drive or windows_path.is_absolute():
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    clean = "/".join(part for part in parts if part not in ("", "."))
    if not clean:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return clean


def _zip_members(zip_path: Path) -> Dict[str, bytes]:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = {}
            folded = set()
            total = 0
            for info in archive.infolist():
                name = _archive_member_name(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP symlink member is not allowed: {info.filename!r}"
                    )
                if name.casefold() in folded:
                    raise ValueError(f"duplicate ZIP member name: {name}")
                folded.add(name.casefold())
                names[name] = info
                total += max(0, int(info.file_size))
                if total > _MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise ValueError("ZIP uncompressed size exceeds 512 MiB")
            # Do not extract or read mask PNGs unless a JSON image record needs one.
            wanted = {}
            for name, info in names.items():
                if (
                    name.casefold().endswith(".json")
                    or Path(name).name.casefold() == "mosaic.png"
                ):
                    wanted[name] = archive.read(info)
            return wanted
    except ValueError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"could not read ZIP {zip_path}: {exc}") from exc


def _zip_lookup(
    entries: Mapping[str, bytes], reference: str
) -> str:
    normalized = _normalized_reference(reference).strip("/")
    exact = [
        name for name in entries if name.casefold() == normalized.casefold()
    ]
    if exact:
        return exact[0]
    if ".." not in PurePosixPath(normalized).parts:
        basename = PurePosixPath(normalized).name.casefold()
        matches = [
            name
            for name in entries
            if PurePosixPath(name).name.casefold() == basename
        ]
        if len(matches) > 1:
            raise ValueError(f"ZIP image reference is ambiguous: {reference!r}")
        if matches:
            return matches[0]
    if PurePosixPath(normalized).stem.casefold() in _PLACEHOLDER_IMAGE_NAMES:
        mosaic = [
            name
            for name in entries
            if PurePosixPath(name).name.casefold() == "mosaic.png"
        ]
        if len(mosaic) == 1:
            return mosaic[0]
    raise FileNotFoundError(f"image referenced in ZIP is missing: {reference}")


def _zip_bundle_tiles(zip_path: Path) -> List[dict]:
    entries = _zip_members(zip_path)
    json_entries = {
        name: data
        for name, data in entries.items()
        if name.casefold().endswith(".json")
    }
    manifest_names = [
        name
        for name in json_entries
        if Path(name).name.casefold() == "manifest.json"
    ]
    if len(manifest_names) > 1:
        raise ValueError("ZIP contains ambiguous manifest.json files")
    manifest = (
        _json_object(
            f"{zip_path}:{manifest_names[0]}",
            json_entries[manifest_names[0]],
        )
        if manifest_names
        else None
    )
    coco_names = [
        name
        for name in json_entries
        if Path(name).name.casefold()
        in {"coco_predictions.json", "coco_masks.json"}
    ]
    if len(coco_names) > 1:
        raise ValueError("ZIP contains ambiguous COCO JSON files")
    if not coco_names:
        raise ValueError(
            f"ZIP {zip_path} contains no supported COCO prediction JSON"
        )
    source = f"{zip_path}:{coco_names[0]}"
    data = _json_object(source, json_entries[coco_names[0]])

    def load_image(reference: str) -> Tuple[Image.Image, str]:
        member = _zip_lookup(entries, reference)
        if member not in entries:
            raise FileNotFoundError(
                f"image referenced in ZIP is missing: {reference}"
            )
        return _load_image_bytes(entries[member], f"{zip_path}:{member}"), member

    return _parse_coco_document(data, source, load_image, manifest=manifest)


def _load_json_path(path: Path) -> Dict[str, Any]:
    try:
        return _json_object(str(path), path.read_bytes())
    except OSError as exc:
        raise FileNotFoundError(f"could not read JSON {path}: {exc}") from exc


def _image_candidates(paths: Sequence[Path]) -> List[Path]:
    candidates = [
        path for path in paths if path.suffix.casefold() in _IMAGE_SUFFIXES
    ]
    seen = {}
    for path in candidates:
        key = path.name.casefold()
        if key in seen:
            raise ValueError(
                f"image name is ambiguous: {path.name} "
                f"({seen[key]} and {path})"
            )
        seen[key] = path
    return candidates


def import_tiles(files: list) -> list:
    """Import image-only files, COCO/LabelMe pairs, or an exported ZIP."""
    if not isinstance(files, list):
        raise ValueError("files must be a list")
    paths = []
    for value in files:
        try:
            path = Path(value)
        except TypeError as exc:
            raise ValueError(f"unsupported input path: {value!r}") from exc
        if not path.exists():
            raise FileNotFoundError(f"input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"input path is not a file: {path}")
        paths.append(path)
    images = _image_candidates(paths)
    json_paths = [
        path for path in paths if path.suffix.casefold() == ".json"
    ]
    zip_paths = [
        path for path in paths if path.suffix.casefold() == ".zip"
    ]
    supported = set(images) | set(json_paths) | set(zip_paths)
    if any(path not in supported for path in paths):
        bad = next(path for path in paths if path not in supported)
        raise ValueError(f"unsupported input file type: {bad}")
    seen_json_names = set()
    for path in json_paths:
        if path.name.casefold() in seen_json_names:
            raise ValueError(f"JSON name is ambiguous: {path.name}")
        seen_json_names.add(path.name.casefold())

    used_images = set()
    tiles = []

    def append(tile: dict) -> None:
        if any(item["name"].casefold() == tile["name"].casefold() for item in tiles):
            raise ValueError(f"tile name is ambiguous: {tile['name']}")
        tiles.append(tile)

    for zip_path in zip_paths:
        for tile in _zip_bundle_tiles(zip_path):
            append(tile)

    for json_path in json_paths:
        data = _load_json_path(json_path)
        if isinstance(data.get("shapes"), list):
            def load_labelme(reference: Optional[str]) -> Tuple[Image.Image, str]:
                if reference:
                    selected = _resolve_external_image(
                        reference, images, json_path
                    )
                elif len(images) == 1:
                    selected = images[0]
                elif not images:
                    raise FileNotFoundError(
                        f"LabelMe JSON {json_path} has no paired image"
                    )
                else:
                    raise ValueError(
                        f"LabelMe JSON {json_path} has ambiguous image files"
                    )
                used_images.add(_path_identity(selected))
                return _load_image_file(selected), str(selected)

            append(_parse_labelme_document(data, str(json_path), load_labelme))
        elif "images" in data or "annotations" in data:
            def load_coco(reference: str) -> Tuple[Image.Image, str]:
                selected = _resolve_external_image(
                    reference, images, json_path, allow_placeholder=True
                )
                used_images.add(_path_identity(selected))
                return _load_image_file(selected), str(selected)

            for tile in _parse_coco_document(
                data, str(json_path), load_coco
            ):
                append(tile)
        else:
            raise ValueError(f"unsupported annotation JSON schema: {json_path}")

    for image_path in images:
        if _path_identity(image_path) in used_images:
            continue
        append(
            make_tile(
                _load_image_file(image_path),
                image_path.name,
                [],
                provenance={
                    "source_format": "image",
                    "source_image": str(image_path),
                },
            )
        )
    return tiles


def _mask_bbox(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    ]


def _encode_binary_mask(mask: np.ndarray) -> dict:
    from sam3_demo.segmentation_evaluation import encode_binary_mask

    return encode_binary_mask(mask)


def export_bundle(
    image: Image.Image, instances: list, directory: Path, manifest: dict
) -> str:
    """Write a self-contained bundle containing mosaic, masks, COCO, manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a dict")
    tile = make_tile(image, "mosaic.png", instances)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"annotated_stitch_{uuid.uuid4().hex}.zip"

    category_names = sorted(
        {instance["category_name"] for instance in tile["instances"]}
    )
    category_ids = {
        name: index + 1 for index, name in enumerate(category_names)
    }
    annotations = []
    masks = []
    skipped_empty = []
    for index, instance in enumerate(tile["instances"], start=1):
        mask = instance["mask"]
        if not mask.any():
            skipped_empty.append(
                {
                    "index": index,
                    "id": instance["id"],
                    "category_name": instance["category_name"],
                }
            )
            continue
        mask_name = f"masks/instance_{index:06d}.png"
        mask_buffer = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            mask_buffer, format="PNG"
        )
        masks.append((mask_name, mask_buffer.getvalue()))
        annotation = {
            "id": len(annotations) + 1,
            "image_id": 1,
            "category_id": category_ids[instance["category_name"]],
            "segmentation": _encode_binary_mask(mask),
            "area": int(mask.sum()),
            "bbox": _mask_bbox(mask),
            "iscrowd": 0,
            "mask_file": mask_name,
            "instance_id": instance["id"],
            "provenance": copy.deepcopy(instance["provenance"]),
        }
        if "score" in instance:
            annotation["score"] = instance["score"]
        annotations.append(annotation)

    manifest_payload = copy.deepcopy(dict(manifest))
    if skipped_empty:
        manifest_payload["skipped_empty_instances"] = skipped_empty
    coco_payload = {
        "info": {
            "description": "Annotated stitch instance masks",
            "version": "1.0",
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "mosaic.png",
                "width": tile["image"].width,
                "height": tile["image"].height,
            }
        ],
        "annotations": annotations,
        "categories": [
            {
                "id": category_ids[name],
                "name": name,
                "supercategory": "object",
            }
            for name in category_names
        ],
    }
    try:
        manifest_bytes = json.dumps(
            manifest_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
        coco_bytes = json.dumps(
            coco_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"manifest or instance provenance is not JSON serializable: {exc}"
        ) from exc
    mosaic_buffer = io.BytesIO()
    tile["image"].save(mosaic_buffer, format="PNG")
    try:
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("mosaic.png", mosaic_buffer.getvalue())
            for name, data in masks:
                archive.writestr(name, data)
            archive.writestr("coco_predictions.json", coco_bytes)
            archive.writestr("manifest.json", manifest_bytes)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        try:
            zip_path.unlink()
        except OSError:
            pass
        raise ValueError(f"could not write bundle {zip_path}: {exc}") from exc
    return str(zip_path)


__all__ = ["export_bundle", "import_tiles", "make_tile"]
