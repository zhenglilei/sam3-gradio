"""Strict, model-free primitives for offline PCS/PVS evaluation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment


RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_NAME = "run_manifest.json"


def _json_error(message: str) -> None:
    raise ValueError(message)


def _reject_constant(value: str) -> None:
    _json_error(f"Non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _json_error(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _json_error("Non-finite JSON number is not allowed")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_numbers(item)


def _strict_json_loads(text: str, *, source: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    _validate_json_numbers(value)
    return value


def _json_bytes(value: Any) -> bytes:
    _validate_json_numbers(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _read_regular_file(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Symlink files are not accepted: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open regular file {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a regular, non-symlink file."""

    return hashlib.sha256(_read_regular_file(Path(path))).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    return text


def _positive_dimension(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if number <= 0 or number != value:
        raise ValueError(f"{field} must be a positive integer")
    return number


def decode_coco_segmentation(
    segmentation: Any,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode one COCO polygon/RLE annotation using COCO rasterization rules."""

    height = _positive_dimension(height, field="height")
    width = _positive_dimension(width, field="width")
    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("COCO polygon segmentation must not be empty")
            polygons: list[list[float]] = []
            for index, polygon in enumerate(segmentation):
                if not isinstance(polygon, (list, tuple)):
                    raise ValueError(f"COCO polygon {index} must be a coordinate list")
                if len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError(f"COCO polygon {index} must contain at least 3 xy points")
                points = np.asarray(polygon, dtype=np.float64)
                if not np.isfinite(points).all():
                    raise ValueError(f"COCO polygon {index} contains a non-finite coordinate")
                polygons.append(points.tolist())
            encoded = coco_mask.frPyObjects(polygons, height, width)
            rle = coco_mask.merge(encoded)
        elif isinstance(segmentation, dict):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise ValueError("COCO RLE size must be [height, width]")
            rle_height = _positive_dimension(size[0], field="RLE height")
            rle_width = _positive_dimension(size[1], field="RLE width")
            if (rle_height, rle_width) != (height, width):
                raise ValueError(
                    f"COCO RLE size {(rle_height, rle_width)} does not match {(height, width)}"
                )
            if isinstance(counts, list):
                if not counts:
                    raise ValueError("Uncompressed COCO RLE counts must not be empty")
                normalized_counts: list[int] = []
                for count in counts:
                    if isinstance(count, bool):
                        raise ValueError("COCO RLE counts must be non-negative integers")
                    try:
                        number = int(count)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("COCO RLE counts must be non-negative integers") from exc
                    if number < 0 or number != count:
                        raise ValueError("COCO RLE counts must be non-negative integers")
                    normalized_counts.append(number)
                if sum(normalized_counts) != height * width:
                    raise ValueError("Uncompressed COCO RLE counts do not cover the image")
                rle = coco_mask.frPyObjects(
                    {"size": [height, width], "counts": normalized_counts},
                    height,
                    width,
                )
            elif isinstance(counts, (str, bytes)):
                if isinstance(counts, str):
                    try:
                        counts = counts.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise ValueError("Compressed COCO RLE counts must be ASCII") from exc
                if not counts:
                    raise ValueError("Compressed COCO RLE counts must not be empty")
                rle = {"size": [height, width], "counts": counts}
            else:
                raise ValueError("COCO RLE counts must be a list, string, or bytes")
        else:
            raise ValueError("COCO segmentation must be a polygon list or RLE object")

        decoded = np.asarray(coco_mask.decode(rle))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid COCO segmentation: {exc}") from exc
    if decoded.ndim != 2 or decoded.shape != (height, width):
        raise ValueError(
            f"Decoded COCO mask shape {decoded.shape} does not match {(height, width)}"
        )
    if not np.isin(decoded, (0, 1)).all():
        raise ValueError("Decoded COCO mask is not binary")
    return np.ascontiguousarray(decoded.astype(bool))


def load_validated_coco_ground_truth(
    annotation_path: Path,
    annotation_id: int | str,
    *,
    expected_dataset: str,
    expected_image_id: int | str,
    image_path: Path,
    expected_annotation_sha256: str,
    expected_image_sha256: str,
    expected_split: str | None = None,
    expected_category_label: str | None = None,
    expected_annotation_label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load a COCO GT mask after binding dataset, annotation, image, and hashes."""

    annotation_path = Path(annotation_path)
    image_path = Path(image_path)
    actual_dataset = annotation_path.parent.parent.name
    if not expected_dataset or actual_dataset != str(expected_dataset):
        raise ValueError(
            f"Dataset identity mismatch: expected {expected_dataset!r}, got {actual_dataset!r}"
        )
    if expected_split is not None:
        expected_name = f"instances_{expected_split}.json"
        if annotation_path.name != expected_name:
            raise ValueError(
                f"Split identity mismatch: expected {expected_name!r}, got {annotation_path.name!r}"
            )

    annotation_bytes = _read_regular_file(annotation_path)
    expected_ann_hash = _require_sha256(
        expected_annotation_sha256, field="expected_annotation_sha256"
    )
    actual_ann_hash = hashlib.sha256(annotation_bytes).hexdigest()
    if actual_ann_hash != expected_ann_hash:
        raise ValueError("COCO annotation file SHA-256 mismatch")
    try:
        annotation_text = annotation_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("COCO annotation file is not UTF-8") from exc
    document = _strict_json_loads(annotation_text, source=str(annotation_path))
    if not isinstance(document, dict):
        raise ValueError("COCO annotation document must be a JSON object")

    def unique_index(records: Any, *, name: str) -> dict[str, dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError(f"COCO {name} must be a list")
        result: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Every COCO {name} record must be an object with id")
            key = str(record["id"])
            if key in result:
                raise ValueError(f"Duplicate COCO {name} id: {key}")
            result[key] = record
        return result

    annotations = unique_index(document.get("annotations"), name="annotation")
    images = unique_index(document.get("images"), name="image")
    categories = unique_index(document.get("categories"), name="category")
    annotation_key = str(annotation_id)
    if annotation_key not in annotations:
        raise ValueError(f"COCO annotation id not found: {annotation_key}")
    annotation = annotations[annotation_key]
    image_key = str(expected_image_id)
    if str(annotation.get("image_id")) != image_key:
        raise ValueError("COCO annotation is bound to a different image id")
    if image_key not in images:
        raise ValueError(f"COCO image id not found: {image_key}")
    image_record = images[image_key]
    category_key = str(annotation.get("category_id"))
    if category_key not in categories:
        raise ValueError(f"COCO category id not found: {category_key}")
    category_record = categories[category_key]
    category_label = category_record.get("name")
    if not isinstance(category_label, str) or not category_label:
        raise ValueError(f"COCO category {category_key} has no non-empty string name")
    if (
        expected_category_label is not None
        and category_label != str(expected_category_label)
    ):
        raise ValueError("COCO category label does not match the prediction record")
    annotation_label = annotation.get("original_label", category_label)
    if not isinstance(annotation_label, str) or not annotation_label:
        raise ValueError(
            f"COCO annotation {annotation_key} has no non-empty string label"
        )
    if (
        expected_annotation_label is not None
        and annotation_label != str(expected_annotation_label)
    ):
        raise ValueError("COCO annotation label does not match the prediction record")

    expected_image_hash = _require_sha256(
        expected_image_sha256, field="expected_image_sha256"
    )
    image_bytes = _read_regular_file(image_path)
    if hashlib.sha256(image_bytes).hexdigest() != expected_image_hash:
        raise ValueError("Source image SHA-256 mismatch")
    record_file_name = str(image_record.get("file_name") or "").replace("\\", "/")
    file_name = PurePosixPath(record_file_name)
    if (
        not record_file_name
        or file_name.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in file_name.parts)
    ):
        raise ValueError("COCO image file_name is not a safe relative path")
    dataset_dir = annotation_path.parent.parent.resolve(strict=True)
    if (
        expected_split is not None
        and file_name.parts[0] != str(expected_split)
    ):
        candidate = dataset_dir.joinpath(str(expected_split), *file_name.parts)
    else:
        candidate = dataset_dir.joinpath(*file_name.parts)
    if candidate.is_symlink() or not candidate.exists() or not candidate.is_file():
        raise ValueError("COCO image file_name does not reference a regular file")
    resolved_candidate = candidate.resolve(strict=True)
    try:
        resolved_image_path = image_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Cannot resolve source image: {image_path}") from exc
    if resolved_image_path != resolved_candidate:
        raise ValueError("COCO image file_name does not match the source image path")
    width = _positive_dimension(image_record.get("width"), field="COCO image width")
    height = _positive_dimension(image_record.get("height"), field="COCO image height")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            actual_size = image.size
    except Exception as exc:
        raise ValueError(f"Cannot decode source image: {exc}") from exc
    if actual_size != (width, height):
        raise ValueError(
            f"Source image size {actual_size} does not match COCO {(width, height)}"
        )
    if "segmentation" not in annotation:
        raise ValueError("COCO annotation has no segmentation")
    mask = decode_coco_segmentation(annotation["segmentation"], height, width)
    if not mask.any():
        raise ValueError(f"COCO annotation {annotation_key} decodes to an empty mask")
    return mask, annotation, image_record


def thresholded_hungarian_matches(
    iou_matrix: Any,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Match GT rows to prediction columns, maximizing count then total IoU."""

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("IoU threshold must be a finite number in [0, 1]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be a finite number in [0, 1]")
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("IoU matrix must be two-dimensional")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("IoU matrix entries must be finite numbers in [0, 1]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return []

    max_pairs = min(matrix.shape)
    cardinality_bonus = float(max_pairs + 1)
    valid = matrix >= threshold
    weights = np.where(valid, cardinality_bonus + matrix, 0.0)
    rows, columns = linear_sum_assignment(-weights)
    matches = [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns)
        if valid[row, column]
    ]
    return sorted(matches, key=lambda item: (item[0], item[1]))


def prepare_empty_output_dir(path: Path) -> Path:
    """Create or accept an empty output directory; never reuse non-empty output."""

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise ValueError(f"Refusing to reuse non-empty output directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _safe_relative_path(root: Path, relative_path: str | Path, *, must_exist: bool) -> Path:
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"Artifact root must not be a symlink: {root}")
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Artifact root is not a directory: {root}")
    raw = str(relative_path)
    posix = PurePosixPath(raw)
    if (
        not raw
        or "\x00" in raw
        or "\\" in raw
        or ":" in raw
        or posix.is_absolute()
        or posix.as_posix() != raw
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"Unsafe artifact relative path: {raw!r}")
    current = root
    for part in posix.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"Artifact path contains a symlink: {raw!r}")
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*posix.parts)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes its root: {raw!r}") from exc
    if must_exist and (candidate.is_symlink() or not candidate.is_file()):
        raise ValueError(f"Artifact is not a regular file: {raw!r}")
    return candidate


def resolve_relative_artifact(root: Path, relative_path: str | Path) -> Path:
    """Resolve an existing regular artifact contained by a non-symlink root."""

    return _safe_relative_path(
        Path(root), relative_path, must_exist=True
    ).resolve(strict=True)


def _normalize_binary_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Binary mask must be a non-empty 2D array")
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Binary mask must contain only 0 and 1")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError("Binary mask must contain only 0 and 1")
    return np.ascontiguousarray(array.astype(bool))


def binary_mask_sha256(mask: Any) -> str:
    """Hash canonical uint8 0/1 pixels of a strict binary mask."""

    normalized = _normalize_binary_mask(mask)
    return hashlib.sha256(normalized.astype(np.uint8).tobytes(order="C")).hexdigest()


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> Path:
    """Serialize strict JSON and atomically replace a non-symlink destination."""

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"JSON destination must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"JSON destination must be a regular file: {path}")
    _atomic_write(path, _json_bytes(value))
    return path


def save_binary_mask(
    root: Path,
    relative_path: str | Path,
    mask: Any,
) -> dict[str, Any]:
    """Atomically save a strict binary PNG and return its integrity record."""

    root = Path(root)
    target = _safe_relative_path(root, relative_path, must_exist=False)
    if target.suffix.lower() != ".png":
        raise ValueError("Binary mask artifact must use a .png path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _safe_relative_path(root, relative_path, must_exist=False)
    if target.exists() and target.is_symlink():
        raise ValueError("Binary mask artifact must not replace a symlink")
    normalized = _normalize_binary_mask(mask)
    buffer = io.BytesIO()
    Image.fromarray(normalized.astype(np.uint8) * 255, mode="L").save(
        buffer, format="PNG"
    )
    payload = buffer.getvalue()
    _atomic_write(target, payload)
    return {
        "path": PurePosixPath(str(relative_path)).as_posix(),
        "shape_hw": [int(normalized.shape[0]), int(normalized.shape[1])],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": binary_mask_sha256(normalized),
    }


def load_binary_mask(root: Path, record: Mapping[str, Any]) -> np.ndarray:
    """Load a binary PNG only after path, file hash, shape, and pixel hash checks."""

    if not isinstance(record, Mapping):
        raise ValueError("Binary mask record must be an object")
    required = {"path", "shape_hw", "file_sha256", "pixel_sha256"}
    if not required.issubset(record):
        raise ValueError(f"Binary mask record is missing fields: {sorted(required - set(record))}")
    path = _safe_relative_path(Path(root), str(record["path"]), must_exist=True)
    if path.suffix.lower() != ".png":
        raise ValueError("Binary mask artifact must use a .png path")
    payload = _read_regular_file(path)
    expected_file_hash = _require_sha256(record["file_sha256"], field="file_sha256")
    if hashlib.sha256(payload).hexdigest() != expected_file_hash:
        raise ValueError("Binary mask file SHA-256 mismatch")
    shape = record["shape_hw"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("Binary mask shape_hw must be [height, width]")
    expected_shape = (
        _positive_dimension(shape[0], field="mask height"),
        _positive_dimension(shape[1], field="mask width"),
    )
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if getattr(image, "n_frames", 1) != 1 or image.mode not in {"1", "L"}:
                raise ValueError("Binary mask PNG must be a single-channel image")
            decoded = np.asarray(image.copy())
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot decode binary mask PNG: {exc}") from exc
    if decoded.shape != expected_shape:
        raise ValueError(
            f"Binary mask shape {decoded.shape} does not match {expected_shape}"
        )
    if decoded.dtype == np.bool_:
        normalized = np.ascontiguousarray(decoded)
    else:
        if not np.isin(decoded, (0, 255)).all():
            raise ValueError("Binary mask PNG contains values other than 0 and 255")
        normalized = np.ascontiguousarray(decoded == 255)
    expected_pixel_hash = _require_sha256(record["pixel_sha256"], field="pixel_sha256")
    if binary_mask_sha256(normalized) != expected_pixel_hash:
        raise ValueError("Binary mask pixel SHA-256 mismatch")
    return normalized


def _parse_selected_manifest(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Selected manifest is not UTF-8: {source}") from exc
    value = _strict_json_loads(text, source=source)
    if not isinstance(value, list) or not value:
        raise ValueError("Selected manifest must be a non-empty JSON list")
    identities: set[str] = set()
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("Every selected manifest entry must be an object")
        identity = record.get("group_id", record.get("sample_id"))
        if not isinstance(identity, str) or not identity:
            raise ValueError("Selected manifest identity must be a non-empty string")
        key = identity
        if key in identities:
            raise ValueError(f"Duplicate selected manifest identity: {key}")
        identities.add(key)
    return value


def _parse_predictions_jsonl(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Predictions JSONL is not UTF-8: {source}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError("Predictions JSONL must contain at least one record")
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank line in predictions JSONL at line {line_number}")
        record = _strict_json_loads(line, source=f"{source}:{line_number}")
        if not isinstance(record, dict):
            raise ValueError(f"Prediction at line {line_number} must be an object")
        identity = record.get("run_id", record.get("group_id", record.get("sample_id")))
        if not isinstance(identity, str) or not identity:
            raise ValueError(
                f"Prediction identity at line {line_number} must be a non-empty string"
            )
        key = identity
        if key in identities:
            raise ValueError(f"Duplicate prediction identity: {key}")
        identities.add(key)
        records.append(record)
    return records


def _artifact_bytes(root: Path, relative_path: str | Path) -> tuple[Path, bytes]:
    path = _safe_relative_path(root, relative_path, must_exist=True)
    return path, _read_regular_file(path)


def _selected_identity(record: Mapping[str, Any]) -> str:
    identity = record.get("group_id", record.get("sample_id"))
    if not isinstance(identity, str) or not identity:
        raise ValueError("Selected manifest identity must be a non-empty string")
    return identity


def _prediction_identity(record: Mapping[str, Any]) -> str:
    identity = record.get("run_id", record.get("group_id", record.get("sample_id")))
    if not isinstance(identity, str) or not identity:
        raise ValueError("Prediction identity must be a non-empty string")
    return identity


def _normalize_expected_identities(
    values: Sequence[str], *, field: str
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a non-empty sequence of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        if value in seen:
            raise ValueError(f"Duplicate identity in {field}: {value}")
        normalized.append(value)
        seen.add(value)
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _identity_sha256(identities: Sequence[str]) -> str:
    return hashlib.sha256(_json_bytes(list(identities))).hexdigest()


def write_complete_run_manifest(
    run_dir: Path,
    *,
    run_kind: str,
    selected_manifest_path: str | Path,
    predictions_jsonl_path: str | Path = "predictions.jsonl",
    expected_selected_ids: Sequence[str],
    expected_prediction_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Validate exact ordered run identities and atomically publish completion."""

    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("run_dir must be an existing non-symlink directory")
    if not isinstance(run_kind, str) or not run_kind.strip():
        raise ValueError("run_kind must be a non-empty string")
    selected_expected = _normalize_expected_identities(
        expected_selected_ids, field="expected_selected_ids"
    )
    predictions_expected = _normalize_expected_identities(
        expected_prediction_ids, field="expected_prediction_ids"
    )
    selected_file, selected_bytes = _artifact_bytes(run_dir, selected_manifest_path)
    predictions_file, predictions_bytes = _artifact_bytes(
        run_dir, predictions_jsonl_path
    )
    selected = _parse_selected_manifest(selected_bytes, source=str(selected_file))
    predictions = _parse_predictions_jsonl(
        predictions_bytes, source=str(predictions_file)
    )
    selected_ids = [_selected_identity(record) for record in selected]
    prediction_ids = [_prediction_identity(record) for record in predictions]
    if selected_ids != selected_expected:
        raise ValueError(
            "Selected manifest ordered identities do not match expected_selected_ids"
        )
    if prediction_ids != predictions_expected:
        raise ValueError(
            "Prediction ordered identities do not match expected_prediction_ids"
        )
    metadata_value = dict(metadata or {})
    _json_bytes(metadata_value)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "run_kind": run_kind.strip(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selected_manifest": {
            "path": PurePosixPath(str(selected_manifest_path)).as_posix(),
            "sha256": hashlib.sha256(selected_bytes).hexdigest(),
            "entry_count": len(selected),
            "identity_sha256": _identity_sha256(selected_ids),
        },
        "predictions_jsonl": {
            "path": PurePosixPath(str(predictions_jsonl_path)).as_posix(),
            "sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "record_count": len(predictions),
            "identity_sha256": _identity_sha256(prediction_ids),
        },
        "metadata": metadata_value,
    }
    manifest_path = run_dir / RUN_MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"Run manifest already exists: {manifest_path}")
    _atomic_write(manifest_path, _json_bytes(manifest))
    return manifest_path


def read_complete_run_manifest(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Strictly verify and load a completed run plus its two authoritative indexes."""

    run_dir = Path(run_dir)
    manifest_file, manifest_bytes = _artifact_bytes(run_dir, RUN_MANIFEST_NAME)
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Run manifest is not UTF-8") from exc
    manifest = _strict_json_loads(manifest_text, source=str(manifest_file))
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")
    expected_fields = {
        "schema_version",
        "status",
        "run_kind",
        "completed_at",
        "selected_manifest",
        "predictions_jsonl",
        "metadata",
    }
    if set(manifest) != expected_fields:
        raise ValueError("Run manifest fields do not match schema version 1")
    if manifest["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported run manifest schema_version")
    if manifest["status"] != "complete":
        raise ValueError("Run manifest is not complete")
    if not isinstance(manifest["run_kind"], str) or not manifest["run_kind"].strip():
        raise ValueError("Run manifest has no run_kind")
    if not isinstance(manifest["completed_at"], str) or not manifest["completed_at"]:
        raise ValueError("Run manifest has no completed_at timestamp")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("Run manifest metadata must be an object")

    def load_index(field: str, count_field: str) -> tuple[bytes, Any]:
        descriptor = manifest[field]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "sha256",
            count_field,
            "identity_sha256",
        }:
            raise ValueError(f"Invalid {field} descriptor")
        path, payload = _artifact_bytes(run_dir, descriptor["path"])
        expected_hash = _require_sha256(
            descriptor["sha256"], field=f"{field}.sha256"
        )
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"{field} SHA-256 mismatch")
        count = descriptor[count_field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{field}.{count_field} must be a positive integer")
        _require_sha256(
            descriptor["identity_sha256"], field=f"{field}.identity_sha256"
        )
        return payload, path

    selected_bytes, selected_path = load_index("selected_manifest", "entry_count")
    predictions_bytes, predictions_path = load_index("predictions_jsonl", "record_count")
    selected = _parse_selected_manifest(selected_bytes, source=str(selected_path))
    predictions = _parse_predictions_jsonl(predictions_bytes, source=str(predictions_path))
    if len(selected) != manifest["selected_manifest"]["entry_count"]:
        raise ValueError("Selected manifest entry_count mismatch")
    if len(predictions) != manifest["predictions_jsonl"]["record_count"]:
        raise ValueError("Predictions JSONL record_count mismatch")
    selected_ids = [_selected_identity(record) for record in selected]
    prediction_ids = [_prediction_identity(record) for record in predictions]
    if (
        _identity_sha256(selected_ids)
        != manifest["selected_manifest"]["identity_sha256"]
    ):
        raise ValueError("Selected manifest identity SHA-256 mismatch")
    if (
        _identity_sha256(prediction_ids)
        != manifest["predictions_jsonl"]["identity_sha256"]
    ):
        raise ValueError("Predictions JSONL identity SHA-256 mismatch")
    return manifest, selected, predictions
