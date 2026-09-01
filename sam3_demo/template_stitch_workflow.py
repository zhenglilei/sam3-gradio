"""Pure periodic-template stitching primitives.

This is the Qt/Gradio-free part of the legacy ``template_old.py`` workflow.
Template tiles are periodic hole-array microscope images: hole centres provide
the period and registration signal.  A 2x2 group tries SIFT/RANSAC layouts
with a closed-loop check; other grid shapes use adjacent-strip NCC followed
by hole-centre refinement.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MIN_HOLE_AREA = 300
HOLE_MIN_CIRCULARITY = 0.50
HOLE_AREA_MIN_RATIO = 0.002
HOLE_AREA_MAX_RATIO = 0.120
HOLE_MATCH_TOL = 0.25
PERIOD_FALLBACK_RATIO = 0.32
MAX_OUTPUT_PIXELS = 100_000_000
_NCC_2X2_COARSE_MAX_SIDE = 320
_NCC_2X2_LAYOUT_TOP_K = 2
_NCC_2X2_FULL_REFINE_RADIUS = 4

Shift = Tuple[int, int]


@dataclass(frozen=True)
class _Hole:
    cx: float
    cy: float
    radius: float
    area: float
    circularity: float


def _as_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    """Return an owned RGB PIL image and reject malformed arrays."""

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("输入图片必须是 H×W×3/4 或灰度数组")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("输入图片尺寸无效")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("输入图片必须是数值数组")
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB").copy()


def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(_as_rgb(image)), cv2.COLOR_RGB2BGR)


def load_template_images(files: Iterable | None) -> Tuple[List[Image.Image], List[str]]:
    """Load template tiles from paths or Gradio-like file objects.

    Input order is preserved.  Returned images are owned RGB PIL images.
    Invalid paths fail explicitly so a group cannot be silently misordered.
    """

    images: List[Image.Image] = []
    names: List[str] = []
    for item in files or []:
        if isinstance(item, Image.Image):
            images.append(_as_rgb(item))
            names.append(str(getattr(item, "filename", f"tile_{len(names):04d}.png")))
            continue
        # pathlib.Path itself has a ``name`` property; use the full path for
        # path-like inputs and reserve ``.name`` for uploaded-file objects.
        path = item if isinstance(item, (str, os.PathLike)) else getattr(item, "name", item)
        if path is None or str(path).strip() == "":
            raise ValueError("模板图片路径为空")
        path_text = os.fspath(path)
        if not str(path_text).lower().endswith(IMG_EXTS):
            raise ValueError(f"不支持的模板图片格式: {path_text}")
        try:
            with Image.open(path_text) as source:
                image = source.convert("RGB").copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取模板图片: {path_text}") from exc
        images.append(image)
        names.append(Path(path_text).name)
    return images, names


def _hole_area_bounds(shape: Tuple[int, int], min_area: int) -> Tuple[int, int]:
    h, w = shape
    area = h * w
    lo = max(int(min_area), int(area * HOLE_AREA_MIN_RATIO))
    hi = max(lo + 1, int(area * HOLE_AREA_MAX_RATIO))
    return lo, hi


def _score_holes(holes: Sequence[_Hole]) -> float:
    n = len(holes)
    if n == 0:
        return -1.0
    circles = np.array([hole.circularity for hole in holes], dtype=np.float64)
    areas = np.array([hole.area for hole in holes], dtype=np.float64)
    count_score = n * 0.5 if n < 3 else (n * 2.0 if n <= 40 else max(0.0, 80.0 - (n - 40) * 2.0))
    circularity_score = float(np.median(circles)) * 15.0
    if len(areas) >= 2 and float(np.mean(areas)) > 0:
        consistency_score = max(0.0, 4.0 - float(np.std(areas) / np.mean(areas)) * 4.0)
    else:
        consistency_score = 0.0
    return count_score + circularity_score + consistency_score


def _detect_holes_threshold(
    gray: np.ndarray,
    area_lo: int,
    area_hi: int,
    min_circularity: float,
) -> List[_Hole]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu = int(cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    thresholds = sorted({45, 55, 65, 75, 85, 95, 105, otsu, max(20, otsu - 20), max(20, otsu - 10), min(180, otsu + 10)})
    best: List[_Hole] = []
    best_score = -1.0
    h, w = gray.shape[:2]
    for threshold in thresholds:
        _, mask = cv2.threshold(blurred, max(20, min(180, int(threshold))), 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        holes: List[_Hole] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < area_lo or area > area_hi:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if cx < 2 or cy < 2 or cx > w - 3 or cy > h - 3:
                continue
            holes.append(_Hole(float(cx), float(cy), float(radius), area, float(circularity)))
        score = _score_holes(holes)
        if score > best_score or (score == best_score and len(holes) > len(best)):
            best, best_score = holes, score
    return best


def _merge_circles(
    circles: Sequence[Tuple[float, float, float]], merge_dist: float
) -> List[Tuple[float, float, float]]:
    merged: List[Tuple[float, float, float]] = []
    for x, y, radius in circles:
        for index, (mx, my, mr) in enumerate(merged):
            if (x - mx) ** 2 + (y - my) ** 2 <= merge_dist * merge_dist:
                merged[index] = ((mx + x) * 0.5, (my + y) * 0.5, (mr + radius) * 0.5)
                break
        else:
            merged.append((float(x), float(y), float(radius)))
    return merged


def _detect_holes_hough(bgr: np.ndarray, area_lo: int, area_hi: int) -> List[_Hole]:
    h, w = bgr.shape[:2]
    min_dist = max(20, int(min(h, w) * 0.11))
    min_radius = max(4, int(min(h, w) * 0.022))
    max_radius = max(min_radius + 4, int(min(h, w) * 0.11))
    sources = [
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        (255 - bgr[:, :, 1]).astype(np.uint8),
        cv2.absdiff(bgr[:, :, 1], bgr[:, :, 2]),
    ]
    raw: List[Tuple[float, float, float]] = []
    for source in sources:
        blurred = cv2.GaussianBlur(source, (5, 5), 0)
        for param2 in (14, 18, 22):
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist, param1=120, param2=param2, minRadius=min_radius, maxRadius=max_radius)
            if circles is not None:
                raw.extend((float(x), float(y), float(r)) for x, y, r in circles[0])
    if not raw:
        return []
    merged = _merge_circles(raw, min_dist * 0.38)
    median_radius = float(np.median([item[2] for item in merged]))
    holes: List[_Hole] = []
    for x, y, radius in merged:
        area = math.pi * radius * radius
        if median_radius and abs(radius - median_radius) > median_radius * 0.45:
            continue
        if not area_lo <= area <= area_hi or x < 2 or y < 2 or x > w - 3 or y > h - 3:
            continue
        holes.append(_Hole(x, y, radius, area, 0.75))
    return holes


def detect_hole_centers(
    image: Image.Image | np.ndarray,
    min_area: int = MIN_HOLE_AREA,
) -> np.ndarray:
    """Detect periodic-hole centres as deterministic ``(N, 2)`` float64."""

    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    area_lo, area_hi = _hole_area_bounds((h, w), int(min_area))
    channels = [
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        (255 - bgr[:, :, 1]).astype(np.uint8),
        cv2.absdiff(bgr[:, :, 1], bgr[:, :, 2]),
    ]
    best: List[_Hole] = []
    best_score = -1.0
    for min_circularity in (HOLE_MIN_CIRCULARITY, 0.40, 0.32):
        for source in channels:
            holes = _detect_holes_threshold(source, area_lo, area_hi, min_circularity)
            score = _score_holes(holes)
            if score > best_score or (score == best_score and len(holes) > len(best)):
                best, best_score = holes, score
    if len(best) < 6:
        hough = _detect_holes_hough(bgr, area_lo, area_hi)
        if _score_holes(hough) > best_score or len(best) < 4:
            if len(hough) >= len(best):
                best = hough
    points = np.array([[hole.cx, hole.cy] for hole in best], dtype=np.float64).reshape((-1, 2))
    if len(points):
        points = points[np.lexsort((points[:, 0], points[:, 1]))]
    return points


def estimate_hole_period(
    centers_list: Sequence[np.ndarray],
    fallback_shape: Optional[Tuple[int, int]] = None,
) -> float:
    """Estimate nearest-neighbour periodic spacing in pixels."""

    periods: List[float] = []
    for centers in centers_list:
        points = np.asarray(centers, dtype=np.float64).reshape((-1, 2))
        if len(points) < 2:
            continue
        distances, _ = cKDTree(points).query(points, k=min(3, len(points)))
        nearest = distances[:, 1] if distances.ndim == 2 else distances
        periods.extend(float(distance) for distance in np.asarray(nearest).ravel() if np.isfinite(distance) and distance > 5)
    if periods:
        period = float(np.median(periods))
        if math.isfinite(period) and period > 0:
            return period
        raise ValueError("孔间距估计无效")
    if fallback_shape is not None:
        h, w = fallback_shape
        if h <= 0 or w <= 0:
            raise ValueError("fallback_shape 无效")
        return float(min(h, w) * PERIOD_FALLBACK_RATIO)
    raise ValueError("未检测到足够孔心，无法估计孔间距")


def overlap_ncc(a: np.ndarray, b: np.ndarray, tx: int, ty: int) -> float:
    """NCC when b's top-left corner is ``(tx, ty)`` in a coordinates."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    x0, y0 = max(0, int(tx)), max(0, int(ty))
    x1, y1 = min(wa, int(tx) + wb), min(ha, int(ty) + hb)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return -1.0
    pa = np.ascontiguousarray(a[y0:y1, x0:x1])
    pb = np.ascontiguousarray(b[y0 - ty:y1 - ty, x0 - tx:x1 - tx])
    pa -= float(pa.mean())
    pb -= float(pb.mean())
    denominator = float(np.linalg.norm(pa) * np.linalg.norm(pb))
    if denominator <= 1e-6:
        return -1.0
    return float(np.sum(pa * pb) / denominator)


def _strip_ncc_shift(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    period: float,
    kind: str,
) -> Tuple[float, np.ndarray]:
    """Bounded adjacent-strip NCC search from the legacy implementation."""

    if kind not in ("horizontal", "vertical"):
        raise ValueError(f"未知配准方向: {kind}")
    h, w = gray_a.shape[:2]
    if gray_b.shape[:2] != (h, w):
        raise ValueError("配准图片尺寸必须一致")
    p = max(float(period), 1.0)
    min_pixels = max(256, int(h * w * 0.10))
    stride = 2
    best_score = -1.0
    best_shift = np.array([0.0, 0.0], dtype=np.float64)

    if kind == "horizontal":
        tx_lo = max(int(p * 0.55), int(w - p * 2.6))
        tx_hi = min(w - int(p * 0.35), w - int(p * 0.55))
        ty_lim = int(max(p * 0.45, 20))
        # Align the coarse lattice to the stride so zero cross-axis offset and
        # the nearest even primary offset are always sampled.  Starting from
        # an arbitrary odd bound can otherwise skip the true registration and
        # let a periodic alias win before the one-pixel refinement runs.
        for tx in range(tx_lo + (-tx_lo) % stride, tx_hi + 1, stride):
            overlap_width = w - tx
            if overlap_width < max(8, int(p * 0.35)):
                continue
            for ty in range(-ty_lim + ty_lim % stride, ty_lim + 1, stride):
                y0, y1 = max(0, ty), min(h, h + ty)
                pa = gray_a[y0:y1, w - overlap_width :]
                pb = gray_b[y0 - ty : y1 - ty, :overlap_width]
                if pa.size < min_pixels or pa.shape != pb.shape:
                    continue
                score = overlap_ncc(pa, pb, 0, 0)
                if score > best_score:
                    best_score, best_shift = score, np.array([float(tx), float(ty)])
        tx0, ty0 = int(best_shift[0]), int(best_shift[1])
        for tx in range(max(tx_lo, tx0 - 4), min(tx_hi, tx0 + 4) + 1):
            overlap_width = w - tx
            if overlap_width < max(8, int(p * 0.35)):
                continue
            for ty in range(max(-ty_lim, ty0 - 4), min(ty_lim, ty0 + 4) + 1):
                y0, y1 = max(0, ty), min(h, h + ty)
                pa = gray_a[y0:y1, w - overlap_width :]
                pb = gray_b[y0 - ty : y1 - ty, :overlap_width]
                if pa.size < min_pixels or pa.shape != pb.shape:
                    continue
                score = overlap_ncc(pa, pb, 0, 0)
                if score > best_score:
                    best_score, best_shift = score, np.array([float(tx), float(ty)])
    else:
        ty_lo = max(int(p * 0.55), int(h - p * 2.6))
        ty_hi = min(h - int(p * 0.35), h - int(p * 0.55))
        tx_lim = int(max(p * 0.45, 20))
        for ty in range(ty_lo + (-ty_lo) % stride, ty_hi + 1, stride):
            overlap_height = h - ty
            if overlap_height < max(8, int(p * 0.35)):
                continue
            for tx in range(-tx_lim + tx_lim % stride, tx_lim + 1, stride):
                x0, x1 = max(0, tx), min(w, w + tx)
                pa = gray_a[h - overlap_height :, x0:x1]
                pb = gray_b[:overlap_height, x0 - tx : x1 - tx]
                if pa.size < min_pixels or pa.shape != pb.shape:
                    continue
                score = overlap_ncc(pa, pb, 0, 0)
                if score > best_score:
                    best_score, best_shift = score, np.array([float(tx), float(ty)])
        ty0, tx0 = int(best_shift[1]), int(best_shift[0])
        for ty in range(max(ty_lo, ty0 - 4), min(ty_hi, ty0 + 4) + 1):
            overlap_height = h - ty
            if overlap_height < max(8, int(p * 0.35)):
                continue
            for tx in range(max(-tx_lim, tx0 - 4), min(tx_lim, tx0 + 4) + 1):
                x0, x1 = max(0, tx), min(w, w + tx)
                pa = gray_a[h - overlap_height :, x0:x1]
                pb = gray_b[:overlap_height, x0 - tx : x1 - tx]
                if pa.size < min_pixels or pa.shape != pb.shape:
                    continue
                score = overlap_ncc(pa, pb, 0, 0)
                if score > best_score:
                    best_score, best_shift = score, np.array([float(tx), float(ty)])
    return best_score, best_shift


def _sift_uint8(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError("SIFT 输入必须是灰度二维数组")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_sift_features(gray: np.ndarray, sift=None):
    detector = sift if sift is not None else cv2.SIFT_create()
    return detector.detectAndCompute(_sift_uint8(gray), None)


def _sift_shift_from_features(features_a, features_b) -> Tuple[Optional[np.ndarray], int]:
    key_a, desc_a = features_a
    key_b, desc_b = features_b
    if desc_a is None or desc_b is None or len(key_a) < 4 or len(key_b) < 4:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [pair[0] for pair in matches if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 6:
        return None, len(good)
    points_b = np.float32([key_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    points_a = np.float32([key_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(points_b, points_a, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None, len(good)
    return np.array([matrix[0, 2], matrix[1, 2]], dtype=np.float64), int(inliers.sum()) if inliers is not None else 0


def _build_sift_pair_cache(
    grays: Sequence[np.ndarray],
) -> dict[Tuple[int, int], Tuple[Optional[np.ndarray], int]]:
    """Extract each tile once and match each unordered pair once."""

    sift = cv2.SIFT_create()
    features = [_extract_sift_features(gray, sift=sift) for gray in grays]
    cache: dict[Tuple[int, int], Tuple[Optional[np.ndarray], int]] = {}
    for i in range(len(grays)):
        for j in range(i + 1, len(grays)):
            shift, inliers = _sift_shift_from_features(features[i], features[j])
            cache[(i, j)] = (shift, inliers)
            cache[(j, i)] = (
                None if shift is None else -np.asarray(shift, dtype=np.float64),
                inliers,
            )
    return cache


def sift_pair_shift(gray_a: np.ndarray, gray_b: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Estimate b's origin in a coordinates via SIFT/RANSAC."""

    sift = cv2.SIFT_create()
    return _sift_shift_from_features(
        _extract_sift_features(gray_a, sift=sift),
        _extract_sift_features(gray_b, sift=sift),
    )


def score_pair_shift(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    shift: np.ndarray,
    period: float,
    tolerance: float = HOLE_MATCH_TOL,
) -> int:
    if len(centers_a) == 0 or len(centers_b) == 0:
        return 0
    points_a = np.asarray(centers_a, dtype=np.float64)
    points_b = np.asarray(centers_b, dtype=np.float64)
    tree = cKDTree(points_a)
    limit = float(period) * float(tolerance)
    distances, _ = tree.query(
        points_b + np.asarray(shift, dtype=np.float64),
        k=1,
        distance_upper_bound=limit,
    )
    return int(np.sum(np.asarray(distances) < limit))


def refine_shift_with_holes(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    coarse: np.ndarray,
    period: float,
    tolerance: float = HOLE_MATCH_TOL,
) -> np.ndarray:
    """Refine the offset that places B's local hole centres in A coordinates."""

    if len(centers_a) == 0 or len(centers_b) == 0:
        return np.asarray(coarse, dtype=np.float64).copy()
    points_a = np.asarray(centers_a, dtype=np.float64)
    points_b = np.asarray(centers_b, dtype=np.float64)
    coarse = np.asarray(coarse, dtype=np.float64).copy()
    tree = cKDTree(points_a)
    search = max(float(period) * 0.12, 8.0)
    limit = float(period) * float(tolerance)

    def measure(candidate: np.ndarray) -> tuple[int, float]:
        distances, _ = tree.query(
            points_b + candidate,
            k=1,
            distance_upper_bound=limit,
        )
        matched = np.asarray(distances) < limit
        count = int(np.sum(matched))
        residual = float(np.mean(np.asarray(distances)[matched])) if count else float("inf")
        return count, residual

    best_shift = coarse.copy()
    best_count, best_residual = measure(best_shift)
    for dx in np.arange(-search, search + 1.0, 1.0):
        for dy in np.arange(-search, search + 1.0, 1.0):
            candidate = coarse + np.array([dx, dy])
            count, residual = measure(candidate)
            if count > best_count or (count == best_count and residual < best_residual):
                best_count, best_residual, best_shift = count, residual, candidate
    if best_count >= 2:
        distances, indices = tree.query(
            points_b + best_shift,
            k=1,
            distance_upper_bound=limit,
        )
        matched = np.asarray(distances) < limit
        if int(np.sum(matched)) >= 2:
            refined = np.mean(points_a[indices[matched]] - points_b[matched], axis=0)
            if np.linalg.norm(refined - coarse) <= max(float(period) * 0.12, 10.0):
                best_shift = refined
    return best_shift


def resolve_pair_shift(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    period: float,
    kind: str,
) -> Tuple[float, np.ndarray]:
    score, coarse = _strip_ncc_shift(gray_a, gray_b, period, kind)
    refined = refine_shift_with_holes(centers_a, centers_b, coarse, period)
    # Hole centres refine a nearby translation but are periodic by definition:
    # their raw match count cannot distinguish a true neighbour from a tile
    # shifted by one or more complete periods.  Rank layout permutations by
    # overlap appearance only, otherwise a diagonal periodic alias can win
    # merely because more repeated holes happen to be visible.
    return score, refined


def _coarse_ncc_pair_cache(
    grays: Sequence[np.ndarray],
    centers_list: Sequence[np.ndarray],
    period: float,
) -> dict[Tuple[int, int, str], Tuple[float, np.ndarray]]:
    """Rank every directed 2x2 edge on bounded low-resolution images."""

    height, width = np.asarray(grays[0]).shape[:2]
    scale = min(1.0, float(_NCC_2X2_COARSE_MAX_SIDE) / float(max(height, width)))
    if scale < 1.0:
        coarse_size = (
            max(16, int(round(width * scale))),
            max(16, int(round(height * scale))),
        )
        coarse_grays = [
            cv2.resize(np.asarray(gray), coarse_size, interpolation=cv2.INTER_AREA)
            for gray in grays
        ]
    else:
        coarse_grays = [np.asarray(gray) for gray in grays]
    coarse_period = max(float(period) * scale, 1.0)
    cache: dict[Tuple[int, int, str], Tuple[float, np.ndarray]] = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for kind in ("horizontal", "vertical"):
                score, coarse_shift = _strip_ncc_shift(
                    coarse_grays[i], coarse_grays[j], coarse_period, kind
                )
                shift = np.asarray(coarse_shift, dtype=np.float64) / scale
                cache[(i, j, kind)] = (
                    score,
                    refine_shift_with_holes(
                        centers_list[i], centers_list[j], shift, period
                    ),
                )
    return cache


def _fullres_ncc_refine(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    initial_shift: np.ndarray,
    period: float,
    kind: str,
    radius: int = _NCC_2X2_FULL_REFINE_RADIUS,
) -> Tuple[float, np.ndarray]:
    """Refine a nearby shift in one full-resolution matchTemplate call."""

    a = np.asarray(gray_a)
    b = np.asarray(gray_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("配准图片尺寸必须一致")
    if kind not in ("horizontal", "vertical"):
        raise ValueError(f"未知配准方向: {kind}")
    height, width = a.shape
    p = max(float(period), 1.0)
    if kind == "horizontal":
        tx_min = max(int(p * 0.55), int(width - p * 2.6))
        tx_max = min(width - int(p * 0.35), width - int(p * 0.55))
        ty_limit = int(max(p * 0.45, 20))
        ty_min, ty_max = -ty_limit, ty_limit
    else:
        ty_min = max(int(p * 0.55), int(height - p * 2.6))
        ty_max = min(height - int(p * 0.35), height - int(p * 0.55))
        tx_limit = int(max(p * 0.45, 20))
        tx_min, tx_max = -tx_limit, tx_limit

    center_x, center_y = (int(round(value)) for value in np.asarray(initial_shift).reshape(2))
    radius = max(int(radius), 0)
    local_tx_min = max(tx_min, center_x - radius)
    local_tx_max = min(tx_max, center_x + radius)
    local_ty_min = max(ty_min, center_y - radius)
    local_ty_max = min(ty_max, center_y + radius)
    if local_tx_min > local_tx_max or local_ty_min > local_ty_max:
        return -1.0, np.array([float(center_x), float(center_y)])

    # Use the overlap common to every local candidate. Moving B by one pixel
    # maps to moving the fixed template by one pixel inside this search window,
    # so OpenCV evaluates the whole local grid in one optimized call.
    ax0, ay0 = max(0, local_tx_max), max(0, local_ty_max)
    ax1 = min(width, local_tx_min + width)
    ay1 = min(height, local_ty_min + height)
    bx0, by0 = ax0 - local_tx_max, ay0 - local_ty_max
    bx1, by1 = ax1 - local_tx_min, ay1 - local_ty_min
    template = np.ascontiguousarray(a[ay0:ay1, ax0:ax1])
    search = np.ascontiguousarray(b[by0:by1, bx0:bx1])
    if template.shape[0] < 16 or template.shape[1] < 16:
        return -1.0, np.array([float(center_x), float(center_y)])
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _minimum, maximum, _minimum_at, maximum_at = cv2.minMaxLoc(result)
    refined = np.array(
        [float(local_tx_max - maximum_at[0]), float(local_ty_max - maximum_at[1])],
        dtype=np.float64,
    )
    return float(maximum), refined


def _ncc_layout_candidate(
    pair_cache: Mapping[Tuple[int, int, str], Tuple[float, np.ndarray]],
    perm: Tuple[int, int, int, int],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tl, tr, bl, br = perm
    score_h_top, shift_h_top = pair_cache[(tl, tr, "horizontal")]
    score_v_left, shift_v_left = pair_cache[(tl, bl, "vertical")]
    score_h_bottom, shift_h_bottom = pair_cache[(bl, br, "horizontal")]
    score_v_right, shift_v_right = pair_cache[(tr, br, "vertical")]
    pos_br_top = shift_h_top + shift_v_right
    pos_br_left = shift_v_left + shift_h_bottom
    closure_error = float(np.linalg.norm(pos_br_top - pos_br_left))
    total = score_h_top + score_v_left + score_h_bottom + score_v_right - closure_error * 5.0
    return total, shift_h_top, shift_v_left, pos_br_top, pos_br_left


def solve_grid_2x2(
    grays: Sequence[np.ndarray],
    centers_list: Sequence[np.ndarray],
    period: float,
) -> Tuple[Mapping[int, np.ndarray], Tuple[int, int, int, int], str]:
    """Enumerate 2x2 layouts with SIFT closed-loop and NCC fallback."""

    if len(grays) != 4 or len(centers_list) != 4:
        raise ValueError("当前自动模板布局仅支持 4 张图 (2x2)")
    shapes = {tuple(np.asarray(gray).shape[:2]) for gray in grays}
    if len(shapes) != 1:
        raise ValueError("2x2 配准图片尺寸必须一致")
    height, width = next(iter(shapes))
    p = max(float(period), 1.0)

    def plausible_edge(shift: np.ndarray, kind: str) -> bool:
        dx, dy = (float(value) for value in np.asarray(shift).reshape(2))
        if kind == "horizontal":
            primary_low = max(p * 0.55, width - p * 2.6)
            primary_high = min(width - p * 0.35, width - p * 0.55)
            return primary_low <= dx <= primary_high and abs(dy) <= max(p * 0.45, 20.0)
        primary_low = max(p * 0.55, height - p * 2.6)
        primary_high = min(height - p * 0.35, height - p * 0.55)
        return primary_low <= dy <= primary_high and abs(dx) <= max(p * 0.45, 20.0)

    sift_cache = _build_sift_pair_cache(grays)
    best = None
    for perm in permutations(range(4)):
        tl, tr, bl, br = perm
        edge_data = [sift_cache[(i, j)] for i, j in ((tl, tr), (tl, bl), (bl, br), (tr, br))]
        if any(shift is None for shift, _ in edge_data):
            continue
        shifts = [shift for shift, _ in edge_data]
        inliers = sum(count for _, count in edge_data)
        pos_tr, pos_bl, shift_h_bottom, shift_v_right = shifts  # type: ignore[misc]
        if not (
            plausible_edge(pos_tr, "horizontal")
            and plausible_edge(pos_bl, "vertical")
            and plausible_edge(shift_h_bottom, "horizontal")
            and plausible_edge(shift_v_right, "vertical")
        ):
            continue
        pos_br_top, pos_br_left = pos_tr + shift_v_right, pos_bl + shift_h_bottom
        closure_error = float(np.linalg.norm(pos_br_top - pos_br_left))
        if closure_error > max(p * 0.35, 8.0):
            continue
        total = float(inliers) - closure_error * 3.0
        candidate = (total, perm, pos_tr, pos_bl, pos_br_top, pos_br_left, closure_error, inliers)
        if best is None or total > best[0]:
            best = candidate
    if best is not None and int(best[7]) >= 20:
        _, perm, pos_tr, pos_bl, pos_br_top, pos_br_left, _closure, _inliers = best
        tl, tr, bl, br = perm
        return ({tl: np.array([0.0, 0.0]), tr: pos_tr, bl: pos_bl, br: (pos_br_top + pos_br_left) * 0.5}, tuple(int(i) for i in perm), "SIFT")

    coarse_cache = _coarse_ncc_pair_cache(grays, centers_list, period)
    coarse_layouts = []
    for perm in permutations(range(4)):
        candidate = _ncc_layout_candidate(coarse_cache, perm)
        coarse_layouts.append((candidate[0], tuple(int(index) for index in perm)))
    coarse_layouts.sort(key=lambda item: (-item[0], item[1]))

    full_cache: dict[Tuple[int, int, str], Tuple[float, np.ndarray]] = {}
    best_ncc = None
    for _coarse_total, perm in coarse_layouts[:_NCC_2X2_LAYOUT_TOP_K]:
        tl, tr, bl, br = perm
        for i, j, kind in (
            (tl, tr, "horizontal"),
            (tl, bl, "vertical"),
            (bl, br, "horizontal"),
            (tr, br, "vertical"),
        ):
            key = (i, j, kind)
            if key in full_cache:
                continue
            _coarse_score, coarse_shift = coarse_cache[key]
            score, shift = _fullres_ncc_refine(
                grays[i], grays[j], coarse_shift, period, kind
            )
            full_cache[key] = (
                score,
                refine_shift_with_holes(
                    centers_list[i], centers_list[j], shift, period
                ),
            )
        total, pos_tr, pos_bl, pos_br_top, pos_br_left = _ncc_layout_candidate(
            full_cache, perm
        )
        candidate = (total, perm, pos_tr, pos_bl, pos_br_top, pos_br_left)
        if best_ncc is None or total > best_ncc[0]:
            best_ncc = candidate
    if best_ncc is None:
        raise ValueError("2x2 模板布局求解失败")
    _, perm, pos_tr, pos_bl, pos_br_top, pos_br_left = best_ncc
    tl, tr, bl, br = perm
    return ({tl: np.array([0.0, 0.0]), tr: pos_tr, bl: pos_bl, br: (pos_br_top + pos_br_left) * 0.5}, tuple(int(i) for i in perm), "NCC")


def solve_grid_by_filename(
    grays: Sequence[np.ndarray],
    centers_list: Sequence[np.ndarray],
    period: float,
    rows: int,
    cols: int,
) -> Mapping[int, np.ndarray]:
    """Solve a row-major grid using adjacent strip registration."""

    count = rows * cols
    if count <= 0 or len(grays) != count or len(centers_list) != count:
        raise ValueError("图片数量与模板网格不匹配")
    positions: dict[int, np.ndarray] = {0: np.array([0.0, 0.0])}
    for index in range(1, count):
        row, col = divmod(index, cols)
        previous = index - 1 if col > 0 else index - cols
        kind = "horizontal" if col > 0 else "vertical"
        _, shift = resolve_pair_shift(grays[previous], grays[index], centers_list[previous], centers_list[index], period, kind)
        positions[index] = positions[previous] + shift
    return positions


def _normalise_positions(positions: Mapping[int, np.ndarray | Sequence[float]]) -> dict[int, np.ndarray]:
    if not positions:
        raise ValueError("positions 为空")
    arrays = {int(index): np.asarray(value, dtype=np.float64).reshape(2) for index, value in positions.items()}
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("positions 含非有限坐标")
    min_xy = np.min(np.stack(list(arrays.values())), axis=0)
    return {index: value - min_xy for index, value in arrays.items()}


def _validate_output_extent(images: Sequence[Image.Image], positions: Mapping[int, np.ndarray]) -> Tuple[int, int, dict[int, np.ndarray]]:
    normalised = _normalise_positions(positions)
    if set(normalised) != set(range(len(images))):
        raise ValueError("positions 必须覆盖每一张图片的原始 index")
    extents = [(float(normalised[index][0]) + image.width, float(normalised[index][1]) + image.height) for index, image in enumerate(images)]
    width = int(math.ceil(max(item[0] for item in extents)))
    height = int(math.ceil(max(item[1] for item in extents)))
    if width <= 0 or height <= 0 or width * height > MAX_OUTPUT_PIXELS:
        raise ValueError(f"输出画布过大: {width}×{height}px")
    return width, height, normalised


def _edge_weights(height: int, width: int) -> np.ndarray:
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    return np.maximum(np.minimum(y[:, None], x[None, :]) / 48.0, 1e-3)


def stitch_images(images: Sequence[Image.Image], positions: Mapping[int, np.ndarray | Sequence[float]]) -> Image.Image:
    """Feather tiles into a deterministic RGB PIL mosaic."""

    if not images:
        raise ValueError("images 为空")
    rgb_images = [_as_rgb(image) for image in images]
    first_size = rgb_images[0].size
    if any(image.size != first_size for image in rgb_images[1:]):
        raise ValueError("模板分块尺寸必须一致")
    canvas_w, canvas_h, normalised = _validate_output_extent(rgb_images, positions)
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for index, image in enumerate(rgb_images):
        x0, y0 = (int(round(value)) for value in normalised[index])
        x1, y1 = x0 + image.width, y0 + image.height
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(canvas_w, x1), min(canvas_h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        sx0, sy0 = cx0 - x0, cy0 - y0
        sx1, sy1 = sx0 + cx1 - cx0, sy0 + cy1 - cy0
        tile = np.asarray(image, dtype=np.float32)[sy0:sy1, sx0:sx1]
        weight = _edge_weights(image.height, image.width)[sy0:sy1, sx0:sx1]
        accum[cy0:cy1, cx0:cx1] += tile * weight[..., None]
        weights[cy0:cy1, cx0:cx1] += weight
    valid = weights > 1e-6
    output = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    output[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def render_template_preview(
    images: Sequence[Image.Image],
    positions: Mapping[int, np.ndarray | Sequence[float]],
    centers_list: Optional[Sequence[np.ndarray]] = None,
    order: Optional[Sequence[int]] = None,
) -> Image.Image:
    """Render a PIL preview with tile boundaries, indexes and hole centres."""

    if not images:
        raise ValueError("images 为空")
    rgb_images = [_as_rgb(image) for image in images]
    _canvas_w, _canvas_h, normalised = _validate_output_extent(rgb_images, positions)
    preview = stitch_images(rgb_images, positions)
    draw = ImageDraw.Draw(preview)
    palette = ((255, 80, 80), (80, 220, 120), (80, 160, 255), (240, 190, 60))
    for display_index, source_index in enumerate(order or range(len(rgb_images))):
        source_index = int(source_index)
        x0, y0 = (int(round(value)) for value in normalised[source_index])
        image = rgb_images[source_index]
        color = palette[display_index % len(palette)]
        draw.rectangle((x0, y0, x0 + image.width - 1, y0 + image.height - 1), outline=color, width=2)
        draw.text((x0 + 4, y0 + 4), str(display_index + 1), fill=color)
        if centers_list is not None and source_index < len(centers_list):
            for center in np.asarray(centers_list[source_index]).reshape((-1, 2)):
                cx, cy = int(round(x0 + float(center[0]))), int(round(y0 + float(center[1])))
                draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=(255, 255, 0), width=1)
    return preview


def stitch_template_group(
    images: Sequence[Image.Image],
    names: Sequence[str],
    rows: int,
    cols: int,
) -> Tuple[Image.Image, dict]:
    """Stitch one periodic template group and return ``(PIL, meta)``."""

    if rows <= 0 or cols <= 0:
        raise ValueError("模板网格行列必须为正数")
    expected = int(rows) * int(cols)
    if len(images) != expected:
        raise ValueError(f"组内图片数量 {len(images)} 与网格 {rows}x{cols} 不匹配")
    if len(names) != expected:
        raise ValueError("names 数量与图片不一致")
    rgb_images = [_as_rgb(image) for image in images]
    width, height = rgb_images[0].size
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效")
    if any(image.size != (width, height) for image in rgb_images):
        raise ValueError("模板分块尺寸必须一致")
    grays = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY) for image in rgb_images]
    centers_list = [detect_hole_centers(image) for image in rgb_images]
    hole_counts = [int(len(centers)) for centers in centers_list]
    # A missing hole set would make both period and seam selection ambiguous;
    # fail before allocating a mosaic rather than silently guessing.
    if any(count < 2 for count in hole_counts):
        raise ValueError(f"孔心数量不足，无法安全拼接: {hole_counts}")
    period = estimate_hole_period(centers_list)
    if rows == 2 and cols == 2:
        positions, order_tuple, method = solve_grid_2x2(grays, centers_list, period)
        order = list(order_tuple)
    else:
        positions = solve_grid_by_filename(grays, centers_list, period, rows, cols)
        order = list(range(expected))
        method = "chain"
    # The UI and metadata use integer pixel positions. Quantise once before
    # both rendering paths so the returned preview geometry is reproducible.
    positions = {index: np.rint(value).astype(np.int64) for index, value in positions.items()}
    ordered_images = [rgb_images[index] for index in order]
    ordered_positions = {new_index: positions[old_index] for new_index, old_index in enumerate(order)}
    stitched = stitch_images(ordered_images, ordered_positions)
    serial_positions = [[int(positions[index][0]), int(positions[index][1])] for index in range(expected)]
    meta = {
        "period": float(period),
        "positions": serial_positions,
        "order": [int(index) for index in order],
        "layout_method": method,
        "hole_counts": hole_counts,
        "names": [str(name) for name in names],
        "centers": [
            [[float(point[0]), float(point[1])] for point in np.asarray(centers).reshape((-1, 2))]
            for centers in centers_list
        ],
    }
    return stitched, meta


template_stitch = stitch_template_group


__all__ = [
    "IMG_EXTS",
    "load_template_images",
    "detect_hole_centers",
    "estimate_hole_period",
    "overlap_ncc",
    "sift_pair_shift",
    "score_pair_shift",
    "refine_shift_with_holes",
    "resolve_pair_shift",
    "solve_grid_2x2",
    "solve_grid_by_filename",
    "stitch_images",
    "render_template_preview",
    "stitch_template_group",
    "template_stitch",
]
