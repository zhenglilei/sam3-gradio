"""Confidence-gated alignment of annotation boundaries to an RGB layout."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

COARSE_LONG_SIDE = 512
COARSE_TOP_K = 8
CENTRAL_SUPPORT_GATE = 0.90
FULL_SUPPORT_GATE = 0.60
ROI_SHIFT_GATE = 3.0
ROTATION_SPREAD_GATE = 0.2
SCALE_SPREAD_GATE = 0.5
DISTINCT_MARGIN_GATE = 0.20


def _mask(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or array.dtype.kind not in "buif":
        raise ValueError("mask must be a non-empty 2-D numeric array")
    if not np.isfinite(array).all():
        raise ValueError("mask must be finite")
    result = array.astype(bool, copy=False)
    if not result.any():
        raise ValueError("mask must contain foreground")
    return result


def _gray(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = np.clip(image[..., :3], 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("base image must be gray, RGB, or RGBA")
    if not gray.size:
        raise ValueError("base image must not be empty")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _edge_distance(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    median = float(np.median(blurred))
    low = int(max(12, 0.55 * median))
    high = int(max(low + 20, min(245, 1.35 * median)))
    edges = cv2.Canny(blurred, low, high, L2gradient=True)
    if np.count_nonzero(edges) < max(64, edges.size // 500):
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        edges = (
            magnitude >= max(1.0, float(np.percentile(magnitude, 82)))
        ).astype(np.uint8) * 255
    distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8), cv2.DIST_L2, 3
    ).astype(np.float32)
    return edges, distance


def _boundary(binary: np.ndarray) -> np.ndarray:
    values = binary.astype(np.uint8)
    edge = values & (1 - cv2.erode(values, np.ones((3, 3), np.uint8)))
    if np.count_nonzero(edge) < 32:
        edge = cv2.morphologyEx(
            values, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
    return edge.astype(bool)


def _points(binary: np.ndarray, limit: int) -> np.ndarray:
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("mask boundary is empty")
    if len(xs) > limit:
        keep = np.linspace(0, len(xs) - 1, limit, dtype=np.int64)
        xs, ys = xs[keep], ys[keep]
    return np.column_stack((xs, ys)).astype(np.float32)


def _rois(edge: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(edge)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    yy, xx = np.indices(edge.shape)
    selectors = []
    spans = ((0.20, 0.80), (0.05, 0.45), (0.55, 0.95))
    if x2 - x1 >= y2 - y1:
        for lo, hi in spans:
            selectors.append((xx >= x1 + (x2 - x1) * lo) & (xx < x1 + (x2 - x1) * hi))
    else:
        for lo, hi in spans:
            selectors.append((yy >= y1 + (y2 - y1) * lo) & (yy < y1 + (y2 - y1) * hi))
    return [_points(edge & region if (edge & region).any() else edge, 3000) for region in selectors]


def _map(
    points: np.ndarray,
    pivot: tuple[float, float],
    center: tuple[float, float],
    scale: float,
    rotation: float,
) -> np.ndarray:
    theta = math.radians(rotation)
    cosine, sine = math.cos(theta), math.sin(theta)
    relative = points - np.asarray(pivot, np.float32)
    result = np.empty_like(relative)
    result[:, 0] = scale * (cosine * relative[:, 0] - sine * relative[:, 1]) + center[0]
    result[:, 1] = scale * (sine * relative[:, 0] + cosine * relative[:, 1]) + center[1]
    return result


def _raster(
    points: np.ndarray,
    shape: tuple[int, int],
    ratio: float,
    pivot: tuple[float, float],
    center: tuple[float, float],
    scale: float,
    rotation: float,
) -> np.ndarray:
    coords = np.rint(
        _map(points, pivot, center, scale, rotation) * ratio
    ).astype(np.int32)
    output = np.zeros(shape, np.float32)
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[1])
        & (coords[:, 1] >= 0) & (coords[:, 1] < shape[0])
    )
    coords = coords[valid]
    if len(coords):
        output[coords[:, 1], coords[:, 0]] = 1
        output = cv2.dilate(output, np.ones((3, 3), np.uint8))
    return output


def _peaks(target_fft: np.ndarray, source: np.ndarray, count: int = 3):
    source = source - float(source.mean())
    norm = float(np.linalg.norm(source))
    if norm <= 1e-6:
        return []
    source_fft = cv2.dft(source, flags=cv2.DFT_COMPLEX_OUTPUT)
    product = cv2.mulSpectrums(target_fft, source_fft, 0, conjB=True)
    correlation = cv2.idft(
        product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    ) / norm
    maxima = correlation == cv2.dilate(
        correlation, np.ones((9, 9), np.uint8)
    )
    ys, xs = np.nonzero(maxima)
    if not len(xs):
        return []
    scores = correlation[ys, xs]
    count = min(count, len(scores))
    order = np.argpartition(scores, -count)[-count:]
    order = order[np.argsort(scores[order])[::-1]]
    height, width = source.shape
    result = []
    for index in order:
        dx, dy = int(xs[index]), int(ys[index])
        if dx > width // 2:
            dx -= width
        if dy > height // 2:
            dy -= height
        result.append((dx, dy, float(scores[index])))
    return result


def _score(
    distance: np.ndarray,
    full: np.ndarray,
    central: np.ndarray,
    candidate: dict[str, float],
) -> dict[str, float]:
    height, width = distance.shape
    pivot = (candidate["pivot_x"], candidate["pivot_y"])
    center = (candidate["center_x"], candidate["center_y"])

    def measure(points: np.ndarray):
        mapped = np.rint(
            _map(points, pivot, center, candidate["scale"], candidate["rotation_deg"])
        ).astype(np.int32)
        visible = (
            (mapped[:, 0] >= 0) & (mapped[:, 0] < width)
            & (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
        )
        visible_ratio = float(visible.mean())
        if not visible.any():
            return 0.0, visible_ratio, 1.0
        values = distance[mapped[visible, 1], mapped[visible, 0]]
        return (
            float(np.mean(values <= 3.0)),
            visible_ratio,
            float(np.mean(np.minimum(values, 12.0)) / 12.0),
        )

    full_support, visible_ratio, full_error = measure(full)
    central_support, _, central_error = measure(central)
    appearance_error = 0.55 * central_error + 0.45 * full_error
    error = appearance_error + 0.25 * (1 - visible_ratio)
    return {
        "central_support": central_support,
        "full_support": full_support,
        "visible_ratio": visible_ratio,
        "appearance_error": float(appearance_error),
        "error": float(error),
        "score": float(max(0, 1 - error)),
    }


def _refine(
    distance: np.ndarray,
    full: np.ndarray,
    central: np.ndarray,
    candidate: dict[str, float],
    step: float,
    min_gain: float = 0.0,
    angle_step: float = 0.2,
    scale_step: float = 0.005,
):
    best = dict(candidate)
    best_score = _score(distance, full, central, best)
    for angle_delta in (-angle_step, 0.0, angle_step):
        for scale_delta in (-scale_step, 0.0, scale_step):
            scale = float(np.clip(candidate["scale"] + scale_delta, 0.9, 1.1))
            for dx in (-step, 0.0, step):
                for dy in (-step, 0.0, step):
                    trial = {
                        **candidate,
                        "center_x": candidate["center_x"] + dx,
                        "center_y": candidate["center_y"] + dy,
                        "scale": scale,
                        "rotation_deg": float(np.clip(
                            candidate["rotation_deg"] + angle_delta, -5, 5
                        )),
                    }
                    trial_score = _score(distance, full, central, trial)
                    if trial_score["error"] < best_score["error"] - min_gain:
                        best, best_score = trial, trial_score
    return best, best_score


def align_mask_boundary(
    base_image: Any,
    mask_value: Any,
    initial_transform: dict[str, Any],
    *,
    top_k: int = COARSE_TOP_K,
) -> dict[str, Any]:
    started = time.perf_counter()
    gray = _gray(base_image)
    binary = _mask(mask_value)
    edge_image, distance = _edge_distance(gray)
    boundary = _boundary(binary)
    full = _points(boundary, 7000)
    rois = _rois(boundary)
    central = rois[0]
    height, width = gray.shape
    ratio = COARSE_LONG_SIDE / max(height, width)
    coarse_shape = (
        max(8, round(height * ratio)),
        max(8, round(width * ratio)),
    )
    likelihood = cv2.resize(
        np.exp(-distance / 2.5).astype(np.float32),
        (coarse_shape[1], coarse_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    likelihood -= float(likelihood.mean())
    target_fft = cv2.dft(likelihood, flags=cv2.DFT_COMPLEX_OUTPUT)
    pivot = (
        float(initial_transform.get("pivot_x", binary.shape[1] / 2)),
        float(initial_transform.get("pivot_y", binary.shape[0] / 2)),
    )
    initial_center = (
        float(initial_transform.get("center_x", width / 2)),
        float(initial_transform.get("center_y", height / 2)),
    )

    raw = []
    for rotation in np.linspace(-5, 5, 11):
        for scale in np.linspace(0.9, 1.1, 9):
            source = _raster(
                central, coarse_shape, ratio, pivot, initial_center,
                float(scale), float(rotation)
            )
            for dx, dy, correlation in _peaks(target_fft, source):
                for sign in (1, -1):
                    raw.append({
                        "pivot_x": pivot[0],
                        "pivot_y": pivot[1],
                        "center_x": initial_center[0] + sign * dx / ratio,
                        "center_y": initial_center[1] + sign * dy / ratio,
                        "scale": float(scale),
                        "rotation_deg": float(rotation),
                        "coarse_score": correlation,
                    })
    raw.sort(key=lambda item: item["coarse_score"], reverse=True)
    unit = max(height, width) / COARSE_LONG_SIDE
    coarse = []
    for candidate in raw:
        duplicate = any(
            math.hypot(
                candidate["center_x"] - kept["center_x"],
                candidate["center_y"] - kept["center_y"],
            ) < 4 * unit
            and abs(candidate["rotation_deg"] - kept["rotation_deg"]) < 0.5
            and abs(candidate["scale"] - kept["scale"]) < 0.01
            for kept in coarse
        )
        if not duplicate:
            coarse.append(candidate)
        if len(coarse) >= max(1, int(top_k)):
            break

    medium_ratio = min(1.0, 1024.0 / max(height, width))
    if medium_ratio < 1.0:
        medium_gray = cv2.resize(
            gray,
            (
                max(8, round(width * medium_ratio)),
                max(8, round(height * medium_ratio)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        _, medium_distance = _edge_distance(medium_gray)
    else:
        medium_distance = distance
    for candidate in coarse:
        medium_candidate = {
            **candidate,
            "pivot_x": candidate["pivot_x"] * medium_ratio,
            "pivot_y": candidate["pivot_y"] * medium_ratio,
            "center_x": candidate["center_x"] * medium_ratio,
            "center_y": candidate["center_y"] * medium_ratio,
        }
        candidate["medium_error"] = _score(
            medium_distance,
            full * medium_ratio,
            central * medium_ratio,
            medium_candidate,
        )["error"]
    validated = sorted(
        coarse,
        key=lambda item: (item["medium_error"], -item["coarse_score"]),
    )[: min(4, len(coarse))]

    refined = []
    for candidate in validated:
        best, metrics = _refine(distance, full, central, candidate, max(1, unit))
        roi_best = [
            _refine(
                distance,
                roi,
                roi,
                best,
                max(1, unit),
                min_gain=0.002,
                angle_step=0.1,
                scale_step=0.0025,
            )[0]
            for roi in rois
        ]
        centers = np.asarray(
            [[item["center_x"], item["center_y"]] for item in roi_best],
            np.float32,
        )
        pairs = [
            float(np.linalg.norm(centers[left] - centers[right]))
            for left in range(len(centers))
            for right in range(left + 1, len(centers))
        ]
        refined.append({
            **best,
            **metrics,
            "roi_shift_spread_preview_px": max(pairs, default=0) / max(unit, 1e-6),
            "rotation_spread_deg": max(x["rotation_deg"] for x in roi_best) - min(x["rotation_deg"] for x in roi_best),
            "scale_spread_percent": 100 * (max(x["scale"] for x in roi_best) - min(x["scale"] for x in roi_best)),
        })
    refined.sort(key=lambda item: (item["error"], -item["coarse_score"]))
    best = refined[0] if refined else None
    distinct = None
    if best:
        for candidate in refined[1:]:
            separated = math.hypot(
                candidate["center_x"] - best["center_x"],
                candidate["center_y"] - best["center_y"],
            ) >= 6 * unit
            if separated or abs(candidate["rotation_deg"] - best["rotation_deg"]) >= 0.5 or abs(candidate["scale"] - best["scale"]) >= 0.01:
                distinct = candidate
                break
    margin = 0.0
    if best and distinct and distinct["appearance_error"] > 1e-9:
        margin = max(
            0.0,
            (
                distinct["appearance_error"] - best["appearance_error"]
            )
            / distinct["appearance_error"],
        )
    automatic = bool(
        best
        and best["central_support"] >= CENTRAL_SUPPORT_GATE
        and best["full_support"] >= FULL_SUPPORT_GATE
        and best["roi_shift_spread_preview_px"] <= ROI_SHIFT_GATE
        and best["rotation_spread_deg"] <= ROTATION_SPREAD_GATE + 1e-9
        and best["scale_spread_percent"] <= SCALE_SPREAD_GATE + 1e-9
        and margin >= DISTINCT_MARGIN_GATE
    )
    candidates = []
    for index, candidate in enumerate(refined[:3], 1):
        candidates.append({
            "candidate_id": f"C{index}",
            "transform": {
                key: candidate[key]
                for key in (
                    "center_x", "center_y", "pivot_x", "pivot_y",
                    "scale", "rotation_deg"
                )
            },
            "metrics": {
                key: candidate[key]
                for key in (
                    "score", "central_support", "full_support", "visible_ratio",
                    "roi_shift_spread_preview_px", "rotation_spread_deg",
                    "scale_spread_percent"
                )
            },
        })
    return {
        "candidates": candidates,
        "auto_apply": automatic,
        "ambiguous": bool(best and margin < DISTINCT_MARGIN_GATE),
        "distinct_margin": margin,
        "coarse_candidate_count": len(coarse),
        "validated_candidate_count": len(validated),
        "elapsed_seconds": time.perf_counter() - started,
        "edge_pixels": int(np.count_nonzero(edge_image)),
    }


__all__ = ["align_mask_boundary"]
