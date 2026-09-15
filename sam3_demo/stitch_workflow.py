"""Qt-free cycle-stitch helpers ported from T4 Stitcher (period stitch tab)."""

from __future__ import annotations

import base64
import io
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

STITCH_LAYOUTS = {
    "横向 1×N（从左到右）": "horizontal",
    "纵向 N×1（从上到下）": "vertical",
    "2×2 网格": "grid_2x2",
    "2×N 网格（两行）": "grid_2xn",
}

LAYOUT_KEYS = ("horizontal", "vertical", "grid_2x2", "grid_2xn")

Shift = Tuple[int, int]
Rotation = float


def normalize_rotation_deg(value: float | int) -> float:
    """Normalize a finite angle to ``(-180, 180]`` degrees."""

    try:
        angle = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(angle):
        return 0.0
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle == -180.0:
        angle = 180.0
    return 0.0 if abs(angle) < 1e-9 else angle


def normalize_rotations(
    rotations: Sequence[float] | None,
    count: int,
) -> List[Rotation]:
    values = list(rotations or [])
    if len(values) != max(0, int(count)):
        return [0.0] * max(0, int(count))
    return [normalize_rotation_deg(value) for value in values]


def normalize_layout(layout: str) -> str:
    key = str(layout or "horizontal").strip()
    return STITCH_LAYOUTS.get(key, key if key in LAYOUT_KEYS else "horizontal")


def pil_rgb(image: Image.Image | np.ndarray | None) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def detect_period(gray_img: np.ndarray):
    h, w = gray_img.shape
    proj_x = gray_img.mean(axis=0)
    fft_x = np.fft.fft(proj_x - proj_x.mean())
    idx_x = int(np.argmax(np.abs(fft_x[2:w // 2])) + 2)
    phase_x = np.angle(fft_x[idx_x])
    proj_y = gray_img.mean(axis=1)
    fft_y = np.fft.fft(proj_y - proj_y.mean())
    idx_y = int(np.argmax(np.abs(fft_y[2:h // 2])) + 2)
    phase_y = np.angle(fft_y[idx_y])
    px = w / idx_x
    py = h / idx_y
    return px, py, phase_x, phase_y


def phase_offset(phase: float, period: float) -> float:
    offset = (-phase / (2.0 * np.pi)) * period
    return offset % period
def crop_to_complete_periods_bounds(
    size: Tuple[int, int],
    px: float,
    py: float,
    x0: float,
    y0: float,
) -> Tuple[int, int, int, int]:
    """Return the exact crop rectangle used for complete periodic cells."""

    W, H = size
    m_min = math.ceil((-x0 - 0.5 * px) / px)
    m_max = math.floor((W - x0 - 0.5 * px) / px)
    n_min = math.ceil((-y0 - 0.5 * py) / py)
    n_max = math.floor((H - y0 - 0.5 * py) / py)
    left = int(round(x0 + (m_min + 0.5) * px))
    right = int(round(x0 + (m_max + 0.5) * px))
    top = int(round(y0 + (n_min + 0.5) * py))
    bottom = int(round(y0 + (n_max + 0.5) * py))
    left = max(0, min(left, W))
    right = max(left, min(right, W))
    top = max(0, min(top, H))
    bottom = max(top, min(bottom, H))
    return left, top, right, bottom




def crop_to_complete_periods(
    img: Image.Image,
    px: float,
    py: float,
    x0: float,
    y0: float,
) -> Image.Image:
    return img.crop(crop_to_complete_periods_bounds(img.size, px, py, x0, y0))



def highpass(img: np.ndarray, k: int = 21) -> np.ndarray:
    if k <= 1:
        return img
    k = k | 1
    blur = cv2.GaussianBlur(img, (k, k), 0)
    return cv2.subtract(img, blur)


# Alignment is deliberately bounded.  The old implementation evaluated every
# period alias at full resolution, which is particularly expensive for large
# periodic layouts.  These limits are algorithmic guardrails rather than
# quality thresholds: the final decision is still made with full-resolution
# NCC below.
_ALIGN_COARSE_MAX_SIDE = 192
_ALIGN_MID_MAX_SIDE = 512
_ALIGN_TOP_K = 8
_ALIGN_MAX_CANDIDATES = 16384
_ALIGN_FULL_BASE_TOP_K = 3
_ALIGN_REFINE_TOP_K = 1
_ALIGN_REFINE_RADIUS = 1
_ALIGN_MAX_DIRECT_SEEDS = 12
_ALIGN_MAX_SCORE_CALLS = (
    _ALIGN_MAX_CANDIDATES
    + _ALIGN_TOP_K
    + _ALIGN_MAX_DIRECT_SEEDS
    + _ALIGN_FULL_BASE_TOP_K
    + 1
)
_ALIGN_MAX_NCC_CALLS = (
    _ALIGN_MAX_CANDIDATES
    + 2 * (_ALIGN_TOP_K + _ALIGN_MAX_DIRECT_SEEDS)
    + 2 * _ALIGN_FULL_BASE_TOP_K
    + (2 * _ALIGN_REFINE_RADIUS + 1) ** 2
    + 2
)


def _prepare_alignment_arrays(
    image: Image.Image,
    hp_kernel: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the raw grayscale and high-pass arrays used by alignment.

    Keeping this small conversion separate lets ``auto_align_images`` prepare
    each tile once.  ``match_translation`` still calls it directly, so its
    public behavior and signature remain unchanged.
    """

    raw = np.array(image.convert("L"), dtype=np.float32)
    return raw, highpass(raw, hp_kernel)


def _resize_alignment_pair(
    first: np.ndarray,
    second: np.ndarray,
    max_side: int = _ALIGN_COARSE_MAX_SIDE,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Resize two tiles with one common geometric scale."""

    first_h, first_w = first.shape[:2]
    second_h, second_w = second.shape[:2]
    limit = max(int(max_side), 8)
    largest_side = max(first_h, first_w, second_h, second_w, 1)
    scale = min(1.0, float(limit) / float(largest_side))

    def resize(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = array.shape[:2]
        if scale >= 1.0:
            return array, 1.0, 1.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(
            array,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        # Return the actual axis scales.  A minimum-size clamp would make the
        # coarse shift coordinates inconsistent with the resized array.
        return resized, new_w / float(w), new_h / float(h)

    first_resized, sx, sy = resize(first)
    second_resized, _second_sx, _second_sy = resize(second)
    return first_resized, second_resized, sx, sy


def _ncc_overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> float:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(wa, dx + wb)
    y1 = min(ha, dy + hb)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return -1.0
    # OpenCV computes the normalized dot product in C++ without constructing
    # several centered/normalized full-size temporary arrays.  Make the crops
    # contiguous because a shifted view is commonly strided; this copy is
    # still substantially cheaper than the old sequence of large temporaries.
    pa = np.ascontiguousarray(a[y0:y1, x0:x1], dtype=np.float32)
    pb = np.ascontiguousarray(
        b[y0 - dy:y1 - dy, x0 - dx:x1 - dx], dtype=np.float32
    )
    _mean_a, std_a = cv2.meanStdDev(pa)
    _mean_b, std_b = cv2.meanStdDev(pb)
    if float(std_a[0, 0]) < 1e-6 or float(std_b[0, 0]) < 1e-6:
        return -1.0
    return float(cv2.matchTemplate(pa, pb, cv2.TM_CCOEFF_NORMED)[0, 0])


def _fold_into(val: float, lo: float, hi: float, period: float) -> float:
    if period <= 1e-6:
        return val
    for _ in range(32):
        if lo < val < hi:
            return val
        if val <= lo:
            val += period
        else:
            val -= period
    return val


def _values_around(
    center: float,
    step: float,
    lo: float,
    hi: float,
    max_values: Optional[int] = None,
) -> List[float]:
    vals = [center]
    limit = None if max_values is None else max(int(max_values), 1)
    if step > 2:
        k = 1
        while center + k * step < hi and (limit is None or len(vals) < limit):
            v = center + k * step
            if v > lo:
                vals.append(v)
            k += 1
        k = 1
        while center - k * step > lo and (limit is None or len(vals) < limit):
            v = center - k * step
            if v < hi:
                vals.append(v)
            k += 1
    return vals


def _phase_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    ha, wa = a.shape
    hb, wb = b.shape
    ph, pw = max(ha, hb), max(wa, wb)
    a_pad = np.zeros((ph, pw), dtype=np.float32)
    b_pad = np.zeros((ph, pw), dtype=np.float32)
    a_pad[:ha, :wa] = a
    b_pad[:hb, :wb] = b
    win = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
    (sx, sy), _resp = cv2.phaseCorrelate(a_pad, b_pad, win)
    return -float(sx), -float(sy)


def _probe_seeds(a: np.ndarray, b: np.ndarray, axis: str) -> List[Tuple[int, int]]:
    ha, wa = a.shape
    hb, wb = b.shape
    seeds: List[Tuple[int, int]] = []
    fracs = (0.06, 0.10, 0.16, 0.24, 0.36)
    if axis == "vertical":
        for frac in fracs:
            ph = max(16, int(ha * frac))
            if ph >= ha or ph >= hb or wa > wb:
                continue
            probe = a[-ph:, :]
            if probe.shape[0] > hb or probe.shape[1] > wb:
                continue
            res = cv2.matchTemplate(b, probe, cv2.TM_CCOEFF_NORMED)
            _mn, maxv, _mnl, maxl = cv2.minMaxLoc(res)
            if maxv < 0.15:
                continue
            x, y = maxl
            seeds.append((-x, (ha - ph) - y))
    else:
        for frac in fracs:
            pw = max(16, int(wa * frac))
            if pw >= wa or pw >= wb or ha > hb:
                continue
            probe = a[:, -pw:]
            if probe.shape[0] > hb or probe.shape[1] > wb:
                continue
            res = cv2.matchTemplate(b, probe, cv2.TM_CCOEFF_NORMED)
            _mn, maxv, _mnl, maxl = cv2.minMaxLoc(res)
            if maxv < 0.15:
                continue
            x, y = maxl
            seeds.append(((wa - pw) - x, -y))
    return seeds


def _shift_candidates(
    seeds: List[Tuple[float, float]],
    px: float,
    py: float,
    axis: str,
    wa: int,
    ha: int,
    max_candidates: int = _ALIGN_MAX_CANDIDATES,
) -> List[Tuple[int, int]]:
    limit = max(int(max_candidates), 1)
    direct = []
    direct_seen = set()
    for dx, dy in seeds:
        candidate = (int(round(dx)), int(round(dy)))
        if candidate not in direct_seen:
            direct_seen.add(candidate)
            direct.append(candidate)
            if len(direct) >= limit:
                break

    # Reserve slots for the directly observed phase/probe seeds.  Alias
    # generation therefore stops as soon as the remaining budget is full,
    # instead of materializing a potentially enormous Cartesian product.
    alias_limit = max(0, limit - len(direct))
    cands: List[Tuple[int, int]] = []
    seen = set()

    def add_alias(candidate: Tuple[int, int]) -> bool:
        if candidate in direct_seen or candidate in seen:
            return True
        if len(cands) >= alias_limit:
            return False
        seen.add(candidate)
        cands.append(candidate)
        return True

    def generate_for_seed(dx: float, dy: float) -> bool:
        if axis == "vertical":
            dy0 = _fold_into(float(dy), 0.05 * ha, 0.95 * ha, float(ha))
            dx0 = float(dx) if abs(dx) <= 0.25 * wa else 0.0
            values_y = _values_around(dy0, py, 0.05 * ha, 0.95 * ha, alias_limit + 1)
            for vy in values_y:
                for ddx in range(-2, 3):
                    if not add_alias((int(round(dx0)) + ddx, int(round(vy)))):
                        return False
        elif axis == "both":
            dx0 = _fold_into(float(dx), 0.05 * wa, 0.95 * wa, float(wa))
            dy0 = _fold_into(float(dy), 0.05 * ha, 0.95 * ha, float(ha))
            values_x = _values_around(dx0, px, 0.05 * wa, 0.95 * wa, alias_limit + 1)
            values_y = _values_around(dy0, py, 0.05 * ha, 0.95 * ha, alias_limit + 1)
            for vx in values_x:
                for vy in values_y:
                    if not add_alias((int(round(vx)), int(round(vy)))):
                        return False
        else:
            dx0 = _fold_into(float(dx), 0.05 * wa, 0.95 * wa, float(wa))
            dy0 = float(dy) if abs(dy) <= 0.25 * ha else 0.0
            values_x = _values_around(dx0, px, 0.05 * wa, 0.95 * wa, alias_limit + 1)
            for vx in values_x:
                for ddy in range(-2, 3):
                    if not add_alias((int(round(vx)), int(round(dy0)) + ddy)):
                        return False
        return True

    for dx, dy in seeds:
        if alias_limit <= len(cands):
            break
        if not generate_for_seed(dx, dy):
            break

    # Direct seeds are appended after generated aliases to preserve the old
    # tie-breaking order as much as possible, while guaranteeing their place
    # within the hard bound.
    return cands + direct


def _refine_offsets(axis: str, radius: int = _ALIGN_REFINE_RADIUS) -> List[Tuple[int, int]]:
    """Return a small deterministic full-resolution refinement neighborhood."""

    radius = max(int(radius), 0)
    if axis not in ("horizontal", "vertical", "both"):
        axis = "horizontal"
    # A local 2-D neighborhood is intentional for all modes: phase/probe
    # estimates can have a small cross-axis error even when the nominal seam
    # is horizontal or vertical.
    return [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]


def _alignment_score(
    a: Tuple[np.ndarray, np.ndarray],
    b: Tuple[np.ndarray, np.ndarray],
    dx: int,
    dy: int,
) -> float:
    """Score a shift using both high-pass and raw grayscale signals."""

    a_raw, a_hp = a
    b_raw, b_hp = b
    return max(
        _ncc_overlap(a_hp, b_hp, dx, dy),
        _ncc_overlap(a_raw, b_raw, dx, dy),
    )


def _highpass_alignment_score(
    a: Tuple[np.ndarray, np.ndarray],
    b: Tuple[np.ndarray, np.ndarray],
    dx: int,
    dy: int,
) -> float:
    """Cheap first-pass score used before raw grayscale confirmation."""

    return _ncc_overlap(a[1], b[1], dx, dy)


def _fallback_translation(axis: str, wa: int, ha: int, score: float) -> Tuple[int, int, float, bool]:
    if axis == "vertical":
        return 0, ha, score, True
    if axis == "both":
        return wa, ha, score, True
    return wa, 0, score, True


def _match_translation_prepared(
    a: Tuple[np.ndarray, np.ndarray],
    b: Tuple[np.ndarray, np.ndarray],
    axis: str,
    ncc_min: float,
) -> Tuple[int, int, float, bool]:
    """Match prepared arrays with coarse screening and full-resolution proof."""

    a_raw, a_hp = a
    b_raw, b_hp = b
    ha, wa = a_hp.shape
    # Probe matching searches large strips over the opposite tile.  Running it
    # at the already-used medium resolution preserves the period-alias search
    # while avoiding repeated full-resolution matchTemplate calls.
    a_raw_mid, b_raw_mid, msx, msy = _resize_alignment_pair(
        a_raw, b_raw, _ALIGN_MID_MAX_SIDE
    )
    a_hp_mid, b_hp_mid, _msx2, _msy2 = _resize_alignment_pair(
        a_hp, b_hp, _ALIGN_MID_MAX_SIDE
    )

    seeds: List[Tuple[float, float]] = []
    seeds.append(_phase_shift(a_hp, b_hp))
    seeds.append(_phase_shift(a_raw, b_raw))
    for dx, dy in _probe_seeds(a_hp_mid, b_hp_mid, axis):
        seeds.append((dx / msx, dy / msy))
    for dx, dy in _probe_seeds(a_raw_mid, b_raw_mid, axis):
        seeds.append((dx / msx, dy / msy))
    px, py, _phx, _phy = detect_period(a_hp)
    candidates = _shift_candidates(
        seeds,
        px,
        py,
        axis,
        wa,
        ha,
        max_candidates=_ALIGN_MAX_CANDIDATES,
    )
    if not candidates:
        return _fallback_translation(axis, wa, ha, -1.0)

    # Screen every candidate on a small bounded thumbnail first.  Keeping this
    # first pass at 192px avoids spending 512px NCC work on hundreds of period
    # aliases for a large tile.
    a_raw_small, b_raw_small, sx, sy = _resize_alignment_pair(
        a_raw, b_raw, _ALIGN_COARSE_MAX_SIDE
    )
    a_hp_small, b_hp_small, _sx2, _sy2 = _resize_alignment_pair(
        a_hp, b_hp, _ALIGN_COARSE_MAX_SIDE
    )
    coarse_a = (a_raw_small, a_hp_small)
    coarse_b = (b_raw_small, b_hp_small)
    coarse_scores = []
    coarse_score_cache = {}
    for index, (dx, dy) in enumerate(candidates):
        scaled = (int(round(dx * sx)), int(round(dy * sy)))
        score = coarse_score_cache.get(scaled)
        if score is None:
            score = _highpass_alignment_score(coarse_a, coarse_b, *scaled)
            coarse_score_cache[scaled] = score
        coarse_scores.append((score, index))
    coarse_scores.sort(key=lambda item: (-item[0], item[1]))

    # Always retain direct phase/probe observations in addition to the top-K
    # thumbnail results.  This protects against a repetitive/low-texture image
    # where downsampling ranks a true high-resolution seam below an alias.
    shortlist: List[Tuple[int, int]] = []
    shortlist_seen = set()
    for _score, index in coarse_scores[:_ALIGN_TOP_K]:
        candidate = candidates[index]
        if candidate not in shortlist_seen:
            shortlist_seen.add(candidate)
            shortlist.append(candidate)
    candidate_set = set(candidates)
    for dx, dy in seeds:
        candidate = (int(round(dx)), int(round(dy)))
        if candidate in candidate_set and candidate not in shortlist_seen:
            shortlist_seen.add(candidate)
            shortlist.append(candidate)

    # A second, 512px screen is retained only for the bounded shortlist.  This
    # is a useful guard against coarse aliasing while keeping the expensive
    # medium-resolution work proportional to top-K rather than all candidates.
    medium_a = (a_raw_mid, a_hp_mid)
    medium_b = (b_raw_mid, b_hp_mid)
    medium_scores = []
    medium_score_cache = {}
    for candidate in shortlist:
        scaled = (
            int(round(candidate[0] * msx)),
            int(round(candidate[1] * msy)),
        )
        score = medium_score_cache.get(scaled)
        if score is None:
            score = _alignment_score(medium_a, medium_b, *scaled)
            medium_score_cache[scaled] = score
        medium_scores.append(
            (
                score,
                candidate,
            )
        )
    medium_scores.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
    if not medium_scores:
        return _fallback_translation(axis, wa, ha, -1.0)

    # Keep a small, spatially separated set of medium-ranked aliases for a
    # full-resolution ambiguity check.  A low medium score is not enough for
    # rejection because detail can be lost during downsampling.
    medium_best, _medium_shift = medium_scores[0]
    period_for_axis = min(px, py) if axis == "both" else (py if axis == "vertical" else px)
    ambiguity_distance = max(8.0, 0.5 * period_for_axis)
    base_candidates: List[Tuple[int, int]] = []
    for _score, candidate in medium_scores:
        if all(
            math.hypot(candidate[0] - selected[0], candidate[1] - selected[1])
            > ambiguity_distance
            for selected in base_candidates
        ):
            base_candidates.append(candidate)
            if len(base_candidates) >= _ALIGN_FULL_BASE_TOP_K:
                break
    if not base_candidates:
        return _fallback_translation(axis, wa, ha, medium_best)

    full_base_scores = [
        (_alignment_score(a, b, candidate[0], candidate[1]), candidate)
        for candidate in base_candidates
    ]
    full_base_scores.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
    full_best, full_best_shift = full_base_scores[0]
    if full_best < ncc_min:
        return _fallback_translation(axis, wa, ha, full_best)
    full_margin = max(0.01, 0.02 * max(abs(full_best), 1.0))
    for score, candidate in full_base_scores[1:]:
        if score >= full_best - full_margin:
            return _fallback_translation(axis, wa, ha, full_best)

    # Full-resolution refinement uses only the best full-resolution alias.  The
    # high-pass channel is sufficient for this tiny local search; one final
    # raw+high-pass score below remains the authoritative validation.  The
    # center is the best full-resolution alias, not merely the thumbnail one.
    full_hp_scores = []
    full_seen = set()
    offsets = _refine_offsets(axis)
    for off_x, off_y in offsets:
        candidate = (full_best_shift[0] + off_x, full_best_shift[1] + off_y)
        if candidate in full_seen:
            continue
        full_seen.add(candidate)
        score = _ncc_overlap(a_hp, b_hp, candidate[0], candidate[1])
        full_hp_scores.append((score, candidate))
    if not full_hp_scores:
        return _fallback_translation(axis, wa, ha, -1.0)
    full_hp_scores.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
    _best_hp, best = full_hp_scores[0]
    best_ncc = (
        full_best
        if best == full_best_shift
        else _alignment_score(a, b, best[0], best[1])
    )
    if full_best > best_ncc:
        best, best_ncc = full_best_shift, full_best
    if best_ncc < ncc_min:
        return _fallback_translation(axis, wa, ha, best_ncc)

    return best[0], best[1], best_ncc, False


def match_translation(
    img_a: Image.Image,
    img_b: Image.Image,
    axis: str = "horizontal",
    hp_kernel: int = 21,
    ncc_min: float = 0.2,
) -> Tuple[int, int, float, bool]:
    return _match_translation_prepared(
        _prepare_alignment_arrays(img_a, hp_kernel),
        _prepare_alignment_arrays(img_b, hp_kernel),
        axis,
        ncc_min,
    )


def _warn(warn: Optional[List[str]], msg: str):
    if warn is not None:
        warn.append(msg)


def _local_positions(shifts: Sequence[Shift]) -> List[Shift]:
    min_x = min(s[0] for s in shifts)
    min_y = min(s[1] for s in shifts)
    return [(s[0] - min_x, s[1] - min_y) for s in shifts]


def _validate_output_extent(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
) -> None:
    """Reject accidental huge gaps before allocating a mosaic canvas."""

    if not images or len(images) != len(shifts):
        return
    local = _local_positions(shifts)
    canvas_w = max(x + image.size[0] for image, (x, _y) in zip(images, local))
    canvas_h = max(y + image.size[1] for image, (_x, y) in zip(images, local))
    total_w = sum(max(1, image.size[0]) for image in images)
    total_h = sum(max(1, image.size[1]) for image in images)
    total_pixels = sum(max(1, image.size[0] * image.size[1]) for image in images)
    if (
        canvas_w > 2 * total_w
        or canvas_h > 2 * total_h
        or canvas_w * canvas_h > 4 * total_pixels
    ):
        raise ValueError(
            f"分块间距过大，输出画布将达到 {canvas_w}×{canvas_h}px；请先缩小坐标偏移"
        )


def _rotated_tile_geometry(
    image: Image.Image,
    shift: Shift,
    rotation_deg: float,
) -> Tuple[Image.Image, float, Image.Image, Shift]:
    """Return the source RGB image and the exact rotated canvas geometry."""

    rgb = image.convert("RGB")
    angle = normalize_rotation_deg(rotation_deg)
    rgba = rgb.convert("RGBA")
    if angle:
        rgba = rgba.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
    origin = (
        int(round(float(shift[0]) + (rgb.width - rgba.width) / 2.0)),
        int(round(float(shift[1]) + (rgb.height - rgba.height) / 2.0)),
    )
    return rgb, angle, rgba, origin


def _rotated_tile(
    image: Image.Image,
    shift: Shift,
    rotation_deg: float,
) -> Tuple[Image.Image, Shift, np.ndarray]:
    """Rotate one tile about its center and return RGBA, origin, and edge weights."""

    rgb, angle, rgba, origin = _rotated_tile_geometry(image, shift, rotation_deg)
    edge = Image.fromarray(
        np.clip(_edge_weight_map(rgb.height, rgb.width) * 255.0, 0, 255).astype(
            np.uint8
        ),
        mode="L",
    )
    if angle:
        edge = edge.rotate(
            -angle,
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=0,
        )
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    weight = np.asarray(edge, dtype=np.float32) / 255.0
    return rgba, origin, weight * alpha


def _rotated_pieces(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    rotations: Sequence[float],
) -> List[Tuple[Image.Image, Shift, np.ndarray]]:
    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images) or len(rotations) != len(images):
        raise ValueError("分块几何数量与图片不一致")
    pieces = [
        _rotated_tile(image, shift, rotation)
        for image, shift, rotation in zip(images, shifts, rotations)
    ]
    min_x = min(origin[0] for _image, origin, _weight in pieces)
    min_y = min(origin[1] for _image, origin, _weight in pieces)
    local = [
        (image, (origin[0] - min_x, origin[1] - min_y), weight)
        for image, origin, weight in pieces
    ]
    canvas_w = max(origin[0] + image.width for image, origin, _weight in local)
    canvas_h = max(origin[1] + image.height for image, origin, _weight in local)
    total_w = sum(max(1, image.width) for image in images)
    total_h = sum(max(1, image.height) for image in images)
    total_pixels = sum(max(1, image.width * image.height) for image in images)
    if (
        canvas_w > 3 * total_w
        or canvas_h > 3 * total_h
        or canvas_w * canvas_h > 6 * total_pixels
    ):
        raise ValueError(
            f"分块间距过大，旋转后输出画布将达到 {canvas_w}×{canvas_h}px；请先缩小坐标偏移"
        )
    return local


def stitch_images_rotated(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    rotations: Sequence[float],
    *,
    blend: bool,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Composite rotated tiles while keeping transparent rotated corners empty."""

    pieces = _rotated_pieces(images, shifts, rotations)
    canvas_w = max(origin[0] + image.width for image, origin, _weight in pieces)
    canvas_h = max(origin[1] + image.height for image, origin, _weight in pieces)
    if not blend:
        canvas = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), bg_color)
        for image, origin, _weight in pieces:
            canvas.paste(image.convert("RGB"), origin, image.getchannel("A"))
        return canvas

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    wsum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint16)
    sources: List[Tuple[np.ndarray, np.ndarray, Shift]] = []
    for i, (image, (x0, y0), weight) in enumerate(pieces):
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
        h, w = source.shape[:2]
        x1, y1 = x0 + w, y0 + h
        roi_acc = acc[y0:y1, x0:x1]
        roi_w = wsum[y0:y1, x0:x1]
        tile = source.astype(np.float32)
        overlap = (roi_w > 1e-6) & alpha
        if i > 0 and overlap.any():
            dst = np.zeros_like(tile)
            dst[overlap] = roi_acc[overlap] / np.maximum(
                roi_w[overlap][:, None], 1e-6
            )
            gain = np.clip(
                dst[overlap].mean(axis=0)
                / np.maximum(tile[overlap].mean(axis=0), 1.0),
                0.75,
                1.35,
            )
            tile = tile.copy()
            tile[overlap] = np.clip(tile[overlap] * gain, 0, 255)
        roi_acc += tile * weight[..., None]
        roi_w += weight
        coverage[y0:y1, x0:x1] += alpha.astype(np.uint16)
        sources.append((source, alpha, (x0, y0)))

    valid = wsum > 1e-6
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    out[valid] = np.clip(acc[valid] / wsum[valid, None], 0, 255).astype(np.uint8)
    for source, alpha, (x0, y0) in sources:
        h, w = source.shape[:2]
        single = (coverage[y0:y0 + h, x0:x0 + w] == 1) & alpha
        if single.any():
            output = out[y0:y0 + h, x0:x0 + w]
            output[single] = source[single]
    return Image.fromarray(out, mode="RGB")


def _intersect_paste(img: Image.Image, origin: Shift, cell: Tuple[int, int, int, int]):
    ix, iy = origin
    iw, ih = img.size
    x0 = max(ix, cell[0])
    y0 = max(iy, cell[1])
    x1 = min(ix + iw, cell[2])
    y1 = min(iy + ih, cell[3])
    if x1 <= x0 or y1 <= y0:
        return None
    cropped = img.crop((x0 - ix, y0 - iy, x1 - ix, y1 - iy))
    return cropped, (x0, y0)


def _stitch_images_2d_legacy(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    layout: str = "horizontal",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    warn: Optional[List[str]] = None,
) -> Image.Image:
    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images):
        raise ValueError("shifts 数量与图片不一致")
    _validate_output_extent(images, shifts)
    layout = normalize_layout(layout)
    local = _local_positions(shifts)
    pieces: List[Tuple[Image.Image, Shift]] = []
    if layout == "horizontal":
        pieces.append((images[0], local[0]))
        for i in range(1, len(images)):
            xi, yi = local[i]
            wi, hi = images[i].size
            prev_x, _prev_y = local[i - 1]
            prev_w, _prev_h = images[i - 1].size
            overlap_w = max(0, prev_x + prev_w - xi)
            crop_left = min(overlap_w, wi)
            if wi - crop_left <= 0:
                _warn(warn, f"图{i + 1}水平裁切后为空，已跳过")
                continue
            cropped = images[i].crop((crop_left, 0, wi, hi))
            pieces.append((cropped, (xi + crop_left, yi)))
    elif layout == "vertical":
        pieces.append((images[0], local[0]))
        for i in range(1, len(images)):
            xi, yi = local[i]
            wi, hi = images[i].size
            _prev_x, prev_y = local[i - 1]
            _prev_w, prev_h = images[i - 1].size
            overlap_h = max(0, prev_y + prev_h - yi)
            crop_top = min(overlap_h, hi)
            if hi - crop_top <= 0:
                _warn(warn, f"图{i + 1}垂直裁切后为空，已跳过")
                continue
            cropped = images[i].crop((0, crop_top, wi, hi))
            pieces.append((cropped, (xi, yi + crop_top)))
    elif layout == "grid_2x2":
        if len(images) != 4:
            raise ValueError("2×2 网格需要每组恰好 4 张图片")
        max_x = max(local[i][0] + images[i].size[0] for i in range(4))
        max_y = max(local[i][1] + images[i].size[1] for i in range(4))
        seam_x = local[1][0]
        seam_y = local[2][1]
        cells = [
            (0, 0, seam_x, seam_y),
            (seam_x, 0, max_x, seam_y),
            (0, seam_y, seam_x, max_y),
            (seam_x, seam_y, max_x, max_y),
        ]
        names = ("左上", "右上", "左下", "右下")
        for img, origin, cell, name in zip(images, local, cells, names):
            pasted = _intersect_paste(img, origin, cell)
            if pasted is None:
                _warn(warn, f"{name}裁切后为空，已跳过")
                continue
            pieces.append(pasted)
    elif layout == "grid_2xn":
        n = len(images)
        cols = max(1, math.ceil(n / 2))
        max_x = max(local[i][0] + images[i].size[0] for i in range(n))
        max_y = max(local[i][1] + images[i].size[1] for i in range(n))
        xs = [0]
        for c in range(1, cols):
            top_i, bot_i = c, cols + c
            if top_i < n:
                xs.append(local[top_i][0])
            elif bot_i < n:
                xs.append(local[bot_i][0])
            else:
                xs.append(xs[-1])
        xs.append(max_x)
        y_mid = local[cols][1] if cols < n else max_y
        ys = [0, y_mid, max_y]
        for i in range(n):
            r, c = divmod(i, cols)
            cell = (xs[c], ys[r], xs[c + 1], ys[r + 1])
            pasted = _intersect_paste(images[i], local[i], cell)
            if pasted is None:
                _warn(warn, f"图{i + 1}裁切后为空，已跳过")
                continue
            pieces.append(pasted)
    else:
        raise ValueError(f"不支持的布局: {layout}")
    if not pieces:
        raise ValueError("拼接后没有有效图像")
    canvas_w = max(xy[0] + im.size[0] for im, xy in pieces)
    canvas_h = max(xy[1] + im.size[1] for im, xy in pieces)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    for im, xy in pieces:
        canvas.paste(im, xy)
    return canvas


def stitch_images_overlay(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images):
        raise ValueError("shifts 数量与图片不一致")
    _validate_output_extent(images, shifts)
    local = _local_positions(shifts)
    canvas_w = max(xy[0] + img.size[0] for img, xy in zip(images, local))
    canvas_h = max(xy[1] + img.size[1] for img, xy in zip(images, local))
    canvas = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), bg_color)
    for img, xy in zip(images, local):
        canvas.paste(img, xy)
    return canvas

def stitch_canvas_geometry(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    *,
    layout: str = "horizontal",
    blend: bool = True,
    rotations: Sequence[float] | None = None,
) -> dict:
    """Describe the exact integer tile placement used by mosaic export.

    ``pieces`` records source crop bounds and normalized canvas origins.  It is
    intentionally image-free so annotation masks can apply the same geometry
    without changing the image export path.
    """

    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images):
        raise ValueError("shifts 数量与图片不一致")
    normalized_rotations = normalize_rotations(rotations, len(images))
    if any(normalized_rotations):
        raw = []
        for index, (image, shift, angle) in enumerate(
            zip(images, shifts, normalized_rotations)
        ):
            _rgb, _angle, rgba, origin = _rotated_tile_geometry(image, shift, angle)
            raw.append((index, image.size, angle, rgba.size, origin))
        min_x = min(item[4][0] for item in raw)
        min_y = min(item[4][1] for item in raw)
        pieces = [
            {
                "tile_index": index,
                "source_bbox_xyxy": (0, 0, size[0], size[1]),
                "origin_xy": (origin[0] - min_x, origin[1] - min_y),
                "raw_origin_xy": origin,
                "rotation_deg": angle,
                "expanded_size": expanded_size,
            }
            for index, size, angle, expanded_size, origin in raw
        ]
        canvas_size = (
            max(piece["origin_xy"][0] + piece["expanded_size"][0] for piece in pieces),
            max(piece["origin_xy"][1] + piece["expanded_size"][1] for piece in pieces),
        )
        return {
            "pieces": pieces,
            "canvas_size": canvas_size,
            "canvas_normalization_xy": (min_x, min_y),
            "rotated": True,
        }

    _validate_output_extent(images, shifts)
    local = _local_positions(shifts)
    layout = normalize_layout(layout)
    pieces = []

    def add(index: int, bbox: Tuple[int, int, int, int], origin: Shift) -> None:
        pieces.append(
            {
                "tile_index": index,
                "source_bbox_xyxy": bbox,
                "origin_xy": origin,
                "raw_origin_xy": shifts[index],
                "rotation_deg": 0.0,
                "expanded_size": (bbox[2] - bbox[0], bbox[3] - bbox[1]),
            }
        )

    if blend:
        for index, (image, origin) in enumerate(zip(images, local)):
            add(index, (0, 0, image.width, image.height), origin)
    elif layout == "horizontal":
        add(0, (0, 0, images[0].width, images[0].height), local[0])
        for index in range(1, len(images)):
            x, y = local[index]
            prev_x, _prev_y = local[index - 1]
            crop_left = min(max(0, prev_x + images[index - 1].width - x), images[index].width)
            if images[index].width - crop_left > 0:
                add(index, (crop_left, 0, images[index].width, images[index].height), (x + crop_left, y))
    elif layout == "vertical":
        add(0, (0, 0, images[0].width, images[0].height), local[0])
        for index in range(1, len(images)):
            x, y = local[index]
            _prev_x, prev_y = local[index - 1]
            crop_top = min(max(0, prev_y + images[index - 1].height - y), images[index].height)
            if images[index].height - crop_top > 0:
                add(index, (0, crop_top, images[index].width, images[index].height), (x, y + crop_top))
    elif layout == "grid_2x2":
        if len(images) != 4:
            raise ValueError("2×2 网格需要每组恰好 4 张图片")
        max_x = max(local[i][0] + images[i].width for i in range(4))
        max_y = max(local[i][1] + images[i].height for i in range(4))
        cells = ((0, 0, local[1][0], local[2][1]), (local[1][0], 0, max_x, local[2][1]), (0, local[2][1], local[1][0], max_y), (local[1][0], local[2][1], max_x, max_y))
        for index, cell in enumerate(cells):
            x, y = local[index]
            right, bottom = x + images[index].width, y + images[index].height
            left, top = max(x, cell[0]), max(y, cell[1])
            end_x, end_y = min(right, cell[2]), min(bottom, cell[3])
            if end_x > left and end_y > top:
                add(index, (left - x, top - y, end_x - x, end_y - y), (left, top))
    elif layout == "grid_2xn":
        count = len(images)
        cols = max(1, math.ceil(count / 2))
        max_x = max(local[i][0] + images[i].width for i in range(count))
        max_y = max(local[i][1] + images[i].height for i in range(count))
        xs = [0]
        for column in range(1, cols):
            top_i, bottom_i = column, cols + column
            xs.append(local[top_i][0] if top_i < count else (local[bottom_i][0] if bottom_i < count else xs[-1]))
        xs.append(max_x)
        y_mid = local[cols][1] if cols < count else max_y
        ys = (0, y_mid, max_y)
        for index, image in enumerate(images):
            row, column = divmod(index, cols)
            cell = (xs[column], ys[row], xs[column + 1], ys[row + 1])
            x, y = local[index]
            left, top = max(x, cell[0]), max(y, cell[1])
            end_x, end_y = min(x + image.width, cell[2]), min(y + image.height, cell[3])
            if end_x > left and end_y > top:
                add(index, (left - x, top - y, end_x - x, end_y - y), (left, top))
    else:
        raise ValueError(f"不支持的布局: {layout}")
    if not pieces:
        raise ValueError("拼接后没有有效图像")
    return {
        "pieces": pieces,
        "canvas_size": (
            max(piece["origin_xy"][0] + piece["expanded_size"][0] for piece in pieces),
            max(piece["origin_xy"][1] + piece["expanded_size"][1] for piece in pieces),
        ),
        "canvas_normalization_xy": (-min(shift[0] for shift in shifts), -min(shift[1] for shift in shifts)),
        "rotated": False,
    }


def _edge_weight_map(h: int, w: int, falloff: int = 48) -> np.ndarray:
    falloff = max(int(falloff), 1)
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    wmap = np.minimum(ys[:, None], xs[None, :])
    # A zero weight makes a tile's unique outermost pixels disappear into the
    # black output canvas.  Keep a tiny positive floor so those pixels remain
    # valid while the interior/overlap feathering is unchanged.
    return np.maximum(np.clip(wmap / falloff, 0.0, 1.0), 1e-3)


def stitch_images_2d(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    layout: str = "horizontal",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    warn: Optional[List[str]] = None,
) -> Image.Image:
    """Paste the shared layout geometry without recomputing crop coordinates."""

    geometry = stitch_canvas_geometry(images, shifts, layout=layout, blend=False)
    pieces = geometry["pieces"]
    present = {piece["tile_index"] for piece in pieces}
    normalized_layout = normalize_layout(layout)
    if normalized_layout == "horizontal":
        for index in range(1, len(images)):
            if index not in present:
                _warn(warn, f"图{index + 1}水平裁切后为空，已跳过")
    elif normalized_layout == "vertical":
        for index in range(1, len(images)):
            if index not in present:
                _warn(warn, f"图{index + 1}垂直裁切后为空，已跳过")
    elif normalized_layout == "grid_2x2":
        for index, name in enumerate(("左上", "右上", "左下", "右下")):
            if index not in present:
                _warn(warn, f"{name}裁切后为空，已跳过")
    elif normalized_layout == "grid_2xn":
        for index in range(len(images)):
            if index not in present:
                _warn(warn, f"图{index + 1}裁切后为空，已跳过")
    canvas = Image.new("RGB", tuple(geometry["canvas_size"]), bg_color)
    for piece in pieces:
        left, top, right, bottom = piece["source_bbox_xyxy"]
        canvas.paste(images[piece["tile_index"]].crop((left, top, right, bottom)), piece["origin_xy"])
    return canvas


def stitch_images_blend(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    falloff: int = 48,
) -> Image.Image:
    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images):
        raise ValueError("shifts 数量与图片不一致")
    _validate_output_extent(images, shifts)
    if len(images) == 1:
        single = pil_rgb(images[0])
        if single is None:
            raise ValueError("图片无效")
        return single.copy()
    local = _local_positions(shifts)
    canvas_w = max(xy[0] + img.size[0] for img, xy in zip(images, local))
    canvas_h = max(xy[1] + img.size[1] for img, xy in zip(images, local))
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    # Float32 is sufficient for 8-bit RGB feathering and halves the peak
    # accumulation memory for large mosaics compared with float64.
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    wsum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint16)

    for i, (img, (ox, oy)) in enumerate(zip(images, local)):
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        ih, iw = arr.shape[:2]
        x0, y0 = int(ox), int(oy)
        x1, y1 = x0 + iw, y0 + ih
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        sx0, sy0 = cx0 - x0, cy0 - y0
        tile = arr[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
        wt = _edge_weight_map(ih, iw, falloff)
        wt = wt[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]

        roi_acc = acc[cy0:cy1, cx0:cx1]
        roi_w = wsum[cy0:cy1, cx0:cx1]
        overlap = roi_w > 1e-6
        if i > 0 and overlap.any():
            dst = np.zeros_like(tile)
            denom = np.maximum(roi_w[overlap], 1e-6)
            dst[overlap] = roi_acc[overlap] / denom[:, None]
            dst_m = dst[overlap].mean(axis=0)
            src_m = tile[overlap].mean(axis=0)
            gain = np.clip(dst_m / np.maximum(src_m, 1.0), 0.75, 1.35)
            # Apply exposure matching only where this tile overlaps an
            # existing tile.  Pixels covered by exactly one tile must remain
            # bit-for-bit equal to their source, including outer borders.
            adjusted = tile.copy()
            adjusted[overlap] = np.clip(tile[overlap] * gain, 0, 255)
            tile = adjusted

        acc[cy0:cy1, cx0:cx1] += tile * wt[..., None]
        wsum[cy0:cy1, cx0:cx1] += wt
        coverage[cy0:cy1, cx0:cx1] += 1

    valid = wsum > 1e-6
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    out[valid] = np.clip(acc[valid] / wsum[valid, None], 0, 255).astype(np.uint8)

    # Feathering uses float32, so restore pixels covered by exactly one tile
    # from the original uint8 source to guarantee bit-for-bit fidelity there.
    for img, (ox, oy) in zip(images, local):
        ih, iw = img.size[1], img.size[0]
        x0, y0 = int(ox), int(oy)
        x1, y1 = x0 + iw, y0 + ih
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        single = coverage[cy0:cy1, cx0:cx1] == 1
        if not single.any():
            continue
        source = np.asarray(img.convert("RGB"), dtype=np.uint8)
        source = source[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        output = out[cy0:cy1, cx0:cx1]
        output[single] = source[single]
    return Image.fromarray(out, mode="RGB")


def default_shifts_for_layout(images: Sequence[Image.Image], layout: str) -> List[Shift]:
    if not images:
        return []
    layout = normalize_layout(layout)
    if layout == "vertical":
        pos = [(0, 0)]
        y = 0
        for i in range(1, len(images)):
            y += images[i - 1].size[1]
            pos.append((0, y))
        return pos
    if layout == "grid_2x2":
        if len(images) != 4:
            raise ValueError("2×2 网格需要每组恰好 4 张图片")
        w0, h0 = images[0].size
        return [(0, 0), (w0, 0), (0, h0), (w0, h0)]
    if layout == "grid_2xn":
        n = len(images)
        cols = max(1, math.ceil(n / 2))
        col_w = []
        for c in range(cols):
            w = 0
            for r in range(2):
                i = r * cols + c
                if i < n:
                    w = max(w, images[i].size[0])
            col_w.append(max(w, 1))
        row_h = []
        for r in range(2):
            h = 0
            for c in range(cols):
                i = r * cols + c
                if i < n:
                    h = max(h, images[i].size[1])
            row_h.append(max(h, 1) if any(r * cols + c < n for c in range(cols)) else 0)
        xs = [0]
        for c in range(cols - 1):
            xs.append(xs[-1] + col_w[c])
        ys = [0, row_h[0]]
        pos = []
        for i in range(n):
            r, c = divmod(i, cols)
            pos.append((xs[c], ys[r]))
        return pos
    pos = [(0, 0)]
    x = 0
    for i in range(1, len(images)):
        x += images[i - 1].size[0]
        pos.append((x, 0))
    return pos


def auto_align_images(
    images: Sequence[Image.Image],
    layout: str = "horizontal",
) -> Tuple[List[Shift], List[str]]:
    """Match neighbors under the selected acquisition topology."""
    layout = normalize_layout(layout)
    if not images:
        return [], []
    prepared = [_prepare_alignment_arrays(image) for image in images]
    if layout in ("grid_2x2", "grid_2xn"):
        if layout == "grid_2x2" and len(images) != 4:
            raise ValueError("2×2 网格需要每组恰好 4 张图片")
        from .stitch_grid_alignment import align_four_tiles, align_two_row_tiles

        matcher = align_four_tiles if len(images) == 4 else align_two_row_tiles
        positions, logs = matcher(prepared)
        return (positions if positions is not None else default_shifts_for_layout(images, layout)), logs

    from .stitch_grid_alignment import match_grid_pair

    axis = "vertical" if layout == "vertical" else "horizontal"
    positions = [(0, 0)]
    logs = []
    for i in range(1, len(images)):
        candidates = match_grid_pair(prepared[i - 1], prepared[i], axis)
        width, height = images[i - 1].size
        extent = height if axis == "vertical" else width
        cross_extent = min(im.width if axis == "vertical" else im.height for im in images[:i + 1])
        previous_cross = positions[-1][0] if axis == "vertical" else positions[-1][1]
        valid = []
        for dx, dy, score in candidates:
            primary, cross = (dy, dx) if axis == "vertical" else (dx, dy)
            if (0.5 * extent <= primary <= extent
                    and abs(cross) <= 0.1 * cross_extent
                    and abs(previous_cross + cross) <= 0.1 * cross_extent):
                valid.append((dx, dy, score))
        if valid:
            valid.sort(key=lambda item: -item[2])
            dx, dy, score = valid[0]
            message = f"邻接 图{i}-图{i + 1} 平移=({dx},{dy}) NCC={score:.3f}"
            if len(valid) > 1 and score - valid[1][2] < 0.02:
                message += "；周期纹理存在近似等分候选，位置仍有歧义"
        else:
            dx, dy = (0, height) if axis == "vertical" else (width, 0)
            message = f"警告：图{i}-图{i + 1} 缺少符合排列方向的匹配证据，按零重叠相邻排列"
        positions.append((positions[-1][0] + dx, positions[-1][1] + dy))
        logs.append(message)
    return positions, logs


def select_worst_tile(shifts: Sequence[Shift], baseline: Sequence[Shift]) -> int:
    if not shifts:
        return 0
    best_i = 0
    best_d = -1.0
    n = min(len(shifts), len(baseline))
    for i in range(n):
        dx = float(shifts[i][0] - baseline[i][0])
        dy = float(shifts[i][1] - baseline[i][1])
        dist = dx * dx + dy * dy
        if i == 0:
            continue
        if dist > best_d:
            best_d = dist
            best_i = i
    if best_d < 0 and len(shifts) > 1:
        return 1
    return best_i


def export_mosaic(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    layout: str = "horizontal",
    blend: bool = True,
    crop_periodic: bool = False,
    rotations: Sequence[float] | None = None,
) -> Tuple[Image.Image, List[str]]:
    if not images:
        raise ValueError("该组没有图片")
    warn: List[str] = []
    normalized_rotations = normalize_rotations(rotations, len(images))
    if any(normalized_rotations):
        mosaic = stitch_images_rotated(
            images,
            shifts,
            normalized_rotations,
            blend=blend,
        )
    elif blend:
        mosaic = stitch_images_blend(images, shifts)
    else:
        mosaic = stitch_images_2d(images, shifts, layout, warn=warn)
    if crop_periodic:
        ref_gray = np.array(images[0].convert("L"), dtype=np.float32)
        px, py, ph_x, ph_y = detect_period(ref_gray)
        x0 = phase_offset(ph_x, px)
        y0 = phase_offset(ph_y, py)
        mosaic = crop_to_complete_periods(mosaic, px, py, x0, y0)
    return mosaic, warn


def image_to_data_url(
    image: Image.Image,
    *,
    max_side: int = 1600,
    quality: int = 85,
) -> str:
    img = pil_rgb(image)
    if img is None:
        return ""
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h, 1)))
    if scale < 1.0:
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def empty_canvas_payload(status: str = "请先上传分块图") -> dict:
    return {
        "tiles": [],
        "selected": 0,
        "nudge_step": 1,
        "diff_mode": False,
        "show_loupe": True,
        "drag_gain": 1.0,
        "status": status,
    }


def canvas_payload(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    *,
    rotations: Sequence[float] | None = None,
    selected: int = 0,
    nudge_step: int = 1,
    diff_mode: bool = False,
    show_loupe: bool = True,
    status: str = "",
) -> dict:
    tiles = []
    normalized_rotations = normalize_rotations(rotations, len(images))
    for i, (img, (x, y), rotation) in enumerate(
        zip(images, shifts, normalized_rotations)
    ):
        rgb = pil_rgb(img)
        if rgb is None:
            continue
        w, h = rgb.size
        tiles.append(
            {
                "index": i,
                "image": image_to_data_url(rgb),
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "rotation_deg": rotation,
            }
        )
    if not tiles:
        return empty_canvas_payload(status or "没有可预览的分块")
    selected = max(0, min(int(selected), len(tiles) - 1))
    return {
        "tiles": tiles,
        "selected": selected,
        "nudge_step": int(nudge_step) if int(nudge_step) in (1, 5, 10) else 1,
        "diff_mode": bool(diff_mode),
        "show_loupe": bool(show_loupe),
        "drag_gain": 1.0,
        "status": status,
    }


def shifts_from_canvas_payload(payload: dict | None, fallback: Sequence[Shift] | None = None) -> List[Shift]:
    fallback_shifts = list(fallback or [])
    if not isinstance(payload, dict):
        return fallback_shifts
    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return fallback_shifts

    expected_count = len(fallback_shifts)
    indexed: dict[int, Shift] = {}
    for item in tiles:
        if not isinstance(item, dict):
            return fallback_shifts
        index = item.get("index")
        x = item.get("x")
        y = item.get("y")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            return fallback_shifts
        index = int(index)
        if index < 0 or (expected_count and index >= expected_count) or index in indexed:
            return fallback_shifts
        if isinstance(x, bool) or isinstance(y, bool):
            return fallback_shifts
        if not isinstance(x, (int, float, np.integer, np.floating)) or not isinstance(
            y, (int, float, np.integer, np.floating)
        ):
            return fallback_shifts
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return fallback_shifts
        indexed[index] = (int(round(float(x))), int(round(float(y))))

    # A fallback gives us the expected image count.  Reject partial/reordered
    # payloads rather than applying a potentially misaligned subset.
    if expected_count:
        if len(indexed) != expected_count or set(indexed) != set(range(expected_count)):
            return fallback_shifts
        return [indexed[i] for i in range(expected_count)]

    # Without a fallback, only a complete zero-based index set is meaningful;
    # otherwise there is no safe way to infer which image a coordinate belongs
    # to.  This path is primarily defensive because callbacks pass a fallback.
    if not indexed or set(indexed) != set(range(max(indexed) + 1)):
        return []
    return [indexed[i] for i in range(max(indexed) + 1)]


def rotations_from_canvas_payload(
    payload: dict | None,
    fallback: Sequence[float] | None = None,
) -> List[Rotation]:
    fallback_rotations = list(fallback or [])
    if not isinstance(payload, dict):
        return fallback_rotations
    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return fallback_rotations
    expected_count = len(fallback_rotations)
    indexed: dict[int, Rotation] = {}
    for item in tiles:
        if not isinstance(item, dict):
            return fallback_rotations
        index = item.get("index")
        value = item.get("rotation_deg")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            return fallback_rotations
        index = int(index)
        if index < 0 or (expected_count and index >= expected_count) or index in indexed:
            return fallback_rotations
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return fallback_rotations
        if not math.isfinite(float(value)):
            return fallback_rotations
        indexed[index] = normalize_rotation_deg(float(value))
    if expected_count:
        if len(indexed) != expected_count or set(indexed) != set(range(expected_count)):
            return fallback_rotations
        return [indexed[i] for i in range(expected_count)]
    if not indexed or set(indexed) != set(range(max(indexed) + 1)):
        return []
    return [indexed[i] for i in range(max(indexed) + 1)]


def selected_from_canvas_payload(payload: dict | None, count: int) -> int:
    if count <= 0:
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        selected = int(payload.get("selected") or 0)
    except (TypeError, ValueError):
        selected = 0
    return max(0, min(selected, count - 1))


def load_images_from_files(files: Iterable) -> List[Image.Image]:
    images: List[Image.Image] = []
    for item in files or []:
        path = item
        if hasattr(item, "name"):
            path = item.name
        if not path:
            continue
        text = str(path)
        if not text.lower().endswith(IMG_EXTS):
            continue
        with Image.open(text) as image:
            images.append(image.convert("RGB"))
    return images
