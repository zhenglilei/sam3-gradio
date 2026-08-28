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


def crop_to_complete_periods(
    img: Image.Image,
    px: float,
    py: float,
    x0: float,
    y0: float,
) -> Image.Image:
    W, H = img.size
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
    return img.crop((left, top, right, bottom))


def highpass(img: np.ndarray, k: int = 21) -> np.ndarray:
    if k <= 1:
        return img
    k = k | 1
    blur = cv2.GaussianBlur(img, (k, k), 0)
    return cv2.subtract(img, blur)


def _ncc_overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> float:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(wa, dx + wb)
    y1 = min(ha, dy + hb)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return -1.0
    pa = a[y0:y1, x0:x1].astype(np.float32).ravel()
    pb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx].astype(np.float32).ravel()
    sa, sb = float(pa.std()), float(pb.std())
    if sa < 1e-6 or sb < 1e-6:
        return -1.0
    return float(np.mean(((pa - pa.mean()) / sa) * ((pb - pb.mean()) / sb)))


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


def _values_around(center: float, step: float, lo: float, hi: float) -> List[float]:
    vals = [center]
    if step > 2:
        k = 1
        while center + k * step < hi:
            v = center + k * step
            if v > lo:
                vals.append(v)
            k += 1
        k = 1
        while center - k * step > lo:
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
) -> List[Tuple[int, int]]:
    cands: List[Tuple[int, int]] = []
    for dx, dy in seeds:
        if axis == "vertical":
            dy0 = _fold_into(float(dy), 0.05 * ha, 0.95 * ha, float(ha))
            dx0 = float(dx) if abs(dx) <= 0.25 * wa else 0.0
            for vy in _values_around(dy0, py, 0.05 * ha, 0.95 * ha):
                for ddx in range(-2, 3):
                    cands.append((int(round(dx0)) + ddx, int(round(vy))))
        elif axis == "both":
            dx0 = _fold_into(float(dx), 0.05 * wa, 0.95 * wa, float(wa))
            dy0 = _fold_into(float(dy), 0.05 * ha, 0.95 * ha, float(ha))
            for vx in _values_around(dx0, px, 0.05 * wa, 0.95 * wa):
                for vy in _values_around(dy0, py, 0.05 * ha, 0.95 * ha):
                    cands.append((int(round(vx)), int(round(vy))))
        else:
            dx0 = _fold_into(float(dx), 0.05 * wa, 0.95 * wa, float(wa))
            dy0 = float(dy) if abs(dy) <= 0.25 * ha else 0.0
            for vx in _values_around(dx0, px, 0.05 * wa, 0.95 * wa):
                for ddy in range(-2, 3):
                    cands.append((int(round(vx)), int(round(dy0)) + ddy))
        cands.append((int(round(dx)), int(round(dy))))
    uniq = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def match_translation(
    img_a: Image.Image,
    img_b: Image.Image,
    axis: str = "horizontal",
    hp_kernel: int = 21,
    ncc_min: float = 0.2,
) -> Tuple[int, int, float, bool]:
    a_raw = np.array(img_a.convert("L"), dtype=np.float32)
    b_raw = np.array(img_b.convert("L"), dtype=np.float32)
    a_hp = highpass(a_raw, hp_kernel)
    b_hp = highpass(b_raw, hp_kernel)
    ha, wa = a_hp.shape
    seeds: List[Tuple[float, float]] = []
    seeds.append(_phase_shift(a_hp, b_hp))
    seeds.append(_phase_shift(a_raw, b_raw))
    seeds.extend(_probe_seeds(a_hp, b_hp, axis))
    seeds.extend(_probe_seeds(a_raw, b_raw, axis))
    px, py, _phx, _phy = detect_period(a_hp)
    best = None
    best_ncc = -1.0
    for cdx, cdy in _shift_candidates(seeds, px, py, axis, wa, ha):
        ncc = max(_ncc_overlap(a_hp, b_hp, cdx, cdy), _ncc_overlap(a_raw, b_raw, cdx, cdy))
        if ncc > best_ncc:
            best_ncc = ncc
            best = (cdx, cdy)
    if best is None or best_ncc < ncc_min:
        if axis == "vertical":
            return 0, ha, best_ncc, True
        if axis == "both":
            return wa, ha, best_ncc, True
        return wa, 0, best_ncc, True
    return best[0], best[1], best_ncc, False


def _warn(warn: Optional[List[str]], msg: str):
    if warn is not None:
        warn.append(msg)


def _local_positions(shifts: Sequence[Shift]) -> List[Shift]:
    min_x = min(s[0] for s in shifts)
    min_y = min(s[1] for s in shifts)
    return [(s[0] - min_x, s[1] - min_y) for s in shifts]


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


def stitch_images_2d(
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
    local = _local_positions(shifts)
    canvas_w = max(xy[0] + img.size[0] for img, xy in zip(images, local))
    canvas_h = max(xy[1] + img.size[1] for img, xy in zip(images, local))
    canvas = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), bg_color)
    for img, xy in zip(images, local):
        canvas.paste(img, xy)
    return canvas


def _edge_weight_map(h: int, w: int, falloff: int = 48) -> np.ndarray:
    falloff = max(int(falloff), 1)
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    wmap = np.minimum(ys[:, None], xs[None, :])
    return np.clip(wmap / falloff, 0.0, 1.0)


def stitch_images_blend(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    falloff: int = 48,
) -> Image.Image:
    if not images:
        raise ValueError("images 为空")
    if len(shifts) != len(images):
        raise ValueError("shifts 数量与图片不一致")
    local = _local_positions(shifts)
    canvas_w = max(xy[0] + img.size[0] for img, xy in zip(images, local))
    canvas_h = max(xy[1] + img.size[1] for img, xy in zip(images, local))
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    wsum = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for i, (img, (ox, oy)) in enumerate(zip(images, local)):
        arr = np.array(img.convert("RGB"), dtype=np.float64)
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
            tile = np.clip(tile * gain, 0, 255)

        acc[cy0:cy1, cx0:cx1] += tile * wt[..., None]
        wsum[cy0:cy1, cx0:cx1] += wt

    valid = wsum > 1e-6
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    out[valid] = np.clip(acc[valid] / wsum[valid, None], 0, 255).astype(np.uint8)
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
    """Neighbor match_translation with T4 closed-loop averaging for 2x2 / 2xn."""
    layout = normalize_layout(layout)
    logs: List[str] = []
    if not images:
        return [], logs

    def pair(i: int, j: int, axis: str) -> Shift:
        dx, dy, ncc, failed = match_translation(images[i], images[j], axis=axis)
        wa, ha = images[i].size
        overlap = max(0, ha - dy) if axis == "vertical" else max(0, wa - dx)
        msg = f"邻接 图{i + 1}-图{j + 1} 平移=({dx},{dy}) 重叠≈{overlap}px NCC={ncc:.3f}"
        if failed:
            msg = "警告：" + msg + "，已按零重叠紧挨兜底"
        logs.append(msg)
        return dx, dy

    if layout == "vertical":
        pos = [(0, 0)]
        for i in range(1, len(images)):
            dx, dy = pair(i - 1, i, "vertical")
            pos.append((pos[-1][0] + dx, pos[-1][1] + dy))
    elif layout == "grid_2x2":
        if len(images) != 4:
            raise ValueError("2×2 网格需要每组恰好 4 张图片")
        p1 = pair(0, 1, "horizontal")
        p2 = pair(0, 2, "vertical")
        p13 = pair(1, 3, "vertical")
        p23 = pair(2, 3, "horizontal")
        pos3 = (
            int(round((p1[0] + p13[0] + p2[0] + p23[0]) / 2.0)),
            int(round((p1[1] + p13[1] + p2[1] + p23[1]) / 2.0)),
        )
        pos = [(0, 0), p1, p2, pos3]
    elif layout == "grid_2xn":
        n = len(images)
        cols = max(1, math.ceil(n / 2))
        pos = [(0, 0)] * n
        for c in range(1, min(cols, n)):
            dx, dy = pair(c - 1, c, "horizontal")
            pos[c] = (pos[c - 1][0] + dx, pos[c - 1][1] + dy)
        if cols < n:
            dx, dy = pair(0, cols, "vertical")
            pos[cols] = (pos[0][0] + dx, pos[0][1] + dy)
        for c in range(1, cols):
            i = cols + c
            if i >= n:
                break
            dx_l, dy_l = pair(i - 1, i, "horizontal")
            p_left = (pos[i - 1][0] + dx_l, pos[i - 1][1] + dy_l)
            dx_t, dy_t = pair(c, i, "vertical")
            p_top = (pos[c][0] + dx_t, pos[c][1] + dy_t)
            pos[i] = (
                int(round((p_left[0] + p_top[0]) / 2.0)),
                int(round((p_left[1] + p_top[1]) / 2.0)),
            )
    else:
        pos = [(0, 0)]
        for i in range(1, len(images)):
            dx, dy = pair(i - 1, i, "horizontal")
            pos.append((pos[-1][0] + dx, pos[-1][1] + dy))
    return pos, logs


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
) -> Tuple[Image.Image, List[str]]:
    if not images:
        raise ValueError("该组没有图片")
    warn: List[str] = []
    if blend:
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
        "drag_gain": 0.25,
        "status": status,
    }


def canvas_payload(
    images: Sequence[Image.Image],
    shifts: Sequence[Shift],
    *,
    selected: int = 0,
    nudge_step: int = 1,
    diff_mode: bool = False,
    show_loupe: bool = True,
    status: str = "",
) -> dict:
    tiles = []
    for i, (img, (x, y)) in enumerate(zip(images, shifts)):
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
        "drag_gain": 0.25,
        "status": status,
    }


def shifts_from_canvas_payload(payload: dict | None, fallback: Sequence[Shift] | None = None) -> List[Shift]:
    if not isinstance(payload, dict):
        return list(fallback or [])
    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return list(fallback or [])
    out: List[Shift] = []
    for item in tiles:
        if not isinstance(item, dict):
            continue
        out.append((int(item.get("x") or 0), int(item.get("y") or 0)))
    return out or list(fallback or [])


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
        images.append(Image.open(text).convert("RGB"))
    return images
