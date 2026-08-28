from __future__ import annotations

import math
import unittest

import numpy as np
from PIL import Image

from sam3_demo.stitch_workflow import (
    auto_align_images,
    default_shifts_for_layout,
    export_mosaic,
    match_translation,
    select_worst_tile,
    stitch_images_blend,
    stitch_images_overlay,
)


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 220, size=(h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    stripe = ((xx // 7 + yy // 5) % 2) * 40
    return np.clip(base.astype(np.int16) + stripe, 0, 255).astype(np.uint8)


def _rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")


class StitchWorkflowTest(unittest.TestCase):
    def test_empty_align(self):
        shifts, logs = auto_align_images([], "horizontal")
        self.assertEqual(shifts, [])
        self.assertEqual(logs, [])

    def test_grid_2x2_requires_four(self):
        imgs = [_rgb(_texture(32, 32, 1)), _rgb(_texture(32, 32, 2))]
        with self.assertRaises(ValueError):
            auto_align_images(imgs, "grid_2x2")
        with self.assertRaises(ValueError):
            default_shifts_for_layout(imgs, "grid_2x2")

    def test_horizontal_match_translation_sign(self):
        overlap = 48
        h, left_w, right_w = 64, 96, 88
        canvas = _texture(h, left_w + right_w - overlap, 7)
        a = _rgb(canvas[:, :left_w])
        b = _rgb(canvas[:, left_w - overlap:left_w - overlap + right_w])
        dx, dy, ncc, failed = match_translation(a, b, axis="horizontal")
        self.assertFalse(failed)
        self.assertGreater(ncc, 0.4)
        self.assertLessEqual(abs(dx - (left_w - overlap)), 3)
        self.assertLessEqual(abs(dy), 2)

    def test_auto_align_horizontal_chain(self):
        overlap = 40
        h, w = 48, 80
        canvas = _texture(h, w * 3 - overlap * 2, 11)
        tiles = [
            _rgb(canvas[:, 0:w]),
            _rgb(canvas[:, w - overlap:2 * w - overlap]),
            _rgb(canvas[:, 2 * (w - overlap):2 * (w - overlap) + w]),
        ]
        shifts, logs = auto_align_images(tiles, "horizontal")
        self.assertEqual(len(shifts), 3)
        self.assertEqual(shifts[0], (0, 0))
        self.assertEqual(len(logs), 2)
        self.assertLessEqual(abs(shifts[1][0] - (w - overlap)), 4)
        self.assertLessEqual(abs(shifts[2][0] - 2 * (w - overlap)), 6)

    def test_grid_2x2_closed_loop_average(self):
        overlap_x, overlap_y = 36, 28
        h, w = 56, 72
        canvas = _texture(h * 2 - overlap_y, w * 2 - overlap_x, 19)
        y1 = h - overlap_y
        x1 = w - overlap_x
        tiles = [
            _rgb(canvas[0:h, 0:w]),
            _rgb(canvas[0:h, x1:x1 + w]),
            _rgb(canvas[y1:y1 + h, 0:w]),
            _rgb(canvas[y1:y1 + h, x1:x1 + w]),
        ]
        shifts, logs = auto_align_images(tiles, "grid_2x2")
        self.assertEqual(len(shifts), 4)
        self.assertEqual(len(logs), 4)
        self.assertEqual(shifts[0], (0, 0))
        expected_x = w - overlap_x
        expected_y = h - overlap_y
        self.assertLessEqual(abs(shifts[1][0] - expected_x), 4)
        self.assertLessEqual(abs(shifts[2][1] - expected_y), 4)
        self.assertLessEqual(abs(shifts[3][0] - expected_x), 5)
        self.assertLessEqual(abs(shifts[3][1] - expected_y), 5)

    def test_select_worst_tile_skips_origin(self):
        baseline = [(0, 0), (100, 0), (0, 80), (100, 80)]
        shifts = [(0, 0), (102, 1), (3, 90), (140, 80)]
        self.assertEqual(select_worst_tile(shifts, baseline), 3)

    def test_blend_and_overlay_export_size(self):
        a = _rgb(_texture(40, 50, 3))
        b = _rgb(_texture(40, 50, 4))
        shifts = [(0, 0), (40, 0)]
        overlay = stitch_images_overlay([a, b], shifts)
        blend = stitch_images_blend([a, b], shifts)
        mosaic, warn = export_mosaic([a, b], shifts, layout="horizontal", blend=True)
        self.assertEqual(overlay.size, (90, 40))
        self.assertEqual(blend.size, overlay.size)
        self.assertEqual(mosaic.size, overlay.size)
        self.assertEqual(warn, [])

    def test_default_horizontal_abut(self):
        imgs = [_rgb(_texture(20, 30, 1)), _rgb(_texture(20, 25, 2))]
        self.assertEqual(default_shifts_for_layout(imgs, "horizontal"), [(0, 0), (30, 0)])


if __name__ == "__main__":
    unittest.main()
