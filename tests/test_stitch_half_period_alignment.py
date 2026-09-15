from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sam3_demo.stitch_grid_alignment import align_four_tiles
from sam3_demo.stitch_workflow import (
    _alignment_evidence, _alignment_quality, _prepare_alignment_arrays, auto_align_images,
)


def _periodic_canvas(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = (
        112.0
        + 34.0 * np.sin(2.0 * np.pi * xx / 14.0)
        + 29.0 * np.cos(2.0 * np.pi * yy / 13.0)
        + 16.0 * np.sin(2.0 * np.pi * (xx + 2.0 * yy) / 9.0)
    )
    landmarks = (
        26.0 * np.exp(-(((xx - 0.31 * width) / 9.0) ** 2
                         + ((yy - 0.42 * height) / 13.0) ** 2))
        + 21.0 * np.exp(-(((xx - 0.74 * width) / 8.0) ** 2
                           + ((yy - 0.68 * height) / 10.0) ** 2))
    )
    gradient = 0.11 * xx + 0.07 * yy
    return np.clip(pattern + landmarks + gradient, 0, 255).astype(np.uint8)


def _rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB")


def _tiles(step_x: int, step_y: int, order: tuple[int, ...]) -> list[Image.Image]:
    tile_width = 120
    tile_height = 120
    canvas = _periodic_canvas(tile_height + step_y, tile_width + step_x)
    slots = [
        _rgb(canvas[0:tile_height, 0:tile_width]),
        _rgb(canvas[0:tile_height, step_x:step_x + tile_width]),
        _rgb(canvas[step_y:step_y + tile_height, 0:tile_width]),
        _rgb(canvas[step_y:step_y + tile_height, step_x:step_x + tile_width]),
    ]
    return [slots[index] for index in order]


class HalfPeriodAlignmentTest(unittest.TestCase):
    def test_equal_agreement_does_not_reward_larger_overlap(self):
        evidence = {"available": True, "raw_median": 0.8, "highpass_median": 0.7}
        narrow = _alignment_quality(0.8, {**evidence, "overlap_ratio": 0.10})
        wide = _alignment_quality(0.8, {**evidence, "overlap_ratio": 0.65})
        self.assertEqual(narrow, wide)

    def test_narrow_seam_has_regional_evidence(self):
        rng = np.random.default_rng(42)
        canvas = rng.integers(0, 256, (120, 224), dtype=np.uint8)
        a = _prepare_alignment_arrays(_rgb(canvas[:, :120]))
        b = _prepare_alignment_arrays(_rgb(canvas[:, 104:224]))
        evidence = _alignment_evidence(a, b, 104, 0)
        self.assertTrue(evidence["available"])
        self.assertGreater(evidence["raw_median"], 0.99)
        self.assertLess(evidence["overlap_ratio"], 0.15)

    def test_horizontal_and_vertical_keep_true_adjacent_steps(self):
        horizontal = _tiles(84, 82, (0, 1))
        horizontal_positions, _ = auto_align_images(horizontal, "horizontal")
        self.assertEqual(horizontal_positions[0], (0, 0))
        self.assertGreaterEqual(horizontal_positions[1][0], 0.65 * horizontal[0].width)
        self.assertLessEqual(abs(horizontal_positions[1][0] - 84), 5)
        self.assertLessEqual(abs(horizontal_positions[1][1]), 5)

        vertical = _tiles(84, 82, (0, 2))
        vertical_positions, _ = auto_align_images(vertical, "vertical")
        self.assertEqual(vertical_positions[0], (0, 0))
        self.assertGreaterEqual(vertical_positions[1][1], 0.65 * vertical[0].height)
        self.assertLessEqual(abs(vertical_positions[1][1] - 82), 5)
        self.assertLessEqual(abs(vertical_positions[1][0]), 5)

    def test_row_major_2x2_returns_original_index_positions(self):
        images = _tiles(84, 82, (0, 1, 2, 3))
        positions, logs = align_four_tiles(
            [_prepare_alignment_arrays(image) for image in images]
        )
        self.assertEqual(positions, [(0, 0), (84, 0), (0, 82), (84, 82)])
        self.assertIn("row-major", logs[0])
        self.assertIn("闭环误差", logs[0])

    def test_clockwise_2x2_reorders_slots_but_preserves_input_indices(self):
        images = _tiles(84, 82, (0, 1, 3, 2))
        positions, logs = align_four_tiles(
            [_prepare_alignment_arrays(image) for image in images]
        )
        self.assertEqual(positions, [(0, 0), (84, 0), (84, 82), (0, 82)])
        self.assertIn("clockwise/snake", logs[0])
        self.assertIn("闭环误差", logs[0])


if __name__ == "__main__":
    unittest.main()
