from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam3_demo.stitch_workflow import auto_align_images, default_shifts_for_layout


def _textured_canvas(height: int, width: int, seed: int) -> np.ndarray:
    """Make one deterministic, locally unique canvas for all tile crops."""

    rng = np.random.RandomState(seed)
    noise = rng.randint(24, 224, size=(height, width), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    signal = noise.astype(np.int16)
    signal += (17 * xx + 31 * yy) % 53
    signal += ((xx // 9 + yy // 7) % 2) * 18
    return np.clip(signal, 0, 255).astype(np.uint8)


def _rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB")


def _crop_tiles(
    canvas: np.ndarray,
    tile_width: int,
    tile_height: int,
    positions: list[tuple[int, int]],
) -> list[Image.Image]:
    return [
        _rgb(canvas[y : y + tile_height, x : x + tile_width])
        for x, y in positions
    ]


class StitchLayoutTopologyTest(unittest.TestCase):
    def _assert_horizontal_topology(
        self, images: list[Image.Image], shifts: list[tuple[int, int]]
    ) -> None:
        self.assertEqual(len(shifts), len(images))
        for index in range(1, len(images)):
            dx = shifts[index][0] - shifts[index - 1][0]
            dy = shifts[index][1] - shifts[index - 1][1]
            predecessor = images[index - 1]
            cross_limit = 0.1 * min(predecessor.height, images[index].height)
            self.assertGreaterEqual(
                dx,
                0.5 * predecessor.width,
                msg=f"horizontal step {index}: {shifts[index - 1]} -> {shifts[index]}",
            )
            self.assertLessEqual(abs(dy), cross_limit + 1e-6)

    def _assert_vertical_topology(
        self, images: list[Image.Image], shifts: list[tuple[int, int]]
    ) -> None:
        self.assertEqual(len(shifts), len(images))
        for index in range(1, len(images)):
            dx = shifts[index][0] - shifts[index - 1][0]
            dy = shifts[index][1] - shifts[index - 1][1]
            predecessor = images[index - 1]
            cross_limit = 0.1 * min(predecessor.width, images[index].width)
            self.assertGreaterEqual(
                dy,
                0.5 * predecessor.height,
                msg=f"vertical step {index}: {shifts[index - 1]} -> {shifts[index]}",
            )
            self.assertLessEqual(abs(dx), cross_limit + 1e-6)

    def test_horizontal_reversed_order_keeps_monotone_layout(self):
        tile_width, tile_height = 56, 48
        step = 32
        positions = [(index * step, 0) for index in range(4)]
        canvas = _textured_canvas(tile_height, tile_width + 3 * step, seed=101)
        images = list(reversed(_crop_tiles(canvas, tile_width, tile_height, positions)))

        shifts, _logs = auto_align_images(images, "horizontal")

        self.assertEqual(shifts[0], (0, 0))
        self._assert_horizontal_topology(images, shifts)

    def test_vertical_reversed_order_keeps_monotone_layout(self):
        tile_width, tile_height = 52, 56
        step = 32
        positions = [(0, index * step) for index in range(4)]
        canvas = _textured_canvas(tile_height + 3 * step, tile_width, seed=102)
        images = list(reversed(_crop_tiles(canvas, tile_width, tile_height, positions)))

        shifts, _logs = auto_align_images(images, "vertical")

        self.assertEqual(shifts[0], (0, 0))
        self._assert_vertical_topology(images, shifts)

    def test_horizontal_chain_bounds_cross_offset_from_first_tile(self):
        images = [Image.new("RGB", (40, 40), (127, 127, 127)) for _ in range(4)]
        candidates = [
            [(28, 4, 0.95), (28, -4, 0.90)]
            for _ in range(3)
        ]

        with patch(
            "sam3_demo.stitch_grid_alignment.match_grid_pair",
            side_effect=candidates,
        ):
            shifts, _logs = auto_align_images(images, "horizontal")

        self.assertEqual(shifts, [(0, 0), (28, 4), (56, 0), (84, 4)])
        for _x, y in shifts:
            self.assertLessEqual(abs(y), 0.1 * images[0].height)

    def test_grid_2xn_six_overlapping_tiles_follow_row_major_loop(self):
        tile_width, tile_height = 48, 44
        step_x, step_y = 34, 31
        positions = [
            (column * step_x, row * step_y)
            for row in range(2)
            for column in range(3)
        ]
        canvas = _textured_canvas(
            tile_height + step_y,
            tile_width + 2 * step_x,
            seed=103,
        )
        images = _crop_tiles(canvas, tile_width, tile_height, positions)

        shifts, _logs = auto_align_images(images, "grid_2xn")

        self.assertEqual(len(shifts), 6)
        self.assertEqual(shifts[0], (0, 0))
        for actual, expected in zip(shifts, positions):
            self.assertLessEqual(
                math.hypot(actual[0] - expected[0], actual[1] - expected[1]),
                6.0,
            )
        for row_start in (0, 3):
            self._assert_horizontal_topology(
                images[row_start : row_start + 3],
                shifts[row_start : row_start + 3],
            )
        for column in range(3):
            dx = shifts[3 + column][0] - shifts[column][0]
            dy = shifts[3 + column][1] - shifts[column][1]
            self.assertGreaterEqual(dy, 0.5 * images[column].height)
            self.assertLessEqual(abs(dx), 0.1 * images[column].width + 1e-6)
        self.assertLess(shifts[1][0] - shifts[0][0], tile_width)
        self.assertLess(shifts[2][0] - shifts[1][0], tile_width)
        self.assertLess(shifts[4][0] - shifts[3][0], tile_width)
        self.assertLess(shifts[5][0] - shifts[4][0], tile_width)

    def test_empty_and_single_image_are_stable_for_supported_layouts(self):
        image = _rgb(_textured_canvas(24, 28, seed=104))
        for layout in ("horizontal", "vertical", "grid_2x2", "grid_2xn"):
            with self.subTest(layout=layout):
                empty_shifts, empty_logs = auto_align_images([], layout)
                self.assertEqual(empty_shifts, [])
                self.assertEqual(empty_logs, [])

        for layout in ("horizontal", "vertical"):
            with self.subTest(layout=layout):
                single_shifts, single_logs = auto_align_images([image], layout)
                self.assertEqual(single_shifts, [(0, 0)])
                self.assertEqual(single_logs, [])

        with self.assertRaises(ValueError):
            auto_align_images([image], "grid_2x2")

    def test_low_texture_uses_layout_fallback_positions(self):
        low_texture = Image.new("RGB", (40, 36), (127, 127, 127))
        for layout in ("horizontal", "vertical"):
            with self.subTest(layout=layout):
                images = [low_texture.copy() for _ in range(2)]
                shifts, _logs = auto_align_images(images, layout)
                self.assertEqual(shifts, default_shifts_for_layout(images, layout))


if __name__ == "__main__":
    unittest.main()
