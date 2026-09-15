from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from sam3_demo.stitch_workflow import auto_align_images, default_shifts_for_layout


_TILE_SIZE = (100, 80)
_MIN_PRIMARY_FRACTION = 0.65


def _images(count: int) -> list[Image.Image]:
    # Uniform tiles keep this contract test independent of NCC/period recovery.
    return [Image.new("RGB", _TILE_SIZE, (127, 127, 127)) for _ in range(count)]


def _axis_candidate(fraction: float, score: float):
    horizontal_step = int(round(_TILE_SIZE[0] * fraction))
    vertical_step = int(round(_TILE_SIZE[1] * fraction))

    def fake_match(_a, _b, axis):
        if axis == "horizontal":
            return [(horizontal_step, 0, score)]
        if axis == "vertical":
            return [(0, vertical_step, score)]
        raise AssertionError(f"unexpected axis: {axis}")

    return fake_match


def _assert_linear_fallback(testcase: unittest.TestCase, logs: list[str]) -> None:
    joined = "\n".join(logs)
    testcase.assertTrue(logs)
    testcase.assertIn("\u8b66\u544a", joined)
    testcase.assertIn("\u96f6\u91cd\u53e0\u76f8\u90bb\u6392\u5217", joined)
    testcase.assertNotIn("\u90bb\u63a5 \u56fe", joined)
    testcase.assertNotIn("\u8d28\u91cf=", joined)


def _assert_grid_fallback(testcase: unittest.TestCase, logs: list[str]) -> None:
    joined = "\n".join(logs)
    testcase.assertTrue(logs)
    testcase.assertIn("\u4fdd\u7559\u89c4\u5219\u7f51\u683c", joined)
    testcase.assertNotIn("\u9009\u62e9\u62d3\u6251", joined)
    testcase.assertNotIn("\u8054\u5408\u5339\u914d", joined)
    testcase.assertNotIn("\u8def\u5f84\u8d28\u91cf", joined)
    testcase.assertNotIn("\u8054\u5408\u8d28\u91cf", joined)


def _assert_step_geometry(
    testcase: unittest.TestCase,
    images: list[Image.Image],
    shifts: list[tuple[int, int]],
    source: int,
    target: int,
    axis: str,
    expected_step: int,
) -> None:
    dx = shifts[target][0] - shifts[source][0]
    dy = shifts[target][1] - shifts[source][1]
    if axis == "horizontal":
        primary = dx
        cross = dy
        extent = images[source].width
        orthogonal = min(images[source].height, images[target].height)
        target_extent = images[target].width
    else:
        primary = dy
        cross = dx
        extent = images[source].height
        orthogonal = min(images[source].width, images[target].width)
        target_extent = images[target].height

    testcase.assertEqual(primary, expected_step)
    testcase.assertGreaterEqual(primary, _MIN_PRIMARY_FRACTION * extent)
    testcase.assertLessEqual(primary, extent)
    testcase.assertLessEqual(abs(cross), 0.1 * orthogonal + 1e-6)

    overlap_extent = max(0, extent - primary)
    newly_covered_extent = max(0, target_extent - overlap_extent)
    newly_covered_area = newly_covered_extent * orthogonal
    testcase.assertGreater(newly_covered_area, 0)
    testcase.assertEqual(newly_covered_area, primary * orthogonal)


class StitchOverlapBudgetTest(unittest.TestCase):
    def test_high_score_40_percent_step_is_rejected_for_linear_layouts(self):
        for layout, axis in (("horizontal", "horizontal"), ("vertical", "vertical")):
            with self.subTest(layout=layout), patch(
                "sam3_demo.stitch_grid_alignment.match_grid_pair",
                side_effect=_axis_candidate(0.40, 0.999),
            ):
                images = _images(2)
                shifts, logs = auto_align_images(images, layout)

            self.assertEqual(shifts, default_shifts_for_layout(images, layout))
            fallback_step = images[0].width if axis == "horizontal" else images[0].height
            actual_step = (
                shifts[1][0] - shifts[0][0]
                if axis == "horizontal"
                else shifts[1][1] - shifts[0][1]
            )
            self.assertEqual(actual_step, fallback_step)
            _assert_linear_fallback(self, logs)

    def test_valid_75_percent_step_is_accepted_for_linear_layouts(self):
        for layout, axis in (("horizontal", "horizontal"), ("vertical", "vertical")):
            with self.subTest(layout=layout), patch(
                "sam3_demo.stitch_grid_alignment.match_grid_pair",
                side_effect=_axis_candidate(0.75, 0.81),
            ):
                images = _images(2)
                shifts, logs = auto_align_images(images, layout)

            expected = [(0, 0), (75, 0)] if axis == "horizontal" else [(0, 0), (0, 60)]
            self.assertEqual(shifts, expected)
            _assert_step_geometry(self, images, shifts, 0, 1, axis, 75 if axis == "horizontal" else 60)
            self.assertTrue(logs)
            self.assertNotIn("\u8b66\u544a", "\n".join(logs))

    def test_high_score_40_percent_step_is_rejected_for_grid_layouts(self):
        for layout, count in (("grid_2x2", 4), ("grid_2xn", 6)):
            with self.subTest(layout=layout), patch(
                "sam3_demo.stitch_grid_alignment.match_grid_pair",
                side_effect=_axis_candidate(0.40, 0.999),
            ):
                images = _images(count)
                shifts, logs = auto_align_images(images, layout)

            self.assertEqual(shifts, default_shifts_for_layout(images, layout))
            _assert_grid_fallback(self, logs)

    def test_valid_75_percent_step_is_accepted_for_grid_layouts(self):
        cases = (
            (
                "grid_2x2",
                4,
                [(0, 0), (75, 0), (0, 60), (75, 60)],
                (
                    (0, 1, "horizontal", 75),
                    (0, 2, "vertical", 60),
                    (1, 3, "vertical", 60),
                    (2, 3, "horizontal", 75),
                ),
            ),
            (
                "grid_2xn",
                6,
                [(0, 0), (75, 0), (150, 0), (0, 60), (75, 60), (150, 60)],
                (
                    (0, 1, "horizontal", 75),
                    (1, 2, "horizontal", 75),
                    (3, 4, "horizontal", 75),
                    (4, 5, "horizontal", 75),
                    (0, 3, "vertical", 60),
                    (1, 4, "vertical", 60),
                    (2, 5, "vertical", 60),
                ),
            ),
        )
        for layout, count, expected, edges in cases:
            with self.subTest(layout=layout), patch(
                "sam3_demo.stitch_grid_alignment.match_grid_pair",
                side_effect=_axis_candidate(0.75, 0.81),
            ):
                images = _images(count)
                shifts, logs = auto_align_images(images, layout)

            self.assertEqual(shifts, expected)
            for source, target, axis, step in edges:
                _assert_step_geometry(
                    self,
                    images,
                    shifts,
                    source,
                    target,
                    axis,
                    step,
                )
            self.assertTrue(logs)
            joined = "\n".join(logs)
            self.assertNotIn("\u4fdd\u7559\u89c4\u5219\u7f51\u683c", joined)


if __name__ == "__main__":
    unittest.main()
