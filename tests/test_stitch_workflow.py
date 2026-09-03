from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import sam3_demo.stitch_workflow as workflow
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


def _periodic_texture(h: int, w: int) -> np.ndarray:
    """Periodic industrial-like texture with a weak non-periodic seam cue."""

    yy, xx = np.mgrid[0:h, 0:w]
    period_x, period_y = 29, 23
    pattern = (
        90.0
        + 38.0 * np.sin(2.0 * np.pi * xx / period_x)
        + 31.0 * np.cos(2.0 * np.pi * yy / period_y)
        + 17.0 * np.sin(2.0 * np.pi * (xx + 2 * yy) / 11.0)
    )
    # A small local landmark prevents exact alias ties without changing the
    # dominant periodic structure under test.
    landmark = np.exp(-(((xx - 0.63 * w) / (0.07 * w)) ** 2 + ((yy - 0.48 * h) / (0.18 * h)) ** 2))
    noise = np.random.RandomState(123).normal(0.0, 24.0, size=(h, w))
    pattern += 100.0 * landmark + 0.03 * xx + 0.02 * yy + noise
    return np.clip(pattern, 0, 255).astype(np.uint8)


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

    def test_blend_single_tile_is_bitwise_identical(self):
        source = np.random.RandomState(31).randint(0, 256, (17, 23, 3), dtype=np.uint8)
        image = Image.fromarray(source, mode="RGB")
        result = stitch_images_blend([image], [(0, 0)])
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_blend_nonoverlap_edge_is_bitwise_identical(self):
        rng = np.random.RandomState(32)
        first = rng.randint(0, 256, (19, 21, 3), dtype=np.uint8)
        second = rng.randint(0, 256, (19, 21, 3), dtype=np.uint8)
        result = stitch_images_blend(
            [Image.fromarray(first, mode="RGB"), Image.fromarray(second, mode="RGB")],
            [(0, 0), (8, 0)],
        )
        # The rightmost 13 columns are covered only by the second tile.  Gain
        # matching must not alter them, and the positive edge floor must keep
        # the outermost row/column from becoming black.
        np.testing.assert_array_equal(np.asarray(result)[:, 21:29], second[:, 13:21])
        np.testing.assert_array_equal(np.asarray(result)[:, 28], second[:, 20])

    def test_export_rejects_extreme_gap_before_canvas_allocation(self):
        images = [
            Image.new("RGB", (10, 10), (10, 20, 30)),
            Image.new("RGB", (10, 10), (30, 20, 10)),
        ]
        with self.assertRaisesRegex(ValueError, "分块间距过大"):
            export_mosaic(images, [(0, 0), (1_000_000, 0)], blend=True)

    def test_periodic_match_uses_full_resolution_refinement(self):
        overlap = 96
        height, width = 128, 192
        canvas = _periodic_texture(height, width * 2 - overlap)
        first = canvas[:, :width]
        second = canvas[:, width - overlap:width - overlap + width]
        dx, dy, ncc, failed = match_translation(_rgb(first), _rgb(second), axis="horizontal")
        self.assertFalse(failed)
        self.assertGreater(ncc, 0.75)
        self.assertLessEqual(abs(dx - (width - overlap)), 3)
        self.assertLessEqual(abs(dy), 3)

    def test_periodic_aliases_use_full_resolution_ambiguity_fallback(self):
        height, width, overlap = 96, 160, 80
        yy, xx = np.mgrid[0:height, 0:width * 2 - overlap]
        canvas = np.clip(
            120.0
            + 70.0 * np.sin(2.0 * np.pi * xx / 16.0)
            + 45.0 * np.cos(2.0 * np.pi * yy / 21.0),
            0,
            255,
        ).astype(np.uint8)
        first = canvas[:, :width]
        second = canvas[:, width - overlap:width - overlap + width]
        _dx, _dy, ncc, failed = match_translation(_rgb(first), _rgb(second), axis="horizontal")
        self.assertTrue(failed)
        self.assertGreaterEqual(ncc, 0.2)

    def test_low_texture_uses_safe_fallback(self):
        first = _rgb(np.full((64, 80), 120, dtype=np.uint8))
        second = _rgb(np.full((64, 80), 120, dtype=np.uint8))
        dx, dy, ncc, failed = match_translation(first, second, axis="horizontal")
        self.assertTrue(failed)
        self.assertEqual((dx, dy), (80, 0))
        self.assertLess(ncc, 0.2)

    def test_shift_candidates_have_explicit_bound(self):
        seeds = [(10.0, 12.0), (25.0, 18.0)]
        candidates = workflow._shift_candidates(
            seeds, px=1.0, py=1.0, axis="both", wa=1000, ha=1000, max_candidates=50
        )
        self.assertLessEqual(len(candidates), 50)
        self.assertIn((10, 12), candidates)
        self.assertIn((25, 18), candidates)

    def test_tiny_period_both_axis_generation_is_bounded(self):
        candidates = workflow._shift_candidates(
            [(500.0, 500.0)],
            px=2.1,
            py=2.1,
            axis="both",
            wa=100_000,
            ha=100_000,
            max_candidates=32,
        )
        self.assertLessEqual(len(candidates), 32)
        self.assertIn((500, 500), candidates)

    def test_match_score_calls_have_hard_budget(self):
        image = _rgb(_texture(64, 64, 51))
        candidate_pool = [(index % 128, index // 128) for index in range(workflow._ALIGN_MAX_CANDIDATES)]
        original = workflow._alignment_score
        original_ncc = workflow._ncc_overlap
        with mock.patch.object(
            workflow, "_shift_candidates", return_value=candidate_pool
        ), mock.patch.object(
            workflow, "_alignment_score", side_effect=original
        ) as score, mock.patch.object(
            workflow, "_ncc_overlap", side_effect=original_ncc
        ) as ncc:
            match_translation(image, image, axis="both")
        self.assertLessEqual(score.call_count, workflow._ALIGN_MAX_SCORE_CALLS)
        self.assertLessEqual(ncc.call_count, workflow._ALIGN_MAX_NCC_CALLS)

    def test_thumbnail_scores_reuse_scaled_candidate_coordinates(self):
        image = _rgb(_texture(512, 512, 53))
        candidates = [(100 + index, 0) for index in range(8)]
        original = workflow._highpass_alignment_score
        with mock.patch.object(
            workflow, "_shift_candidates", return_value=candidates
        ), mock.patch.object(
            workflow, "_highpass_alignment_score", side_effect=original
        ) as score:
            match_translation(image, image, axis="horizontal")
        self.assertLess(score.call_count, len(candidates))

    def test_probe_seeds_use_medium_resolution(self):
        image = _rgb(_texture(640, 960, 54))
        original = workflow._probe_seeds
        observed_sizes = []

        def capture(first, second, axis):
            observed_sizes.append(max(first.shape + second.shape))
            return original(first, second, axis)

        with mock.patch.object(workflow, "_probe_seeds", side_effect=capture):
            match_translation(image, image, axis="horizontal")
        self.assertEqual(len(observed_sizes), 2)
        self.assertTrue(all(size <= workflow._ALIGN_MID_MAX_SIDE for size in observed_sizes))

    def test_canvas_shifts_use_tile_index_not_client_order(self):
        fallback = [(0, 0), (20, 0), (40, 0)]
        payload = {
            "tiles": [
                {"index": 2, "x": 42, "y": 4},
                {"index": 0, "x": 3, "y": 1},
                {"index": 1, "x": 23, "y": 2},
            ]
        }
        self.assertEqual(
            workflow.shifts_from_canvas_payload(payload, fallback),
            [(3, 1), (23, 2), (42, 4)],
        )

    def test_canvas_payload_uses_original_rate_drag_by_default(self):
        image = _rgb(_texture(20, 30, 52))
        self.assertEqual(workflow.empty_canvas_payload()["drag_gain"], 1.0)
        self.assertEqual(workflow.canvas_payload([image], [(0, 0)])["drag_gain"], 1.0)

    def test_rotation_normalization_and_indexed_canvas_roundtrip(self):
        image = _rgb(_texture(20, 30, 61))
        payload = workflow.canvas_payload(
            [image, image],
            [(0, 0), (30, 0)],
            rotations=[540, -181],
        )
        self.assertEqual(
            [tile["rotation_deg"] for tile in payload["tiles"]],
            [180.0, 179.0],
        )
        reordered = {
            "tiles": [
                {"index": 1, "rotation_deg": -10.25},
                {"index": 0, "rotation_deg": 3.5},
            ]
        }
        self.assertEqual(
            workflow.rotations_from_canvas_payload(reordered, [0.0, 0.0]),
            [3.5, -10.25],
        )

    def test_canvas_rotation_rejects_partial_or_nonfinite_payload(self):
        fallback = [1.0, 2.0]
        invalid_payloads = [
            {"tiles": [{"index": 0, "rotation_deg": 5.0}]},
            {
                "tiles": [
                    {"index": 0, "rotation_deg": float("nan")},
                    {"index": 1, "rotation_deg": 2.0},
                ]
            },
        ]
        for payload in invalid_payloads:
            self.assertEqual(
                workflow.rotations_from_canvas_payload(payload, fallback),
                fallback,
            )

    def test_zero_rotation_keeps_existing_export_bitwise_identical(self):
        images = [
            _rgb(_texture(25, 31, 62)),
            _rgb(_texture(25, 31, 63)),
        ]
        shifts = [(0, 0), (21, 0)]
        baseline, _ = export_mosaic(images, shifts, blend=True)
        rotated_api, _ = export_mosaic(
            images,
            shifts,
            blend=True,
            rotations=[0.0, 360.0],
        )
        np.testing.assert_array_equal(np.asarray(rotated_api), np.asarray(baseline))

    def test_rotated_export_expands_around_tile_center_without_white_corners(self):
        source = np.zeros((10, 20, 3), dtype=np.uint8)
        source[2:8, 3:17] = (220, 40, 20)
        image = Image.fromarray(source, mode="RGB")
        mosaic, warnings = export_mosaic(
            [image],
            [(0, 0)],
            blend=False,
            rotations=[45.0],
        )
        self.assertGreater(mosaic.width, image.width)
        self.assertGreater(mosaic.height, image.height)
        array = np.asarray(mosaic)
        self.assertEqual(tuple(array[0, 0]), (0, 0, 0))
        self.assertGreater(int(array[..., 0].max()), 150)
        self.assertEqual(warnings, [])

    def test_positive_rotation_matches_layout_clockwise_convention(self):
        source = np.zeros((5, 9, 3), dtype=np.uint8)
        source[0:2, 0:3] = (255, 30, 10)
        image = Image.fromarray(source, mode="RGB")
        mosaic, _ = export_mosaic(
            [image],
            [(0, 0)],
            blend=False,
            rotations=[90.0],
        )
        expected_rgba = image.convert("RGBA").rotate(
            -90.0,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        expected = Image.new("RGB", expected_rgba.size, (0, 0, 0))
        expected.paste(
            expected_rgba.convert("RGB"),
            (0, 0),
            expected_rgba.getchannel("A"),
        )
        np.testing.assert_array_equal(np.asarray(mosaic), np.asarray(expected))

    def test_canvas_shifts_reject_invalid_indexed_payload(self):
        fallback = [(0, 0), (20, 0)]
        invalid_payloads = [
            {"tiles": [{"index": 0, "x": 1, "y": 2}]},
            {"tiles": [{"index": 0, "x": 1, "y": 2}, {"index": 0, "x": 3, "y": 4}]},
            {"tiles": [{"index": 2, "x": 1, "y": 2}, {"index": 0, "x": 3, "y": 4}]},
            {"tiles": [{"index": 0, "x": np.nan, "y": 2}, {"index": 1, "x": 3, "y": 4}]},
        ]
        for payload in invalid_payloads:
            self.assertEqual(workflow.shifts_from_canvas_payload(payload, fallback), fallback)

    def test_auto_align_prepares_each_tile_once(self):
        images = [_rgb(_texture(48, 64, seed)) for seed in (41, 42, 43)]
        original = workflow._prepare_alignment_arrays
        with mock.patch.object(
            workflow,
            "_prepare_alignment_arrays",
            side_effect=lambda image, hp_kernel=21: original(image, hp_kernel),
        ) as prepare:
            auto_align_images(images, "horizontal")
        self.assertEqual(prepare.call_count, len(images))

    def test_default_horizontal_abut(self):
        imgs = [_rgb(_texture(20, 30, 1)), _rgb(_texture(20, 25, 2))]
        self.assertEqual(default_shifts_for_layout(imgs, "horizontal"), [(0, 0), (30, 0)])


if __name__ == "__main__":
    unittest.main()
