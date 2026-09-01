from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import template_stitch_workflow as workflow

from sam3_demo.template_stitch_workflow import (
    detect_hole_centers,
    estimate_hole_period,
    load_template_images,
    overlap_ncc,
    refine_shift_with_holes,
    render_template_preview,
    score_pair_shift,
    stitch_template_group,
    stitch_images,
)


def _tile(width: int = 192, height: int = 160, seed: int = 0) -> Image.Image:
    """Deterministic periodic dark-hole tile used by all tests."""

    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    texture = rng.integers(0, 7, size=(height, width, 1), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) - texture, 0, 255).astype(np.uint8)
    yy, xx = np.ogrid[:height, :width]
    for cy in (40, 88, 136):
        for cx in (40, 88, 136):
            hole = (xx - cx) ** 2 + (yy - cy) ** 2 <= 12**2
            image[hole] = (18, 18, 18)
    return Image.fromarray(image, mode="RGB")


class TemplateStitchWorkflowTests(unittest.TestCase):
    def test_2x2_sift_extracts_each_image_once_and_reuses_reverse_pair(self) -> None:
        grays = [np.full((32, 32), index, dtype=np.uint8) for index in range(4)]
        features = [(f"key-{index}", f"desc-{index}") for index in range(4)]
        shifts = [
            (np.array([float(index + 1), float(index + 2)]), index + 10)
            for index in range(6)
        ]
        with mock.patch.object(
            workflow,
            "_extract_sift_features",
            side_effect=features,
        ) as extract, mock.patch.object(
            workflow,
            "_sift_shift_from_features",
            side_effect=shifts,
        ) as match:
            cache = workflow._build_sift_pair_cache(grays)
        self.assertEqual(extract.call_count, 4)
        self.assertEqual(match.call_count, 6)
        np.testing.assert_array_equal(cache[(1, 0)][0], -cache[(0, 1)][0])
        self.assertEqual(cache[(1, 0)][1], cache[(0, 1)][1])

    def test_2x2_ncc_coarse_search_is_downsampled(self) -> None:
        grays = [np.zeros((800, 1200), dtype=np.uint8) for _ in range(4)]
        centers = [np.array([[40.0, 40.0], [80.0, 80.0]]) for _ in range(4)]
        seen_shapes = []

        def coarse(gray_a, gray_b, period, kind):
            seen_shapes.append(gray_a.shape)
            shift = np.array([gray_a.shape[1] * 0.6, 0.0])
            if kind == "vertical":
                shift = np.array([0.0, gray_a.shape[0] * 0.6])
            return 0.5, shift

        with mock.patch.object(workflow, "_strip_ncc_shift", side_effect=coarse), mock.patch.object(
            workflow,
            "refine_shift_with_holes",
            side_effect=lambda _a, _b, shift, _period: shift,
        ):
            cache = workflow._coarse_ncc_pair_cache(grays, centers, 80.0)
        self.assertEqual(len(cache), 24)
        self.assertEqual(len(seen_shapes), 24)
        self.assertTrue(all(max(shape) <= 320 for shape in seen_shapes))

    def test_2x2_full_resolution_ncc_is_limited_to_top_layouts(self) -> None:
        grays = [np.zeros((80, 96), dtype=np.uint8) for _ in range(4)]
        centers = [np.array([[20.0, 20.0], [40.0, 40.0]]) for _ in range(4)]
        sift_cache = {
            (i, j): (None, 0) for i in range(4) for j in range(4) if i != j
        }
        coarse_cache = {}
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                coarse_cache[(i, j, "horizontal")] = (0.8, np.array([60.0, 0.0]))
                coarse_cache[(i, j, "vertical")] = (0.8, np.array([0.0, 48.0]))

        with mock.patch.object(
            workflow, "_build_sift_pair_cache", return_value=sift_cache
        ), mock.patch.object(
            workflow, "_coarse_ncc_pair_cache", return_value=coarse_cache
        ), mock.patch.object(
            workflow,
            "_fullres_ncc_refine",
            side_effect=lambda _a, _b, shift, _period, _kind: (0.9, shift),
        ) as refine, mock.patch.object(
            workflow,
            "refine_shift_with_holes",
            side_effect=lambda _a, _b, shift, _period: shift,
        ):
            positions, order, method = workflow.solve_grid_2x2(grays, centers, 20.0)
        self.assertEqual(set(positions), {0, 1, 2, 3})
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertEqual(method, "NCC")
        self.assertLessEqual(refine.call_count, 8)
        self.assertLess(refine.call_count, 24)

    def test_full_resolution_ncc_refines_nearby_translation(self) -> None:
        rng = np.random.default_rng(41)
        canvas = rng.integers(0, 256, size=(130, 150), dtype=np.uint8)
        gray_a = canvas[:100, :100]
        gray_b = canvas[3:103, 40:140]
        score, shift = workflow._fullres_ncc_refine(
            gray_a,
            gray_b,
            np.array([39.0, 2.0]),
            period=25.0,
            kind="horizontal",
        )
        np.testing.assert_array_equal(shift, np.array([40.0, 3.0]))
        self.assertGreater(score, 0.99)

    def test_hole_detection_and_period_are_deterministic(self) -> None:
        image = _tile()
        first = detect_hole_centers(image, min_area=100)
        second = detect_hole_centers(image, min_area=100)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(len(first), 4)
        self.assertAlmostEqual(estimate_hole_period([first]), 48.0, delta=2.5)

    def test_zero_or_insufficient_holes_fail_safely(self) -> None:
        blank = Image.new("RGB", (192, 160), (230, 230, 230))
        with self.assertRaises(ValueError):
            stitch_template_group([blank] * 4, ["a", "b", "c", "d"], 2, 2)
        one = np.asarray(blank).copy()
        one[70:90, 86:106] = 0
        single = Image.fromarray(one, mode="RGB")
        with self.assertRaises(ValueError):
            stitch_template_group([single] * 4, ["a", "b", "c", "d"], 2, 2)

    def test_2x2_returns_pil_layout_meta_and_preview(self) -> None:
        images = [_tile(seed=index) for index in range(4)]
        result, meta = stitch_template_group(images, [f"tile_{i}.png" for i in range(4)], 2, 2)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(len(meta["positions"]), 4)
        self.assertEqual(sorted(meta["order"]), [0, 1, 2, 3])
        self.assertIn(meta["layout_method"], {"SIFT", "NCC"})
        self.assertEqual(len(meta["hole_counts"]), 4)
        preview = render_template_preview(images, {i: p for i, p in enumerate(meta["positions"])}, [np.asarray(c) for c in meta["centers"]], meta["order"])
        self.assertIsInstance(preview, Image.Image)
        self.assertEqual(preview.size, result.size)

    def test_2x2_recovers_overlapping_crop_extent(self) -> None:
        rng = np.random.default_rng(20260831)
        canvas = rng.integers(205, 246, size=(260, 300, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:260, :300]
        for cy in (40, 88, 136, 184, 232):
            for cx in (40, 88, 136, 184, 232, 280):
                canvas[(xx - cx) ** 2 + (yy - cy) ** 2 <= 11**2] = (18, 18, 18)
        offsets = [(108, 100), (0, 0), (108, 0), (0, 100)]
        images = [
            Image.fromarray(canvas[y : y + 160, x : x + 192], mode="RGB")
            for x, y in offsets
        ]
        result, meta = stitch_template_group(
            images,
            [f"crop_{index}.png" for index in range(4)],
            2,
            2,
        )
        self.assertIn(meta["layout_method"], {"SIFT", "NCC"})
        self.assertEqual(sorted(meta["order"]), [0, 1, 2, 3])
        self.assertAlmostEqual(result.width, 300, delta=4)
        self.assertAlmostEqual(result.height, 260, delta=4)

    def test_2xn_chain_and_output_are_stable(self) -> None:
        images = [_tile(seed=index) for index in range(6)]
        first, meta_first = stitch_template_group(images, [str(i) for i in range(6)], 2, 3)
        second, meta_second = stitch_template_group(images, [str(i) for i in range(6)], 2, 3)
        self.assertEqual(first.size, second.size)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(meta_first["positions"], meta_second["positions"])
        self.assertEqual(meta_first["layout_method"], "chain")
        self.assertEqual(meta_first["order"], list(range(6)))

    def test_hole_refinement_uses_b_to_a_translation_direction(self) -> None:
        centers_a = np.array(
            [[80.0, 40.0], [128.0, 40.0], [80.0, 88.0], [128.0, 88.0]]
        )
        expected = np.array([64.0, 3.0])
        centers_b = centers_a - expected
        refined = refine_shift_with_holes(
            centers_a,
            centers_b,
            coarse=np.array([61.0, 5.0]),
            period=48.0,
        )
        np.testing.assert_allclose(refined, expected, atol=0.01)
        self.assertEqual(
            score_pair_shift(centers_a, centers_b, refined, period=48.0),
            len(centers_a),
        )

    def test_ncc_and_invalid_inputs(self) -> None:
        array = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.assertAlmostEqual(overlap_ncc(array, array, 0, 0), 1.0, places=5)
        with self.assertRaises(ValueError):
            stitch_template_group([_tile()] * 4, ["a"], 2, 2)
        with self.assertRaises(ValueError):
            stitch_template_group([_tile()] * 3, ["a", "b", "c"], 2, 2)
        with self.assertRaises(ValueError):
            stitch_images([_tile(), _tile()], {0: (0, 0), 1: (1_000_000, 0)})
        with self.assertRaises(ValueError):
            stitch_images([_tile(192, 160), _tile(160, 160)], {0: (0, 0), 1: (1, 0)})

    def test_load_template_images_preserves_order_and_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                path = Path(directory) / f"tile_{index}.png"
                Image.new("RGB", (8, 6), color).save(path)
                paths.append(path)
            images, names = load_template_images(paths)
            self.assertEqual(names, ["tile_0.png", "tile_1.png"])
            self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
            self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
