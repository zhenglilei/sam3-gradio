from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from layout_transform_utils import build_layout_affine_matrix
from sam3_demo.overlap_geometry import (
    OverlapGeometryError,
    calculate_next_layer_overlap,
    current_and_next_layer,
    default_layer_ordinal,
    overlap_metrics,
    predict_next_transform,
    render_overlap_preview,
)


def _transform(
    *,
    center_x: float,
    center_y: float,
    pivot_x: float = 1.5,
    pivot_y: float = 1.5,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
) -> dict:
    return {
        "transform_version": 2,
        "center_x": center_x,
        "center_y": center_y,
        "pivot_x": pivot_x,
        "pivot_y": pivot_y,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "preview_alpha": 0.35,
    }


class OverlapGeometryTests(unittest.TestCase):
    def test_default_layer_order(self):
        self.assertEqual(
            [default_layer_ordinal(x) for x in ("BSM", "ACT", "GE1", "GE2")],
            [0, 1, 2, 3],
        )
        with self.assertRaises(OverlapGeometryError):
            default_layer_ordinal("metal")

    def test_current_and_next_respects_authored_order(self):
        layers = [
            {"id": "ge1", "layer_type": "GE1", "ordinal": 1},
            {"id": "act", "layer_type": "ACT", "ordinal": 0},
        ]
        current, following = current_and_next_layer(layers, "act")
        self.assertEqual(current["id"], "act")
        self.assertEqual(following["id"], "ge1")
        with self.assertRaisesRegex(OverlapGeometryError, "no next layer"):
            current_and_next_layer(layers, "ge1")

    def test_predict_next_transform_propagates_similarity_delta(self):
        reference_current = _transform(center_x=10, center_y=12, pivot_x=2, pivot_y=3)
        reference_next = _transform(
            center_x=18,
            center_y=15,
            pivot_x=4,
            pivot_y=5,
            scale=0.8,
            rotation_deg=-7,
        )
        observed_current = _transform(
            center_x=22,
            center_y=30,
            pivot_x=2,
            pivot_y=3,
            scale=1.25,
            rotation_deg=11,
        )
        predicted = predict_next_transform(
            reference_current, reference_next, observed_current
        )
        ref_current_matrix = np.vstack(
            [build_layout_affine_matrix(reference_current), [0, 0, 1]]
        )
        ref_next_matrix = np.vstack(
            [build_layout_affine_matrix(reference_next), [0, 0, 1]]
        )
        observed_matrix = np.vstack(
            [build_layout_affine_matrix(observed_current), [0, 0, 1]]
        )
        expected = observed_matrix @ np.linalg.inv(ref_current_matrix) @ ref_next_matrix
        actual = build_layout_affine_matrix(predicted)
        np.testing.assert_allclose(actual, expected[:2], atol=1e-5)

    def test_calculate_overlap_warps_and_intersects(self):
        current_mask = np.zeros((4, 4), dtype=np.uint8)
        current_mask[1:3, 1:3] = 1
        next_mask = current_mask.copy()
        layers = [
            {
                "id": "act",
                "layer_type": "ACT",
                "ordinal": 0,
                "mask": current_mask,
                "transform": _transform(center_x=4.5, center_y=4.5),
            },
            {
                "id": "ge1",
                "layer_type": "GE1",
                "ordinal": 1,
                "mask": next_mask,
                "transform": _transform(center_x=5.5, center_y=4.5),
            },
        ]
        result = calculate_next_layer_overlap(layers, "act", (10, 10))
        self.assertEqual(result["metrics"]["current_area"], 4)
        self.assertEqual(result["metrics"]["next_area"], 4)
        self.assertEqual(result["metrics"]["intersection_area"], 2)
        self.assertAlmostEqual(result["metrics"]["iou"], 1 / 3)
        self.assertEqual(result["metrics"]["intersection_bbox_xyxy"], [5, 4, 6, 6])

    def test_observed_current_moves_predicted_next_together(self):
        mask = np.ones((3, 3), dtype=np.uint8)
        layers = [
            {
                "id": "bsm",
                "layer_type": "BSM",
                "ordinal": 0,
                "mask": mask,
                "transform": _transform(center_x=4, center_y=4),
            },
            {
                "id": "act",
                "layer_type": "ACT",
                "ordinal": 1,
                "mask": mask,
                "transform": _transform(center_x=7, center_y=4),
            },
        ]
        observed = _transform(center_x=6, center_y=5)
        result = calculate_next_layer_overlap(
            layers,
            "bsm",
            (14, 10),
            observed_current_transform=observed,
        )
        self.assertAlmostEqual(result["predicted_next_transform"]["center_x"], 9)
        self.assertAlmostEqual(result["predicted_next_transform"]["center_y"], 5)
        self.assertTrue(result["predicted_next_transform"]["predicted_from_current"])

    def test_overlap_metrics_empty_intersection(self):
        left = np.zeros((4, 5), dtype=bool)
        right = np.zeros((4, 5), dtype=bool)
        left[:, :2] = True
        right[:, 3:] = True
        metrics = overlap_metrics(left, right)
        self.assertEqual(metrics["intersection_area"], 0)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertIsNone(metrics["intersection_bbox_xyxy"])

    def test_render_preview_marks_intersection(self):
        base = Image.new("RGB", (20, 20), "black")
        current = np.zeros((20, 20), dtype=bool)
        following = np.zeros((20, 20), dtype=bool)
        current[1:15, 1:15] = True
        following[6:19, 6:19] = True
        preview = np.asarray(render_overlap_preview(base, current, following))
        self.assertEqual(preview.shape, (20, 20, 3))
        self.assertGreater(int(preview[10, 10, 0]), int(preview[3, 3, 0]))

    def test_rejects_non_contiguous_order_and_blank_masks(self):
        layers = [
            {"id": "a", "layer_type": "ACT", "ordinal": 0},
            {"id": "b", "layer_type": "GE1", "ordinal": 2},
        ]
        with self.assertRaisesRegex(OverlapGeometryError, "contiguous"):
            current_and_next_layer(layers, "a")
        layers[1]["ordinal"] = 1
        layers[0].update(
            mask=np.zeros((3, 3), dtype=np.uint8),
            transform=_transform(center_x=1, center_y=1),
        )
        layers[1].update(
            mask=np.ones((3, 3), dtype=np.uint8),
            transform=_transform(center_x=1, center_y=1),
        )
        with self.assertRaisesRegex(OverlapGeometryError, "no foreground"):
            calculate_next_layer_overlap(layers, "a", (5, 5))


if __name__ == "__main__":
    unittest.main()
