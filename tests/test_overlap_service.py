from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam3_demo.overlap_service import OverlapPredictionService, OverlapServiceError
from sam3_demo.overlap_store import OverlapLayoutStore


def _png(array: np.ndarray, mode: str) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _transform(center_x: float, center_y: float) -> dict:
    return {
        "transform_version": 2,
        "center_x": center_x,
        "center_y": center_y,
        "pivot_x": 1.5,
        "pivot_y": 1.5,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "preview_alpha": 0.35,
    }


class OverlapPredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = OverlapLayoutStore(
            Path(self.tempdir.name) / "overlap.sqlite3"
        )
        base = np.zeros((10, 12, 3), dtype=np.uint8)
        base[:, :, 1] = 40
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        self.snapshot = self.store.create_layout(
            "O3 sample",
            _png(base, "RGB"),
            12,
            10,
            [
                {
                    "name": "ACT main",
                    "layer_type": "ACT",
                    "mask_png": _png(mask, "L"),
                    "mask_width": 4,
                    "mask_height": 4,
                    "transform": _transform(4.5, 4.5),
                },
                {
                    "name": "GE1 contact",
                    "layer_type": "GE1",
                    "mask_png": _png(mask, "L"),
                    "mask_width": 4,
                    "mask_height": 4,
                    "transform": _transform(5.5, 4.5),
                },
                {
                    "name": "GE2 top",
                    "layer_type": "GE2",
                    "mask_png": _png(mask, "L"),
                    "mask_width": 4,
                    "mask_height": 4,
                    "transform": _transform(6.5, 4.5),
                },
            ],
        )
        self.service = OverlapPredictionService(self.store)

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def test_choices_use_stable_ids_and_exclude_terminal_layer(self):
        layout_choices = self.service.layout_choices()
        self.assertEqual(layout_choices[0][1], self.snapshot["id"])
        self.assertIn("O3 sample", layout_choices[0][0])
        layer_choices = self.service.layer_choices(self.snapshot["id"])
        self.assertEqual(
            [value for _, value in layer_choices],
            [layer["id"] for layer in self.snapshot["layers"][:-1]],
        )
        self.assertIn("ACT", layer_choices[0][0])

    def test_render_decodes_masks_and_returns_report(self):
        result = self.service.render(
            self.snapshot["id"], self.snapshot["layers"][0]["id"]
        )
        self.assertEqual(result["preview"].size, (12, 10))
        self.assertEqual(result["current_layer"]["layer_type"], "ACT")
        self.assertEqual(result["next_layer"]["layer_type"], "GE1")
        self.assertEqual(result["metrics"]["intersection_area"], 2)
        self.assertIn("当前层：ACT", result["report"])
        self.assertIn("下一层：GE1", result["report"])
        self.assertNotIn("base_image_png", result["layout"])

    def test_render_propagates_observed_current_pose(self):
        observed = _transform(6.5, 5.5)
        result = self.service.render(
            self.snapshot["id"],
            self.snapshot["layers"][0]["id"],
            observed_current_transform=observed,
        )
        predicted = result["predicted_next_transform"]
        self.assertAlmostEqual(predicted["center_x"], 7.5)
        self.assertAlmostEqual(predicted["center_y"], 5.5)
        self.assertTrue(predicted["predicted_from_current"])

    def test_deleted_and_corrupt_assets_fail_closed(self):
        deleted = self.store.delete_layout(
            self.snapshot["id"], self.snapshot["revision"]
        )
        with self.assertRaisesRegex(OverlapServiceError, "不存在"):
            self.service.render(deleted["id"], deleted["layers"][0]["id"])

        replacement = self.store.create_layout(
            "bad size",
            self.snapshot["base_image_png"],
            11,
            10,
            self.snapshot["layers"],
        )
        with self.assertRaisesRegex(OverlapServiceError, "does not match"):
            self.service.render(
                replacement["id"], replacement["layers"][0]["id"]
            )


if __name__ == "__main__":
    unittest.main()
