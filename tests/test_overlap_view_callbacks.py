from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import overlap_view_callbacks as callbacks
from sam3_demo.overlap_service import OverlapPredictionService
from sam3_demo.overlap_store import OverlapLayoutStore


def _png(array: np.ndarray, mode: str) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _transform(center_x: float) -> dict:
    return {
        "transform_version": 2,
        "center_x": center_x,
        "center_y": 4.0,
        "pivot_x": 1.5,
        "pivot_y": 1.5,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "preview_alpha": 0.35,
    }


class OverlapViewCallbackTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = OverlapLayoutStore(Path(self.tempdir.name) / "view.sqlite3")
        self.service = OverlapPredictionService(self.store)

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def _create_layout(self):
        base = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        return self.store.create_layout(
            "viewer",
            _png(base, "RGB"),
            10,
            8,
            [
                {
                    "name": "ACT",
                    "layer_type": "ACT",
                    "mask_png": _png(mask, "L"),
                    "mask_width": 4,
                    "mask_height": 4,
                    "transform": _transform(4.0),
                },
                {
                    "name": "GE1",
                    "layer_type": "GE1",
                    "mask_png": _png(mask, "L"),
                    "mask_width": 4,
                    "mask_height": 4,
                    "transform": _transform(5.0),
                },
            ],
        )

    def test_empty_store_returns_independent_empty_state(self):
        with mock.patch.object(
            callbacks, "get_overlap_service", return_value=self.service
        ):
            layout_update, layer_update, preview, report = (
                callbacks.refresh_overlap_view(None)
            )
        self.assertEqual(layout_update["choices"], [])
        self.assertEqual(layer_update["choices"], [])
        self.assertIsNone(preview)
        self.assertIn("尚无", report)

    def test_refresh_select_and_render(self):
        snapshot = self._create_layout()
        with mock.patch.object(
            callbacks, "get_overlap_service", return_value=self.service
        ):
            layout_update, layer_update, preview, report = (
                callbacks.refresh_overlap_view(None)
            )
            selected_preview, selected_report = callbacks.render_overlap_layer(
                snapshot["id"], snapshot["layers"][0]["id"]
            )
        self.assertEqual(layout_update["value"], snapshot["id"])
        self.assertEqual(layer_update["value"], snapshot["layers"][0]["id"])
        self.assertEqual(preview.size, (10, 8))
        self.assertEqual(selected_preview.size, (10, 8))
        self.assertIn("下一层：GE1", report)
        self.assertEqual(selected_report, report)


if __name__ == "__main__":
    unittest.main()
