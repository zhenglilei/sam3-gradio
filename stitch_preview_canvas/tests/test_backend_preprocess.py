from __future__ import annotations

import sys
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT / "backend"))

from gradio_stitch_preview_canvas import StitchPreviewCanvas


class StitchPreviewCanvasPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.component = StitchPreviewCanvas()

    def test_preprocess_keeps_tile_geometry_and_settings(self) -> None:
        payload = {
            "tiles": [
                {
                    "index": 0,
                    "image": "data:image/jpeg;base64,forged",
                    "x": 1.5,
                    "y": 2.5,
                    "width": 100,
                    "height": 80,
                    "extra": "drop-me",
                },
                {
                    "index": 1,
                    "image": "data:image/png;base64,forged-2",
                    "x": 10,
                    "y": 20,
                    "width": 50,
                    "height": 40,
                },
                {"index": "bad"},
            ],
            "selected": 1,
            "nudge_step": 5,
            "diff_mode": True,
            "show_loupe": False,
            "drag_gain": 0.25,
            "status": "aligned",
            "forged": "drop-me",
        }

        sanitized = self.component.preprocess(payload)

        self.assertEqual(
            set(sanitized),
            {
                "tiles",
                "selected",
                "nudge_step",
                "diff_mode",
                "show_loupe",
                "drag_gain",
                "status",
            },
        )
        self.assertEqual(len(sanitized["tiles"]), 2)
        self.assertNotIn("image", sanitized["tiles"][0])
        self.assertEqual(sanitized["tiles"][0]["x"], 1)
        self.assertEqual(sanitized["tiles"][0]["y"], 2)
        self.assertEqual(sanitized["tiles"][1]["index"], 1)
        self.assertEqual(sanitized["tiles"][1]["width"], 50)
        self.assertEqual(sanitized["tiles"][1]["height"], 40)
        self.assertNotIn("extra", sanitized["tiles"][0])
        self.assertEqual(sanitized["selected"], 1)
        self.assertEqual(sanitized["nudge_step"], 5)
        self.assertTrue(sanitized["diff_mode"])
        self.assertFalse(sanitized["show_loupe"])
        self.assertEqual(sanitized["drag_gain"], 0.25)
        self.assertEqual(sanitized["status"], "aligned")

    def test_preprocess_drops_invalid_geometry_and_duplicate_indices(self) -> None:
        payload = {
            "tiles": [
                {
                    "index": 0,
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 80,
                    "image": "data:image/png;base64,forged",
                },
                {
                    "index": 0,
                    "x": 10,
                    "y": 20,
                    "width": 50,
                    "height": 40,
                },
                {
                    "index": 1,
                    "x": float("nan"),
                    "y": 0,
                    "width": 20,
                    "height": 20,
                },
                {
                    "index": 2,
                    "x": 0,
                    "y": float("inf"),
                    "width": 20,
                    "height": 20,
                },
                {
                    "index": 3,
                    "x": 0,
                    "y": 0,
                    "width": -1,
                    "height": 20,
                },
                {
                    "index": 4,
                    "x": 0,
                    "y": 0,
                    "width": 20,
                    "height": 0,
                },
                {
                    "index": 5,
                    "x": 10_000_001,
                    "y": 0,
                    "width": 20,
                    "height": 20,
                },
                {
                    "index": 6,
                    "x": 0,
                    "y": 0,
                    "width": 20,
                    "height": 20,
                    "image": "data:image/jpeg;base64,forged-2",
                    "extra": "drop-me",
                },
            ],
            "forged": "drop-me",
        }

        sanitized = self.component.preprocess(payload)

        self.assertEqual([tile["index"] for tile in sanitized["tiles"]], [0, 6])
        self.assertEqual(sanitized["tiles"][1]["x"], 0)
        self.assertNotIn("image", sanitized["tiles"][1])
        self.assertNotIn("extra", sanitized["tiles"][1])

    def test_preprocess_sanitizes_control_fields(self) -> None:
        payload = {
            "tiles": [
                {"index": 0, "x": 0, "y": 0, "width": 10, "height": 10},
            ],
            "selected": float("inf"),
            "nudge_step": float("nan"),
            "drag_gain": float("inf"),
            "diff_mode": 1,
            "show_loupe": "false",
            "status": "x" * 1025,
        }

        sanitized = self.component.preprocess(payload)

        self.assertEqual(sanitized["tiles"][0]["index"], 0)
        self.assertNotIn("selected", sanitized)
        self.assertNotIn("nudge_step", sanitized)
        self.assertNotIn("drag_gain", sanitized)
        self.assertNotIn("diff_mode", sanitized)
        self.assertNotIn("show_loupe", sanitized)
        self.assertNotIn("status", sanitized)

    def test_preprocess_empty_payload(self) -> None:
        self.assertEqual(self.component.preprocess(None), {})
        self.assertEqual(self.component.preprocess("not-json"), {})


if __name__ == "__main__":
    unittest.main()
