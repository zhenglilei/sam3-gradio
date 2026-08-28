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
        self.assertEqual(sanitized["tiles"][0]["x"], 1.5)
        self.assertEqual(sanitized["tiles"][1]["index"], 1)
        self.assertNotIn("extra", sanitized["tiles"][0])
        self.assertEqual(sanitized["selected"], 1)
        self.assertEqual(sanitized["nudge_step"], 5)
        self.assertTrue(sanitized["diff_mode"])
        self.assertFalse(sanitized["show_loupe"])
        self.assertEqual(sanitized["drag_gain"], 0.25)
        self.assertEqual(sanitized["status"], "aligned")

    def test_preprocess_empty_payload(self) -> None:
        self.assertEqual(self.component.preprocess(None), {})
        self.assertEqual(self.component.preprocess("not-json"), {})


if __name__ == "__main__":
    unittest.main()
