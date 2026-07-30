from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT / "backend"))

from gradio_image_gesture_overlay import ImageGestureOverlay


class ImageGestureOverlayPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.component = ImageGestureOverlay()

    def test_preprocess_only_keeps_client_intent(self) -> None:
        payload = {
            "server_view": {
                "enabled": True,
                "natural_width": 1920,
                "natural_height": 1080,
                "image_id": "forged-server-id",
                "image_sha256": "forged-server-hash",
                "revision": 99,
                "interaction": "crop",
                "image": "data:image/png;base64,forged",
            },
            "client_intent": {
                "gesture": "drag",
                "start_xy": [10, 20],
                "end_xy": [300.5, 240.25],
                "expected_revision": 7,
                "image_id": "image-1",
                "image_sha256": "hash-1",
                "natural_width": 999,
                "bbox_xyxy": [1, 2, 3, 4],
            },
        }

        sanitized = self.component.preprocess(payload)

        self.assertEqual(
            sanitized,
            {
                "gesture": "drag",
                "start_xy": [10.0, 20.0],
                "end_xy": [300.5, 240.25],
                "expected_revision": 7,
                "image_id": "image-1",
                "image_sha256": "hash-1",
            },
        )

    def test_invalid_values_are_cleared(self) -> None:
        sanitized = self.component.preprocess(
            {
                "client_intent": {
                    "gesture": "forged",
                    "start_xy": [True, 2],
                    "end_xy": [math.inf, 4],
                    "expected_revision": True,
                }
            }
        )

        self.assertEqual(sanitized["gesture"], "")
        self.assertEqual(sanitized["start_xy"], [])
        self.assertEqual(sanitized["end_xy"], [])
        self.assertIsNone(sanitized["expected_revision"])

    def test_flat_legacy_intent_is_sanitized(self) -> None:
        sanitized = self.component.preprocess(
            {
                "gesture": "click",
                "start_xy": [3, 4],
                "end_xy": [3, 4],
                "expected_revision": 2,
                "image_id": "image-2",
                "image_sha256": "hash-2",
                "server_view": {"enabled": True},
            }
        )

        self.assertEqual(sanitized["gesture"], "click")
        self.assertEqual(sanitized["start_xy"], [3.0, 4.0])
        self.assertEqual(sanitized["image_id"], "image-2")
        self.assertNotIn("server_view", sanitized)


if __name__ == "__main__":
    unittest.main()
