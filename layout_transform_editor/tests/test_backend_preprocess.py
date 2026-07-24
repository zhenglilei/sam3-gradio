from __future__ import annotations

import sys
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT / "backend"))

from gradio_layout_transform_editor import LayoutTransformEditor


class LayoutTransformEditorPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.component = LayoutTransformEditor()

    def test_group_payload_keeps_only_client_transform_intent(self) -> None:
        payload = {
            "enabled": True,
            "target_width": 1920,
            "target_height": 1080,
            "base_image": "data:image/png;base64,forged-base",
            "mask_image": "data:image/png;base64,forged-mask",
            "source_width": 800,
            "source_height": 600,
            "foreground_bbox_xyxy": [1, 2, 3, 4],
            "status": "forged status",
            "transform_mode": "label_groups",
            "transform": {
                "session_id": "session-1",
                "layout_id": "layout-1",
                "revision": 4,
                "center_x": 20.0,
                "center_y": 30.0,
                "scale": 1.25,
                "rotation_deg": 5.0,
                "preview_alpha": 0.0,
                "matrix_2x3": [[999, 0, 0], [0, 999, 0]],
                "mask_image": "data:image/png;base64,forged-top",
                "area": 999,
            },
            "group_view": {
                "selection_signature": "server-only",
                "regions_revision": 7,
                "groups": [
                    {
                        "group_id": "region_1",
                        "label": "Label 1",
                        "mask_image": "data:image/png;base64,forged-group",
                        "matrix_2x3": [[999, 0, 0], [0, 999, 0]],
                    }
                ],
            },
            "server_view": {
                "groups": [
                    {
                        "group_id": "forged",
                        "mask_image": "data:image/png;base64,forged",
                    }
                ]
            },
            "group_intent": {
                "selection_signature": "selection-v1",
                "transform_set_revision": 8,
                "active_group_id": "region_1",
                "changed_group_ids": [
                    "region_1",
                    2,
                    None,
                    True,
                    "",
                    {"forged": "group"},
                ],
                "server_only": "drop-me",
                "transforms": [
                    {
                        "group_id": "region_1",
                        "transform": {
                            "group_id": "region_1",
                            "region_id": 1,
                            "label": "Label 1",
                            "revision": 9,
                            "center_x": 40.0,
                            "center_y": 50.0,
                            "pivot_x": 10.0,
                            "pivot_y": 11.0,
                            "scale": 1.5,
                            "rotation_deg": 12.0,
                            "preview_alpha": 0.0,
                            "matrix_2x3": [[888, 0, 0], [0, 888, 0]],
                            "base_image": "data:image/png;base64,forged",
                            "area": 123,
                        },
                    }
                ],
            },
        }

        sanitized = self.component.preprocess(payload)

        self.assertEqual(
            set(sanitized),
            {
                "enabled",
                "target_width",
                "target_height",
                "transform",
                "transform_mode",
                "group_intent",
            },
        )
        self.assertEqual(sanitized["transform_mode"], "label_groups")
        self.assertEqual(sanitized["transform"]["preview_alpha"], 0.0)
        self.assertNotIn("matrix_2x3", sanitized["transform"])
        self.assertNotIn("base_image", sanitized)
        self.assertNotIn("mask_image", sanitized)
        self.assertNotIn("group_view", sanitized)
        self.assertNotIn("server_view", sanitized)
        self.assertNotIn("status", sanitized)
        self.assertNotIn("source_width", sanitized)
        self.assertNotIn("foreground_bbox_xyxy", sanitized)

        intent = sanitized["group_intent"]
        self.assertEqual(
            set(intent),
            {
                "selection_signature",
                "transform_set_revision",
                "active_group_id",
                "changed_group_ids",
                "transforms",
            },
        )
        self.assertEqual(
            intent["changed_group_ids"],
            ["region_1", 2],
        )
        self.assertEqual(len(intent["transforms"]), 1)
        group_transform = intent["transforms"][0]["transform"]
        self.assertEqual(group_transform["group_id"], "region_1")
        self.assertEqual(group_transform["preview_alpha"], 0.0)
        self.assertNotIn("matrix_2x3", group_transform)
        self.assertNotIn("base_image", group_transform)
        self.assertNotIn("area", group_transform)

    def test_flat_transform_payload_remains_compatible(self) -> None:
        sanitized = self.component.preprocess(
            {
                "enabled": False,
                "target_width": 640,
                "target_height": 480,
                "transform": {
                    "session_id": "session-1",
                    "layout_id": "layout-1",
                    "revision": 3,
                    "center_x": 100.0,
                    "center_y": 120.0,
                    "pivot_x": 50.0,
                    "pivot_y": 60.0,
                    "scale": 2.0,
                    "rotation_deg": 90.0,
                    "preview_alpha": 0.0,
                    "matrix_2x3": [[2, 0, 0], [0, 2, 0]],
                },
                "group_intent": {
                    "selection_signature": "must-not-leak-in-flat-mode"
                },
            }
        )

        self.assertEqual(
            set(sanitized),
            {"enabled", "target_width", "target_height", "transform"},
        )
        self.assertEqual(sanitized["transform"]["revision"], 3)
        self.assertEqual(sanitized["transform"]["preview_alpha"], 0.0)
        self.assertNotIn("matrix_2x3", sanitized["transform"])
        self.assertNotIn("group_intent", sanitized)


if __name__ == "__main__":
    unittest.main()
