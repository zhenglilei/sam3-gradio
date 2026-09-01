from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_demo import template_stitch_callbacks as callbacks


class TemplateStitchCallbacksTests(unittest.TestCase):
    def test_success_preserves_owner_and_publishes_under_session(self):
        previous = callbacks.new_template_stitch_state("a" * 32, "owner")
        images = [Image.new("RGB", (12, 8), (index, 20, 30)) for index in range(4)]
        meta = {
            "period": 24.0,
            "layout_method": "SIFT",
            "positions": [[0, 0], [8, 0], [0, 6], [8, 6]],
            "order": [0, 1, 2, 3],
            "hole_counts": [4, 4, 4, 4],
        }
        published = []
        with mock.patch.object(
            callbacks,
            "load_template_images",
            return_value=(images, [f"tile_{index}.png" for index in range(4)]),
        ), mock.patch.object(
            callbacks,
            "stitch_template_group",
            return_value=(Image.new("RGB", (20, 14), "white"), meta),
        ):
            result = callbacks.run_template_stitch(
                ["unused"] * 4,
                2,
                2,
                previous,
                publish_mosaic=lambda image, session: published.append((image.size, session)) or "/tmp/template.png",
            )
        state = result[0]
        self.assertEqual(state["session_id"], "a" * 32)
        self.assertEqual(state["owner_token"], "owner")
        self.assertEqual(state["mosaic"].size, (20, 14))
        self.assertEqual(state["generated_revision"], state["revision"])
        self.assertEqual(published, [((20, 14), "a" * 32)])
        self.assertTrue(result[5]["interactive"])
        self.assertEqual(callbacks.mosaic_for_handoff(state).size, (20, 14))

    def test_publish_failure_keeps_valid_mosaic_and_handoff(self):
        previous = callbacks.new_template_stitch_state("d" * 32, "owner")
        images = [Image.new("RGB", (12, 8), "white") for _ in range(4)]
        meta = {"period": 24.0, "layout_method": "NCC"}
        with mock.patch.object(
            callbacks,
            "load_template_images",
            return_value=(images, ["a", "b", "c", "d"]),
        ), mock.patch.object(
            callbacks,
            "stitch_template_group",
            return_value=(Image.new("RGB", (20, 14), "white"), meta),
        ):
            result = callbacks.run_template_stitch(
                ["unused"] * 4,
                2,
                2,
                previous,
                publish_mosaic=mock.Mock(side_effect=OSError("disk unavailable")),
            )
        self.assertEqual(result[0]["mosaic"].size, (20, 14))
        self.assertEqual(callbacks.mosaic_for_handoff(result[0]).size, (20, 14))
        self.assertIn("下载文件发布失败", result[4])
        self.assertTrue(result[5]["interactive"])

    def test_invalid_grid_clears_old_result_without_publishing(self):
        previous = callbacks.new_template_stitch_state("b" * 32, "owner")
        previous.update(
            {
                "revision": 2,
                "generated_revision": 2,
                "mosaic": Image.new("RGB", (4, 4), "white"),
            }
        )
        publisher = mock.Mock()
        with mock.patch.object(
            callbacks,
            "load_template_images",
            return_value=([Image.new("RGB", (8, 8), "white")] * 3, ["a", "b", "c"]),
        ):
            result = callbacks.run_template_stitch(
                ["unused"] * 3,
                2,
                2,
                previous,
                publish_mosaic=publisher,
            )
        self.assertIsNone(result[0]["mosaic"])
        self.assertIsNone(result[0]["generated_revision"])
        self.assertIn("需要 4 张", result[4])
        self.assertFalse(result[5]["interactive"])
        publisher.assert_not_called()

    def test_handoff_rejects_stale_revision(self):
        state = callbacks.new_template_stitch_state("c" * 32, "owner")
        state.update(
            {
                "revision": 4,
                "generated_revision": 3,
                "mosaic": Image.new("RGB", (4, 4), "white"),
            }
        )
        with self.assertRaisesRegex(ValueError, "先生成"):
            callbacks.mosaic_for_handoff(state)

    def test_ui_and_binding_contract(self):
        root = Path(__file__).resolve().parents[1]
        tab = (root / "sam3_demo" / "ui" / "template_stitch_tab.py").read_text(encoding="utf-8")
        bindings = (root / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn('with gr.TabItem("模板拼接"', tab)
        self.assertIn("template_stitch_run_btn.click", bindings)
        self.assertIn("template_handoff_event.success", bindings)
        self.assertIn("outputs=[source_image_upload]", bindings)


if __name__ == "__main__":
    unittest.main()
