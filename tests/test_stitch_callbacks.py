from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_demo import stitch_callbacks as callbacks


class StitchCallbacksTest(unittest.TestCase):
    def _loaded_state(self):
        state = callbacks.new_stitch_state("a" * 32, "owner-token")
        state.update(
            {
                "images": [
                    Image.new("RGB", (12, 8), (30, 60, 90)),
                    Image.new("RGB", (12, 8), (90, 60, 30)),
                ],
                "shifts": [(0, 0), (10, 0)],
                "revision": 3,
            }
        )
        return state

    def test_load_preserves_server_owned_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.png"
            Image.new("RGB", (10, 6), (10, 20, 30)).save(path)
            previous = callbacks.new_stitch_state("b" * 32, "secret")
            result = callbacks.load_tiles(
                [str(path)],
                "horizontal",
                previous,
                1,
                False,
                True,
                True,
                False,
            )
        state = result[0]
        self.assertEqual(state["session_id"], "b" * 32)
        self.assertEqual(state["owner_token"], "secret")
        self.assertEqual(len(state["images"]), 1)
        self.assertIsNone(state["mosaic"])
        self.assertFalse(result[7]["interactive"])
        self.assertEqual(len(result), 10)
        self.assertEqual(result[-1], 0.0)

    def test_load_trims_black_border_before_default_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                pixels = Image.new("RGB", (20, 14), (0, 0, 0))
                content = Image.new("RGB", (14, 10), (80 + index * 20, 130, 180))
                content.putpixel((4, 4), (240, 220, 180))
                pixels.paste(content, (3, 2))
                path = Path(tmp) / f"framed_{index}.png"
                pixels.save(path)
                paths.append(str(path))

            result = callbacks.load_tiles(
                paths,
                "horizontal",
                callbacks.new_stitch_state("d" * 32, "owner"),
                1,
                False,
                True,
                True,
                False,
                True,
            )

        state = result[0]
        self.assertEqual([image.size for image in state["images"]], [(14, 10), (14, 10)])
        self.assertEqual(state["shifts"], [(0, 0), (14, 0)])
        self.assertTrue(all(record["applied"] for record in state["black_border_records"]))
        self.assertIn("自动去黑边 2/2 张", result[4])

    def test_load_can_disable_black_border_trimming(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Image.new("RGB", (20, 14), (0, 0, 0))
            image.paste(Image.new("RGB", (14, 10), (120, 170, 210)), (3, 2))
            path = Path(tmp) / "framed.png"
            image.save(path)
            result = callbacks.load_tiles(
                [str(path)],
                "horizontal",
                callbacks.new_stitch_state("e" * 32, "owner"),
                1,
                False,
                True,
                True,
                False,
                False,
            )

        state = result[0]
        self.assertEqual(state["images"][0].size, (20, 14))
        self.assertFalse(state["remove_black_border"])
        self.assertEqual(state["black_border_records"], [])

    def test_invalid_grid_load_is_recoverable_without_callback_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f"tile_{index}.png"
                Image.new("RGB", (10, 6), (10 + index, 20, 30)).save(path)
                paths.append(str(path))
            result = callbacks.load_tiles(
                paths,
                "grid_2x2",
                callbacks.new_stitch_state("c" * 32, "owner"),
                1,
                False,
                True,
                True,
                False,
            )

        state = result[0]
        self.assertEqual(len(state["images"]), 2)
        self.assertEqual(state["shifts"], [])
        self.assertIn("需要每组恰好 4 张", result[4])
        recovered = callbacks.apply_layout("horizontal", state)
        self.assertEqual(recovered[0]["shifts"], [(0, 0), (10, 0)])
        self.assertEqual(len(recovered[2]["tiles"]), 2)

    def test_geometry_change_invalidates_generated_result(self):
        state = self._loaded_state()
        generated = callbacks.generate_mosaic(
            state,
            True,
            False,
            publish_mosaic=lambda _image, _session: "/tmp/stitch.png",
        )
        state = generated[0]
        self.assertTrue(generated[4]["interactive"])
        self.assertTrue(generated[6]["server_view"]["enabled"])
        self.assertIsNotNone(callbacks.mosaic_for_handoff(state))

        changed = callbacks.apply_numeric_shift(9, 1, state)
        state = changed[0]
        self.assertIsNone(state["mosaic"])
        self.assertIsNone(state["generated_revision"])
        self.assertFalse(changed[7]["interactive"])
        with self.assertRaisesRegex(ValueError, "过期|先生成"):
            callbacks.mosaic_for_handoff(state)

    def test_selection_only_canvas_change_keeps_result(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, False, False)[0]
        before_revision = state["revision"]
        payload = {
            "selected": 1,
            "tiles": [
                {"index": 0, "x": 0, "y": 0, "width": 12, "height": 8},
                {"index": 1, "x": 10, "y": 0, "width": 12, "height": 8},
            ],
        }
        result = callbacks.canvas_changed(payload, state)
        self.assertIsNotNone(result[0]["mosaic"])
        self.assertEqual(result[0]["revision"], before_revision)

    def test_canvas_rotation_invalidates_result_and_syncs_numeric_angle(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, False, False)[0]
        payload = {
            "selected": 1,
            "tiles": [
                {
                    "index": 0,
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 8,
                    "rotation_deg": 0,
                },
                {
                    "index": 1,
                    "x": 10,
                    "y": 0,
                    "width": 12,
                    "height": 8,
                    "rotation_deg": 2.75,
                },
            ],
        }
        result = callbacks.canvas_changed(payload, state)
        self.assertEqual(result[0]["rotations"], [0.0, 2.75])
        self.assertIsNone(result[0]["mosaic"])
        self.assertEqual(result[-1], 2.75)
        self.assertIn("旋转=2.75°", result[3])

    def test_numeric_transform_updates_selected_angle(self):
        state = self._loaded_state()
        result = callbacks.apply_numeric_transform(9, 1, -181, state)
        self.assertEqual(result[0]["shifts"][0], (9, 1))
        self.assertEqual(result[0]["rotations"], [179.0, 0.0])
        self.assertEqual(result[-1], 179.0)
        self.assertEqual(result[1]["tiles"][0]["rotation_deg"], 179.0)
        self.assertEqual(len(result), 10)

    def test_auto_align_resets_manual_rotations(self):
        state = self._loaded_state()
        state["rotations"] = [3.0, -2.0]
        with mock.patch.object(
            callbacks,
            "auto_align_images",
            return_value=([(0, 0), (10, 0)], ["aligned"]),
        ):
            result = callbacks.auto_align(state, "horizontal", 1, False, True)
        self.assertEqual(result[0]["rotations"], [0.0, 0.0])
        self.assertEqual(result[-1], 0.0)

    def test_export_option_change_invalidates_result(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, True, False)[0]
        result = callbacks.apply_export_options(False, False, state)
        self.assertIsNone(result[0]["mosaic"])
        self.assertFalse(result[4]["interactive"])
        self.assertIn("重新生成", result[1])

    def test_layout_change_resets_geometry_and_invalidates_result(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, True, False)[0]
        result = callbacks.apply_layout("vertical", state)
        next_state = result[0]
        self.assertEqual(next_state["layout"], "vertical")
        self.assertEqual(next_state["shifts"], [(0, 0), (0, 8)])
        self.assertIsNone(next_state["mosaic"])
        self.assertEqual(result[1]["value"], "vertical")
        self.assertFalse(result[8]["interactive"])
        self.assertIn("重新自动对齐", result[5])

    def test_invalid_grid_layout_invalidates_old_result(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, True, False)[0]
        result = callbacks.apply_layout("grid_2x2", state)
        self.assertEqual(result[0]["layout"], "horizontal")
        self.assertEqual(result[1]["value"], "horizontal")
        self.assertIsNone(result[0]["mosaic"])
        self.assertFalse(result[8]["interactive"])
        self.assertIn("需要每组恰好 4 张", result[5])

    def test_failed_auto_align_does_not_commit_requested_layout(self):
        state = self._loaded_state()
        state = callbacks.generate_mosaic(state, True, False)[0]
        with mock.patch.object(
            callbacks,
            "auto_align_images",
            side_effect=ValueError("synthetic failure"),
        ):
            result = callbacks.auto_align(state, "vertical", 1, False, True)
        self.assertEqual(result[0]["layout"], "horizontal")
        self.assertIsNone(result[0]["mosaic"])
        self.assertFalse(result[7]["interactive"])

    def test_handoff_status_reflects_source_upload_result(self):
        self.assertIn("已写入", callbacks.handoff_status("已载入完整原图 20x20；当前使用整图"))
        self.assertIn("失败", callbacks.handoff_status("完整原图加载失败: bad image"))

    def test_manual_rectangle_crop_and_restore_full_mosaic(self):
        state = callbacks.generate_mosaic(
            self._loaded_state(),
            False,
            False,
            publish_mosaic=lambda _image, _session: "/tmp/stitch.png",
        )[0]
        full_size = state["mosaic"].size
        view = callbacks.mosaic_crop_payload(state)["server_view"]
        intent = {
            "gesture": "drag",
            "start_xy": [2, 1],
            "end_xy": [9, 6],
            "expected_revision": view["revision"],
            "image_id": view["image_id"],
            "image_sha256": view["image_sha256"],
        }
        cropped = callbacks.crop_mosaic_preview(
            intent,
            state,
            publish_mosaic=lambda _image, _session: "/tmp/cropped.png",
        )
        state = cropped[0]
        self.assertEqual(state["mosaic"].size, (7, 5))
        self.assertEqual(state["mosaic_full"].size, full_size)
        self.assertEqual(state["mosaic_crop_bbox_xyxy"], [2, 1, 9, 6])
        self.assertTrue(cropped[4]["interactive"])
        self.assertTrue(cropped[7]["interactive"])

        restored = callbacks.restore_full_mosaic(
            state,
            publish_mosaic=lambda _image, _session: "/tmp/full.png",
        )
        self.assertEqual(restored[0]["mosaic"].size, full_size)
        self.assertIsNone(restored[0]["mosaic_crop_bbox_xyxy"])
        self.assertFalse(restored[7]["interactive"])

    def test_manual_crop_rejects_stale_identity_without_changing_result(self):
        state = callbacks.generate_mosaic(self._loaded_state(), False, False)[0]
        before = state["mosaic"].tobytes()
        view = callbacks.mosaic_crop_payload(state)["server_view"]
        intent = {
            "gesture": "drag",
            "start_xy": [0, 0],
            "end_xy": [8, 6],
            "expected_revision": view["revision"],
            "image_id": view["image_id"],
            "image_sha256": "stale",
        }
        result = callbacks.crop_mosaic_preview(intent, state)
        self.assertEqual(result[0]["mosaic"].tobytes(), before)
        self.assertIn("hash 已过期", result[3])

    def test_stitch_ui_is_standalone_and_handoff_cleanup_is_success_only(self):
        root = Path(__file__).resolve().parents[1]
        tab_source = (root / "sam3_demo" / "ui" / "stitch_tab.py").read_text(
            encoding="utf-8"
        )
        binding_source = (root / "sam3_demo" / "ui" / "bindings.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("LayoutTransformEditor", tab_source)
        self.assertNotIn("版图对齐", tab_source)
        self.assertIn('with gr.TabItem("周期拼接"', tab_source)
        self.assertIn('elem_id="stitch_mosaic_preview"', tab_source)
        self.assertIn("visible=True", tab_source)
        self.assertIn("stitch_mosaic_crop_overlay", tab_source)
        for tab_id in ("stitch_images", "stitch_alignment", "stitch_export"):
            self.assertIn(f'id="{tab_id}"', tab_source)
        self.assertIn('elem_classes="stitch-sidebar"', tab_source)
        self.assertIn('elem_classes="stitch-control-tabs"', tab_source)
        self.assertNotIn("_load_stitch_layout_mask", binding_source)
        self.assertIn("handoff_event.success(", binding_source)
        self.assertIn("outputs=[source_image_upload]", binding_source)
        self.assertNotIn("handoff_event.then(", binding_source)

    def test_stitch_sidebar_uses_tabs_and_two_column_fields(self):
        root = Path(__file__).resolve().parents[1]
        style_source = (root / "sam3_demo" / "ui" / "styles.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(".stitch-control-tabs", style_source)
        self.assertIn(".stitch-crop-fields > .form", style_source)
        self.assertIn("repeat(2, minmax(0, 1fr))", style_source)
        self.assertIn(".stitch-crop-fields label.block", style_source)


if __name__ == "__main__":
    unittest.main()
