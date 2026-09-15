import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam3_demo import ui_repair_core as repair


def mask_data_url(array):
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeLama:
    def inpaint(self, image, mask):
        result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        result[np.asarray(mask) > 0] = (1, 2, 3)
        return Image.fromarray(result, mode="RGB")


class RepairCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runtime"
        self.session = {"session_id": "a" * 32}
        self.state = repair.owned_repair_state(None, self.session)
        self.paths = []
        for index, color in enumerate(("red", "yellow")):
            path = Path(self.temp.name) / f"tile_{index}.png"
            image = Image.new("RGB", (37, 29), "gray")
            draw = ImageDraw.Draw(image)
            draw.rectangle((5, 7, 13, 16), fill=color)
            image.save(path)
            self.paths.append(str(path))

    def test_upload_is_session_owned_and_persisted(self):
        added = repair.add_uploaded_images(self.state, self.paths, self.root)
        self.assertEqual(len(added), 2)
        self.assertEqual(self.state["active_id"], added[0]["id"])
        self.assertEqual(self.state["selected_ids"], [item["id"] for item in added])
        self.assertTrue(Path(added[0]["source_path"]).is_file())
        self.assertTrue(Path(added[0]["mask_path"]).is_file())
        reset = repair.owned_repair_state(self.state, {"session_id": "b" * 32})
        self.assertEqual(reset["items"], [])
        self.assertEqual(len(self.state["items"]), 2)

    def test_color_detection_adds_rectangles_and_invalidates_result(self):
        items = repair.add_uploaded_images(self.state, self.paths, self.root)
        items[0]["result_path"] = items[0]["source_path"]
        affected, regions = repair.apply_color_detection(
            self.state,
            [item["id"] for item in items],
            ["red", "yellow"],
            saturation=80,
            value=80,
            min_area=2,
            padding=1,
            merge_distance=0,
        )
        self.assertEqual((affected, regions), (2, 2))
        self.assertIsNone(items[0]["result_path"])
        self.assertEqual(items[0]["status"], "已标记")
        with Image.open(items[0]["mask_path"]) as opened:
            self.assertGreater(np.asarray(opened).sum(), 0)

    def test_editor_commit_rejects_old_image_and_keeps_exact_size(self):
        items = repair.add_uploaded_images(self.state, self.paths, self.root)
        payload = repair.editor_payload(items[0])
        edited = np.zeros((29, 37), dtype=np.uint8)
        edited[3:8, 4:10] = 255
        payload["mask_png"] = mask_data_url(edited)
        payload["revision"] = 1
        repair.commit_editor_value(self.state, payload)
        self.assertEqual(items[0]["revision"], 1)
        with Image.open(items[0]["mask_path"]) as opened:
            np.testing.assert_array_equal(np.asarray(opened), edited)
        self.state["active_id"] = items[1]["id"]
        with self.assertRaisesRegex(ValueError, "图片已切换"):
            repair.commit_editor_value(self.state, payload)

    def test_clear_and_repair_selected(self):
        items = repair.add_uploaded_images(self.state, self.paths[:1], self.root)
        repair.apply_color_detection(
            self.state,
            [items[0]["id"]],
            ["red"],
            min_area=2,
            merge_distance=0,
        )
        completed, failures = repair.repair_items(
            self.state,
            [items[0]["id"]],
            FakeLama(),
        )
        self.assertEqual((completed, failures), (1, []))
        self.assertEqual(items[0]["status"], "已修复")
        self.assertEqual(repair.repaired_paths(self.state, [items[0]["id"]]), [items[0]["result_path"]])
        repair.clear_active_mask(self.state)
        self.assertEqual(items[0]["status"], "待标记")
        self.assertIsNone(items[0]["result_path"])

    def test_queue_navigation_stays_on_stable_ids(self):
        items = repair.add_uploaded_images(self.state, self.paths, self.root)
        gallery, choices = repair.queue_view(self.state)
        self.assertEqual(len(gallery), 2)
        self.assertEqual([value for _label, value in choices], [item["id"] for item in items])
        self.assertEqual(repair.move_active(self.state, 1)["id"], items[1]["id"])
        self.assertEqual(repair.move_active(self.state, 99)["id"], items[1]["id"])


if __name__ == "__main__":
    unittest.main()
