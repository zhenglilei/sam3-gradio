import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
from sam3_demo import app, batch_workspace as batch
from sam3_demo.stitch_callbacks import new_stitch_state


class BatchWorkspaceTests(unittest.TestCase):
    def setUp(self):
        # These shared fixtures test workspace behavior, not durable storage.
        self.enterContext(mock.patch("os.fsync"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.patch = mock.patch.object(app, "runtime_dir", Path(self.tmp.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.session = {"session_id": "a" * 32}
        self.state = batch.owned_batch({}, self.session)
        self.paths = []
        for n in range(2):
            path = Path(self.tmp.name) / f"tile{n}.png"
            Image.new("RGB", (24, 20), ("white", "green")[n]).save(path)
            self.paths.append(str(path))
        batch.add_images(self.state, self.paths)

    def test_owner_change_clears_previous_queue(self):
        result = batch.owned_batch(self.state, {"session_id": "b" * 32})
        self.assertEqual(result["items"], [])
        self.assertEqual(len(self.state["items"]), 2)

    def test_restore_preserves_independent_masks_and_prompts(self):
        first, second = self.state["items"]
        values = list(batch.restore_item(app, self.state, first, self.session))
        mask = np.ones((20, 24), dtype=bool)
        mask[5:12, 7:13] = False
        inst = app._make_inst(1, "pvs", mask, app._mask_box(mask), 0.8, history=[])
        values[3]["instances"] = {1: inst}
        values[3]["active_instance_id"] = 1
        values[4]["polygon_points"] = [[1, 2], [4, 5]]
        batch.save_snapshot(app, self.state, *values)
        blank = batch.restore_item(app, self.state, second, self.session)
        self.assertEqual(blank[3]["instances"], {})
        restored = batch.restore_item(app, self.state, first, self.session)
        np.testing.assert_array_equal(restored[3]["instances"][1]["mask_fullres_bool"], mask)
        self.assertEqual(restored[4]["polygon_points"], [[1, 2], [4, 5]])
        self.assertNotEqual(values[1]["image_id"], restored[1]["image_id"])

    def test_crop_is_saved_and_not_accumulated(self):
        item = self.state["items"][0]
        values = list(batch.restore_item(app, self.state, item, self.session))
        values[0]["pending_crop_bbox_xyxy"] = [2, 3, 20, 18]
        cropped = app._apply_source_crop(values[0], values[6], self.session, values[5])
        values[:5] = [cropped[0], cropped[3], cropped[4], cropped[5], cropped[6]]
        batch.save_snapshot(app, self.state, *values)
        for _ in range(2):
            restored = batch.restore_item(app, self.state, item, self.session)
            self.assertEqual((restored[1]["width"], restored[1]["height"]), (18, 15))
            self.assertEqual(item["original"].size, (24, 20))

    def test_retry_filters_only_selected_failed_items(self):
        a, b = self.state["items"]
        a["status"], b["status"] = "failed", "done"
        self.assertEqual(batch.selected_for_run(self.state, [a["id"], b["id"]], True), [a["id"]])
        self.assertEqual(batch.selected_for_run(self.state, [], True), [])

    def test_send_stitch_updates_stable_id(self):
        item = self.state["items"][0]
        values = list(batch.restore_item(app, self.state, item, self.session))
        mask = np.ones((20, 24), dtype=bool)
        values[3]["instances"] = {1: app._make_inst(1, "pvs", mask, [0, 0, 24, 20], 0.8, history=[])}
        batch.save_snapshot(app, self.state, *values)
        stitch = new_stitch_state("a" * 32)
        for _ in range(2):
            stitch, count = batch.send_tiles(app, self.state, [item["id"]], stitch)
            self.assertEqual(count, 1)
            self.assertEqual(len(stitch["saved_tiles"]), 1)
            self.assertEqual(stitch["saved_tiles"][0]["tile_id"], item["id"])

if __name__ == "__main__":
    unittest.main()
