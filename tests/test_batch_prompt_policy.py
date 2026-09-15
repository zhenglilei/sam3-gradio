import inspect
import unittest
from copy import deepcopy
from unittest import mock

import test_batch_workspace as fixtures
from sam3_demo import app, batch_workspace as batch
from sam3_demo.stitch_callbacks import new_stitch_state


class BatchPromptPolicyTests(unittest.TestCase):
    setUp = fixtures.BatchWorkspaceTests.setUp

    def _callback(self, name):
        demo = app.create_demo()
        return inspect.unwrap(next(f.fn for f in demo.fns.values()
                                   if getattr(f.fn, "__name__", "") == name))

    def _call(self, callback, values, selected=None, files=None):
        source, image, pcs, pvs, prompt, layout, mode, tool, text, threshold = values
        return callback(self.state, self.session, source, image, pcs, pvs,
                        prompt, layout, mode, tool, text, threshold,
                        selected or [], files, new_stitch_state("a" * 32))

    def test_blank_text_accepts_own_positive_boxes(self):
        pcs = {"positive_boxes": [[1, 2, 8, 10]], "negative_boxes": [[12, 10, 20, 18]]}
        copied = batch.pcs_input_for_batch(pcs, "")
        self.assertEqual(copied, pcs)
        copied["positive_boxes"][0][0] = 9
        self.assertEqual(pcs["positive_boxes"][0][0], 1)

    def test_shared_text_accepts_no_boxes(self):
        self.assertEqual(batch.pcs_input_for_batch({}, "circle"), {})

    def test_blank_text_without_positive_box_is_actionable(self):
        for pcs in ({}, {"positive_boxes": [], "negative_boxes": [[1, 2, 8, 10]]}):
            with self.assertRaisesRegex(ValueError, "正样本框"):
                batch.pcs_input_for_batch(pcs, " ")

    def test_batch_steps_keep_each_images_own_boxes(self):
        boxes = [[1, 2, 8, 10], [4, 5, 12, 16]]
        for item, box in zip(self.state["items"], boxes):
            values = list(batch.restore_item(app, self.state, item, self.session))
            values[2]["positive_boxes"] = [box]
            values[6] = "PCS Auto"
            batch.save_snapshot(app, self.state, *values)
        ids = [i["id"] for i in self.state["items"]]
        self.state.update(running=True, pending=ids[:], batch_text="", batch_threshold=0.4)
        seen = []

        def predict(image, pcs, pvs, mode, text, threshold):
            seen.append(deepcopy(pcs["positive_boxes"]))
            self.assertEqual(text, "")
            return (pcs, *app._view(image, pcs, pvs, mode, "PCS found no instances"))

        callback = self._callback("batch_step")
        with mock.patch.object(app, "_run_pcs", side_effect=predict):
            self._call(callback, values, ids)
            self._call(callback, values, ids)
        self.assertEqual(seen, [[boxes[0]], [boxes[1]]])
        self.assertFalse(self.state["running"])
        self.assertEqual([i["status"] for i in self.state["items"]], ["done", "done"])

    def test_start_batch_does_not_require_text(self):
        item = self.state["items"][0]
        values = list(batch.restore_item(app, self.state, item, self.session))
        values[2]["positive_boxes"] = [[1, 2, 8, 10]]
        with mock.patch.object(batch.gr, "Info"):
            self._call(self._callback("batch_run"), values, [item["id"]])
        self.assertTrue(self.state["running"])
        self.assertEqual(self.state["batch_text"], "")

    def test_upload_and_select_all_keep_choice_values(self):
        values = batch.restore_item(app, self.state, self.state["items"][0], self.session)
        out = self._call(self._callback("batch_upload"), values, files=self.paths)
        self.assertEqual(len(out[2]["value"]), 2)
        out = self._call(self._callback("batch_select_all"), values)
        self.assertEqual(len(out[2]["value"]), 4)
        out = self._call(self._callback("batch_select_none"), values)
        self.assertEqual(out[2]["value"], [])

    def test_rapid_run_uses_latest_server_selection(self):
        values = batch.restore_item(app, self.state, self.state["items"][0], self.session)
        self._call(self._callback("batch_select_all"), values)
        with mock.patch.object(batch.gr, "Info"):
            self._call(self._callback("batch_run"), values, selected=[])
        self.assertEqual(len(self.state["pending"]), 2)

    def test_manual_selection_updates_server_selection(self):
        values = batch.restore_item(app, self.state, self.state["items"][0], self.session)
        self._call(self._callback("batch_select_all"), values)
        first = self.state["items"][0]["id"]
        self._call(self._callback("batch_selection"), values, selected=[first])
        self.assertEqual(self.state["selected_ids"], [first])

    def test_failed_step_keeps_previous_pvs_snapshot(self):
        values = list(batch.restore_item(app, self.state, self.state["items"][0], self.session))
        mask = fixtures.np.ones((20, 24), dtype=bool)
        values[3]["instances"] = {1: app._make_inst(1, "pvs", mask, [0, 0, 24, 20], 0.8, history=[])}
        batch.save_snapshot(app, self.state, *values)
        item = self.state["items"][0]
        self.state.update(running=True, pending=[item["id"]], batch_text="", batch_threshold=0.4)
        callback = self._callback("batch_step")
        with mock.patch.object(app, "_run_pcs") as infer, mock.patch.object(batch.gr, "Warning"):
            self._call(callback, values)
        infer.assert_not_called()
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["snapshot"]["mode"], "PVS Manual")
        fixtures.np.testing.assert_array_equal(item["snapshot"]["pvs"]["instances"][1]["mask_fullres_bool"], mask)


if __name__ == "__main__":
    unittest.main()
