import inspect
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr
from PIL import Image

from sam3_demo import app, batch_workspace
from sam3_demo.session_guard import SessionGuardError


class BatchQueueRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = mock.patch.object(app, "runtime_dir", Path(self.temp.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        session_hash = uuid.uuid4().hex
        self.request = gr.Request(
            username="batch-owner",
            session_hash=session_hash,
            client=SimpleNamespace(host="10.0.0.1"),
            headers={},
        )
        self.session = app._SESSION_REGISTRY.bind(app.request_identity(self.request))
        self.addCleanup(app._SESSION_REGISTRY.close, self.session)
        self.demo = app.create_demo()
        self.paths = []
        for index in range(4):
            path = Path(self.temp.name) / f"tile_{index}.png"
            Image.new("RGB", (32, 24), (index * 30, 80, 90)).save(path)
            self.paths.append(str(path))
        self.values = {}

    def event(self, name):
        return [e for e in self.demo.fns.values() if getattr(e.fn, "__name__", "") == name][-1]

    def invoke(self, name, _request=None, **overrides):
        event = self.event(name)
        names = list(inspect.signature(inspect.unwrap(event.fn)).parameters)
        for key, component in zip(names, event.inputs):
            if key in overrides:
                self.values[component._id] = overrides[key]
        args = [overrides.get(key, self.values.get(component._id))
                for key, component in zip(names, event.inputs)]
        result = event.fn(*args, _request or self.request)
        result = result if isinstance(result, tuple) else (result,)
        for component, value in zip(event.outputs, result):
            if isinstance(value, dict) and value.get("__type__") == "update":
                if "value" in value:
                    self.values[component._id] = value["value"]
            else:
                self.values[component._id] = value
        return result

    def upload(self):
        states = app._session_recovery_states(self.session)
        self.invoke("batch_upload", batch=None, **states, mode="PVS Manual",
                    click_tool="bbox", text="", threshold=0.4, selected=[], files=self.paths)
        self.batch_id = self.event("batch_upload").inputs[0]._id
        return self.values[self.batch_id]

    def business_states(self):
        event = self.event("batch_upload")
        names = list(inspect.signature(inspect.unwrap(event.fn)).parameters)
        return {name: self.values[c._id] for name, c in zip(names, event.inputs)
                if name.endswith("_state") and name != "session_state"}

    def test_delete_noncurrent_keeps_live_workspace_and_unsaved_prompt(self):
        batch = self.upload()
        before = deepcopy(self.business_states())
        target = batch["items"][2]["id"]
        self.invoke("batch_selection", selected=[target])
        self.invoke("batch_delete_selected")
        # A guarded callback may replace the component's state mapping.
        batch = self.values[self.batch_id]
        self.assertEqual(len(batch["items"]), 3)
        self.assertNotIn(target, [i["id"] for i in batch["items"]])
        after = self.business_states()
        self.assertEqual(before["image_state"], after["image_state"])
        self.assertEqual(before["prompt_state"], after["prompt_state"])

    def test_different_owner_cannot_send_annotation_queue_to_stitch(self):
        batch = self.upload()
        before = deepcopy(batch)
        wrong_request = gr.Request(
            username="different-owner",
            session_hash=self.request.session_hash,
            client=SimpleNamespace(host="10.0.0.1"),
            headers={},
        )
        with self.assertRaises(SessionGuardError):
            self.invoke("batch_stitch", _request=wrong_request)
        self.assertEqual(batch, before)

    def test_delete_current_selects_next_and_preserves_remaining_ids(self):
        batch = self.upload()
        ids = [i["id"] for i in batch["items"]]
        self.invoke("batch_delete_current")
        self.assertEqual([i["id"] for i in batch["items"]], ids[1:])
        self.assertEqual(batch["active_id"], ids[1])
        self.invoke("_switch_mode_with_layout_editor", mode="PCS Auto")
        self.assertTrue(self.business_states()["image_state"]["image_id"])

    def test_delete_all_clears_canvas_and_owned_states_and_allows_reupload(self):
        batch = self.upload()
        self.invoke("batch_delete_selected")
        self.assertEqual(batch["items"], [])
        self.assertIsNone(batch["active_id"])
        self.assertEqual(batch["selected_ids"], [])
        states = self.business_states()
        self.assertIsNone(states["image_state"]["image_id"])
        self.assertFalse(states["pvs_state"]["instances"])
        for value in states.values():
            self.assertEqual(value["session_id"], self.session["session_id"])
            self.assertEqual(value["owner_token"], self.session["owner_token"])
        self.invoke("_switch_mode_with_layout_editor", mode="PCS Auto")
        self.invoke("batch_upload", files=self.paths[:1])
        self.assertEqual(len(batch["items"]), 1)
        self.assertTrue(self.business_states()["image_state"]["image_id"])

    def test_empty_selection_does_not_remove_active_image(self):
        batch = self.upload()
        ids = [i["id"] for i in batch["items"]]
        self.invoke("batch_selection", selected=[])
        self.invoke("batch_delete_selected")
        self.assertEqual([i["id"] for i in batch["items"]], ids)

    def test_running_batch_rejects_deletion(self):
        batch = self.upload()
        batch["running"] = True
        self.invoke("batch_delete_current")
        self.assertEqual(len(batch["items"]), 4)

    def test_saved_stitch_queue_and_source_files_survive_removal(self):
        batch = self.upload()
        stitch = self.business_states()["stitch_state"]
        stitch["saved_tiles"] = [{"tile_id": batch["items"][0]["id"], "name": "saved"}]
        self.invoke("batch_delete_selected")
        self.assertEqual(len(stitch["saved_tiles"]), 1)
        self.assertTrue(all(Path(p).is_file() for p in self.paths))

    def test_multiple_removal_chooses_nearest_survivor(self):
        batch = self.upload()
        ids = [i["id"] for i in batch["items"]]
        batch["active_id"] = ids[2]
        removed, changed = batch_workspace.remove_images(batch, [ids[1], ids[2]])
        self.assertEqual((removed, changed), (2, True))
        self.assertEqual(batch["active_id"], ids[3])


if __name__ == "__main__":
    unittest.main()
