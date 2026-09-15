import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_demo import app
from sam3_demo import ui_repair
from sam3_demo import ui_repair_core as repair


class RepairQueueRemovalTests(unittest.TestCase):
    session_id = "b" * 32

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repair-runtime"
        self.session = {"session_id": self.session_id}
        self.state = repair.owned_repair_state(None, self.session)

    def _add_images(self, count=4):
        paths = []
        for index in range(count):
            path = Path(self.temp.name) / f"repair_{index}.png"
            Image.new("RGB", (32 + index, 24 + index), "white").save(path)
            paths.append(str(path))
        return repair.add_uploaded_images(self.state, paths, self.root)

    @staticmethod
    def _callback(demo, name):
        callbacks = [
            event
            for event in demo.fns.values()
            if getattr(inspect.unwrap(event.fn), "__name__", "") == name
        ]
        if len(callbacks) != 1:
            raise AssertionError(f"expected one callback {name!r}, got {len(callbacks)}")
        return callbacks[0]

    @staticmethod
    def _invoke(callback, values):
        function = inspect.unwrap(callback.fn)
        parameters = inspect.signature(function).parameters
        return function(
            **{
                parameter.name: values[parameter.name]
                for parameter in parameters.values()
                if parameter.kind
                in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                and parameter.name in values
            }
        )

    @staticmethod
    def _update_value(update):
        if isinstance(update, dict):
            return update.get("value")
        return getattr(update, "value", None)

    def test_remove_active_uses_next_at_same_index_and_keeps_artifacts(self):
        items = self._add_images()
        self.state["active_id"] = items[1]["id"]
        self.state["selected_ids"] = [
            items[0]["id"],
            items[1]["id"],
            items[3]["id"],
        ]
        result_paths = []
        artifact_bytes = {}
        for item in items:
            result_path = Path(item["source_path"]).with_name("result.png")
            with Image.open(item["source_path"]) as source:
                source.save(result_path)
            item["result_path"] = str(result_path)
            artifact_bytes[item["id"]] = (
                Path(item["mask_path"]).read_bytes(),
                result_path.read_bytes(),
            )
            result_paths.append(result_path)

        removed = repair.remove_repair_items(self.state, [items[1]["id"]])

        self.assertEqual(removed, [items[1]["id"]])
        self.assertEqual(
            [item["id"] for item in self.state["items"]],
            [items[0]["id"], items[2]["id"], items[3]["id"]],
        )
        self.assertEqual(self.state["active_id"], items[2]["id"])
        self.assertEqual(
            self.state["selected_ids"],
            [items[0]["id"], items[3]["id"]],
        )
        for item, result_path in zip(items, result_paths):
            self.assertTrue(Path(item["source_path"]).is_file())
            self.assertTrue(Path(item["mask_path"]).is_file())
            self.assertTrue(result_path.is_file())
            self.assertEqual(
                (
                    Path(item["mask_path"]).read_bytes(),
                    result_path.read_bytes(),
                ),
                artifact_bytes[item["id"]],
            )

    def test_remove_nonactive_multiple_preserves_active_and_stable_survivors(self):
        items = self._add_images()
        self.state["active_id"] = items[2]["id"]
        self.state["selected_ids"] = [item["id"] for item in items]

        removed = repair.remove_repair_items(
            self.state,
            [items[0]["id"], items[3]["id"]],
        )

        self.assertEqual(removed, [items[0]["id"], items[3]["id"]])
        self.assertEqual(
            [item["id"] for item in self.state["items"]],
            [items[1]["id"], items[2]["id"]],
        )
        self.assertEqual(self.state["active_id"], items[2]["id"])
        self.assertEqual(self.state["selected_ids"], [items[1]["id"], items[2]["id"]])

    def test_remove_last_active_uses_previous_item(self):
        items = self._add_images(3)
        self.state["active_id"] = items[-1]["id"]

        removed = repair.remove_repair_items(self.state, [items[-1]["id"]])

        self.assertEqual(removed, [items[-1]["id"]])
        self.assertEqual(self.state["active_id"], items[-2]["id"])
        self.assertEqual(
            [item["id"] for item in self.state["items"]],
            [items[0]["id"], items[1]["id"]],
        )

    def test_remove_all_clears_queue_selection_and_active_without_deleting_files(self):
        items = self._add_images()
        self.state["active_id"] = items[1]["id"]
        self.state["selected_ids"] = [item["id"] for item in items]
        source_paths = [Path(item["source_path"]) for item in items]
        mask_paths = [Path(item["mask_path"]) for item in items]

        removed = repair.remove_repair_items(
            self.state,
            [item["id"] for item in items],
        )

        self.assertEqual(removed, [item["id"] for item in items])
        self.assertEqual(self.state["items"], [])
        self.assertIsNone(self.state["active_id"])
        self.assertEqual(self.state["selected_ids"], [])
        self.assertTrue(all(path.is_file() for path in source_paths + mask_paths))

    def test_empty_or_none_selection_is_a_noop(self):
        items = self._add_images(2)
        original_items = self.state["items"]
        original_active = self.state["active_id"]
        original_selected = list(self.state["selected_ids"])

        self.assertEqual(repair.remove_repair_items(self.state, []), [])
        self.assertEqual(repair.remove_repair_items(self.state, None), [])
        self.assertIs(self.state["items"], original_items)
        self.assertEqual(self.state["active_id"], original_active)
        self.assertEqual(self.state["selected_ids"], original_selected)
        self.assertEqual(len(items), 2)

    def test_guarded_delete_callbacks_are_serial_and_return_empty_view(self):
        demo = app.create_demo()
        current_event = self._callback(demo, "remove_current")
        selected_event = self._callback(demo, "remove_selected")
        for event in (current_event, selected_event):
            self.assertEqual(event.concurrency_id, "ui-repair-state")
            self.assertEqual(event.concurrency_limit, 1)
            self.assertTrue(event.queue)
        self.assertIsNot(current_event.fn, inspect.unwrap(current_event.fn))

        items = self._add_images(1)
        outputs = self._invoke(
            current_event,
            {
                "repair_state": self.state,
                "session_state": self.session,
                "selected": [],
                "tool": "brush",
                "brush_size": 20,
                "alpha": 0.45,
            },
        )

        self.assertEqual(len(outputs), 7)
        self.assertEqual(outputs[0]["items"], [])
        self.assertEqual(outputs[1], [])
        self.assertEqual(self._update_value(outputs[2]), [])
        self.assertIsNone(outputs[3]["image_id"])
        self.assertIsNone(outputs[4])
        self.assertIn("已删除", outputs[5])
        self.assertEqual(outputs[6], "尚未添加图片")
        self.assertTrue(Path(items[0]["source_path"]).is_file())

    def test_batch_callback_empty_selection_does_not_fallback_to_active(self):
        demo = app.create_demo()
        selected_event = self._callback(demo, "remove_selected")
        items = self._add_images(2)
        original_ids = [item["id"] for item in items]

        outputs = self._invoke(
            selected_event,
            {
                "repair_state": self.state,
                "session_state": self.session,
                "selected": [],
                "tool": "brush",
                "brush_size": 20,
                "alpha": 0.45,
            },
        )

        self.assertEqual(
            [item["id"] for item in outputs[0]["items"]],
            original_ids,
        )
        self.assertEqual(outputs[0]["active_id"], original_ids[0])
        self.assertEqual(self._update_value(outputs[2]), [])
        self.assertIn("未选择", outputs[5])
        self.assertIn("请先选择", outputs[5])

    def test_stale_editor_event_after_delete_cannot_resurrect_image(self):
        demo = app.create_demo()
        editor_event = self._callback(demo, "editor_changed")
        items = self._add_images(2)
        stale_payload = repair.editor_payload(items[0])
        repair.remove_repair_items(self.state, [items[0]["id"]])

        with mock.patch.object(ui_repair.gr, "Warning"):
            outputs = self._invoke(
                editor_event,
                {
                    "repair_state": self.state,
                    "session_state": self.session,
                    "editor_value": stale_payload,
                },
            )

        self.assertEqual(
            [item["id"] for item in outputs[0]["items"]],
            [items[1]["id"]],
        )
        self.assertNotIn(items[0]["id"], [item["id"] for item in outputs[0]["items"]])


if __name__ == "__main__":
    unittest.main()
