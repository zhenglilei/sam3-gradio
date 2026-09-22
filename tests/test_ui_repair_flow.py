import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from sam3_demo import app
from sam3_demo import ui_repair
from sam3_demo import ui_repair_core as repair
from sam3_demo.stitch_callbacks import new_stitch_state


class UIRepairFlowTests(unittest.TestCase):
    session_id = "a" * 32

    def test_action_availability_tracks_selection_and_result_invalidation(self):
        self.assertEqual(ui_repair.repair_action_availability(None, []), (False, False, False))
        state = {"active_id": "a", "items": [{"id": "a"}, {"id": "b", "result_path": "result.png"}]}
        self.assertEqual(ui_repair.repair_action_availability(state, []), (True, True, False))
        self.assertEqual(ui_repair.repair_action_availability(state, ["b"]), (True, True, True))
        state["items"][1]["result_path"] = None
        self.assertEqual(ui_repair.repair_action_availability(state, ["b"]), (True, True, False))
        self.assertEqual(ui_repair.repair_action_availability(state, ["removed"]), (True, True, False))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repair-runtime"
        self.session = {"session_id": self.session_id}
        self.state = repair.owned_repair_state(None, self.session)
        self.runtime_patch = mock.patch.object(app, "runtime_dir", Path(self.temp.name))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    @staticmethod
    def _callback(demo, name, occurrence=0):
        callbacks = [
            inspect.unwrap(event.fn)
            for event in demo.fns.values()
            if getattr(event.fn, "__name__", "") == name
        ]
        if len(callbacks) <= occurrence:
            raise AssertionError(f"missing callback {name!r} occurrence {occurrence}")
        return callbacks[occurrence]

    @staticmethod
    def _invoke(callback, values):
        parameters = inspect.signature(callback).parameters
        missing = [
            parameter.name
            for parameter in parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.name not in values
        ]
        if missing:
            raise AssertionError(f"missing callback inputs: {missing}")
        return callback(
            **{
                parameter.name: values[parameter.name]
                for parameter in parameters.values()
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and parameter.name in values
            }
        )

    def _add_repair_images(self, count=2):
        paths = []
        for index, color in enumerate(("red", "yellow")[:count]):
            path = Path(self.temp.name) / f"repair_{index}.png"
            image = Image.new("RGB", (32, 24), "white")
            ImageDraw.Draw(image).rectangle((4, 5, 14, 16), fill=color)
            image.save(path)
            paths.append(str(path))
        items = repair.add_uploaded_images(self.state, paths, self.root)
        for index, item in enumerate(items):
            result_path = Path(item["source_path"]).with_name(f"repaired_{index}.png")
            with Image.open(item["source_path"]) as source:
                source.save(result_path)
            item["result_path"] = str(result_path)
            item["status"] = "已修复"
        return items

    def test_repair_selection_accepts_dynamic_image_ids(self):
        demo = app.create_demo()
        component = next(
            component
            for component in demo.get_config_file()["components"]
            if component["type"] == "dropdown"
            and component.get("props", {}).get("label") == "\u9009\u4e2d\u7684\u56fe\u7247"
        )
        self.assertTrue(component["props"].get("allow_custom_value"))

    def test_handoff_switches_tab_before_delayed_batch_restore(self):
        config = app.create_demo().get_config_file()
        main_tabs = next(
            component["id"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "main_tabs"
        )
        component_types = {
            component["id"]: component.get("type")
            for component in config["components"]
        }
        send = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "send_to_segmentation"
        )
        settle = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "wait_for_segmentation_tab"
        )
        upload = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "batch_upload_1"
        )
        workspace_overlay = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "batch_refresh_workspace_overlay"
        )
        source_overlay = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "batch_refresh_source_overlay"
        )
        layout_editor = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "batch_refresh_layout_editor"
        )
        self.assertNotIn(main_tabs, send["outputs"])
        self.assertTrue(
            any(
                not dependency.get("backend_fn")
                and "#main_tabs" in str(dependency.get("js") or "")
                and dependency.get("targets") == send.get("targets")
                for dependency in config["dependencies"]
            )
        )
        self.assertEqual(settle.get("trigger_after"), send["id"])
        self.assertEqual(upload.get("trigger_after"), settle["id"])
        self.assertNotIn(main_tabs, upload["outputs"])
        output_types = [component_types[component_id] for component_id in upload["outputs"]]
        self.assertEqual(output_types.count("imagegestureoverlay"), 0)
        self.assertNotIn("layouttransformeditor", output_types)
        self.assertEqual(workspace_overlay.get("trigger_after"), upload["id"])
        self.assertEqual(source_overlay.get("trigger_after"), workspace_overlay["id"])
        self.assertEqual(layout_editor.get("trigger_after"), source_overlay["id"])
        self.assertEqual(
            [component_types[item] for item in workspace_overlay["outputs"]],
            ["imagegestureoverlay"],
        )
        self.assertEqual(
            [component_types[item] for item in source_overlay["outputs"]],
            ["imagegestureoverlay"],
        )
        self.assertEqual(
            [component_types[item] for item in layout_editor["outputs"]],
            ["layouttransformeditor"],
        )


    def test_batch_repair_detects_red_and_yellow_before_lama(self):
        items = self._add_repair_images()
        selected = [item["id"] for item in items]
        events = []
        lama = object()

        def detect(state, image_ids, colors, **parameters):
            events.append(("detect", list(image_ids), tuple(colors), parameters))
            return len(image_ids), len(colors)

        def run_lama(state, image_ids, runtime):
            events.append(("lama", list(image_ids), runtime))
            return len(image_ids), []

        demo = app.create_demo()
        callback = self._callback(demo, "repair_selected")
        with (
            mock.patch.object(ui_repair.core, "apply_color_detection", side_effect=detect),
            mock.patch.object(ui_repair.core, "repair_items", side_effect=run_lama),
            mock.patch.object(ui_repair, "get_lama_runtime", return_value=lama),
        ):
            self._invoke(
                callback,
                {
                    "repair_state": self.state,
                    "session_state": self.session,
                    "selected": selected,
                    "editor_value": None,
                    "tool": "brush",
                    "brush_size": 33,
                    "alpha": 0.55,
                    "saturation": 133,
                    "value": 147,
                    "min_area": 9,
                    "padding": 4,
                    "merge_distance": 6,
                },
            )

        self.assertEqual([event[0] for event in events], ["detect", "lama"])
        self.assertEqual(events[0][1], selected)
        self.assertEqual(events[0][2], ("red", "yellow"))
        self.assertEqual(
            events[0][3],
            {
                "saturation": 133,
                "value": 147,
                "min_area": 9,
                "padding": 4,
                "merge_distance": 6,
            },
        )
        self.assertEqual(events[1][1], selected)
        self.assertIs(events[1][2], lama)

    def test_repaired_handoff_uses_batch_upload_restore_for_first_workspace(self):
        items = self._add_repair_images()
        selected = [item["id"] for item in items]
        expected_paths = [item["result_path"] for item in items]
        demo = app.create_demo()

        send_callback = self._callback(demo, "send_to_segmentation")
        handoff = self._invoke(
            send_callback,
            {
                "repair_state": self.state,
                "session_state": self.session,
                "selected": selected,
                "editor_value": None,
                "batch": None,
                "batch_selected": [],
            },
        )
        handoff_paths = handoff[-1]
        self.assertEqual(handoff_paths, expected_paths)

        upload_events = [
            event
            for event in demo.fns.values()
            if getattr(event.fn, "__name__", "") == "batch_upload"
        ]
        self.assertGreaterEqual(len(upload_events), 2)
        upload_event = upload_events[-1]
        upload_callback = inspect.unwrap(upload_event.fn)
        restored = self._invoke(
            upload_callback,
            {
                "batch": None,
                "session_state": self.session,
                "source_image_state": {},
                "image_state": {},
                "pcs_state": {},
                "pvs_state": {},
                "prompt_state": {},
                "layout_state": {},
                "mode": "PVS Manual",
                "click_tool": "bbox",
                "text": "",
                "threshold": 0.4,
                "selected": [],
                "files": handoff_paths,
                "stitch_state": new_stitch_state(self.session_id),
            },
        )

        batch_state = restored[0]
        first = batch_state["items"][0]
        self.assertEqual(batch_state["active_id"], first["id"])
        self.assertEqual(
            batch_state["selected_ids"],
            [item["id"] for item in batch_state["items"]],
        )
        self.assertEqual(first["name"], Path(expected_paths[0]).name)
        self.assertIsNotNone(first["original"])
        self.assertEqual(first["original"].size, (32, 24))

        source_state = restored[5]
        image_state = restored[6]
        self.assertTrue(source_state.get("source_image_id"))
        self.assertTrue(source_state.get("workspace_image_id"))
        self.assertTrue(image_state.get("image_id"))
        self.assertEqual(image_state.get("source_image_id"), source_state["source_image_id"])
        self.assertEqual(source_state["workspace_image_id"], image_state["image_id"])
        self.assertIsNotNone(restored[11])
        workspace_callback = self._callback(demo, "batch_refresh_workspace_overlay")
        workspace_payload = self._invoke(
            workspace_callback,
            {
                "session_state": self.session,
                "image_state": image_state,
                "mode": "PVS Manual",
                "click_tool": "bbox",
            },
        )
        self.assertTrue(workspace_payload.get("server_view"))


if __name__ == "__main__":
    unittest.main()
