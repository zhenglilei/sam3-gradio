import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import gradio as gr
from PIL import Image
from sam3_demo import app, ui_repair_core
from sam3_demo.session_guard import SessionGuardError


class BatchOwnedRestoreTests(unittest.TestCase):
    def test_four_image_restore_keeps_guarded_mode_callback_usable(self):
        request = gr.Request(
            username="batch-owner",
            session_hash="batch-owned-test",
            client=SimpleNamespace(host="10.0.0.1"),
            headers={},
        )
        session = app._SESSION_REGISTRY.bind(app.request_identity(request))
        self.addCleanup(app._SESSION_REGISTRY.close, session)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(app, "runtime_dir", Path(folder)):
            paths = []
            for index in range(4):
                path = Path(folder) / f"tile_{index}.png"
                Image.new("RGB", (32, 24), "green").save(path)
                paths.append(str(path))
            demo = app.create_demo()
            repair_state = ui_repair_core.owned_repair_state(None, session)
            items = ui_repair_core.add_uploaded_images(repair_state, paths, Path(folder) / "repair")
            for item in items:
                item["result_path"] = item["source_path"]
                item["status"] = "已修复"
            send = next(e for e in demo.fns.values() if getattr(e.fn, "__name__", "") == "send_to_segmentation")
            wrong_request = gr.Request(
                username="different-owner",
                session_hash="batch-owned-test",
                client=SimpleNamespace(host="10.0.0.1"),
                headers={},
            )
            before_repair = deepcopy(repair_state)
            with self.assertRaises(SessionGuardError):
                send.fn(
                    repair_state,
                    session,
                    [item["id"] for item in items],
                    None,
                    wrong_request,
                )
            self.assertEqual(repair_state, before_repair)
            handoff = send.fn(repair_state, session, [item["id"] for item in items], None, request)
            paths = handoff[-1]
            event = [e for e in demo.fns.values() if getattr(e.fn, "__name__", "") == "batch_upload"][-1]
            states = app._session_recovery_states(session)
            values = [None, session, states["source_image_state"], states["image_state"],
                      states["pcs_state"], states["pvs_state"], states["prompt_state"], states["layout_state"],
                      "PVS Manual", "bbox", "", 0.4, [], paths, states["stitch_state"]]
            before_states = deepcopy(values[1:8])
            with self.assertRaises(SessionGuardError):
                event.fn(*values, wrong_request)
            self.assertEqual(values[1:8], before_states)
            result = event.fn(*values, request)
            state_by_id = {component._id: value for component, value in zip(event.outputs, result)}
            config = demo.get_config_file()
            mode = next(d for d in config["dependencies"] if d.get("api_name") == "_switch_mode_with_layout_editor")
            arguments = ["PCS Auto", *[state_by_id[cid] for cid in mode["inputs"][1:]]]
            # Do not unwrap: real UI calls must retain ownership across the handoff.
            demo.fns[mode["id"]].fn(*arguments, request)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(Path(p).is_file() for p in paths))
            for cid in mode["inputs"][1:]:
                self.assertEqual(state_by_id[cid]["session_id"], session["session_id"])
                self.assertEqual(state_by_id[cid]["owner_token"], session["owner_token"])


if __name__ == "__main__":
    unittest.main()
