"""Large preview images must never be echoed in layout control requests."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_demo.layout.mask_callbacks import (
    _sync_layout_controls_from_editor_with_prompt_epoch_impl as sync_editor,
)
from sam3_demo.ui.bindings import _layout_editor_intent_js


class LayoutEditorTransportTests(unittest.TestCase):
    def test_server_snapshot_is_small_before_editor_mounts(self):
        preview = "data:image/png;base64," + "A" * (3 * 1024 * 1024)
        transform = {"center_x": 40, "center_y": 30, "scale": 1, "revision": 3}
        group_intent = {
            "selection_signature": "saved-labels",
            "active_group_id": "region_2",
            "transforms": [
                {"group_id": name, "transform": transform}
                for name in ["region_1", "region_2"]
            ],
        }
        payload = {
            "enabled": True,
            "base_image": preview,
            "mask_image": preview,
            "group_view": {"groups": [{"mask_image": preview}]},
            "transform": transform,
            "transform_mode": "label_groups",
            "group_intent": group_intent,
            "target_width": 2464,
            "target_height": 1898,
        }
        for position in (1, 11):
            with self.subTest(editor_input=position):
                args = [None] * position + [payload, ["region_label:1", "region_label:2"]]
                code = (
                    "const fs = require('fs');"
                    "const args = JSON.parse(fs.readFileSync(0, 'utf8'));"
                    f"const trim = {_layout_editor_intent_js(position)};"
                    "process.stdout.write(JSON.stringify(trim(...args)));"
                )
                result = subprocess.run(
                    ["node", "-e", code],
                    input=json.dumps(args),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertLess(len(result.stdout), 4096)
                actual = json.loads(result.stdout)
                self.assertEqual(actual[:position], args[:position])
                self.assertEqual(actual[position + 1:], args[position + 1:])
                self.assertEqual(actual[position]["transform"], transform)
                self.assertEqual(actual[position]["group_intent"], group_intent)
                self.assertTrue(actual[position]["enabled"])
                self.assertNotIn("group_view", actual[position])
                self.assertNotIn("data:image", result.stdout)


class LayoutEditorAcknowledgementTests(unittest.TestCase):
    def setUp(self):
        self.state = {"prompt_mask_scope": "region_labels"}
        self.commit = mock.Mock(
            return_value=(self.state, [], {}, None, {"label": "left"})
        )
        self.advance = mock.Mock()
        self.deps = {
            "_LAYOUT_PROMPT_SCOPE_REGION_LABELS": "region_labels",
            "_advance_layout_prompt_epoch": self.advance,
            "_commit_layout_group_transforms": self.commit,
            "_layout_editor_transform": lambda payload: payload.get("transform"),
            "_layout_group_control_values": lambda *_: (2, 3, 1, 0, 0.35),
            "_sync_layout_controls_from_editor": mock.Mock(),
            "gr": gr,
        }

    def payload(self, revision):
        return {
            "payload_kind": "intent",
            "transform": {"image_id": "image-1"},
            "target_width": 48,
            "target_height": 36,
            "group_intent": {
                "editor_session_id": "",
                "document_epoch": "selection-1",
                "selection_signature": "selection-1",
                "client_revision": revision,
                "transform_set_revision": revision,
                "payload_hash": f"intent-{revision}",
            },
        }

    def test_each_accepted_edit_gets_a_matching_small_ack(self):
        for revision in (1, 2):
            result = sync_editor(self.deps, self.state, self.payload(revision))
            self.assertEqual(len(result), 9)
            self.assertEqual(result[1:7], (True, 2, 3, 1, 0, 0.35))
            self.assertEqual(result[-1], {
                "payload_kind": "ack",
                "editor_session_id": "",
                "document_epoch": "selection-1",
                "server_ack_revision": revision,
                "payload_hash": f"intent-{revision}",
            })
            self.assertLess(len(json.dumps(result[-1])), 512)
        self.assertEqual(self.commit.call_count, 2)

    def test_rejected_transform_is_not_acknowledged_as_saved(self):
        self.commit.side_effect = ValueError("stale selection")
        result = sync_editor(self.deps, self.state, self.payload(1))
        self.assertEqual(result[-1]["payload_kind"], "ack_error")
        self.assertIn("stale selection", result[-1]["status"])
        self.assertEqual(result[-1]["payload_hash"], "intent-1")
        self.assertEqual(result[1], gr.skip())

    def test_ack_echo_does_not_invalidate_or_resubmit_transforms(self):
        for kind in ("ack", "ack_error"):
            result = sync_editor(self.deps, self.state, {"payload_kind": kind})
            self.assertEqual(result, (gr.skip(),) * 9)
        self.commit.assert_not_called()
        self.advance.assert_not_called()

    def test_invalid_revision_is_rejected_before_commit(self):
        payload = self.payload(2)
        payload["group_intent"]["client_revision"] = 1
        result = sync_editor(self.deps, self.state, payload)
        self.assertEqual(result[-1]["payload_kind"], "ack_error")
        self.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
