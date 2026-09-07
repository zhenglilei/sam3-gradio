from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam3_demo.overlap_authoring import OverlapAuthoringService
from sam3_demo.overlap_authoring_callbacks import (
    add_authoring_layer,
    apply_authoring_editor,
    delete_authoring_layout,
    inspect_authoring_annotation,
    draft_to_view,
    load_authoring_layout,
    new_authoring_draft,
    propose_authoring_alignment,
    refresh_authoring_layouts,
    remove_authoring_layer,
    save_authoring_layout,
    select_authoring_layer,
    set_authoring_base_image,
    set_authoring_name,
)
from sam3_demo.overlap_service import OverlapPredictionService
from sam3_demo.overlap_store import OverlapLayoutStore, OverlapStoreError


def _advance_intent(payload: dict, amount: int = 1) -> None:
    intent = payload["group_intent"]
    revision = int(intent.get("client_revision") or 0) + amount
    intent["client_revision"] = revision
    intent["transform_set_revision"] = revision
    intent["payload_hash"] = f"test:{revision}"


def _base(width=16, height=12):
    return Image.new("RGB", (width, height), (20, 30, 40))


def _mask(width=6, height=5):
    array = np.zeros((height, width), dtype=np.uint8)
    array[1:4, 1:5] = 255
    return Image.fromarray(array, mode="L")


class OverlapAuthoringCallbackTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = OverlapLayoutStore(Path(self.tempdir.name) / "layouts.sqlite3")
        self.authoring = OverlapAuthoringService(self.store)
        self.viewer = OverlapPredictionService(self.store)
        self.patchers = [
            patch(
                "sam3_demo.overlap_authoring_callbacks.overlap_runtime.get_overlap_authoring_service",
                return_value=self.authoring,
            ),
            patch(
                "sam3_demo.overlap_authoring_callbacks.overlap_runtime.get_overlap_service",
                return_value=self.viewer,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.store.close()
        self.tempdir.cleanup()

    def _draft_with_layers(self):
        draft = self.authoring.new_draft()
        draft = self.authoring.set_name(draft, "sample")
        draft = self.authoring.set_base_image(draft, _base())
        draft = self.authoring.add_layer(draft, _mask(), "ACT", "active")
        draft = self.authoring.add_layer(draft, _mask(), "GE1", "gate")
        return draft

    def _write_coco(self):
        path = Path(self.tempdir.name) / "labels.json"
        path.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 7,
                            "file_name": "sample.png",
                            "width": 6,
                            "height": 5,
                        }
                    ],
                    "annotations": [
                        {
                            "image_id": 7,
                            "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_empty_and_view_contract_are_stable(self):
        view = new_authoring_draft()
        self.assertEqual(len(view), 6)
        self.assertEqual(view[0]["layers"], [])
        self.assertIsNone(view[2])
        self.assertEqual(view[3]["choices"], [])
        self.assertFalse(view[4]["enabled"])
        self.assertIn("整体版图", view[5])
        view_again = draft_to_view({}, service=self.authoring)
        self.assertEqual(view_again[0]["layers"], [])

    def test_mutating_callback_success_paths(self):
        draft = new_authoring_draft()[0]
        draft = set_authoring_name("sample", draft)[0]
        draft = set_authoring_base_image(_base(), draft)[0]
        coco = self._write_coco()
        draft = add_authoring_layer("ACT", coco, draft)[0]
        draft = add_authoring_layer("custom-metal", coco, draft)[0]
        self.assertEqual(len(draft["layers"]), 2)
        first, second = [layer["id"] for layer in draft["layers"]]
        selected = select_authoring_layer(first, draft)
        self.assertEqual(selected[0]["active_layer_id"], first)
        removed = remove_authoring_layer(second, selected[0])
        self.assertEqual(len(removed[0]["layers"]), 1)
        self.assertIn("groups", removed[4]["group_view"])

    def test_failures_preserve_previous_draft(self):
        draft = self._draft_with_layers()
        original = copy.deepcopy(draft)
        failed = add_authoring_layer("", self._write_coco(), draft)
        self.assertEqual(failed[0], original)
        self.assertIn("失败", failed[-1])
        failed = set_authoring_base_image(None, draft)
        self.assertEqual(failed[0], original)
        self.assertIn("失败", failed[-1])

    def test_runtime_getter_failures_are_rendered_without_throwing(self):
        draft = self._draft_with_layers()
        with patch(
            "sam3_demo.overlap_authoring_callbacks.overlap_runtime.get_overlap_authoring_service",
            side_effect=OverlapStoreError("runtime unavailable"),
        ):
            callback_results = [
                new_authoring_draft(),
                set_authoring_name("next", draft),
                set_authoring_base_image(_base(), draft),

                add_authoring_layer("ACT", self._write_coco(), draft),
                select_authoring_layer(draft["layers"][0]["id"], draft),
                remove_authoring_layer(draft["layers"][0]["id"], draft),
                apply_authoring_editor({}, draft),
                load_authoring_layout("missing", draft),
                save_authoring_layout(draft),
                delete_authoring_layout(draft),
            ]
        self.assertEqual(len(callback_results[0]), 6)
        for result in callback_results[1:6]:
            self.assertEqual(result[0], draft)
            self.assertIn("失败", result[-1])
        self.assertEqual(len(callback_results[6]), 3)
        self.assertEqual(callback_results[6][1]["payload_kind"], "ack_error")
        self.assertIn("失败", callback_results[6][-1])
        for result in callback_results[7:]:
            self.assertEqual(len(result), 7)
            self.assertEqual(result[0], draft)
            self.assertIn("失败", result[-2])

    def test_corrupt_layer_still_returns_recoverable_runtime_error(self):
        draft = self._draft_with_layers()
        draft["layers"] = [None]
        with patch(
            "sam3_demo.overlap_authoring_callbacks.overlap_runtime.get_overlap_authoring_service",
            side_effect=OverlapStoreError("runtime unavailable"),
        ):
            result = set_authoring_name("next", draft)
        self.assertEqual(result[0], draft)
        self.assertFalse(result[4]["enabled"])
        self.assertIn("失败", result[-1])

    def test_editor_intent_reorder_and_save_update(self):
        draft = self._draft_with_layers()
        first, second = [layer["id"] for layer in draft["layers"]]
        payload = draft_to_view(draft, service=self.authoring)[4]
        intent = payload["group_intent"]
        _advance_intent(payload)
        intent["changed_group_ids"] = [first]
        intent["ordered_group_ids"] = [second, first]
        item = next(x for x in intent["transforms"] if x["group_id"] == first)
        item["transform"]["center_x"] += 1
        updated = apply_authoring_editor(payload, draft)
        self.assertEqual(updated[1]["payload_kind"], "ack")
        self.assertEqual([x["id"] for x in updated[0]["layers"]], [second, first])
        saved = save_authoring_layout(updated[0])
        self.assertEqual(len(saved), 7)
        self.assertEqual(saved[0]["layout_id"], saved[6]["value"])
        self.assertEqual(len(self.store.list_layouts()), 1)

    def test_save_applies_latest_unsynced_editor_snapshot(self):
        draft = self._draft_with_layers()
        payload = draft_to_view(draft, service=self.authoring)[4]
        first = draft["layers"][0]["id"]
        _advance_intent(payload)
        intent = payload["group_intent"]
        intent["changed_group_ids"] = [first]
        item = next(x for x in intent["transforms"] if x["group_id"] == first)
        item["transform"]["center_x"] = 11.25

        saved = save_authoring_layout(draft, "latest", payload)

        persisted = self.store.get_layout(saved[0]["layout_id"])
        layer = next(
            x for x in persisted["layers"] if x["layer_type"] == "ACT"
        )
        self.assertEqual(layer["transform"]["center_x"], 11.25)
        self.assertEqual(saved[0]["editor_sync_revision"], 0)

    def test_alignment_proposal_callback_does_not_replace_draft(self):
        draft = self._draft_with_layers()
        proposal = {
            "payload_kind": "alignment_proposal",
            "candidates": [],
            "auto_apply": False,
            "ambiguous": True,
            "status": "存在周期歧义",
        }
        with patch.object(
            self.authoring,
            "alignment_proposal",
            return_value=proposal,
        ):
            result = propose_authoring_alignment(draft)
        self.assertEqual(result, (proposal, "存在周期歧义"))
        self.assertNotIn("draft_state", result[0])

    def test_annotation_inspection_returns_binary_preview(self):
        preview, status = inspect_authoring_annotation(self._write_coco())
        self.assertEqual(preview.mode, "L")
        self.assertTrue(np.asarray(preview).any())
        self.assertIn("COCO", status)

        preview, status = inspect_authoring_annotation(None)
        self.assertIsNone(preview)
        self.assertIn("失败", status)

    def test_save_uses_current_name_without_waiting_for_blur(self):
        draft = self._draft_with_layers()
        draft = self.authoring.set_name(draft, "stale name")
        saved = save_authoring_layout(draft, "current field name")
        self.assertEqual(saved[0]["name"], "current field name")
        self.assertEqual(self.store.list_layouts()[0]["name"], "current field name")

    def test_load_delete_refresh_and_store_error(self):
        saved = save_authoring_layout(self._draft_with_layers())
        layout_id = saved[0]["layout_id"]
        loaded = load_authoring_layout(layout_id, saved[0])
        self.assertEqual(loaded[0]["layout_id"], layout_id)
        choices, status = refresh_authoring_layouts(layout_id)
        self.assertEqual(choices["value"], layout_id)
        self.assertIn("刷新", status)
        deleted = delete_authoring_layout(loaded[0])
        self.assertIsNone(deleted[0]["layout_id"])
        self.assertEqual(deleted[6]["choices"], [])
        with patch.object(
            self.viewer,
            "layout_choices",
            side_effect=OverlapStoreError("database unavailable"),
        ):
            choices, status = refresh_authoring_layouts(layout_id)
        self.assertNotIn("database unavailable", str(choices))
        self.assertIn("失败", status)


if __name__ == "__main__":
    unittest.main()
