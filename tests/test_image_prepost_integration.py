from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_demo import app as demo


class ImagePrepostIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_runtime_export_dir = demo.runtime_export_dir
        self.old_public_download_dir = demo.public_download_dir
        demo.runtime_export_dir = self.root / "internal_exports"
        demo.runtime_export_dir.mkdir()
        demo.public_download_dir = self.root / "public_downloads"
        demo._clear_workspace_cache()
        demo._clear_source_image_cache()
        self.session_state = {"session_id": "image-prepost-session"}
        self.layout_state = demo._new_layout_state("image-prepost-session")

    def tearDown(self):
        demo._clear_workspace_cache()
        demo._clear_source_image_cache()
        demo.runtime_export_dir = self.old_runtime_export_dir
        demo.public_download_dir = self.old_public_download_dir
        self.temporary.cleanup()

    @staticmethod
    def _source_image():
        pixels = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
        return Image.fromarray(pixels, mode="RGB")

    def _upload(self):
        return demo._source_upload_workspace(
            self._source_image(),
            demo.MODE_PVS,
            self.session_state,
            self.layout_state,
        )

    @staticmethod
    def _gesture(identity, gesture, start, end, revision=None):
        return {
            "gesture": gesture,
            "start_xy": list(start),
            "end_xy": list(end),
            "expected_revision": int(
                revision
                if revision is not None
                else identity.get("interaction_revision")
                or identity.get("source_revision")
                or 0
            ),
            "image_id": str(
                identity.get("image_id") or identity.get("source_image_id") or ""
            ),
            "image_sha256": str(
                identity.get("target_image_sha256")
                or identity.get("source_image_sha256")
                or ""
            ),
        }

    def test_crop_and_use_full_reinitialize_authoritative_workspace(self):
        uploaded = self._upload()
        source_state = uploaded[0]
        image_state = uploaded[3]
        self.assertEqual((image_state["width"], image_state["height"]), (20, 12))
        self.assertEqual(source_state["crop_bbox_xyxy"], [0, 0, 20, 12])
        self.assertEqual(uploaded[1]["server_view"]["selection_state"], "")
        self.assertEqual(uploaded[1]["client_intent"], {})

        gesture = self._gesture(source_state, "drag", [3, 2], [13, 10])
        source_state, gesture_payload, status = demo._record_source_crop_gesture(
            source_state,
            gesture,
        )
        self.assertIn("[3, 2, 13, 10]", status)
        self.assertEqual(source_state["pending_crop_bbox_xyxy"], [3, 2, 13, 10])
        self.assertEqual(gesture_payload["server_view"]["selection_state"], "draft")

        cropped = demo._apply_source_crop(
            source_state,
            demo.MODE_PVS,
            self.session_state,
            self.layout_state,
        )
        source_state = cropped[0]
        cropped_image_state = cropped[3]
        self.assertEqual((cropped_image_state["width"], cropped_image_state["height"]), (10, 8))
        self.assertEqual(cropped_image_state["crop_bbox_xyxy"], [3, 2, 13, 10])
        self.assertIsNone(source_state["pending_crop_bbox_xyxy"])
        self.assertEqual(cropped[1]["server_view"]["selection_state"], "applied")
        expected = np.asarray(self._source_image())[2:10, 3:13]
        np.testing.assert_array_equal(
            np.asarray(demo._workspace(cropped_image_state)["image"]),
            expected,
        )

        restored = demo._use_full_source_image(
            source_state,
            demo.MODE_PVS,
            self.session_state,
            self.layout_state,
        )
        self.assertEqual((restored[3]["width"], restored[3]["height"]), (20, 12))
        self.assertEqual(restored[0]["crop_bbox_xyxy"], [0, 0, 20, 12])
        self.assertEqual(restored[1]["client_intent"], {})

    def test_failed_crop_initialization_preserves_previous_workspace(self):
        uploaded = self._upload()
        source_state = uploaded[0]
        previous_image_state = uploaded[3]
        previous_workspace_image = np.asarray(
            demo._workspace(previous_image_state)["image"]
        ).copy()
        gesture = self._gesture(source_state, "drag", [3, 2], [13, 10])
        source_state, _, _ = demo._record_source_crop_gesture(source_state, gesture)
        with mock.patch.object(
            demo,
            "_source_image_cache_get",
            side_effect=RuntimeError("simulated source cache failure"),
        ):
            failed = demo._apply_source_crop(
                source_state,
                demo.MODE_PVS,
                self.session_state,
                self.layout_state,
            )

        self.assertIn("simulated source cache failure", failed[2])
        self.assertEqual(failed[0]["crop_bbox_xyxy"], [0, 0, 20, 12])
        self.assertEqual(failed[0]["pending_crop_bbox_xyxy"], [3, 2, 13, 10])
        np.testing.assert_array_equal(
            np.asarray(demo._workspace(previous_image_state)["image"]),
            previous_workspace_image,
        )

    def test_use_full_is_noop_when_workspace_already_uses_full_image(self):
        uploaded = self._upload()
        source_state = uploaded[0]
        previous_image_state = uploaded[3]
        result = demo._use_full_source_image(
            source_state, demo.MODE_PVS, self.session_state, self.layout_state
        )
        self.assertIn("无需重复应用", result[2])
        self.assertEqual(demo._workspace(previous_image_state)["image"].size, (20, 12))

    def test_stale_crop_intent_cannot_reuse_previous_pending_box(self):
        source_state = self._upload()[0]
        valid = self._gesture(source_state, "drag", [2, 2], [10, 9])
        source_state, _, _ = demo._record_source_crop_gesture(source_state, valid)
        self.assertIsNotNone(source_state["pending_crop_bbox_xyxy"])

        stale = self._gesture(
            source_state,
            "drag",
            [4, 3],
            [12, 10],
            revision=source_state["source_revision"] + 1,
        )
        source_state, _, status = demo._record_source_crop_gesture(source_state, stale)
        self.assertIsNone(source_state["pending_crop_bbox_xyxy"])
        self.assertIn("revision", status)

    def test_workspace_drag_adds_one_bbox_and_rejects_small_drag(self):
        image_state = self._upload()[3]
        pcs_state = demo._new_pcs_state()
        pvs_state = demo._new_pvs_state()
        prompt_state = demo._new_prompt_state()

        drag = self._gesture(image_state, "drag", [2, 2], [11, 9])
        output = demo._workspace_gesture_input(
            image_state,
            pcs_state,
            pvs_state,
            demo.MODE_PVS,
            "bbox",
            "Positive exemplar",
            prompt_state,
            drag,
        )
        payload = json.loads(output[1])
        self.assertEqual(payload["box_xyxy_px"], [2.0, 2.0, 11.0, 9.0])
        self.assertEqual(output[5]["pending_boxes"], [[2.0, 2.0, 11.0, 9.0]])

        small = self._gesture(image_state, "drag", [2, 2], [4, 4])
        rejected = demo._workspace_gesture_input(
            image_state,
            demo._new_pcs_state(),
            demo._new_pvs_state(),
            demo.MODE_PVS,
            "bbox",
            "Positive exemplar",
            demo._new_prompt_state(),
            small,
        )
        self.assertEqual(rejected[5]["pending_boxes"], [])
        self.assertIn("bbox", rejected[14])

    def test_template_export_is_independent_of_pvs_state_and_sam3(self):
        uploaded = self._upload()
        source_state = uploaded[0]
        image_state = uploaded[3]
        seed = np.zeros((12, 20), dtype=bool)
        seed[3:8, 4:10] = True
        pvs_state = demo._new_pvs_state()
        pvs_state["instances"][1] = demo._make_inst(
            1,
            "manual_pvs",
            seed,
            [4, 3, 10, 8],
            0.9,
        )
        pvs_state["active_instance_id"] = 1
        pvs_state["next_instance_id"] = 2
        before = copy.deepcopy(pvs_state)
        matched = np.zeros((12, 20), dtype=bool)
        matched[3:8, 12:18] = True
        workflow = {
            "result": {
                "schema_version": 1,
                "match_count": 1,
                "matches": [
                    {
                        "match_id": 1,
                        "score": 0.93,
                        "bbox_xyxy": [12, 3, 18, 8],
                        "translation_xy": [8, 0],
                    }
                ],
                "parameters": {
                    "match_threshold": 0.7,
                    "expand_threshold": 20,
                    "nms_threshold": 0.3,
                },
                "blockers": {
                    "seed_instance_id": 1,
                    "accepted_instance_ids": [],
                },
            },
            "seed_mask_fullres_bool": seed.copy(),
            "match_masks_fullres_bool": [matched],
            "overlay_rgb": np.zeros((12, 20, 3), dtype=np.uint8),
        }

        with mock.patch.object(
            demo._template_matching,
            "run_template_match_workflow",
            return_value=workflow,
        ) as run_workflow, mock.patch.object(
            demo,
            "_predict_inst",
            side_effect=AssertionError("template matching must not call SAM3"),
        ) as predict_inst:
            result = demo._run_template_matching(
                source_state,
                image_state,
                pvs_state,
                demo.MODE_PVS,
                0.7,
                None,
                0.3,
            )

        template_state, preview, zip_path, status = result
        run_workflow.assert_called_once()
        self.assertEqual(run_workflow.call_args.kwargs["expand_threshold"], 20)
        predict_inst.assert_not_called()
        self.assertEqual(template_state["active_instance_id"], 1)
        self.assertEqual(template_state["result"]["match_count"], 1)
        self.assertIsInstance(preview, Image.Image)
        self.assertIn("1 matches", status)

        archive_path = Path(zip_path)
        self.assertTrue(archive_path.is_file())
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            self.assertTrue(
                {
                    "original_image.png",
                    "seed_mask.png",
                    "template_match_overlay.png",
                    "matches.json",
                    "masks/match_0001.png",
                }.issubset(names)
            )
            manifest = json.loads(archive.read("matches.json").decode("utf-8"))
        self.assertEqual(manifest["match_count"], 1)
        self.assertEqual(
            manifest["matches"][0]["mask_file"],
            "masks/match_0001.png",
        )
        self.assertEqual(
            manifest["source_image"]["image_id"],
            source_state["source_image_id"],
        )
        self.assertEqual(
            manifest["workspace"]["crop_bbox_xyxy"],
            [0, 0, 20, 12],
        )

        self.assertEqual(
            pvs_state["active_instance_id"],
            before["active_instance_id"],
        )
        self.assertEqual(pvs_state["next_instance_id"], before["next_instance_id"])
        self.assertEqual(set(pvs_state["instances"]), set(before["instances"]))
        current_instance = pvs_state["instances"][1]
        previous_instance = before["instances"][1]
        np.testing.assert_array_equal(
            current_instance["mask_fullres_bool"],
            previous_instance["mask_fullres_bool"],
        )
        for key in (
            "id",
            "source",
            "box_xyxy_px",
            "score",
            "status",
            "prompt_history",
        ):
            self.assertEqual(current_instance[key], previous_instance[key])

    def test_create_demo_exposes_image_prepost_and_removes_video(self):
        app = demo.create_demo()
        config = app.config
        components = config["components"]
        by_elem_id = {
            component.get("props", {}).get("elem_id"): component
            for component in components
            if component.get("props", {}).get("elem_id")
        }
        source_preview = by_elem_id["source_input_image"]
        template_preview = by_elem_id["template_match_preview"]
        self.assertEqual(source_preview["type"], "image")
        self.assertEqual(source_preview["props"]["height"], 320)
        self.assertEqual(template_preview["props"]["height"], 320)
        self.assertIn(
            "aligned-prepost-preview",
            source_preview["props"]["elem_classes"],
        )
        self.assertIn(
            "aligned-prepost-preview",
            template_preview["props"]["elem_classes"],
        )

        prepost_row = by_elem_id["image_prepost_row"]
        source_column = by_elem_id["source_prepost_column"]
        template_column = by_elem_id["template_prepost_column"]
        self.assertTrue(prepost_row["props"]["equal_height"])

        def find_layout_node(node, component_id):
            if node.get("id") == component_id:
                return node
            for child in node.get("children", []):
                match = find_layout_node(child, component_id)
                if match is not None:
                    return match
            return None

        row_layout = find_layout_node(config["layout"], prepost_row["id"])
        self.assertIsNotNone(row_layout)
        self.assertEqual(
            [child.get("id") for child in row_layout.get("children", [])],
            [source_column["id"], template_column["id"]],
        )

        self.assertEqual(by_elem_id["input_image"]["type"], "image")
        self.assertFalse(by_elem_id["input_image"]["props"]["interactive"])

        tab_ids = {
            component.get("props", {}).get("id")
            for component in components
            if component.get("type") == "tabitem"
        }
        self.assertNotIn("tab_video", tab_ids)

        api_names = {
            str(dependency.get("api_name") or "")
            for dependency in config["dependencies"]
        }
        for expected in (
            "_source_upload_workspace",
            "_workspace_gesture_input",
            "_run_template_matching",
        ):
            self.assertIn(expected, api_names)
        self.assertFalse(any("video" in name.lower() for name in api_names))


if __name__ == "__main__":
    unittest.main()
