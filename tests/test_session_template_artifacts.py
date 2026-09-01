from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_demo import image_prepost_callbacks as callbacks


def _states(session_id="a" * 32):
    source_state = {
        "session_id": session_id,
        "source_image_id": "source-1",
        "source_image_sha256": "source-hash",
        "source_width": 8,
        "source_height": 6,
        "crop_bbox_xyxy": [0, 0, 8, 6],
        "workspace_image_id": "workspace-1",
        "workspace_hash": "workspace-hash",
    }
    image_state = {
        "session_id": session_id,
        "image_id": "workspace-1",
        "target_image_sha256": "workspace-hash",
        "source_image_id": "source-1",
        "source_image_sha256": "source-hash",
        "crop_bbox_xyxy": [0, 0, 8, 6],
    }
    pvs_state = {
        "session_id": session_id,
        "active_instance_id": 1,
        "instances": {1: {"id": 1}},
    }
    return source_state, image_state, pvs_state


def _workflow():
    seed = np.zeros((6, 8), dtype=bool)
    seed[1:3, 1:3] = True
    match = np.zeros((6, 8), dtype=bool)
    match[1:3, 5:7] = True
    return {
        "result": {
            "schema_version": 1,
            "match_count": 1,
            "matches": [
                {
                    "match_id": 1,
                    "score": 0.9,
                    "bbox_xyxy": [5, 1, 7, 3],
                }
            ],
        },
        "seed_mask_fullres_bool": seed,
        "match_masks_fullres_bool": [match],
        "overlay_rgb": np.zeros((6, 8, 3), dtype=np.uint8),
    }


class SessionTemplateArtifactTests(unittest.TestCase):
    def test_clear_internal_path_can_retain_session_owner(self):
        deps = {
            "_new_template_match_state": lambda session_id=None: {
                "session_id": str(session_id or "")
            }
        }
        result = callbacks._clear_template_match_outputs_impl(
            deps, "failed", session_id="a" * 32
        )
        self.assertEqual(result[0]["session_id"], "a" * 32)
        self.assertEqual(result[3], "failed")

    def test_export_is_scoped_to_validated_session(self):
        source_state, image_state, _ = _states()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def publish(export_dir, zip_name, session_id=None):
                calls.append((export_dir, zip_name, session_id))
                return export_dir / zip_name

            zip_path, manifest = callbacks._publish_template_match_export_impl(
                {
                    "_publish_segmentation_zip": publish,
                    "runtime_export_dir": root,
                },
                Image.new("RGB", (8, 6), "white"),
                _workflow(),
                source_state,
                image_state,
            )

            self.assertEqual(len(calls), 1)
            export_dir, _, session_id = calls[0]
            self.assertEqual(session_id, "a" * 32)
            self.assertEqual(export_dir.parent, root / ("a" * 32))
            self.assertTrue(export_dir.is_dir())
            self.assertTrue((export_dir / "matches.json").is_file())
            self.assertEqual(Path(zip_path), export_dir / zip_path.name)
            self.assertEqual(manifest["source_image"]["image_id"], "source-1")

    def test_export_rejects_mismatched_or_unsafe_session_without_write(self):
        source_state, image_state, _ = _states()
        image_state["session_id"] = "b" * 32
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "session_id must match"):
                callbacks._publish_template_match_export_impl(
                    {
                        "_publish_segmentation_zip": lambda *_: None,
                        "runtime_export_dir": root,
                    },
                    Image.new("RGB", (8, 6), "white"),
                    _workflow(),
                    source_state,
                    image_state,
                )
            self.assertEqual(list(root.iterdir()), [])

        source_state, image_state, _ = _states("../escape")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "invalid session_id"):
                callbacks._publish_template_match_export_impl(
                    {
                        "_publish_segmentation_zip": lambda *_: None,
                        "runtime_export_dir": Path(tmp),
                    },
                    Image.new("RGB", (8, 6), "white"),
                    _workflow(),
                    source_state,
                    image_state,
                )

    def test_run_rejects_cross_session_and_preserves_owner_on_error(self):
        source_state, image_state, pvs_state = _states()
        pvs_state["session_id"] = "b" * 32
        clear_calls = []

        def clear(state, status):
            clear_calls.append((state, status))
            return dict(state or {}), None, None, status

        workflow_calls = []
        result = callbacks._run_template_matching_impl(
            {
                "_clear_template_match_outputs": clear,
                "_is_pvs_pool_mode": lambda mode: True,
                "_publish_template_match_export": lambda *args: (_ for _ in ()).throw(
                    AssertionError("must not export")
                ),
                "_source_image_cache_get": lambda state: (_ for _ in ()).throw(
                    AssertionError("must not load source")
                ),
                "_workspace": lambda state: (_ for _ in ()).throw(
                    AssertionError("must not load workspace")
                ),
            },
            source_state,
            image_state,
            pvs_state,
            "PVS Manual",
            0.7,
            20,
            0.3,
        )

        self.assertEqual(len(clear_calls), 1)
        self.assertEqual(clear_calls[0][0]["session_id"], "a" * 32)
        self.assertEqual(result[0]["session_id"], "a" * 32)
        self.assertIn("session_id", result[3])
        self.assertEqual(workflow_calls, [])

    def test_run_internal_failure_preserves_valid_session(self):
        source_state, image_state, pvs_state = _states()
        result = callbacks._run_template_matching_impl(
            {
                "_clear_template_match_outputs": lambda state, status: (
                    dict(state or {}),
                    None,
                    None,
                    status,
                ),
                "_is_pvs_pool_mode": lambda mode: True,
                "_publish_template_match_export": lambda *args: None,
                "_source_image_cache_get": lambda state: None,
                "_workspace": lambda state: (_ for _ in ()).throw(
                    RuntimeError("workspace expired")
                ),
            },
            source_state,
            image_state,
            pvs_state,
            "PVS Manual",
            0.7,
            20,
            0.3,
        )
        self.assertEqual(result[0]["session_id"], "a" * 32)
        self.assertIn("workspace expired", result[3])

    def test_run_requires_pvs_session_identity(self):
        source_state, image_state, pvs_state = _states()
        pvs_state.pop("session_id")
        result = callbacks._run_template_matching_impl(
            {
                "_clear_template_match_outputs": lambda state, status: (
                    dict(state or {}),
                    None,
                    None,
                    status,
                ),
                "_is_pvs_pool_mode": lambda mode: True,
                "_publish_template_match_export": lambda *args: None,
                "_source_image_cache_get": lambda state: None,
                "_workspace": lambda state: None,
            },
            source_state,
            image_state,
            pvs_state,
            "PVS Manual",
            0.7,
            20,
            0.3,
        )
        self.assertEqual(result[0]["session_id"], "a" * 32)
        self.assertIn("pvs_state session_id is missing", result[3])

    def test_template_instance_preview_is_independent_from_active_editor_instance(self):
        source_state, image_state, pvs_state = _states()
        first = np.zeros((6, 8), dtype=bool)
        first[1:3, 1:3] = True
        second = np.zeros((6, 8), dtype=bool)
        second[2:5, 4:7] = True
        pvs_state["instances"] = {
            1: {"id": 1, "status": "draft", "mask_fullres_bool": first},
            2: {"id": 2, "status": "accepted", "mask_fullres_bool": second},
        }

        result = callbacks._preview_template_instance_impl(
            {
                "_clear_template_match_outputs": lambda state, status: (
                    dict(state or {}), None, None, status
                ),
                "_new_template_match_state": lambda session_id=None: {
                    "session_id": str(session_id or ""),
                    "result": None,
                },
                "_source_image_cache_get": lambda state: Image.new("RGB", (8, 6), "white"),
            },
            source_state,
            image_state,
            pvs_state,
            "2",
        )

        next_pvs, template_state, preview, download, status = result
        self.assertEqual(next_pvs["active_instance_id"], 1)
        self.assertEqual(next_pvs["template_match_instance_id"], 2)
        self.assertEqual(template_state["active_instance_id"], 2)
        self.assertIsInstance(preview, Image.Image)
        self.assertIsNone(download)
        self.assertIn("PVS #2", status)

    def test_export_selected_matches_rebuilds_only_requested_masks(self):
        source_state, image_state, pvs_state = _states()
        seed = np.zeros((6, 8), dtype=bool)
        seed[1:3, 1:3] = True
        pvs_state["instances"] = {
            1: {"id": 1, "status": "draft", "mask_fullres_bool": seed},
        }
        template_state = {
            "session_id": "a" * 32,
            "source_image_id": "source-1",
            "workspace_image_id": "workspace-1",
            "active_instance_id": 1,
            "result": {
                "schema_version": 1,
                "match_count": 2,
                "matches": [
                    {"match_id": 1, "score": 0.91, "translation_xy": [4, 0]},
                    {"match_id": 2, "score": 0.89, "translation_xy": [0, 3]},
                ],
            },
        }
        published = []

        def publish(source, workflow, source_state_arg, image_state_arg):
            published.append(workflow)
            return Path("selected.zip"), workflow["result"]

        download, status = callbacks._export_template_match_selection_impl(
            {
                "_publish_template_match_export": publish,
                "_source_image_cache_get": lambda state: Image.new("RGB", (8, 6), "white"),
            },
            source_state,
            image_state,
            pvs_state,
            template_state,
            "selected",
            ["2"],
        )

        self.assertEqual(download, "selected.zip")
        self.assertIn("1 个匹配实例", status)
        self.assertEqual(len(published), 1)
        workflow = published[0]
        self.assertEqual(workflow["result"]["match_count"], 1)
        self.assertEqual(workflow["result"]["matches"][0]["match_id"], 2)
        self.assertEqual(workflow["result"]["selection"]["selected_match_ids"], [2])
        self.assertEqual(len(workflow["match_masks_fullres_bool"]), 1)
        expected = np.zeros((6, 8), dtype=bool)
        expected[4:6, 1:3] = True
        np.testing.assert_array_equal(workflow["match_masks_fullres_bool"][0], expected)

    def test_export_selected_matches_rejects_unknown_match(self):
        source_state, image_state, pvs_state = _states()
        pvs_state["instances"][1].update(
            {
                "status": "draft",
                "mask_fullres_bool": np.ones((6, 8), dtype=bool),
            }
        )
        template_state = {
            "session_id": "a" * 32,
            "source_image_id": "source-1",
            "workspace_image_id": "workspace-1",
            "active_instance_id": 1,
            "result": {"matches": [{"match_id": 1, "translation_xy": [0, 0]}]},
        }
        download, status = callbacks._export_template_match_selection_impl(
            {
                "_publish_template_match_export": lambda *args: (_ for _ in ()).throw(
                    AssertionError("must not publish")
                ),
                "_source_image_cache_get": lambda state: Image.new("RGB", (8, 6), "white"),
            },
            source_state,
            image_state,
            pvs_state,
            template_state,
            "selected",
            ["99"],
        )
        self.assertIsNone(download)
        self.assertIn("M99", status)


if __name__ == "__main__":
    unittest.main()
