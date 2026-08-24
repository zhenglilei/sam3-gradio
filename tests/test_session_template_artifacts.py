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


if __name__ == "__main__":
    unittest.main()
