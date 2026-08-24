from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_demo import feedback_export as module
from sam3_demo import segmentation_evaluation as legacy_export


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class SessionArtifactIsolationTest(unittest.TestCase):
    SESSION_A = "a" * 32
    SESSION_B = "b" * 32

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.feedback_root = root / "feedback"
        self.export_root = root / "exports"
        self.public_root = root / "public"
        self.images = {
            self.SESSION_A: Image.new("RGB", (10, 8), (20, 30, 40)),
            self.SESSION_B: Image.new("RGB", (10, 8), (50, 60, 70)),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _image_state(self, session_id):
        return {
            "session_id": session_id,
            "image_id": f"image-{session_id[:4]}",
            "target_image_sha256": "hash-" + session_id[:4],
        }

    @staticmethod
    def _instance(instance_id=1):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:7] = True
        return {
            "id": instance_id,
            "source": "pcs",
            "status": "draft",
            "score": 0.9,
            "box_xyxy_px": [3, 2, 7, 6],
            "mask_fullres_bool": mask,
            "prompt_history": [],
        }

    def _workspace(self, image_state):
        session_id = image_state["session_id"]
        return {
            "image": self.images[session_id],
            "session_id": session_id,
        }

    def _feedback_deps(self):
        return {
            "_FEEDBACK_WRITE_LOCK": threading.Lock(),
            "_active_instances": lambda pool: list(pool.get("instances", [])),
            "_history_json": lambda history: [],
            "_is_pcs_mode": lambda mode: mode == "PCS Auto",
            "_is_pvs_pool_mode": lambda mode: mode == "PVS Manual",
            "_latest_layout_prompt_from_instances": lambda instances: None,
            "_result_image": lambda *args: Image.new("RGB", (10, 8), (1, 2, 3)),
            "_view": lambda image_state, pcs_state, pvs_state, mode, info: info,
            "_workspace": self._workspace,
            "_write_feedback_layout_artifacts": lambda sample_dir, prompt: {},
            "runtime_feedback_dir": self.feedback_root,
        }

    def _export_deps(self):
        return {
            "_active_instances": lambda pool: list(pool.get("instances", [])),
            "_history_json": lambda history: [],
            "_overlay": lambda *args: Image.new("RGB", (10, 8), (4, 5, 6)),
            "_publish_segmentation_zip": lambda export_dir, zip_name, session_id=None: str(
                self.public_root / zip_name
            ),
            "_workspace": self._workspace,
            "compare_with_coco": lambda *args: {"summary_lines": []},
            "create_prediction_coco_json": lambda *args, **kwargs: {"images": []},
            "mask_to_polygons": lambda mask: [],
            "runtime_export_dir": self.export_root,
        }

    def test_feedback_is_scoped_for_pvs_and_pcs(self):
        pvs_state = {
            "active_instance_id": 1,
            "instances": {1: self._instance()},
        }
        result_a = module._submit_feedback_impl(
            self._feedback_deps(),
            self._image_state(self.SESSION_A),
            {},
            pvs_state,
            "PVS Manual",
            1,
            ["quality"],
            "a",
        )
        pcs_state = {"instances": [self._instance()]}
        result_b = module._submit_feedback_impl(
            self._feedback_deps(),
            self._image_state(self.SESSION_B),
            pcs_state,
            {},
            "PCS Auto",
            0,
            ["quality"],
            "b",
        )

        self.assertIsInstance(result_a, str)
        self.assertIsInstance(result_b, str)
        for session_id in (self.SESSION_A, self.SESSION_B):
            session_root = self.feedback_root / session_id
            self.assertTrue((session_root / "feedback.jsonl").is_file())
            samples = list((session_root / "samples").glob("*/feedback.json"))
            self.assertEqual(len(samples), 1)
            payload = json.loads(samples[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["session_id"], session_id)
            self.assertTrue(str(samples[0]).startswith(str(session_root)))
        self.assertFalse((self.feedback_root / "feedback.jsonl").exists())
        self.assertFalse((self.feedback_root / "samples").exists())

    def test_pcs_and_pvs_exports_are_scoped(self):
        for session_id, pool_name in (
            (self.SESSION_A, "pcs"),
            (self.SESSION_B, "pvs"),
        ):
            state = {"instances": [self._instance()]}
            image_state = self._image_state(session_id)
            path, info = module._export_pool_impl(
                self._export_deps(),
                image_state,
                state,
                state,
                "PCS Auto" if pool_name == "pcs" else "PVS Manual",
                pool_name,
                "GE1_coco",
                "",
                "auto",
                "overlap",
                None,
            )
            self.assertIn("Exported 1", info)
            self.assertTrue(path.endswith(".zip"))
            session_exports = list((self.export_root / session_id).iterdir())
            self.assertEqual(len(session_exports), 1)
            prediction = json.loads(
                (session_exports[0] / "prediction.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["session_id"], session_id)
            self.assertEqual(prediction["pool"], pool_name)

        self.assertEqual(
            {item.name for item in self.export_root.iterdir()},
            {self.SESSION_A, self.SESSION_B},
        )

    def test_feedback_rejects_foreign_layout_prompt_before_writing(self):
        deps = self._feedback_deps()
        deps["_latest_layout_prompt_from_instances"] = lambda instances: {
            "session_id": self.SESSION_B,
            "layout_id": "layout-b",
        }
        result = module._submit_feedback_impl(
            deps,
            self._image_state(self.SESSION_A),
            {},
            {"active_instance_id": 1, "instances": {1: self._instance()}},
            "PVS Manual",
            1,
            [],
            "",
        )
        self.assertIn("another server session", result)
        self.assertFalse(self.feedback_root.exists())

    def test_legacy_export_requires_and_scopes_server_session(self):
        image = Image.new("RGB", (10, 8), (20, 30, 40))
        with self.assertRaisesRegex(ValueError, "server session"):
            legacy_export.create_segmentation_export(
                image,
                image,
                {},
                {},
                "GE1_coco",
                "",
                "auto",
                "overlap",
            )

        state = {
            "session_id": self.SESSION_A,
            "masks": _FakeTensor(np.zeros((1, 8, 10), dtype=np.uint8)),
            "boxes": _FakeTensor(np.asarray([[1, 1, 5, 5]], dtype=np.float32)),
            "scores": _FakeTensor(np.asarray([0.9], dtype=np.float32)),
        }
        published = []

        def publish(export_dir, zip_name, session_id):
            published.append((Path(export_dir), zip_name, session_id))
            return self.public_root / session_id / zip_name

        with (
            mock.patch.object(legacy_export, "runtime_export_dir", self.export_root),
            mock.patch.object(
                legacy_export,
                "compare_with_coco",
                return_value={"summary_lines": []},
            ),
            mock.patch.object(legacy_export, "_publish_segmentation_zip", side_effect=publish),
        ):
            path, _ = legacy_export.create_segmentation_export(
                image,
                image,
                state,
                {},
                "GE1_coco",
                "",
                "auto",
                "overlap",
            )

        self.assertEqual(len(published), 1)
        export_dir, _, published_session = published[0]
        self.assertEqual(export_dir.parent, self.export_root / self.SESSION_A)
        self.assertEqual(published_session, self.SESSION_A)
        self.assertTrue(path.startswith(str(self.public_root / self.SESSION_A)))
        payload = json.loads(
            (export_dir / "prediction.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["session_id"], self.SESSION_A)

    def test_invalid_or_mismatched_session_never_creates_artifact(self):
        deps = self._export_deps()
        invalid = self._image_state("../escape")
        invalid["session_id"] = "../escape"
        path, info = module._export_pool_impl(
            deps,
            invalid,
            {"instances": [self._instance()]},
            {},
            "PCS Auto",
            "pcs",
            "GE1_coco",
            "",
            "auto",
            "overlap",
            None,
        )
        self.assertIsNone(path)
        self.assertIn("Export failed", info)
        self.assertFalse(self.export_root.exists())

        mismatched = self._image_state(self.SESSION_A)
        original_workspace = self._feedback_deps()["_workspace"]
        feedback_deps = self._feedback_deps()
        feedback_deps["_workspace"] = lambda state: {
            "image": original_workspace(state)["image"],
            "session_id": self.SESSION_B,
        }
        feedback = module._submit_feedback_impl(
            feedback_deps,
            mismatched,
            {},
            {"active_instance_id": 1, "instances": {1: self._instance()}},
            "PVS Manual",
            1,
            [],
            "",
        )
        self.assertIsInstance(feedback, str)
        self.assertFalse(self.feedback_root.exists())


if __name__ == "__main__":
    unittest.main()
