import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3_gradio_demo as demo_module


class WorkspaceTransactionsTest(unittest.TestCase):
    def setUp(self):
        with demo_module._WORKSPACE_CACHE_LOCK:
            self.previous_cache = dict(demo_module._WORKSPACE_CACHE)
            demo_module._WORKSPACE_CACHE.clear()

    def tearDown(self):
        with demo_module._WORKSPACE_CACHE_LOCK:
            demo_module._WORKSPACE_CACHE.clear()
            demo_module._WORKSPACE_CACHE.update(self.previous_cache)

    @staticmethod
    def _workspace_entry(image_id, session_id, color):
        image = Image.new("RGB", (12, 10), color)
        target_hash = demo_module._layout_tx.image_pixel_sha256(image)
        image_state = {
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
            "session_id": session_id,
            "target_image_sha256": target_hash,
        }
        workspace = {
            "image": image,
            "base_state": {"owner": session_id},
            "session_id": session_id,
            "target_image_sha256": target_hash,
        }
        return image_state, workspace

    def test_new_upload_only_invalidates_the_current_session(self):
        state_a, workspace_a = self._workspace_entry(
            "old-a",
            "session-a",
            (10, 20, 30),
        )
        state_b, workspace_b = self._workspace_entry(
            "image-b",
            "session-b",
            (40, 50, 60),
        )
        with demo_module._WORKSPACE_CACHE_LOCK:
            demo_module._WORKSPACE_CACHE.update(
                {
                    state_a["image_id"]: workspace_a,
                    state_b["image_id"]: workspace_b,
                }
            )

        predictor = mock.Mock()
        predictor.set_image.return_value = {
            "original_height": 10,
            "original_width": 12,
            "backbone_out": {},
        }
        input_image = np.full((10, 12, 3), 90, dtype=np.uint8)
        with mock.patch.object(demo_module, "image_predictor", predictor):
            result = demo_module._init_workspace(
                input_image,
                demo_module.MODE_PVS,
                {"session_id": "session-a"},
            )

        new_state = result[0]
        with demo_module._WORKSPACE_CACHE_LOCK:
            cache_snapshot = dict(demo_module._WORKSPACE_CACHE)
        self.assertNotIn("old-a", cache_snapshot)
        self.assertIn("image-b", cache_snapshot)
        self.assertIn(new_state["image_id"], cache_snapshot)
        self.assertEqual(
            cache_snapshot[new_state["image_id"]]["session_id"],
            "session-a",
        )
        self.assertIs(demo_module._workspace(state_b), workspace_b)

        forged_state = dict(state_b)
        forged_state["session_id"] = "session-a"
        with self.assertRaisesRegex(ValueError, "current session"):
            demo_module._workspace(forged_state)

    def test_pvs_batch_commits_only_after_every_prediction_succeeds(self):
        pvs_state = demo_module._new_pvs_state()
        pvs_state["instances"] = {7: {"id": 7, "sentinel": True}}
        pvs_state["next_instance_id"] = 8
        pvs_state["pending_bbox_records"] = [
            {"id": 1, "box": [1, 1, 5, 5]},
            {"id": 2, "box": [6, 1, 10, 5]},
        ]
        pvs_state["pending_boxes"] = [
            [1, 1, 5, 5],
            [6, 1, 10, 5],
        ]
        original_state = copy.deepcopy(pvs_state)
        mask = np.zeros((8, 12), dtype=bool)
        mask[1:5, 1:5] = True
        prediction = {
            "masks": np.asarray([mask]),
            "scores": np.asarray([0.9], dtype=np.float32),
            "lowres_logits": np.ones((1, 4, 4), dtype=np.float32),
        }

        with (
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module,
                "_predict_inst",
                side_effect=[prediction, RuntimeError("second bbox failed")],
            ),
            mock.patch.object(
                demo_module,
                "_view",
                return_value=(None,) * 8,
            ) as failed_view,
        ):
            failed = demo_module._create_pvs_from_pending_boxes(
                {"image_id": "unused"},
                demo_module._new_pcs_state(),
                pvs_state,
                demo_module.MODE_PVS,
                progress=None,
            )

        self.assertIs(failed[0], pvs_state)
        self.assertEqual(pvs_state, original_state)
        self.assertIn("second bbox failed", failed_view.call_args.args[4])

        with (
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module,
                "_predict_inst",
                side_effect=[prediction, prediction],
            ),
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            succeeded = demo_module._create_pvs_from_pending_boxes(
                {"image_id": "unused"},
                demo_module._new_pcs_state(),
                pvs_state,
                demo_module.MODE_PVS,
                progress=None,
            )

        self.assertIs(succeeded[0], pvs_state)
        self.assertEqual(sorted(pvs_state["instances"]), [7, 8, 9])
        self.assertTrue(pvs_state["instances"][7]["sentinel"])
        self.assertEqual(pvs_state["next_instance_id"], 10)
        self.assertEqual(pvs_state["active_instance_id"], 9)
        self.assertEqual(pvs_state["pending_bbox_records"], [])
        self.assertEqual(pvs_state["pending_boxes"], [])


if __name__ == "__main__":
    unittest.main()
