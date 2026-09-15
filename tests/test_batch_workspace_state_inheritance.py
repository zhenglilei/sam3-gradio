import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import app, batch_workspace as batch
from sam3_demo.stitch_callbacks import new_stitch_state


class BatchWorkspaceStateInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime_patch = mock.patch.object(app, "runtime_dir", Path(self.tmp.name))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

        self.session = {"session_id": "a" * 32}
        self.state = batch.owned_batch({}, self.session)
        self.state["items"] = [
            {
                "id": f"tile-{index}",
                "name": f"tile-{index}.png",
                "original": Image.new("RGB", (24, 20), color),
                "snapshot": None,
                "status": "pending",
                "error": "",
            }
            for index, color in enumerate(("white", "green"))
        ]
        self.state["active_id"] = None

    def _restore(self, index):
        return list(batch.restore_item(
            app, self.state, self.state["items"][index], self.session
        ))

    @staticmethod
    def _config(mode="PCS Auto", tool="polygon", text="shared", threshold=0.73):
        return {
            "mode": mode,
            "tool": tool,
            "text": text,
            "threshold": threshold,
        }

    def test_next_unprocessed_item_inherits_only_reusable_config(self):
        first_values = self._restore(0)
        config = self._config()
        restored = batch.restore_item(
            app, self.state, self.state["items"][1], self.session,
            inherit_config=config,
        )

        self.assertEqual(restored[6:10], (
            config["mode"], config["tool"], config["text"], config["threshold"]
        ))
        self.assertIsNone(self.state["items"][1]["snapshot"])

    def test_next_unprocessed_item_drops_masks_instances_and_geometry(self):
        first_values = self._restore(0)
        mask = np.ones((20, 24), dtype=bool)
        first_values[2]["positive_boxes"] = [[1, 2, 10, 12]]
        first_values[2]["negative_boxes"] = [[12, 10, 20, 18]]
        first_values[2]["instances"] = {
            1: {"mask_fullres_bool": mask, "category_name": "old"}
        }
        first_values[3]["pending_boxes"] = [[2, 3, 9, 11]]
        first_values[3]["instances"] = {
            2: {"mask_fullres_bool": mask, "prompt_history": [{"op": "old"}]}
        }
        first_values[4]["polygon_points"] = [[3, 4], [8, 9]]
        first_values[5] = {"layout_id": "old-layout", "tx": 99.0}

        restored = batch.restore_item(
            app, self.state, self.state["items"][1], self.session,
            inherit_config=self._config(),
        )

        self.assertEqual(restored[2]["positive_boxes"], [])
        self.assertEqual(restored[2]["negative_boxes"], [])
        self.assertEqual(restored[2]["instances"], {})
        self.assertEqual(restored[3]["pending_boxes"], [])
        self.assertEqual(restored[3]["instances"], {})
        self.assertEqual(restored[4]["polygon_points"], [])
        self.assertEqual(restored[5], app._new_layout_state(self.session["session_id"]))
        self.assertNotEqual(restored[0]["source_image_id"], first_values[0]["source_image_id"])
        self.assertNotEqual(restored[1]["image_id"], first_values[1]["image_id"])

    def test_next_with_own_snapshot_keeps_own_state(self):
        second_values = self._restore(1)
        mask = np.zeros((20, 24), dtype=bool)
        mask[4:11, 6:15] = True
        second_values[3]["instances"] = {
            7: {"mask_fullres_bool": mask, "category_name": "own"}
        }
        second_values[4]["polygon_points"] = [[5, 6], [10, 12]]
        second_values[6:10] = ["PVS Manual", "bbox", "own prompt", 0.21]
        self.assertTrue(batch.save_snapshot(
            app, self.state, *second_values
        ))

        first_values = self._restore(0)
        restored = batch.restore_item(
            app, self.state, self.state["items"][1], self.session,
            inherit_config=self._config(
                mode="PCS Auto", tool="point", text="previous", threshold=0.91
            ),
        )

        self.assertEqual(restored[6:10], ("PVS Manual", "bbox", "own prompt", 0.21))
        self.assertEqual(restored[4]["polygon_points"], [[5, 6], [10, 12]])
        np.testing.assert_array_equal(
            restored[3]["instances"][7]["mask_fullres_bool"], mask
        )

    def test_empty_text_prompt_is_inherited(self):
        restored = batch.restore_item(
            app, self.state, self.state["items"][1], self.session,
            inherit_config=self._config(
                mode="PVS Manual", tool="bbox", text="", threshold=0.88
            ),
        )

        self.assertEqual(restored[6], "PVS Manual")
        self.assertEqual(restored[7], "bbox")
        self.assertEqual(restored[8], "")
        self.assertEqual(restored[9], 0.88)

    def test_next_saves_current_snapshot_before_restore(self):
        first_values = self._restore(0)
        first_values[6:10] = ["PCS Auto", "polygon", "shared", 0.67]
        events = []
        real_save = batch.save_snapshot
        real_restore = batch.restore_item

        def save(*args, **kwargs):
            result = real_save(*args, **kwargs)
            events.append(("save", result))
            return result

        def restore(*args, **kwargs):
            events.append(("restore", kwargs.get("inherit_config")))
            return real_restore(*args, **kwargs)

        demo = app.create_demo()
        callback = inspect.unwrap(next(
            fn.fn for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", "") == "batch_next"
        ))
        with (
            mock.patch.object(batch, "save_snapshot", side_effect=save),
            mock.patch.object(batch, "restore_item", side_effect=restore),
        ):
            callback(
                self.state,
                self.session,
                *first_values,
                [],
                None,
                new_stitch_state(self.session["session_id"]),
            )

        self.assertEqual(events, [
            ("save", True),
            ("restore", {
                "mode": "PCS Auto", "tool": "polygon",
                "text": "shared", "threshold": 0.67,
            }),
        ])
        self.assertIsNotNone(self.state["items"][0]["snapshot"])


    def test_first_selection_without_active_item_restores_target(self):
        values = self._restore(0)
        self.state["active_id"] = None
        demo = app.create_demo()
        callback = inspect.unwrap(next(
            fn.fn for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", "") == "batch_select"
        ))

        callback(
            self.state,
            self.session,
            *values,
            [],
            None,
            new_stitch_state(self.session["session_id"]),
            type("SelectEvent", (), {"index": 0})(),
        )

        self.assertEqual(self.state["active_id"], "tile-0")

    def test_failed_save_blocks_switch_when_active_item_exists(self):
        values = self._restore(0)
        demo = app.create_demo()
        callback = inspect.unwrap(next(
            fn.fn for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", "") == "batch_next"
        ))

        with (
            mock.patch.object(batch, "save_snapshot", return_value=False),
            mock.patch.object(batch, "restore_item") as restore,
        ):
            callback(
                self.state,
                self.session,
                *values,
                [],
                None,
                new_stitch_state(self.session["session_id"]),
            )

        restore.assert_not_called()
        self.assertEqual(self.state["active_id"], "tile-0")

if __name__ == "__main__":
    unittest.main()
