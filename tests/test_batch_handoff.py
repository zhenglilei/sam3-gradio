import inspect
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import app, batch_workspace as batch
from sam3_demo.annotated_stitch_io import make_tile
from sam3_demo.stitch_callbacks import new_stitch_state


class BatchHandoffTests(unittest.TestCase):
    session_id = "a" * 32

    def setUp(self):
        self.image = Image.new("RGB", (24, 20), "white")
        self.live_mask = np.zeros((20, 24), dtype=bool)
        self.live_mask[3:17, 5:20] = True
        self.live_mask[8:12, 10:15] = False
        self.live_values = self._live_values(
            self.live_mask, image_id="image-live", source_id="source-live"
        )

    def _source(self, source_id):
        return {
            "session_id": self.session_id,
            "source_image_id": source_id,
            "source_image_sha256": f"hash-{source_id}",
            "source_width": self.image.width,
            "source_height": self.image.height,
            "crop_bbox_xyxy": [0, 0, self.image.width, self.image.height],
        }

    def _pool(self, mask, *, category="scratch", instance_id=1):
        return {
            "text_prompt": category,
            "instances": {
                instance_id: {
                    "id": instance_id,
                    "status": "active",
                    "mask_fullres_bool": mask,
                    "pcs_fullres_prob": np.full(mask.shape, 0.25, dtype=np.float16),
                    "score": 0.9,
                    "category_name": category,
                    "prompt_history": [{"op": "test"}],
                }
            },
        }

    def _live_values(self, mask, *, image_id, source_id, category="scratch"):
        source = self._source(source_id)
        image_state = {
            "session_id": self.session_id,
            "image_id": image_id,
            "target_image_sha256": f"target-{image_id}",
        }
        return (
            source,
            image_state,
            self._pool(mask, category=category),
            {"instances": {}},
            {},
            {},
            "PCS Auto",
            "bbox",
            "",
            0.4,
        )

    def _item(self, item_id, *, snapshot=None, status="pending",
              image_id="image-live", source_id="source-live"):
        return {
            "id": item_id,
            "name": f"{item_id}.png",
            "original": self.image,
            "snapshot": snapshot,
            "status": status,
            "error": "",
            "live_image_id": image_id,
            "live_source_id": source_id,
        }

    def _batch(self, *items, selected=None):
        return {
            "session_id": self.session_id,
            "items": list(items),
            "active_id": items[0]["id"] if items else None,
            "selected_ids": list(selected or []),
        }

    def _snapshot(self, mask, *, image_id, source_id, category="saved"):
        return {
            "source": self._source(source_id),
            "image": self.image.copy(),
            "pcs": self._pool(mask, category=category),
            "pvs": {"instances": {}},
            "prompt": {},
            "layout": {},
            "mode": "PCS Auto",
            "tool": "bbox",
            "text": "",
            "threshold": 0.4,
        }

    def _send(self, batch_state, selected, stitch_state=None, live_values=None):
        return batch.send_tiles(
            app,
            batch_state,
            selected,
            stitch_state or new_stitch_state(self.session_id),
            live_values=live_values,
        )

    def test_unsaved_live_pcs_mask_wins_and_is_detached(self):
        item = self._item("live")
        batch_state = self._batch(item, selected=["live"])
        live_before = self.live_mask.copy()

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            stitch, count = self._send(
                batch_state, ["live"], live_values=self.live_values
            )

        self.assertEqual(count, 1)
        self.assertEqual([tile["tile_id"] for tile in stitch["saved_tiles"]], ["live"])
        handed_off = stitch["saved_tiles"][0]["instances"][0]["mask"]
        np.testing.assert_array_equal(handed_off, live_before)
        self.assertIsNot(handed_off, self.live_mask)
        self.live_mask[3:17, 5:20] = False
        np.testing.assert_array_equal(handed_off, live_before)
        self.assertIsNone(item["snapshot"])

    def test_selected_snapshot_and_live_tile_keep_holes_and_nonselected_is_untouched(self):
        saved_mask = np.zeros((20, 24), dtype=bool)
        saved_mask[1:8, 2:10] = True
        saved_item = self._item(
            "saved",
            snapshot=self._snapshot(
                saved_mask, image_id="image-saved", source_id="source-saved"
            ),
            image_id="image-saved",
            source_id="source-saved",
        )
        live_item = self._item("live")
        ignored_mask = np.ones((20, 24), dtype=bool)
        ignored_item = self._item(
            "ignored",
            snapshot=self._snapshot(
                ignored_mask, image_id="image-ignored", source_id="source-ignored"
            ),
            image_id="image-ignored",
            source_id="source-ignored",
        )
        ignored_snapshot = ignored_item["snapshot"]
        batch_state = self._batch(
            live_item, saved_item, ignored_item, selected=["live", "saved"]
        )

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            stitch, count = self._send(
                batch_state, ["live", "saved"], live_values=self.live_values
            )

        self.assertEqual(count, 2)
        self.assertEqual(
            [tile["tile_id"] for tile in stitch["saved_tiles"]], ["live", "saved"]
        )
        np.testing.assert_array_equal(
            stitch["saved_tiles"][0]["instances"][0]["mask"], self.live_mask
        )
        np.testing.assert_array_equal(
            stitch["saved_tiles"][1]["instances"][0]["mask"], saved_mask
        )
        self.assertIs(ignored_item["snapshot"], ignored_snapshot)

    def test_stable_id_replaces_existing_tile_without_duplicate(self):
        old_mask = np.zeros((20, 24), dtype=bool)
        old_mask[0:2, 0:2] = True
        existing = make_tile(
            self.image,
            "old.png",
            [{"id": 1, "category_name": "old", "mask": old_mask}],
            tile_id="live",
        )
        stitch = new_stitch_state(self.session_id)
        stitch["saved_tiles"] = [existing]
        item = self._item("live")

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            updated, count = self._send(
                self._batch(item, selected=["live"]),
                ["live"],
                stitch,
                self.live_values,
            )

        self.assertEqual(count, 1)
        self.assertEqual(len(updated["saved_tiles"]), 1)
        self.assertEqual(updated["saved_tiles"][0]["tile_id"], "live")
        self.assertNotEqual(
            updated["saved_tiles"][0]["instances"][0]["category_name"], "old"
        )

    def test_live_source_mismatch_falls_back_to_matching_snapshot(self):
        snapshot_mask = np.zeros((20, 24), dtype=bool)
        snapshot_mask[2:6, 3:9] = True
        item = self._item(
            "tile",
            snapshot=self._snapshot(
                snapshot_mask, image_id="image-live", source_id="source-saved"
            ),
            image_id="image-live",
            source_id="source-saved",
        )
        mismatched = self._live_values(
            np.ones((20, 24), dtype=bool),
            image_id="image-live",
            source_id="source-other",
        )

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            stitch, count = self._send(
                self._batch(item, selected=["tile"]),
                ["tile"],
                live_values=mismatched,
            )

        self.assertEqual(count, 1)
        np.testing.assert_array_equal(
            stitch["saved_tiles"][0]["instances"][0]["mask"], snapshot_mask
        )

    def test_same_source_crop_workspace_image_id_uses_current_mask(self):
        snapshot_mask = np.zeros((20, 24), dtype=bool)
        snapshot_mask[1:5, 2:8] = True
        item = self._item(
            "crop",
            snapshot=self._snapshot(
                snapshot_mask, image_id="image-live", source_id="source-live"
            ),
        )
        live_values = list(self.live_values)
        live_values[0] = dict(
            live_values[0], workspace_image_id="image-cropped"
        )
        live_values[1] = dict(live_values[1], image_id="image-cropped")

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            stitch, count = self._send(
                self._batch(item, selected=["crop"]),
                ["crop"],
                live_values=tuple(live_values),
            )

        self.assertEqual(count, 1)
        np.testing.assert_array_equal(
            stitch["saved_tiles"][0]["instances"][0]["mask"], self.live_mask
        )

    def test_same_source_unknown_image_id_falls_back_to_snapshot(self):
        snapshot_mask = np.zeros((20, 24), dtype=bool)
        snapshot_mask[2:6, 3:9] = True
        item = self._item(
            "unknown",
            snapshot=self._snapshot(
                snapshot_mask, image_id="image-live", source_id="source-live"
            ),
        )
        unknown_live = self._live_values(
            np.ones((20, 24), dtype=bool),
            image_id="image-unknown",
            source_id="source-live",
        )

        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            stitch, count = self._send(
                self._batch(item, selected=["unknown"]),
                ["unknown"],
                live_values=unknown_live,
            )

        self.assertEqual(count, 1)
        np.testing.assert_array_equal(
            stitch["saved_tiles"][0]["instances"][0]["mask"], snapshot_mask
        )

    def test_empty_live_source_id_does_not_match(self):
        item = self._item("empty-source", source_id="")
        empty_source_live = self._live_values(
            np.ones((20, 24), dtype=bool),
            image_id="image-live",
            source_id="",
        )

        with (
            mock.patch.object(app, "_workspace", return_value={"image": self.image}),
            self.assertRaises(ValueError),
        ):
            self._send(
                self._batch(item, selected=["empty-source"]),
                ["empty-source"],
                live_values=empty_source_live,
            )

    def test_empty_selection_has_actionable_failure(self):

        item = self._item("live")
        with self.assertRaises(ValueError) as caught:
            self._send(self._batch(item), [])
        expected = "".join(
            chr(value)
            for value in (
                0x8BF7, 0x52FE, 0x9009, 0x5DF2, 0x6807, 0x6CE8, 0x56FE, 0x7247,
            )
        )
        self.assertIn(expected, str(caught.exception))

    def test_batch_stitch_skips_snapshot_archive_for_live_pcs(self):
        demo = app.create_demo()
        callback = inspect.unwrap(
            next(
                fn.fn for fn in demo.fns.values()
                if getattr(fn.fn, "__name__", "") == "batch_stitch"
            )
        )
        item = self._item("live")
        state = self._batch(item, selected=["live"])
        args = (
            state,
            {"session_id": self.session_id},
            self.live_values[0],
            self.live_values[1],
            self.live_values[2],
            self.live_values[3],
            self.live_values[4],
            self.live_values[5],
            self.live_values[6],
            self.live_values[7],
            self.live_values[8],
            self.live_values[9],
            ["live"],
            None,
            new_stitch_state(self.session_id),
        )
        with (
            mock.patch.object(app, "_workspace", return_value={"image": self.image}),
            mock.patch.object(batch.np, "savez_compressed") as save_archive,
            mock.patch.object(batch.gr, "Warning"),
        ):
            result = callback(*args)

        save_archive.assert_not_called()
        stitch = next(
            value for value in result
            if isinstance(value, dict) and "saved_tiles" in value
        )
        self.assertEqual(len(stitch["saved_tiles"]), 1)
        np.testing.assert_array_equal(
            stitch["saved_tiles"][0]["instances"][0]["mask"], self.live_mask
        )


    def test_batch_stitch_outputs_only_handoff_components_and_selects_stitch_tab(self):
        demo = app.create_demo()
        block_fn = next(
            fn for fn in demo.fns.values()
            if getattr(fn.fn, "__name__", "") == "batch_stitch"
        )
        output_types = [type(component).__name__ for component in block_fn.outputs]
        self.assertEqual(
            output_types,
            ["State", "Markdown", "State", "Gallery", "Dropdown", "Tabs"],
        )
        self.assertNotIn("Image", output_types)
        self.assertNotIn("ImageGestureOverlay", output_types)

        item = self._item("live")
        args = (
            self._batch(item, selected=["live"]),
            {"session_id": self.session_id},
            *self.live_values,
            ["live"],
            None,
            new_stitch_state(self.session_id),
        )
        with mock.patch.object(app, "_workspace", return_value={"image": self.image}):
            result = inspect.unwrap(block_fn.fn)(*args)

        self.assertEqual(len(result), 6)
        self.assertIsInstance(result[-1], dict)
        self.assertEqual(result[-1].get("selected"), "tab_stitch")

if __name__ == "__main__":
    unittest.main()
