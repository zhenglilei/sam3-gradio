import unittest
import numpy as np
from PIL import Image
from sam3_demo import annotated_stitch as queue
from sam3_demo import stitch_callbacks as cb


class AnnotatedQueueTest(unittest.TestCase):
    def tile(self, name="one"):
        mask = np.zeros((20, 30), bool)
        mask[2:18, 3:27] = True
        mask[7:12, 9:15] = False
        return {"tile_id": name, "name": name, "image": Image.new("RGB", (30, 20), "white"),
                "instances": [{"id": 4, "category_name": "object", "mask": mask}], "provenance": {}}

    def test_order_and_remove_preserve_masks_and_ids(self):
        a, b = self.tile("a"), self.tile("b")
        state = dict(cb.new_stitch_state(), saved_tiles=[a, b], annotated_mode=True)
        state = queue.edit_queue(state, "a", "down")
        self.assertEqual([tile["tile_id"] for tile in state["saved_tiles"]], ["b", "a"])
        self.assertIs(state["saved_tiles"][1]["instances"][0]["mask"], a["instances"][0]["mask"])
        self.assertTrue(state["queue_dirty"])
        self.assertIsNone(state["generated_revision"])
        state = queue.edit_queue(state, "b", "remove")
        self.assertEqual([tile["tile_id"] for tile in state["saved_tiles"]], ["a"])

    def test_manual_and_black_crop_share_coordinates(self):
        tile = self.tile()
        annotations, records = queue.prepare_annotations([tile],
            {"top": 1, "bottom": 2, "left": 3, "right": 4},
            [{"trim": {"top": 2, "bottom": 1, "left": 1, "right": 2}}])
        np.testing.assert_array_equal(annotations[0][0]["mask"], tile["instances"][0]["mask"][3:17, 4:24])
        self.assertEqual(records[0]["source_crop_xyxy"], [4, 3, 24, 17])

    def test_loading_images_does_not_erase_saved_queue(self):
        tile = self.tile()
        state = dict(cb.new_stitch_state(), saved_tiles=[tile])
        result = cb.load_tiles(None, "horizontal", state, 1, False, True, True, False,
                               False, 1, 2, 3, 4, source_tiles=[tile])
        loaded = result[0]
        self.assertEqual(loaded["saved_tiles"][0]["tile_id"], "one")
        self.assertEqual(loaded["images"][0].size, (23, 17))
        self.assertEqual(loaded["annotations"][0][0]["mask"].shape, (17, 23))
        plain = cb._fresh_owned_state(loaded)
        self.assertEqual(len(plain["saved_tiles"]), 1)
        self.assertFalse(plain.get("annotated_mode", False))

    def test_preview_does_not_modify_authoritative_image_or_mask(self):
        tile = self.tile()
        pixels = np.asarray(tile["image"]).copy()
        mask = tile["instances"][0]["mask"].copy()
        state = {"annotations": [tile["instances"]], "annotation_visible": True, "annotation_alpha": .5}
        shown = queue.preview_images(state, [tile["image"]])
        self.assertFalse(np.array_equal(np.asarray(shown[0]), pixels))
        np.testing.assert_array_equal(np.asarray(tile["image"]), pixels)
        np.testing.assert_array_equal(tile["instances"][0]["mask"], mask)

    def test_geometry_change_clears_old_annotations(self):
        state = dict(cb.new_stitch_state(), mosaic_instances=[{"old": True}], annotation_manifest={"old": True})
        cb._invalidate_mosaic(state)
        self.assertEqual(state["mosaic_instances"], [])
        self.assertEqual(state["annotation_manifest"], {})
