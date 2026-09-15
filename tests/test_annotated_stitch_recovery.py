import unittest
import numpy as np
from PIL import Image
from sam3_demo import app, stitch_callbacks as cb


class AnnotatedRecoveryTests(unittest.TestCase):
    def state(self):
        mask = np.ones((20, 30), dtype=bool)
        mask[5:10, 7:12] = False
        tile = {"tile_id": "one", "name": "one.png", "image": Image.new("RGB", (30, 20), "white"),
                "instances": [{"id": 1, "category_name": "object", "mask": mask}], "provenance": {}}
        return dict(cb.new_stitch_state(), saved_tiles=[tile], generated_revision=7)

    def test_empty_import_preserves_queue_and_outputs(self):
        state = self.state()
        result = app._import_stitch_annotations([], state)
        self.assertEqual(len(result), 10)
        self.assertIs(result[0], state)
        self.assertEqual(state["generated_revision"], 7)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[5], {"__type__": "update"})

    def test_invalid_file_preserves_queue(self):
        state = self.state()
        result = app._import_stitch_annotations(["/nonexistent/invalid.json"], state)
        self.assertIs(result[0], state)
        self.assertEqual(len(state["saved_tiles"]), 1)
        self.assertEqual(state["generated_revision"], 7)

    def test_bottom_button_crops_queue_and_does_not_accumulate(self):
        original = self.state()
        state = original
        for _ in range(2):
            state = app._load_stitch_tiles(None, "horizontal", state, 1, False, False,
                                           True, False, False, 0, 8, 0, 0)[0]
            self.assertEqual(state["images"][0].size, (30, 12))
            np.testing.assert_array_equal(state["annotations"][0][0]["mask"],
                                          original["saved_tiles"][0]["instances"][0]["mask"][:12])
            self.assertEqual(state["saved_tiles"][0]["image"].size, (30, 20))
