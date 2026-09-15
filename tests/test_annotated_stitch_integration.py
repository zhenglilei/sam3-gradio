import unittest
from contextlib import nullcontext
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import app
from sam3_demo import stitch_callbacks


def _instance(instance_id, mask, *, category="defect", score=0.8, missing=False):
    item = {
        "id": instance_id,
        "mask_fullres_bool": np.asarray(mask, dtype=bool),
        "category_name": category,
        "source": "test",
        "prompt_history": [],
    }
    if missing:
        item["score_missing"] = True
    else:
        item["score"] = score
    return item


class AnnotatedStitchIntegrationTests(unittest.TestCase):
    def test_save_uses_only_active_mode_pool_and_update_keeps_tile_id(self):
        image = Image.new("RGB", (6, 5), "white")
        image_state = {"image_id": "image-a", "target_image_sha256": "hash-a", "session_id": "a" * 32}
        pcs_state = {"text_prompt": "pcs prompt", "instances": {1: _instance(1, np.ones((5, 6)), category="pcs-cat")}}
        pvs_state = {"instances": {9: _instance(9, np.ones((5, 6)), category="pvs-cat")}}
        stitch_state = stitch_callbacks.new_stitch_state("a" * 32, "owner")
        with (
            mock.patch.object(app, "_workspace", return_value={"image": image}),
            mock.patch.object(app, "_publish_annotated_image", return_value="/cache/annotated_stitch.zip") as publish,
        ):
            saved = app._save_annotated_tile(image_state, pcs_state, pvs_state, app.MODE_PCS, stitch_state, "one.png")
            self.assertEqual(saved[4], "/cache/annotated_stitch.zip")
            state = saved[0]
            tile = state["saved_tiles"][0]
            self.assertEqual(tile["instances"][0]["category_name"], "pcs-cat")
            self.assertEqual(tile["instances"][0]["provenance"]["pool"], "pcs")
            self.assertNotEqual(tile["instances"][0]["category_name"], "pvs-cat")
            tile_id = tile["tile_id"]
            updated = app._save_annotated_tile(image_state, pcs_state, pvs_state, app.MODE_PCS, state, "renamed.png", tile_id, True)
        self.assertEqual(len(updated[0]["saved_tiles"]), 1)
        self.assertEqual(updated[0]["saved_tiles"][0]["tile_id"], tile_id)
        self.assertEqual(updated[0]["saved_tiles"][0]["name"], "renamed.png")
        self.assertEqual(updated[4], "/cache/annotated_stitch.zip")
        self.assertEqual(publish.call_count, 2)
        self.assertEqual(publish.call_args.args[2], image_state["session_id"])

    def test_apply_keeps_categories_masks_and_marks_missing_score(self):
        first = np.zeros((5, 6), dtype=bool)
        first[1:4, 1:3] = True
        second = np.zeros((5, 6), dtype=bool)
        second[0:2, 4:6] = True
        stitch_state = {
            "annotated_mode": True,
            "mosaic_instances": [
                {"id": "one", "category_name": "scratch", "mask": first, "score": 0.7, "provenance": {"tile": 0}},
                {"id": "two", "category_name": "hole", "mask": second, "provenance": {"tile": 1}},
            ],
        }
        with mock.patch.object(app, "_view", return_value=(None,) * 8):
            result = app._apply_stitch_instances(stitch_state, {}, {}, app._new_pvs_state(), app.MODE_PVS)
        imported = result[0]["instances"]
        self.assertEqual([item["category_name"] for item in imported.values()], ["scratch", "hole"])
        self.assertTrue(np.array_equal(imported[1]["mask_fullres_bool"], first))
        self.assertTrue(np.array_equal(imported[2]["mask_fullres_bool"], second))
        self.assertFalse(imported[1]["score_missing"])
        self.assertTrue(imported[2]["score_missing"])

    def test_generate_crop_restore_keeps_annotated_masks_consistent(self):
        image = Image.new("RGB", (6, 5), "white")
        mask = np.zeros((5, 6), dtype=bool)
        mask[1:4, 1:5] = True
        state = stitch_callbacks.new_stitch_state("b" * 32, "owner")
        state.update(
            images=[image], shifts=[(0, 0)], rotations=[0.0], layout="horizontal",
            annotations=[[{"id": "tile", "category_name": "scratch", "mask": mask, "provenance": {}}]],
            annotated_mode=True, annotation_sources=[], queue_dirty=False,
        )
        generated = stitch_callbacks.generate_mosaic(state, True, False)[0]
        original = generated["mosaic_instances"][0]["mask"].copy()
        view = stitch_callbacks.mosaic_crop_payload(generated)["server_view"]
        payload = {"gesture": "drag", "expected_revision": view["revision"], "image_id": view["image_id"], "image_sha256": view["image_sha256"], "start_xy": [1, 1], "end_xy": [5, 5]}
        cropped = stitch_callbacks.crop_mosaic_preview(payload, generated)[0]
        self.assertEqual(cropped["mosaic_instances"][0]["mask"].shape, (4, 4))
        restored = stitch_callbacks.restore_full_mosaic(cropped)[0]
        self.assertTrue(np.array_equal(restored["mosaic_instances"][0]["mask"], original))


    def test_annotation_prompt_logits_preserve_hole_and_reject_ordinary_missing_logits(self):
        from sam3_demo import pcs_pvs_callbacks

        mask = np.ones((8, 8), dtype=bool)
        mask[2:6, 2:6] = False
        logits = pcs_pvs_callbacks._instance_prompt_logits(
            {"mask_fullres_bool": mask, "annotation_provenance": {"tile": 0}},
            (4, 4),
        )
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(logits.shape, (4, 4))
        self.assertEqual(logits[0, 0], 10.0)
        self.assertEqual(logits[1, 1], -10.0)
        self.assertIsNone(
            pcs_pvs_callbacks._instance_prompt_logits(
                {"mask_fullres_bool": mask}, (4, 4)
            )
        )

    def test_point_refine_lazily_converts_imported_annotation_logits(self):
        mask = np.ones((8, 8), dtype=bool)
        mask[2:6, 2:6] = False
        imported = app._make_inst(1, "annotated_stitch", mask, app._mask_box(mask), 0.0, history=[])
        imported.update(annotation_provenance={"tile": 0}, score_missing=True)
        pvs_state = app._new_pvs_state()
        pvs_state.update(instances={1: imported}, active_instance_id=1, next_instance_id=2)
        prediction = {
            "masks": np.asarray([mask]),
            "scores": np.asarray([0.75], dtype=np.float32),
            "lowres_logits": np.ones((1, 4, 4), dtype=np.float32),
        }
        with (
            mock.patch.object(app, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(app, "_fresh_state", return_value={}),
            mock.patch.object(app, "_predict_inst", return_value=prediction) as predict,
            mock.patch.object(app.SUPERVISOR, "lease", return_value=nullcontext()),
        ):
            result = app._refine_active_pvs_with_point(
                {"width": 8, "height": 8}, pvs_state, [1, 1], 1
            )
        self.assertEqual(result, (1, "positive_point"))
        prompt_logits = predict.call_args.kwargs["mask_input_lowres_logits"]
        self.assertEqual(prompt_logits.dtype, np.float32)
        self.assertEqual(prompt_logits[1, 1], -10.0)
        self.assertFalse(pvs_state["instances"][1]["score_missing"])
if __name__ == "__main__":
    unittest.main()
