import copy
import hashlib
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from template_match_workflow import (
    map_crop_mask_to_source,
    render_template_overlay,
    run_template_match_workflow,
    translate_source_mask,
)


def _repeated_source(count=4):
    height, width = 72, 172
    image = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234)
    patch = rng.integers(10, 245, size=(24, 24, 3), dtype=np.uint8)
    origins = [(8 + 40 * index, 20) for index in range(count)]
    for x, y in origins:
        image[y : y + 24, x : x + 24] = patch

    seed_mask = np.zeros((height, width), dtype=bool)
    x, y = origins[0]
    seed_mask[y + 4 : y + 10, x + 4 : x + 10] = True
    seed_mask[y + 6 : y + 8, x + 6 : x + 8] = False
    seed_mask[y + 12 : y + 19, x + 13 : x + 18] = True
    return image, origins, seed_mask


def _instance(instance_id, mask, status):
    return {
        "id": instance_id,
        "mask_fullres_bool": np.asarray(mask).copy(),
        "status": status,
        "source": "manual_pvs",
    }


class TemplateMatchWorkflowTest(unittest.TestCase):
    def test_crop_mask_is_mapped_to_full_source_coordinates(self):
        crop_mask = np.zeros((5, 7), dtype=bool)
        crop_mask[1:4, 2:6] = True

        mapped = map_crop_mask_to_source(crop_mask, (12, 20), [4, 3, 11, 8])

        expected = np.zeros((12, 20), dtype=bool)
        expected[4:7, 6:10] = True
        np.testing.assert_array_equal(mapped, expected)

    def test_only_seed_and_accepted_instances_are_blockers(self):
        image, _, seed = _repeated_source()
        accepted = translate_source_mask(seed, 40, 0)
        draft = translate_source_mask(seed, 80, 0)
        deleted = translate_source_mask(seed, 120, 0)
        state = {
            "active_instance_id": 1,
            "instances": {
                1: _instance(1, seed, "draft"),
                2: _instance(2, accepted, "accepted"),
                3: _instance(3, draft, "draft"),
                4: _instance(4, deleted, "deleted"),
            },
        }
        before = copy.deepcopy(state)

        workflow = run_template_match_workflow(
            image,
            [0, 0, image.shape[1], image.shape[0]],
            state,
            match_threshold=0.99,
            expand_threshold=2,
            nms_threshold=0.3,
        )

        translations = [item["translation_xy"] for item in workflow["result"]["matches"]]
        self.assertEqual(translations, [[80, 0], [120, 0]])
        self.assertEqual(
            workflow["result"]["blockers"],
            {"seed_instance_id": 1, "accepted_instance_ids": [2]},
        )
        self.assertEqual(state.keys(), before.keys())
        self.assertEqual(state["active_instance_id"], before["active_instance_id"])
        for instance_id in state["instances"]:
            self.assertEqual(
                state["instances"][instance_id]["status"],
                before["instances"][instance_id]["status"],
            )
            np.testing.assert_array_equal(
                state["instances"][instance_id]["mask_fullres_bool"],
                before["instances"][instance_id]["mask_fullres_bool"],
            )

    def test_empty_accepted_instance_is_ignored_as_blocker(self):
        image, _, seed = _repeated_source(count=2)
        empty = np.zeros_like(seed)
        state = {
            "active_instance_id": 1,
            "instances": {
                1: _instance(1, seed, "draft"),
                2: _instance(2, empty, "accepted"),
            },
        }

        workflow = run_template_match_workflow(
            image,
            [0, 0, image.shape[1], image.shape[0]],
            state,
            match_threshold=0.99,
            expand_threshold=2,
            nms_threshold=0.3,
        )

        self.assertEqual(workflow["result"]["match_count"], 1)
        self.assertEqual(
            workflow["result"]["blockers"],
            {"seed_instance_id": 1, "accepted_instance_ids": []},
        )

    def test_exact_seed_is_suppressed_when_nms_threshold_is_one(self):
        image, _, seed = _repeated_source(count=2)
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "draft")},
        }

        workflow = run_template_match_workflow(
            image,
            [0, 0, image.shape[1], image.shape[0]],
            state,
            match_threshold=0.99,
            expand_threshold=2,
            nms_threshold=1.0,
        )

        self.assertEqual(workflow["result"]["match_count"], 1)
        self.assertEqual(
            workflow["result"]["matches"][0]["translation_xy"],
            [40, 0],
        )

    def test_translated_masks_preserve_holes_and_components(self):
        image, _, seed = _repeated_source(count=3)
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "draft")},
        }

        workflow = run_template_match_workflow(
            image,
            [0, 0, image.shape[1], image.shape[0]],
            state,
            match_threshold=0.99,
            expand_threshold=2,
            nms_threshold=0.3,
        )

        self.assertEqual(workflow["result"]["match_count"], 2)
        for match, translated in zip(
            workflow["result"]["matches"],
            workflow["match_masks_fullres_bool"],
        ):
            dx, dy = match["translation_xy"]
            expected = translate_source_mask(seed, dx, dy)
            np.testing.assert_array_equal(translated, expected)
            self.assertEqual(np.count_nonzero(translated), np.count_nonzero(seed))
            component_count, _ = cv2.connectedComponents(
                translated.astype(np.uint8),
                connectivity=8,
            )
            self.assertEqual(component_count - 1, 2)
        self.assertTrue(
            np.array_equal(
                workflow["overlay_rgb"].shape,
                np.array(image.shape),
            )
        )
        json.dumps(workflow["result"])

    def test_zero_matches_returns_seed_only_overlay_and_serializable_result(self):
        image, _, seed = _repeated_source(count=1)
        image = image[:, :40].copy()
        seed = seed[:, :40].copy()
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "draft")},
        }

        workflow = run_template_match_workflow(
            image,
            [0, 0, image.shape[1], image.shape[0]],
            state,
            match_threshold=0.99,
            expand_threshold=2,
            nms_threshold=0.3,
        )

        self.assertEqual(workflow["result"]["match_count"], 0)
        self.assertEqual(workflow["result"]["matches"], [])
        self.assertEqual(workflow["match_masks_fullres_bool"], [])
        self.assertEqual(workflow["overlay_rgb"].shape, image.shape)
        self.assertEqual(
            workflow["result"]["seed"]["mask_pixel_sha256"],
            hashlib.sha256(
                np.ascontiguousarray(seed, dtype=np.uint8).tobytes()
            ).hexdigest(),
        )
        json.dumps(workflow["result"])

    def test_corrupt_active_masks_are_rejected_before_matching(self):
        image, _, seed = _repeated_source(count=1)
        corrupt_masks = {
            "non_2d": seed[..., None],
            "wrong_shape": seed[:-1],
            "non_numeric": np.full(seed.shape, "foreground", dtype="U10"),
            "nan": np.full(seed.shape, np.nan, dtype=np.float32),
            "infinity": np.full(seed.shape, np.inf, dtype=np.float32),
        }

        for label, corrupt_mask in corrupt_masks.items():
            with self.subTest(label=label), mock.patch(
                "template_match_workflow.match_periodic_instances"
            ) as matcher:
                state = {
                    "active_instance_id": 1,
                    "instances": {1: _instance(1, corrupt_mask, "draft")},
                }
                with self.assertRaisesRegex(ValueError, "crop mask"):
                    run_template_match_workflow(
                        image,
                        [0, 0, image.shape[1], image.shape[0]],
                        state,
                    )
                matcher.assert_not_called()

    def test_corrupt_accepted_mask_is_rejected_before_matching(self):
        image, _, seed = _repeated_source(count=1)
        accepted = np.full(seed.shape, np.nan, dtype=np.float32)
        state = {
            "active_instance_id": 1,
            "instances": {
                1: _instance(1, seed, "draft"),
                2: _instance(2, accepted, "accepted"),
            },
        }

        with mock.patch(
            "template_match_workflow.match_periodic_instances"
        ) as matcher:
            with self.assertRaisesRegex(ValueError, "finite"):
                run_template_match_workflow(
                    image,
                    [0, 0, image.shape[1], image.shape[0]],
                    state,
                )
            matcher.assert_not_called()

    def test_translate_source_mask_rejects_edge_clipping(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 7:10] = True

        with self.assertRaisesRegex(ValueError, "clipped by source bounds"):
            translate_source_mask(mask, 1, 0)

    def test_overflow_match_candidate_is_skipped(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        seed = np.zeros((20, 20), dtype=bool)
        seed[4:8, 4:8] = True
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "draft")},
        }
        overflow_match = {
            "segmentation": [[17, 4], [19, 4], [19, 7], [17, 7]],
            "label": "template",
            "matchScore": 0.95,
        }

        with mock.patch(
            "template_match_workflow.match_periodic_instances",
            return_value=[overflow_match],
        ):
            workflow = run_template_match_workflow(
                image,
                [0, 0, image.shape[1], image.shape[0]],
                state,
            )

        self.assertEqual(workflow["result"]["match_count"], 0)
        self.assertEqual(workflow["match_masks_fullres_bool"], [])

    def test_overlay_draws_fixed_match_id_and_score_label(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        seed = np.zeros((60, 100), dtype=bool)
        seed[10:20, 10:20] = True
        match = np.zeros((60, 100), dtype=bool)
        match[30:40, 40:50] = True

        with mock.patch(
            "template_match_workflow.cv2.putText",
            wraps=cv2.putText,
        ) as put_text:
            overlay = render_template_overlay(
                image,
                seed,
                [match],
                [{"match_id": 3, "score": 0.934}],
            )

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(put_text.call_args.args[1], "M3 0.93")
        legacy_overlay = render_template_overlay(image, seed, [match])
        self.assertEqual(legacy_overlay.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
