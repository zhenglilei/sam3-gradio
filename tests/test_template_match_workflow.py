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

import template_match_workflow as workflow_module
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

    def test_all_non_deleted_instances_are_blockers(self):
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
        self.assertEqual(translations, [[120, 0]])
        self.assertEqual(
            workflow["result"]["blockers"],
            {"seed_instance_id": 1, "instance_ids": [2, 3]},
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

    def test_empty_instance_is_ignored_as_blocker(self):
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
            {"seed_instance_id": 1, "instance_ids": []},
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

    def test_translate_source_mask_allows_only_safe_edge_clipping(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 10:20] = True

        translated = translate_source_mask(
            mask,
            1,
            0,
            allow_clip=True,
            min_visible_ratio=0.9,
        )

        self.assertEqual(np.count_nonzero(translated), 45)
        with self.assertRaisesRegex(ValueError, "too little visible area"):
            translate_source_mask(
                mask,
                2,
                0,
                allow_clip=True,
                min_visible_ratio=0.9,
            )

    def test_horizontal_edge_refinement_requires_score_gain(self):
        image = np.zeros((20, 40), dtype=np.uint8)
        template = np.arange(25, dtype=np.uint8).reshape(5, 5)

        def score_with_small_gain(_image, _template, x, _y, **_kwargs):
            return (0.52 if x == 1 else 0.50), 1.0

        with mock.patch(
            "template_match_workflow._partial_ccoeff_at",
            side_effect=score_with_small_gain,
        ):
            rejected = workflow_module._refine_horizontal_edge_translation(
                image,
                template,
                [0, 5, 5, 10],
                edge_margin=5,
                search_radius=2,
            )
        self.assertEqual(rejected["delta_x"], 0)

        def score_with_clear_gain(_image, _template, x, _y, **_kwargs):
            return (0.54 if x == 1 else 0.50), 1.0

        with mock.patch(
            "template_match_workflow._partial_ccoeff_at",
            side_effect=score_with_clear_gain,
        ):
            accepted = workflow_module._refine_horizontal_edge_translation(
                image,
                template,
                [0, 5, 5, 10],
                edge_margin=5,
                search_radius=2,
            )
        self.assertEqual(accepted["delta_x"], 1)
        self.assertAlmostEqual(accepted["score_gain"], 0.04)

    def test_workflow_refines_right_edge_and_safely_clips_mask(self):
        image = np.full((60, 100, 3), 30, dtype=np.uint8)
        rng = np.random.default_rng(20260903)
        template = rng.integers(50, 240, size=(10, 10, 3), dtype=np.uint8)
        image[10:20, 10:20] = template
        image[30:40, 91:100] = template[:, :9]
        seed = np.zeros((60, 100), dtype=bool)
        seed[10:20, 10:20] = True
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "accepted")},
        }
        coarse_match = {
            "segmentation": [[90, 30], [99, 30], [99, 39], [90, 39]],
            "label": "template",
            "matchScore": 0.95,
        }

        with mock.patch(
            "template_match_workflow.match_periodic_instances",
            return_value=[coarse_match],
        ):
            result = run_template_match_workflow(
                image,
                [0, 0, image.shape[1], image.shape[0]],
                state,
                expand_threshold=20,
            )

        self.assertEqual(result["result"]["match_count"], 1)
        match = result["result"]["matches"][0]
        self.assertEqual(match["translation_xy"], [81, 20])
        self.assertEqual(match["coarse_translation_xy"], [80, 20])
        self.assertEqual(match["edge_refine_delta_xy"], [1, 0])
        self.assertTrue(match["edge_refined"])
        self.assertAlmostEqual(match["visible_ratio"], 0.9)
        self.assertEqual(np.count_nonzero(result["match_masks_fullres_bool"][0]), 90)

    def test_workflow_does_not_refine_an_interior_candidate(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        seed = np.zeros((60, 100), dtype=bool)
        seed[10:20, 10:20] = True
        state = {
            "active_instance_id": 1,
            "instances": {1: _instance(1, seed, "accepted")},
        }
        coarse_match = {
            "segmentation": [[45, 30], [54, 30], [54, 39], [45, 39]],
            "label": "template",
            "matchScore": 0.95,
        }

        with mock.patch(
            "template_match_workflow.match_periodic_instances",
            return_value=[coarse_match],
        ), mock.patch(
            "template_match_workflow._partial_ccoeff_at"
        ) as scorer:
            result = run_template_match_workflow(
                image,
                [0, 0, image.shape[1], image.shape[0]],
                state,
                expand_threshold=10,
            )

        scorer.assert_not_called()
        match = result["result"]["matches"][0]
        self.assertEqual(match["translation_xy"], [35, 20])
        self.assertNotIn("edge_refined", match)

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

    def test_orientation_gate_switches_on_strong_consistent_evidence(self):
        context = {"extents": [8, 8], "prototype_cache": {}}
        centers = {0: [20, 20], 1: [20, 20]}

        def score(_context, group_index, _center, _patch_size, _band):
            combined = 0.50 if group_index == 0 else 0.72
            return {
                "combined": combined,
                "appearance": combined,
                "edge": combined,
                "coverage": 0.95,
            }

        with mock.patch.object(
            workflow_module,
            "_orientation_patch_sizes",
            return_value=((51, 53, 55), 2),
        ), mock.patch.object(
            workflow_module,
            "_score_template_orientation",
            side_effect=score,
        ) as score_mock:
            decision = workflow_module.choose_template_orientation_group(
                context,
                [0, 1],
                centers,
                0,
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["selected_group_index"], 1)
        self.assertEqual(decision["mode"], "strong")
        self.assertGreaterEqual(decision["improvement"], 0.18)
        self.assertEqual(score_mock.call_count, 6)

    def test_orientation_gate_requires_three_scale_consensus_for_midrange_gain(self):
        context = {"extents": [8, 8], "prototype_cache": {}}
        centers = {0: [20, 20], 1: [20, 20]}

        def consistent_score(_context, group_index, _center, _patch_size, _band):
            combined = 0.60 if group_index == 0 else 0.76
            return {
                "combined": combined,
                "appearance": combined,
                "edge": combined,
                "coverage": 0.95,
            }

        with mock.patch.object(
            workflow_module,
            "_orientation_patch_sizes",
            return_value=((51, 53, 55), 2),
        ), mock.patch.object(
            workflow_module,
            "_score_template_orientation",
            side_effect=consistent_score,
        ) as score_mock:
            accepted = workflow_module.choose_template_orientation_group(
                context,
                [0, 1],
                centers,
                0,
            )

        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["selected_group_index"], 1)
        self.assertEqual(accepted["mode"], "multiscale_consensus")
        self.assertAlmostEqual(accepted["improvement"], 0.16)
        self.assertEqual(score_mock.call_count, 6)

        def disagreeing_score(_context, group_index, _center, patch_size, _band):
            if patch_size == 55:
                combined = 0.76 if group_index == 0 else 0.60
            else:
                combined = 0.60 if group_index == 0 else 0.76
            return {
                "combined": combined,
                "appearance": combined,
                "edge": combined,
                "coverage": 0.95,
            }

        with mock.patch.object(
            workflow_module,
            "_orientation_patch_sizes",
            return_value=((51, 53, 55), 2),
        ), mock.patch.object(
            workflow_module,
            "_score_template_orientation",
            side_effect=disagreeing_score,
        ):
            rejected = workflow_module.choose_template_orientation_group(
                context,
                [0, 1],
                centers,
                0,
            )

        self.assertIsNone(rejected)

    def test_orientation_gate_selects_the_best_of_three_template_groups(self):
        context = {"extents": [8, 8, 8], "prototype_cache": {}}
        centers = {0: [20, 20], 1: [20, 20], 2: [20, 20]}
        values = {0: 0.50, 1: 0.68, 2: 0.76}

        def score(_context, group_index, _center, _patch_size, _band):
            combined = values[group_index]
            return {
                "combined": combined,
                "appearance": combined,
                "edge": combined,
                "coverage": 0.95,
            }

        with mock.patch.object(
            workflow_module,
            "_orientation_patch_sizes",
            return_value=((51, 53, 55), 2),
        ), mock.patch.object(
            workflow_module,
            "_score_template_orientation",
            side_effect=score,
        ) as score_mock:
            decision = workflow_module.choose_template_orientation_group(
                context,
                [0, 1, 2],
                centers,
                0,
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["selected_group_index"], 2)
        self.assertEqual(decision["mode"], "strong")
        self.assertAlmostEqual(decision["margin"], 0.08)
        self.assertEqual(score_mock.call_count, 9)

    def test_orientation_gate_safely_rejects_empty_and_tiny_templates(self):
        source = np.zeros((64, 64, 3), dtype=np.uint8)
        empty = np.zeros((64, 64), dtype=bool)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            workflow_module.prepare_template_orientation_context(
                source,
                [empty],
            )

        tiny = np.zeros((64, 64), dtype=bool)
        tiny[8, 8] = True
        context = workflow_module.prepare_template_orientation_context(
            source,
            [tiny, tiny],
        )
        decision = workflow_module.choose_template_orientation_group(
            context,
            [0, 1],
            {0: [8, 8], 1: [8, 8]},
            0,
        )
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
