import unittest

import cv2
import numpy as np

from sam3_demo.layout.mask_quality import compare_masks, validate_candidate_transition
from sam3_demo.layout.preprocess_registry import build_candidates, params_from_controls


class LayoutMaskAgentPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.current = params_from_controls(12, False, 0, 0, 0, "all", 0)

    def test_auto_candidates_are_bounded_and_deduplicated(self):
        candidates = build_candidates(self.current, profile_mode="Auto", image_shape=(100, 120))
        self.assertLessEqual(len(candidates), 6)
        signatures = {
            tuple(item["params"][key] for key in self.current)
            for item in candidates
        }
        self.assertEqual(len(signatures), len(candidates))
        self.assertEqual(candidates[0]["params"], self.current)

    def test_act_fills_with_close_without_morph(self):
        current = dict(self.current, close_kernel=15)
        candidates = build_candidates(
            current,
            profile_mode="ACT",
            message="\u518d\u586b\u4e00\u70b9",
            image_shape=(100, 120),
        )
        selected = next(item for item in candidates if item["label"] == "ACT fill a little more")
        self.assertEqual(selected["params"]["close_kernel"], 17)
        self.assertEqual(selected["params"]["morph_pixels"], 0)

    def test_ge1_and_ge2_priors_do_not_cross(self):
        ge1 = build_candidates(self.current, profile_mode="GE1", image_shape=(100, 120))
        self.assertTrue({0, 1, 2}.issubset({row["params"]["morph_pixels"] for row in ge1}))
        ge2 = build_candidates(
            dict(self.current, close_kernel=15),
            profile_mode="GE2",
            image_shape=(100, 120),
        )
        identity = next(row for row in ge2 if row["label"] == "GE2 preserve holes")
        self.assertEqual(identity["params"]["close_kernel"], 0)
        self.assertEqual(identity["params"]["morph_pixels"], 0)

    def test_explicit_value_must_be_in_ui_range(self):
        candidates = build_candidates(self.current, message="close=17", image_shape=(100, 120))
        self.assertTrue(any(row["params"]["close_kernel"] == 17 for row in candidates))
        with self.assertRaisesRegex(ValueError, "outside the UI range"):
            build_candidates(self.current, message="close=99", image_shape=(100, 120))

    def test_empty_full_merge_and_large_hole_loss_are_rejected(self):
        base = np.zeros((100, 100), dtype=bool)
        base[10:40, 10:40] = True
        base[10:40, 50:80] = True
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_candidate_transition(base, np.zeros_like(base))
        with self.assertRaisesRegex(ValueError, "full image"):
            validate_candidate_transition(base, np.ones_like(base))
        merged = base.copy()
        merged[20:30, 40:50] = True
        with self.assertRaisesRegex(ValueError, "merges"):
            validate_candidate_transition(base, merged)

        ring = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ring, (10, 10), (90, 90), 1, -1)
        cv2.rectangle(ring, (30, 30), (70, 70), 0, -1)
        filled = ring.astype(bool)
        filled[30:71, 30:71] = True
        report = compare_masks(ring.astype(bool), filled)
        self.assertTrue(report["large_holes_lost"])
        with self.assertRaisesRegex(ValueError, "large hole"):
            validate_candidate_transition(ring.astype(bool), filled)


if __name__ == "__main__":
    unittest.main()
