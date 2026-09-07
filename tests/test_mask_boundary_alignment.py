from __future__ import annotations

import time
import unittest

import cv2
import numpy as np

from sam3_demo.mask_boundary_alignment import align_mask_boundary


def _initial(width: int, height: int) -> dict:
    return {
        "pivot_x": width / 2,
        "pivot_y": height / 2,
        "center_x": width / 2,
        "center_y": height / 2,
        "scale": 1.0,
        "rotation_deg": 0.0,
    }


class MaskBoundaryAlignmentTests(unittest.TestCase):
    def test_unique_shape_is_aligned_and_can_auto_apply(self):
        height, width = 320, 420
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (170, 135), (75, 28), 17, 0, 360, 1, -1)
        cv2.rectangle(mask, (265, 190), (320, 245), 1, -1)
        cv2.circle(mask, (82, 235), 21, 1, -1)
        shifted = cv2.warpAffine(
            mask,
            np.float32([[1, 0, 17], [0, 1, -11]]),
            (width, height),
        )
        image = np.repeat((shifted * 190 + 25)[..., None], 3, axis=2)

        result = align_mask_boundary(image, mask, _initial(width, height))

        self.assertTrue(result["auto_apply"])
        self.assertFalse(result["ambiguous"])
        best = result["candidates"][0]
        self.assertAlmostEqual(best["transform"]["center_x"], width / 2 + 17, delta=2)
        self.assertAlmostEqual(best["transform"]["center_y"], height / 2 - 11, delta=2)
        self.assertGreaterEqual(best["metrics"]["central_support"], 0.90)
        self.assertGreaterEqual(best["metrics"]["full_support"], 0.60)

    def test_periodic_symmetry_returns_candidates_without_auto_apply(self):
        height, width = 360, 480
        mask = np.zeros((height, width), np.uint8)
        for y in range(45, height, 70):
            for x in range(40, width, 80):
                cv2.circle(mask, (x, y), 18, 1, -1)
        shifted = cv2.warpAffine(
            mask,
            np.float32([[1, 0, 13], [0, 1, -9]]),
            (width, height),
        )
        image = np.repeat((shifted * 180 + 40)[..., None], 3, axis=2)

        started = time.perf_counter()
        result = align_mask_boundary(image, mask, _initial(width, height))

        self.assertFalse(result["auto_apply"])
        self.assertTrue(result["ambiguous"])
        self.assertGreaterEqual(len(result["candidates"]), 2)
        self.assertLessEqual(len(result["candidates"]), 3)
        self.assertLess(time.perf_counter() - started, 3.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            align_mask_boundary(
                np.zeros((20, 20, 3), np.uint8),
                np.zeros((20, 20), np.uint8),
                _initial(20, 20),
            )


if __name__ == "__main__":
    unittest.main()
