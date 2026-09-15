import unittest
from unittest.mock import patch

from PIL import Image

from sam3_demo.stitch_workflow import auto_align_images, default_shifts_for_layout


class GridSafetyTest(unittest.TestCase):
    def setUp(self):
        self.images = [Image.new("RGB", (552, 560)) for _ in range(4)]

    def test_observed_periodic_collapse_is_rejected(self):
        matches = [(54, 0, .729, False), (-77, 27, .670, False),
                   (11, -33, .724, False), (407, 0, .827, False)]
        for layout in ("grid_2x2", "grid_2xn"):
            with self.subTest(layout=layout), patch(
                "sam3_demo.stitch_grid_alignment.match_grid_pair",
                side_effect=[[m[:3]] for m in matches],
            ):
                shifts, logs = auto_align_images(self.images, layout)
                self.assertEqual(shifts, default_shifts_for_layout(self.images, layout))
                self.assertIn("保留规则网格", logs[-1])

    def test_incompatible_paths_are_not_averaged(self):
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair",
                   side_effect=[[(400, 0, .9)], [(0, 400, .9)],
                                [(0, 450, .9)], [(500, 0, .9)]]):
            shifts, logs = auto_align_images(self.images, "grid_2x2")
        self.assertEqual(shifts, default_shifts_for_layout(self.images, "grid_2x2"))
        self.assertTrue(any("一致解" in line for line in logs))

    def test_consistent_overlapping_grid_is_preserved(self):
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair",
                   side_effect=[[(400, 0, .9)], [(0, 420, .9)],
                                [(0, 420, .9)], [(400, 0, .9)]]):
            shifts, _ = auto_align_images(self.images, "grid_2x2")
        self.assertEqual(shifts, [(0, 0), (400, 0), (0, 420), (400, 420)])

    def test_horizontal_collapsing_candidate_is_rejected(self):
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair",
                   return_value=[(54, 0, .9)]):
            shifts, _ = auto_align_images(self.images[:2], "horizontal")
        self.assertEqual(shifts, [(0, 0), (552, 0)])
