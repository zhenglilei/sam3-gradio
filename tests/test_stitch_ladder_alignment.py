import unittest
from unittest.mock import patch

from PIL import Image
from sam3_demo.stitch_workflow import auto_align_images, default_shifts_for_layout


class LadderAlignmentTest(unittest.TestCase):
    def test_future_cell_can_reject_locally_best_rung(self):
        images = [Image.new("RGB", (100, 100)) for _ in range(6)]
        # Fetch order: top01, rung0, rung1, bottom01, top12, rung2, bottom12.
        candidates = [[(60, 0, .9)], [(0, 60, .99), (0, 80, .8)],
                      [(0, 60, .99), (0, 80, .8)], [(60, 0, .9)],
                      [(60, 0, .9)], [(0, 80, .9)], [(60, 0, .9)]]
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair", side_effect=candidates):
            shifts, _ = auto_align_images(images, "grid_2xn")
        self.assertEqual(shifts, [(0, 0), (60, 0), (120, 0),
                                  (0, 80), (60, 80), (120, 80)])

    def test_odd_count_retains_top_row_tail(self):
        images = [Image.new("RGB", (100, 100)) for _ in range(5)]
        candidates = [[(60, 0, .9)], [(0, 70, .9)], [(0, 70, .9)],
                      [(60, 0, .9)], [(60, 0, .9)]]
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair", side_effect=candidates):
            shifts, _ = auto_align_images(images, "grid_2xn")
        self.assertEqual(shifts, [(0, 0), (60, 0), (120, 0), (0, 70), (60, 70)])

    def test_missing_grid_evidence_retains_defaults(self):
        images = [Image.new("RGB", (40, 36)) for _ in range(6)]
        shifts, logs = auto_align_images(images, "grid_2xn")
        self.assertEqual(shifts, default_shifts_for_layout(images, "grid_2xn"))
        self.assertTrue(logs)

    def test_single_and_two_tile_grid(self):
        images = [Image.new("RGB", (100, 100)) for _ in range(2)]
        self.assertEqual(auto_align_images(images[:1], "grid_2xn"), ([(0, 0)], []))
        with patch("sam3_demo.stitch_grid_alignment.match_grid_pair", return_value=[(0, 60, .9)]):
            shifts, _ = auto_align_images(images, "grid_2xn")
        self.assertEqual(shifts, [(0, 0), (0, 60)])
