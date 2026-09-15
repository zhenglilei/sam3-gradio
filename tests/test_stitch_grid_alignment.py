from __future__ import annotations

import unittest

from sam3_demo.stitch_grid_alignment import solve_grid_candidates


class SolveGridCandidatesTest(unittest.TestCase):
    def test_prefers_globally_consistent_candidate_over_highest_scoring_edge(self):
        edges = [
            [
                (516, -8, 0.913),
                (452, -8, 0.804),
            ],
            [(-4, 488, 0.829)],
            [(20, 496, 0.799)],
            [(472, 0, 0.806)],
        ]

        selected = solve_grid_candidates(edges, tolerance=10)

        self.assertEqual(
            selected,
            (
                (452, -8, 0.804),
                (-4, 488, 0.829),
                (20, 496, 0.799),
                (472, 0, 0.806),
            ),
        )

    def test_returns_none_when_no_combination_is_consistent(self):
        edges = [
            [(10, 0, 0.9)],
            [(0, 10, 0.8)],
            [(10, 10, 0.7)],
            [(0, 0, 0.6)],
        ]

        self.assertIsNone(solve_grid_candidates(edges, tolerance=5))

    def test_returns_none_when_an_edge_is_empty(self):
        edges = [
            [(100, 0, 0.9)],
            [],
            [(0, 100, 0.8)],
            [(100, 0, 0.7)],
        ]

        self.assertIsNone(solve_grid_candidates(edges, tolerance=10))

    def test_selects_highest_scoring_exact_consistent_combination(self):
        edges = [
            [(120, -4, 0.91), (120, -4, 0.70)],
            [(-6, 250, 0.84)],
            [(18, 252, 0.82), (19, 252, 0.95)],
            [(144, -2, 0.80), (145, -2, 0.60)],
        ]

        selected = solve_grid_candidates(edges, tolerance=0)

        self.assertEqual(
            selected,
            (
                (120, -4, 0.91),
                (-6, 250, 0.84),
                (18, 252, 0.82),
                (144, -2, 0.80),
            ),
        )


if __name__ == "__main__":
    unittest.main()
