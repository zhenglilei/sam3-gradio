import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import periodic_template_matching as periodic
from periodic_template_matching import match_periodic_instances, smart_annotation


def _periodic_image():
    image = np.zeros((72, 150, 3), dtype=np.uint8)
    origins = [(10, 20), (55, 20), (100, 20)]
    for x, y in origins:
        cv2.rectangle(image, (x + 2, y + 2), (x + 16, y + 16), (220, 220, 220), -1)
        cv2.rectangle(image, (x + 9, y + 2), (x + 16, y + 8), (80, 80, 80), -1)
        cv2.circle(image, (x + 5, y + 13), 2, (255, 255, 255), -1)
    seed = [[12, 22], [26, 22], [26, 36], [12, 36]]
    return image, origins, seed


def _edge_periodic_image():
    image = np.full((100, 100, 3), 30, dtype=np.uint8)
    origins = [(45, 45), (0, 45), (89, 45), (45, 0), (45, 89)]
    for x, y in origins:
        cv2.rectangle(image, (x, y), (x + 10, y + 10), (220, 220, 220), -1)
        cv2.rectangle(image, (x + 3, y + 3), (x + 7, y + 7), (80, 80, 80), -1)
        cv2.circle(image, (x + 2, y + 8), 1, (255, 255, 255), -1)
    seed = [[45, 45], [55, 45], [55, 55], [45, 55]]
    return image, origins, seed


class PeriodicTemplateMatchingTest(unittest.TestCase):
    def test_repeated_polygon_is_translated_and_existing_annotation_is_blocked(self):
        image, _, seed = _periodic_image()
        existing = [[x + 45, y] for x, y in seed]

        matches = match_periodic_instances(
            image,
            seed,
            label="periodic-part",
            match_threshold=0.99,
            expand_threshold=5,
            nms_threshold=0.3,
            all_segmentations=[existing],
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["segmentation"], [[x + 90, y] for x, y in seed])
        self.assertEqual(matches[0]["label"], "periodic-part")
        self.assertGreaterEqual(matches[0]["matchScore"], 0.99)

    def test_seed_is_not_returned_when_no_repeat_exists(self):
        image, _, seed = _periodic_image()
        image[:, 45:] = 0

        matches = match_periodic_instances(
            image,
            seed,
            label="seed",
            match_threshold=0.99,
            expand_threshold=5,
        )

        self.assertEqual(matches, [])

    def test_smart_annotation_and_cli_end_to_end(self):
        image, _, seed = _periodic_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "periodic.png"
            request_path = tmp_path / "request.json"
            output_path = tmp_path / "matches.json"
            cv2.imwrite(str(image_path), image)

            request = {
                "picPath": str(image_path),
                "label": "part",
                "matchThreshold": 0.99,
                "expandThreshold": 5,
                "nmsThreshold": 0.3,
                "segmentation": seed,
                "allSegmentation": [seed],
            }
            request_path.write_text(json.dumps(request), encoding="utf-8")

            direct_matches = smart_annotation(request)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "run_periodic_template_matching.py"),
                    "--request",
                    str(request_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cli_matches = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(cli_matches, direct_matches)
        self.assertEqual(len(cli_matches), 2)
        self.assertIn("matched 2 repeated instances", result.stdout)

    def test_spatially_uniform_templates_are_rejected(self):
        images = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.full((32, 32, 3), [20, 120, 220], dtype=np.uint8),
        ]
        for image in images:
            with self.subTest(color=image[0, 0].tolist()):
                with self.assertRaisesRegex(ValueError, "no intensity variation"):
                    match_periodic_instances(
                        image,
                        [[4, 4], [12, 4], [12, 12], [4, 12]],
                        label="blank",
                        expand_threshold=2,
                    )

    def test_padding_finds_legal_instances_on_all_four_image_edges(self):
        image, origins, seed = _edge_periodic_image()

        matches = match_periodic_instances(
            image,
            seed,
            label="edge-part",
            match_threshold=0.99,
            expand_threshold=5,
            nms_threshold=0.3,
        )

        matched_origins = {
            (min(point[0] for point in match["segmentation"]),
             min(point[1] for point in match["segmentation"]))
            for match in matches
        }
        self.assertEqual(
            matched_origins,
            set(origins[1:]),
        )
        self.assertEqual(len(matches), 4)
        for match in matches:
            self.assertTrue(
                all(
                    0 <= x < image.shape[1] and 0 <= y < image.shape[0]
                    for x, y in match["segmentation"]
                )
            )

    def test_edge_blocker_suppresses_match_and_padding_does_not_emit_external_peak(self):
        image, _, seed = _edge_periodic_image()
        left_edge = [[0, 45], [10, 45], [10, 55], [0, 55]]

        matches = match_periodic_instances(
            image,
            seed,
            label="edge-part",
            match_threshold=0.99,
            expand_threshold=5,
            nms_threshold=0.3,
            all_segmentations=[left_edge],
        )

        self.assertEqual(len(matches), 3)
        self.assertNotIn(
            (0, 45),
            {
                (
                    min(point[0] for point in match["segmentation"]),
                    min(point[1] for point in match["segmentation"]),
                )
                for match in matches
            },
        )
        for match in matches:
            self.assertTrue(
                all(
                    0 <= x < image.shape[1] and 0 <= y < image.shape[0]
                    for x, y in match["segmentation"]
                )
                )

    def test_zero_expansion_keeps_original_template_match_domain(self):
        image, _, seed = _periodic_image()

        matches = match_periodic_instances(
            image,
            seed,
            label="periodic-part",
            match_threshold=0.99,
            expand_threshold=0,
            nms_threshold=0.3,
        )

        self.assertEqual(
            {
                (
                    min(point[0] for point in match["segmentation"]),
                    min(point[1] for point in match["segmentation"]),
                )
                for match in matches
            },
            {(57, 22), (102, 22)},
        )

    def test_excessive_local_peak_count_is_rejected_before_nms(self):
        score_map = np.zeros((130, 130), dtype=np.float32)
        score_map[::2, ::2] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            "too many local template matches",
        ):
            periodic._local_peak_candidates(
                score_map,
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
