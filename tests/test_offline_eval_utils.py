import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import offline_eval_utils as eval_utils


def _uncompressed_counts(mask: np.ndarray) -> list[int]:
    counts: list[int] = []
    previous = 0
    run = 0
    for value in np.asarray(mask, dtype=np.uint8).ravel(order="F"):
        pixel = int(value)
        if pixel == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = pixel
    counts.append(run)
    return counts


class CocoSegmentationTest(unittest.TestCase):
    def test_polygon_uses_exact_coco_fractional_and_multipart_rasterization(self):
        segmentation = [
            [0.2, 0.2, 2.7, 0.2, 2.7, 2.7, 0.2, 2.7],
            [4.1, 3.1, 5.8, 3.1, 5.8, 5.8, 4.1, 5.8],
        ]
        expected_rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, 7, 7))
        expected = coco_mask.decode(expected_rle).astype(bool)

        actual = eval_utils.decode_coco_segmentation(segmentation, 7, 7)

        self.assertEqual(actual.dtype, np.bool_)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compressed_and_uncompressed_rle_decode_identically(self):
        expected = np.zeros((6, 8), dtype=np.uint8)
        expected[1:5, 2:6] = 1
        expected[2:4, 3:5] = 0
        encoded = coco_mask.encode(np.asfortranarray(expected))
        compressed = {
            "size": [6, 8],
            "counts": encoded["counts"].decode("ascii"),
        }
        uncompressed = {
            "size": [6, 8],
            "counts": _uncompressed_counts(expected),
        }

        for segmentation in (compressed, uncompressed):
            with self.subTest(type=type(segmentation["counts"]).__name__):
                actual = eval_utils.decode_coco_segmentation(segmentation, 6, 8)
                self.assertTrue(np.array_equal(actual, expected.astype(bool)))

    def test_invalid_coco_data_is_never_silently_converted_to_empty_mask(self):
        invalid_values = [
            ([[0, 0, 1, 1]], 4, 4),
            ([[0, 0, 1, 0, float("nan"), 1]], 4, 4),
            ({"size": [3, 4], "counts": [12]}, 4, 4),
            ({"size": [4, 4], "counts": [3, 4]}, 4, 4),
            ({"size": [4, 4], "counts": "not-an-rle"}, 4, 4),
            ([], 4, 4),
        ]
        for segmentation, height, width in invalid_values:
            with self.subTest(segmentation=segmentation):
                with self.assertRaises(ValueError):
                    eval_utils.decode_coco_segmentation(segmentation, height, width)


class CocoIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "ACT_coco"
        self.annotation_path = self.dataset / "annotations" / "instances_val.json"
        self.image_path = self.dataset / "val" / "sample.png"
        self.annotation_path.parent.mkdir(parents=True)
        self.image_path.parent.mkdir(parents=True)
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[:, :, 1] = 120
        Image.fromarray(image, mode="RGB").save(self.image_path)
        self.document = {
            "images": [
                {"id": 11, "file_name": "sample.png", "width": 7, "height": 5}
            ],
            "annotations": [
                {
                    "id": 21,
                    "image_id": 11,
                    "category_id": 1,
                    "original_label": "ACT-1",
                    "segmentation": [[1, 1, 6, 1, 6, 4, 1, 4]],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "ACT"}],
        }
        self._write_document()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_document(self):
        self.annotation_path.write_text(
            json.dumps(self.document), encoding="utf-8"
        )

    def _load(self, **overrides):
        kwargs = {
            "expected_dataset": "ACT_coco",
            "expected_image_id": 11,
            "image_path": self.image_path,
            "expected_annotation_sha256": eval_utils.file_sha256(
                self.annotation_path
            ),
            "expected_image_sha256": eval_utils.file_sha256(self.image_path),
            "expected_split": "val",
            "expected_category_label": "ACT",
            "expected_annotation_label": "ACT-1",
        }
        kwargs.update(overrides)
        return eval_utils.load_validated_coco_ground_truth(
            self.annotation_path, 21, **kwargs
        )

    def test_ground_truth_binds_dataset_annotation_image_dimensions_and_hashes(self):
        mask, annotation, image = self._load()

        self.assertEqual(annotation["id"], 21)
        self.assertEqual(image["id"], 11)
        expected = eval_utils.decode_coco_segmentation(
            self.document["annotations"][0]["segmentation"], 5, 7
        )
        self.assertTrue(np.array_equal(mask, expected))

    def test_each_identity_mismatch_is_rejected(self):
        bad_hash = "0" * 64
        cases = [
            {"expected_dataset": "GE1_coco"},
            {"expected_image_id": 12},
            {"expected_annotation_sha256": bad_hash},
            {"expected_image_sha256": bad_hash},
            {"expected_split": "train"},
            {"expected_category_label": "GE1"},
            {"expected_annotation_label": "ACT-2"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self._load(**overrides)

    def test_duplicate_coco_ids_and_image_record_drift_are_rejected(self):
        duplicate = copy.deepcopy(self.document["annotations"][0])
        self.document["annotations"].append(duplicate)
        self._write_document()
        with self.assertRaisesRegex(ValueError, "Duplicate COCO annotation id"):
            self._load()

        self.document["annotations"] = self.document["annotations"][:1]
        self.document["images"][0]["width"] = 8
        self._write_document()
        with self.assertRaisesRegex(ValueError, "does not match COCO"):
            self._load()


    def test_canonical_image_path_rejects_same_basename_and_traversal(self):
        wrong_image = self.dataset / "other" / self.image_path.name
        wrong_image.parent.mkdir()
        wrong_image.write_bytes(self.image_path.read_bytes())

        with self.assertRaisesRegex(ValueError, "does not match the source image path"):
            self._load(
                image_path=wrong_image,
                expected_image_sha256=eval_utils.file_sha256(wrong_image),
            )

        root_decoy = self.dataset / self.image_path.name
        root_decoy.write_bytes(self.image_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "does not match the source image path"):
            self._load(
                image_path=root_decoy,
                expected_image_sha256=eval_utils.file_sha256(root_decoy),
            )

        self.document["images"][0]["file_name"] = "../sample.png"
        self._write_document()
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            self._load()

    def test_category_and_original_label_must_be_non_empty_strings(self):
        self.document["categories"][0]["name"] = ""
        self._write_document()
        with self.assertRaisesRegex(ValueError, "category.*non-empty string name"):
            self._load()

        self.document["categories"][0]["name"] = "ACT"
        self.document["annotations"][0]["original_label"] = 17
        self._write_document()
        with self.assertRaisesRegex(ValueError, "annotation.*non-empty string label"):
            self._load()

    def test_empty_ground_truth_mask_is_rejected(self):
        encoded = coco_mask.encode(
            np.asfortranarray(np.zeros((5, 7), dtype=np.uint8))
        )
        self.document["annotations"][0]["segmentation"] = {
            "size": [5, 7],
            "counts": encoded["counts"].decode("ascii"),
        }
        self._write_document()

        with self.assertRaisesRegex(ValueError, "empty mask"):
            self._load()


class ThresholdedMatchingTest(unittest.TestCase):
    def test_maximum_cardinality_wins_over_largest_single_iou(self):
        matrix = np.asarray([[0.96, 0.95], [0.95, 0.0]], dtype=np.float64)

        matches = eval_utils.thresholded_hungarian_matches(matrix, 0.95)

        self.assertEqual(matches, [(0, 1, 0.95), (1, 0, 0.95)])

    def test_total_iou_breaks_ties_after_cardinality(self):
        matrix = np.asarray([[0.91, 0.80], [0.70, 0.89]], dtype=np.float64)
        self.assertEqual(
            eval_utils.thresholded_hungarian_matches(matrix, 0.5),
            [(0, 0, 0.91), (1, 1, 0.89)],
        )

    def test_threshold_is_inclusive_and_empty_axes_are_supported(self):
        self.assertEqual(
            eval_utils.thresholded_hungarian_matches([[0.95]], 0.95),
            [(0, 0, 0.95)],
        )
        self.assertEqual(
            eval_utils.thresholded_hungarian_matches(np.zeros((0, 3)), 0.5), []
        )

    def test_invalid_threshold_or_matrix_is_rejected(self):
        for matrix, threshold in (
            ([[float("nan")]], 0.5),
            ([[1.1]], 0.5),
            ([[0.5]], float("inf")),
            ([0.5], 0.5),
        ):
            with self.subTest(matrix=matrix, threshold=threshold):
                with self.assertRaises(ValueError):
                    eval_utils.thresholded_hungarian_matches(matrix, threshold)


class OutputAndMaskArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_output_directory_must_be_empty(self):
        output = self.root / "output"
        self.assertEqual(eval_utils.prepare_empty_output_dir(output), output)
        self.assertEqual(eval_utils.prepare_empty_output_dir(output), output)
        (output / "existing.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            eval_utils.prepare_empty_output_dir(output)
        self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep")

    def test_binary_mask_roundtrip_verifies_file_pixels_and_shape(self):
        output = eval_utils.prepare_empty_output_dir(self.root / "run")
        mask = np.zeros((12, 15), dtype=bool)
        mask[2:10, 3:11] = True
        mask[5:7, 6:8] = False

        record = eval_utils.save_binary_mask(output, "masks/pred-1.png", mask)
        decoded = eval_utils.load_binary_mask(output, record)

        self.assertTrue(np.array_equal(decoded, mask))
        self.assertEqual(record["shape_hw"], [12, 15])
        self.assertEqual(record["pixel_sha256"], eval_utils.binary_mask_sha256(mask))
        with Image.open(output / record["path"]) as image:
            self.assertEqual(set(np.unique(np.asarray(image)).tolist()), {0, 255})

        wrong_shape = {**record, "shape_hw": [12, 14]}
        with self.assertRaisesRegex(ValueError, "shape"):
            eval_utils.load_binary_mask(output, wrong_shape)
        wrong_pixel_hash = {**record, "pixel_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "pixel SHA-256"):
            eval_utils.load_binary_mask(output, wrong_pixel_hash)

    def test_mask_corruption_and_unsafe_paths_are_rejected(self):
        output = eval_utils.prepare_empty_output_dir(self.root / "run")
        mask = np.eye(5, dtype=bool)
        record = eval_utils.save_binary_mask(output, "masks/pred.png", mask)
        path = output / record["path"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "file SHA-256"):
            eval_utils.load_binary_mask(output, record)

        for unsafe in ("../escape.png", "/absolute.png", "masks\\escape.png"):
            with self.subTest(path=unsafe):
                with self.assertRaises(ValueError):
                    eval_utils.save_binary_mask(output, unsafe, mask)
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            eval_utils.save_binary_mask(output, "masks/not-binary.png", mask * 255)

    def test_mask_symlink_is_rejected(self):
        output = eval_utils.prepare_empty_output_dir(self.root / "run")
        mask = np.eye(4, dtype=bool)
        record = eval_utils.save_binary_mask(output, "masks/original.png", mask)
        link = output / "masks" / "link.png"
        os.symlink(output / record["path"], link)
        linked_record = {**record, "path": "masks/link.png"}
        with self.assertRaisesRegex(ValueError, "symlink"):
            eval_utils.load_binary_mask(output, linked_record)


    def test_relative_artifact_resolution_and_atomic_json_are_strict(self):
        output = eval_utils.prepare_empty_output_dir(self.root / "run")
        destination = output / "indexes" / "selected.json"
        value = {"groups": ["g1", "g2"], "revision": 1}

        self.assertEqual(
            eval_utils.write_json_atomic(destination, value), destination
        )
        self.assertEqual(
            json.loads(destination.read_text(encoding="utf-8")), value
        )
        self.assertEqual(
            eval_utils.resolve_relative_artifact(
                output, "indexes/selected.json"
            ),
            destination.resolve(),
        )

        previous = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            eval_utils.write_json_atomic(destination, {"bad": float("nan")})
        self.assertEqual(destination.read_bytes(), previous)

        link = output / "indexes" / "selected-link.json"
        os.symlink(destination, link)
        with self.assertRaisesRegex(ValueError, "symlink"):
            eval_utils.resolve_relative_artifact(
                output, "indexes/selected-link.json"
            )
        with self.assertRaisesRegex(ValueError, "symlink"):
            eval_utils.write_json_atomic(link, {"replace": True})
        with self.assertRaises(ValueError):
            eval_utils.resolve_relative_artifact(output, "../escape.json")


class RunManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        self.selected_path = self.run / "selected_groups.json"
        self.predictions_path = self.run / "predictions.jsonl"
        self.selected = [{"group_id": "g1"}, {"group_id": "g2"}]
        self.predictions = [
            {"group_id": "g1", "items": []},
            {"group_id": "g2", "items": []},
        ]
        self._write_indexes()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_indexes(self):
        self.selected_path.write_text(
            json.dumps(self.selected), encoding="utf-8"
        )
        self.predictions_path.write_text(
            "".join(json.dumps(record) + "\n" for record in self.predictions),
            encoding="utf-8",
        )

    def _publish(self, **overrides):
        kwargs = {
            "run_kind": "pvs_bbox_grouped_eval",
            "selected_manifest_path": "selected_groups.json",
            "expected_selected_ids": ["g1", "g2"],
            "expected_prediction_ids": ["g1", "g2"],
            "metadata": {"model_sha256": "a" * 64},
        }
        kwargs.update(overrides)
        return eval_utils.write_complete_run_manifest(self.run, **kwargs)

    def test_complete_manifest_roundtrip_verifies_both_index_hashes(self):
        manifest_path = self._publish()

        manifest, selected, predictions = eval_utils.read_complete_run_manifest(
            self.run
        )

        self.assertEqual(manifest_path, self.run / "run_manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["selected_manifest"]["entry_count"], 2)
        self.assertEqual(manifest["predictions_jsonl"]["record_count"], 2)
        self.assertRegex(
            manifest["selected_manifest"]["identity_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            manifest["predictions_jsonl"]["identity_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(selected, self.selected)
        self.assertEqual(predictions, self.predictions)
        self.assertEqual(
            manifest["selected_manifest"]["sha256"],
            eval_utils.file_sha256(self.selected_path),
        )
        self.assertEqual(
            manifest["predictions_jsonl"]["sha256"],
            eval_utils.file_sha256(self.predictions_path),
        )

    def test_changed_index_or_incomplete_manifest_is_rejected(self):
        self._publish()
        self.predictions_path.write_text(
            self.predictions_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            eval_utils.read_complete_run_manifest(self.run)

        self._write_indexes()
        manifest_path = self.run / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_identity_hash = manifest["predictions_jsonl"]["identity_sha256"]
        manifest["predictions_jsonl"]["identity_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity SHA-256 mismatch"):
            eval_utils.read_complete_run_manifest(self.run)

        manifest["predictions_jsonl"]["identity_sha256"] = original_identity_hash
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not complete"):
            eval_utils.read_complete_run_manifest(self.run)

    def test_writer_rejects_wrong_or_reordered_exact_identities(self):
        invalid_expectations = [
            {"expected_selected_ids": ["g2", "g1"]},
            {"expected_prediction_ids": ["g1", "missing"]},
            {"expected_prediction_ids": ["g1", "g1"]},
            {"expected_selected_ids": []},
            {"expected_prediction_ids": ["g1", 2]},
        ]
        for overrides in invalid_expectations:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self._publish(**overrides)
                self.assertFalse((self.run / "run_manifest.json").exists())

        self._publish()
        with self.assertRaises(FileExistsError):
            self._publish()

    def test_non_string_run_identities_are_never_coerced(self):
        self.selected[0]["group_id"] = 1
        self._write_indexes()
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            self._publish(expected_selected_ids=["1", "g2"])
        self.assertFalse((self.run / "run_manifest.json").exists())

        self.selected[0]["group_id"] = "g1"
        self.predictions[0]["group_id"] = 1
        self._write_indexes()
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            self._publish(expected_prediction_ids=["1", "g2"])
        self.assertFalse((self.run / "run_manifest.json").exists())

    def test_duplicate_or_nonfinite_prediction_records_are_rejected(self):
        invalid_payloads = [
            '{"group_id":"g1"}\n{"group_id":"g1"}\n',
            '{"group_id":"g1","score":NaN}\n{"group_id":"g2"}\n',
            '{"group_id":"g1"}\n\n{"group_id":"g2"}\n',
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temporary:
                    run = Path(temporary)
                    (run / "selected_groups.json").write_text(
                        json.dumps(self.selected), encoding="utf-8"
                    )
                    (run / "predictions.jsonl").write_text(
                        payload, encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        eval_utils.write_complete_run_manifest(
                            run,
                            run_kind="test",
                            selected_manifest_path="selected_groups.json",
                            expected_selected_ids=["g1", "g2"],
                            expected_prediction_ids=["g1", "g2"],
                        )


if __name__ == "__main__":
    unittest.main()
