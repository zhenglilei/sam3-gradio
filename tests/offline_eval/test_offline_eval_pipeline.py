from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts import offline_eval_utils as eval_utils
from scripts import run_mask_iou95_eval as scorer
from scripts import run_pcs_o3_grouped_eval as pcs


class FakePcsPredictor:
    def __init__(self, masks: list[np.ndarray], boxes: list[list[float]]) -> None:
        self.masks = np.stack(masks)
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.thresholds: list[float] = []

    def set_image(self, _image):
        return {
            "masks": self.masks,
            "boxes": self.boxes,
            "scores": np.asarray([0.91, 0.87], dtype=np.float32),
        }

    def add_geometric_prompt(self, _box, _positive, state):
        return state

    def set_confidence_threshold(self, threshold, state):
        self.thresholds.append(float(threshold))
        return state


class OfflineEvaluationPipelineTest(unittest.TestCase):
    def _write_o3_fixture(self, root: Path):
        dataset = root / "ACT_coco"
        annotation_path = dataset / "annotations" / "instances_val.json"
        image_path = dataset / "val" / "x.png"
        annotation_path.parent.mkdir(parents=True)
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)
        document = {
            "categories": [{"id": 7, "name": "metal"}],
            "images": [
                {
                    "id": 11,
                    "file_name": "x.png",
                    "width": 12,
                    "height": 8,
                }
            ],
            "annotations": [
                {
                    "id": 21,
                    "image_id": 11,
                    "category_id": 7,
                    "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                    "area": 9,
                    "bbox": [1, 1, 3, 3],
                    "iscrowd": 0,
                },
                {
                    "id": 22,
                    "image_id": 11,
                    "category_id": 7,
                    "segmentation": [[6, 1, 9, 1, 9, 4, 6, 4]],
                    "area": 9,
                    "bbox": [6, 1, 3, 3],
                    "iscrowd": 0,
                },
            ],
        }
        annotation_path.write_text(json.dumps(document), encoding="utf-8")
        annotation_hash = eval_utils.file_sha256(annotation_path)
        image_hash = eval_utils.file_sha256(image_path)
        masks = []
        for annotation_id in ("21", "22"):
            mask, _, _ = eval_utils.load_validated_coco_ground_truth(
                annotation_path,
                annotation_id,
                expected_dataset="ACT_coco",
                expected_split="val",
                expected_image_id="11",
                image_path=image_path,
                expected_annotation_sha256=annotation_hash,
                expected_image_sha256=image_hash,
                expected_category_label="metal",
            )
            masks.append(mask)
        return annotation_path, image_path, annotation_hash, image_hash, masks

    def _write_pvs_run(
        self,
        run_dir: Path,
        image_path: Path,
        annotation_hash: str,
        image_hash: str,
        mask: np.ndarray,
    ) -> None:
        eval_utils.prepare_empty_output_dir(run_dir)
        item = {
            "sample_id": "o3_ACT_val_11_x__ann21",
            "source": "O3",
            "dataset": "ACT_coco",
            "split": "val",
            "image_path": str(image_path),
            "image_id": "11",
            "annotation_id": "21",
            "label": "metal",
            "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
            "category_label": "metal",
            "annotation_relpath": "ACT_coco/annotations/instances_val.json",
            "annotation_sha256": annotation_hash,
            "image_sha256": image_hash,
            "label_shape_type": "",
        }
        group = {
            "group_id": "o3_ACT_val_11_x",
            "source": "O3",
            "layer": "ACT",
            "dataset": "ACT_coco",
            "split": "val",
            "image_path": str(image_path),
            "image_id": "11",
            "annotation_relpath": "ACT_coco/annotations/instances_val.json",
            "annotation_sha256": annotation_hash,
            "image_sha256": image_hash,
            "items": [item],
            "category_label": "ACT",
            "category_labels": ["metal"],
        }
        artifact = eval_utils.save_binary_mask(
            run_dir, "groups/o3_ACT_val_11_x/masks/0001.png", mask
        )
        prediction = copy.deepcopy(group)
        prediction["items"][0].update(
            {
                "output_label": "metal",
                "score": 0.93,
                "pred_bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
                "best_index": 0,
                "candidate_scores": [0.93],
                "mask_artifact": artifact,
            }
        )
        eval_utils.write_json_atomic(run_dir / "selected_groups.json", [group])
        (run_dir / "predictions.jsonl").write_text(
            json.dumps(prediction) + "\n", encoding="utf-8"
        )
        eval_utils.write_complete_run_manifest(
            run_dir,
            run_kind="pvs_bbox_grouped_eval",
            selected_manifest_path="selected_groups.json",
            expected_selected_ids=[group["group_id"]],
            expected_prediction_ids=[group["group_id"]],
            metadata={"source_group_counts": {"O3": 1}},
        )

    def _write_pcs_run(
        self,
        run_dir: Path,
        image_path: Path,
        annotation_hash: str,
        image_hash: str,
        masks: list[np.ndarray],
    ) -> FakePcsPredictor:
        boxes = [[1.0, 1.0, 4.0, 4.0], [6.0, 1.0, 9.0, 4.0]]
        group = pcs.PcsCategoryImage(
            group_id="pcs_o3_ACT_metal_01_x",
            layer="ACT",
            dataset="ACT_coco",
            split="val",
            label="metal",
            image_path=str(image_path),
            image_id="11",
            image_file_name="x.png",
            category_id=7,
            boxes=boxes,
            annotation_ids=["21", "22"],
            annotation_relpath="ACT_coco/annotations/instances_val.json",
            annotation_sha256=annotation_hash,
            image_sha256=image_hash,
        )
        predictor = FakePcsPredictor(masks, boxes)
        with mock.patch.object(
            pcs.base, "init_image_predictor", return_value=predictor
        ):
            pcs.run_groups(
                [group],
                run_dir,
                prompt_counts=[1],
                threshold=0.73,
                device="cpu",
                dry_run=False,
            )
        return predictor

    def test_fixed_artifacts_are_scored_without_reinference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o3_root = root / "O3_coco"
            _, image_path, ann_hash, image_hash, masks = self._write_o3_fixture(
                o3_root
            )
            pvs_run = root / "pvs_run"
            pcs_run = root / "pcs_run"
            self._write_pvs_run(
                pvs_run, image_path, ann_hash, image_hash, masks[0]
            )
            predictor = self._write_pcs_run(
                pcs_run, image_path, ann_hash, image_hash, masks
            )
            self.assertEqual(predictor.thresholds, [0.73])

            _, _, pcs_predictions = eval_utils.read_complete_run_manifest(pcs_run)
            self.assertEqual(pcs_predictions[0]["confidence"], 0.73)
            self.assertEqual(len(pcs_predictions[0]["pred_instances"]), 2)
            for instance in pcs_predictions[0]["pred_instances"]:
                eval_utils.load_binary_mask(
                    pcs_run, instance["mask_artifact"]
                )

            for run_dir in (pvs_run, pcs_run):
                stale = run_dir / "groups" / "stale_unindexed"
                stale.mkdir(parents=True)
                (stale / "prediction.json").write_text("{}", encoding="utf-8")

            out_dir = root / "evaluation"
            argv = [
                "run_mask_iou95_eval.py",
                "--pvs-dir",
                str(pvs_run),
                "--pcs-dir",
                str(pcs_run),
                "--o3-root",
                str(o3_root),
                "--out-dir",
                str(out_dir),
                "--threshold",
                "0.95",
                "--device",
                "cuda",
                "--only",
                "all",
            ]
            with mock.patch.object(sys, "argv", argv):
                scorer.main()

            pvs_rows = [
                json.loads(line)
                for line in (
                    out_dir / "pvs" / "pvs_mask_iou95_predictions.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            pcs_rows = [
                json.loads(line)
                for line in (
                    out_dir / "pcs" / "pcs_mask_iou95_runs.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(pvs_rows), 1)
            self.assertEqual(pvs_rows[0]["mask_iou"], 1.0)
            self.assertEqual(pvs_rows[0]["success"], 1)
            self.assertEqual(len(pcs_rows), 1)
            self.assertEqual(pcs_rows[0]["matched_count"], 2)
            self.assertEqual(pcs_rows[0]["strict_all_gt_success"], 1)

            manifest = json.loads(
                (out_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["model_inference_performed"])
            self.assertEqual(manifest["deprecated_device_argument"], "cuda")
            self.assertEqual(manifest["outputs"]["pvs_items"]["row_count"], 1)
            self.assertEqual(manifest["outputs"]["pcs_runs"]["row_count"], 1)

    def test_pcs_selected_group_without_prediction_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o3_root = root / "O3_coco"
            _, image_path, ann_hash, image_hash, masks = self._write_o3_fixture(
                o3_root
            )
            pcs_run = root / "pcs_run"
            self._write_pcs_run(
                pcs_run, image_path, ann_hash, image_hash, masks
            )
            _, selected, predictions = eval_utils.read_complete_run_manifest(
                pcs_run
            )
            missing_group = copy.deepcopy(selected[0])
            missing_group["group_id"] = "pcs_o3_ACT_metal_02_x"
            output = root / "evaluation"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "no prediction runs"):
                scorer.evaluate_pcs(
                    pcs_run,
                    selected + [missing_group],
                    predictions,
                    output,
                    o3_root=o3_root,
                    threshold=0.95,
                )


    def test_legacy_directory_without_complete_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy = Path(temporary) / "legacy"
            (legacy / "groups" / "old").mkdir(parents=True)
            (legacy / "groups" / "old" / "prediction.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "regenerated|Legacy|incomplete"):
                scorer._load_producer_run(legacy, "pvs_bbox_grouped_eval")

    def test_scorer_source_has_no_model_or_directory_glob_fallback(self):
        source = Path(scorer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("init_image_predictor", source)
        self.assertNotIn("predict_from_box", source)
        self.assertNotIn(".glob(", source)
        self.assertNotIn(".rglob(", source)


if __name__ == "__main__":
    unittest.main()
