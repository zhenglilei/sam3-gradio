import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import layout_region_utils as regions
import sam3_gradio_demo as demo_module


def _signature(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _signature(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    return value


class LayoutRegionPvsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout_masks = self.root / "layout_masks"
        self.layout_regions = self.root / "layout_regions"
        self.session_id = "session1"
        self.layout_id = "layout1"
        self.source_mask = np.zeros((36, 48), dtype=np.uint8)
        self.source_mask[5:31, 8:40] = 1
        self.source_hash = regions.mask_pixel_sha256(self.source_mask)
        layout_dir = self.layout_masks / self.session_id / self.layout_id
        layout_dir.mkdir(parents=True)
        cv2.imwrite(str(layout_dir / "source_mask.png"), self.source_mask * 255)
        (layout_dir / "layout_meta.json").write_text(
            json.dumps(
                {"source_mask_pixel_sha256": self.source_hash}
            ),
            encoding="utf-8",
        )
        self.store = regions.LayoutRegionStore(
            layout_masks_root=self.layout_masks,
            layout_regions_root=self.layout_regions,
            categories_path=ROOT / "layout_categories.json",
        )
        self.old_store = demo_module._LAYOUT_REGION_STORE
        demo_module._LAYOUT_REGION_STORE = self.store
        self.image = Image.new("RGB", (48, 36), (32, 48, 64))
        self.image_state = {
            "image_id": "target-image",
            "session_id": self.session_id,
            "width": self.image.width,
            "height": self.image.height,
            "target_image_sha256": (
                demo_module._layout_tx.image_pixel_sha256(self.image)
            ),
        }
        self.layout_state = {
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
        }

    def tearDown(self):
        demo_module._LAYOUT_REGION_STORE = self.old_store
        self.temporary.cleanup()

    def _save_two_regions(self):
        document, _ = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=[[8, 5], [22, 5], [22, 30], [8, 30]],
            class_label="metal",
            name="left",
        )
        document, _ = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=document["regions_revision"],
            lasso_polygon=[[26, 5], [39, 5], [39, 30], [26, 30]],
            class_label="metal",
            name="right",
        )
        return document

    def _selection_state(self, document, class_label="metal"):
        state = copy.deepcopy(self.layout_state)
        records = regions.active_regions_for_class(document, class_label)
        state.update(
            {
                "prompt_mask_scope": (
                    demo_module._LAYOUT_PROMPT_SCOPE_REGION_CLASS
                ),
                "prompt_class_label": class_label,
                "prompt_regions_revision": document["regions_revision"],
                "prompt_region_ids": [
                    int(record["region_id"]) for record in records
                ],
            }
        )
        return state

    def _prediction(self, offset=0):
        masks = np.zeros((2, 36, 48), dtype=bool)
        masks[0, 2:8, 2:8] = True
        masks[1, 8 + offset:18 + offset, 12:28] = True
        return {
            "masks": masks,
            "scores": np.asarray([0.2, 0.9], dtype=np.float32),
            "lowres_logits": np.stack(
                [
                    np.full((4, 4), 10 + offset, dtype=np.float32),
                    np.full((4, 4), 20 + offset, dtype=np.float32),
                ]
            ),
        }

    def _run_batch(self, document, pvs_state, predict_side_effect=None):
        state = self._selection_state(document)
        decoded_inputs = []
        identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        transform = {
            "revision": 7,
            "matrix_2x3": identity_matrix,
            "source_mask_pixel_sha256": self.source_hash,
            "target_image_sha256": self.image_state[
                "target_image_sha256"
            ],
        }

        def commit(
            _image_state,
            incoming_state,
            *_args,
            **_kwargs,
        ):
            committed = dict(incoming_state)
            committed["revision"] = 7
            return committed, self.source_mask.astype(bool), copy.deepcopy(
                transform
            )

        def to_logits(mask):
            binary = np.asarray(mask, dtype=bool)
            decoded_inputs.append(binary.copy())
            lowres = cv2.resize(
                binary.astype(np.uint8),
                (4, 4),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            return (lowres * 20.0 - 10.0).astype(np.float32)

        predictions = (
            predict_side_effect
            if predict_side_effect is not None
            else [self._prediction(0), self._prediction(2)]
        )
        with (
            mock.patch.object(
                demo_module,
                "_commit_layout_transform",
                side_effect=commit,
            ),
            mock.patch.object(
                demo_module,
                "_layout_prompt_metadata",
                return_value={
                    "type": "layout_mask",
                    "session_id": self.session_id,
                    "layout_id": self.layout_id,
                    "source_mask_pixel_sha256": self.source_hash,
                    "target_width": 48,
                    "target_height": 36,
                },
            ),
            mock.patch.object(
                demo_module,
                "_mask_to_lowres_logits",
                side_effect=to_logits,
            ),
            mock.patch.object(
                demo_module,
                "_prompt_mask_size",
                return_value=(4, 4),
            ),
            mock.patch.object(
                demo_module,
                "_fresh_state",
                return_value={"fresh": True},
            ),
            mock.patch.object(
                demo_module,
                "_predict_inst",
                side_effect=predictions,
            ) as predictor,
            mock.patch.object(
                demo_module,
                "_validate_layout_transform_snapshot",
            ),
            mock.patch.object(
                demo_module,
                "_workspace",
                return_value={"image": self.image},
            ),
            mock.patch.object(
                demo_module,
                "_layout_editor_payload",
                return_value={"editor": True},
            ),
            mock.patch.object(
                demo_module,
                "_view",
                return_value=(None,) * 8,
            ),
            mock.patch.object(demo_module, "_pvs_progress"),
        ):
            result = demo_module._create_pvs_from_layout_selection(
                self.image_state,
                demo_module._new_pcs_state(),
                pvs_state,
                demo_module.MODE_LAYOUT,
                state,
                True,
                0.0,
                0.0,
                1.0,
                0.0,
                0.35,
                {"transform": transform},
                demo_module._layout_prompt_selection_token(
                    demo_module._LAYOUT_PROMPT_SCOPE_REGION_CLASS,
                    "metal",
                ),
                progress=None,
            )
        return result, decoded_inputs, predictor

    def test_same_class_regions_create_independent_instances(self):
        document = self._save_two_regions()
        pvs_state = demo_module._new_pvs_state()
        result, prompt_masks, predictor = self._run_batch(
            document,
            pvs_state,
        )

        candidate_state = result[0]
        self.assertIsNot(candidate_state, pvs_state)
        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(sorted(candidate_state["instances"]), [1, 2])
        self.assertEqual(candidate_state["next_instance_id"], 3)
        self.assertEqual(candidate_state["active_instance_id"], 2)
        expected_masks = [
            mask
            for _, mask in regions.decode_region_masks(
                regions.active_regions_for_class(document, "metal"),
                self.source_mask.shape,
            )
        ]
        self.assertEqual(len(prompt_masks), 2)
        for actual, expected in zip(prompt_masks, expected_masks):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(prompt_masks[0], prompt_masks[1]))

        prompts = [
            candidate_state["instances"][instance_id][
                "prompt_history"
            ][0]["prompt"]
            for instance_id in (1, 2)
        ]
        self.assertEqual([item["region_id"] for item in prompts], [1, 2])
        self.assertEqual(
            [item["batch_index"] for item in prompts],
            [1, 2],
        )
        self.assertTrue(
            all(item["batch_region_ids"] == [1, 2] for item in prompts)
        )
        self.assertTrue(
            all(item["mask_scope"] == "region" for item in prompts)
        )
        self.assertTrue(
            all(
                item["matrix_2x3"]
                == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                for item in prompts
            )
        )
    def test_later_prediction_failure_leaves_pvs_state_unchanged(self):
        document = self._save_two_regions()
        pvs_state = demo_module._new_pvs_state()
        before = _signature(pvs_state)
        result, _, predictor = self._run_batch(
            document,
            pvs_state,
            predict_side_effect=[
                self._prediction(0),
                RuntimeError("second Region failed"),
            ],
        )

        self.assertEqual(predictor.call_count, 2)
        self.assertIs(result[0], pvs_state)
        self.assertEqual(_signature(pvs_state), before)
        self.assertIn("整批未提交", result[3])

    def test_empty_best_candidate_aborts_entire_batch(self):
        document = self._save_two_regions()
        pvs_state = demo_module._new_pvs_state()
        before = _signature(pvs_state)
        empty_best = self._prediction(2)
        empty_best["masks"][1] = False
        result, _, predictor = self._run_batch(
            document,
            pvs_state,
            predict_side_effect=[self._prediction(0), empty_best],
        )
        self.assertEqual(predictor.call_count, 2)
        self.assertIs(result[0], pvs_state)
        self.assertEqual(_signature(pvs_state), before)
        self.assertIn("最佳候选 mask 为空", result[3])

    def test_region_revision_change_discards_completed_batch(self):
        document = self._save_two_regions()
        pvs_state = demo_module._new_pvs_state()
        before = _signature(pvs_state)
        call_count = 0

        def predict_and_delete(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                self.store.delete_region(
                    session_id=self.session_id,
                    layout_id=self.layout_id,
                    source_mask_hash=self.source_hash,
                    expected_revision=document["regions_revision"],
                    region_id=2,
                )
            return self._prediction(call_count - 1)

        result, _, predictor = self._run_batch(
            document,
            pvs_state,
            predict_side_effect=predict_and_delete,
        )
        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(result[0].get("__type__"), "update")
        self.assertEqual(_signature(pvs_state), before)
        self.assertIn("Region", result[3])
        self.assertIn("整批未提交", result[3])

    def test_selection_epoch_change_discards_completed_batch(self):
        document = self._save_two_regions()
        pvs_state = demo_module._new_pvs_state()
        before = _signature(pvs_state)
        call_count = 0

        def predict_and_switch_selection(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                demo_module._advance_layout_prompt_epoch(
                    self.image_state,
                    self._selection_state(document),
                )
            return self._prediction(call_count - 1)

        result, _, predictor = self._run_batch(
            document,
            pvs_state,
            predict_side_effect=predict_and_switch_selection,
        )
        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(result[0].get("__type__"), "update")
        self.assertEqual(_signature(pvs_state), before)
        self.assertIn("identity", result[3])
        self.assertIn("整批未提交", result[3])

    def test_canvas_transform_actions_discard_completed_batch(self):
        document = self._save_two_regions()
        actions = {
            "canvas sync": lambda state: (
                demo_module
                ._sync_layout_controls_from_editor_with_prompt_epoch(
                    state,
                    {
                        "enabled": True,
                        "target_width": 48,
                        "target_height": 36,
                        "transform": {
                            "session_id": self.session_id,
                            "layout_id": self.layout_id,
                            "center_x": 24.0,
                            "center_y": 18.0,
                            "pivot_x": 24.0,
                            "pivot_y": 18.0,
                            "scale": 1.25,
                            "rotation_deg": 5.0,
                            "preview_alpha": 0.35,
                            "revision": 7,
                            "source_mask_pixel_sha256": self.source_hash,
                        },
                    },
                )
            ),
            "reset": lambda state: (
                demo_module._reset_layout_controls_with_prompt_epoch(
                    self.image_state,
                    state,
                )
            ),
        }

        for action_name, action in actions.items():
            with self.subTest(action=action_name):
                pvs_state = demo_module._new_pvs_state()
                before = _signature(pvs_state)
                call_count = 0

                def predict_and_change(*_args, **_kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 2:
                        action(self._selection_state(document))
                    return self._prediction(call_count - 1)

                result, _, predictor = self._run_batch(
                    document,
                    pvs_state,
                    predict_side_effect=predict_and_change,
                )
                self.assertEqual(predictor.call_count, 2)
                self.assertEqual(result[0].get("__type__"), "update")
                self.assertEqual(len(result), 12)
                self.assertEqual(result[3], result[10])
                for index in (0, 1, 2, 4, 5, 6, 7, 8, 9, 11):
                    self.assertEqual(result[index].get("__type__"), "update")
                self.assertEqual(_signature(pvs_state), before)
                self.assertIn("identity", result[3])
                self.assertIn("整批未提交", result[3])

    def test_multicomponent_region_still_predicts_once(self):
        mask = np.zeros_like(self.source_mask, dtype=bool)
        mask[7:13, 10:16] = True
        mask[20:27, 30:37] = True
        document = regions.new_regions_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        document, record = regions.append_region(
            document,
            class_label="metal",
            name="two-components",
            region_mask=mask,
            allowed_categories=["metal"],
        )
        self.assertEqual(record["component_count"], 2)
        regions.write_json_atomic(
            self.store.regions_path(self.session_id, self.layout_id),
            document,
        )

        result, prompt_masks, predictor = self._run_batch(
            document,
            demo_module._new_pvs_state(),
            predict_side_effect=[self._prediction(0)],
        )
        self.assertEqual(predictor.call_count, 1)
        self.assertEqual(len(prompt_masks), 1)
        self.assertEqual(len(result[0]["instances"]), 1)
        self.assertEqual(
            result[0]["instances"][1]["prompt_history"][0]["prompt"][
                "region_id"
            ],
            1,
        )

    def test_full_selection_delegates_and_missing_selection_is_rejected(self):
        pvs_state = demo_module._new_pvs_state()
        full_state = copy.deepcopy(self.layout_state)
        full_state["prompt_mask_scope"] = (
            demo_module._LAYOUT_PROMPT_SCOPE_FULL
        )
        sentinel = tuple(range(12))
        with (
            mock.patch.object(
                demo_module,
                "_create_pvs_from_layout_mask",
                return_value=sentinel,
            ) as full_create,
            mock.patch.object(
                demo_module,
                "_layout_editor_payload",
                return_value={},
            ),
            mock.patch.object(
                demo_module,
                "_view",
                return_value=(None,) * 8,
            ),
        ):
            result = demo_module._create_pvs_from_layout_selection(
                self.image_state,
                demo_module._new_pcs_state(),
                pvs_state,
                demo_module.MODE_LAYOUT,
                full_state,
                True,
                0.0,
                0.0,
                1.0,
                0.0,
                0.35,
                {},
                demo_module._LAYOUT_PROMPT_SCOPE_FULL,
                progress=None,
            )
            rejected = demo_module._create_pvs_from_layout_selection(
                self.image_state,
                demo_module._new_pcs_state(),
                pvs_state,
                demo_module.MODE_LAYOUT,
                full_state,
                True,
                0.0,
                0.0,
                1.0,
                0.0,
                0.35,
                {},
                None,
                progress=None,
            )

        self.assertEqual(result, sentinel)
        full_create.assert_called_once()
        self.assertIs(rejected[0], pvs_state)
        self.assertIn("选择无效", rejected[3])

    def test_selector_failure_restores_browser_value(self):
        document = self._save_two_regions()
        full_state = copy.deepcopy(self.layout_state)
        full_state["prompt_mask_scope"] = (
            demo_module._LAYOUT_PROMPT_SCOPE_FULL
        )
        failed = demo_module._select_layout_prompt_mask(
            self.image_state,
            full_state,
            demo_module._layout_prompt_selection_token(
                demo_module._LAYOUT_PROMPT_SCOPE_REGION_CLASS,
                "via",
            ),
        )
        self.assertEqual(len(failed), 4)
        self.assertEqual(
            failed[0]["prompt_mask_scope"],
            demo_module._LAYOUT_PROMPT_SCOPE_FULL,
        )
        self.assertEqual(
            failed[2]["value"],
            demo_module._LAYOUT_PROMPT_SCOPE_FULL,
        )
        self.assertIn("已保留原选择", failed[3])

        selected = demo_module._select_layout_prompt_mask(
            self.image_state,
            full_state,
            demo_module._layout_prompt_selection_token(
                demo_module._LAYOUT_PROMPT_SCOPE_REGION_CLASS,
                "metal",
            ),
        )
        self.assertEqual(
            selected[0]["prompt_region_ids"],
            [record["region_id"] for record in document["regions"]],
        )
        self.assertEqual(
            selected[2]["value"],
            demo_module._layout_prompt_selection_token(
                demo_module._LAYOUT_PROMPT_SCOPE_REGION_CLASS,
                "metal",
            ),
        )
    def test_feedback_reconstructs_frozen_region_after_soft_delete(self):


        document = self._save_two_regions()
        record = regions.active_regions_for_class(document, "metal")[0]
        region_mask = regions.decode_binary_mask(
            record["mask_rle"],
            self.source_mask.shape,
        )
        prompt = {
            "type": "layout_mask",
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
            "target_width": 54,
            "target_height": 42,
            "matrix_2x3": [
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 2.0],
            ],
            "mask_scope": "region",
            "class_label": "metal",
            "region_id": int(record["region_id"]),
            "region_mask_pixel_sha256": regions.mask_pixel_sha256(
                region_mask.astype(np.uint8)
            ),
            "regions_revision": document["regions_revision"],
        }
        document, _ = self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=document["regions_revision"],
            region_id=int(record["region_id"]),
        )
        self.assertGreater(
            document["regions_revision"],
            prompt["regions_revision"],
        )

        sample_dir = self.root / "feedback"
        sample_dir.mkdir()
        artifacts = demo_module._write_feedback_layout_artifacts(
            sample_dir,
            prompt,
        )
        reconstructed = cv2.imread(
            artifacts["layout_transformed_mask_file"],
            cv2.IMREAD_GRAYSCALE,
        )
        expected = demo_module._layout_tx.warp_layout_mask(
            region_mask,
            prompt["matrix_2x3"],
            (prompt["target_width"], prompt["target_height"]),
        )
        np.testing.assert_array_equal(reconstructed >= 128, expected)
        metadata = json.loads(
            Path(artifacts["layout_transform_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            metadata["reconstruction_method"],
            "frozen_region_rle",
        )
        self.assertEqual(metadata["reconstruction_status"], "ok")
        self.assertIsNotNone(metadata["region_deleted_at"])


    def test_legacy_full_feedback_ignores_corrupt_region_document(self):
        self._save_two_regions()
        self.store.regions_path(
            self.session_id,
            self.layout_id,
        ).write_text("{broken", encoding="utf-8")
        prompt = {
            "type": "layout_mask",
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
            "target_width": 48,
            "target_height": 36,
            "matrix_2x3": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
        sample_dir = self.root / "feedback-full"
        sample_dir.mkdir()
        artifacts = demo_module._write_feedback_layout_artifacts(
            sample_dir,
            prompt,
        )
        reconstructed = cv2.imread(
            artifacts["layout_transformed_mask_file"],
            cv2.IMREAD_GRAYSCALE,
        )
        np.testing.assert_array_equal(
            reconstructed >= 128,
            self.source_mask.astype(bool),
        )
        metadata = json.loads(
            Path(artifacts["layout_transform_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            metadata["reconstruction_method"],
            "frozen_full_mask",
        )
        self.assertEqual(metadata["reconstruction_status"], "ok")

    def test_feedback_hash_mismatch_does_not_fallback_to_full_mask(self):
        document = self._save_two_regions()
        record = regions.active_regions_for_class(document, "metal")[0]
        prompt = {
            "type": "layout_mask",
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
            "target_width": 48,
            "target_height": 36,
            "matrix_2x3": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "mask_scope": "region",
            "class_label": "metal",
            "region_id": int(record["region_id"]),
            "region_mask_pixel_sha256": "0" * 64,
        }
        sample_dir = self.root / "feedback-bad-hash"
        sample_dir.mkdir()
        artifacts = demo_module._write_feedback_layout_artifacts(
            sample_dir,
            prompt,
        )
        self.assertIsNone(artifacts["layout_transformed_mask_file"])
        self.assertFalse(
            (sample_dir / "layout_transformed_mask.png").exists()
        )
        metadata = json.loads(
            Path(artifacts["layout_transform_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            metadata["reconstruction_status"],
            "unavailable",
        )
        self.assertIn("hash", metadata["reconstruction_error"])


if __name__ == "__main__":
    unittest.main()
