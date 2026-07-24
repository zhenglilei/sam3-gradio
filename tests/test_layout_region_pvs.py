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
            label="left",
        )
        document, _ = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=document["regions_revision"],
            lasso_polygon=[[26, 5], [39, 5], [39, 30], [26, 30]],
            label="right",
        )
        return document

    def _selection_state(self, document, region_ids=None):
        records = regions.active_regions(document)
        if region_ids is not None:
            wanted = {int(region_id) for region_id in region_ids}
            records = [
                record
                for record in records
                if int(record["region_id"]) in wanted
            ]
        state = copy.deepcopy(self.layout_state)
        state.update(
            {
                "prompt_mask_scope": (
                    demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS
                ),
                "prompt_class_label": None,
                "prompt_labels": [
                    regions.region_label(record) for record in records
                ],
                "prompt_regions_revision": document["regions_revision"],
                "prompt_region_ids": [
                    int(record["region_id"]) for record in records
                ],
                "image_id": self.image_state["image_id"],
                "target_image_sha256": self.image_state[
                    "target_image_sha256"
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

    def _run_batch(
        self,
        document,
        pvs_state,
        predict_side_effect=None,
        *,
        controls=None,
        commit_capture=None,
    ):
        state = self._selection_state(document)
        records = regions.active_regions(document)
        decoded = regions.decode_region_masks(
            records,
            self.source_mask.shape,
        )
        decoded_inputs = []
        group_matrices = {
            demo_module._layout_prompt_group_id(1): [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            demo_module._layout_prompt_group_id(2): [
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 0.0],
            ],
        }
        transforms = {}
        for record, _ in decoded:
            group_id = demo_module._layout_prompt_group_id(
                record["region_id"]
            )
            transforms[group_id] = {
                "revision": 7,
                "group_id": group_id,
                "region_id": int(record["region_id"]),
                "label": regions.region_label(record),
                "matrix_2x3": copy.deepcopy(group_matrices[group_id]),
                "preview_alpha": 0.35,
                "source_mask_pixel_sha256": self.source_hash,
                "target_image_sha256": self.image_state[
                    "target_image_sha256"
                ],
            }
        snapshot = {
            "selection_signature": "labels-selection-v1",
            "regions_revision": int(document["regions_revision"]),
            "transform_set_revision": 8,
            "active_group_id": demo_module._layout_prompt_group_id(
                records[-1]["region_id"]
            ),
            "region_ids": [
                int(record["region_id"]) for record in records
            ],
            "labels": [
                regions.region_label(record) for record in records
            ],
            "target_image_sha256": self.image_state[
                "target_image_sha256"
            ],
            "transforms": copy.deepcopy(transforms),
        }

        def commit(_image_state, incoming_state, *_args, **kwargs):
            numeric_override = kwargs.get("numeric_override")
            if commit_capture is not None:
                commit_capture["numeric_override"] = numeric_override
            effective = copy.deepcopy(transforms)
            effective_snapshot = copy.deepcopy(snapshot)
            if controls is not None and numeric_override is not None:
                active_group_id = effective_snapshot["active_group_id"]
                active_record, active_mask = next(
                    (record, mask)
                    for record, mask in decoded
                    if demo_module._layout_prompt_group_id(
                        record["region_id"]
                    )
                    == active_group_id
                )
                tx, ty, active_scale, active_rotation, active_alpha = (
                    numeric_override
                )
                pivot = demo_module._layout_tx.pivot_from_bbox_xyxy(
                    demo_module._layout_tx.foreground_bbox_xyxy(
                        active_mask
                    )
                )
                active_transform = (
                    demo_module._layout_tx.make_layout_transform_v2(
                        session_id=self.session_id,
                        layout_id=self.layout_id,
                        image_id=self.image_state["image_id"],
                        target_size=(self.image.width, self.image.height),
                        source_mask=active_mask,
                        center_x=self.image.width / 2.0 + float(tx),
                        center_y=self.image.height / 2.0 + float(ty),
                        pivot_xy=pivot,
                        scale=float(active_scale),
                        rotation_deg=float(active_rotation),
                        preview_alpha=float(active_alpha),
                        revision=effective[active_group_id]["revision"] + 1,
                        source_mask_pixel_sha256=self.source_hash,
                        target_image_sha256=self.image_state[
                            "target_image_sha256"
                        ],
                    )
                )
                active_transform = (
                    demo_module._layout_tx.transform_with_derived_fields(
                        active_transform,
                        (self.image.width, self.image.height),
                    )
                )
                active_transform.update(
                    {
                        "group_id": active_group_id,
                        "region_id": int(active_record["region_id"]),
                        "label": regions.region_label(active_record),
                    }
                )
                effective[active_group_id] = active_transform
                effective_snapshot["transforms"] = copy.deepcopy(
                    effective
                )
                effective_snapshot["transform_set_revision"] += 1

            committed = dict(incoming_state)
            committed.update(
                {
                    "prompt_group_transforms": copy.deepcopy(effective),
                    "prompt_active_group_id": effective_snapshot[
                        "active_group_id"
                    ],
                    "prompt_selection_signature": effective_snapshot[
                        "selection_signature"
                    ],
                    "prompt_transform_set_revision": effective_snapshot[
                        "transform_set_revision"
                    ],
                }
            )
            transformed = []
            union = np.zeros_like(self.source_mask, dtype=bool)
            for record, mask in decoded:
                mask = np.asarray(mask, dtype=bool)
                transformed.append((record, mask.copy()))
                warped = demo_module._layout_tx.warp_layout_mask(
                    mask,
                    effective[
                        demo_module._layout_prompt_group_id(
                            record["region_id"]
                        )
                    ]["matrix_2x3"],
                    (self.image.width, self.image.height),
                )
                union = np.logical_or(union, warped)
            return (
                committed,
                transformed,
                effective_snapshot,
                union,
                copy.deepcopy(
                    effective[effective_snapshot["active_group_id"]]
                ),
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
                "_commit_layout_group_transforms",
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
                "_validate_layout_group_transform_snapshot",
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
                *(controls or (0.0, 0.0, 1.0, 0.0, 0.35)),
                {
                    "group_intent": {
                        "active_group_id": snapshot["active_group_id"]
                    }
                },
                [
                    demo_module._layout_prompt_selection_token(
                        demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                        record["region_id"],
                    )
                    for record in records
                ],
                progress=None,
            )
        return result, decoded_inputs, predictor

    def test_selected_labels_use_independent_transforms_and_instances(self):
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

        records = regions.active_regions(document)
        decoded = regions.decode_region_masks(
            records,
            self.source_mask.shape,
        )
        matrices = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]],
        ]
        expected_masks = [
            demo_module._layout_tx.warp_layout_mask(
                mask,
                matrix,
                (self.image.width, self.image.height),
            )
            for (_, mask), matrix in zip(decoded, matrices)
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
        self.assertEqual([item["label"] for item in prompts], ["left", "right"])
        self.assertEqual(
            [item["group_id"] for item in prompts],
            ["region_1", "region_2"],
        )
        self.assertEqual(
            [item["batch_index"] for item in prompts],
            [1, 2],
        )
        self.assertTrue(
            all(item["batch_region_ids"] == [1, 2] for item in prompts)
        )
        self.assertTrue(
            all(item["batch_labels"] == ["left", "right"] for item in prompts)
        )
        self.assertTrue(
            all(item["mask_scope"] == "region" for item in prompts)
        )
        self.assertEqual(
            [item["matrix_2x3"] for item in prompts],
            matrices,
        )

    def test_create_applies_pending_numeric_values_to_active_label(self):
        document = self._save_two_regions()
        controls = (5.0, -3.0, 1.4, 17.0, 0.2)
        capture = {}
        result, prompt_masks, predictor = self._run_batch(
            document,
            demo_module._new_pvs_state(),
            controls=controls,
            commit_capture=capture,
        )

        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(capture["numeric_override"], controls)
        prompts = [
            result[0]["instances"][instance_id]["prompt_history"][0][
                "prompt"
            ]
            for instance_id in (1, 2)
        ]
        self.assertEqual(prompts[1]["label"], "right")
        self.assertEqual(prompts[1]["group_id"], "region_2")
        self.assertAlmostEqual(
            prompts[1]["transform"]["center_x"],
            self.image.width / 2.0 + controls[0],
        )
        self.assertAlmostEqual(
            prompts[1]["transform"]["center_y"],
            self.image.height / 2.0 + controls[1],
        )
        self.assertAlmostEqual(
            prompts[1]["transform"]["scale"],
            controls[2],
        )
        self.assertAlmostEqual(
            prompts[1]["transform"]["rotation_deg"],
            controls[3],
        )
        self.assertAlmostEqual(
            prompts[1]["transform"]["preview_alpha"],
            controls[4],
        )

        right_record = regions.active_regions(document)[1]
        right_mask = regions.decode_binary_mask(
            right_record["mask_rle"],
            self.source_mask.shape,
        )
        expected_right = demo_module._layout_tx.warp_layout_mask(
            right_mask,
            prompts[1]["matrix_2x3"],
            (self.image.width, self.image.height),
        )
        np.testing.assert_array_equal(prompt_masks[1], expected_right)
        self.assertEqual(
            prompts[0]["matrix_2x3"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        self.assertEqual(prompts[0]["transform"]["revision"], 7)
        self.assertEqual(prompts[0]["transform"]["preview_alpha"], 0.35)

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

    def test_commit_accepts_all_changed_groups_after_active_switch(self):
        source = self.source_mask.astype(bool)
        group_masks = []
        for start, stop in ((8, 18), (18, 29), (29, 40)):
            mask = np.zeros_like(source)
            mask[5:31, start:stop] = source[5:31, start:stop]
            group_masks.append(mask)
        records = [
            regions.region_record_from_mask(
                index,
                None,
                "",
                mask,
                label=label,
            )
            for index, (label, mask) in enumerate(
                zip(("A", "B", "C"), group_masks),
                start=1,
            )
        ]
        document = {
            "regions_revision": 3,
            "regions": records,
        }
        decoded = list(zip(records, group_masks))
        state = self._selection_state(document)
        state["prompt_active_group_id"] = "region_1"
        state["prompt_selection_signature"] = "stable-selection"
        state["prompt_transform_set_revision"] = 0

        target_size = (self.image.width, self.image.height)
        base_transform = demo_module._layout_tx.make_layout_transform_v2(
            session_id=self.session_id,
            layout_id=self.layout_id,
            image_id=self.image_state["image_id"],
            target_size=target_size,
            source_mask=source,
            center_x=24.0,
            center_y=18.0,
            pivot_xy=[24.0, 18.0],
            scale=1.0,
            rotation_deg=0.0,
            preview_alpha=0.35,
            revision=0,
            source_mask_pixel_sha256=self.source_hash,
            target_image_sha256=self.image_state[
                "target_image_sha256"
            ],
        )
        base_transform = (
            demo_module._layout_tx.transform_with_derived_fields(
                base_transform,
                target_size,
            )
        )
        initial = {}
        for record, mask in decoded:
            group_id = demo_module._layout_prompt_group_id(
                record["region_id"]
            )
            initial[group_id] = demo_module._layout_group_transform(
                state,
                base_transform,
                record,
                mask,
                target_size,
            )[0]
        state["prompt_group_transforms"] = copy.deepcopy(initial)

        incoming = copy.deepcopy(initial)
        incoming["region_1"]["center_x"] += 2.0
        incoming["region_1"]["revision"] += 1
        incoming["region_2"]["center_y"] += 3.0
        incoming["region_2"]["revision"] += 1
        incoming["region_3"]["center_x"] += 100.0
        incoming["region_3"]["revision"] += 9
        editor = {
            "group_intent": {
                "selection_signature": "stable-selection",
                "transform_set_revision": 1,
                "active_group_id": "region_2",
                "changed_group_ids": ["region_1", "region_2"],
                "transforms": [
                    {
                        "group_id": group_id,
                        "transform": transform,
                    }
                    for group_id, transform in incoming.items()
                ],
            }
        }
        cache = {"source_mask": source}
        base_payload = {
            "target_width": target_size[0],
            "target_height": target_size[1],
            "transform": base_transform,
        }

        with (
            mock.patch.object(
                demo_module,
                "_layout_cache_get",
                return_value=cache,
            ),
            mock.patch.object(
                demo_module,
                "_layout_editor_payload",
                return_value=base_payload,
            ),
            mock.patch.object(
                demo_module,
                "_layout_prompt_group_data",
                return_value=(
                    document,
                    decoded,
                    "stable-selection",
                ),
            ),
        ):
            updated, _, snapshot, _, active = (
                demo_module._commit_layout_group_transforms(
                    self.image_state,
                    state,
                    editor,
                )
            )

        authoritative = updated["prompt_group_transforms"]
        self.assertEqual(active["group_id"], "region_2")
        self.assertEqual(snapshot["active_group_id"], "region_2")
        self.assertEqual(authoritative["region_1"]["revision"], 1)
        self.assertEqual(authoritative["region_2"]["revision"], 1)
        self.assertAlmostEqual(
            authoritative["region_1"]["center_x"],
            incoming["region_1"]["center_x"],
        )
        self.assertAlmostEqual(
            authoritative["region_2"]["center_y"],
            incoming["region_2"]["center_y"],
        )
        self.assertEqual(
            _signature(authoritative["region_3"]),
            _signature(initial["region_3"]),
        )

    def test_numeric_update_and_reset_only_change_active_label_group(self):
        document = self._save_two_regions()
        state = self._selection_state(document)
        decoded = regions.decode_region_masks(
            regions.active_regions(document),
            self.source_mask.shape,
        )
        target_size = (self.image.width, self.image.height)
        base_transform = demo_module._layout_tx.make_layout_transform_v2(
            session_id=self.session_id,
            layout_id=self.layout_id,
            image_id=self.image_state["image_id"],
            target_size=target_size,
            source_mask=self.source_mask.astype(bool),
            center_x=24.0,
            center_y=18.0,
            pivot_xy=[24.0, 18.0],
            scale=1.0,
            rotation_deg=0.0,
            preview_alpha=0.35,
            revision=0,
            source_mask_pixel_sha256=self.source_hash,
            target_image_sha256=self.image_state[
                "target_image_sha256"
            ],
        )
        base_transform = (
            demo_module._layout_tx.transform_with_derived_fields(
                base_transform,
                target_size,
            )
        )
        initial = {}
        for record, mask in decoded:
            group_id = demo_module._layout_prompt_group_id(
                record["region_id"]
            )
            initial[group_id] = demo_module._layout_group_transform(
                state,
                base_transform,
                record,
                mask,
                target_size,
            )[0]
        state["prompt_group_transforms"] = copy.deepcopy(initial)
        state["prompt_active_group_id"] = "region_1"
        state["prompt_selection_signature"] = "stable-selection"
        state["prompt_transform_set_revision"] = 0
        editor = {
            "group_intent": {
                "selection_signature": "stable-selection",
                "transform_set_revision": 0,
                "active_group_id": "region_1",
                "transforms": [
                    {
                        "group_id": group_id,
                        "transform": copy.deepcopy(transform),
                    }
                    for group_id, transform in initial.items()
                ],
            }
        }
        cache = {"source_mask": self.source_mask.astype(bool)}
        base_payload = {
            "target_width": target_size[0],
            "target_height": target_size[1],
            "transform": base_transform,
        }

        with (
            mock.patch.object(
                demo_module,
                "_layout_cache_get",
                return_value=cache,
            ),
            mock.patch.object(
                demo_module,
                "_layout_editor_payload",
                return_value=base_payload,
            ),
            mock.patch.object(
                demo_module,
                "_layout_prompt_group_data",
                return_value=(
                    document,
                    decoded,
                    "stable-selection",
                ),
            ),
        ):
            first_editor = copy.deepcopy(editor)
            first_active = next(
                item["transform"]
                for item in first_editor["group_intent"]["transforms"]
                if item["group_id"] == "region_1"
            )
            first_active["center_x"] += 2.0
            first_active["revision"] += 1
            first_active["preview_alpha"] = 0.0
            first_drag, _, _, _, first_result = (
                demo_module._commit_layout_group_transforms(
                    self.image_state,
                    state,
                    first_editor,
                )
            )
            self.assertEqual(first_result["revision"], 1)
            self.assertEqual(first_result["preview_alpha"], 0.0)
            self.assertEqual(
                first_drag["prompt_group_transforms"]["region_2"][
                    "revision"
                ],
                initial["region_2"]["revision"],
            )
            self.assertEqual(
                _signature(
                    first_drag["prompt_group_transforms"]["region_2"]
                ),
                _signature(initial["region_2"]),
            )

            second_editor = {
                "group_intent": {
                    "selection_signature": "stable-selection",
                    "transform_set_revision": first_drag[
                        "prompt_transform_set_revision"
                    ],
                    "active_group_id": "region_1",
                    "transforms": [
                        {
                            "group_id": group_id,
                            "transform": copy.deepcopy(transform),
                        }
                        for group_id, transform in first_drag[
                            "prompt_group_transforms"
                        ].items()
                    ],
                }
            }
            second_active = next(
                item["transform"]
                for item in second_editor["group_intent"]["transforms"]
                if item["group_id"] == "region_1"
            )
            second_active["center_y"] += 3.0
            second_active["revision"] += 1
            second_active["preview_alpha"] = 0.0
            second_drag, _, _, _, second_result = (
                demo_module._commit_layout_group_transforms(
                    self.image_state,
                    first_drag,
                    second_editor,
                )
            )
            self.assertEqual(second_result["revision"], 2)
            self.assertEqual(second_result["preview_alpha"], 0.0)
            self.assertEqual(
                _signature(
                    second_drag["prompt_group_transforms"]["region_2"]
                ),
                _signature(initial["region_2"]),
            )

            numeric_editor = {
                "group_intent": {
                    "selection_signature": "stable-selection",
                    "transform_set_revision": second_drag[
                        "prompt_transform_set_revision"
                    ],
                    "active_group_id": "region_1",
                    "transforms": [
                        {
                            "group_id": group_id,
                            "transform": copy.deepcopy(transform),
                        }
                        for group_id, transform in second_drag[
                            "prompt_group_transforms"
                        ].items()
                    ],
                }
            }
            updated, _, _, _, active = (
                demo_module._commit_layout_group_transforms(
                    self.image_state,
                    second_drag,
                    numeric_editor,
                    numeric_override=(5.0, 6.0, 1.5, 20.0, 0.4),
                )
            )
            self.assertEqual(active["group_id"], "region_1")
            self.assertAlmostEqual(active["center_x"], 29.0)
            self.assertAlmostEqual(active["center_y"], 24.0)
            self.assertAlmostEqual(active["scale"], 1.5)
            self.assertAlmostEqual(active["rotation_deg"], 20.0)
            self.assertAlmostEqual(active["preview_alpha"], 0.4)

            untouched = updated["prompt_group_transforms"]["region_2"]
            self.assertEqual(
                _signature(untouched),
                _signature(initial["region_2"]),
            )

            reset_editor = {
                "group_intent": {
                    "selection_signature": "stable-selection",
                    "transform_set_revision": updated[
                        "prompt_transform_set_revision"
                    ],
                    "active_group_id": "region_1",
                    "transforms": [
                        {
                            "group_id": group_id,
                            "transform": copy.deepcopy(transform),
                        }
                        for group_id, transform in updated[
                            "prompt_group_transforms"
                        ].items()
                    ],
                }
            }
            reset, _, _, _, reset_active = (
                demo_module._commit_layout_group_transforms(
                    self.image_state,
                    updated,
                    reset_editor,
                    reset_active=True,
                )
            )

        self.assertAlmostEqual(
            reset_active["center_x"],
            initial["region_1"]["center_x"],
        )
        self.assertAlmostEqual(
            reset_active["center_y"],
            initial["region_1"]["center_y"],
        )
        for key in (
            "center_x",
            "center_y",
            "scale",
            "rotation_deg",
            "preview_alpha",
        ):
            self.assertAlmostEqual(
                reset["prompt_group_transforms"]["region_2"][key],
                untouched[key],
            )

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
            label="two-components",
            region_mask=mask,
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
                [demo_module._LAYOUT_PROMPT_SCOPE_FULL],
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

    def test_duplicate_historical_labels_get_distinct_choice_titles(self):
        shape = (4, 9)
        masks = []
        for start in (0, 3, 6):
            mask = np.zeros(shape, dtype=bool)
            mask[1:3, start : start + 2] = True
            masks.append(mask)
        document = {
            "regions": [
                regions.region_record_from_mask(
                    1,
                    "metal",
                    "",
                    masks[0],
                ),
                regions.region_record_from_mask(
                    2,
                    "metal",
                    "",
                    masks[1],
                ),
                regions.region_record_from_mask(
                    3,
                    None,
                    "",
                    masks[2],
                    label="via",
                ),
            ]
        }
        metal_1 = demo_module._layout_prompt_selection_token(
            demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            1,
        )
        metal_2 = demo_module._layout_prompt_selection_token(
            demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            2,
        )
        via = demo_module._layout_prompt_selection_token(
            demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            3,
        )

        update = demo_module._layout_prompt_choice_update(
            document,
            [metal_2, via],
        )

        self.assertEqual(
            update["choices"],
            [
                ("全部版图 mask", demo_module._LAYOUT_PROMPT_SCOPE_FULL),
                ("metal (R1)", metal_1),
                ("metal (R2)", metal_2),
                ("via", via),
            ],
        )
        self.assertEqual(update["value"], [metal_2, via])

    def test_selector_uses_checkbox_values_and_restores_on_failure(self):
        document = self._save_two_regions()
        full_state = copy.deepcopy(self.layout_state)
        full_state["prompt_mask_scope"] = (
            demo_module._LAYOUT_PROMPT_SCOPE_FULL
        )
        missing_token = demo_module._layout_prompt_selection_token(
            demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
            999,
        )
        failed = demo_module._select_layout_prompt_mask(
            self.image_state,
            full_state,
            [missing_token],
        )
        self.assertEqual(len(failed), 10)
        self.assertEqual(
            failed[0]["prompt_mask_scope"],
            demo_module._LAYOUT_PROMPT_SCOPE_FULL,
        )
        self.assertEqual(
            failed[2]["value"],
            [demo_module._LAYOUT_PROMPT_SCOPE_FULL],
        )
        self.assertIn("已保留原选择", failed[3])

        selected_tokens = [
            demo_module._layout_prompt_selection_token(
                demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS,
                record["region_id"],
            )
            for record in regions.active_regions(document)
        ]
        active_transform = {
            "center_x": 24.0,
            "center_y": 18.0,
            "scale": 1.0,
            "rotation_deg": 0.0,
            "preview_alpha": 0.35,
        }

        def commit(_image_state, state, _editor):
            return (
                dict(state),
                [],
                {},
                np.zeros_like(self.source_mask, dtype=bool),
                active_transform,
            )

        with (
            mock.patch.object(
                demo_module,
                "_layout_editor_payload",
                return_value={"enabled": True},
            ),
            mock.patch.object(
                demo_module,
                "_commit_layout_group_transforms",
                side_effect=commit,
            ),
        ):
            selected = demo_module._select_layout_prompt_mask(
                self.image_state,
                full_state,
                selected_tokens,
            )

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            selected[0]["prompt_region_ids"],
            [record["region_id"] for record in document["regions"]],
        )
        self.assertEqual(selected[2]["value"], selected_tokens)
        self.assertIn("2 个 Label", selected[3])

    def test_workspace_overlay_uses_ascii_region_id_for_unicode_label(self):
        transformed = np.zeros(
            (self.image.height, self.image.width),
            dtype=bool,
        )
        transformed[8:20, 10:30] = True
        cache = {
            "prompt_group_transformed_masks": {
                "region_7": transformed,
            },
            "prompt_group_snapshot": {
                "active_group_id": "region_7",
                "transforms": {
                    "region_7": {
                        "region_id": 7,
                        "label": "金属互连",
                        "preview_alpha": 0.35,
                    }
                },
            },
        }
        layout_state = {
            "enabled": True,
            "layout_id": self.layout_id,
            "prompt_mask_scope": (
                demo_module._LAYOUT_PROMPT_SCOPE_REGION_LABELS
            ),
        }
        rendered_text = []

        def capture_put_text(image, text, *_args, **_kwargs):
            rendered_text.append(text)
            return image

        with (
            mock.patch.object(
                demo_module,
                "_workspace",
                return_value={"image": self.image},
            ),
            mock.patch.object(
                demo_module,
                "_layout_cache_get",
                return_value=cache,
            ),
            mock.patch.object(
                cv2,
                "putText",
                side_effect=capture_put_text,
            ),
        ):
            demo_module._overlay(
                self.image_state,
                demo_module._new_pcs_state(),
                demo_module._new_pvs_state(),
                demo_module.MODE_LAYOUT,
                show_layout_overlay=True,
                layout_state=layout_state,
            )

        self.assertIn("L7", rendered_text)
        self.assertNotIn("金属互连", rendered_text)

    def test_feedback_reconstructs_frozen_region_after_soft_delete(self):


        document = self._save_two_regions()
        record = regions.active_regions(document)[0]
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
            "label": regions.region_label(record),
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
        record = regions.active_regions(document)[0]
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
            "label": regions.region_label(record),
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
