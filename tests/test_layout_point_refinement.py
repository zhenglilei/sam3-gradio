import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3_gradio_demo as demo_module


def _signature(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((str(key), _signature(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    return value


class LayoutPointRefinementTest(unittest.TestCase):
    image_state = {"image_id": "image", "width": 8, "height": 6}

    @staticmethod
    def _prediction(offset=0.0):
        masks = np.zeros((3, 6, 8), dtype=bool)
        masks[0, :2, :2] = True
        masks[1, 1:5, 2:7] = True
        masks[2, 3:6, :4] = True
        return {
            "masks": masks,
            "scores": np.asarray([0.2, 0.9, 0.5], dtype=np.float32),
            "lowres_logits": np.stack(
                [
                    np.full((4, 4), offset + 10, dtype=np.float32),
                    np.full((4, 4), offset + 20, dtype=np.float32),
                    np.full((4, 4), offset + 30, dtype=np.float32),
                ]
            ),
        }

    @staticmethod
    def _state(source="manual_pvs_layout_mask", status="accepted", logits=None, history=None):
        logits = np.ones((4, 4), dtype=np.float32) if logits is None else logits
        history = [{"op": "create_from_layout_mask"}] if history is None else history
        active_mask = np.zeros((6, 8), dtype=bool)
        active_mask[1:4, 1:5] = True
        other_mask = np.zeros((6, 8), dtype=bool)
        other_mask[:2, 5:8] = True
        active = demo_module._make_inst(
            1, source, active_mask, [1, 1, 5, 4], 0.4,
            pvs_logits=logits, history=copy.deepcopy(history),
        )
        active.update(status=status, sentinel="keep")
        other = demo_module._make_inst(
            2, "manual_pvs_bbox_batch", other_mask, [5, 0, 8, 2], 0.7,
            pvs_logits=np.full((4, 4), 7, dtype=np.float32),
            history=[{"op": "create_from_pending_bbox"}],
        )
        return {
            "instances": {1: active, 2: other},
            "active_instance_id": 1,
            "next_instance_id": 3,
            "pending_boxes": [],
            "pending_bbox_records": [],
            "next_pending_bbox_id": 1,
        }

    def test_helper_uses_previous_logits_across_rounds_and_only_updates_active(self):
        state = self._state()
        other_before = _signature(state["instances"][2])
        first, second = self._prediction(), self._prediction(100)
        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={"fresh": True}),
            mock.patch.object(demo_module, "_predict_inst", side_effect=[first, second]) as predict,
        ):
            result = demo_module._refine_active_pvs_with_point(
                self.image_state, state, [2, 3], 1
            )
            first_logits = state["instances"][1]["pvs_lowres_logits"].copy()
            demo_module._refine_active_pvs_with_point(
                self.image_state, state, [4, 2], 0
            )

        self.assertEqual(result, (1, "positive_point"))
        np.testing.assert_array_equal(
            predict.call_args_list[0].kwargs["mask_input_lowres_logits"],
            np.ones((4, 4), dtype=np.float32),
        )
        self.assertEqual(predict.call_args_list[0].kwargs["point_labels"], [1])
        np.testing.assert_array_equal(
            predict.call_args_list[1].kwargs["mask_input_lowres_logits"],
            first_logits,
        )
        self.assertEqual(predict.call_args_list[1].kwargs["point_labels"], [0])
        np.testing.assert_array_equal(first_logits, first["lowres_logits"][1])
        np.testing.assert_array_equal(
            state["instances"][1]["pvs_lowres_logits"], second["lowres_logits"][1]
        )
        np.testing.assert_array_equal(
            state["instances"][1]["mask_fullres_bool"], second["masks"][1]
        )
        self.assertEqual(state["instances"][1]["source"], "manual_pvs_layout_mask")
        self.assertEqual(state["instances"][1]["status"], "accepted")
        self.assertEqual(state["instances"][1]["sentinel"], "keep")
        self.assertEqual(
            [event["op"] for event in state["instances"][1]["prompt_history"][-2:]],
            ["positive_point_refine", "negative_point_refine"],
        )
        self.assertEqual(_signature(state["instances"][2]), other_before)
        self.assertEqual(state["next_instance_id"], 3)

    def test_helper_validation_and_prediction_failures_are_atomic(self):
        invalid_cases = [
            lambda state: state.update(active_instance_id=None),
            lambda state: state.update(active_instance_id=99),
            lambda state: state["instances"][1].update(status="deleted"),
            lambda state: state["instances"][1].update(pvs_lowres_logits=None),
            lambda state: state["instances"][1].update(
                pvs_lowres_logits=np.ones((4, 4), dtype=np.float64)
            ),
            lambda state: state["instances"][1].update(
                pvs_lowres_logits=np.ones((3, 4), dtype=np.float32)
            ),
            lambda state: state["instances"][1].update(
                pvs_lowres_logits=np.full((4, 4), np.nan, dtype=np.float32)
            ),
        ]
        for mutate in invalid_cases:
            state = self._state()
            mutate(state)
            before = _signature(state)
            with (
                mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
                mock.patch.object(demo_module, "_predict_inst") as predict,
            ):
                with self.assertRaises(ValueError):
                    demo_module._refine_active_pvs_with_point(
                        self.image_state, state, [2, 3], 1
                    )
            predict.assert_not_called()
            self.assertEqual(_signature(state), before)

        bad_predictions = [
            RuntimeError("predict failed"),
            {
                "masks": np.zeros((1, 6, 8), dtype=bool),
                "scores": np.asarray([np.nan], dtype=np.float32),
                "lowres_logits": np.zeros((1, 4, 4), dtype=np.float32),
            },
            {
                "masks": np.zeros((1, 6, 8), dtype=bool),
                "scores": np.asarray([0.2, 0.9], dtype=np.float32),
                "lowres_logits": np.zeros((2, 4, 4), dtype=np.float32),
            },
            {
                "masks": np.zeros((1, 6, 8), dtype=bool),
                "scores": np.asarray([0.9], dtype=np.float32),
                "lowres_logits": np.zeros((1, 3, 4), dtype=np.float32),
            },
        ]
        for bad in bad_predictions:
            state = self._state()
            before = _signature(state)
            with (
                mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
                mock.patch.object(demo_module, "_fresh_state", return_value={}),
                mock.patch.object(
                    demo_module,
                    "_predict_inst",
                    side_effect=bad if isinstance(bad, Exception) else None,
                    return_value=None if isinstance(bad, Exception) else bad,
                ),
            ):
                with self.assertRaises((RuntimeError, ValueError)):
                    demo_module._refine_active_pvs_with_point(
                        self.image_state, state, [2, 3], 1
                    )
            self.assertEqual(_signature(state), before)

    def test_helper_rejects_state_change_during_prediction(self):
        state = self._state()
        active_before = _signature(state["instances"][1])

        def switch_active(*_args, **_kwargs):
            state["active_instance_id"] = 2
            return self._prediction()

        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(demo_module, "_predict_inst", side_effect=switch_active),
        ):
            with self.assertRaisesRegex(ValueError, "预测期间发生变化"):
                demo_module._refine_active_pvs_with_point(
                    self.image_state, state, [2, 3], 1
                )
        self.assertEqual(state["active_instance_id"], 2)
        self.assertEqual(_signature(state["instances"][1]), active_before)

        state = self._state()
        active_before = _signature(state["instances"][1])
        original_mask_box = demo_module._mask_box

        def switch_while_staging(mask):
            state["active_instance_id"] = 2
            return original_mask_box(mask)

        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(demo_module, "_predict_inst", return_value=self._prediction()),
            mock.patch.object(demo_module, "_mask_box", side_effect=switch_while_staging),
        ):
            with self.assertRaisesRegex(ValueError, "预测期间发生变化"):
                demo_module._refine_active_pvs_with_point(self.image_state, state, [2, 3], 1)
        self.assertEqual(_signature(state["instances"][1]), active_before)

    def test_layout_wrapper_consumes_success_and_retains_failures(self):
        state = self._state()
        prompt = demo_module._new_prompt_state()
        prompt["last_point"] = [2.0, 3.0]
        payload = json.dumps({"point_xy_px": [2.0, 3.0]})
        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module, "_predict_inst", return_value=self._prediction()
            ) as predict,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8) as view,
        ):
            result = demo_module._layout_point_refine(
                self.image_state, demo_module._new_pcs_state(), state,
                demo_module.MODE_LAYOUT, payload, "negative", prompt, progress=None,
            )
        self.assertIsNone(result[0]["last_point"])
        self.assertEqual(result[1], "")
        self.assertEqual(predict.call_args.kwargs["point_labels"], [0])
        self.assertIn("已应用 negative_point", view.call_args.args[4])

        for kind in ("positive", "negative"):
            failed_state = self._state()
            failed_state["active_instance_id"] = None
            failed_prompt = demo_module._new_prompt_state()
            failed_prompt["last_point"] = [2.0, 3.0]
            before = _signature(failed_state)
            with (
                mock.patch.object(demo_module, "_predict_inst") as failed_predict,
                mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
            ):
                failed = demo_module._layout_point_refine(
                    self.image_state, demo_module._new_pcs_state(), failed_state,
                    demo_module.MODE_LAYOUT, payload, kind, failed_prompt, progress=None,
                )
            failed_predict.assert_not_called()
            self.assertEqual(failed[0]["last_point"], [2.0, 3.0])
            self.assertEqual(failed[1], payload)
            self.assertEqual(_signature(failed_state), before)

        wrong_source = self._state(source="manual_pvs_point")
        wrong_prompt = demo_module._new_prompt_state()
        wrong_prompt["last_point"] = [2.0, 3.0]
        before = _signature(wrong_source)
        with (
            mock.patch.object(demo_module, "_predict_inst") as failed_predict,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            failed = demo_module._layout_point_refine(
                self.image_state, demo_module._new_pcs_state(), wrong_source,
                demo_module.MODE_LAYOUT, payload, "positive", wrong_prompt, progress=None,
            )
        failed_predict.assert_not_called()
        self.assertEqual(failed[1], payload)
        self.assertEqual(_signature(wrong_source), before)

        failure_cases = [
            ("wrong mode", demo_module.MODE_PVS, lambda state: None),
            (
                "deleted",
                demo_module.MODE_LAYOUT,
                lambda state: state["instances"][1].update(status="deleted"),
            ),
            (
                "missing creation history",
                demo_module.MODE_LAYOUT,
                lambda state: state["instances"][1].update(
                    prompt_history=[{"op": "positive_point_refine"}]
                ),
            ),
            (
                "missing logits",
                demo_module.MODE_LAYOUT,
                lambda state: state["instances"][1].update(pvs_lowres_logits=None),
            ),
        ]
        for name, mode, mutate in failure_cases:
            with self.subTest(name=name):
                failed_state = self._state()
                mutate(failed_state)
                failed_prompt = demo_module._new_prompt_state()
                failed_prompt["last_point"] = [2.0, 3.0]
                before = _signature(failed_state)
                with (
                    mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
                    mock.patch.object(demo_module, "_predict_inst") as failed_predict,
                    mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
                ):
                    failed = demo_module._layout_point_refine(
                        self.image_state, demo_module._new_pcs_state(), failed_state,
                        mode, payload, "positive", failed_prompt, progress=None,
                    )
                failed_predict.assert_not_called()
                self.assertEqual(failed[0]["last_point"], [2.0, 3.0])
                self.assertEqual(failed[1], payload)
                self.assertEqual(_signature(failed_state), before)

        predict_failure_state = self._state()
        predict_failure_prompt = demo_module._new_prompt_state()
        predict_failure_prompt["last_point"] = [2.0, 3.0]
        before = _signature(predict_failure_state)
        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module, "_predict_inst", side_effect=RuntimeError("predict failed")
            ),
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            failed = demo_module._layout_point_refine(
                self.image_state, demo_module._new_pcs_state(), predict_failure_state,
                demo_module.MODE_LAYOUT, payload, "positive",
                predict_failure_prompt, progress=None,
            )
        self.assertEqual(failed[0]["last_point"], [2.0, 3.0])
        self.assertEqual(failed[1], payload)
        self.assertEqual(_signature(predict_failure_state), before)

    def test_layout_postcommit_progress_failure_does_not_retain_point(self):
        state = self._state()
        prompt = demo_module._new_prompt_state()
        prompt["last_point"] = [2.0, 3.0]
        payload = json.dumps({"point_xy_px": [2.0, 3.0]})

        def progress_side_effect(_progress, value, _desc, **_kwargs):
            if value == 0.78:
                raise RuntimeError("progress delivery failed")

        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module, "_predict_inst", return_value=self._prediction()
            ),
            mock.patch.object(
                demo_module, "_pvs_progress", side_effect=progress_side_effect
            ),
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8) as view,
        ):
            result = demo_module._layout_point_refine(
                self.image_state, demo_module._new_pcs_state(), state,
                demo_module.MODE_LAYOUT, payload, "positive", prompt,
                progress=object(),
            )
        self.assertIsNone(result[0]["last_point"])
        self.assertEqual(result[1], "")
        self.assertEqual(
            state["instances"][1]["prompt_history"][-1]["op"],
            "positive_point_refine",
        )
        self.assertNotIn("修缮失败", view.call_args.args[4])
        self.assertIn("结果已保存", view.call_args.args[4])

    def test_layout_view_failure_still_returns_consumed_point_state(self):
        state = self._state()
        prompt = demo_module._new_prompt_state()
        prompt["last_point"] = [2.0, 3.0]
        payload = json.dumps({"point_xy_px": [2.0, 3.0]})
        with (
            mock.patch.object(demo_module, "_prompt_mask_size", return_value=(4, 4)),
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(demo_module, "_predict_inst", return_value=self._prediction()),
            mock.patch.object(demo_module, "_view", side_effect=RuntimeError("render failed")),
        ):
            result = demo_module._layout_point_refine(
                self.image_state, demo_module._new_pcs_state(), state,
                demo_module.MODE_LAYOUT, payload, "positive", prompt, progress=None,
            )
        self.assertEqual(len(result), 11)
        self.assertIsNone(result[0]["last_point"])
        self.assertEqual(result[1], "")
        self.assertIn("界面刷新失败", result[9])

    def test_layout_click_records_one_point_without_prediction(self):
        prompt = demo_module._new_prompt_state()
        with (
            mock.patch.object(demo_module, "_predict_inst") as predict,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            first = demo_module._workspace_select(
                {"width": 8, "height": 6}, demo_module._new_pcs_state(),
                demo_module._new_pvs_state(), demo_module.MODE_LAYOUT, "layout",
                "Positive exemplar", prompt, SimpleNamespace(index=[1, 2]),
            )
            second = demo_module._workspace_select(
                {"width": 8, "height": 6}, demo_module._new_pcs_state(),
                demo_module._new_pvs_state(), demo_module.MODE_LAYOUT, "layout",
                "Positive exemplar", first[0], SimpleNamespace(index=[5, 4]),
            )
        predict.assert_not_called()
        self.assertEqual(second[0]["last_point"], [5.0, 4.0])
        self.assertEqual(json.loads(second[2])["point_xy_px"], [5.0, 4.0])

    def test_pvs_manual_active_branch_uses_common_helper(self):
        payload = json.dumps({"point_xy_px": [2.0, 3.0]})
        state = self._state()
        with (
            mock.patch.object(
                demo_module,
                "_refine_active_pvs_with_point",
                return_value=(1, "positive_point"),
            ) as refine,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            demo_module._pvs_point_prompt(
                self.image_state, demo_module._new_pcs_state(), state,
                demo_module.MODE_PVS, payload, "positive", progress=None,
            )
        refine.assert_called_once_with(
            self.image_state, state, [2.0, 3.0], 1, progress=None
        )

    def test_pvs_manual_creation_and_mode_cleanup_regressions(self):
        payload = json.dumps({"point_xy_px": [2.0, 3.0]})
        pvs_state = demo_module._new_pvs_state()
        with (
            mock.patch.object(demo_module, "_fresh_state", return_value={}),
            mock.patch.object(
                demo_module, "_predict_inst", return_value=self._prediction()
            ) as predict,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            demo_module._pvs_point_prompt(
                self.image_state, demo_module._new_pcs_state(), pvs_state,
                demo_module.MODE_PVS, payload, "positive", progress=None,
            )
        self.assertEqual(predict.call_args.kwargs["point_labels"], [1])
        self.assertEqual(pvs_state["instances"][1]["source"], "manual_pvs_point")

        negative_state = demo_module._new_pvs_state()
        with (
            mock.patch.object(demo_module, "_predict_inst") as negative_predict,
            mock.patch.object(demo_module, "_view", return_value=(None,) * 8),
        ):
            demo_module._pvs_point_prompt(
                self.image_state, demo_module._new_pcs_state(), negative_state,
                demo_module.MODE_PVS, payload, "negative", progress=None,
            )
        negative_predict.assert_not_called()
        self.assertEqual(negative_state["instances"], {})

        with mock.patch.object(demo_module, "_view", return_value=(None,) * 8):
            layout_result = demo_module._switch_mode(
                demo_module.MODE_LAYOUT, self.image_state,
                demo_module._new_pcs_state(), demo_module._new_pvs_state(),
            )
            pvs_result = demo_module._switch_mode(
                demo_module.MODE_PVS, self.image_state,
                demo_module._new_pcs_state(), demo_module._new_pvs_state(),
            )
        self.assertEqual(len(layout_result), 27)
        self.assertIsNone(layout_result[0]["last_point"])
        self.assertEqual(layout_result[2], "")
        self.assertTrue(layout_result[18]["visible"])
        self.assertFalse(pvs_result[18]["visible"])

        selected_state = self._state()
        with mock.patch.object(demo_module, "_view", return_value=(None,) * 8):
            demo_module._set_active_pvs(
                self.image_state, demo_module._new_pcs_state(), selected_state,
                demo_module.MODE_LAYOUT, None,
            )
        self.assertIsNone(selected_state["active_instance_id"])
        self.assertIsNone(demo_module._pvs_choice_update(selected_state)["value"])
        self.assertEqual(demo_module._clear_pending_point_payload(), "")


if __name__ == "__main__":
    unittest.main()
