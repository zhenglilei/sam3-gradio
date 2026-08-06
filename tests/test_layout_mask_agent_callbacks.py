import copy
import unittest

import numpy as np
from PIL import Image

from sam3_demo.layout.agent_callbacks import (
    classify_layout_agent_message,
    apply_layout_mask_agent_params,
    run_layout_mask_agent_turn,
    undo_layout_mask_agent_draft,
    set_layout_mask_agent_consent,
    validate_layout_mask_agent_context,
)
from sam3_demo.layout.preprocess_registry import params_from_controls
from sam3_demo.state import _new_layout_mask_agent_state


class LayoutMaskAgentCallbackTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 60), "white")
        self.controls = params_from_controls(12, False, 0, 15, 0, "all", 0)
        self.calls = []

    @staticmethod
    def compute_draft(image, threshold, invert, open_kernel, close_kernel, min_area, region_mode, morph_pixels):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[10:50, 10:70] = True
        mask[25:28, 35:38] = False
        if close_kernel >= 17:
            mask[25:28, 35:38] = True
        params = params_from_controls(
            threshold,
            invert,
            open_kernel,
            close_kernel,
            min_area,
            region_mode,
            morph_pixels,
        )
        return image, mask, [], params

    def vlm(self, **kwargs):
        self.calls.append(kwargs)
        return (
            {
                "schema_version": 4,
                "profile": "ACT",
                "confidence": 0.92,
                "period_total": 4,
                "period_rows": 2,
                "parameters": dict(self.controls, close_kernel=17),
                "explanation": "建议将 close 增加到 17 并保持线宽。",
                "manual_review": False,
            },
            {"total_tokens": 55},
            0.002,
            "skill-test",
        )

    def run_turn(self, state=None, consent=True, controls=None, vlm=None):
        state = set_layout_mask_agent_consent(
            state or _new_layout_mask_agent_state("session"),
            session_id="session",
            image=self.image,
            consent=consent,
            profile_mode="ACT",
        )
        return run_layout_mask_agent_turn(
            state=state,
            session_id="session",
            image=self.image,
            consent=consent,
            profile_mode="ACT",
            user_message="\u518d\u586b\u4e00\u70b9",
            controls=controls or self.controls,
            compute_draft=self.compute_draft,
            vlm_options={},
            vlm_call=vlm or self.vlm,
        )

    def test_turn_uses_one_request_and_keeps_only_lightweight_state(self):
        result = self.run_turn()
        self.assertEqual(len(self.calls), 1)
        state = result["state"]
        self.assertGreaterEqual(state["last_latency_seconds"], 0.0)
        self.assertEqual(state["current_draft_params"]["close_kernel"], 17)
        self.assertEqual(state["undo_stack"][-1]["close_kernel"], 15)
        self.assertEqual(state["turn_count"], 1)
        self.assertEqual(state["last_usage"]["total_tokens"], 55)
        self.assertNotIn("mask", json_keys(state))
        self.assertIsNotNone(result["draft_preview"])
        self.assertNotIn("candidate_preview", result)

    def test_consent_and_vlm_failure_do_not_mutate_input_state(self):
        state = _new_layout_mask_agent_state("session")
        original = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, "outbound"):
            self.run_turn(state=state, consent=False)
        self.assertEqual(state, original)
        self.assertEqual(len(self.calls), 0)

        def fail(**kwargs):
            raise RuntimeError("network down")

        with self.assertRaisesRegex(RuntimeError, "network down"):
            self.run_turn(state=state, vlm=fail)
        self.assertEqual(state, original)

    def test_undo_and_apply_are_local(self):
        result = self.run_turn()
        call_count = len(self.calls)
        undone = undo_layout_mask_agent_draft(
            result["state"],
            self.image,
            self.compute_draft,
        )
        self.assertEqual(undone["state"]["current_draft_params"]["close_kernel"], 15)
        self.assertEqual(len(self.calls), call_count)
        applied_state, applied = apply_layout_mask_agent_params(result["state"])
        self.assertEqual(applied["close_kernel"], 17)
        self.assertEqual(
            applied_state["applied_revision"],
            result["state"]["conversation_revision"],
        )
        self.assertEqual(len(self.calls), call_count)

    def test_manual_controls_become_new_baseline(self):
        first = self.run_turn()

        def select_current(**kwargs):
            self.calls.append(kwargs)
            return (
                {
                    "schema_version": 4,
                    "profile": "ACT",
                    "confidence": 0.8,
                    "period_total": 4,
                    "period_rows": 2,
                    "parameters": dict(kwargs["current_parameters"]),
                    "explanation": "保持当前手动参数。",
                    "manual_review": False,
                },
                None,
                None,
                "skill-test",
            )

        manual = dict(self.controls, close_kernel=13)
        result = self.run_turn(
            state=first["state"],
            controls=manual,
            vlm=select_current,
        )
        self.assertEqual(result["state"]["baseline_params"]["close_kernel"], 13)
        self.assertEqual(result["state"]["current_draft_params"]["close_kernel"], 13)
        self.assertTrue(
            any(row.get("kind") == "manual_override" for row in result["state"]["history"])
        )

    def test_local_chat_commands_are_keyword_routed_without_vlm(self):
        self.assertEqual(classify_layout_agent_message("撤回"), "undo")
        self.assertEqual(classify_layout_agent_message("恢复 上一版"), "undo")
        self.assertEqual(classify_layout_agent_message("重置"), "reset")
        self.assertEqual(classify_layout_agent_message("应用推荐参数"), "apply")
        self.assertEqual(
            classify_layout_agent_message("不要加粗但修补断口"),
            "vlm",
        )


    def test_second_turn_continues_from_unapplied_draft(self):
        first = self.run_turn()

        def select_fill_candidate(**kwargs):
            self.calls.append(kwargs)
            recommended = dict(kwargs["current_parameters"])
            recommended["close_kernel"] += 2
            return (
                {
                    "schema_version": 4,
                    "profile": "ACT",
                    "confidence": 0.9,
                    "period_total": 4,
                    "period_rows": 2,
                    "parameters": recommended,
                    "explanation": "在当前 close 基础上增加 2。",
                    "manual_review": False,
                },
                None,
                None,
                "skill-test",
            )

        second = self.run_turn(state=first["state"], vlm=select_fill_candidate)
        self.assertEqual(second["state"]["current_draft_params"]["close_kernel"], 19)
        self.assertEqual(second["state"]["undo_stack"][-1]["close_kernel"], 17)

    def test_consent_is_bound_to_one_image_identity(self):
        state = set_layout_mask_agent_consent(
            _new_layout_mask_agent_state("session"),
            session_id="session",
            image=self.image,
            consent=True,
            profile_mode="ACT",
        )
        other_image = Image.new("RGB", (80, 60), "black")
        with self.assertRaisesRegex(ValueError, "outbound"):
            run_layout_mask_agent_turn(
                state=state,
                session_id="session",
                image=other_image,
                consent=True,
                profile_mode="ACT",
                user_message="analyze",
                controls=self.controls,
                compute_draft=self.compute_draft,
                vlm_options={},
                vlm_call=self.vlm,
            )

    def test_local_context_rejects_other_session_or_image(self):
        state = set_layout_mask_agent_consent(
            _new_layout_mask_agent_state("session"),
            session_id="session",
            image=self.image,
            consent=True,
            profile_mode="ACT",
        )
        validated = validate_layout_mask_agent_context(
            state,
            session_id="session",
            image=self.image,
        )
        self.assertEqual(validated.size, self.image.size)
        with self.assertRaisesRegex(ValueError, "different session"):
            validate_layout_mask_agent_context(
                state,
                session_id="other",
                image=self.image,
            )
        with self.assertRaisesRegex(ValueError, "current layout screenshot"):
            validate_layout_mask_agent_context(
                state,
                session_id="session",
                image=Image.new("RGB", self.image.size, "black"),
            )

def json_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(json_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(json_keys(item))
        return keys
    return set()


if __name__ == "__main__":
    unittest.main()
