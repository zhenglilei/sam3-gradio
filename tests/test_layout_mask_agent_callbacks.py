import copy
import unittest

import numpy as np
from PIL import Image

from sam3_demo.layout.agent_callbacks import (
    apply_layout_mask_agent_params,
    run_layout_mask_agent_turn,
    undo_layout_mask_agent_draft,
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
                "schema_version": 1,
                "intent": "revise",
                "profile": "ACT",
                "confidence": 0.92,
                "selected_candidate_id": "C2",
                "observations": ["small hole"],
                "assistant_message": "Fill a little more.",
                "manual_review": False,
                "warnings": [],
            },
            {"total_tokens": 55},
            0.002,
            "skill-test",
        )

    def run_turn(self, state=None, consent=True, controls=None, vlm=None):
        return run_layout_mask_agent_turn(
            state=state or _new_layout_mask_agent_state("session"),
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
        self.assertEqual(state["current_draft_params"]["close_kernel"], 17)
        self.assertEqual(state["undo_stack"][-1]["close_kernel"], 15)
        self.assertEqual(state["turn_count"], 1)
        self.assertEqual(state["last_usage"]["total_tokens"], 55)
        self.assertNotIn("mask", json_keys(state))
        self.assertIsNotNone(result["draft_preview"])
        self.assertIsNotNone(result["candidate_preview"])

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
                    "schema_version": 1,
                    "intent": "revise",
                    "profile": "ACT",
                    "confidence": 0.8,
                    "selected_candidate_id": "C1",
                    "observations": [],
                    "assistant_message": "Keep manual controls.",
                    "manual_review": False,
                    "warnings": [],
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
