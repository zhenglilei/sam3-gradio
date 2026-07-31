from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from sam3_demo import app as demo


ROOT = Path(__file__).resolve().parents[1]


class LayoutMaskAgentUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = demo.create_demo()
        cls.config = cls.demo.config

    def test_assistant_is_only_constructed_in_layout_mask_tab(self):
        labels = {
            (component.get("props") or {}).get("label")
            for component in self.config["components"]
        }
        self.assertIn("AI Mask \u4fee\u590d\u52a9\u624b", labels)
        image_tab_source = (ROOT / "sam3_demo" / "ui" / "image_tab.py").read_text(encoding="utf-8")
        self.assertNotIn("layout_agent_", image_tab_source)

    def test_paid_callback_has_dedicated_serial_concurrency(self):
        dependency = next(
            item
            for item in self.config["dependencies"]
            if item.get("api_name") == "_layout_mask_agent_run"
        )
        self.assertTrue(dependency.get("backend_fn"))
        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn('concurrency_id="layout-mask-vlm"', binding_source)
        self.assertIn("concurrency_limit=1", binding_source)

    def test_agent_callbacks_do_not_accept_layout_state(self):
        run_parameters = inspect.signature(
            demo._layout_mask_agent_run_callback
        ).parameters
        apply_parameters = inspect.signature(
            demo._layout_mask_agent_apply_callback
        ).parameters
        self.assertNotIn("layout_state", run_parameters)
        self.assertEqual(list(apply_parameters), ["agent_state"])
        source = (ROOT / "sam3_demo" / "layout" / "agent_callbacks.py").read_text(encoding="utf-8")
        self.assertNotIn("_save_layout_mask_files", source)
        self.assertNotIn("runtime_layout_dir", source)

    def test_existing_generate_contract_is_unchanged(self):
        dependency = next(
            item
            for item in self.config["dependencies"]
            if item.get("api_name") == "_run_layout_mask_page"
        )
        self.assertEqual(len(dependency["inputs"]), 10)
        self.assertEqual(len(dependency["outputs"]), 8)

    def test_reset_clears_consent_and_draft_without_vlm(self):
        result = demo._layout_mask_agent_reset_callback(
            {"session_id": "ui-test"},
            None,
            "ACT",
        )
        self.assertEqual(len(result), 9)
        self.assertFalse(result[1])
        self.assertEqual(result[2], [])
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        self.assertEqual(result[0]["profile_mode"], "ACT")
        self.assertIsNone(result[0]["consent_image_sha256"])

    def test_saved_baseline_only_runs_after_successful_generation(self):
        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn("run_layout_mask_event.success(", binding_source)

    def test_consent_callback_binds_authorization_to_image(self):
        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn("layout_agent_consent.change(", binding_source)
        self.assertIn("fn=_layout_mask_agent_consent_callback", binding_source)


if __name__ == "__main__":
    unittest.main()
