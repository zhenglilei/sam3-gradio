from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_demo import app as demo
from sam3_demo.layout.agent_callbacks import set_layout_mask_agent_consent
from sam3_demo.layout.preprocess_registry import params_from_controls
from sam3_demo.state import _new_layout_mask_agent_state


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
        self.assertNotIn("允许发送当前图像缩略图", labels)
        image_tab_source = (ROOT / "sam3_demo" / "ui" / "image_tab.py").read_text(encoding="utf-8")
        self.assertNotIn("layout_agent_", image_tab_source)
        sidebar = next(
            component
            for component in self.config["components"]
            if component.get("type") == "sidebar"
        )
        self.assertEqual(sidebar["props"]["label"], "AI Mask 修复助手")
        self.assertEqual(sidebar["props"]["position"], "left")
        layout_tab_source = (ROOT / "sam3_demo" / "ui" / "layout_mask_tab.py").read_text(encoding="utf-8")
        for removed in ("layout_agent_auto_btn", "layout_agent_undo_btn", "layout_agent_reset_btn"):
            self.assertNotIn(removed, layout_tab_source)
        apply_button = next(
            component
            for component in self.config["components"]
            if component.get("type") == "button"
            and (component.get("props") or {}).get("value") == "应用推荐参数"
        )
        self.assertEqual(apply_button["props"]["variant"], "secondary")
        chatbot = next(
            component
            for component in self.config["components"]
            if component.get("type") == "chatbot"
            and (component.get("props") or {}).get("label") == "对话"
        )
        self.assertEqual(chatbot["props"]["buttons"], [])
        self.assertNotIn("feedback_options", chatbot["props"])
        self.assertNotIn("候选对比图", layout_tab_source)
        self.assertIn(r"\u4e0a\u4f20\u7248\u56fe\u622a\u56fe\u540e\u4f1a\u81ea\u52a8\u8c03\u7528 VLM", layout_tab_source)

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
        self.assertNotIn("layout-mask-agent-local", binding_source)
        self.assertGreaterEqual(
            binding_source.count('concurrency_id="layout-mask-vlm"'),
            4,
        )

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

    def test_reset_keeps_automatic_access_and_clears_draft_without_vlm(self):
        result = demo._layout_mask_agent_reset_callback(
            {"session_id": "ui-test"},
            None,
            "ACT",
        )
        self.assertEqual(len(result), 10)
        self.assertTrue(result[1])
        self.assertEqual(result[2], [])
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        self.assertEqual(result[0]["profile_mode"], "ACT")
        self.assertIsNone(result[0]["consent_image_sha256"])

    def test_upload_immediately_stages_one_automatic_analysis(self):
        image = Image.new("RGB", (80, 60), "white")
        result = demo._layout_mask_agent_prepare_upload_callback(
            {"session_id": "ui-test"},
            image,
            "Auto",
        )
        self.assertEqual(len(result), 10)
        self.assertEqual(result[2][0], {"role": "user", "content": "自动分析"})
        self.assertEqual(result[2][1]["role"], "assistant")
        self.assertIn("正在分析", result[2][1]["content"])
        self.assertEqual(result[7], "自动分析")
        self.assertFalse(result[8]["interactive"])
        self.assertFalse(result[9]["interactive"])

        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn("fn=_layout_mask_agent_prepare_upload_callback", binding_source)
        self.assertIn("bind_layout_agent_result(layout_agent_upload_event)", binding_source)

    def test_clearing_upload_does_not_stage_vlm_request(self):
        result = demo._layout_mask_agent_prepare_upload_callback(
            {"session_id": "ui-test"},
            None,
            "Auto",
        )
        self.assertEqual(result[2], [])
        self.assertEqual(result[7], "")
        self.assertTrue(result[8]["interactive"])
        self.assertTrue(result[9]["interactive"])

    def test_chat_submit_immediately_shows_waiting_state(self):
        result = demo._layout_mask_agent_begin_chat_callback([], "再填一点")
        self.assertEqual(result[0][0]["role"], "user")
        self.assertEqual(result[0][1]["role"], "assistant")
        self.assertIn("180", result[0][1]["content"])
        self.assertEqual(result[1], "再填一点")
        self.assertFalse(result[2]["interactive"])
        self.assertFalse(result[4]["interactive"])
        finished = demo._layout_mask_agent_finish_chat(result[0], "完成")
        self.assertEqual(len(finished), 2)
        self.assertEqual(finished[-1]["content"], "完成")

    def test_local_apply_command_does_not_show_vlm_waiting_state(self):
        result = demo._layout_mask_agent_begin_chat_callback([], "应用推荐参数")
        self.assertIn("本地操作", result[0][1]["content"])
        self.assertNotIn("180", result[0][1]["content"])

    def test_applied_params_refresh_page_previews(self):
        image = Image.new("RGB", (80, 60), "white")
        state = set_layout_mask_agent_consent(
            _new_layout_mask_agent_state("ui-test"),
            session_id="ui-test",
            image=image,
            consent=True,
            profile_mode="ACT",
        )
        params = params_from_controls(12, False, 0, 15, 0, "all", 0)
        state["baseline_params"] = dict(params)
        state["current_draft_params"] = dict(params)
        state["conversation_revision"] = 2
        state["applied_revision"] = 2
        mask_preview, contour_preview = (
            demo._layout_mask_agent_applied_preview_callback(
                {"session_id": "ui-test"}, state, image
            )
        )
        self.assertIsInstance(mask_preview, Image.Image)
        self.assertIsInstance(contour_preview, Image.Image)

    def test_empty_chat_message_does_not_enter_waiting_state(self):
        result = demo._layout_mask_agent_begin_chat_callback([], "   ")
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], "")
        self.assertTrue(result[2]["interactive"])
        self.assertIn("请输入", result[3])
        self.assertTrue(result[4]["interactive"])

    def test_topology_safety_errors_are_actionable_and_keep_internal_text_hidden(self):
        message = demo._layout_mask_agent_error_message(
            ValueError("Candidate merges previously separate components")
        )
        self.assertIn("原本分离的图形粘连", message)
        self.assertIn("保留当前参数和已有 Draft", message)
        self.assertIn("减少粘连", message)
        self.assertNotIn("Candidate merges", message)

        fallback = demo._layout_mask_agent_error_message(RuntimeError("network down"))
        self.assertEqual(fallback, "请求失败：network down")

    def test_chat_callback_preserves_state_when_topology_gate_rejects(self):
        state = {"sentinel": "unchanged"}
        chat = [
            {"role": "user", "content": "分析"},
            {"role": "assistant", "content": "正在分析"},
        ]
        with mock.patch.object(
            demo,
            "_layout_mask_agent_turn_result",
            side_effect=ValueError("Candidate merges previously separate components"),
        ):
            result = demo._layout_mask_agent_chat_callback(
                {"session_id": "ui-test"},
                state,
                Image.new("RGB", (80, 60), "white"),
                True,
                "Auto",
                "分析",
                chat,
                12,
                False,
                0,
                0,
                0,
                "all",
                0,
            )

        self.assertIs(result[0], state)
        self.assertIn("原本分离的图形粘连", result[1][-1]["content"])
        self.assertEqual(result[5], result[1][-1]["content"])
        self.assertNotIn("Candidate merges", result[1][-1]["content"])
        self.assertTrue(result[8]["interactive"])
        self.assertTrue(result[9]["interactive"])

    def test_saved_baseline_only_runs_after_successful_generation(self):
        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertIn("run_layout_mask_event.success(", binding_source)
        self.assertIn("layout_agent_apply_btn.click", binding_source)
        self.assertIn("_layout_mask_agent_applied_preview_callback", binding_source)

    def test_consent_checkbox_and_binding_are_removed(self):
        layout_source = (ROOT / "sam3_demo" / "ui" / "layout_mask_tab.py").read_text(encoding="utf-8")
        binding_source = (ROOT / "sam3_demo" / "ui" / "bindings.py").read_text(encoding="utf-8")
        self.assertNotIn("允许发送当前图像缩略图", layout_source)
        self.assertNotIn("layout_agent_consent.input(", binding_source)
        self.assertNotIn("_layout_mask_agent_consent_callback", binding_source)

    def test_vlm_turn_automatically_binds_current_image_before_request(self):
        image = Image.new("RGB", (80, 60), "white")
        stale_state = _new_layout_mask_agent_state("ui-test")
        controls = params_from_controls(12, False, 0, 15, 0, "all", 0)
        sentinel = object()

        with mock.patch.object(
            demo,
            "run_layout_mask_agent_turn",
            return_value=sentinel,
        ) as run_turn:
            result = demo._layout_mask_agent_turn_result(
                {"session_id": "ui-test"},
                stale_state,
                image,
                False,
                "ACT",
                "自动分析",
                controls["threshold"],
                controls["invert"],
                controls["open_kernel"],
                controls["close_kernel"],
                controls["min_component_area"],
                controls["region_mode"],
                controls["morph_pixels"],
            )

        self.assertIs(result, sentinel)
        submitted_state = run_turn.call_args.kwargs["state"]
        self.assertTrue(run_turn.call_args.kwargs["consent"])
        self.assertEqual(
            submitted_state["consent_image_sha256"],
            submitted_state["image_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
