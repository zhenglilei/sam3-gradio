from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr

from sam3_demo.ui.top_bar import (
    _button_update,
    _model_status_view,
    _render_status,
    _status_presentation,
    build_top_bar,
)


class TopBarPresentationTest(unittest.TestCase):
    def test_state_mapping_has_text_colour_and_button_policy(self):
        cases = {
            "UNLOADED": ("模型未加载", "#dc2626", True),
            "SEARCHING_GPU": ("启动中", "#d97706", False),
            "WAITING_GPU": ("等待 GPU", "#d97706", False),
            "LOADING": ("启动中", "#d97706", False),
            "WARMUP": ("启动中", "#d97706", False),
            "REHYDRATING": ("启动中", "#d97706", False),
            "STOPPING": ("启动中", "#d97706", False),
            "READY": ("模型已加载", "#16a34a", False),
            "RUNNING": ("推理中", "#16a34a", False),
            "ERROR": ("模型加载失败：磁盘不可用", "#dc2626", True),
        }
        for state, expected in cases.items():
            with self.subTest(state=state):
                snapshot = {
                    "state": state,
                    "last_error": "磁盘不可用" if state == "ERROR" else "",
                }
                self.assertEqual(_status_presentation(snapshot)[1:], expected)

    def test_error_is_compacted_and_html_escaped(self):
        html = _render_status(
            {
                "state": "ERROR",
                "last_error": "<bad>&\n" + ("x" * 300),
            }
        )
        self.assertIn("&lt;bad&gt;&amp;", html)
        self.assertNotIn("<bad>", html)
        # The compact label is rendered both as aria-label and visible text.
        self.assertLessEqual(html.count("x"), 2 * 159)

    def test_unknown_state_is_recoverable(self):
        state, label, colour, enabled = _status_presentation({"state": "future"})
        self.assertEqual(state, "FUTURE")
        self.assertIn("模型状态未知", label)
        self.assertEqual(colour, "#dc2626")
        self.assertTrue(enabled)
        self.assertEqual(_button_update({"state": "future"}), {"value": "启动模型", "interactive": True, "__type__": "update"})

    def test_callbacks_call_only_the_injected_functions(self):
        snapshots = [{"state": "READY"}]
        starts = [{"state": "LOADING"}]
        calls = []

        def snapshot_fn():
            calls.append("snapshot")
            return snapshots[0]

        def start_fn():
            calls.append("start")
            return starts[0]

        status, button = _model_status_view(snapshot_fn)
        self.assertIn("模型已加载", status)
        self.assertFalse(button["interactive"])
        self.assertEqual(calls, ["snapshot"])

        from sam3_demo.ui.top_bar import _request_model_start

        status, button = _request_model_start(start_fn)
        self.assertIn("启动中", status)
        self.assertFalse(button["interactive"])
        self.assertEqual(calls, ["snapshot", "start"])


class TopBarBuildTest(unittest.TestCase):
    def test_build_is_side_effect_free_and_registers_two_nonqueued_events(self):
        calls = []

        def snapshot_fn():
            calls.append("snapshot")
            return {"state": "UNLOADED"}

        def request_start_fn():
            calls.append("start")
            return {"state": "SEARCHING_GPU"}

        with gr.Blocks() as demo:
            refs = build_top_bar(
                snapshot_fn=snapshot_fn,
                request_start_fn=request_start_fn,
                title="Test title",
                subtitle="Test subtitle",
            )

        self.assertEqual(calls, [])
        self.assertTrue(hasattr(refs, "model_status"))
        self.assertTrue(hasattr(refs, "model_start_btn"))
        self.assertTrue(hasattr(refs, "model_status_timer"))

        status_id = refs.model_status._id
        start_id = refs.model_start_btn._id
        timer_id = refs.model_status_timer._id
        matching = []
        for dependency in demo.config["dependencies"]:
            targets = dependency.get("targets") or []
            target_ids = {target[0] for target in targets if target and target[0] is not None}
            if timer_id in target_ids or start_id in target_ids:
                matching.append(dependency)

        self.assertEqual(len(matching), 2)
        self.assertTrue(any(timer_id in {target[0] for target in (d.get("targets") or [])} for d in matching))
        for dependency in matching:
            self.assertFalse(dependency["queue"])
            self.assertEqual(dependency.get("show_progress"), "hidden")
            self.assertIsNone(dependency.get("concurrency_id"))
            self.assertEqual(dependency["outputs"], [status_id, start_id])

        timer_component = next(component for component in demo.config["components"] if component["id"] == timer_id)
        self.assertEqual(timer_component["type"], "timer")


if __name__ == "__main__":
    unittest.main()
