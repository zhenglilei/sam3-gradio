from __future__ import annotations

import unittest

import gradio as gr

from sam3_demo.overlap_view_callbacks import bind_overlap_prediction_events
from sam3_demo.ui.overlap_prediction_tab import build_overlap_prediction_tab


class OverlapPredictionTabTests(unittest.TestCase):
    def test_tab_and_event_contract_are_standalone(self):
        with gr.Blocks() as demo:
            refs = build_overlap_prediction_tab()
            bind_overlap_prediction_events(demo=demo, refs=refs)
        self.assertIsNone(refs.authoring)

        tab_ids = {
            component.get("props", {}).get("id")
            for component in demo.config["components"]
            if component.get("type") == "tabitem"
        }
        self.assertIn("tab_overlap_prediction", tab_ids)
        names = {
            dependency.get("api_name")
            for dependency in demo.config["dependencies"]
        }
        self.assertEqual(
            names,
            {
                "refresh_overlap_view",
                "refresh_overlap_view_1",
                "select_overlap_layout",
                "render_overlap_layer",
            },
        )
        functions = [
            block_fn
            for block_fn in demo.fns.values()
            if block_fn.name
            in {
                "refresh_overlap_view",
                "select_overlap_layout",
                "render_overlap_layer",
            }
        ]
        self.assertEqual(len(functions), 4)
        self.assertEqual(
            [fn.name for fn in functions].count("refresh_overlap_view"),
            2,
        )
        self.assertTrue(
            all(fn.concurrency_id == "overlap-read-view" for fn in functions)
        )
        self.assertTrue(all(fn.concurrency_limit == 2 for fn in functions))
        self.assertTrue(all(fn.queue is True for fn in functions))
        self.assertFalse(
            any(
                any(target[1] == "load" for target in dep.get("targets") or [])
                and dep.get("api_name", "").startswith("refresh_overlap_view")
                for dep in demo.config["dependencies"]
            )
        )


if __name__ == "__main__":
    unittest.main()
