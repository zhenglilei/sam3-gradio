from __future__ import annotations

import unittest

import gradio as gr

from sam3_demo.overlap_view_callbacks import bind_overlap_prediction_events
from sam3_demo.ui.overlap_authoring_panel import (
    bind_overlap_authoring_events,
    build_overlap_authoring_panel,
)
from sam3_demo.ui.overlap_prediction_tab import build_overlap_prediction_tab


class OverlapAuthoringPanelTests(unittest.TestCase):
    def test_optional_panel_binds_without_pvs_or_session_state(self):
        with gr.Blocks() as demo:
            viewer_refs = build_overlap_prediction_tab(
                authoring_builder=build_overlap_authoring_panel,
                authoring_kwargs={
                    "LayoutTransformEditor": None,
                    "import_error": "not installed",
                },
            )
            bind_overlap_prediction_events(demo=demo, refs=viewer_refs)
            bind_overlap_authoring_events(
                demo=demo,
                refs=viewer_refs.authoring,
                viewer_refs=viewer_refs,
            )

        elem_ids = {
            component.get("props", {}).get("elem_id")
            for component in demo.config["components"]
        }
        self.assertIn("overlap_authoring_panel", elem_ids)

        authoring_names = {
            "new_authoring_draft",
            "refresh_authoring_layouts",
            "load_authoring_layout",
            "set_authoring_name",
            "set_authoring_base_image",
            "add_authoring_layer",
            "select_authoring_layer",
            "remove_authoring_layer",
            "apply_authoring_editor",
            "save_authoring_layout",
            "delete_authoring_layout",
        }
        authoring_fns = [
            fn for fn in demo.fns.values() if fn.name in authoring_names
        ]
        self.assertEqual(len(authoring_fns), 13)
        self.assertTrue(
            all(
                fn.concurrency_id == "overlap-authoring-write"
                for fn in authoring_fns
            )
        )
        self.assertTrue(all(fn.concurrency_limit == 1 for fn in authoring_fns))
        self.assertTrue(all(fn.queue is True for fn in authoring_fns))
        self.assertFalse(
            any(
                "pvs" in fn.name.lower() or "session" in fn.name.lower()
                for fn in authoring_fns
            )
        )

        dependencies = demo.config["dependencies"]
        base_input = next(
            dep
            for dep in dependencies
            if dep.get("api_name") == "set_authoring_base_image"
        )
        self.assertEqual(base_input["targets"][0][1], "input")
        save = next(
            dep
            for dep in dependencies
            if dep.get("api_name") == "save_authoring_layout"
        )
        self.assertEqual(len(save["inputs"]), 2)

    def test_read_only_tab_needs_no_authoring_builder(self):
        with gr.Blocks() as demo:
            refs = build_overlap_prediction_tab()
            bind_overlap_prediction_events(demo=demo, refs=refs)
        self.assertIsNone(refs.authoring)
        self.assertFalse(
            any(
                "authoring" in str(dep.get("api_name") or "")
                for dep in demo.config["dependencies"]
            )
        )


if __name__ == "__main__":
    unittest.main()
