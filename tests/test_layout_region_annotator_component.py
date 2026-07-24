import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "layout_region_annotator" / "backend"))

from gradio_layout_region_annotator import LayoutRegionAnnotator


class LayoutRegionAnnotatorComponentTest(unittest.TestCase):
    def test_preprocess_whitelists_only_client_intent(self):
        component = LayoutRegionAnnotator()
        sanitized = component.preprocess(
            {
                "server_view": {
                    "regions": [{"mask_rle": {"counts": "forged"}}],
                    "saved_region_overlay_image": "forged",
                },
                "client_intent": {
                    "tool_mode": "lasso",
                    "lasso_polygon": [[1, 2], [3, 4], [5, 6]],
                    "expected_regions_revision": 7,
                    "session_id": "session1",
                    "layout_id": "layout1",
                    "source_mask_hash": "hash1",
                    "area": 999,
                    "bbox_xywh": [0, 0, 1, 1],
                    "mask_rle": {"counts": "forged"},
                },
            }
        )
        self.assertEqual(
            set(sanitized),
            {
                "tool_mode",
                "lasso_polygon",
                "expected_regions_revision",
                "session_id",
                "layout_id",
                "source_mask_hash",
            },
        )
        self.assertNotIn("server_view", sanitized)
        self.assertNotIn("mask_rle", sanitized)
        self.assertNotIn("area", sanitized)
        self.assertNotIn("bbox_xywh", sanitized)

    def test_frontend_dispatches_input_only_from_explicit_finish(self):
        source = (ROOT / "layout_region_annotator" / "frontend" / "Index.svelte").read_text(encoding="utf-8")
        self.assertEqual(source.count('gradio.dispatch("input")'), 1)
        pointer_move = source.split("function onPointerMove", 1)[1].split("function appendFinalPoint", 1)[0]
        self.assertNotIn("dispatch", pointer_move)
        pointer_up = source.split("function onPointerUp", 1)[1].split("function cancelDraft", 1)[0]
        self.assertNotIn("updateBrowserValue(true)", pointer_up)
        self.assertIn("updateBrowserValue(false)", pointer_up)
        finish_draft = source.split("function finishDraft", 1)[1].split(
            "function onPointerUp", 1
        )[0]
        self.assertIn("updateBrowserValue(true)", finish_draft)
        self.assertIn("MAX_LASSO_POINTS = 4096", source)
        self.assertIn("SAMPLE_DISTANCE_CSS_PX = 3", source)
        self.assertIn("黄色：Draft", source)
        self.assertIn("绿色：Saved Label", source)

    def test_frontend_supports_mixed_freehand_and_straight_segments(self):
        source = (
            ROOT / "layout_region_annotator" / "frontend" / "Index.svelte"
        ).read_text(encoding="utf-8")
        self.assertIn('let openDraft = $state(false)', source)
        self.assertIn("function finishDraft", source)
        finish_click = source.split("function finishDraft", 1)[1].split(
            "function onPointerUp", 1
        )[0]
        self.assertIn("points.length < 3", finish_click)
        self.assertIn("uniquePointCount() < 3", finish_click)
        self.assertIn("updateBrowserValue(true)", finish_click)
        pointer_up = source.split("function onPointerUp", 1)[1].split(
            "function cancelDraft", 1
        )[0]
        self.assertIn("if (freehandGesture)", pointer_up)
        self.assertIn("finishPointer(event, true)", pointer_up)
        self.assertGreaterEqual(pointer_up.count("openDraft = true"), 2)
        self.assertIn("updateBrowserValue(false)", pointer_up)
        self.assertNotIn("updateBrowserValue(true)", pointer_up)
        self.assertIn("完成套索", source)
        self.assertIn("!drawing && !openDraft", source)

    def test_frontend_binds_loaded_images_and_draft_to_layout_identity(self):
        source = (
            ROOT / "layout_region_annotator" / "frontend" / "Index.svelte"
        ).read_text(encoding="utf-8")
        self.assertIn("const generation = ++imageLoadGeneration", source)
        self.assertIn("generation === imageLoadGeneration", source)
        self.assertIn("const nextIdentity = layoutIdentity(intent)", source)
        self.assertIn("if (!imagesReady)", source)
        self.assertIn("activeDraftIdentity = loadedIdentity", source)
        self.assertGreaterEqual(
            source.count("activeDraftIdentity !== loadedIdentity"),
            2,
        )
        self.assertIn("Layout 已切换，Draft 已清除", source)

    def test_frontend_publishes_only_small_client_intent(self):
        source = (
            ROOT / "layout_region_annotator" / "frontend" / "Index.svelte"
        ).read_text(encoding="utf-8")
        body = source.split(
            "function updateBrowserValue",
            1,
        )[1].split("function setToolMode", 1)[0]

        self.assertIn("const outbound: LayoutRegionAnnotatorValue", body)
        self.assertIn("client_intent:", body)
        self.assertIn("gradio.props.value = outbound", body)
        self.assertNotIn("gradio.props.value = localValue", body)
        for server_only_field in (
            "server_view:",
            "source_image:",
            "source_mask_image:",
            "saved_region_overlay_image:",
            "draft_region_overlay_image:",
            "data:image",
        ):
            self.assertNotIn(server_only_field, body)

    def test_frontend_canonicalizes_region_summaries_to_label(self):
        types_source = (
            ROOT / "layout_region_annotator" / "frontend" / "types.ts"
        ).read_text(encoding="utf-8")
        index_source = (
            ROOT / "layout_region_annotator" / "frontend" / "Index.svelte"
        ).read_text(encoding="utf-8")

        self.assertIn("label?: string;", types_source)
        self.assertIn("class_label?: string;", types_source)
        self.assertIn("name?: string;", types_source)
        self.assertIn("function regionSummaryLabel", types_source)
        self.assertIn("${classLabel} / ${name}", types_source)
        self.assertIn("name || classLabel || `R${region.region_id}`", types_source)
        self.assertIn("label: regionSummaryLabel(region)", index_source)


if __name__ == "__main__":
    unittest.main()
