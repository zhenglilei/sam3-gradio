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

    def test_frontend_dispatches_input_only_from_valid_pointerup_path(self):
        source = (ROOT / "layout_region_annotator" / "frontend" / "Index.svelte").read_text(encoding="utf-8")
        self.assertEqual(source.count('gradio.dispatch("input")'), 1)
        pointer_move = source.split("function onPointerMove", 1)[1].split("function appendFinalPoint", 1)[0]
        self.assertNotIn("dispatch", pointer_move)
        pointer_up = source.split("function onPointerUp", 1)[1].split("function cancelDraft", 1)[0]
        self.assertIn("updateBrowserValue(true)", pointer_up)
        self.assertIn("MAX_LASSO_POINTS = 4096", source)
        self.assertIn("SAMPLE_DISTANCE_CSS_PX = 3", source)
        self.assertIn("黄色：Draft", source)
        self.assertIn("绿色：Saved Region", source)

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


if __name__ == "__main__":
    unittest.main()
