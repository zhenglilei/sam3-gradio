import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import layout_region_utils as regions
import sam3_gradio_demo as demo_module


def _function_dump(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"missing function: {name}")


def _descendant_ids(layout, target_id):
    def find(node):
        if node.get("id") == target_id:
            return node
        for child in node.get("children") or []:
            found = find(child)
            if found is not None:
                return found
        return None

    target = find(layout)
    if target is None:
        return set()

    ids = set()

    def collect(node):
        if node.get("id") is not None:
            ids.add(node["id"])
        for child in node.get("children") or []:
            collect(child)

    collect(target)
    return ids


class LayoutRegionCallbacksTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout_masks = self.root / "layout_masks"
        self.layout_regions = self.root / "layout_regions"
        self.session_id = "session1"
        self.layout_id = "layout1"
        self.source_mask = np.zeros((36, 48), dtype=np.uint8)
        self.source_mask[5:30, 8:40] = 1
        self.source_hash = regions.mask_pixel_sha256(self.source_mask)
        layout_dir = self.layout_masks / self.session_id / self.layout_id
        layout_dir.mkdir(parents=True)
        cv2.imwrite(str(layout_dir / "source_mask.png"), self.source_mask * 255)
        Image.new("RGB", (48, 36), (32, 48, 64)).save(layout_dir / "source_image.png")
        (layout_dir / "layout_meta.json").write_text(
            json.dumps({"source_mask_pixel_sha256": self.source_hash}), encoding="utf-8"
        )
        self.store = regions.LayoutRegionStore(
            layout_masks_root=self.layout_masks,
            layout_regions_root=self.layout_regions,
            categories_path=ROOT / "layout_categories.json",
        )
        self.old_store = demo_module._LAYOUT_REGION_STORE
        self.old_runtime_layout_dir = demo_module.runtime_layout_dir
        demo_module._LAYOUT_REGION_STORE = self.store
        demo_module.runtime_layout_dir = self.layout_masks
        self.layout_state = {
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
        }

    def tearDown(self):
        demo_module._LAYOUT_REGION_STORE = self.old_store
        demo_module.runtime_layout_dir = self.old_runtime_layout_dir
        self.temporary.cleanup()

    def test_preview_save_and_soft_delete_round_trip(self):
        loaded = demo_module._load_layout_region_context(self.layout_state)
        region_state, editor = loaded[0], loaded[1]
        self.assertEqual(editor["server_view"]["regions"], [])

        intent_editor = copy.deepcopy(editor)
        intent_editor["client_intent"]["lasso_polygon"] = [
            [4, 3],
            [44, 3],
            [44, 33],
            [4, 33],
        ]
        previewed = demo_module._preview_layout_region(self.layout_state, region_state, intent_editor)
        region_state, editor = previewed[0], previewed[1]
        self.assertIn("Draft 预览完成", previewed[3])
        self.assertTrue(editor["server_view"]["draft_region_overlay_image"].startswith("data:image/png;base64,"))

        saved = demo_module._save_layout_region(
            self.layout_state,
            region_state,
            editor,
            "metal",
            "M1_power",
        )
        region_state, editor = saved[0], saved[1]
        self.assertIn("已保存 R1 metal", saved[7])
        self.assertEqual(editor["server_view"]["draft_region_overlay_image"], "")
        self.assertEqual(editor["server_view"]["regions"][0]["name"], "M1_power")
        self.assertTrue(editor["server_view"]["saved_region_overlay_image"].startswith("data:image/png;base64,"))

        deleted = demo_module._delete_layout_region(
            self.layout_state,
            region_state,
            editor,
            1,
        )
        self.assertIn("已软删除 R1", deleted[7])
        self.assertEqual(deleted[1]["server_view"]["regions"], [])
        document, _ = self.store.load_document(self.session_id, self.layout_id, self.source_hash)
        self.assertIsNotNone(document["regions"][0]["deleted_at"])
        self.assertIn("mask_rle", document["regions"][0])

    def test_create_demo_keeps_region_controls_in_layout_tab(self):
        app = demo_module.create_demo()
        config = app.config
        component_by_id = {component["id"]: component for component in config["components"]}
        tab_ids = {
            component.get("props", {}).get("id"): component["id"]
            for component in config["components"]
            if component.get("type") == "tabitem"
        }
        layout_ids = _descendant_ids(config["layout"], tab_ids["tab_layout_mask"])
        image_ids = _descendant_ids(config["layout"], tab_ids["tab_image"])
        region_ids = {
            component_id
            for component_id, component in component_by_id.items()
            if "Region" in str(component.get("props", {}).get("label", ""))
            or component.get("props", {}).get("elem_id") == "layout_region_annotator"
        }
        self.assertTrue(region_ids)
        self.assertTrue(region_ids <= layout_ids)
        self.assertFalse(region_ids & image_ids)

        dependencies = {item.get("api_name"): item for item in config["dependencies"]}
        run_dependency = dependencies["_run_layout_mask_page"]
        clear_dependency = dependencies["_clear_current_layout_mask"]
        self.assertEqual((len(run_dependency["inputs"]), len(run_dependency["outputs"])), (9, 8))
        self.assertEqual((len(clear_dependency["inputs"]), len(clear_dependency["outputs"])), (2, 8))

    def test_protected_function_bodies_match_baseline(self):
        baseline = subprocess.run(
            ["git", "show", "fef70b7:sam3_gradio_demo.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        current = (ROOT / "sam3_gradio_demo.py").read_text(encoding="utf-8")
        protected = [
            "_finish_native_polygon",
            "_run_pcs",
            "_create_pvs_from_pending_boxes",
            "_pvs_point_prompt",
            "_sync_layout_controls_from_editor",
            "_run_layout_mask_page",
            "_clear_current_layout_mask",
            "_commit_layout_transform",
            "_update_layout_preview",
            "_create_pvs_from_layout_mask",
        ]
        for name in protected:
            with self.subTest(name=name):
                self.assertEqual(_function_dump(current, name), _function_dump(baseline, name))


if __name__ == "__main__":
    unittest.main()
