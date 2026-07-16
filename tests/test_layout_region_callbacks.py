import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

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
        self.old_runtime_export_dir = demo_module.runtime_export_dir
        self.old_public_download_dir = demo_module.public_download_dir
        demo_module._LAYOUT_REGION_STORE = self.store
        demo_module.runtime_layout_dir = self.layout_masks
        demo_module.runtime_export_dir = self.root / "internal_exports"
        demo_module.runtime_export_dir.mkdir()
        demo_module.public_download_dir = self.root / "public_downloads"
        self.layout_state = {
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_pixel_sha256": self.source_hash,
        }

    def tearDown(self):
        demo_module._LAYOUT_REGION_STORE = self.old_store
        demo_module.runtime_layout_dir = self.old_runtime_layout_dir
        demo_module.runtime_export_dir = self.old_runtime_export_dir
        demo_module.public_download_dir = self.old_public_download_dir
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

        archive_path, export_status = demo_module._export_layout_regions(
            self.layout_state,
            region_state,
        )
        self.assertIn("Region", export_status)
        archive_path = Path(archive_path).resolve()
        self.assertIn(
            (self.root / "public_downloads" / "region_annotation_exports").resolve(),
            archive_path.parents,
        )
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"manifest.json", "regions.json", "source_mask.png"},
            )
            exported_document = json.loads(archive.read("regions.json"))
            exported_manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(exported_document["regions"][0]["name"], "M1_power")
        self.assertEqual(exported_manifest["active_region_count"], 1)
        self.assertEqual(exported_manifest["regions_revision"], 1)

        stale_state = copy.deepcopy(region_state)
        stale_state["regions_revision"] = 0
        stale_path, stale_status = demo_module._export_layout_regions(
            self.layout_state,
            stale_state,
        )
        self.assertIsNone(stale_path)
        self.assertIn("stale regions revision", stale_status)

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

    def test_stale_draft_cannot_bind_to_a_different_layout(self):
        loaded_a = demo_module._load_layout_region_context(self.layout_state)
        region_state_a, editor_a = loaded_a[0], loaded_a[1]
        stale_editor = copy.deepcopy(editor_a)
        stale_editor["client_intent"]["lasso_polygon"] = [
            [4, 3],
            [44, 3],
            [44, 33],
            [4, 33],
        ]

        layout_id_b = "layout2"
        source_mask_b = np.zeros((36, 48), dtype=np.uint8)
        source_mask_b[10:25, 12:36] = 1
        source_hash_b = regions.mask_pixel_sha256(source_mask_b)
        layout_dir_b = self.layout_masks / self.session_id / layout_id_b
        layout_dir_b.mkdir(parents=True)
        cv2.imwrite(str(layout_dir_b / "source_mask.png"), source_mask_b * 255)
        Image.new("RGB", (48, 36), (80, 64, 48)).save(
            layout_dir_b / "source_image.png"
        )
        (layout_dir_b / "layout_meta.json").write_text(
            json.dumps({"source_mask_pixel_sha256": source_hash_b}),
            encoding="utf-8",
        )
        layout_state_b = {
            "session_id": self.session_id,
            "layout_id": layout_id_b,
            "source_mask_pixel_sha256": source_hash_b,
        }
        loaded_b = demo_module._load_layout_region_context(layout_state_b)
        region_state_b = loaded_b[0]

        stale_preview = demo_module._preview_layout_region(
            layout_state_b,
            region_state_b,
            stale_editor,
        )
        self.assertIn("does not match current layout", stale_preview[3])
        self.assertEqual(
            stale_preview[1]["client_intent"]["layout_id"],
            layout_id_b,
        )
        self.assertEqual(stale_preview[1]["client_intent"]["lasso_polygon"], [])
        self.assertEqual(
            stale_preview[1]["server_view"]["draft_region_overlay_image"],
            "",
        )

        retagged_editor = copy.deepcopy(stale_editor)
        retagged_editor["client_intent"].update(
            {
                "layout_id": layout_id_b,
                "source_mask_hash": source_hash_b,
                "expected_regions_revision": region_state_b["regions_revision"],
            }
        )
        retagged_preview = demo_module._preview_layout_region(
            layout_state_b,
            region_state_a,
            retagged_editor,
        )
        self.assertIn("Region state layout_id does not match", retagged_preview[3])
        self.assertEqual(
            retagged_preview[1]["client_intent"]["layout_id"],
            layout_id_b,
        )
        self.assertEqual(retagged_preview[1]["client_intent"]["lasso_polygon"], [])

        failed_save = demo_module._save_layout_region(
            layout_state_b,
            region_state_a,
            retagged_editor,
            "metal",
            "must_not_save",
        )
        self.assertIn("Region state layout_id does not match", failed_save[7])
        self.assertEqual(
            failed_save[1]["client_intent"]["layout_id"],
            layout_id_b,
        )
        self.assertEqual(failed_save[1]["client_intent"]["lasso_polygon"], [])
        self.assertEqual(
            failed_save[1]["server_view"]["draft_region_overlay_image"],
            "",
        )
        document_b, _ = self.store.load_document(
            self.session_id,
            layout_id_b,
            source_hash_b,
        )
        self.assertEqual(regions.active_regions(document_b), [])
        self.assertEqual(document_b["regions_revision"], 0)

    def test_layout_download_wrapper_publishes_copies_only_for_file_outputs(self):
        internal_dir = self.root / "internal_layout"
        internal_dir.mkdir()
        internal_mask = internal_dir / "source_mask.png"
        internal_contours = internal_dir / "contours.json"
        internal_mask.write_bytes(b"mask")
        internal_contours.write_text('{"contours": []}', encoding="utf-8")
        original_result = (
            {"layout_id": self.layout_id},
            None,
            None,
            None,
            None,
            str(internal_mask),
            str(internal_contours),
            f"mask: {internal_mask}\ncontours: {internal_contours}",
        )

        with mock.patch.object(
            demo_module,
            "_run_layout_mask_page",
            return_value=original_result,
        ):
            result = demo_module._run_layout_mask_page_with_downloads(*([None] * 9))

        public_mask = Path(result[5]).resolve()
        public_contours = Path(result[6]).resolve()
        public_category = (
            self.root / "public_downloads" / "layout_mask_exports"
        ).resolve()
        self.assertIn(public_category, public_mask.parents)
        self.assertEqual(public_mask.parent, public_contours.parent)
        self.assertEqual(public_mask.read_bytes(), b"mask")
        self.assertEqual(
            public_contours.read_text(encoding="utf-8"),
            '{"contours": []}',
        )
        self.assertNotIn(str(internal_dir), result[7])

    def test_gradio_file_access_boundary_blocks_internal_paths(self):
        repository = self.root / "repository"
        runtime = repository / ".runtime"
        public = repository / "public_downloads"
        gradio_runtime = runtime / "gradio"
        video_runtime = runtime / "videos"
        for directory in (
            repository / "models",
            repository / ".gradio",
            repository / "layout_region_annotator",
            runtime / "layout_masks",
            runtime / "layout_regions",
            runtime / "feedback",
            runtime / "exports",
            gradio_runtime,
            video_runtime,
            public,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        source_file = repository / "sam3_gradio_demo.py"
        source_file.write_text("secret", encoding="utf-8")

        with mock.patch.multiple(
            demo_module,
            current_dir=repository,
            runtime_dir=runtime,
            runtime_gradio_dir=gradio_runtime,
            runtime_video_dir=video_runtime,
            public_download_dir=public,
        ):
            allowed = demo_module._gradio_allowed_paths()
            blocked = set(demo_module._gradio_blocked_paths())

        self.assertEqual(allowed, [str(public.resolve())])
        self.assertIn(str(source_file.resolve()), blocked)
        self.assertIn(str((repository / "models").resolve()), blocked)
        self.assertIn(str((runtime / "layout_masks").resolve()), blocked)
        self.assertIn(str((runtime / "layout_regions").resolve()), blocked)
        self.assertIn(str((runtime / "feedback").resolve()), blocked)
        self.assertIn(str((runtime / "exports").resolve()), blocked)
        self.assertNotIn(str(public.resolve()), blocked)
        self.assertNotIn(str((repository / ".gradio").resolve()), blocked)
        self.assertNotIn(str(gradio_runtime.resolve()), blocked)
        self.assertNotIn(str(video_runtime.resolve()), blocked)

    def test_pcs_export_pool_returns_public_zip(self):
        image_id = "export-image"
        image = Image.new("RGB", (12, 10), (20, 30, 40))
        image_state = {
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
        }
        pcs_state = demo_module._new_pcs_state()
        pvs_state = demo_module._new_pvs_state()
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[2:8, 3:9] = True
        pcs_state["instances"][1] = demo_module._make_inst(
            1,
            "pcs",
            mask,
            [3, 2, 9, 8],
            0.9,
        )
        demo_module._WORKSPACE_CACHE[image_id] = {"image": image}
        try:
            with mock.patch.object(
                demo_module,
                "compare_with_coco",
                return_value={"summary_lines": []},
            ):
                archive_path, info = demo_module._export_pool(
                    image_state,
                    pcs_state,
                    pvs_state,
                    demo_module.MODE_PCS,
                    "pcs",
                    "GE1_coco",
                    "",
                    "auto",
                    demo_module.coco_eval_scope_overlap,
                    None,
                )
        finally:
            demo_module._WORKSPACE_CACHE.pop(image_id, None)

        archive_path = Path(archive_path).resolve()
        public_category = (
            self.root / "public_downloads" / "pcs_pvs_exports"
        ).resolve()
        self.assertIn(public_category, archive_path.parents)
        self.assertIn("Exported 1 PCS", info)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        self.assertIn("overlay.png", names)
        self.assertIn("prediction.json", names)
        self.assertIn("metrics.json", names)
        self.assertIn("coco_masks.json", names)
        self.assertIn("masks/pcs_001.png", names)

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
        self.assertIn("_export_layout_regions", dependencies)
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
