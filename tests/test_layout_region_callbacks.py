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
        self.assertEqual(len(loaded), 7)
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
            "M1_power",
        )
        self.assertEqual(len(saved), 7)
        region_state, editor = saved[0], saved[1]
        self.assertIn("已保存 Label M1_power", saved[6])
        self.assertEqual(editor["server_view"]["draft_region_overlay_image"], "")
        self.assertEqual(editor["server_view"]["regions"][0]["label"], "M1_power")
        self.assertTrue(editor["server_view"]["saved_region_overlay_image"].startswith("data:image/png;base64,"))

        archive_path, export_status = demo_module._export_layout_regions(
            self.layout_state,
            region_state,
        )
        self.assertIn("label 标注", export_status)
        archive_path = Path(archive_path).resolve()
        self.assertIn(
            (self.root / "public_downloads" / "region_annotation_exports").resolve(),
            archive_path.parents,
        )
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "manifest.json",
                    "regions.json",
                    "source_mask.png",
                    "region_label_index.png",
                    "labels.json",
                    "label_masks/label_0001_R1.png",
                },
            )
            exported_document = json.loads(archive.read("regions.json"))
            exported_manifest = json.loads(archive.read("manifest.json"))
            exported_labels = json.loads(archive.read("labels.json"))
            exported_source = cv2.imdecode(
                np.frombuffer(archive.read("source_mask.png"), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            exported_index = cv2.imdecode(
                np.frombuffer(
                    archive.read("region_label_index.png"),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_UNCHANGED,
            )
            exported_label_mask = cv2.imdecode(
                np.frombuffer(
                    archive.read("label_masks/label_0001_R1.png"),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_GRAYSCALE,
            )
        self.assertEqual(exported_document["regions"][0]["label"], "M1_power")
        self.assertEqual(exported_manifest["active_region_count"], 1)
        self.assertEqual(exported_manifest["regions_revision"], 1)
        self.assertEqual(exported_manifest["label_mask_count"], 1)
        self.assertEqual(
            exported_manifest["label_mask_files"],
            ["label_masks/label_0001_R1.png"],
        )
        self.assertIn("label_masks/label_0001_R1.png", exported_manifest["files"])
        self.assertEqual(exported_index.dtype, np.uint16)
        self.assertEqual(exported_labels["labels"][0]["label"], "M1_power")
        self.assertEqual(
            exported_labels["labels"][0]["mask_file"],
            "label_masks/label_0001_R1.png",
        )
        np.testing.assert_array_equal(
            exported_source >= 128,
            self.source_mask.astype(bool),
        )
        expected_region_mask = regions.decode_binary_mask(
            exported_document["regions"][0]["mask_rle"],
            self.source_mask.shape,
        )
        np.testing.assert_array_equal(exported_index == 1, expected_region_mask)
        self.assertEqual(set(np.unique(exported_label_mask)), {0, 255})
        np.testing.assert_array_equal(
            exported_label_mask == 255,
            expected_region_mask,
        )

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
        self.assertEqual(len(deleted), 7)
        self.assertIn("已软删除 Label M1_power", deleted[6])
        self.assertEqual(deleted[1]["server_view"]["regions"], [])
        document, _ = self.store.load_document(self.session_id, self.layout_id, self.source_hash)
        self.assertIsNotNone(document["regions"][0]["deleted_at"])
        self.assertIn("mask_rle", document["regions"][0])

    def test_export_writes_one_binary_mask_per_active_label(self):
        polygons = [
            ([[8, 5], [17, 5], [17, 29], [8, 29]], "left"),
            ([[19, 5], [27, 5], [27, 29], [19, 29]], "middle"),
            ([[29, 5], [39, 5], [39, 29], [29, 29]], "right"),
        ]
        document = None
        for expected_revision, (polygon, label) in enumerate(polygons):
            document, _ = self.store.save_region(
                session_id=self.session_id,
                layout_id=self.layout_id,
                source_mask_hash=self.source_hash,
                expected_revision=expected_revision,
                lasso_polygon=polygon,
                label=label,
            )
        document, _ = self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=3,
            region_id=2,
        )
        self.assertEqual(document["regions_revision"], 4)

        region_state = {
            "session_id": self.session_id,
            "layout_id": self.layout_id,
            "source_mask_hash": self.source_hash,
            "regions_revision": 4,
        }
        archive_path, status = demo_module._export_layout_regions(
            self.layout_state,
            region_state,
        )
        self.assertIn("label_masks=2", status)

        expected_files = [
            "label_masks/label_0001_R1.png",
            "label_masks/label_0002_R3.png",
        ]
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            exported_document = json.loads(archive.read("regions.json"))
            exported_manifest = json.loads(archive.read("manifest.json"))
            exported_labels = json.loads(archive.read("labels.json"))
            exported_masks = {
                path: cv2.imdecode(
                    np.frombuffer(archive.read(path), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in expected_files
            }

        self.assertTrue(set(expected_files) <= names)
        self.assertNotIn("label_masks/label_0002_R2.png", names)
        self.assertEqual(exported_manifest["label_mask_count"], 2)
        self.assertEqual(exported_manifest["label_mask_files"], expected_files)
        self.assertEqual(
            [entry["region_id"] for entry in exported_labels["labels"]],
            [1, 3],
        )
        self.assertEqual(
            [entry["label"] for entry in exported_labels["labels"]],
            ["left", "right"],
        )
        self.assertEqual(
            [entry["mask_file"] for entry in exported_labels["labels"]],
            expected_files,
        )

        record_by_id = {
            int(record["region_id"]): record
            for record in exported_document["regions"]
        }
        for entry in exported_labels["labels"]:
            mask = exported_masks[entry["mask_file"]]
            self.assertTrue(set(np.unique(mask)) <= {0, 255})
            expected_mask = regions.decode_binary_mask(
                record_by_id[int(entry["region_id"])]["mask_rle"],
                self.source_mask.shape,
            )
            np.testing.assert_array_equal(mask == 255, expected_mask)

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
            "must_not_save",
        )
        self.assertEqual(len(failed_save), 7)
        self.assertIn("Region state layout_id does not match", failed_save[6])
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
        self.assertIn(str(video_runtime.resolve()), blocked)

    def test_pcs_export_pool_returns_public_zip(self):
        image_id = "export-image"
        session_id = "export-session"
        image = Image.new("RGB", (12, 10), (20, 30, 40))
        target_hash = demo_module._layout_tx.image_pixel_sha256(image)
        image_state = {
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
            "session_id": session_id,
            "target_image_sha256": target_hash,
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
        demo_module._WORKSPACE_CACHE[image_id] = {
            "image": image,
            "session_id": session_id,
            "target_image_sha256": target_hash,
        }
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
        point_refine_accordions = [
            component
            for component in config["components"]
            if component.get("type") == "accordion"
            and component.get("props", {}).get("label") == "点提示修缮"
        ]
        self.assertEqual(len(point_refine_accordions), 1)
        point_refine_accordion = point_refine_accordions[0]
        self.assertIn(point_refine_accordion["id"], image_ids)
        self.assertNotIn(point_refine_accordion["id"], layout_ids)
        self.assertFalse(point_refine_accordion["props"]["visible"])

        layout_prompt_selectors = [
            component
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
            == "layout_prompt_mask_selector"
        ]
        self.assertEqual(len(layout_prompt_selectors), 1)
        layout_prompt_selector = layout_prompt_selectors[0]
        self.assertEqual(layout_prompt_selector["type"], "checkboxgroup")
        self.assertIn(layout_prompt_selector["id"], image_ids)
        self.assertNotIn(layout_prompt_selector["id"], layout_ids)
        self.assertEqual(
            layout_prompt_selector["props"]["value"],
            [demo_module._LAYOUT_PROMPT_SCOPE_FULL],
        )
        self.assertFalse(layout_prompt_selector["props"]["interactive"])

        morph_controls = [
            component
            for component in config["components"]
            if component.get("props", {}).get("label")
            == "膨胀/腐蚀像素（正数膨胀，负数腐蚀）"
        ]
        self.assertEqual(len(morph_controls), 1)
        morph_control = morph_controls[0]
        self.assertIn(morph_control["id"], layout_ids)
        self.assertEqual(morph_control["props"]["minimum"], -31)
        self.assertEqual(morph_control["props"]["maximum"], 31)
        self.assertEqual(morph_control["props"]["value"], 0)

        dependencies = {item.get("api_name"): item for item in config["dependencies"]}
        run_dependency = dependencies["_run_layout_mask_page"]
        self.assertIn(morph_control["id"], run_dependency["inputs"])
        clear_dependency = dependencies["_clear_current_layout_mask"]
        self.assertIn("_export_layout_regions", dependencies)
        self.assertEqual((len(run_dependency["inputs"]), len(run_dependency["outputs"])), (10, 8))
        self.assertEqual((len(clear_dependency["inputs"]), len(clear_dependency["outputs"])), (2, 8))
        layout_point_dependency = dependencies["_layout_point_refine"]
        pvs_point_dependency = dependencies["_pvs_point_prompt"]
        mode_dependency = dependencies["_switch_mode_with_layout_editor"]
        upload_point_cleanup = dependencies["_clear_pending_point_payload"]
        self.assertEqual(
            (len(layout_point_dependency["inputs"]), len(layout_point_dependency["outputs"])),
            (7, 11),
        )
        self.assertEqual(
            (len(pvs_point_dependency["inputs"]), len(pvs_point_dependency["outputs"])),
            (6, 9),
        )
        self.assertEqual(
            (len(upload_point_cleanup["inputs"]), len(upload_point_cleanup["outputs"])),
            (0, 1),
        )
        self.assertEqual(len(mode_dependency["outputs"]), 28)
        self.assertIn(point_refine_accordion["id"], mode_dependency["outputs"])
        self.assertIsNotNone(upload_point_cleanup.get("trigger_after"))

        use_current_dependency = dependencies["_use_current_layout_mask"]
        load_choices_dependency = dependencies["_load_layout_prompt_choices"]
        select_prompt_dependency = dependencies["_select_layout_prompt_mask"]
        create_region_pvs_dependency = dependencies[
            "_create_pvs_from_layout_selection"
        ]
        self.assertEqual(
            (
                len(use_current_dependency["inputs"]),
                len(use_current_dependency["outputs"]),
            ),
            (2, 3),
        )
        self.assertEqual(
            (
                len(load_choices_dependency["inputs"]),
                len(load_choices_dependency["outputs"]),
            ),
            (2, 4),
        )
        self.assertIsNotNone(load_choices_dependency.get("trigger_after"))
        self.assertEqual(
            (
                len(select_prompt_dependency["inputs"]),
                len(select_prompt_dependency["outputs"]),
            ),
            (3, 10),
        )
        self.assertEqual(
            (
                len(create_region_pvs_dependency["inputs"]),
                len(create_region_pvs_dependency["outputs"]),
            ),
            (13, 12),
        )
        sync_transform_dependency = dependencies[
            "_sync_layout_controls_from_editor_with_prompt_epoch"
        ]
        reset_transform_dependency = dependencies[
            "_reset_layout_controls_with_prompt_epoch"
        ]
        self.assertEqual(
            (
                len(sync_transform_dependency["inputs"]),
                len(sync_transform_dependency["outputs"]),
            ),
            (2, 8),
        )
        self.assertEqual(
            (
                len(reset_transform_dependency["inputs"]),
                len(reset_transform_dependency["outputs"]),
            ),
            (2, 9),
        )

        reset_dependencies = [
            item
            for item in config["dependencies"]
            if str(item.get("api_name") or "").startswith(
                "_reset_layout_prompt_selection"
            )
        ]
        self.assertGreaterEqual(len(reset_dependencies), 6)
        self.assertTrue(
            all(
                (len(item["inputs"]), len(item["outputs"])) == (2, 3)
                and item.get("trigger_after") is not None
                for item in reset_dependencies
            )
        )

        source = (ROOT / "sam3_gradio_demo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        create_demo_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_demo"
        )
        common_assignment = next(
            node for node in ast.walk(create_demo_node)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "common"
                for target in node.targets
            )
        )
        self.assertEqual(
            [item.id for item in common_assignment.value.elts],
            [
                "image_upload", "result_image", "analysis_report", "pcs_summary",
                "pvs_summary", "active_pvs", "interaction_info", "pvs_pending_count",
            ],
        )
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
            "_clear_current_layout_mask",
            "_commit_layout_transform",
            "_create_pvs_from_layout_mask",
        ]
        for name in protected:
            with self.subTest(name=name):
                self.assertEqual(_function_dump(current, name), _function_dump(baseline, name))


if __name__ == "__main__":
    unittest.main()
