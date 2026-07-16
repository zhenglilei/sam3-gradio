from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script(module_name: str, script_name: str):
    """Load an offline collector without letting import-time setup touch .runtime."""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / script_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with mock.patch.object(Path, "mkdir", autospec=True):
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module

PCS_EVAL = _load_script("offline_eval_pcs_collector_test", "run_pcs_o3_grouped_eval.py")

LABEL_EVAL = _load_script("offline_eval_label_collector_test", "run_pvs_bbox_label_eval.py")
GROUPED_EVAL = _load_script("offline_eval_group_collector_test", "run_pvs_bbox_grouped_eval.py")


class T4CocoFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.layer_dir = root / "ACT"
        self.annotation_path = self.layer_dir / "annotations" / "instances_all.json"
        self.image_path = self.layer_dir / "images" / "x.png"

    def write(
        self,
        *,
        write_image: bool = True,
        image_id: int = 11,
        original_label: Any = "ACT_HOLE_1",
        annotation_image_id: int = 11,
        legacy_labelme: bool = False,
    ) -> dict:
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        if write_image:
            self.image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), (30, 60, 90)).save(self.image_path)

        if legacy_labelme:
            document = {
                "version": "5.0.1",
                "imagePath": "x.png",
                "imageWidth": 8,
                "imageHeight": 6,
                "shapes": [
                    {
                        "label": "ACT-1",
                        "shape_type": "polygon",
                        "points": [[1, 1], [5, 1], [5, 4], [1, 4]],
                    }
                ],
            }
        else:
            document = {
                "info": {"description": "synthetic current T4 COCO fixture"},
                "licenses": [],
                "categories": [{"id": 7, "name": "ACT", "supercategory": "layout_layer"}],
                "images": [
                    {
                        "id": image_id,
                        "file_name": "images/x.png",
                        "width": 8,
                        "height": 6,
                    }
                ],
                "annotations": [
                    {
                        "id": 23,
                        "image_id": annotation_image_id,
                        "category_id": 7,
                        "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
                        "area": 12,
                        "bbox": [1, 1, 4, 3],
                        "iscrowd": 0,
                        "original_label": original_label,
                        "source_shape_index": 9,
                    }
                ],
            }
        self.annotation_path.write_text(json.dumps(document), encoding="utf-8")
        return document

    @property
    def annotation_sha256(self) -> str:
        return hashlib.sha256(self.annotation_path.read_bytes()).hexdigest()


class OfflineT4CocoCollectorTest(unittest.TestCase):
    def test_current_coco_fixture_is_collected_with_provenance_and_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = T4CocoFixture(Path(tmp) / "T4" / "original_size")
            fixture.write()

            samples = LABEL_EVAL.collect_t4_samples(fixture.root, "all")
            groups = GROUPED_EVAL.collect_t4_groups(fixture.root, "all")

            self.assertEqual(len(samples), 1)
            self.assertEqual(len(groups), 1)
            sample = samples[0]
            group = groups[0]
            self.assertEqual(sample.source, "T4")
            self.assertEqual(sample.category_label, "ACT")
            self.assertEqual(sample.label, "ACT_HOLE_1")
            self.assertEqual(sample.dataset, "original_size/ACT")
            self.assertEqual(sample.split, "original_size")
            self.assertEqual(sample.image_path, str(fixture.image_path))
            self.assertEqual(sample.image_id, "11")
            self.assertEqual(sample.annotation_id, "23")
            self.assertEqual(sample.annotation_relpath, "ACT/annotations/instances_all.json")
            self.assertEqual(sample.annotation_sha256, fixture.annotation_sha256)
            self.assertEqual(sample.bbox_xyxy, [1.0, 1.0, 5.0, 4.0])

            self.assertEqual(group.source, "T4")
            self.assertEqual(group.layer, "ACT")
            self.assertEqual(group.category_label, "ACT")
            self.assertEqual(group.dataset, "original_size/ACT")
            self.assertEqual(group.split, "original_size")
            self.assertEqual(group.image_path, str(fixture.image_path))
            self.assertEqual(group.image_id, "11")
            self.assertEqual(group.annotation_relpath, "ACT/annotations/instances_all.json")
            self.assertEqual(group.annotation_sha256, fixture.annotation_sha256)
            self.assertEqual(len(group.items), 1)
            group_item = group.items[0]
            self.assertEqual(group_item.category_label, sample.category_label)
            self.assertEqual(group_item.label, sample.label)
            self.assertEqual(group_item.annotation_id, sample.annotation_id)
            self.assertEqual(group_item.annotation_relpath, sample.annotation_relpath)
            self.assertEqual(group_item.annotation_sha256, sample.annotation_sha256)

    def test_missing_root_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            for collector in (LABEL_EVAL.collect_t4_samples, GROUPED_EVAL.collect_t4_groups):
                with self.subTest(collector=collector.__module__):
                    with self.assertRaisesRegex(FileNotFoundError, "T4|root|exist"):
                        collector(missing, "all")

    def test_legacy_labelme_schema_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = T4CocoFixture(Path(tmp) / "T4" / "original_size")
            fixture.write(legacy_labelme=True)
            for collector in (LABEL_EVAL.collect_t4_samples, GROUPED_EVAL.collect_t4_groups):
                with self.subTest(collector=collector.__module__):
                    with self.assertRaisesRegex(ValueError, "COCO|schema|images|annotations"):
                        collector(fixture.root, "all")

    def test_missing_image_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = T4CocoFixture(Path(tmp) / "T4" / "original_size")
            fixture.write(write_image=False)
            for collector in (LABEL_EVAL.collect_t4_samples, GROUPED_EVAL.collect_t4_groups):
                with self.subTest(collector=collector.__module__):
                    with self.assertRaisesRegex(FileNotFoundError, "image|x.png"):
                        collector(fixture.root, "all")

    def test_annotation_with_unknown_image_id_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = T4CocoFixture(Path(tmp) / "T4" / "original_size")
            fixture.write(annotation_image_id=99)
            for collector in (LABEL_EVAL.collect_t4_samples, GROUPED_EVAL.collect_t4_groups):
                with self.subTest(collector=collector.__module__):
                    with self.assertRaisesRegex(ValueError, "image_id|99|reference"):
                        collector(fixture.root, "all")

    def test_invalid_original_label_is_rejected_consistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("", 123):
                fixture = T4CocoFixture(
                    Path(tmp) / f"case_{str(value) or 'empty'}" / "original_size"
                )
                fixture.write(original_label=value)
                for collector in (
                    LABEL_EVAL.collect_t4_samples,
                    GROUPED_EVAL.collect_t4_groups,
                ):
                    with self.subTest(value=value, collector=collector.__module__):
                        with self.assertRaisesRegex(ValueError, "original_label"):
                            collector(fixture.root, "all")

    def test_split_prefixed_o3_image_path_is_shared_by_all_collectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "O3_coco"
            dataset = root / "ACT_coco"
            annotation_path = dataset / "annotations" / "instances_val.json"
            image_path = dataset / "val" / "x.png"
            annotation_path.parent.mkdir(parents=True)
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
            annotation_path.write_text(
                json.dumps(
                    {
                        "categories": [{"id": 7, "name": "ACT"}],
                        "images": [
                            {
                                "id": 11,
                                "file_name": "val/x.png",
                                "width": 8,
                                "height": 6,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 23,
                                "image_id": 11,
                                "category_id": 7,
                                "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
                                "area": 12,
                                "bbox": [1, 1, 4, 3],
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            samples = LABEL_EVAL.collect_o3_samples(root, 1, {"val"})
            groups = GROUPED_EVAL.collect_o3_groups(root, 1, ["val"])
            pcs_groups = PCS_EVAL.collect_o3_category_images(root, 1, ["val"], 1)
            self.assertEqual(samples[0].image_path, str(image_path))
            self.assertEqual(groups[0].image_path, str(image_path))
            self.assertEqual(pcs_groups[0].image_path, str(image_path))


    def test_explicitly_requested_empty_source_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "t4|zero"):
            LABEL_EVAL.require_requested_sources({"t4"}, {"o3": 1, "t4": 0})
        with self.assertRaisesRegex(ValueError, "T4|zero"):
            GROUPED_EVAL._require_requested_sources({"T4"}, [])


if __name__ == "__main__":
    unittest.main()
