import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3_demo.annotated_stitch_io import export_bundle, import_tiles, make_tile


def _save_image(path: Path, size=(8, 6)):
    Image.new("RGB", size, (20, 40, 60)).save(path)


def _write_coco(path, image_name, width, height, annotations, categories=None):
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 11,
                        "file_name": image_name,
                        "width": width,
                        "height": height,
                    }
                ],
                "annotations": annotations,
                "categories": (
                    categories
                    if categories is not None
                    else [{"id": 1, "name": "object"}]
                ),
            }
        ),
        encoding="utf-8",
    )


class AnnotatedStitchIOTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_make_tile_detaches_rgb_and_accepts_zero_instances(self):
        source = Image.new("L", (4, 3), 127)
        tile = make_tile(source, "tile.png", [])

        self.assertEqual(tile["image"].mode, "RGB")
        self.assertEqual(tile["instances"], [])
        self.assertRegex(tile["tile_id"], r"[0-9a-f]{32}")
        source.putpixel((0, 0), 0)
        self.assertEqual(tile["image"].getpixel((0, 0)), (127, 127, 127))

    def test_coco_hole_rle_is_decoded_pixel_for_pixel(self):
        image_path = self.root / "hole.png"
        json_path = self.root / "annotations.json"
        _save_image(image_path, (7, 6))
        expected = np.zeros((6, 7), dtype=np.uint8)
        expected[1:5, 1:6] = 1
        expected[2:4, 3:5] = 0
        encoded = coco_mask.encode(np.asfortranarray(expected))
        encoded["counts"] = encoded["counts"].decode("ascii")
        _write_coco(
            json_path,
            image_path.name,
            7,
            6,
            [
                {
                    "id": 23,
                    "image_id": 11,
                    "category_id": 1,
                    "segmentation": encoded,
                }
            ],
        )

        tiles = import_tiles([json_path, image_path])

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0]["instances"][0]["id"], 23)
        self.assertEqual(tiles[0]["instances"][0]["mask"].dtype, np.bool_)
        self.assertTrue(
            np.array_equal(
                tiles[0]["instances"][0]["mask"], expected.astype(bool)
            )
        )

    def test_coco_polygon_uses_pycocotools_rasterization(self):
        image_path = self.root / "polygon.png"
        json_path = self.root / "annotations.json"
        _save_image(image_path, (8, 7))
        polygon = [0.2, 0.2, 4.7, 0.2, 4.7, 3.7, 0.2, 3.7]
        expected = coco_mask.decode(
            coco_mask.merge(coco_mask.frPyObjects([polygon], 7, 8))
        ).astype(bool)
        _write_coco(
            json_path,
            image_path.name,
            8,
            7,
            [
                {
                    "id": 5,
                    "image_id": 11,
                    "category_id": 1,
                    "segmentation": [polygon],
                }
            ],
        )

        actual = import_tiles([image_path, json_path])[0]["instances"][0]["mask"]

        self.assertTrue(np.array_equal(actual, expected))

    def test_labelme_groups_by_group_and_label_but_unlabeled_shapes_stay_separate(
        self,
    ):
        image_path = self.root / "labelme.png"
        json_path = self.root / "labelme.json"
        _save_image(image_path, (10, 8))
        json_path.write_text(
            json.dumps(
                {
                    "imagePath": image_path.name,
                    "imageWidth": 10,
                    "imageHeight": 8,
                    "shapes": [
                        {
                            "label": "A",
                            "group_id": 7,
                            "shape_type": "polygon",
                            "points": [[1, 1], [3, 1], [3, 3], [1, 3]],
                        },
                        {
                            "label": "A",
                            "group_id": 7,
                            "shape_type": "rectangle",
                            "points": [[5, 1], [7, 3]],
                        },
                        {
                            "label": "A",
                            "shape_type": "polygon",
                            "points": [[1, 5], [3, 5], [3, 7], [1, 7]],
                        },
                        {
                            "label": "A",
                            "shape_type": "polygon",
                            "points": [[5, 5], [7, 5], [7, 7], [5, 7]],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        instances = import_tiles([json_path, image_path])[0]["instances"]

        self.assertEqual(len(instances), 3)
        self.assertEqual(instances[0]["provenance"]["shape_indices"], [0, 1])
        self.assertEqual(instances[1]["provenance"]["shape_indices"], [2])
        self.assertEqual(instances[2]["provenance"]["shape_indices"], [3])
        self.assertTrue(instances[0]["mask"][2, 2])
        self.assertTrue(instances[0]["mask"][2, 6])
        self.assertFalse(instances[1]["mask"][2, 2])

    def test_labelme_rejects_unsupported_shape_type(self):
        image_path = self.root / "shape.png"
        json_path = self.root / "shape.json"
        _save_image(image_path)
        json_path.write_text(
            json.dumps(
                {
                    "imagePath": image_path.name,
                    "shapes": [
                        {"label": "x", "shape_type": "circle", "points": []}
                    ],
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError, "unsupported LabelMe shape_type"
        ):
            import_tiles([json_path, image_path])

    def test_score_is_not_fabricated_when_missing(self):
        image_path = self.root / "score.png"
        json_path = self.root / "score.json"
        _save_image(image_path)
        _write_coco(
            json_path,
            image_path.name,
            8,
            6,
            [
                {
                    "id": 2,
                    "image_id": 11,
                    "category_id": 1,
                    "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
                }
            ],
        )

        instance = import_tiles([image_path, json_path])[0]["instances"][0]

        self.assertNotIn("score", instance)

    def test_dimension_mismatch_is_rejected(self):
        image_path = self.root / "wrong.png"
        json_path = self.root / "wrong.json"
        _save_image(image_path, (8, 6))
        _write_coco(json_path, image_path.name, 7, 6, [])

        with self.assertRaisesRegex(ValueError, "expected"):
            import_tiles([json_path, image_path])
        with self.assertRaisesRegex(ValueError, "mask shape"):
            make_tile(
                Image.new("RGB", (5, 4)),
                "bad.png",
                [
                    {
                        "id": 1,
                        "category_name": "x",
                        "mask": np.zeros((3, 5)),
                    }
                ],
            )

    def test_missing_json_image_is_not_treated_as_image_only(self):
        image_path = self.root / "present.png"
        json_path = self.root / "missing.json"
        _save_image(image_path)
        _write_coco(json_path, "missing.png", 8, 6, [])

        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            import_tiles([json_path, image_path])

    def test_zip_path_traversal_is_rejected(self):
        bundle = self.root / "unsafe.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("../evil.txt", b"x")

        with self.assertRaisesRegex(ValueError, "unsafe ZIP member path"):
            import_tiles([bundle])

    def test_duplicate_image_names_are_rejected(self):
        first = self.root / "one" / "same.png"
        second = self.root / "two" / "same.png"
        first.parent.mkdir()
        second.parent.mkdir()
        _save_image(first)
        _save_image(second)

        with self.assertRaisesRegex(ValueError, "image name is ambiguous"):
            import_tiles([first, second])

    def test_export_bundle_preserves_tile_manifest_and_recomputes_geometry(self):
        image = Image.new("RGB", (7, 6), (1, 2, 3))
        first = np.zeros((6, 7), dtype=bool)
        first[1:4, 1:4] = True
        second = np.zeros((6, 7), dtype=bool)
        second[2:5, 2:6] = True
        empty = np.zeros((6, 7), dtype=bool)
        instances = [
            {
                "id": "first",
                "category_name": "zeta",
                "mask": first,
                "provenance": {"source": "a"},
            },
            {
                "id": 99,
                "category_name": "alpha",
                "mask": second,
                "score": 0.75,
                "provenance": {"source": "b"},
            },
            {"id": "empty", "category_name": "alpha", "mask": empty},
        ]
        manifest = {
            "kind": "tile",
            "tile_id": "tile-fixed",
            "source_name": "source-tile.png",
            "provenance": {"source": "unit"},
        }

        bundle = Path(export_bundle(image, instances, self.root, manifest))

        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            coco = json.loads(archive.read("coco_predictions.json"))
            saved_manifest = json.loads(archive.read("manifest.json"))
        self.assertTrue(
            {"mosaic.png", "coco_predictions.json", "manifest.json"} <= names
        )
        self.assertIn("masks/instance_000001.png", names)
        self.assertIn("masks/instance_000002.png", names)
        self.assertNotIn("masks/instance_000003.png", names)
        self.assertEqual(
            [item["name"] for item in coco["categories"]], ["alpha", "zeta"]
        )
        self.assertEqual(
            [item["id"] for item in coco["annotations"]], [1, 2]
        )
        self.assertEqual(coco["annotations"][0]["category_id"], 2)
        self.assertEqual(coco["annotations"][1]["category_id"], 1)
        self.assertEqual(coco["annotations"][0]["area"], int(first.sum()))
        self.assertEqual(
            coco["annotations"][0]["bbox"], [1.0, 1.0, 3.0, 3.0]
        )
        self.assertNotIn("score", coco["annotations"][0])
        self.assertEqual(coco["annotations"][1]["score"], 0.75)
        self.assertEqual(saved_manifest["kind"], "tile")
        self.assertEqual(
            saved_manifest["skipped_empty_instances"][0]["id"], "empty"
        )

        tile = import_tiles([bundle])[0]
        self.assertEqual(tile["name"], "source-tile.png")
        self.assertEqual(tile["tile_id"], "tile-fixed")
        self.assertEqual(tile["provenance"], {"source": "unit"})
        self.assertEqual(
            [item["id"] for item in tile["instances"]], ["first", 99]
        )
        self.assertTrue(np.array_equal(tile["instances"][0]["mask"], first))
        self.assertTrue(np.array_equal(tile["instances"][1]["mask"], second))


if __name__ == "__main__":
    unittest.main()
