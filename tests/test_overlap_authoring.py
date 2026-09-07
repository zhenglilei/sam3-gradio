from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from layout_transform_utils import build_layout_affine_matrix

from sam3_demo.overlap_authoring import (
    MAX_COCO_DIMENSION,
    MAX_COCO_JSON_BYTES,
    OverlapAuthoringError,
    OverlapAuthoringService,
)
from sam3_demo.overlap_store import OverlapLayoutStore


def _base(width=12, height=10):
    return Image.new("RGB", (width, height), (20, 30, 40))


def _mask(width=5, height=4):
    array = np.zeros((height, width), dtype=np.uint8)
    array[1:3, 1:4] = 255
    return Image.fromarray(array, mode="L")


def _uncompressed_counts(mask: np.ndarray) -> list[int]:
    counts: list[int] = []
    previous = 0
    run = 0
    for value in np.asarray(mask, dtype=np.uint8).ravel(order="F"):
        pixel = int(value)
        if pixel == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = pixel
    counts.append(run)
    return counts


class OverlapAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = OverlapLayoutStore(
            Path(self.tempdir.name) / "authoring.sqlite3"
        )
        self.service = OverlapAuthoringService(self.store)

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def _draft_with_layers(self):
        draft = self.service.new_draft()
        draft = self.service.set_name(draft, "sample")
        draft = self.service.set_base_image(draft, _base())
        draft = self.service.add_layer(draft, _mask(), "ACT", "active")
        draft = self.service.add_layer(draft, _mask(), "GE1", "gate")
        return draft

    def _write_coco(self, payload, name="labels.json"):
        path = Path(self.tempdir.name) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _coco_payload(self):
        return {
            "images": [
                {"id": 11, "file_name": "first.png", "width": 7, "height": 6},
                {"id": "11", "file_name": "second.png", "width": 8, "height": 5},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 11,
                    "segmentation": [
                        [1, 1, 5, 1, 5, 4, 1, 4],
                        [4, 3, 6, 3, 6, 5, 4, 5],
                    ],
                },
                {
                    "id": 2,
                    "image_id": "11",
                    "segmentation": [[2, 1, 7, 1, 7, 4, 2, 4]],
                },
            ],
        }

    def test_coco_choices_union_and_add_layer_rereads_json(self):
        path = self._write_coco(self._coco_payload())
        choices = self.service.coco_image_choices(path)
        self.assertEqual(
            choices,
            [
                ("first.png (7×6)", "int:11"),
                ("second.png (8×5)", "str:11"),
            ],
        )
        selected_choices, selected_id, preview = self.service.inspect_coco_json(path)
        self.assertEqual(selected_choices, choices)
        self.assertEqual(selected_id, "int:11")
        expected_rle = coco_mask.merge(
            coco_mask.frPyObjects(
                self._coco_payload()["annotations"][0]["segmentation"],
                6,
                7,
            )
        )
        expected = coco_mask.decode(expected_rle).astype(bool)
        self.assertTrue(
            np.array_equal(np.asarray(preview) > 0, expected)
        )

        second_preview = self.service.coco_image_mask(path, "str:11")
        self.assertEqual(second_preview.size, (8, 5))
        self.assertTrue(np.asarray(second_preview).any())
        self.assertTrue(
            np.array_equal(
                np.asarray(second_preview),
                np.asarray(self.service.coco_image_mask(path, "11")),
            )
        )

        draft = self.service.set_base_image(self.service.new_draft(), _base())
        added = self.service.add_layer_from_coco(
            draft, path, "int:11", "ACT", "from JSON"
        )
        self.assertEqual(added["layers"][0]["name"], "from JSON")
        with Image.open(BytesIO(added["layers"][0]["mask_png"])) as mask_image:
            self.assertTrue(
                np.array_equal(np.asarray(mask_image) > 0, expected)
            )

        changed = self._coco_payload()
        changed["annotations"][0]["segmentation"] = [
            [0, 0, 2, 0, 2, 2, 0, 2]
        ]
        self._write_coco(changed)
        reread = self.service.add_layer_from_coco(
            draft, path, "int:11", "GE1"
        )
        with Image.open(BytesIO(reread["layers"][0]["mask_png"])) as mask_image:
            self.assertFalse(
                np.array_equal(np.asarray(mask_image) > 0, expected)
            )

    def test_coco_compressed_and_uncompressed_rle(self):
        expected = np.zeros((6, 8), dtype=np.uint8)
        expected[1:5, 2:6] = 1
        expected[2:4, 3:5] = 0
        encoded = coco_mask.encode(np.asfortranarray(expected))
        segmentations = [
            {
                "size": [6, 8],
                "counts": encoded["counts"].decode("ascii"),
            },
            {
                "size": [6, 8],
                "counts": _uncompressed_counts(expected),
            },
        ]
        for index, segmentation in enumerate(segmentations):
            with self.subTest(rle_type=type(segmentation["counts"]).__name__):
                path = self._write_coco(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": f"rle-{index}.png",
                                "width": 8,
                                "height": 6,
                            }
                        ],
                        "annotations": [
                            {"image_id": 1, "segmentation": segmentation}
                        ],
                    },
                    f"rle-{index}.json",
                )
                actual = np.asarray(self.service.coco_image_mask(path)) > 0
                self.assertTrue(np.array_equal(actual, expected.astype(bool)))

    def test_coco_rejects_invalid_or_oversized_documents(self):
        valid_image = {
            "id": 1,
            "file_name": "one.png",
            "width": 8,
            "height": 6,
        }
        valid_annotation = {
            "image_id": 1,
            "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
        }

        cases = [
            (
                {"images": [], "annotations": []},
                "images",
                "choices",
            ),
            (
                {"images": [valid_image], "annotations": []},
                "annotation",
                "choices",
            ),
            (
                {
                    "images": [valid_image, dict(valid_image)],
                    "annotations": [valid_annotation],
                },
                "重复",
                "choices",
            ),
            (
                {
                    "images": [{**valid_image, "width": 0}],
                    "annotations": [valid_annotation],
                },
                "width",
                "choices",
            ),
            (
                {
                    "images": [
                        {
                            **valid_image,
                            "width": MAX_COCO_DIMENSION + 1,
                        }
                    ],
                    "annotations": [valid_annotation],
                },
                "上限",
                "choices",
            ),
        ]
        for index, (payload, pattern, method) in enumerate(cases):
            with self.subTest(case=index):
                path = self._write_coco(payload, f"invalid-{index}.json")
                with self.assertRaisesRegex(OverlapAuthoringError, pattern):
                    getattr(self.service, f"coco_image_{method}")(path)

        bad_segmentation = dict(valid_annotation)
        bad_segmentation["segmentation"] = [[0, 0, 1]]
        path = self._write_coco(
            {
                "images": [valid_image],
                "annotations": [bad_segmentation],
            },
            "bad-segmentation.json",
        )
        with self.assertRaisesRegex(OverlapAuthoringError, "polygon"):
            self.service.coco_image_mask(path)

        empty_segmentation = dict(valid_annotation)
        empty_segmentation["segmentation"] = [[0, 0, 0, 0, 0, 0]]
        path = self._write_coco(
            {
                "images": [valid_image],
                "annotations": [empty_segmentation],
            },
            "empty-mask.json",
        )
        with self.assertRaisesRegex(OverlapAuthoringError, "mask 为空"):
            self.service.coco_image_mask(path)

        path = self._write_coco(
            {
                "images": [valid_image],
                "annotations": [empty_segmentation, valid_annotation],
            },
            "mixed-empty-mask.json",
        )
        self.assertTrue(np.asarray(self.service.coco_image_mask(path)).any())


        path = Path(self.tempdir.name) / "broken.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(OverlapAuthoringError, "解析"):
            self.service.coco_image_choices(path)

        path = self._write_coco(
            {
                "images": [valid_image],
                "annotations": [valid_annotation],
            },
            "oversized.json",
        )
        with patch(
            "sam3_demo.overlap_authoring.MAX_COCO_JSON_BYTES",
            len(path.read_bytes()) - 1,
        ):
            with self.assertRaisesRegex(OverlapAuthoringError, "大小上限"):
                self.service.coco_image_choices(path)

        no_annotation_for_selected = self._write_coco(
            {
                "images": [
                    valid_image,
                    {
                        "id": 2,
                        "file_name": "two.png",
                        "width": 8,
                        "height": 6,
                    },
                ],
                "annotations": [valid_annotation],
            },
            "missing-selected-annotation.json",
        )
        with self.assertRaisesRegex(OverlapAuthoringError, "没有 annotation"):
            self.service.coco_image_mask(no_annotation_for_selected, "int:2")

        with self.assertRaisesRegex(OverlapAuthoringError, "不存在"):
            self.service.coco_image_mask(no_annotation_for_selected, "int:99")

    def test_draft_methods_are_immutable_and_transform_is_per_layer(self):
        original = self.service.new_draft()
        original_copy = copy.deepcopy(original)
        named = self.service.set_name(original, "  board  ")
        self.assertEqual(original, original_copy)
        self.assertEqual(named["name"], "board")
        based = self.service.set_base_image(named, _base())
        layered = self.service.add_layer(based, _mask(), "ACT")
        self.assertEqual(based["layers"], [])
        layer = layered["layers"][0]
        self.assertEqual(layer["transform"]["transform_version"], 2)
        self.assertEqual(layer["transform"]["center_x"], 6.0)
        self.assertEqual(layer["transform"]["center_y"], 5.0)
        self.assertEqual(layer["transform"]["scale"], 1.0)
        self.assertEqual(layered["active_layer_id"], layer["id"])

    def test_full_canvas_mask_keeps_its_original_pixel_position(self):
        draft = self.service.set_base_image(self.service.new_draft(), _base())
        full_mask = Image.new("L", (12, 10), 0)
        array = np.asarray(full_mask).copy()
        array[1:3, 1:4] = 255
        draft = self.service.add_layer(
            draft, Image.fromarray(array, mode="L"), "ACT"
        )
        transform = draft["layers"][0]["transform"]
        self.assertEqual(transform["center_x"], transform["pivot_x"])
        self.assertEqual(transform["center_y"], transform["pivot_y"])
        self.assertEqual(
            build_layout_affine_matrix(transform).tolist(),
            [[1.0, -0.0, 0.0], [0.0, 1.0, 0.0]],
        )

    def test_reorder_remove_and_active_fallback(self):
        draft = self._draft_with_layers()
        first, second = [layer["id"] for layer in draft["layers"]]
        reordered = self.service.reorder_layers(draft, [second, first])
        self.assertEqual(
            [layer["id"] for layer in reordered["layers"]],
            [second, first],
        )
        self.assertEqual(
            [layer["ordinal"] for layer in reordered["layers"]], [0, 1]
        )
        with self.assertRaises(OverlapAuthoringError):
            self.service.reorder_layers(draft, [first, first])
        removed = self.service.remove_layer(reordered, second)
        self.assertEqual(removed["active_layer_id"], first)

    def test_editor_intent_updates_one_layer_and_order(self):
        draft = self._draft_with_layers()
        payload = self.service.editor_payload(draft)
        first, second = [layer["id"] for layer in draft["layers"]]
        intent = payload["group_intent"]
        intent["transform_set_revision"] += 1
        intent["changed_group_ids"] = [first]
        intent["ordered_group_ids"] = [second, first]
        first_item = next(
            item for item in intent["transforms"] if item["group_id"] == first
        )
        first_item["transform"].update(
            center_x=8.5,
            center_y=7.0,
            scale=1.4,
            rotation_deg=12.0,
            preview_alpha=0.5,
            pivot_x=999,
            matrix_2x3=[[999, 0, 0], [0, 999, 0]],
        )
        updated = self.service.apply_editor_intent(draft, payload)
        by_id = {layer["id"]: layer for layer in updated["layers"]}
        self.assertEqual([layer["id"] for layer in updated["layers"]], [second, first])
        self.assertEqual(by_id[first]["transform"]["center_x"], 8.5)
        self.assertEqual(by_id[first]["transform"]["scale"], 1.4)
        self.assertNotEqual(by_id[first]["transform"]["pivot_x"], 999)
        self.assertEqual(
            by_id[second]["transform"], draft["layers"][1]["transform"]
        )
        self.assertEqual(draft["layers"][0]["transform"]["center_x"], 6.0)

    def test_editor_intent_rejects_stale_and_forged_collections(self):
        draft = self._draft_with_layers()
        payload = self.service.editor_payload(draft)
        with self.assertRaisesRegex(OverlapAuthoringError, "revision"):
            self.service.apply_editor_intent(draft, payload)
        payload = self.service.editor_payload(draft)
        payload["group_intent"]["transform_set_revision"] += 1
        payload["group_intent"]["ordered_group_ids"] = ["forged"]
        with self.assertRaisesRegex(OverlapAuthoringError, "顺序"):
            self.service.apply_editor_intent(draft, payload)
        payload = self.service.editor_payload(draft)
        payload["group_intent"]["transform_set_revision"] += 1
        payload["group_intent"]["transforms"].pop()
        with self.assertRaisesRegex(OverlapAuthoringError, "集合"):
            self.service.apply_editor_intent(draft, payload)

    def test_editor_intent_accepts_coalesced_revision_gap(self):
        draft = self._draft_with_layers()
        payload = self.service.editor_payload(draft)
        layer_id = draft["layers"][0]["id"]
        intent = payload["group_intent"]
        intent["transform_set_revision"] += 2
        intent["changed_group_ids"] = [layer_id]
        item = next(
            value
            for value in intent["transforms"]
            if value["group_id"] == layer_id
        )
        item["transform"]["center_x"] = 9.0

        updated = self.service.apply_editor_intent(draft, payload)

        self.assertEqual(
            updated["draft_revision"], intent["transform_set_revision"]
        )
        self.assertEqual(updated["layers"][0]["transform"]["center_x"], 9.0)

    def test_base_resize_recenters_layers(self):
        draft = self._draft_with_layers()
        payload = self.service.editor_payload(draft)
        layer_id = draft["layers"][0]["id"]
        payload["group_intent"]["transform_set_revision"] += 1
        payload["group_intent"]["changed_group_ids"] = [layer_id]
        item = next(
            value
            for value in payload["group_intent"]["transforms"]
            if value["group_id"] == layer_id
        )
        item["transform"]["scale"] = 1.75
        item["transform"]["rotation_deg"] = 8.0
        draft = self.service.apply_editor_intent(draft, payload)
        resized = self.service.set_base_image(draft, _base(20, 16))
        layer = next(x for x in resized["layers"] if x["id"] == layer_id)
        self.assertEqual(layer["transform"]["center_x"], 10.0)
        self.assertEqual(layer["transform"]["center_y"], 8.0)
        self.assertEqual(layer["transform"]["scale"], 1.75)
        self.assertEqual(layer["transform"]["rotation_deg"], 8.0)

    def test_create_update_load_delete(self):
        draft = self._draft_with_layers()
        saved = self.service.save(draft)
        self.assertIsNotNone(saved["layout_id"])
        self.assertEqual(saved["store_revision"], 1)
        loaded = self.service.load_layout(saved["layout_id"])
        self.assertEqual(loaded["name"], "sample")
        changed = self.service.set_name(loaded, "updated")
        updated = self.service.save(changed)
        self.assertEqual(updated["store_revision"], 2)
        self.assertEqual(self.store.get_layout(updated["layout_id"])["name"], "updated")
        empty = self.service.delete(updated)
        self.assertIsNone(empty["layout_id"])
        self.assertEqual(self.store.list_layouts(), [])

    def test_rejects_missing_inputs_and_too_few_layers(self):
        draft = self.service.new_draft()
        with self.assertRaisesRegex(OverlapAuthoringError, "整体版图"):
            self.service.add_layer(draft, _mask(), "ACT")
        draft = self.service.set_base_image(draft, _base())
        with self.assertRaisesRegex(OverlapAuthoringError, "前景"):
            self.service.add_layer(
                draft, Image.new("L", (4, 4), 0), "ACT"
            )
        draft = self.service.set_name(draft, "one")
        draft = self.service.add_layer(draft, _mask(), "ACT")
        with self.assertRaisesRegex(OverlapAuthoringError, "至少需要两层"):
            self.service.save(draft)


if __name__ == "__main__":
    unittest.main()
