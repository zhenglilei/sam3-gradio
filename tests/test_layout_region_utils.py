import copy
import json
import multiprocessing
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import layout_region_utils as regions


def _save_region_in_subprocess(
    layout_masks,
    layout_regions,
    categories_path,
    session_id,
    layout_id,
    source_hash,
    started,
    result_queue,
):
    store = regions.LayoutRegionStore(
        layout_masks_root=layout_masks,
        layout_regions_root=layout_regions,
        categories_path=categories_path,
    )
    started.set()
    try:
        document, record = store.save_region(
            session_id=session_id,
            layout_id=layout_id,
            source_mask_hash=source_hash,
            expected_revision=0,
            lasso_polygon=[[0, 0], [39, 0], [39, 31], [0, 31]],
            class_label="via",
            name="child",
        )
        result_queue.put(("saved", document["regions_revision"], record["region_id"]))
    except Exception as exc:
        result_queue.put((type(exc).__name__, str(exc)))


class LayoutRegionUtilsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout_masks = self.root / "layout_masks"
        self.layout_regions = self.root / "layout_regions"
        self.categories_path = self.root / "layout_categories.json"
        self.session_id = "session1"
        self.layout_id = "layout1"
        self.source_mask = np.zeros((32, 40), dtype=np.uint8)
        self.source_mask[3:15, 4:18] = 1
        self.source_mask[7:10, 8:12] = 0
        self.source_mask[20:27, 26:35] = 1
        self.source_hash = regions.mask_pixel_sha256(self.source_mask)
        layout_dir = self.layout_masks / self.session_id / self.layout_id
        layout_dir.mkdir(parents=True)
        cv2.imwrite(str(layout_dir / "source_mask.png"), self.source_mask * 255)
        (layout_dir / "layout_meta.json").write_text(
            json.dumps({"source_mask_pixel_sha256": self.source_hash}), encoding="utf-8"
        )
        self._write_categories(["metal", "via"])
        self.store = regions.LayoutRegionStore(
            layout_masks_root=self.layout_masks,
            layout_regions_root=self.layout_regions,
            categories_path=self.categories_path,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_categories(self, values):
        self.categories_path.write_text(
            json.dumps({"schema_version": 1, "categories": values}), encoding="utf-8"
        )

    def _polygon(self):
        return [[0, 0], [39, 0], [39, 31], [0, 31], [0, 0]]

    def _save(self, revision=0, category="metal", name=""):
        return self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=revision,
            lasso_polygon=self._polygon(),
            class_label=category,
            name=name,
        )

    def test_duplicate_cleanup_and_invalid_lasso(self):
        points = [[1, 1], [1, 1], [10, 1], [10, 10], [1, 1]]
        normalized = regions.normalize_lasso_points(points, (20, 20))
        self.assertEqual(normalized.tolist(), [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0]])
        with self.assertRaises(regions.RegionValidationError):
            regions.normalize_lasso_points([[1, 1], [1, 1], [2, 2]], (20, 20))
        with self.assertRaises(regions.RegionValidationError):
            regions.normalize_lasso_points([[0, 0]] * 4097, (20, 20))

    def test_rasterize_clips_background_and_preserves_hole_and_components(self):
        region_mask, document = self.store.preview_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=self._polygon(),
        )
        self.assertTrue(np.array_equal(region_mask, self.source_mask.astype(bool)))
        self.assertFalse(region_mask[8, 9])
        updated, record = self._save(name="M1")
        decoded = regions.decode_binary_mask(record["mask_rle"], self.source_mask.shape)
        self.assertTrue(np.array_equal(decoded, self.source_mask.astype(bool)))
        self.assertEqual(record["area"], int(self.source_mask.sum()))
        self.assertEqual(record["component_count"], 2)
        self.assertEqual(updated["regions_revision"], 1)
        self.assertEqual(document["regions_revision"], 0)

    def test_soft_delete_preserves_record_and_ids_are_monotonic(self):
        polygons = [
            [[0, 0], [9, 0], [9, 18], [0, 18]],
            [[10, 0], [24, 0], [24, 18], [10, 18]],
            [[20, 18], [39, 18], [39, 31], [20, 31]],
        ]
        records = []
        document = None
        for revision, polygon in enumerate(polygons):
            document, record = self.store.save_region(
                session_id=self.session_id,
                layout_id=self.layout_id,
                source_mask_hash=self.source_hash,
                expected_revision=revision,
                lasso_polygon=polygon,
                class_label="metal",
                name=f"R{revision + 1}",
            )
            records.append(record)
        first, second, third = records
        document, deleted = self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=3,
            region_id=second["region_id"],
        )
        self.assertIsNotNone(deleted["deleted_at"])
        self.assertEqual([r["region_id"] for r in regions.active_regions(document)], [1, 3])
        decoded_deleted = regions.decode_binary_mask(deleted["mask_rle"], self.source_mask.shape)
        self.assertTrue(decoded_deleted.any())
        document, fourth = self._save(4, name="R4")
        self.assertEqual([first["region_id"], second["region_id"], third["region_id"], fourth["region_id"]], [1, 2, 3, 4])
        self.assertEqual(document["next_region_id"], 5)

    def test_category_changes_do_not_invalidate_history(self):
        document, record = self._save(0, category="metal")
        self.assertEqual(record["class_label"], "metal")
        self._write_categories(["via"])
        restored, _ = self.store.load_document(self.session_id, self.layout_id, self.source_hash)
        self.assertEqual(restored["regions"][0]["class_label"], "metal")
        with self.assertRaises(regions.RegionValidationError):
            self._save(1, category="metal")
        self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=1,
            region_id=record["region_id"],
        )
        updated, via = self._save(2, category="via")
        self.assertEqual(via["class_label"], "via")
        self.assertEqual(updated["regions_revision"], 3)

    def test_canonical_label_save_and_legacy_restore_are_compatible(self):
        updated, record = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=self._polygon(),
            label="M1_power",
        )
        self.assertEqual(record["label"], "M1_power")
        self.assertNotIn("class_label", record)
        self.assertNotIn("name", record)
        self.assertEqual(regions.region_label(record), "M1_power")
        self.assertEqual(updated["regions_revision"], 1)

        self._write_categories(["via"])
        restored, _ = self.store.load_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        self.assertEqual(restored["regions"][0]["label"], "M1_power")

        path = self.store.regions_path(self.session_id, self.layout_id)
        label_only = json.loads(path.read_text(encoding="utf-8"))
        label_only["regions"][0].pop("class_label", None)
        label_only["regions"][0].pop("name", None)
        path.write_text(json.dumps(label_only), encoding="utf-8")
        restored, _ = self.store.load_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        self.assertEqual(restored["regions"][0]["label"], "M1_power")
        self.assertNotIn("class_label", restored["regions"][0])
        self.assertNotIn("name", restored["regions"][0])

        legacy = regions.region_record_from_mask(
            2,
            "metal",
            "M2_signal",
            np.pad(np.ones((2, 2), dtype=bool), ((0, 30), (0, 38))),
        )
        self.assertNotIn("label", legacy)
        self.assertEqual(regions.region_label(legacy), "metal / M2_signal")
        self.assertEqual(
            regions.region_label({"class_label": "via", "name": ""}),
            "via",
        )

    def test_unicode_canonical_label_restore_enforces_80_character_limit(self):
        label_80 = "界" * 80
        document, record = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=self._polygon(),
            label=label_80,
        )
        self.assertEqual(record["label"], label_80)
        restored, _ = self.store.load_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        self.assertEqual(restored["regions"][0]["label"], label_80)

        corrupted = copy.deepcopy(document)
        corrupted["regions"][0]["label"] = "界" * 81
        self.store.regions_path(
            self.session_id,
            self.layout_id,
        ).write_text(
            json.dumps(corrupted, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            regions.RegionValidationError,
            "at most 80 characters",
        ):
            self.store.load_document(
                self.session_id,
                self.layout_id,
                self.source_hash,
            )

    def test_blank_label_uses_region_sequence_and_active_labels_are_unique(self):
        document, first = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=[[0, 0], [20, 0], [20, 18], [0, 18]],
            label="  ",
        )
        self.assertEqual(first["label"], "Label 1")
        self.assertEqual(regions.region_label(first), "Label 1")

        remaining = np.logical_and(
            self.source_mask.astype(bool),
            np.logical_not(
                regions.decode_binary_mask(
                    first["mask_rle"],
                    self.source_mask.shape,
                )
            ),
        )
        with self.assertRaisesRegex(
            regions.RegionValidationError,
            "active Region label already exists",
        ):
            regions.append_region(
                document,
                label="Label 1",
                region_mask=remaining,
            )

        document, second = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=1,
            lasso_polygon=self._polygon(),
            label="",
        )
        self.assertEqual(second["region_id"], 2)
        self.assertEqual(second["label"], "Label 2")
        self.assertEqual(
            [
                regions.region_label(record)
                for record in regions.active_regions(document)
            ],
            ["Label 1", "Label 2"],
        )

    def test_blank_label_skips_an_explicitly_occupied_sequence_name(self):
        document, explicit = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=[[0, 0], [20, 0], [20, 18], [0, 18]],
            label="Label 2",
        )
        self.assertEqual(explicit["region_id"], 1)
        self.assertEqual(explicit["label"], "Label 2")

        document, automatic = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=document["regions_revision"],
            lasso_polygon=self._polygon(),
            label="",
        )
        self.assertEqual(automatic["region_id"], 2)
        self.assertEqual(automatic["label"], "Label 3")
        self.assertEqual(
            [
                regions.region_label(record)
                for record in regions.active_regions(document)
            ],
            ["Label 2", "Label 3"],
        )

    def test_class_region_helpers_keep_regions_independent_and_ordered(self):
        shape = (8, 10)
        first_mask = np.zeros(shape, dtype=bool)
        first_mask[1:3, 1:4] = True
        third_mask = np.zeros(shape, dtype=bool)
        third_mask[5:7, 6:9] = True
        deleted_mask = np.zeros(shape, dtype=bool)
        deleted_mask[3:5, 4:6] = True
        other_mask = np.zeros(shape, dtype=bool)
        other_mask[0:2, 8:10] = True

        first = regions.region_record_from_mask(1, "legacy-metal", "first", first_mask)
        third = regions.region_record_from_mask(3, "legacy-metal", "third", third_mask)
        deleted = regions.region_record_from_mask(2, "legacy-metal", "deleted", deleted_mask)
        deleted["deleted_at"] = regions.utc_now_iso()
        other = regions.region_record_from_mask(4, "via", "other", other_mask)
        document = {"regions": [third, other, deleted, first]}

        selected = regions.active_regions_for_class(document, "legacy-metal")
        self.assertEqual([record["region_id"] for record in selected], [1, 3])

        decoded = regions.decode_region_masks(selected, shape)
        self.assertEqual([record["region_id"] for record, _ in decoded], [1, 3])
        self.assertTrue(np.array_equal(decoded[0][1], first_mask))
        self.assertTrue(np.array_equal(decoded[1][1], third_mask))
        self.assertEqual(regions.decode_region_masks([], shape), [])

        preview = regions.class_region_preview_mask(document, "legacy-metal", shape)
        self.assertTrue(np.array_equal(preview, np.logical_or(first_mask, third_mask)))
        self.assertFalse(np.logical_and(preview, deleted_mask).any())
        self.assertFalse(np.logical_and(preview, other_mask).any())

    def test_class_region_helpers_validate_label_and_allow_no_matches(self):
        document = {"regions": []}
        for invalid_label in (None, "", "   "):
            with self.subTest(class_label=invalid_label):
                with self.assertRaisesRegex(
                    regions.RegionValidationError,
                    "region category is required",
                ):
                    regions.active_regions_for_class(document, invalid_label)

        preview = regions.class_region_preview_mask(document, "historical", (5, 7))
        self.assertEqual(preview.dtype, np.bool_)
        self.assertEqual(preview.shape, (5, 7))
        self.assertFalse(preview.any())

    def test_label_helpers_filter_preview_and_build_uint16_index(self):
        shape = (8, 10)
        via_mask = np.zeros(shape, dtype=bool)
        via_mask[0:2, 0:3] = True
        deleted_mask = np.zeros(shape, dtype=bool)
        deleted_mask[2:4, 3:5] = True
        metal_mask = np.zeros(shape, dtype=bool)
        metal_mask[4:6, 5:8] = True
        legacy_mask = np.zeros(shape, dtype=bool)
        legacy_mask[6:8, 8:10] = True

        via = regions.region_record_from_mask(
            1, "via", "", via_mask, label="via"
        )
        deleted = regions.region_record_from_mask(
            2, "via", "", deleted_mask, label="via"
        )
        deleted["deleted_at"] = regions.utc_now_iso()
        metal = regions.region_record_from_mask(
            3, "metal", "", metal_mask, label="metal"
        )
        legacy = regions.region_record_from_mask(
            4, "metal", "M1_power", legacy_mask
        )
        document = {"regions": [metal, legacy, deleted, via]}

        selected = regions.active_regions_for_labels(
            document,
            ["metal", "via"],
        )
        self.assertEqual(
            [record["region_id"] for record in selected],
            [1, 3],
        )
        preview = regions.labels_region_preview_mask(
            document,
            ["metal", "via"],
            shape,
        )
        self.assertTrue(
            np.array_equal(preview, np.logical_or(via_mask, metal_mask))
        )
        self.assertFalse(np.logical_and(preview, deleted_mask).any())
        self.assertFalse(np.logical_and(preview, legacy_mask).any())

        legacy_selected = regions.active_regions_for_labels(
            document,
            ["metal / M1_power"],
        )
        self.assertEqual(
            [record["region_id"] for record in legacy_selected],
            [4],
        )
        long_legacy_label = regions.region_label(
            {"class_label": "metal", "name": "x" * 90}
        )
        self.assertEqual(
            regions.normalize_region_label_selection([long_legacy_label]),
            [long_legacy_label],
        )
        with self.assertRaisesRegex(
            regions.RegionValidationError,
            "duplicate region label",
        ):
            regions.active_regions_for_labels(document, ["metal", "metal"])
        with self.assertRaisesRegex(
            regions.RegionValidationError,
            "active Region label does not exist",
        ):
            regions.active_regions_for_labels(document, ["missing"])

        index_mask, labels = regions.region_label_index(document, shape)
        self.assertEqual(index_mask.dtype, np.uint16)
        self.assertEqual(
            labels,
            [
                {"index": 1, "region_id": 1, "label": "via", "area": 6},
                {"index": 2, "region_id": 3, "label": "metal", "area": 6},
                {
                    "index": 3,
                    "region_id": 4,
                    "label": "metal / M1_power",
                    "area": 4,
                },
            ],
        )
        self.assertTrue(np.all(index_mask[via_mask] == 1))
        self.assertTrue(np.all(index_mask[metal_mask] == 2))
        self.assertTrue(np.all(index_mask[legacy_mask] == 3))
        self.assertTrue(np.all(index_mask[deleted_mask] == 0))

    def test_stale_revision_and_hash_mismatch_are_rejected(self):
        self._save(0)
        with self.assertRaises(regions.StaleRegionsRevisionError):
            self._save(0)
        with self.assertRaises(regions.RegionValidationError):
            self.store.load_document(self.session_id, self.layout_id, "wrong-hash")

    def test_corrupt_rle_rejects_entire_restore(self):
        document, _ = self._save(0)
        path = self.store.regions_path(self.session_id, self.layout_id)
        corrupt = copy.deepcopy(document)
        corrupt["regions"][0]["mask_rle"]["counts"] = "not-a-valid-rle"
        path.write_text(json.dumps(corrupt), encoding="utf-8")
        with self.assertRaises(regions.RegionValidationError):
            self.store.load_document(self.session_id, self.layout_id, self.source_hash)

    def test_restore_rejects_oversized_canonical_label(self):
        document, _ = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=self._polygon(),
            label="valid",
        )
        corrupt = copy.deepcopy(document)
        corrupt["regions"][0]["label"] = "x" * (
            regions.MAX_REGION_LABEL_LENGTH + 1
        )
        path = self.store.regions_path(self.session_id, self.layout_id)
        path.write_text(json.dumps(corrupt), encoding="utf-8")
        with self.assertRaises(regions.RegionValidationError):
            self.store.load_document(self.session_id, self.layout_id, self.source_hash)


    def test_restore_rejects_overlapping_active_regions(self):
        document, first = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=[[0, 0], [12, 0], [12, 18], [0, 18]],
            class_label="metal",
            name="first",
        )
        first_mask = regions.decode_binary_mask(
            first["mask_rle"],
            self.source_mask.shape,
        )
        overlapping = regions.region_record_from_mask(
            2,
            "via",
            "overlap",
            first_mask,
        )
        corrupted = copy.deepcopy(document)
        corrupted["regions"].append(overlapping)
        corrupted["regions_revision"] = 2
        corrupted["next_region_id"] = 3
        path = self.store.regions_path(self.session_id, self.layout_id)
        path.write_text(json.dumps(corrupted), encoding="utf-8")

        with self.assertRaisesRegex(
            regions.RegionValidationError,
            "R2 overlaps another active Region",
        ):
            self.store.load_document(
                self.session_id,
                self.layout_id,
                self.source_hash,
            )

        corrupted["regions"][0]["deleted_at"] = regions.utc_now_iso()
        path.write_text(json.dumps(corrupted), encoding="utf-8")
        restored, _ = self.store.load_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        self.assertEqual(len(regions.active_regions(restored)), 1)

    def test_atomic_replace_failure_preserves_existing_document(self):
        target = self.root / "regions.json"
        original = {"revision": 1}
        target.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(OSError):
            regions.write_json_atomic(
                target,
                {"revision": 2},
                replace=mock.Mock(side_effect=OSError("replace failed")),
            )
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), original)
        self.assertEqual(list(target.parent.glob(".regions.json.*.tmp")), [])

    def test_cross_process_write_lock_prevents_lost_update(self):
        context = multiprocessing.get_context("spawn")
        started = context.Event()
        result_queue = context.Queue()
        process = context.Process(
            target=_save_region_in_subprocess,
            args=(
                str(self.layout_masks),
                str(self.layout_regions),
                str(self.categories_path),
                self.session_id,
                self.layout_id,
                self.source_hash,
                started,
                result_queue,
            ),
        )

        try:
            with self.store._layout_write_lock(self.session_id, self.layout_id):
                process.start()
                self.assertTrue(started.wait(timeout=5))
                with self.assertRaises(queue.Empty):
                    result_queue.get(timeout=1)

                document, source_mask = self.store.load_document(
                    self.session_id,
                    self.layout_id,
                    self.source_hash,
                )
                region_mask = regions.rasterize_uncovered_region_mask(
                    source_mask,
                    self._polygon(),
                    document,
                )
                updated, _ = regions.append_region(
                    document,
                    class_label="metal",
                    name="parent",
                    region_mask=region_mask,
                    allowed_categories=["metal", "via"],
                )
                regions.write_json_atomic(
                    self.store.regions_path(self.session_id, self.layout_id),
                    updated,
                )
        finally:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        outcome = result_queue.get(timeout=2)
        self.assertEqual(outcome[0], "StaleRegionsRevisionError")
        restored, _ = self.store.load_document(
            self.session_id,
            self.layout_id,
            self.source_hash,
        )
        self.assertEqual(restored["regions_revision"], 1)
        self.assertEqual(
            [record["name"] for record in regions.active_regions(restored)],
            ["parent"],
        )

    def test_saved_overlay_excludes_soft_deleted_region(self):
        document, _ = self._save(0)
        active_overlay = np.asarray(
            regions.render_saved_region_overlay(document["regions"], self.source_mask.shape)
        )
        self.assertGreater(int(active_overlay[..., 3].sum()), 0)
        document, _ = self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=1,
            region_id=1,
        )
        deleted_overlay = np.asarray(
            regions.render_saved_region_overlay(document["regions"], self.source_mask.shape)
        )
        self.assertEqual(int(deleted_overlay[..., 3].sum()), 0)

    def test_saved_overlay_uses_ascii_fallback_for_unicode_label(self):
        mask = np.zeros_like(self.source_mask, dtype=bool)
        mask[3:8, 4:10] = True
        record = {
            "region_id": 7,
            "label": "金属层",
            "mask_rle": regions.encode_binary_mask(mask),
            "deleted_at": None,
        }
        with mock.patch.object(regions, "_draw_region_label") as draw_label:
            regions.render_saved_region_overlay([record], mask.shape)

        self.assertEqual(draw_label.call_args.args[2], "L7")


    def test_new_regions_only_use_pixels_not_covered_by_active_regions(self):
        left_polygon = [[0, 0], [22, 0], [22, 18], [0, 18]]
        document, first = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=0,
            lasso_polygon=left_polygon,
            class_label="metal",
            name="left",
        )
        first_mask = regions.decode_binary_mask(first["mask_rle"], self.source_mask.shape)

        second_mask, _ = self.store.preview_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=1,
            lasso_polygon=self._polygon(),
        )
        self.assertFalse(np.logical_and(first_mask, second_mask).any())
        self.assertTrue(np.array_equal(np.logical_or(first_mask, second_mask), self.source_mask.astype(bool)))

        document, second = self.store.save_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=1,
            lasso_polygon=self._polygon(),
            class_label="via",
            name="remaining",
        )
        decoded_second = regions.decode_binary_mask(second["mask_rle"], self.source_mask.shape)
        self.assertTrue(np.array_equal(decoded_second, second_mask))
        with self.assertRaises(regions.RegionValidationError):
            self.store.preview_region(
                session_id=self.session_id,
                layout_id=self.layout_id,
                source_mask_hash=self.source_hash,
                expected_revision=2,
                lasso_polygon=self._polygon(),
            )

        document, _ = self.store.delete_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=2,
            region_id=first["region_id"],
        )
        reclaimed_mask, _ = self.store.preview_region(
            session_id=self.session_id,
            layout_id=self.layout_id,
            source_mask_hash=self.source_hash,
            expected_revision=3,
            lasso_polygon=left_polygon,
        )
        self.assertTrue(np.array_equal(reclaimed_mask, first_mask))


if __name__ == "__main__":
    unittest.main()
