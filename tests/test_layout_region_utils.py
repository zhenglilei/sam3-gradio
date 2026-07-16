import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import layout_region_utils as regions


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
