import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam3_demo.layout import region_callbacks


class _Regions:
    class RegionValidationError(ValueError):
        pass

    @staticmethod
    def active_regions(document):
        return [record for record in document.get("regions", []) if not record.get("deleted_at")]

    @staticmethod
    def region_label_index(document, shape):
        del document
        return np.zeros(shape, dtype=np.uint16), []

    @staticmethod
    def utc_now_iso():
        return "2026-08-21T00:00:00Z"


class _Store:
    def __init__(self, document, source_mask):
        self.document = document
        self.source_mask = source_mask

    def load_document(self, session_id, layout_id, source_mask_hash):
        self.loaded = (session_id, layout_id, source_mask_hash)
        return self.document, self.source_mask


class _Downloads:
    def __init__(self):
        self.staging_dir = None

    def publish_zip(
        self,
        public_dir,
        category,
        staging_dir,
        filename,
        *,
        session_id=None,
    ):
        del public_dir, category, filename
        self.session_id = session_id
        self.staging_dir = Path(staging_dir)
        return self.staging_dir.parent / "published.zip"


class RegionSessionArtifactTest(unittest.TestCase):
    def test_fallback_state_keeps_layout_session_id(self):
        def new_state():
            return {
                "session_id": None,
                "layout_id": None,
                "source_mask_hash": None,
                "regions_revision": 0,
                "next_region_id": 1,
                "selected_region_id": None,
            }

        state = region_callbacks._new_layout_region_state_for_layout_impl(
            {"_new_layout_region_state": new_state},
            {"session_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        )
        self.assertEqual(state["session_id"], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.assertIsNone(state["layout_id"])
        self.assertEqual(state["regions_revision"], 0)

    def test_region_export_stages_under_validated_session_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            export_root = Path(temporary)
            source_mask = np.zeros((8, 10), dtype=np.uint8)
            document = {
                "session_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "layout_id": "layout-a",
                "source_mask_hash": "hash-a",
                "regions_revision": 0,
                "regions": [],
            }
            store = _Store(document, source_mask)
            downloads = _Downloads()
            archive_path, status = region_callbacks._export_layout_regions_impl(
                {
                    "Path": Path,
                    "_LAYOUT_REGION_STORE": store,
                    "_layout_region_identity": lambda layout: (
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "layout-a",
                        "hash-a",
                    ),
                    "_layout_regions": _Regions,
                    "_prune_public_downloads": lambda: None,
                    "_public_downloads": downloads,
                    "cv2": cv2,
                    "json": __import__("json"),
                    "np": np,
                    "public_download_dir": export_root / "public",
                    "runtime_export_dir": export_root,
                    "tempfile": tempfile,
                },
                {
                    "session_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "layout_id": "layout-a",
                    "source_mask_pixel_sha256": "client-value",
                },
                {
                    "session_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "layout_id": "layout-a",
                    "source_mask_hash": "hash-a",
                    "regions_revision": 0,
                },
            )
            self.assertIn("revision=0", status)
            self.assertEqual(Path(archive_path).name, "published.zip")
            self.assertIsNotNone(downloads.staging_dir)
            self.assertEqual(downloads.staging_dir.parent, export_root / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            self.assertEqual(downloads.session_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            self.assertTrue((export_root / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").is_dir())
            self.assertEqual(
                {path.name for path in export_root.iterdir()},
                {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            )


if __name__ == "__main__":
    unittest.main()
