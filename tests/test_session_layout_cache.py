import json
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import layout_transform_utils
from sam3_demo.layout import mask_callbacks


class SessionLayoutCacheTests(unittest.TestCase):
    def setUp(self):
        self.sid_a = "a" * 32
        self.sid_b = "b" * 32
        self.cache = {
            f"{self.sid_a}:layout-1": "a1",
            f"{self.sid_a}:layout-2": "a2",
            f"{self.sid_b}:layout-1": "b1",
        }
        self.deps = {
            "_LAYOUT_CACHE": self.cache,
            "_LAYOUT_CACHE_LOCK": threading.RLock(),
            "_layout_cache_key": lambda sid, lid: f"{sid}:{lid}",
        }

    def test_cache_and_disk_keys_require_server_session_id(self):
        deps = {
            "_layout_tx": layout_transform_utils,
            "runtime_layout_dir": __import__("pathlib").Path("/tmp/layouts"),
        }
        for invalid in (None, "", "default", "A" * 32, "../escape"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    mask_callbacks._layout_cache_key_impl(deps, invalid, "layout-1")
                with self.assertRaises(ValueError):
                    mask_callbacks._layout_disk_dir_impl(deps, invalid, "layout-1")

    def test_disk_restore_rejects_metadata_from_another_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / self.sid_a / "layout-1"
            out_dir.mkdir(parents=True)
            (out_dir / "source_mask.png").write_bytes(b"not-a-png")
            (out_dir / "layout_meta.json").write_text(
                json.dumps(
                    {
                        "session_id": self.sid_b,
                        "layout_id": "layout-1",
                    }
                ),
                encoding="utf-8",
            )
            deps = {
                "Image": Image,
                "_layout_disk_dir": lambda session_id, layout_id: out_dir,
                "_layout_mask_to_preview": lambda mask: None,
                "_layout_tx": layout_transform_utils,
                "cv2": cv2,
                "json": json,
                "np": np,
            }
            with self.assertRaisesRegex(ValueError, "metadata identity"):
                mask_callbacks._restore_layout_cache_from_disk_impl(
                    deps,
                    self.sid_a,
                    "layout-1",
                )

    def test_clear_removes_only_exact_session_layout(self):
        mask_callbacks._clear_layout_cache_impl(
            self.deps,
            {"session_id": self.sid_a, "layout_id": "layout-1"},
        )
        self.assertNotIn(f"{self.sid_a}:layout-1", self.cache)
        self.assertIn(f"{self.sid_a}:layout-2", self.cache)
        self.assertIn(f"{self.sid_b}:layout-1", self.cache)

    def test_missing_layout_id_is_session_local_noop(self):
        before = dict(self.cache)
        mask_callbacks._clear_layout_cache_impl(
            self.deps,
            {"session_id": self.sid_a, "layout_id": None},
        )
        self.assertEqual(self.cache, before)

    def test_missing_session_cannot_clear_cache(self):
        for state in (None, {}, {"layout_id": "layout-1"}):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    mask_callbacks._clear_layout_cache_impl(self.deps, state)
                self.assertEqual(len(self.cache), 3)


if __name__ == "__main__":
    unittest.main()
