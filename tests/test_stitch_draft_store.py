from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sam3_demo.stitch_draft_store import StitchDraftStore


class _Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def _state():
    return {
        "schema_version": 2,
        "session_id": "server-session",
        "owner_token": "secret",
        "images": [
            Image.new("RGB", (4, 3), (10, 20, 30)),
            Image.new("RGBA", (2, 2), (1, 2, 3, 128)),
        ],
        "shifts": [(0, 0), (2.5, -1)],
        "rotations": [0.0, 3.25],
        "layout": "horizontal",
        "selected": 1,
        "mosaic": Image.new("RGB", (8, 3), "white"),
        "mosaic_full": Image.new("RGB", (8, 3), "black"),
        "mosaic_crop_bbox_xyxy": [1, 2, 7, 3],
        "revision": 7,
        "generated_revision": 7,
        "mosaic_view_revision": 4,
        "nudge_step": 5,
        "diff_mode": True,
        "show_loupe": False,
        "blend": True,
        "crop_periodic": True,
        "remove_black_border": False,
        "black_border_records": [{"applied": True}],
        "status": "ready",
        "logs": ["loaded"],
        "warnings": ["periodic"],
        "nested": {"session_id": "must-not-leak", "keep": 1},
    }


class StitchDraftStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = _Clock()
        self.store = StitchDraftStore(Path(self.tmp.name) / "drafts", 10, self.clock)
        self.resume_id = "a" * 64

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_persists_business_state_and_png_tiles(self):
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def flush(fd):
            events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            return real_fsync(fd)

        def replace(source, target):
            events.append(Path(target).name)
            return real_replace(source, target)

        with mock.patch("sam3_demo.stitch_draft_store.os.fsync", side_effect=flush), mock.patch(
            "sam3_demo.stitch_draft_store.os.replace", side_effect=replace
        ):
            self.store.save(self.resume_id, _state())
        commit_index = events.index("manifest.json")
        self.assertEqual(events[:commit_index].count("file"), 3)
        self.assertEqual(events[commit_index + 1:], ["directory", "directory"])

        loaded = self.store.load(self.resume_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["shifts"], [[0, 0], [2.5, -1]])
        self.assertEqual(loaded["rotations"], [0.0, 3.25])
        self.assertEqual(loaded["layout"], "horizontal")
        self.assertEqual(loaded["selected"], 1)
        self.assertEqual(loaded["revision"], 7)
        self.assertEqual(loaded["logs"], ["loaded"])
        self.assertEqual(loaded["warnings"], ["periodic"])
        self.assertEqual(len(loaded["images"]), 2)
        self.assertEqual(loaded["images"][0].mode, "RGB")
        self.assertEqual(loaded["images"][0].getpixel((1, 1)), (10, 20, 30))
        self.assertEqual(loaded["images"][1].mode, "RGBA")
        self.assertEqual(loaded["images"][1].getpixel((1, 1)), (1, 2, 3, 128))
        draft = Path(self.tmp.name) / "drafts" / self.resume_id
        self.assertTrue((draft / "images").is_dir())
        self.assertTrue(list((draft / "images").glob("*.png")))

    # These validation cases still use real files; the round-trip and failure cases
    # above/below exercise durable flushes and atomic replacement without stubbing.
    @mock.patch("sam3_demo.stitch_draft_store.os.fsync")
    def test_session_identity_and_derived_images_are_excluded(self, _flush):
        self.store.save(self.resume_id, _state())
        manifest_path = Path(self.tmp.name) / "drafts" / self.resume_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        encoded = json.dumps(manifest, ensure_ascii=False)
        self.assertNotIn("session_id", encoded)
        self.assertNotIn("owner_token", encoded)
        self.assertNotIn("mosaic", manifest["state"])
        self.assertNotIn("mosaic_full", manifest["state"])
        self.assertNotIn("generated_revision", manifest["state"])
        loaded = self.store.load(self.resume_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertNotIn("session_id", loaded)
        self.assertNotIn("owner_token", loaded)
        self.assertEqual(loaded["nested"], {"keep": 1})

    @mock.patch("sam3_demo.stitch_draft_store.os.fsync")
    def test_corrupt_manifest_missing_image_and_unsafe_path_fail_closed(self, _flush):
        self.store.save(self.resume_id, _state())
        draft = Path(self.tmp.name) / "drafts" / self.resume_id
        manifest_path = draft / "manifest.json"
        original = manifest_path.read_text(encoding="utf-8")

        manifest_path.write_text("{not-json", encoding="utf-8")
        self.assertIsNone(self.store.load(self.resume_id))
        manifest_path.write_text(original, encoding="utf-8")

        manifest = json.loads(original)
        (draft / manifest["images"][0]).unlink()
        self.assertIsNone(self.store.load(self.resume_id))

        self.store.save(self.resume_id, _state())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outside = Path(self.tmp.name) / "outside.png"
        Image.new("RGB", (1, 1), "red").save(outside)
        manifest["images"][0] = "../../outside.png"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIsNone(self.store.load(self.resume_id))
        self.assertTrue(outside.is_file())

    def test_resume_id_validation_and_draft_symlink_are_safe(self):
        for invalid in ("", "A" * 64, "a" * 63, "a" * 65, "../" + "a" * 61):
            with self.assertRaises(ValueError):
                self.store.load(invalid)
            with self.assertRaises(ValueError):
                self.store.save(invalid, _state())

        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (outside / "marker").write_text("keep", encoding="utf-8")
        draft = Path(self.tmp.name) / "drafts" / ("b" * 64)
        try:
            draft.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            self.store.load("b" * 64)
        with self.assertRaises(ValueError):
            self.store.delete("b" * 64)
        self.assertTrue((outside / "marker").is_file())

    @mock.patch("sam3_demo.stitch_draft_store.os.fsync")
    def test_ttl_load_and_prune(self, _flush):
        first = "1" * 64
        second = "2" * 64
        self.store.save(first, _state())
        self.clock.value = 105
        self.store.save(second, _state())
        self.clock.value = 111
        self.assertIsNone(self.store.load(first))
        self.assertIsNotNone(self.store.load(second))
        self.clock.value = 116
        self.assertEqual(self.store.prune(), [second])
        self.assertFalse((Path(self.tmp.name) / "drafts" / first).exists())
        self.assertFalse((Path(self.tmp.name) / "drafts" / second).exists())

    def test_failed_manifest_replace_keeps_previous_revision(self):
        self.store.save(self.resume_id, _state())
        next_state = _state()
        next_state["revision"] = 99
        real_replace = os.replace

        def fail_manifest(source, target):
            if Path(target).name == "manifest.json":
                raise OSError("simulated commit failure")
            return real_replace(source, target)

        with mock.patch("sam3_demo.stitch_draft_store.os.replace", side_effect=fail_manifest):
            with self.assertRaises(OSError):
                self.store.save(self.resume_id, next_state)
        loaded = self.store.load(self.resume_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["revision"], 7)

    def test_opaque_tensor_like_values_are_rejected(self):
        state = _state()
        state["gpu_tensor"] = object()
        with self.assertRaises(ValueError):
            self.store.save(self.resume_id, state)


if __name__ == "__main__":
    unittest.main()
