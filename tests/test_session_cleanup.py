"""Tests for isolated server-session cleanup."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from sam3_demo.session_cleanup import (
    cleanup_session_resources,
    validate_server_session_id,
)


class SessionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.sid_a = "a" * 32
        self.sid_b = "b" * 32
        self.layout = {
            self.sid_a: object(),
            self.sid_a + ":layout-1": object(),
            self.sid_b: object(),
            self.sid_a + "x": object(),
            7: object(),
        }
        self.epochs = {self.sid_a: 1, self.sid_b: 2}

    def _cleanup(self, sid, *, workspace=None, source=None, roots=()):
        return cleanup_session_resources(
            sid,
            clear_workspace_cache=workspace or (lambda value: None),
            clear_source_image_cache=source or (lambda value: None),
            layout_cache=self.layout,
            layout_cache_lock=threading.RLock(),
            prompt_epochs=self.epochs,
            prompt_epoch_lock=threading.RLock(),
            persistent_roots=roots,
        )

    def test_validation_is_exact_and_happens_before_mutation(self):
        self.assertEqual(validate_server_session_id(self.sid_a), self.sid_a)
        for value in (
            "",
            "A" * 32,
            "a" * 31,
            "a" * 33,
            self.sid_a + ":x",
            "../" + self.sid_a,
            None,
            123,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_server_session_id(value)

        calls = []
        with self.assertRaises(ValueError):
            cleanup_session_resources(
                "../" + self.sid_a,
                clear_workspace_cache=lambda value: calls.append(("workspace", value)),
                clear_source_image_cache=lambda value: calls.append(("source", value)),
                layout_cache=self.layout,
                layout_cache_lock=threading.RLock(),
                prompt_epochs=self.epochs,
                prompt_epoch_lock=threading.RLock(),
            )
        self.assertEqual(calls, [])
        self.assertIn(self.sid_a, self.layout)
        self.assertIn(self.sid_a, self.epochs)

    def test_only_target_session_is_removed_from_all_resources(self):
        workspace_calls = []
        source_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "layout"
            root.mkdir()
            (root / self.sid_a / "nested").mkdir(parents=True)
            (root / self.sid_a / "nested" / "mask.bin").write_bytes(b"a")
            (root / self.sid_b / "mask.bin").parent.mkdir(parents=True)
            (root / self.sid_b / "mask.bin").write_bytes(b"b")
            result = self._cleanup(
                self.sid_a,
                workspace=lambda sid: workspace_calls.append(sid),
                source=lambda sid: source_calls.append(sid),
                roots=(root,),
            )

            self.assertEqual(workspace_calls, [self.sid_a])
            self.assertEqual(source_calls, [self.sid_a])
            self.assertNotIn(self.sid_a, self.layout)
            self.assertNotIn(self.sid_a + ":layout-1", self.layout)
            self.assertIn(self.sid_b, self.layout)
            self.assertIn(self.sid_a + "x", self.layout)
            self.assertNotIn(self.sid_a, self.epochs)
            self.assertEqual(self.epochs[self.sid_b], 2)
            self.assertFalse((root / self.sid_a).exists())
            self.assertTrue((root / self.sid_b / "mask.bin").exists())
            self.assertTrue(root.exists())
            self.assertFalse(result["errors"])
            self.assertGreaterEqual(result["persistent_roots"][0]["count"], 3)

    def test_symlink_is_unlinked_without_following_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "do-not-delete").write_text("keep", encoding="utf-8")
            os.symlink(outside, root / self.sid_a)

            result = self._cleanup(self.sid_a, roots=(root,))

            self.assertFalse((root / self.sid_a).exists())
            self.assertTrue((outside / "do-not-delete").exists())
            self.assertTrue(root.exists())
            self.assertEqual(result["errors"], [])

    def test_symlink_ancestor_root_is_rejected_without_deleting_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            real_root = outside / "root"
            real_root.mkdir(parents=True)
            target = real_root / self.sid_a / "keep"
            target.parent.mkdir()
            target.write_text("keep", encoding="utf-8")
            alias = base / "alias"
            os.symlink(outside, alias)

            result = self._cleanup(self.sid_a, roots=(alias / "root",))

            self.assertTrue(target.exists())
            self.assertTrue(result["errors"])
            self.assertIn("must not be a symlink", result["errors"][0]["error"])

    def test_nested_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            outside = base / "outside"
            (root / self.sid_a).mkdir(parents=True)
            outside.mkdir()
            (outside / "keep").write_text("keep", encoding="utf-8")
            os.symlink(outside / "keep", root / self.sid_a / "link")

            self._cleanup(self.sid_a, roots=(root,))

            self.assertTrue((outside / "keep").exists())
            self.assertTrue(root.exists())
            self.assertFalse((root / self.sid_a).exists())

    def test_failure_does_not_prevent_later_resources(self):
        called = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            (root / self.sid_a).mkdir(parents=True)
            (root / self.sid_a / "x").write_text("x", encoding="utf-8")

            def fail(_sid):
                called.append("workspace")
                raise RuntimeError("synthetic failure")

            result = self._cleanup(
                self.sid_a,
                workspace=fail,
                source=lambda sid: called.append(("source", sid)),
                roots=(root,),
            )

            self.assertEqual(called, ["workspace", ("source", self.sid_a)])
            self.assertTrue(result["errors"])
            self.assertEqual(result["errors"][0]["resource"], "workspace_cache")
            self.assertFalse((root / self.sid_a).exists())
            self.assertNotIn(self.sid_a, self.layout)
            self.assertNotIn(self.sid_a, self.epochs)


if __name__ == "__main__":
    unittest.main()
