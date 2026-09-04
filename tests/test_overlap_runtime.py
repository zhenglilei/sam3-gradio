from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sam3_demo import overlap_runtime


class OverlapRuntimeTests(unittest.TestCase):
    def tearDown(self):
        overlap_runtime.close_overlap_runtime()

    def test_store_and_service_are_lazy_singletons_and_reopen(self):
        overlap_runtime.close_overlap_runtime()
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "layouts.sqlite3"
            with mock.patch.object(overlap_runtime, "overlap_layout_db_path", db_path):
                self.assertIsNone(overlap_runtime._STORE)
                first_store = overlap_runtime.get_overlap_store()
                self.assertIs(first_store, overlap_runtime.get_overlap_store())
                first_service = overlap_runtime.get_overlap_service()
                self.assertIs(first_service, overlap_runtime.get_overlap_service())
                self.assertIs(first_service._store, first_store)
                overlap_runtime.close_overlap_runtime()
                second_store = overlap_runtime.get_overlap_store()
                self.assertIsNot(second_store, first_store)
                self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()
