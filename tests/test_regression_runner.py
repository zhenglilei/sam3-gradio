import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import run_tests


class RegressionRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = [
            "tests/test_web.py", "tests/offline_eval/test_offline.py",
            "first/tests/test_backend.py", "second/tests/test_backend.py",
        ]
        for relative in self.files:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in ("first", "second"):
            config = self.root / name / "frontend" / "gradio.config.js"
            config.parent.mkdir()
            config.touch()
        archived = self.root / "archive" / "tests" / "test_old.py"
        archived.parent.mkdir(parents=True)
        archived.touch()

    def test_groups_partition_all_test_files_without_duplicates(self):
        groups = run_tests.test_groups(self.root, "all")
        paths = [path for _, directory in groups for path in directory.glob("test_*.py")]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), {self.root / relative for relative in self.files})
        for suite, count in (("web", 1), ("offline", 1), ("components", 2)):
            with self.subTest(suite=suite):
                self.assertEqual(len(run_tests.test_groups(self.root, suite)), count)

    def test_runner_isolates_component_imports_and_propagates_failure(self):
        results = [subprocess.CompletedProcess([], code) for code in (0, 1, 0, 0)]
        with mock.patch.object(run_tests, "ROOT", self.root), mock.patch.object(
            run_tests.subprocess, "run", side_effect=results
        ) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_tests.main(["--suite", "all"]), 1)
        self.assertEqual(run.call_count, 4)
        for call, (_, directory) in zip(run.call_args_list, run_tests.test_groups(self.root, "all")):
            self.assertEqual(call.args[0][0], sys.executable)
            self.assertEqual(call.kwargs["cwd"], self.root)
            self.assertEqual(call.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], str(directory))

    def test_listing_does_not_run_tests_and_empty_group_is_an_error(self):
        with mock.patch.object(run_tests, "ROOT", self.root), mock.patch.object(
            run_tests.subprocess, "run"
        ) as run, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_tests.main(["--suite", "all", "--list"]), 0)
            run.assert_not_called()
            (self.root / self.files[0]).unlink()
            with self.assertRaises(SystemExit) as error:
                run_tests.main(["--suite", "web"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
