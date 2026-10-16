"""Container entrypoint contract without starting a model or a web server."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
class XingzhouEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        shutil.copyfile(ROOT / "run.sh", self.directory / "run.sh")
        python = self.directory / "python-probe"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'interpreter': os.path.basename(sys.argv[0]), 'cwd': os.getcwd(), "
            "'host': os.environ['SAM3_SERVER_NAME'], 'port': os.environ['SAM3_SERVER_PORT'], "
            "'cache': os.environ['XDG_CACHE_HOME']}))\n"
        )
        python.chmod(0o700)
        self.env = {
            "PATH": os.environ["PATH"],
            "PYTHON_BIN": str(python),
            "SAM3_ALLOWED_ORIGINS": "https://sam3.example.test",
            "SAM3_SESSION_COOKIE_SECURE": "1",
            "SAM3_SERVER_PORT": "7893",
            "SAM3_SERVER_NAME": "127.0.0.1",
        }

    def launch(self, *args):
        return subprocess.run(
            ["/bin/sh", str(self.directory / "run.sh"), *args], cwd="/", env=self.env,
            capture_output=True, text=True, timeout=10,
        )

    def use_interpreter_path(self, *names):
        directory = self.directory / "bin"
        directory.mkdir()
        (directory / "dirname").symlink_to(shutil.which("dirname"))
        for name in names:
            (directory / name).symlink_to(self.directory / "python-probe")
        self.env["PATH"] = str(directory)
        self.env.pop("PYTHON_BIN", None)

    def test_python3_only_is_discovered_and_check_is_forwarded(self):
        self.use_interpreter_path("python3")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["interpreter"], "python3")
        self.assertEqual(record["args"], [str(self.directory / "xingzhou_startup.py"), "--check"])

    def test_python_only_is_discovered_with_empty_override(self):
        self.use_interpreter_path("python")
        self.env["PYTHON_BIN"] = ""
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["interpreter"], "python")

    def test_python3_is_preferred_when_both_are_available(self):
        self.use_interpreter_path("python3", "python")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["interpreter"], "python3")

    def test_missing_interpreters_fail_before_launch(self):
        self.use_interpreter_path()
        result = self.launch()
        self.assertEqual(result.returncode, 127)
        self.assertEqual(result.stdout, "")
        self.assertIn("Python interpreter not found", result.stderr)
        self.assertIn("PYTHON_BIN", result.stderr)

    def test_invalid_explicit_interpreter_does_not_fall_back(self):
        self.use_interpreter_path("python3", "python")
        nonexecutable = self.directory / "nonexecutable-python"
        nonexecutable.write_text("#!/bin/sh\nexit 0\n")
        nonexecutable.chmod(0o600)
        for value in ("missing-python", str(nonexecutable), str(self.directory)):
            with self.subTest(value=value):
                self.env["PYTHON_BIN"] = value
                result = self.launch()
                self.assertEqual(result.returncode, 127)
                self.assertEqual(result.stdout, "")
                self.assertIn("Invalid PYTHON_BIN", result.stderr)
                self.assertIn("executable Python interpreter", result.stderr)

    def test_explicit_interpreter_name_overrides_python3_preference(self):
        self.use_interpreter_path("python3", "python")
        self.env["PYTHON_BIN"] = "python"
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["interpreter"], "python")

    def test_entrypoint_fixed_port_and_bundle_workdir(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["args"], [str(self.directory / "xingzhou_startup.py")])
        self.assertEqual(records[0]["host"], "0.0.0.0")
        self.assertEqual(records[0]["port"], "8030")
        self.assertEqual(records[0]["cwd"], str(self.directory))
        self.assertEqual(records[0]["cache"], str(self.directory / ".runtime/cache"))

    def test_check_only_is_forwarded_without_shell_security_defaults(self):
        self.env.pop("SAM3_ALLOWED_ORIGINS")
        self.env.pop("SAM3_SESSION_COOKIE_SECURE")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["args"], [
            str(self.directory / "xingzhou_startup.py"), "--check"
        ])

    def test_explicit_cache_is_preserved(self):
        self.env["XDG_CACHE_HOME"] = "/cache/sam3"
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["cache"], "/cache/sam3")

    def test_config_cache_override(self):
        tree = ast.parse((ROOT / "sam3_demo/config.py").read_text())
        assignment = next(
            node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "qiyuan_cache_dir" for target in node.targets)
        )
        for cache, expected in (
            (None, "/data/zhengqiyuan/.cache"),
            ("", "/data/zhengqiyuan/.cache"),
            ("/src/.runtime/cache", "/src/.runtime/cache"),
        ):
            with self.subTest(cache=cache), patch.dict(os.environ, {}, clear=True):
                if cache is not None:
                    os.environ["XDG_CACHE_HOME"] = cache
                value = eval(
                    compile(ast.Expression(assignment.value), "config.py", "eval"),
                    {"os": os, "Path": Path},
                )
                self.assertEqual(value, Path(expected))


if __name__ == "__main__":
    unittest.main()
