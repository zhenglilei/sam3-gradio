"""Container entrypoint contract without starting a model or a web server."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "models/sam3.pt", "models/big-lama.pt", "assets/bpe_simple_vocab_16e6.txt.gz"
)


class XingzhouEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        shutil.copyfile(ROOT / "run.sh", self.directory / "run.sh")
        for name in ASSETS:
            target = self.directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
        python = self.directory / "python-probe"
        python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
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

    def launch(self):
        return subprocess.run(
            ["sh", str(self.directory / "run.sh")], cwd="/", env=self.env,
            capture_output=True, text=True, timeout=10,
        )

    def test_entrypoint_fixed_port_and_bundle_workdir(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(records[0]["args"], [
            "-c", "from sam3_demo.session_web import load_session_cookie_settings; load_session_cookie_settings()"
        ])
        self.assertEqual(records[1]["args"], ["-m", "sam3_demo.app"])
        self.assertEqual(records[1]["host"], "0.0.0.0")
        self.assertEqual(records[1]["port"], "8030")
        self.assertEqual(records[1]["cwd"], str(self.directory))
        self.assertEqual(records[1]["cache"], str(self.directory / ".runtime/cache"))

    def test_missing_security_configuration_fails_before_launch(self):
        for name in ("SAM3_ALLOWED_ORIGINS", "SAM3_SESSION_COOKIE_SECURE"):
            with self.subTest(name=name):
                value = self.env.pop(name)
                result = self.launch()
                self.env[name] = value
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertFalse(result.stdout)

    def test_missing_asset_fails_without_download_prompt(self):
        for name in ASSETS:
            with self.subTest(name=name):
                target = self.directory / name
                target.unlink()
                result = self.launch()
                target.write_bytes(b"fixture")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertFalse(result.stdout)

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
