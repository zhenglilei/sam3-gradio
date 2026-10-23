"""Container entrypoint contract without starting a model or a web server."""

import ast
import hashlib
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
        self.requirements = self.directory / "requirements-offline.txt"
        self.requirements.write_text("fixturedep==1.2.3\n")
        (self.directory / "wheels").mkdir()
        self.log = self.directory / "probe.jsonl"
        python = self.directory / "python-probe"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "log = pathlib.Path(os.environ['PROBE_LOG'])\n"
            "def record(value):\n"
            "    with log.open('a') as stream:\n"
            "        stream.write(json.dumps(value) + '\\n')\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-c':\n"
            "    exec(compile(sys.argv[2], '<probe>', 'exec'), {'__name__': '__main__'})\n"
            "elif len(sys.argv) > 2 and sys.argv[1:3] == ['-m', 'pip']:\n"
            "    args = sys.argv[3:]\n"
            "    record({'kind': 'pip', 'args': args})\n"
            "    target = pathlib.Path(args[args.index('--target') + 1])\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    if os.environ.get('PROBE_PIP_FAIL') == '1':\n"
            "        raise SystemExit(23)\n"
            "elif len(sys.argv) > 1 and sys.argv[1].endswith('xingzhou_envreport.py'):\n"
            "    record({'kind': 'envreport', 'pythonpath': os.environ.get('PYTHONPATH', '')})\n"
            "    if os.environ.get('PROBE_REPORT_FAIL') == '1':\n"
            "        raise SystemExit(5)\n"
            "elif len(sys.argv) > 1 and sys.argv[1].endswith('xingzhou_startup.py'):\n"
            "    record({'kind': 'startup', 'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'pythonpath': os.environ.get('PYTHONPATH', '')})\n"
            "    print(json.dumps({'args': sys.argv[1:], 'interpreter': os.path.basename(sys.argv[0]), "
            "'cwd': os.getcwd(), 'host': os.environ['SAM3_SERVER_NAME'], "
            "'port': os.environ['SAM3_SERVER_PORT'], 'cache': os.environ['XDG_CACHE_HOME'], "
            "'pythonpath': os.environ.get('PYTHONPATH', '')}))\n"
            "    mode = os.environ.get('PROBE_STARTUP_EXIT')\n"
            "    if mode == 'kill':\n"
            "        os.kill(os.getpid(), 9)\n"
            "    elif mode:\n"
            "        raise SystemExit(int(mode))\n"
            "else:\n"
            "    record({'kind': 'unknown', 'args': sys.argv[1:]})\n"
            "    raise SystemExit(97)\n"
        )
        python.chmod(0o700)
        self.env = {
            "PATH": os.environ["PATH"],
            "PROBE_LOG": str(self.log),
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

    def records(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def expected_deps_dir(self):
        digest = hashlib.sha256(self.requirements.read_bytes()).hexdigest()[:16]
        return self.directory / ".runtime" / (
            f"python-deps-{sys.version_info[0]}{sys.version_info[1]}-{digest}"
        )

    def use_interpreter_path(self, *names):
        directory = self.directory / "bin"
        directory.mkdir()
        (directory / "dirname").symlink_to(shutil.which("dirname"))
        (directory / "touch").symlink_to(shutil.which("touch"))
        (directory / "date").symlink_to(shutil.which("date"))
        for name in names:
            (directory / name).symlink_to(self.directory / "python-probe")
        self.env["PATH"] = str(directory)

    def test_python3_only_is_discovered_and_check_is_forwarded(self):
        self.use_interpreter_path("python3")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["interpreter"], "python3")
        self.assertEqual(record["args"], [str(self.directory / "xingzhou_startup.py"), "--check"])

    def test_python_only_is_discovered(self):
        self.use_interpreter_path("python")
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

    def test_entrypoint_fixed_port_and_bundle_workdir(self):
        self.use_interpreter_path("python3")
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
        self.use_interpreter_path("python3")
        self.env.pop("SAM3_ALLOWED_ORIGINS")
        self.env.pop("SAM3_SESSION_COOKIE_SECURE")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["args"], [
            str(self.directory / "xingzhou_startup.py"), "--check"
        ])

    def test_explicit_cache_is_preserved(self):
        self.use_interpreter_path("python3")
        self.env["XDG_CACHE_HOME"] = "/cache/sam3"
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["cache"], "/cache/sam3")

    def test_offline_install_has_no_network_or_gpu_flags_and_uses_versioned_target(self):
        self.use_interpreter_path("python3")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        pip_records = [record for record in self.records() if record["kind"] == "pip"]
        self.assertEqual(len(pip_records), 1)
        args = pip_records[0]["args"]
        for flag in ("--isolated", "--no-index", "--only-binary=:all:", "--no-deps", "--upgrade", "--no-compile"):
            self.assertIn(flag, args)
        self.assertNotIn("--index-url", args)
        self.assertEqual(args[args.index("--find-links") + 1], str(self.directory / "wheels"))
        self.assertEqual(args[args.index("--target") + 1], str(self.expected_deps_dir()))
        self.assertEqual(args[args.index("-r") + 1], str(self.requirements))
        command = " ".join(args).lower()
        for gpu_name in ("torch", "torchvision", "torchaudio", "triton", "nvidia-", "cuda"):
            self.assertNotIn(gpu_name, command)

    def test_failed_offline_install_blocks_launch_and_leaves_no_marker(self):
        self.use_interpreter_path("python3")
        self.env["PROBE_PIP_FAIL"] = "1"
        result = self.launch("--check")
        self.assertEqual(result.returncode, 23)
        self.assertEqual(result.stdout, "")
        self.assertFalse(any(record["kind"] == "startup" for record in self.records()))
        self.assertFalse((self.expected_deps_dir() / ".installed").exists())
        self.assertIn("stage=offline_install status=failed exit_code=23", result.stderr)
        self.assertNotIn("stage=python status=begin", result.stderr)

    def test_successful_marker_skips_reinstall_and_prepends_pythonpath_overlay(self):
        self.use_interpreter_path("python3")
        existing = self.directory / "existing-pythonpath"
        self.env["PYTHONPATH"] = str(existing)
        first = self.launch("--check")
        second = self.launch("--check")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        records = self.records()
        self.assertEqual(sum(record["kind"] == "pip" for record in records), 1)
        startup = [record for record in records if record["kind"] == "startup"]
        self.assertEqual(len(startup), 2)
        overlay = str(self.expected_deps_dir())
        for record in startup:
            pythonpath = record["pythonpath"].split(os.pathsep)
            self.assertEqual(pythonpath[0], overlay)
            self.assertIn(str(existing), pythonpath)
        self.assertTrue((self.expected_deps_dir() / ".installed").is_file())
        self.assertIn("stage=offline_install status=skipped cache=hit", second.stderr)

    def test_environment_report_runs_first_without_overlay_and_never_blocks(self):
        self.use_interpreter_path("python3")
        self.env["PROBE_REPORT_FAIL"] = "1"
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        kinds = [record["kind"] for record in self.records()]
        self.assertEqual(kinds, ["envreport", "pip", "startup"])
        self.assertNotIn(str(self.expected_deps_dir()), self.records()[0]["pythonpath"])
        self.assertIn("stage=environment_report status=warning exit_code=5 continuing=true", result.stderr)

    def test_boot_stages_distinguish_pip_return_marker_and_exec(self):
        self.use_interpreter_path("python3")
        result = self.launch("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        stages = [
            "stage=entry status=begin", "stage=environment_report status=begin",
            "stage=environment_report status=ok", "stage=dependency_target status=begin",
            "stage=dependency_target status=ok", "stage=offline_install status=begin cache=miss",
            "stage=offline_install status=ok exit_code=0", "stage=dependency_marker status=ok",
            "stage=python status=begin", "stage=python status=ok exit_code=0",
        ]
        positions = [result.stderr.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("elapsed_s=", result.stderr)

    def test_python_exit_code_and_signal_are_logged_and_returned(self):
        self.use_interpreter_path("python3")
        for value, code, line in (("9", 9, "stage=python status=failed exit_code=9"),
                                  ("kill", 137, "stage=python status=failed exit_code=137 signal=9")):
            with self.subTest(value=value):
                self.env["PROBE_STARTUP_EXIT"] = value
                result = self.launch("--check")
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn(line, result.stderr)

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
