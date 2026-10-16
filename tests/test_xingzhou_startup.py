"""Xingzhou startup checks using tiny bundles and mocked CUDA, never live models."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xingzhou_startup", ROOT / "xingzhou_startup.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class XingzhouStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"schema": 2, "container_port": 8030, "entrypoint": "/src/run.sh",
                         "git_sha": "a" * 40, "files": {}}
        for name in startup.REQUIRED:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            self.manifest["files"][name] = {"size": 7}
        profile = (ROOT / "xingzhou-config.json").read_bytes()
        (self.root / "xingzhou-config.json").write_bytes(profile)
        self.manifest["files"]["xingzhou-config.json"]["size"] = len(profile)
        self.save_manifest()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def save_manifest(self):
        (self.root / "bundle-manifest.json").write_text(json.dumps(self.manifest))

    def test_packaged_http_profile_needs_no_injected_variables_or_secret(self):
        result = startup.configure(self.root)
        self.assertEqual(result["deployment_id"], "sam3-xingzhou-10259")
        self.assertEqual(result["origins"], ["http://10.120.105.205:30231"])
        self.assertFalse(result["secure_cookie"])
        self.assertEqual(os.environ["SAM3_ALLOW_INSECURE_COOKIE"], "1")
        self.assertNotIn("SAM3_SESSION_SECRET_FILE", os.environ)

    def test_explicit_https_settings_are_preserved(self):
        os.environ.update(SAM3_ALLOWED_ORIGINS="https://example.test",
                          SAM3_SESSION_COOKIE_SECURE="1",
                          SAM3_SESSION_DEPLOYMENT_ID="another-deployment",
                          SAM3_SESSION_COOKIE_NAME="another_owner_cookie")
        result = startup.configure(self.root)
        self.assertEqual(result["origins"], ["https://example.test:443"])
        self.assertTrue(result["secure_cookie"])
        self.assertEqual(result["deployment_id"], "another-deployment")
        self.assertEqual(os.environ["SAM3_SESSION_COOKIE_NAME"], "another_owner_cookie")

    def test_invalid_explicit_session_settings_are_not_replaced_by_defaults(self):
        for name, value in (
            ("SAM3_ALLOWED_ORIGINS", ""), ("SAM3_ALLOWED_ORIGINS", "http://host/inferenceStart"),
            ("SAM3_ALLOWED_ORIGINS", "*"), ("SAM3_SESSION_COOKIE_SECURE", "maybe"),
            ("SAM3_ALLOW_INSECURE_COOKIE", "0"), ("SAM3_SESSION_COOKIE_NAME", "bad"),
        ):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    startup.configure(self.root)

    def test_http_secure_cookie_and_https_insecure_cookie_mismatch_fail(self):
        for overrides in ({"SAM3_SESSION_COOKIE_SECURE": "1"},
                          {"SAM3_ALLOWED_ORIGINS": "https://example.test"}):
            with self.subTest(overrides=overrides), patch.dict(os.environ, overrides, clear=True):
                with self.assertRaisesRegex(startup.StartupError, "must match"):
                    startup.configure(self.root)

    def test_profile_cannot_inject_unrelated_environment(self):
        profile = json.loads((self.root / "xingzhou-config.json").read_text())
        profile["PYTHONPATH"] = "/unreviewed"
        (self.root / "xingzhou-config.json").write_text(json.dumps(profile))
        with self.assertRaisesRegex(startup.StartupError, "five documented"):
            startup.configure(self.root)
        self.assertNotIn("PYTHONPATH", os.environ)

    def test_manifest_sha_reaches_runtime_environment_without_git(self):
        result = startup.check_bundle(self.root)
        self.assertEqual(result["git_sha"], "a" * 40)
        self.assertEqual(os.environ["SAM3_BUILD_GIT_SHA"], "a" * 40)
        self.assertNotIn("PATH", os.environ)

    def test_malformed_or_conflicting_build_sha_fails(self):
        for sha in ("unknown", "abc123", "a" * 39 + ";", 123):
            with self.subTest(sha=sha):
                self.manifest["git_sha"] = sha
                self.save_manifest()
                with self.assertRaisesRegex(startup.StartupError, "full Git SHA"):
                    startup.check_bundle(self.root)
        self.manifest["git_sha"] = "a" * 40
        self.save_manifest()
        os.environ["SAM3_BUILD_GIT_SHA"] = "b" * 40
        with self.assertRaisesRegex(startup.StartupError, "conflicts"):
            startup.check_bundle(self.root)

    def test_missing_manifest_or_required_file_fails(self):
        (self.root / "bundle-manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            startup.check_bundle(self.root)
        self.save_manifest()
        for name in startup.REQUIRED:
            with self.subTest(name=name):
                target = self.root / name
                content = target.read_bytes()
                target.unlink()
                with self.assertRaisesRegex(startup.StartupError, "missing or empty"):
                    startup.check_bundle(self.root)
                target.write_bytes(content)

    def test_unlisted_required_resource_and_wrong_port_fail(self):
        name = startup.SKILL + "SKILL.md"
        entry = self.manifest["files"].pop(name)
        self.save_manifest()
        with self.assertRaisesRegex(startup.StartupError, "Missing required manifest entry"):
            startup.check_bundle(self.root)
        self.manifest["files"][name] = entry
        self.manifest["container_port"] = 7893
        self.save_manifest()
        with self.assertRaisesRegex(startup.StartupError, "contract"):
            startup.check_bundle(self.root)

    def test_size_change_and_unsafe_manifest_path_fail(self):
        (self.root / "run.sh").write_bytes(b"changed contents")
        with self.assertRaisesRegex(startup.StartupError, "size differs"):
            startup.check_bundle(self.root)
        (self.root / "run.sh").write_bytes(b"fixture")
        self.manifest["files"]["../escape"] = {"size": 0}
        self.save_manifest()
        with self.assertRaisesRegex(startup.StartupError, "Unsafe relative path"):
            startup.check_bundle(self.root)

    def test_external_models_and_empty_package_init_are_supported(self):
        for name in ("models/sam3.pt", "models/big-lama.pt"):
            del self.manifest["files"][name]
        name = "sam3_demo/__init__.py"
        (self.root / name).write_bytes(b"")
        self.manifest["files"][name] = {"size": 0}
        self.save_manifest()
        startup.check_bundle(self.root)

    def test_dependency_failure_names_module_without_exception_secrets(self):
        with patch.object(startup.importlib, "import_module", side_effect=ImportError("sensitive value")):
            with self.assertRaisesRegex(startup.StartupError, "Cannot import torch") as result:
                startup.check_dependencies(self.root)
            self.assertNotIn("sensitive value", str(result.exception))

    def test_gpu_unavailable_fails_and_low_memory_only_warns(self):
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(startup.StartupError, "CUDA is unavailable"):
                startup.check_gpu()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="fixture")), \
             patch.object(torch.cuda, "mem_get_info", return_value=(1024**3, 11 * 1024**3)), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = startup.check_gpu()
            self.assertEqual(result["free_mib"], 1024)
            self.assertEqual(json.loads(output.getvalue())["status"], "warning")

    def test_directory_write_failure_is_actionable(self):
        directories = SimpleNamespace(runtime_dir=self.root / ".runtime", runtime_tmp_dir=self.root / ".runtime/tmp",
                                      public_download_dir=self.root / "downloads", qiyuan_cache_dir=self.root / "cache",
                                      runtime_feedback_dir=self.root / ".runtime/feedback")
        with patch.dict(sys.modules, {"sam3_demo.config": directories}), \
             patch.object(startup.tempfile, "TemporaryFile", side_effect=PermissionError()):
            with self.assertRaisesRegex(startup.StartupError, "not writable"):
                startup.check_directories(self.root)

    def test_main_check_only_does_not_exec_and_success_execs_same_python(self):
        with patch.object(startup, "check_bundle", return_value={}), \
             patch.object(startup, "configure", return_value={}), \
             patch.object(startup, "check_dependencies", return_value={}), \
             patch.object(startup, "check_directories", return_value={}), \
             patch.object(startup, "check_gpu", return_value={}), \
             patch.object(startup.os, "execv") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(startup.main(["--check"]), 0)
            execute.assert_not_called()
            self.assertEqual(startup.main([]), 0)
            execute.assert_called_once_with(sys.executable, [sys.executable, "-m", "sam3_demo.app"])

    def test_main_stops_before_launch_and_suppresses_unknown_exception_values(self):
        with patch.object(startup, "check_bundle", side_effect=ValueError("secret value")), \
             patch.object(startup, "configure") as configure, patch.object(startup.os, "execv") as execute, \
             contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(startup.main([]), 1)
            configure.assert_not_called()
            execute.assert_not_called()
            self.assertNotIn("secret value", output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["check"], "bundle")


if __name__ == "__main__":
    unittest.main()
