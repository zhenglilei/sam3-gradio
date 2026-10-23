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
        (self.root / "xingzhou-config.json").write_bytes((ROOT / "xingzhou-config.json").read_bytes())
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def edit_profile(self, **changes):
        profile = json.loads((self.root / "xingzhou-config.json").read_text())
        profile.update(changes)
        (self.root / "xingzhou-config.json").write_text(json.dumps(profile))

    def test_packaged_http_profile_needs_no_injected_variables_or_secret(self):
        result = startup.configure(self.root)
        self.assertEqual(result["deployment_id"], "sam3-xingzhou-10259")
        self.assertEqual(result["origins"], ["http://10.120.105.205:30231"])
        self.assertFalse(result["secure_cookie"])
        self.assertEqual(os.environ["SAM3_ALLOW_INSECURE_COOKIE"], "1")
        self.assertNotIn("SAM3_SESSION_SECRET_FILE", os.environ)
        self.assertEqual(set(result["configuration_sources"].values()), {"profile"})

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
        self.assertEqual(result["configuration_sources"]["SAM3_ALLOWED_ORIGINS"], "environment")
        self.assertEqual(result["configuration_sources"]["SAM3_ALLOW_INSECURE_COOKIE"], "profile")

    def test_profile_can_carry_additional_settings(self):
        self.edit_profile(LAYOUT_MASK_VLM_BASE_URL="http://vlm.example.test/v1",
                          SAM3_ALLOWED_ORIGINS="http://10.0.0.1:30231,http://sam3.example.test")
        result = startup.configure(self.root)
        self.assertEqual(os.environ["LAYOUT_MASK_VLM_BASE_URL"], "http://vlm.example.test/v1")
        self.assertEqual(result["origins"], ["http://10.0.0.1:30231", "http://sam3.example.test:80"])

    def test_non_string_profile_is_rejected(self):
        self.edit_profile(SAM3_SESSION_COOKIE_SECURE=0)
        with self.assertRaisesRegex(startup.StartupError, "string settings"):
            startup.configure(self.root)

    def test_invalid_explicit_session_settings_are_not_replaced_by_defaults(self):
        for name, value in (
            ("SAM3_ALLOWED_ORIGINS", "http://host/inferenceStart"),
            ("SAM3_ALLOWED_ORIGINS", "*"), ("SAM3_SESSION_COOKIE_SECURE", "maybe"),
            ("SAM3_ALLOW_INSECURE_COOKIE", "0"), ("SAM3_SESSION_COOKIE_NAME", "bad"),
        ):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    startup.configure(self.root)

    def test_secure_cookie_with_http_origin_fails(self):
        with patch.dict(os.environ, {"SAM3_SESSION_COOKIE_SECURE": "1"}, clear=True):
            with self.assertRaisesRegex(startup.StartupError, "cannot be used with http://"):
                startup.configure(self.root)

    def test_https_origin_with_non_secure_cookie_is_allowed(self):
        with patch.dict(os.environ, {"SAM3_ALLOWED_ORIGINS": "https://example.test"}, clear=True):
            self.assertEqual(startup.configure(self.root)["origins"], ["https://example.test:443"])

    def test_empty_origin_list_means_same_origin_only(self):
        self.edit_profile(SAM3_ALLOWED_ORIGINS="")
        self.assertEqual(startup.configure(self.root)["origins"], [])

    def test_manifest_sha_is_passed_through_without_validation(self):
        startup.configure(self.root)
        self.assertNotIn("SAM3_BUILD_GIT_SHA", os.environ)
        (self.root / "bundle-manifest.json").write_text(json.dumps({"git_sha": "a" * 40}))
        startup.configure(self.root)
        self.assertEqual(os.environ["SAM3_BUILD_GIT_SHA"], "a" * 40)

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
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([(row["check"], row["status"]) for row in records], [
                ("torch_import", "begin"), ("torch_import", "ok"), ("cuda_init", "begin"), ("gpu", "warning")
            ])

    def test_main_check_only_does_not_exec_and_success_execs_same_python(self):
        with patch.object(startup, "configure", return_value={}), \
             patch.object(startup, "check_gpu", return_value={}), \
             patch.object(startup.os, "execv") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(startup.main(["--check"]), 0)
            execute.assert_not_called()
            self.assertEqual(startup.main([]), 0)
            execute.assert_called_once_with(sys.executable, [sys.executable, "-m", "sam3_demo.app"])

    def test_main_stops_before_launch_and_suppresses_unknown_exception_values(self):
        with patch.object(startup, "configure", side_effect=ValueError("secret value")), \
             patch.object(startup, "check_gpu") as gpu, patch.object(startup.os, "execv") as execute, \
             contextlib.redirect_stderr(io.StringIO()) as output, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(startup.main([]), 1)
            gpu.assert_not_called()
            execute.assert_not_called()
            self.assertNotIn("secret value", output.getvalue())
            self.assertEqual(json.loads(output.getvalue().splitlines()[0])["check"], "session")
            self.assertIn("File ", output.getvalue())

    def test_failure_preserves_dependency_cause_and_does_not_exec(self):
        def broken_import():
            try:
                raise ImportError("libcuda.so.1: cannot open shared object file")
            except ImportError as exc:
                raise RuntimeError("Torch initialization failed") from exc

        with patch.object(startup, "configure", return_value={}), \
             patch.object(startup, "check_gpu", side_effect=broken_import), \
             patch.object(startup.os, "execv") as execute, \
             contextlib.redirect_stderr(io.StringIO()) as output, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(startup.main([]), 1)
            execute.assert_not_called()
            self.assertIn("libcuda.so.1: cannot open shared object file", output.getvalue())
            self.assertIn("RuntimeError: Torch initialization failed", output.getvalue())
            self.assertIn("Traceback (most recent call last)", output.getvalue())

    def test_startup_stage_order_and_timing(self):
        with patch.object(startup, "configure", return_value={}), \
             patch.object(startup, "check_gpu", return_value={}), \
             patch.object(startup.os, "execv"), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(startup.main([]), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([(row["check"], row["status"]) for row in records], [
            ("entry", "begin"), ("session", "begin"), ("session", "ok"),
            ("gpu", "begin"), ("gpu", "ok"), ("preflight", "ok"), ("app_exec", "begin")
        ])
        self.assertTrue(all(row["elapsed_s"] >= 0 and row["pid"] == os.getpid() for row in records))
        self.assertTrue(all(row["duration_s"] >= 0 for row in records if "duration_s" in row))


if __name__ == "__main__":
    unittest.main()
