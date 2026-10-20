"""Base-image environment report: shape, isolation from stdout and secret filtering."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class XingzhouEnvReportTests(unittest.TestCase):
    def run_report(self, **environment):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "requirements-offline.txt").write_text("# lock\npip==0.0.1\nnot-installed==1.0\n")
            return subprocess.run(
                [sys.executable, str(ROOT / "xingzhou_envreport.py")], cwd=directory,
                env={**os.environ, **environment}, capture_output=True, text=True, timeout=120,
            )

    def test_report_writes_json_sections_to_stderr_only(self):
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
        self.assertEqual([record["section"] for record in records], ["system", "gpu", "torch", "packages"])
        self.assertTrue(all(record["event"] == "xingzhou_env" for record in records))
        system = records[0]
        self.assertEqual(system["python"], sys.version.split()[0])
        packages = records[3]
        self.assertIn("pip", packages["image_packages"])
        self.assertIn("pip", packages["bundle_overrides_image"])
        self.assertNotIn("not-installed", packages["bundle_overrides_image"])

    def test_only_cuda_names_without_credentials_are_reported(self):
        result = self.run_report(CUDA_VERSION="12.4.1", NVIDIA_API_KEY="hidden-value",
                                 UNRELATED_SETTING="hidden-value")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("hidden-value", result.stderr)
        gpu = next(json.loads(line) for line in result.stderr.splitlines() if '"section": "gpu"' in line)
        self.assertEqual(gpu["environment"]["CUDA_VERSION"], "12.4.1")


if __name__ == "__main__":
    unittest.main()
