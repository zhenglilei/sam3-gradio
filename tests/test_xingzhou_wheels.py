"""Offline wheel metadata contracts without installing dependencies."""

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy/xingzhou"
sys.path.insert(0, str(DEPLOY_DIR))
try:
    import prepare_wheels
finally:
    sys.path.remove(str(DEPLOY_DIR))


class XingzhouWheelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    @staticmethod
    def write_wheel(path, name, version, metadata=True):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
            if metadata:
                wheel.writestr(
                    f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                )

    def test_lock_accepts_exact_non_gpu_pins(self):
        self.assertEqual(
            prepare_wheels.read_lock("# fixture\nfixture-dep==1.2.3\n"),
            {"fixture-dep": "1.2.3"},
        )

    def test_lock_rejects_gpu_dependencies(self):
        with self.assertRaisesRegex(ValueError, "GPU dependency"):
            prepare_wheels.read_lock("torch==2.7.0\n")
        with self.assertRaisesRegex(ValueError, "GPU dependency"):
            prepare_wheels.read_lock("nvidia-cublas-cu12==12.6.4.1\n")

    def test_wheel_files_validates_metadata_and_rejects_gpu_wheels(self):
        lock = "fixture-dep==1.2.3\n"
        renamed = self.directory / "renamed.whl"
        self.write_wheel(renamed, "fixture-dep", "1.2.3")
        self.assertEqual(prepare_wheels.wheel_files(self.directory, lock), [renamed])

        self.write_wheel(self.directory / "torch-2.7.0.whl", "torch", "2.7.0")
        with self.assertRaisesRegex(ValueError, "does not match application lock"):
            prepare_wheels.wheel_files(self.directory, lock)

    def test_wheel_files_rejects_missing_metadata(self):
        self.write_wheel(self.directory / "fixture-dep-1.2.3.whl", "fixture-dep", "1.2.3", metadata=False)
        with self.assertRaisesRegex(ValueError, "no unique metadata"):
            prepare_wheels.wheel_files(self.directory, "fixture-dep==1.2.3\n")


if __name__ == "__main__":
    unittest.main()
