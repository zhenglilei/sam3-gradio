"""Production ZIP contracts, using tiny fixtures and no model execution."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("production_bundle", Path(__file__).resolve().parents[1] / "deploy/xingzhou/bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class ProductionBundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        names = list(bundle.REQUIRED)
        for component in bundle.COMPONENTS:
            names.append(component + "/backend/pkg/templates/component/index.js")
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"production fixture")
        listing = self.root / bundle.FILE_LIST
        listing.parent.mkdir(parents=True)
        listing.write_text("\n".join(names) + "\n")
        for name in ["tests/test_bad.py", ".runtime/retired/legacy.key", "reference_code.py", "sam3_demo/app.py.orig"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"never package")
        self.git("init", "-q")
        self.commit()
        self.archive = self.base / "bundle.zip"

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.STDOUT)

    def commit(self):
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")

    def test_reviewed_manifest_contains_only_supported_runtime_files(self):
        listing = Path(__file__).resolve().parents[1] / bundle.FILE_LIST
        for name in listing.read_text().splitlines():
            if name and not name.startswith("#"):
                self.assertTrue(bundle.selected(name), name)

    def test_flat_production_zip_uses_committed_bytes_and_excludes_unrelated_files(self):
        (self.root / "run.sh").write_bytes(b"uncommitted override")
        (self.root / "sam3_demo/untracked.py").write_bytes(b"untracked")
        bundle.build(self.root, self.archive)
        manifest = bundle.verify(self.archive)
        self.assertEqual(manifest["models"], "external mount")
        self.assertEqual(manifest["source"], "committed Git blobs")
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.read("run.sh"), b"production fixture")
            self.assertNotIn("repo/run.sh", archive.namelist())
            for name in ["tests/test_bad.py", ".runtime/retired/legacy.key", "reference_code.py",
                         "sam3_demo/app.py.orig", "sam3_demo/untracked.py", *bundle.MODELS]:
                self.assertNotIn(name, archive.namelist())
        with self.assertRaises(FileExistsError):
            bundle.build(self.root, self.archive)

    def test_models_are_only_included_explicitly_and_symlinks_become_file_bytes(self):
        external = self.base / "model.pt"
        external.write_bytes(b"trusted model fixture")
        for name in bundle.MODELS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(external)
        bundle.build(self.root, self.archive, with_models=True)
        self.assertEqual(bundle.verify(self.archive)["models"], "included")
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.read(bundle.MODELS[0]), external.read_bytes())

    def test_committed_source_symlink_is_rejected(self):
        path = self.root / "sam3_demo/app.py"
        path.unlink()
        path.symlink_to(self.base / "outside.py")
        self.commit()
        with self.assertRaisesRegex(ValueError, "committed regular file"):
            bundle.build(self.root, self.archive)

    def test_nonproduction_allowlist_entry_is_rejected(self):
        with (self.root / bundle.FILE_LIST).open("a") as target:
            target.write("tests/test_bad.py\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Non-production"):
            bundle.build(self.root, self.archive)

    def test_missing_compiled_component_is_rejected(self):
        listing = self.root / bundle.FILE_LIST
        listing.write_text("\n".join(line for line in listing.read_text().splitlines()
                                     if not line.startswith(bundle.COMPONENTS[0] + "/")))
        self.commit()
        with self.assertRaisesRegex(ValueError, "Missing compiled component"):
            bundle.build(self.root, self.archive)

    def test_tampering_and_path_traversal_are_rejected(self):
        bundle.build(self.root, self.archive)
        changed = self.base / "tampered.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(changed, "w") as target:
            for item in source.infolist():
                target.writestr(item, b"wrong" if item.filename == "run.sh" else source.read(item))
        with self.assertRaisesRegex(ValueError, "Size mismatch|Checksum mismatch"):
            bundle.verify(changed)
        traversal = self.base / "traversal.zip"
        with zipfile.ZipFile(traversal, "w") as archive:
            archive.writestr("../escape", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
            bundle.verify(traversal)


if __name__ == "__main__":
    unittest.main()
