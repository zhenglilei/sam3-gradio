"""Production ZIP contracts, using tiny fixtures and no model execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy/xingzhou"
sys.path.insert(0, str(DEPLOY_DIR))
try:
    spec = importlib.util.spec_from_file_location("production_bundle", DEPLOY_DIR / "bundle.py")
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
finally:
    sys.path.remove(str(DEPLOY_DIR))


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
        (self.root / "requirements-offline.txt").write_text("fixturedep==1.2.3\n")
        self.wheels_dir = self.root / ".runtime/xingzhou-wheels"
        self.wheels_dir.mkdir(parents=True)
        self.write_wheel(self.wheels_dir / "fixturedep-1.2.3-py3-none-any.whl", "fixturedep", "1.2.3")
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

    @staticmethod
    def write_wheel(path, name, version):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
            wheel.writestr(
                f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            )

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
            self.assertIn("wheels/fixturedep-1.2.3-py3-none-any.whl", archive.namelist())
            self.assertNotIn("repo/run.sh", archive.namelist())
            for name in ["tests/test_bad.py", ".runtime/retired/legacy.key", "reference_code.py",
                         "sam3_demo/app.py.orig", "sam3_demo/untracked.py", *bundle.MODELS]:
                self.assertNotIn(name, archive.namelist())
        with self.assertRaises(FileExistsError):
            bundle.build(self.root, self.archive)

    def test_missing_offline_wheel_is_rejected(self):
        next(self.wheels_dir.glob("*.whl")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing offline wheels"):
            bundle.build(self.root, self.archive)

    def test_wrong_version_offline_wheel_is_rejected(self):
        next(self.wheels_dir.glob("*.whl")).unlink()
        self.write_wheel(self.wheels_dir / "fixturedep-1.2.4-py3-none-any.whl", "fixturedep", "1.2.4")
        with self.assertRaisesRegex(ValueError, "does not match application lock"):
            bundle.build(self.root, self.archive)

    def test_gpu_offline_wheel_is_rejected(self):
        self.write_wheel(self.wheels_dir / "torch-2.7.0-py3-none-any.whl", "torch", "2.7.0")
        with self.assertRaisesRegex(ValueError, "does not match application lock"):
            bundle.build(self.root, self.archive)

    def test_runtime_documents_are_explicitly_allowlisted_and_packaged(self):
        expected = {
            "sam3_demo/layout/skills/layout-mask-preprocess/SKILL.md",
            "sam3_demo/layout/skills/layout-mask-preprocess/references/operation-catalog.md",
            "sam3_demo/layout/skills/layout-mask-preprocess/references/profile-priors.md",
            "sam3_demo/layout/skills/layout-mask-preprocess/references/response-schema.json",
        }
        listing = Path(__file__).resolve().parents[1] / bundle.FILE_LIST
        listed = {line for line in listing.read_text().splitlines() if line and not line.startswith("#")}
        self.assertEqual(set(bundle.RUNTIME_DOCS), expected)
        self.assertTrue(expected.issubset(bundle.REQUIRED))
        self.assertTrue(expected.issubset(listed))
        self.assertTrue(all(bundle.selected(name) for name in expected))
        self.assertFalse(bundle.selected("sam3_demo/layout/skills/layout-mask-preprocess/references/extra.md"))

        bundle.build(self.root, self.archive)
        manifest = bundle.verify(self.archive)
        self.assertTrue(expected.issubset(manifest["files"]))
        with zipfile.ZipFile(self.archive) as archive:
            self.assertTrue(expected.issubset(archive.namelist()))

    def test_missing_runtime_document_is_rejected_during_build(self):
        missing = bundle.RUNTIME_DOCS[0]
        listing = self.root / bundle.FILE_LIST
        listing.write_text("\n".join(line for line in listing.read_text().splitlines() if line != missing) + "\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "missing required runtime files"):
            bundle.build(self.root, self.archive)

    def test_verify_rejects_archive_missing_runtime_document(self):
        bundle.build(self.root, self.archive)
        missing = bundle.RUNTIME_DOCS[0]
        incomplete = self.base / "incomplete.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(incomplete, "w") as target:
            for item in source.infolist():
                if item.filename == missing:
                    continue
                data = source.read(item)
                if item.filename == bundle.MANIFEST:
                    manifest = json.loads(data)
                    manifest["files"].pop(missing)
                    data = json.dumps(manifest).encode()
                target.writestr(item, data)
        with self.assertRaisesRegex(ValueError, "missing runtime files"):
            bundle.verify(incomplete)

    def test_arbitrary_markdown_is_rejected(self):
        arbitrary = "sam3_demo/layout/skills/layout-mask-preprocess/references/extra.md"
        self.assertFalse(bundle.selected(arbitrary))
        with (self.root / bundle.FILE_LIST).open("a") as target:
            target.write(arbitrary + "\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Non-production"):
            bundle.build(self.root, self.archive)

    def test_packaged_real_skill_fixture_loads_without_external_api(self):
        source_root = Path(__file__).resolve().parents[1]
        for name in bundle.RUNTIME_DOCS:
            target = self.root / name
            target.write_bytes((source_root / name).read_bytes())
        self.commit()
        bundle.build(self.root, self.archive)
        extracted = self.base / "extracted"
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(extracted)

        sys.path.insert(0, str(source_root))
        try:
            from sam3_demo.layout.agent_runtime import load_skill_bundle
        finally:
            sys.path.remove(str(source_root))
        text, _, response_schema = load_skill_bundle(
            extracted / "sam3_demo/layout/skills/layout-mask-preprocess"
        )
        self.assertEqual(response_schema["properties"]["schema_version"]["const"], 4)
        for name in bundle.RUNTIME_DOCS:
            relative_name = name.split("layout-mask-preprocess/", 1)[1]
            self.assertIn(f"## {relative_name}\n", text)

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
