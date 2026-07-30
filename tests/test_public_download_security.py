import ast
import os
import re
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from public_download_utils import (
    PUBLIC_DOWNLOAD_CATEGORIES,
    PublicDownloadError,
    ensure_public_download_dirs,
    prune_public_downloads,
    publish_files,
    publish_zip,
)

ROOT = Path(__file__).resolve().parents[1]


class PublicDownloadSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "public_downloads"
        self.internal = Path(self.temp_dir.name) / "internal"
        self.internal.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_random_export_dir(self, path, category):
        self.assertEqual(path.parent, self.root.resolve() / category)
        self.assertRegex(path.name, re.compile(r"^[0-9a-f]{32}$"))

    def test_ensure_public_download_dirs_creates_only_fixed_categories(self):
        public_root = ensure_public_download_dirs(self.root)

        self.assertEqual(public_root, self.root.resolve())
        self.assertEqual(
            {path.name for path in public_root.iterdir()},
            set(PUBLIC_DOWNLOAD_CATEGORIES),
        )
        self.assertTrue(all(path.is_dir() for path in public_root.iterdir()))

    def test_publish_files_uses_random_export_dir_and_fixed_public_names(self):
        mask = self.internal / "private-mask-name.png"
        contour = self.internal / "private-contours-name.json"
        mask.write_bytes(b"mask-v1")
        contour.write_text('{"contours": []}', encoding="utf-8")

        export_dir = publish_files(
            self.root,
            "layout_mask_exports",
            {"mask.png": mask, "contours.json": contour},
        )

        self.assert_random_export_dir(export_dir, "layout_mask_exports")
        self.assertEqual((export_dir / "mask.png").read_bytes(), b"mask-v1")
        self.assertEqual(
            (export_dir / "contours.json").read_text(encoding="utf-8"),
            '{"contours": []}',
        )
        mask.write_bytes(b"mask-v2")
        self.assertEqual((export_dir / "mask.png").read_bytes(), b"mask-v1")

        second_export = publish_files(
            self.root,
            "layout_mask_exports",
            [mask, contour],
        )
        self.assert_random_export_dir(second_export, "layout_mask_exports")
        self.assertNotEqual(export_dir, second_export)
        self.assertEqual((second_export / mask.name).read_bytes(), b"mask-v2")

    def test_publish_files_rejects_unknown_category_names_and_symlinks(self):
        source = self.internal / "source.txt"
        source.write_text("private", encoding="utf-8")

        with self.assertRaises(PublicDownloadError):
            publish_files(self.root, "../layout_mask_exports", [source])
        with self.assertRaises(PublicDownloadError):
            publish_files(
                self.root,
                "layout_mask_exports",
                {"../source.txt": source},
            )

        symlink = self.internal / "source-link.txt"
        symlink.symlink_to(source)
        with self.assertRaises(PublicDownloadError):
            publish_files(self.root, "layout_mask_exports", [symlink])

        category_dir = self.root / "layout_mask_exports"
        self.assertEqual(list(category_dir.iterdir()), [])

    def test_publish_zip_copies_nested_regular_files_only(self):
        staging = self.internal / "region-export"
        nested = staging / "metadata"
        nested.mkdir(parents=True)
        (staging / "regions.json").write_text('{"regions": []}', encoding="utf-8")
        (staging / "source_mask.png").write_bytes(b"png")
        (nested / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")
        expected_date_time = (2026, 7, 30, 12, 34, 56)
        modified_at = time.mktime((*expected_date_time, 0, 0, -1))
        for source in (
            staging / "regions.json",
            staging / "source_mask.png",
            nested / "manifest.json",
        ):
            os.utime(source, (modified_at, modified_at))

        archive_path = publish_zip(
            self.root,
            "region_annotation_exports",
            staging,
            "region_annotations.zip",
        )

        self.assert_random_export_dir(
            archive_path.parent,
            "region_annotation_exports",
        )
        self.assertEqual(archive_path.name, "region_annotations.zip")
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                ["metadata/manifest.json", "regions.json", "source_mask.png"],
            )
            self.assertEqual(archive.read("regions.json"), b'{"regions": []}')
            self.assertEqual(
                {item.date_time for item in archive.infolist()},
                {expected_date_time},
            )

    def test_publish_zip_clamps_pre_1980_member_timestamps(self):
        staging = self.internal / "legacy-export"
        staging.mkdir()
        source = staging / "result.json"
        source.write_text("{}", encoding="utf-8")
        os.utime(source, (0, 0))

        archive_path = publish_zip(
            self.root,
            "pcs_pvs_exports",
            staging,
            "legacy.zip",
        )

        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(
                archive.getinfo("result.json").date_time,
                (1980, 1, 1, 0, 0, 0),
            )

    def test_publish_zip_rejects_symlinks_and_cross_platform_traversal_names(self):
        outside = self.internal / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        staging = self.internal / "staging"
        staging.mkdir()
        (staging / "safe.txt").write_text("safe", encoding="utf-8")
        (staging / "escape.txt").symlink_to(outside)

        with self.assertRaises(PublicDownloadError):
            publish_zip(self.root, "pcs_pvs_exports", staging)
        category_dir = self.root / "pcs_pvs_exports"
        self.assertTrue(not category_dir.exists() or not any(category_dir.iterdir()))

        (staging / "escape.txt").unlink()
        unsafe_name = staging / "..\\escape.txt"
        unsafe_name.write_text("unsafe", encoding="utf-8")
        with self.assertRaises(PublicDownloadError):
            publish_zip(self.root, "pcs_pvs_exports", staging)

    def test_public_root_and_zip_source_directories_must_not_be_symlinks(self):
        actual_root = Path(self.temp_dir.name) / "actual-public"
        actual_root.mkdir()
        root_link = Path(self.temp_dir.name) / "public-link"
        root_link.symlink_to(actual_root, target_is_directory=True)
        with self.assertRaises(PublicDownloadError):
            ensure_public_download_dirs(root_link)

        staging = self.internal / "staging"
        staging.mkdir()
        (staging / "result.json").write_text("{}", encoding="utf-8")
        staging_link = self.internal / "staging-link"
        staging_link.symlink_to(staging, target_is_directory=True)
        with self.assertRaises(PublicDownloadError):
            publish_zip(self.root, "pcs_pvs_exports", staging_link)

    def test_prune_removes_only_expired_category_entries(self):
        source = self.internal / "result.txt"
        source.write_text("result", encoding="utf-8")
        old_export = publish_files(self.root, "pcs_pvs_exports", [source])
        recent_export = publish_files(self.root, "pcs_pvs_exports", [source])
        old_timestamp = 1_000.0
        os.utime(old_export, (old_timestamp, old_timestamp))
        now = old_timestamp + 100.0

        removed = prune_public_downloads(
            self.root,
            max_age_seconds=50,
            now=now,
        )

        self.assertEqual(removed, [old_export])
        self.assertFalse(old_export.exists())
        self.assertTrue(recent_export.exists())
        self.assertTrue((self.root / "pcs_pvs_exports").is_dir())

    def test_prune_rejects_non_positive_ttl(self):
        for invalid_ttl in (0, -1, False):
            with self.subTest(invalid_ttl=invalid_ttl):
                with self.assertRaises(PublicDownloadError):
                    prune_public_downloads(self.root, invalid_ttl)


    def test_main_launch_only_allows_public_downloads_and_sets_blocklist(self):
        source = (ROOT / "sam3_demo" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        launch = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "launch"
        )
        keywords = {keyword.arg: keyword.value for keyword in launch.keywords}
        self.assertEqual(ast.literal_eval(keywords["debug"]), False)
        self.assertEqual(
            ast.unparse(keywords["allowed_paths"]),
            "_gradio_allowed_paths()",
        )
        self.assertEqual(
            ast.unparse(keywords["blocked_paths"]),
            "_gradio_blocked_paths()",
        )
        self.assertNotIn("allowed_paths=[str(current_dir)]", source)
        self.assertIn(
            "public_downloads/",
            (ROOT / ".gitignore").read_text(encoding="utf-8"),
        )

    def test_demo_configures_gradio_cache_ttl(self):
        source = (ROOT / "sam3_demo" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        create_demo = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_demo"
        )
        blocks_call = next(
            node
            for node in ast.walk(create_demo)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "gr"
            and node.func.attr == "Blocks"
        )
        keywords = {keyword.arg: keyword.value for keyword in blocks_call.keywords}
        self.assertEqual(
            ast.unparse(keywords["delete_cache"]),
            "(3600, _PUBLIC_DOWNLOAD_TTL_SECONDS)",
        )


if __name__ == "__main__":
    unittest.main()
