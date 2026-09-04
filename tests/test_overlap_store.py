"""Focused, dependency-free tests for the shared overlap layout store."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

from sam3_demo.overlap_store import (
    ConflictError,
    MAX_LAYERS,
    MAX_PNG_BYTES,
    NotFoundError,
    OverlapLayoutStore,
    ValidationError,
)


PNG = b"\x89PNG\r\n\x1a\nfixture"


def make_transform(**overrides):
    transform = {
        "transform_version": 2,
        "center_x": 12.0,
        "center_y": 24.0,
        "pivot_x": 3.0,
        "pivot_y": 4.0,
        "scale": 1.0,
        "rotation_deg": 15.0,
        "preview_alpha": 0.35,
        "matrix_2x3": [[1.0, 0.0, 12.0], [0.0, 1.0, 24.0]],
    }
    transform.update(overrides)
    return transform


def make_layer(name="mask", layer_type="BSM", ordinal=None, **overrides):
    layer = {
        "name": name,
        "layer_type": layer_type,
        "mask_png": PNG,
        "mask_width": 10,
        "mask_height": 20,
        "transform": make_transform(),
    }
    if ordinal is not None:
        layer["ordinal"] = ordinal
    layer.update(overrides)
    return layer


class OverlapStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "overlap.sqlite3"
        self.store = OverlapLayoutStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def create_default(self, *, name="  board  ", layers=None):
        if layers is None:
            layers = [
                make_layer("act", "ACT"),
                make_layer("bsm", "BSM"),
                make_layer("ge2", "GE2"),
            ]
        return self.store.create_layout(name, PNG, 100, 200, layers)

    def test_crud_and_snapshot_isolation(self):
        created = self.create_default()
        self.assertTrue(uuid.UUID(created["id"]))
        self.assertEqual(created["name"], "board")
        self.assertEqual(created["board_type"], "O3")
        self.assertEqual(created["revision"], 1)
        self.assertEqual(created["base_image_png"], PNG)
        self.assertEqual(
            [layer["layer_type"] for layer in created["layers"]],
            ["ACT", "BSM", "GE2"],
        )
        self.assertEqual([layer["ordinal"] for layer in created["layers"]], [0, 1, 2])
        self.assertEqual(
            created["layers"][0]["mask_sha256"], hashlib.sha256(PNG).hexdigest()
        )
        self.assertNotIn("matrix_2x3", created["layers"][0]["transform"])

        # Returned values are fresh snapshots and list summaries never contain
        # image/mask BLOBs.
        created["layers"][0]["transform"]["scale"] = 99
        created["layers"].reverse()
        created["base_image_png"] = b"changed"
        fetched = self.store.get_layout(created["id"])
        self.assertEqual(fetched["base_image_png"], PNG)
        self.assertEqual(fetched["layers"][0]["transform"]["scale"], 1.0)
        self.assertEqual([x["ordinal"] for x in fetched["layers"]], [0, 1, 2])

        summaries = self.store.list_layouts()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["id"], fetched["id"])
        self.assertNotIn("base_image_png", summaries[0])
        self.assertNotIn("layers", summaries[0])
        self.assertNotIn("mask_png", summaries[0])

        updated = self.store.update_layout(
            fetched["id"],
            fetched["revision"],
            name=" updated ",
            base_image_png=PNG + b"-new",
            width=300,
            height=400,
            layers=[
                dict(fetched["layers"][2], name="last", ordinal=0),
                dict(fetched["layers"][0], name="first", ordinal=1),
            ],
        )
        self.assertEqual(updated["name"], "updated")
        self.assertEqual(updated["revision"], 2)
        self.assertEqual([x["ordinal"] for x in updated["layers"]], [0, 1])
        self.assertEqual([x["name"] for x in updated["layers"]], ["last", "first"])
        self.assertEqual(
            {x["id"] for x in updated["layers"]},
            {fetched["layers"][0]["id"], fetched["layers"][2]["id"]},
        )

    def test_layer_ordinals_and_arbitrary_type_order(self):
        # Explicit ordinals determine canonical order, and the type sequence
        # is intentionally not constrained to the UI's default order.
        snapshot = self.create_default(
            layers=[
                make_layer("ge1", "GE1", ordinal=2),
                make_layer("bsm", "BSM", ordinal=0),
                make_layer("act", "ACT", ordinal=1),
            ]
        )
        self.assertEqual(
            [layer["layer_type"] for layer in snapshot["layers"]],
            ["BSM", "ACT", "GE1"],
        )
        with self.assertRaises(ValidationError):
            self.create_default(
                layers=[make_layer("a", ordinal=0), make_layer("b", ordinal=2)]
            )
        with self.assertRaises(ValidationError):
            self.create_default(
                layers=[make_layer("a", ordinal=0), make_layer("b")]
            )
        with self.assertRaises(ValidationError):
            self.create_default(layers=[])
        with self.assertRaises(ValidationError):
            self.create_default(
                layers=[make_layer(str(index)) for index in range(MAX_LAYERS + 1)]
            )
        with self.assertRaises(ValidationError):
            self.create_default(layers=[make_layer("bad", "OTHER")])

    def test_soft_delete_and_visibility(self):
        snapshot = self.create_default()
        deleted = self.store.delete_layout(snapshot["id"], snapshot["revision"])
        self.assertEqual(deleted["revision"], 2)
        self.assertIsNotNone(deleted["deleted_at"])
        self.assertEqual(self.store.list_layouts(), [])
        self.assertEqual(len(self.store.list_layouts(include_deleted=True)), 1)
        with self.assertRaises(NotFoundError):
            self.store.get_layout(snapshot["id"])
        self.assertEqual(
            self.store.get_layout(snapshot["id"], include_deleted=True)["id"],
            snapshot["id"],
        )
        with self.assertRaises(NotFoundError):
            self.store.delete_layout(snapshot["id"], deleted["revision"])
        with self.assertRaises(NotFoundError):
            self.store.update_layout(
                snapshot["id"],
                deleted["revision"],
                name="no",
                base_image_png=PNG,
                width=1,
                height=1,
                layers=[make_layer()],
            )

    def test_stale_revision_rejected_without_write(self):
        original = self.create_default()
        current = self.store.update_layout(
            original["id"],
            original["revision"],
            name="current",
            base_image_png=PNG,
            width=100,
            height=200,
            layers=original["layers"],
        )
        with self.assertRaises(ConflictError) as context:
            self.store.update_layout(
                original["id"],
                original["revision"],
                name="stale",
                base_image_png=PNG + b"stale",
                width=999,
                height=999,
                layers=[make_layer("replacement", "GE2")],
            )
        self.assertEqual(context.exception.expected_revision, 1)
        self.assertEqual(context.exception.current_revision, 2)
        preserved = self.store.get_layout(original["id"])
        self.assertEqual(preserved["name"], current["name"])
        self.assertEqual(preserved["width"], current["width"])
        self.assertEqual(len(preserved["layers"]), len(current["layers"]))

        with self.assertRaises(ConflictError):
            self.store.delete_layout(original["id"], original["revision"])
        self.assertIsNone(self.store.get_layout(original["id"])["deleted_at"])

    def test_invalid_transform_nan_and_scale(self):
        invalid_values = [
            {"transform_version": 1},
            dict(make_transform(), transform_version=2.0),
            dict(make_transform(), center_x=float("nan")),
            dict(make_transform(), rotation_deg=float("inf")),
            dict(make_transform(), scale=0),
            dict(make_transform(), scale=-1),
            dict(make_transform(), preview_alpha=float("-inf")),
        ]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValidationError):
                    self.create_default(layers=[make_layer(transform=invalid)])

        encoded = json.dumps(make_transform())
        with self.assertRaises(ValidationError):
            self.create_default(layers=[make_layer(transform=encoded[:-1])])

    def test_invalid_png_dimensions_names_and_hash(self):
        checks = [
            {"base_image_png": b"not-png"},
            {"base_image_png": b""},
            {"width": 0},
            {"height": -1},
            {"name": " "},
            {"name": "x" * 129},
        ]
        for overrides in checks:
            with self.subTest(overrides=overrides):
                kwargs = {
                    "name": "ok",
                    "base_image_png": PNG,
                    "width": 1,
                    "height": 1,
                    "layers": [make_layer()],
                }
                kwargs.update(overrides)
                with self.assertRaises(ValidationError):
                    self.store.create_layout(**kwargs)

        for overrides in (
            {"mask_png": b"bad"},
            {"mask_png": b""},
            {"mask_width": 0},
            {"mask_height": -1},
            {"mask_sha256": "z" * 64},
            {"mask_sha256": "a" * 63},
            {"mask_sha256": "0" * 64},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValidationError):
                    self.create_default(layers=[make_layer(**overrides)])

        with self.assertRaises(ValidationError):
            self.create_default(layers=[make_layer(mask_png=b"\x89PNG")])

        too_large = PNG + b"x" * (MAX_PNG_BYTES - len(PNG) + 1)
        with self.assertRaises(ValidationError):
            self.create_default(layers=[make_layer(mask_png=too_large)])

    def test_update_failure_preserves_old_snapshot(self):
        original = self.create_default()
        with self.assertRaises(ValidationError):
            self.store.update_layout(
                original["id"],
                original["revision"],
                name="new",
                base_image_png=PNG + b"new",
                width=999,
                height=888,
                layers=[make_layer("invalid", transform=make_transform(scale=0))],
            )
        current = self.store.get_layout(original["id"])
        self.assertEqual(current["name"], original["name"])
        self.assertEqual(current["base_image_png"], original["base_image_png"])
        self.assertEqual(current["width"], original["width"])
        self.assertEqual(current["revision"], original["revision"])
        self.assertEqual(current["layers"], original["layers"])

    def test_two_store_instances_only_one_stale_write_succeeds(self):
        original = self.create_default()
        store_a = OverlapLayoutStore(self.db_path)
        store_b = OverlapLayoutStore(self.db_path)
        barrier = threading.Barrier(2)
        outcomes = []
        outcome_lock = threading.Lock()

        def writer(store, name):
            barrier.wait()
            try:
                result = store.update_layout(
                    original["id"],
                    1,
                    name=name,
                    base_image_png=PNG,
                    width=100,
                    height=200,
                    layers=original["layers"],
                )
                outcome = ("ok", result["name"])
            except ConflictError:
                outcome = ("conflict", None)
            with outcome_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=writer, args=(store_a, "writer-a")),
            threading.Thread(target=writer, args=(store_b, "writer-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        store_a.close()
        store_b.close()

        self.assertEqual(sorted(result[0] for result in outcomes), ["conflict", "ok"])
        final = self.store.get_layout(original["id"])
        self.assertEqual(final["revision"], 2)
        self.assertIn(final["name"], {"writer-a", "writer-b"})

    def test_schema_reopen_and_connection_pragmas(self):
        original = self.create_default()
        self.store.close()
        reopened = OverlapLayoutStore(self.db_path)
        self.store = reopened
        self.assertEqual(reopened.get_layout(original["id"]), original)
        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 0)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(reopened._conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(
            reopened._conn.execute("PRAGMA busy_timeout").fetchone()[0],
            30_000,
        )
        self.assertEqual(
            reopened._conn.execute("SELECT COUNT(*) FROM layers").fetchone()[0], 3
        )
        self.assertEqual(
            reopened._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'layers'"
            ).fetchone()[0].count("ON DELETE CASCADE"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
