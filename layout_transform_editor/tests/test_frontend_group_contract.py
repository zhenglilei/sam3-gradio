from __future__ import annotations

import re
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
INDEX_SOURCE = COMPONENT_ROOT / "frontend" / "Index.svelte"
TYPES_SOURCE = COMPONENT_ROOT / "frontend" / "types.ts"


def _function_body(source: str, function_name: str) -> str:
    marker = f"function {function_name}("
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"unterminated function: {function_name}")


class LayoutTransformEditorGroupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_source = INDEX_SOURCE.read_text(encoding="utf-8")
        cls.types_source = TYPES_SOURCE.read_text(encoding="utf-8")

    def test_group_payload_types_are_declared(self) -> None:
        for field in (
            "transform_mode?: string",
            "group_view?: LayoutTransformGroupServerView",
            "group_intent?: LayoutTransformGroupIntent",
            "selection_signature: string",
            "transform_set_revision: number",
            "active_group_id: LayoutTransformGroupId | null",
            "changed_group_ids?: LayoutTransformGroupId[]",
        ):
            self.assertIn(field, self.types_source)

    def test_pointer_move_never_dispatches_python_change(self) -> None:
        body = _function_body(self.index_source, "onPointerMove")
        self.assertNotIn('dispatch("change")', body)
        self.assertNotIn("syncValue(", body)
        self.assertNotIn("publishGroupIntent(", body)

    def test_pointer_up_commits_through_single_sync_path(self) -> None:
        body = _function_body(self.index_source, "onPointerUp")
        self.assertEqual(body.count("syncValue("), 1)
        self.assertNotIn('dispatch("change")', body)

    def test_group_sync_only_replaces_group_intent(self) -> None:
        body = _function_body(self.index_source, "publishGroupIntent")
        assignment = re.search(r"localValue\s*=\s*\{(?P<body>.*?)\n\s*\};", body, re.DOTALL)
        self.assertIsNotNone(assignment)
        assigned_body = assignment.group("body")
        self.assertIn("...localValue", assigned_body)
        self.assertIn("group_intent:", assigned_body)
        self.assertNotIn("group_view:", assigned_body)
        self.assertNotIn("transform:", assigned_body.split("group_intent:", 1)[0])

    def test_dirty_groups_survive_active_switch_and_debounced_wheels(self) -> None:
        bump_body = _function_body(
            self.index_source,
            "bumpActiveTransformRevision",
        )
        self.assertIn("dirtyGroupKeys.add(changedGroupKey)", bump_body)
        self.assertNotIn("dirtyGroupKeys.clear", bump_body)
        self.assertNotIn("dirtyGroupKeys = new Set", bump_body)

        publish_body = _function_body(
            self.index_source,
            "publishGroupIntent",
        )
        self.assertIn("dirtyGroupKeys.has(groupKey(group.group_id))", publish_body)
        self.assertIn("changed_group_ids: changedGroupIds", publish_body)
        self.assertNotIn("dirtyGroupKeys.clear", publish_body)
        self.assertNotIn("dirtyGroupKeys = new Set", publish_body)

        switch_body = _function_body(
            self.index_source,
            "setActiveGroupLocal",
        )
        self.assertNotIn("dirtyGroupKeys.clear", switch_body)
        self.assertNotIn("dirtyGroupKeys = new Set", switch_body)

    def test_publish_uses_small_client_intent_without_server_images(self) -> None:
        body = _function_body(self.index_source, "publishClientIntent")
        for field in (
            "enabled:",
            "transform:",
            "target_width:",
            "target_height:",
            'transform_mode = "label_groups"',
            "outbound.group_intent",
        ):
            self.assertIn(field, body)
        for server_only_field in (
            "base_image:",
            "mask_image:",
            "group_view:",
            "source_width:",
            "source_height:",
            "foreground_bbox_xyxy:",
            "status:",
            "data:image",
        ):
            self.assertNotIn(server_only_field, body)
        self.assertIn("gradio.props.value = outbound", body)

        group_body = _function_body(
            self.index_source,
            "publishGroupIntent",
        )
        self.assertIn("publishClientIntent()", group_body)
        self.assertNotIn("gradio.props.value", group_body)

    def test_toolbar_names_internal_reset_center_normalize(self) -> None:
        self.assertIn(
            ">居中归一</button>",
            self.index_source,
        )
        reset_body = _function_body(
            self.index_source,
            "resetTransform",
        )
        self.assertIn('"居中归一 "', reset_body)
        self.assertNotIn('"重置 "', reset_body)

    def test_async_image_loads_are_generation_guarded(self) -> None:
        load_body = _function_body(self.index_source, "loadImage")
        self.assertGreaterEqual(load_body.count("generation === imageLoadGeneration"), 3)
        ingest_body = _function_body(self.index_source, "ingestValue")
        self.assertIn("imageLoadGeneration += 1", ingest_body)


if __name__ == "__main__":
    unittest.main()
