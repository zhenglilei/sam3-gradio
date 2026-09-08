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
            "allow_reorder?: boolean",
            "ordered_group_ids?: LayoutTransformGroupId[]",
        ):
            self.assertIn(field, self.types_source)

    def test_pointer_move_never_dispatches_python_change(self) -> None:
        body = _function_body(self.index_source, "onPointerMove")
        self.assertNotIn('dispatch("change")', body)
        self.assertNotIn("syncValue(", body)
        self.assertNotIn("publishGroupIntent(", body)

    def test_pointer_up_debounces_group_and_keeps_legacy_immediate(self) -> None:
        body = _function_body(self.index_source, "onPointerUp")
        self.assertEqual(body.count('queueSyncValue("canvas", status)'), 1)
        self.assertEqual(body.count('syncValue("canvas", status)'), 1)
        self.assertRegex(
            body,
            r"if \(isGroupMode\(\)\) \{\s*"
            r'queueSyncValue\("canvas", status\);\s*'
            r"\} else \{\s*"
            r'syncValue\("canvas", status\);\s*'
            r"\}",
        )
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

    def test_group_reorder_preserves_maps_and_publishes_order_once(self) -> None:
        publish_body = _function_body(self.index_source, "publishGroupIntent")
        self.assertIn("ordered_group_ids", publish_body)
        self.assertIn("canReorderGroups()", publish_body)
        self.assertIn("groupViews().map", publish_body)
        self.assertEqual(publish_body.count('gradio.dispatch("change")'), 0)
        self.assertIn("scheduleIntentDispatch(0)", publish_body)
        dispatch_body = _function_body(
            self.index_source,
            "dispatchLatestIntent",
        )
        self.assertEqual(dispatch_body.count('gradio.dispatch("change")'), 2)
        self.assertIn("syncInFlightRevision !== null", dispatch_body)

        drop_body = _function_body(self.index_source, "onGroupDrop")
        self.assertEqual(drop_body.count('publishGroupIntent("reorder_groups"'), 1)
        self.assertIn("false, true", drop_body)

        reorder_body = _function_body(self.index_source, "reorderGroupViews")
        for forbidden in (
            "dirtyGroupKeys =",
            "dirtyGroupKeys.clear",
            "groupTransformByKey = new Map",
            "groupRuntimeByKey = new Map",
        ):
            self.assertNotIn(forbidden, reorder_body)

        ingest_body = _function_body(self.index_source, "ingestGroupValue")
        self.assertIn("restoreGroupOrder", ingest_body)
        self.assertIn("ordered_group_ids", ingest_body)
        self.assertIn("allow_reorder === true", ingest_body)

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

    def test_wheel_is_precise_local_and_delta_mode_normalized(self) -> None:
        body = _function_body(self.index_source, "onWheel")
        self.assertIn("wheelPixels(evt)", body)
        self.assertIn("evt.shiftKey ? 0.1", body)
        self.assertIn("evt.altKey ? 5", body)
        self.assertIn("queueSyncValue(", body)
        self.assertNotIn('dispatch("change")', body)
        normalizer = _function_body(self.index_source, "wheelPixels")
        self.assertIn("WheelEvent.DOM_DELTA_LINE", normalizer)
        self.assertIn("WheelEvent.DOM_DELTA_PAGE", normalizer)
        self.assertIn("Math.log(1.01) / 100", self.index_source)

    def test_view_visibility_and_solo_are_browser_only(self) -> None:
        self.assertIn('let displayMode = $state<"align" | "overlap">("align")', self.index_source)
        visible = _function_body(self.index_source, "defaultGroupVisible")
        self.assertIn('displayMode === "overlap"', visible)
        for name in (
            "setDisplayMode",
            "toggleGroupVisibility",
            "toggleSoloGroup",
            "resetView",
            "zoomView",
        ):
            body = _function_body(self.index_source, name)
            self.assertNotIn('dispatch("change")', body)
            self.assertNotIn("syncValue(", body)
            self.assertNotIn("queueSyncValue(", body)
            self.assertNotIn("publishGroupIntent(", body)

    def test_canvas_pan_never_changes_layer_transform(self) -> None:
        begin = _function_body(self.index_source, "beginCanvasPan")
        self.assertIn("viewPanX", begin)
        self.assertNotIn("setActiveTransform", begin)
        move = _function_body(self.index_source, "onPointerMove")
        pan_block = move.split("if (panning)", 1)[1].split("const p =", 1)[0]
        self.assertIn("viewPanX", pan_block)
        self.assertIn("requestDraw()", pan_block)
        self.assertNotIn("setActiveTransform", pan_block)
        self.assertNotIn('dispatch("change")', pan_block)

    def test_numeric_editing_only_syncs_on_commit(self) -> None:
        input_body = _function_body(self.index_source, "onNumericInput")
        self.assertNotIn("syncValue(", input_body)
        self.assertNotIn("queueSyncValue(", input_body)
        commit_body = _function_body(self.index_source, "commitNumericField")
        self.assertIn('queueSyncValue("numeric"', commit_body)
        self.assertIn('evt.key === "Escape"', _function_body(self.index_source, "onNumericKeydown"))

    def test_overlap_preview_is_focused_and_patterned(self) -> None:
        draw_body = _function_body(self.index_source, "draw")
        self.assertIn("visibleGroupKeys()", draw_body)
        self.assertIn("preview_alpha ?? 0.22", draw_body)
        self.assertIn("? 0.10", draw_body)
        self.assertIn("canvasEl.width !== renderWidth", draw_body)
        self.assertIn("drawOverlapPreview", draw_body)
        overlap_body = _function_body(self.index_source, "drawOverlapPreview")
        self.assertIn('"destination-in"', overlap_body)
        self.assertIn('"source-atop"', overlap_body)
        self.assertIn("#ff3fa4", overlap_body)
        self.assertNotIn("StatusTracker", self.index_source)

    def test_async_group_mask_readiness_recomputes_controls(self) -> None:
        self.assertIn("let groupRuntimeRevision = $state(0)", self.index_source)
        self.assertIn("groupRuntimeRevision += 1", self.index_source)
        editable = _function_body(self.index_source, "hasEditableMask")
        self.assertIn("groupRuntimeRevision;", editable)
        self.assertIn("groupRuntimeByKey.get(activeGroupKey)?.ready", editable)

    def test_ack_requires_exact_session_epoch_revision_and_payload_hash(self) -> None:
        handler = _function_body(self.index_source, "handleServerAck")
        self.assertIn("ackSession !== editorSessionId", handler)
        self.assertIn("ackEpoch !== documentEpoch", handler)
        self.assertIn("ackRevision !== syncInFlightRevision", handler)
        self.assertIn("ackHash !== syncInFlightHash", handler)
        self.assertIn("ackRevision !== clientRevision", handler)
        self.assertIn("localValue.group_intent?.payload_hash", handler)
        dispatch = _function_body(self.index_source, "dispatchLatestIntent")
        self.assertIn("syncInFlightHash = String(intent.payload_hash", dispatch)

    def test_alignment_proposal_never_rehydrates_local_geometry(self) -> None:
        self.assertIn(
            'incoming?.payload_kind === "alignment_proposal"',
            self.index_source,
        )
        handler = _function_body(
            self.index_source,
            "handleAlignmentProposal",
        )
        self.assertIn("publishClientIntent()", handler)
        self.assertNotIn("ingestValue(", handler)
        self.assertNotIn('dispatch("change")', handler)

    def test_late_auto_alignment_respects_manual_touch_and_revision(self) -> None:
        apply_body = _function_body(
            self.index_source,
            "applyAlignmentCandidate",
        )
        self.assertIn("manuallyTouchedGroupKeys.has(key)", apply_body)
        self.assertIn("start_transform_revision", apply_body)
        self.assertIn("setActiveTransform({", apply_body)
        self.assertIn("}, false)", apply_body)
        self.assertIn('queueSyncValue(', apply_body)




if __name__ == "__main__":
    unittest.main()
