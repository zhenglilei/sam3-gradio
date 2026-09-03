from __future__ import annotations

import unittest
from pathlib import Path


FRONTEND_SOURCE = Path(__file__).resolve().parents[1] / "frontend" / "Index.svelte"


class StitchPreviewCanvasFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = FRONTEND_SOURCE.read_text(encoding="utf-8")

    def test_cancel_events_do_not_use_pointerup_commit_path(self) -> None:
        self.assertIn('onpointerup={onPointerUp}', self.source)
        self.assertIn('onpointercancel={onPointerCancel}', self.source)
        self.assertIn('function onPointerCancel(): void', self.source)
        self.assertIn('cancelPointerInteraction();', self.source)
        self.assertIn('function onWindowBlur(): void', self.source)
        self.assertIn('window.addEventListener("blur", onWindowBlur)', self.source)

    def test_loupe_uses_world_space_center_after_pan_and_zoom(self) -> None:
        loupe = self.source[
            self.source.index("function drawLoupe") : self.source.index("function drawSelectionOverlay")
        ]
        self.assertIn('const centerWorld = screenToWorld(cx, cy);', loupe)
        scale = loupe.index('ctx.scale(LOUPE_SCALE * viewZoom, LOUPE_SCALE * viewZoom);')
        translate = loupe.index('ctx.translate(-centerWorld.x, -centerWorld.y);')
        self.assertLess(scale, translate)
        self.assertNotIn('const srcRadiusWorld', self.source)
        self.assertNotIn(
            'ctx.translate(-centerWorld.x * viewZoom - viewPanX / LOUPE_SCALE', loupe
        )

    def test_pointer_move_only_publishes_local_value_until_pointerup(self) -> None:
        pointer_move = self.source[
            self.source.index("function onPointerMove") : self.source.index("function onPointerUp")
        ]
        drag_branch = pointer_move[
            pointer_move.index("if (pointerIdActive === evt.pointerId") :
        ]
        self.assertIn("publishClientValue(statusText);", drag_branch)
        self.assertNotIn("queueSyncValue(", drag_branch)
        self.assertNotIn('gradio.dispatch("change")', drag_branch)

    def test_default_drag_is_one_x_and_shift_is_explicit_precision_mode(self) -> None:
        pointer_move = self.source[
            self.source.index("function onPointerMove") : self.source.index("function onPointerUp")
        ]
        self.assertIn("const DEFAULT_DRAG_GAIN = 1.0;", self.source)
        self.assertIn("const PRECISE_DRAG_GAIN = 0.25;", self.source)
        self.assertIn(
            "const gain = evt.shiftKey ? preciseDragGain() : dragGain();",
            pointer_move,
        )
        self.assertIn("((screen.x - pointerLastX) / viewZoom) * gain", pointer_move)
        self.assertIn("((screen.y - pointerLastY) / viewZoom) * gain", pointer_move)
        self.assertNotIn("? 1.0 : dragGain()", pointer_move)

    def test_escape_uses_the_same_complete_cancel_path(self) -> None:
        keydown = self.source[
            self.source.index("function onKeyDown") : self.source.index("function onKeyUp")
        ]
        self.assertIn('cancelPointerInteraction("已取消拖动");', keydown)

    def test_rotation_handle_is_connected_to_pointer_lifecycle(self) -> None:
        pointer_down = self.source[
            self.source.index("function onPointerDown") :
            self.source.index("function onPointerMove")
        ]
        pointer_move = self.source[
            self.source.index("function onPointerMove") :
            self.source.index("function onPointerUp")
        ]
        pointer_up = self.source[
            self.source.index("function onPointerUp") :
            self.source.index("function onPointerCancel")
        ]
        self.assertIn("const rotateHit = hitRotateHandle(world.x, world.y);", pointer_down)
        self.assertIn("rotateStartAngle =", pointer_down)
        self.assertIn("if (pointerIdActive === evt.pointerId && rotateTileIndex >= 0)", pointer_move)
        self.assertIn("runtime.tile.rotation_deg = normalizeRotation(", pointer_move)
        self.assertIn("publishClientValue(statusText);", pointer_move)
        self.assertNotIn('gradio.dispatch("change")', pointer_move)
        self.assertIn("if (pointerIdActive === evt.pointerId && rotateTileIndex >= 0)", pointer_up)
        self.assertIn("syncValue(", pointer_up)

    def test_rotation_uses_center_clockwise_geometry_everywhere(self) -> None:
        self.assertIn("function tileCenter(", self.source)
        self.assertIn("function tileCorners(", self.source)
        self.assertIn("ctx.rotate((tile.rotation_deg * Math.PI) / 180);", self.source)
        self.assertIn("pointInTile(tile", self.source)
        self.assertIn("rotationHandleForTile(tile)", self.source)

    def test_debug_hud_is_not_rendered(self) -> None:
        self.assertNotIn("function drawHud", self.source)
        self.assertNotIn('ctx.fillRect(10, 10, boxW, boxH);', self.source)
        self.assertIn("function drawSelectionOverlay", self.source)

    def test_left_drag_on_background_pans_without_syncing_tile_value(self) -> None:
        pointer_down = self.source[
            self.source.index("function onPointerDown") :
            self.source.index("function onPointerMove")
        ]
        background_branch = pointer_down[pointer_down.index("} else {") :]
        self.assertIn("panning = true;", background_branch)
        self.assertIn("pointerIdActive = evt.pointerId;", background_branch)
        self.assertIn('cursorStyle = "grabbing";', background_branch)
        self.assertIn("canvasEl.setPointerCapture(evt.pointerId);", background_branch)

        pointer_up = self.source[
            self.source.index("function onPointerUp") :
            self.source.index("function onPointerCancel")
        ]
        pan_branch = pointer_up[: pointer_up.index("if (pointerIdActive === evt.pointerId")]
        self.assertIn("if (panning)", pan_branch)
        self.assertNotIn("syncValue(", pan_branch)
        self.assertNotIn("publishClientValue(", pan_branch)

    def test_rotation_is_undoable_and_cancel_restores_origin(self) -> None:
        self.assertIn(
            "tiles: { index: number; x: number; y: number; rotation_deg: number }[];",
            self.source,
        )
        cancel = self.source[
            self.source.index("function cancelPointerInteraction") :
            self.source.index("function onWindowBlur")
        ]
        self.assertIn("rotateRuntime.tile.rotation_deg = rotateOrigin;", cancel)
        self.assertIn(
            "runtime.tile.rotation_deg = normalizeRotation(item.rotation_deg);",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
