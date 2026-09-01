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
            self.source.index("function drawLoupe") : self.source.index("function drawHud")
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


if __name__ == "__main__":
    unittest.main()
