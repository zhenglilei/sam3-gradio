from __future__ import annotations

import sys
import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1] / "repair_mask_editor"
INDEX_SOURCE = COMPONENT_ROOT / "frontend" / "Index.svelte"
TEMPLATE_INDEX = (
    COMPONENT_ROOT
    / "backend"
    / "gradio_repair_mask_editor"
    / "templates"
    / "component"
    / "index.js"
)

sys.path.insert(0, str(COMPONENT_ROOT / "backend"))
from gradio_repair_mask_editor import RepairMaskEditor


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


class RepairMaskEditorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = INDEX_SOURCE.read_text(encoding="utf-8")

    def test_canvas_is_native_2d_and_forbids_gpu_editors(self) -> None:
        self.assertIn('getContext("2d"', self.source)
        self.assertIn("<canvas", self.source)
        for forbidden in (
            "ImageEditor",
            "Pixi",
            "WebGL",
            "WebGPU",
            'getContext("webgl',
        ):
            self.assertNotIn(forbidden, self.source)

    def test_source_space_and_contain_resize_contract(self) -> None:
        self.assertIn("function containRect(", self.source)
        self.assertIn("Math.min(width / size.width, height / size.height)", self.source)
        self.assertIn("source_width", self.source)
        self.assertIn("source_height", self.source)
        self.assertIn("maskCanvas", self.source)
        self.assertIn("tintCanvas", self.source)
        self.assertIn("new ResizeObserver", self.source)
        self.assertIn("function resizeDisplayCanvas(", self.source)
        self.assertIn("context.drawImage(tintCanvas", self.source)

    def test_pointermove_stays_local_and_pointerup_commits(self) -> None:
        pointer_move = _function_body(self.source, "onPointerMove")
        pointer_up = _function_body(self.source, "onPointerUp")
        self.assertNotIn("gradio.dispatch", pointer_move)
        self.assertNotIn("gradio.props.value", pointer_move)
        self.assertIn("drawStroke", pointer_move)
        self.assertIn("commitValue()", pointer_up)
        self.assertIn('gradio.dispatch("change")', self.source)

    def test_internal_toolbar_is_absent(self) -> None:
        for fragment in ("toolbar", "Brush", "Eraser", "Add rectangle", "Delete rectangle", "Commit", "range-control"):
            self.assertNotIn(fragment, self.source)

    def test_tools_and_png_payload_are_present(self) -> None:
        for tool in ("brush", "eraser", "rect_add", "rect_erase"):
            self.assertIn(tool, self.source)
        self.assertIn(
            'mask_png: maskCanvas ? maskCanvas.toDataURL("image/png") : null',
            self.source,
        )
        self.assertIn("preview_alpha", self.source)
        self.assertIn("brush_size", self.source)
        self.assertIn("image_id", self.source)
        self.assertIn("revision", self.source)

    def test_narrow_layout_has_no_horizontal_overflow(self) -> None:
        for fragment in (
            "min-width: 0",
            "max-width: 100%",
            "overflow: hidden",
        ):
            self.assertIn(fragment, self.source)

    def test_backend_sanitizes_the_value_contract(self) -> None:
        component = RepairMaskEditor()
        sanitized = component.preprocess(
            {
                "image_id": "img-1",
                "revision": 3.8,
                "source_width": 640,
                "source_height": 480,
                "base_image": "data:image/png;base64,base",
                "mask_png": "data:image/png;base64,mask",
                "preview_alpha": 2,
                "tool": "rect_erase",
                "brush_size": 0,
                "status": "ready",
                "unexpected": "drop",
            }
        )
        self.assertEqual(sanitized["image_id"], "img-1")
        self.assertEqual(sanitized["revision"], 3)
        self.assertEqual(sanitized["source_width"], 640)
        self.assertEqual(sanitized["source_height"], 480)
        self.assertEqual(sanitized["mask_png"], "data:image/png;base64,mask")
        self.assertEqual(sanitized["preview_alpha"], 1.0)
        self.assertEqual(sanitized["tool"], "rect_erase")
        self.assertNotIn("unexpected", sanitized)
        self.assertNotIn("brush_size", sanitized)

    def test_built_component_template_is_present(self) -> None:
        self.assertTrue(
            TEMPLATE_INDEX.exists(),
            "run the component frontend build before shipping",
        )


if __name__ == "__main__":
    unittest.main()
