from __future__ import annotations

import unittest
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
INDEX_SOURCE = COMPONENT_ROOT / "frontend" / "Index.svelte"


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


class ImageGestureOverlayFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = INDEX_SOURCE.read_text(encoding="utf-8")

    def test_pointer_move_never_dispatches(self) -> None:
        body = _function_body(self.source, "onPointerMove")
        self.assertNotIn("gradio.dispatch", body)
        self.assertNotIn("publishGesture", body)

    def test_pointer_up_has_one_publish_path(self) -> None:
        body = _function_body(self.source, "onPointerUp")
        self.assertEqual(body.count("publishGesture("), 1)
        self.assertNotIn("gradio.dispatch", body)

    def test_cancel_and_blur_do_not_publish(self) -> None:
        for name in ("onPointerCancel", "onLostPointerCapture", "onWindowBlur"):
            body = _function_body(self.source, name)
            self.assertNotIn("publishGesture", body)
            self.assertNotIn("gradio.dispatch", body)
        self.assertIn("on:lostpointercapture={onLostPointerCapture}", self.source)

    def test_pointer_slop_uses_css_distance_and_pointer_type(self) -> None:
        body = _function_body(self.source, "onPointerUp")
        self.assertIn("clickDistanceCssPx(pointerType)", body)
        self.assertIn("minDragCssPx(pointerType)", body)
        self.assertIn('pointerType === "touch"', _function_body(self.source, "clickDistanceCssPx"))
        self.assertIn('pointerType === "touch"', _function_body(self.source, "minDragCssPx"))

    def test_target_measurement_uses_actual_image_and_contain_geometry(self) -> None:
        body = _function_body(self.source, "measureTarget")
        self.assertIn("targetImage.getBoundingClientRect()", body)
        self.assertIn("Math.min", body)
        self.assertIn("objectPosition", body)
        self.assertIn("ResizeObserver", self.source)
        self.assertIn('window.addEventListener("scroll", measureTarget, true)', self.source)
        self.assertIn('window.visualViewport?.addEventListener("resize", measureTarget)', self.source)
        self.assertIn('window.visualViewport?.removeEventListener("scroll", measureTarget)', self.source)
        bind_body = _function_body(self.source, "bindTargetImage")
        self.assertIn("if (!targetImage)", bind_body)
        self.assertIn("targetReady = false", bind_body)

    def test_image_identity_is_separate_from_interaction_revision(self) -> None:
        image_body = _function_body(self.source, "imageIdentitySignature")
        for field in ("view.image_id", "view.image_sha256", "naturalWidth()", "naturalHeight()"):
            self.assertIn(field, image_body)
        self.assertNotIn("view.revision", image_body)
        self.assertNotIn("interaction()", image_body)

        view_body = _function_body(self.source, "viewIdentitySignature")
        self.assertIn("view.revision", view_body)
        self.assertIn("interaction()", view_body)

    def test_image_change_waits_for_new_loaded_dom_source(self) -> None:
        ingest_body = _function_body(self.source, "ingestValue")
        self.assertIn("if (imageChanged) bindTargetImage()", ingest_body)
        transition_body = _function_body(self.source, "beginImageTransition")
        self.assertIn("targetReady = false", transition_body)
        self.assertIn("waitingPreviousSource = previousSource", transition_body)

        activation_body = _function_body(self.source, "activateTargetImage")
        self.assertIn("await image.decode()", activation_body)
        self.assertIn("generation !== targetImageGeneration", activation_body)
        self.assertIn("identity !== loadedImageIdentity", activation_body)
        self.assertIn("source !== targetSource(image)", activation_body)

        request_body = _function_body(self.source, "requestTargetActivation")
        self.assertIn("source === waitingPreviousSource", request_body)
        self.assertIn("targetReady = false", request_body)

    def test_same_identity_new_source_is_decoded_and_rebound(self) -> None:
        body = _function_body(self.source, "requestTargetActivation")
        self.assertNotIn("boundImageIdentity === loadedImageIdentity", body)
        self.assertNotIn("source !== boundImageSource", body)
        self.assertIn("++targetImageGeneration", body)
        self.assertIn("activateTargetImage(image, loadedImageIdentity, source, generation)", body)

    def test_same_source_identity_transition_has_bounded_wait(self) -> None:
        self.assertIn("const IMAGE_TRANSITION_WAIT_MS = 500", self.source)
        begin_body = _function_body(self.source, "beginImageTransition")
        self.assertIn("clearTransitionTimer()", begin_body)
        self.assertIn("setTimeout", begin_body)
        self.assertIn("releaseTransitionWait(expectedIdentity)", begin_body)

        release_body = _function_body(self.source, "releaseTransitionWait")
        self.assertIn("expectedIdentity !== loadedImageIdentity", release_body)
        self.assertIn('waitingPreviousSource = ""', release_body)
        self.assertIn("requestTargetActivation(targetImage)", release_body)

        request_body = _function_body(self.source, "requestTargetActivation")
        self.assertIn("source === waitingPreviousSource", request_body)
        self.assertIn("clearTransitionTimer()", request_body)
        self.assertIn('waitingPreviousSource = ""', request_body)

    def test_letterbox_hit_test_is_explicit_for_standalone_fallback(self) -> None:
        point_body = _function_body(self.source, "eventToNatural")
        self.assertIn("inside", point_body)
        self.assertIn("if (!inside && !clampOutside) return null", point_body)

    def test_disabled_overlay_does_not_capture_pointer_events(self) -> None:
        self.assertIn("class:disabled={!enabled()}", self.source)
        self.assertIn("pointer-events: none", self.source)
        self.assertIn("selection-rectangle.draft-selection", self.source)
        self.assertIn("selection-rectangle.applied-selection", self.source)


if __name__ == "__main__":
    unittest.main()
