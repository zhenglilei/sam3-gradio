import io
import json
import unittest

import numpy as np
from PIL import Image

from sam3_demo.config import layout_mask_agent_skill_dir
from sam3_demo.layout.agent_runtime import (
    LayoutMaskVLMError,
    build_candidate_contact_sheet,
    call_qwen_layout_mask,
    validate_vlm_response,
)


def canonical_response(**updates):
    payload = {
        "schema_version": 1,
        "intent": "analyze",
        "profile": "ACT",
        "confidence": 0.9,
        "selected_candidate_id": "C1",
        "observations": ["small concavities"],
        "assistant_message": "Use C1.",
        "manual_review": False,
        "warnings": [],
    }
    payload.update(updates)
    return payload


class FakeResponse:
    def __init__(self, payload):
        self.buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self.buffer

    def __exit__(self, exc_type, exc, traceback):
        return False


class LayoutMaskAgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 60), "white")
        self.baseline = np.zeros((60, 80), dtype=bool)
        self.baseline[10:40, 15:45] = True
        self.candidates = [
            {
                "candidate_id": "C1",
                "label": "ACT close=15",
                "params": {
                    "threshold": 12,
                    "invert": False,
                    "open_kernel": 0,
                    "close_kernel": 15,
                    "morph_pixels": 0,
                    "min_component_area": 0,
                    "region_mode": "all",
                },
                "reason": "ACT baseline",
                "mask": self.baseline,
                "report": {"manual_review": False},
                "error": None,
            }
        ]

    def test_strict_response_rejects_unknown_fields_candidate_and_profile(self):
        response = canonical_response(extra=True)
        with self.assertRaisesRegex(LayoutMaskVLMError, "canonical schema"):
            validate_vlm_response(json.dumps(response), {"C1"}, "Auto")
        with self.assertRaisesRegex(LayoutMaskVLMError, "unknown candidate"):
            validate_vlm_response(
                json.dumps(canonical_response(selected_candidate_id="C9")),
                {"C1"},
                "Auto",
            )
        with self.assertRaisesRegex(LayoutMaskVLMError, "forced profile"):
            validate_vlm_response(
                json.dumps(canonical_response(profile="GE1")),
                {"C1"},
                "ACT",
            )

    def test_invalid_json_and_nonfinite_confidence_are_rejected(self):
        with self.assertRaisesRegex(LayoutMaskVLMError, "strict JSON"):
            validate_vlm_response("```json\n{}\n```", {"C1"}, "Auto")
        with self.assertRaisesRegex(LayoutMaskVLMError, "outside"):
            validate_vlm_response(
                json.dumps(canonical_response(confidence=float("inf"))),
                {"C1"},
                "Auto",
            )

    def test_call_sends_two_resized_images_and_makes_one_request(self):
        contact = build_candidate_contact_sheet(self.image, self.baseline, self.candidates)
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(canonical_response())
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "total_tokens": 125,
                        "cost": 0.01,
                    },
                }
            )

        response, usage, cost, version = call_qwen_layout_mask(
            image=self.image,
            contact_sheet=contact,
            candidates=self.candidates,
            message="analyze",
            profile_mode="Auto",
            history=[],
            skill_dir=layout_mask_agent_skill_dir,
            base_url="http://example.test/v1",
            model="qwen3.5-122b",
            timeout_seconds=180,
            max_tokens=1024,
            temperature=0.1,
            urlopen=fake_urlopen,
        )
        self.assertEqual(len(calls), 1)
        request_body = json.loads(calls[0][0].data.decode("utf-8"))
        user_content = request_body["messages"][1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        self.assertTrue(all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in image_items))
        self.assertEqual(response["selected_candidate_id"], "C1")
        self.assertEqual(usage["total_tokens"], 125)
        self.assertEqual(cost, 0.01)
        self.assertEqual(len(version), 16)


if __name__ == "__main__":
    unittest.main()
