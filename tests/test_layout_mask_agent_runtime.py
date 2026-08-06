import io
import json
import os
import unittest

from PIL import Image

from sam3_demo.config import (
    _LAYOUT_MASK_VLM_API_KEY,
    _LAYOUT_MASK_VLM_BASE_URL,
    _LAYOUT_MASK_VLM_MAX_TOKENS,
    _LAYOUT_MASK_VLM_MODEL,
    _LAYOUT_MASK_VLM_TEMPERATURE,
    _LAYOUT_MASK_VLM_TIMEOUT_SECONDS,
    layout_mask_agent_skill_dir,
)
from sam3_demo.layout.agent_runtime import (
    LayoutMaskVLMError,
    _estimate_act_period_grid,
    call_qwen_layout_mask,
    load_skill_bundle,
    validate_vlm_response,
)


def canonical_response(**updates):
    payload = {
        "schema_version": 4,
        "profile": "ACT",
        "confidence": 0.9,
        "period_total": 4,
        "period_rows": 2,
        "parameters": {
            "threshold": 12,
            "invert": False,
            "open_kernel": 0,
            "close_kernel": 15,
            "morph_pixels": 0,
            "min_component_area": 0,
            "region_mode": "all",
        },
        "explanation": "边缘存在小凹坑，建议使用 close=15 填补并保持线宽。",
        "manual_review": False,
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
        self.current_parameters = canonical_response()["parameters"]

    def test_strict_response_rejects_unknown_fields_and_forced_profile(self):
        with self.assertRaisesRegex(LayoutMaskVLMError, "canonical schema"):
            validate_vlm_response(
                json.dumps(canonical_response(extra=True)),
                "Auto",
                image_shape=(60, 80),
            )
        with self.assertRaisesRegex(LayoutMaskVLMError, "forced profile"):
            validate_vlm_response(
                json.dumps(canonical_response(profile="GE1")),
                "ACT",
                image_shape=(60, 80),
            )

    def test_invalid_json_nonfinite_confidence_and_noop_kernels_are_rejected(self):
        with self.assertRaisesRegex(LayoutMaskVLMError, "strict JSON"):
            validate_vlm_response("not-json", "Auto")
        with self.assertRaisesRegex(LayoutMaskVLMError, "outside"):
            validate_vlm_response(
                json.dumps(canonical_response(confidence=float("inf"))),
                "Auto",
            )
        invalid = canonical_response()
        invalid["parameters"]["close_kernel"] = 1
        with self.assertRaisesRegex(LayoutMaskVLMError, "odd integer"):
            validate_vlm_response(json.dumps(invalid), "Auto")
        invalid = canonical_response()
        invalid["parameters"]["open_kernel"] = 4
        with self.assertRaisesRegex(LayoutMaskVLMError, "odd integer"):
            validate_vlm_response(json.dumps(invalid), "Auto")

    def test_period_grid_must_match_profile(self):
        missing_period = canonical_response(period_total=0)
        with self.assertRaisesRegex(LayoutMaskVLMError, "positive period counts"):
            validate_vlm_response(json.dumps(missing_period), "Auto")
        non_act = canonical_response(
            profile="GE1",
            period_total=4,
            period_rows=2,
        )
        with self.assertRaisesRegex(LayoutMaskVLMError, "zero period counts"):
            validate_vlm_response(json.dumps(non_act), "Auto")

    def test_act_period_grid_estimator_counts_rows_and_ignores_legs(self):
        image = Image.new("L", (240, 200), 255)
        pixels = image.load()
        for row_y in (5, 100):
            for column_x in (10, 125):
                for y in range(row_y + 40, row_y + 65):
                    for x in range(column_x, column_x + 95):
                        pixels[x, y] = 0
                for offset_x in (35, 52):
                    for y in range(row_y, row_y + 25):
                        for x in range(column_x + offset_x, column_x + offset_x + 9):
                            pixels[x, y] = 0
                for y in range(row_y + 23, row_y + 31):
                    for x in range(column_x + 35, column_x + 61):
                        pixels[x, y] = 0
                for y in range(row_y + 23, row_y + 45):
                    for x in range(column_x + 44, column_x + 52):
                        pixels[x, y] = 0
                for leg_x in (column_x + 15, column_x + 70):
                    for y in range(row_y + 70, min(image.height, row_y + 88)):
                        for x in range(leg_x, leg_x + 8):
                            pixels[x, y] = 0
        grid = _estimate_act_period_grid(image.convert("RGB"))
        self.assertEqual(
            grid,
            {"period_total": 4, "period_columns": 2, "period_rows": 2},
        )

    def test_act_period_counts_deterministically_set_close(self):
        response = canonical_response(
            period_total=24,
            period_rows=4,
        )
        response["parameters"]["close_kernel"] = 9
        validated = validate_vlm_response(
            json.dumps(response),
            "Auto",
            image_shape=(60, 80),
        )
        self.assertEqual(validated["period_columns"], 6)
        self.assertEqual(validated["parameters"]["close_kernel"], 5)
        self.assertIn("数据中台按周期尺度规则采用close=5", validated["explanation"])

    def test_explanation_must_be_one_concise_line(self):
        with self.assertRaisesRegex(LayoutMaskVLMError, "one concise line"):
            validate_vlm_response(
                json.dumps(canonical_response(explanation="第一句。\n第二句。")),
                "Auto",
            )

    def test_skill_bundle_uses_profile_priors_without_dialogue_examples(self):
        text, version, schema = load_skill_bundle(layout_mask_agent_skill_dir)
        self.assertIn("references/profile-priors.md", text)
        self.assertIn("references/operation-catalog.md", text)
        self.assertNotIn("references/dialogue-examples.md", text)
        self.assertNotIn("references/keyword-routing.md", text)
        self.assertEqual(schema["properties"]["schema_version"]["const"], 4)
        self.assertIn("period_total", schema["required"])
        self.assertIn("period_rows", schema["required"])
        self.assertEqual(len(version), 16)

    def test_call_sends_one_image_and_strict_parameter_schema(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(
                {
                    "choices": [
                        {"message": {"content": json.dumps(canonical_response())}}
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
            current_parameters=self.current_parameters,
            message="填补凹坑但不要变粗",
            profile_mode="Auto",
            history=[{"candidate_id": "C2", "warnings": ["legacy"], "user_message": "上一轮", "params": self.current_parameters}],
            skill_dir=layout_mask_agent_skill_dir,
            base_url="http://example.test/v1",
            model="qwen3.5-122b",
            timeout_seconds=180,
            max_tokens=512,
            temperature=0.1,
            urlopen=fake_urlopen,
        )
        self.assertEqual(len(calls), 1)
        request_body = json.loads(calls[0][0].data.decode("utf-8"))
        user_content = request_body["messages"][1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 1)
        self.assertTrue(
            image_items[0]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        user_payload = json.loads(user_content[0]["text"])
        system_prompt = request_body["messages"][0]["content"]
        self.assertIn("禁止默认 ACT", system_prompt)
        self.assertIn("不足两个证据时才返回 Unknown", system_prompt)
        self.assertIn("解释若写符合某类", system_prompt)
        self.assertIn("单元内部拓扑优先于整图排列", system_prompt)
        self.assertIn("开口 U 槽不是 GE2", system_prompt)
        self.assertIn("先定位单个重复单元", user_payload["classification_policy"])
        self.assertIn("直接可见缺陷", user_payload["parameter_policy"])
        self.assertIn("period_total只数全图", user_payload["kernel_scale_policy"])
        self.assertIn("period_rows只数", user_payload["kernel_scale_policy"])
        self.assertIn("period_total/period_rows", user_payload["kernel_scale_policy"])
        self.assertIn("1-2列取15", user_payload["kernel_scale_policy"])
        self.assertIn("5-7列取5", user_payload["kernel_scale_policy"])
        prompt_text = system_prompt + user_content[0]["text"]
        self.assertNotIn("ACTMASK", prompt_text)
        self.assertNotIn("AST_ccfill", prompt_text)
        self.assertIn("不得用“可能、常见、通常”", system_prompt)
        self.assertEqual(user_payload["current_parameters"], self.current_parameters)
        self.assertNotIn("candidate_registry", user_payload)
        history_summary = user_payload["prior_turn_summaries"][0]
        self.assertNotIn("candidate_id", history_summary)
        self.assertNotIn("warnings", history_summary)
        self.assertNotIn("profile", history_summary)
        self.assertNotIn("assistant_message", history_summary)
        self.assertEqual(
            set(history_summary),
            {"user_message", "params"},
        )
        self.assertIn("不得从历史继承profile", user_payload["history_policy"])
        self.assertIn("禁止继承历史 profile", system_prompt)
        self.assertEqual(
            request_body["reasoning"],
            {"effort": "none", "exclude": True},
        )
        self.assertTrue(request_body["response_format"]["json_schema"]["strict"])
        self.assertEqual(response["parameters"]["close_kernel"], 15)
        self.assertNotIn("selected_candidate_id", response)
        self.assertEqual(usage["total_tokens"], 125)
        self.assertEqual(cost, 0.01)
        self.assertEqual(len(version), 16)

    def test_request_constraints_reject_thickening_and_implicit_largest(self):
        def response_for(payload):
            def fake_urlopen(request, timeout):
                return FakeResponse(
                    {"choices": [{"message": {"content": json.dumps(payload)}}]}
                )
            return fake_urlopen

        thick = canonical_response()
        thick["parameters"]["morph_pixels"] = 1
        with self.assertRaisesRegex(LayoutMaskVLMError, "preserve line width"):
            call_qwen_layout_mask(
                image=self.image,
                current_parameters=self.current_parameters,
                message="不要变粗",
                profile_mode="Auto",
                history=[],
                skill_dir=layout_mask_agent_skill_dir,
                base_url="http://example.test/v1",
                model="qwen3.5-122b",
                timeout_seconds=180,
                max_tokens=512,
                temperature=0.1,
                urlopen=response_for(thick),
            )

        largest = canonical_response()
        largest["parameters"]["region_mode"] = "largest"
        with self.assertRaisesRegex(LayoutMaskVLMError, "explicit user request"):
            call_qwen_layout_mask(
                image=self.image,
                current_parameters=self.current_parameters,
                message="自动分析",
                profile_mode="Auto",
                history=[],
                skill_dir=layout_mask_agent_skill_dir,
                base_url="http://example.test/v1",
                model="qwen3.5-122b",
                timeout_seconds=180,
                max_tokens=512,
                temperature=0.1,
                urlopen=response_for(largest),
            )

    @unittest.skipUnless(
        os.environ.get("RUN_LAYOUT_MASK_VLM_SMOKE") == "1",
        "paid VLM smoke is opt-in",
    )
    def test_paid_qwen_vlm_smoke(self):
        response, usage, _cost, _version = call_qwen_layout_mask(
            image=self.image,
            current_parameters=self.current_parameters,
            message="填补凹坑但不要变粗",
            profile_mode="ACT",
            history=[],
            skill_dir=layout_mask_agent_skill_dir,
            base_url=_LAYOUT_MASK_VLM_BASE_URL,
            model=_LAYOUT_MASK_VLM_MODEL,
            timeout_seconds=_LAYOUT_MASK_VLM_TIMEOUT_SECONDS,
            max_tokens=_LAYOUT_MASK_VLM_MAX_TOKENS,
            temperature=_LAYOUT_MASK_VLM_TEMPERATURE,
            api_key=_LAYOUT_MASK_VLM_API_KEY,
        )
        self.assertEqual(response["profile"], "ACT")
        self.assertIn("parameters", response)
        self.assertIsInstance(usage, (dict, type(None)))


if __name__ == "__main__":
    unittest.main()
