"""Strict single-image Qwen VLM client for layout-mask parameter drafts."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
from numbers import Real
from pathlib import Path

from PIL import Image

from sam3_demo.layout.preprocess_registry import PARAM_KEYS, normalize_params


ALLOWED_RESPONSE_KEYS = {
    "schema_version",
    "profile",
    "confidence",
    "parameters",
    "explanation",
    "manual_review",
}
ALLOWED_PROFILES = {"ACT", "GE1", "GE2", "Unknown"}
VALID_MORPH_KERNELS = {0, *range(3, 32, 2)}


class LayoutMaskVLMError(RuntimeError):
    pass


def load_skill_bundle(skill_dir):
    """Load only the single-image instructions used by the paid request."""
    root = Path(skill_dir)
    relative_paths = (
        "SKILL.md",
        "references/operation-catalog.md",
        "references/profile-priors.md",
        "references/response-schema.json",
    )
    sections = []
    digest = hashlib.sha256()
    response_schema = None
    for relative_path in relative_paths:
        path = root / relative_path
        data = path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        text = data.decode("utf-8")
        if relative_path.endswith(".json"):
            response_schema = json.loads(text)
        sections.append(f"## {relative_path}\n{text}")
    if not isinstance(response_schema, dict):
        raise LayoutMaskVLMError("Layout mask response schema is missing")
    return "\n\n".join(sections), digest.hexdigest()[:16], response_schema


def _png_data_url(image, max_side):
    image = image.convert("RGB")
    if max(image.size) > max_side:
        ratio = float(max_side) / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _validate_raw_parameters(value):
    if not isinstance(value, dict) or set(value) != set(PARAM_KEYS):
        raise LayoutMaskVLMError("VLM parameters do not match the canonical schema")
    integer_fields = (
        "threshold",
        "open_kernel",
        "close_kernel",
        "morph_pixels",
        "min_component_area",
    )
    for field in integer_fields:
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise LayoutMaskVLMError(f"VLM {field} must be an integer")
    if not isinstance(value["invert"], bool):
        raise LayoutMaskVLMError("VLM invert must be boolean")
    if not isinstance(value["region_mode"], str):
        raise LayoutMaskVLMError("VLM region_mode must be a string")
    for field in ("open_kernel", "close_kernel"):
        if value[field] not in VALID_MORPH_KERNELS:
            raise LayoutMaskVLMError(
                f"VLM {field} must be 0 or an odd integer from 3 to 31"
            )


def validate_vlm_response(content, profile_mode, image_shape=None):
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LayoutMaskVLMError("VLM response is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise LayoutMaskVLMError("VLM response must be a JSON object")
    if set(payload) != ALLOWED_RESPONSE_KEYS:
        raise LayoutMaskVLMError("VLM response fields do not match the canonical schema")
    if payload["schema_version"] != 2:
        raise LayoutMaskVLMError("Unsupported VLM response schema")
    if payload["profile"] not in ALLOWED_PROFILES:
        raise LayoutMaskVLMError("Unknown VLM profile")
    if profile_mode in {"ACT", "GE1", "GE2"} and payload["profile"] != profile_mode:
        raise LayoutMaskVLMError("VLM profile conflicts with the forced profile")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise LayoutMaskVLMError("VLM confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise LayoutMaskVLMError("VLM confidence is outside [0, 1]")
    _validate_raw_parameters(payload["parameters"])
    try:
        parameters = normalize_params(payload["parameters"], image_shape=image_shape)
    except (TypeError, ValueError) as exc:
        raise LayoutMaskVLMError(f"VLM parameters are invalid: {exc}") from exc
    explanation = payload["explanation"]
    if (
        not isinstance(explanation, str)
        or not explanation.strip()
        or len(explanation) > 240
        or "\n" in explanation
        or "\r" in explanation
    ):
        raise LayoutMaskVLMError("VLM explanation must be one concise line")
    if not isinstance(payload["manual_review"], bool):
        raise LayoutMaskVLMError("VLM manual_review must be boolean")
    result = dict(payload)
    result["confidence"] = confidence
    result["parameters"] = parameters
    result["explanation"] = explanation.strip()
    return result


def _validate_request_constraints(response, message):
    compact = "".join(str(message or "").casefold().split())
    avoid_thickening = any(
        keyword in compact
        for keyword in ("不要加粗", "不加粗", "不要变粗", "保持线宽")
    )
    if avoid_thickening and response["parameters"]["morph_pixels"] > 0:
        raise LayoutMaskVLMError(
            "VLM recommendation conflicts with the request to preserve line width"
        )
    largest_requested = any(
        keyword in compact
        for keyword in ("只保留最大", "最大主体", "最大连通")
    )
    if response["parameters"]["region_mode"] == "largest" and not largest_requested:
        raise LayoutMaskVLMError(
            "VLM selected largest region without an explicit user request"
        )


def _history_summaries(history):
    allowed = (
        "user_message",
        "assistant_message",
        "profile",
        "params",
        "manual_review",
    )
    rows = []
    for item in list(history or [])[-6:]:
        if not isinstance(item, dict):
            continue
        rows.append({key: item[key] for key in allowed if key in item})
    return rows


def _usage_summary(result):
    raw = result.get("usage") if isinstance(result, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            usage[key] = int(value)
    cost = raw.get("cost")
    if not isinstance(cost, Real) or isinstance(cost, bool) or not math.isfinite(float(cost)):
        hidden = result.get("_hidden_params") if isinstance(result, dict) else None
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    cost = (
        float(cost)
        if isinstance(cost, Real)
        and not isinstance(cost, bool)
        and math.isfinite(float(cost))
        else None
    )
    return usage or None, cost


def call_qwen_layout_mask(
    *,
    image,
    current_parameters,
    message,
    profile_mode,
    history,
    skill_dir,
    base_url,
    model,
    timeout_seconds,
    max_tokens,
    temperature,
    api_key="",
    urlopen=urllib.request.urlopen,
):
    """Send exactly one image and receive strict preprocessing parameters."""
    image = image.convert("RGB")
    current_parameters = normalize_params(
        current_parameters,
        image_shape=(image.height, image.width),
    )
    skill_text, skill_version, response_schema = load_skill_bundle(skill_dir)
    response_schema = dict(response_schema)
    response_schema.pop("$schema", None)
    system_prompt = (
        skill_text
        + "\n\n只返回一个原始 JSON 对象，不要使用 Markdown 代码块。"
        + "VLM 只推荐参数；4090 后端执行图像处理和 mask 生成。"
    )
    user_text = json.dumps(
        {
            "request": str(message or ""),
            "profile_mode": profile_mode,
            "current_parameters": current_parameters,
            "image_size": [image.width, image.height],
            "prior_turn_summaries": _history_summaries(history),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(image, 1024)},
                    },
                ],
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "reasoning": {"effort": "none", "exclude": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "layout_mask_recommendation",
                "strict": True,
                "schema": response_schema,
            },
        },
        "stream": False,
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{str(base_url).rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise LayoutMaskVLMError(f"LiteLLM HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LayoutMaskVLMError("LiteLLM request failed or timed out") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise LayoutMaskVLMError("LiteLLM returned an error response")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LayoutMaskVLMError("LiteLLM response has no choices")
    message_payload = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message_payload.get("content") if isinstance(message_payload, dict) else None
    if not isinstance(content, str):
        raise LayoutMaskVLMError("LiteLLM response content is missing")
    validated = validate_vlm_response(
        content,
        profile_mode,
        image_shape=(image.height, image.width),
    )
    _validate_request_constraints(validated, message)
    usage, cost = _usage_summary(result)
    return validated, usage, cost, skill_version
