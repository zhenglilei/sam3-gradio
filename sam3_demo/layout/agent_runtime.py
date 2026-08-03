"""Strict Qwen VLM client and visual payload helpers for layout-mask drafts."""

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

import numpy as np
from PIL import Image, ImageDraw


ALLOWED_RESPONSE_KEYS = {
    "schema_version",
    "intent",
    "profile",
    "confidence",
    "selected_candidate_id",
    "observations",
    "assistant_message",
    "manual_review",
    "warnings",
}
ALLOWED_INTENTS = {"analyze", "revise", "compare", "explain"}
ALLOWED_PROFILES = {"ACT", "GE1", "GE2", "Unknown"}


class LayoutMaskVLMError(RuntimeError):
    pass


def load_skill_bundle(skill_dir):
    root = Path(skill_dir)
    relative_paths = (
        "SKILL.md",
        "references/operation-catalog.md",
        "references/profile-priors.md",
        "references/keyword-routing.md",
        "references/response-schema.json",
        "references/dialogue-examples.md",
    )
    sections = []
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        data = path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        sections.append(f"## {relative_path}\n{data.decode('utf-8')}")
    return "\n\n".join(sections), digest.hexdigest()[:16]


def _fit_image(image, max_size):
    image = image.convert("RGB")
    image.thumbnail((int(max_size[0]), int(max_size[1])), Image.Resampling.LANCZOS)
    return image


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
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_candidate_contact_sheet(image, baseline_mask, candidates):
    """Render candidate IDs and deltas; RLE or full masks never enter JSON."""
    image = image.convert("RGB")
    baseline = np.asarray(baseline_mask, dtype=bool)
    tile_width, image_height, header_height = 360, 240, 64
    columns = 2
    rows = max(1, math.ceil(len(candidates) / columns))
    sheet = Image.new("RGB", (tile_width * columns, (image_height + header_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(candidates):
        column = index % columns
        row = index // columns
        left = column * tile_width
        top = row * (image_height + header_height)
        mask = np.asarray(candidate["mask"], dtype=bool)
        rgb = np.asarray(image, dtype=np.uint8).astype(np.float32)
        green = mask
        rgb[green] = rgb[green] * 0.50 + np.array([40, 220, 80], dtype=np.float32) * 0.50
        added = mask & ~baseline
        removed = baseline & ~mask
        rgb[added] = rgb[added] * 0.25 + np.array([255, 40, 40], dtype=np.float32) * 0.75
        rgb[removed] = rgb[removed] * 0.25 + np.array([40, 90, 255], dtype=np.float32) * 0.75
        tile = _fit_image(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), (tile_width, image_height))
        tile_left = left + (tile_width - tile.width) // 2
        tile_top = top + header_height + (image_height - tile.height) // 2
        sheet.paste(tile, (tile_left, tile_top))
        report = candidate.get("report") or {}
        risk = bool(candidate.get("error") or report.get("manual_review"))
        border = (240, 190, 0) if risk else (70, 130, 180)
        draw.rectangle((left + 1, top + 1, left + tile_width - 2, top + image_height + header_height - 2), outline=border, width=4 if risk else 2)
        title = f"{candidate['candidate_id']}  {candidate['label']}"
        draw.text((left + 8, top + 7), title[:50], fill=(20, 20, 20))
        params = candidate["params"]
        summary = (
            f"thr={params['threshold']} open={params['open_kernel']} "
            f"close={params['close_kernel']} morph={params['morph_pixels']}"
        )
        draw.text((left + 8, top + 29), summary, fill=(45, 45, 45))
        warning = candidate.get("error") or ("manual review" if risk else "safe")
        draw.text((left + 8, top + 47), str(warning)[:56], fill=(140, 90, 0) if risk else (40, 110, 50))
    return sheet


def _validate_text_list(value, field_name):
    if not isinstance(value, list) or len(value) > 6:
        raise LayoutMaskVLMError(f"{field_name} must be a list with at most six items")
    result = []
    for item in value:
        if not isinstance(item, str) or len(item) > 160:
            raise LayoutMaskVLMError(f"{field_name} contains an invalid item")
        result.append(item)
    return result


def validate_vlm_response(content, candidate_ids, profile_mode):
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LayoutMaskVLMError("VLM response is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise LayoutMaskVLMError("VLM response must be a JSON object")
    if set(payload) != ALLOWED_RESPONSE_KEYS:
        raise LayoutMaskVLMError("VLM response fields do not match the canonical schema")
    if payload["schema_version"] != 1:
        raise LayoutMaskVLMError("Unsupported VLM response schema")
    if payload["intent"] not in ALLOWED_INTENTS:
        raise LayoutMaskVLMError("Unknown VLM intent")
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
    candidate_id = payload["selected_candidate_id"]
    if not isinstance(candidate_id, str) or candidate_id not in set(candidate_ids):
        raise LayoutMaskVLMError("VLM selected an unknown candidate")
    message = payload["assistant_message"]
    if not isinstance(message, str) or not message.strip() or len(message) > 500:
        raise LayoutMaskVLMError("VLM assistant message is invalid")
    if not isinstance(payload["manual_review"], bool):
        raise LayoutMaskVLMError("VLM manual_review must be boolean")
    result = dict(payload)
    result["confidence"] = confidence
    result["observations"] = _validate_text_list(payload["observations"], "observations")
    result["warnings"] = _validate_text_list(payload["warnings"], "warnings")
    return result


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
    cost = float(cost) if isinstance(cost, Real) and not isinstance(cost, bool) and math.isfinite(float(cost)) else None
    return usage or None, cost


def call_qwen_layout_mask(
    *,
    image,
    contact_sheet,
    candidates,
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
    skill_text, skill_version = load_skill_bundle(skill_dir)
    public_candidates = [
        {
            "candidate_id": item["candidate_id"],
            "label": item["label"],
            "params": item["params"],
            "reason": item["reason"],
            "report": item.get("report"),
            "error": item.get("error"),
        }
        for item in candidates
    ]
    prior_turns = list(history or [])[-6:]
    system_prompt = (
        skill_text
        + "\n\nReturn one raw JSON object only. Never wrap it in markdown. "
        + "Candidate IDs and backend warnings are authoritative."
    )
    user_text = json.dumps(
        {
            "latest_user_message": str(message or ""),
            "profile_mode": profile_mode,
            "candidate_registry": public_candidates,
            "prior_turn_summaries": prior_turns,
            "color_legend": {
                "green": "candidate foreground",
                "red": "new pixels",
                "blue": "removed pixels",
                "yellow": "manual review or rejected topology",
            },
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
                    {"type": "image_url", "image_url": {"url": _png_data_url(image, 1024)}},
                    {"type": "image_url", "image_url": {"url": _png_data_url(contact_sheet, 1400)}},
                ],
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
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
        [item["candidate_id"] for item in candidates],
        profile_mode,
    )
    usage, cost = _usage_summary(result)
    return validated, usage, cost, skill_version
