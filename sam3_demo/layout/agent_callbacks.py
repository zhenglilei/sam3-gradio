"""In-memory orchestration for the constrained layout-mask assistant."""

from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
from PIL import Image

from sam3_demo.layout.agent_runtime import call_qwen_layout_mask
from sam3_demo.layout.mask_quality import (
    compare_masks,
    render_mask_delta,
    validate_candidate_transition,
)
from sam3_demo.layout.preprocess_registry import PARAM_KEYS, normalize_params
from sam3_demo.state import _new_layout_mask_agent_state


def layout_agent_image_sha256(image):
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()

def validate_layout_mask_agent_context(state, *, session_id, image):
    """Return the current image only when an Agent Draft belongs to it."""
    if image is None:
        raise ValueError("Upload a layout screenshot first")
    image = image.convert("RGB")
    if not isinstance(state, dict):
        raise ValueError("No Agent Draft is available")
    if state.get("session_id") != session_id:
        raise ValueError("Agent Draft belongs to a different session")
    image_hash = layout_agent_image_sha256(image)
    if state.get("image_sha256") != image_hash:
        raise ValueError(
            "Agent Draft does not belong to the current layout screenshot"
        )
    return image



def _compute_mask(compute_draft, image, params):
    result = compute_draft(
        image,
        params["threshold"],
        params["invert"],
        params["open_kernel"],
        params["close_kernel"],
        params["min_component_area"],
        params["region_mode"],
        params["morph_pixels"],
    )
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise ValueError("Layout draft computation returned an invalid result")
    computed_image = result[0].convert("RGB")
    mask = np.asarray(result[1], dtype=bool)
    if mask.shape != (computed_image.height, computed_image.width):
        raise ValueError("Layout draft mask shape does not match the image")
    return computed_image, mask


def _params_signature(params):
    return tuple(params[key] for key in PARAM_KEYS)


def _history_chat(history):
    messages = []
    for row in history or []:
        user_message = row.get("user_message")
        assistant_message = row.get("assistant_message")
        if user_message:
            messages.append({"role": "user", "content": str(user_message)})
        if assistant_message:
            messages.append({"role": "assistant", "content": str(assistant_message)})
    return messages


def classify_layout_agent_message(message):
    compact = "".join(str(message or "").strip().casefold().split())
    for prefix in ("\u8bf7\u5e2e\u6211", "\u9ebb\u70e6\u5e2e\u6211", "\u9ebb\u70e6", "\u5e2e\u6211", "\u8bf7"):
        if compact.startswith(prefix):
            compact = compact[len(prefix) :]
            break
    for suffix in ("\u4e00\u4e0b", "\u5427", "\u3002", "\uff01", "!"):
        if compact.endswith(suffix):
            compact = compact[: -len(suffix)]
            break
    if compact in {"undo", "\u64a4\u56de", "\u64a4\u56de\u4e0a\u4e00\u7248", "\u6062\u590d\u4e0a\u4e00\u7248"}:
        return "undo"
    if compact in {"reset", "\u91cd\u7f6e", "\u91cd\u65b0\u5f00\u59cb"}:
        return "reset"
    if compact in {"apply", "\u5e94\u7528", "\u5e94\u7528\u53c2\u6570", "\u5e94\u7528\u63a8\u8350\u53c2\u6570"}:
        return "apply"
    return "vlm"


def format_parameter_diff(baseline, current):
    if not baseline or not current:
        return "No active Agent Draft."
    rows = []
    for key in PARAM_KEYS:
        before = baseline.get(key)
        after = current.get(key)
        marker = "changed" if before != after else "same"
        rows.append(f"- {key}: {before} -> {after} ({marker})")
    return "\n".join(rows)


def _manual_override(state, controls):
    current = state.get("current_draft_params")
    baseline = state.get("baseline_params")
    control_signature = _params_signature(controls)
    if (
        current is None
        or _params_signature(current) == control_signature
        or (baseline is not None and _params_signature(baseline) == control_signature)
    ):
        return state
    updated = copy.deepcopy(state)
    updated["baseline_params"] = dict(controls)
    updated["current_draft_params"] = dict(controls)
    updated["undo_stack"] = []
    history = list(updated.get("history") or [])
    history.append(
        {
            "kind": "manual_override",
            "params": dict(controls),
            "assistant_message": "Manual controls became the new Agent baseline.",
        }
    )
    updated["history"] = history[-6:]
    return updated


def set_layout_mask_agent_consent(
    state,
    *,
    session_id,
    image,
    consent,
    profile_mode,
):
    """Bind outbound consent to one exact image and profile locally."""
    current = copy.deepcopy(state or _new_layout_mask_agent_state(session_id))
    image_hash = None
    if image is not None:
        image_hash = layout_agent_image_sha256(image.convert("RGB"))
    if (
        current.get("session_id") != session_id
        or current.get("image_sha256") != image_hash
        or current.get("profile_mode") != profile_mode
    ):
        current = reset_layout_mask_agent(session_id, image, profile_mode)
    current["consent_image_sha256"] = image_hash if consent and image_hash else None
    return current


def run_layout_mask_agent_turn(
    *,
    state,
    session_id,
    image,
    consent,
    profile_mode,
    user_message,
    controls,
    compute_draft,
    vlm_options,
    vlm_call=call_qwen_layout_mask,
):
    """Execute exactly one paid VLM turn and return a fully in-memory result."""
    if image is None:
        raise ValueError("Upload a layout screenshot first")
    image = image.convert("RGB")
    image_hash = layout_agent_image_sha256(image)
    current_state = copy.deepcopy(state or _new_layout_mask_agent_state(session_id))
    if (
        current_state.get("session_id") != session_id
        or current_state.get("image_sha256") != image_hash
    ):
        current_state = _new_layout_mask_agent_state(session_id)
        current_state["image_sha256"] = image_hash
    if current_state.get("profile_mode") != profile_mode:
        current_state = _new_layout_mask_agent_state(session_id)
        current_state["image_sha256"] = image_hash
        current_state["profile_mode"] = profile_mode
    if not consent or current_state.get("consent_image_sha256") != image_hash:
        raise ValueError("Confirm outbound image sharing for this image")
    message = str(user_message or "").strip()
    if not message:
        raise ValueError("Enter a request for the mask assistant")
    if len(message) > 1000:
        raise ValueError("Mask assistant message is too long")
    normalized_controls = normalize_params(controls, image_shape=(image.height, image.width))
    current_state = _manual_override(current_state, normalized_controls)
    if current_state.get("current_draft_params") is None:
        current_state["baseline_params"] = dict(normalized_controls)
        current_state["current_draft_params"] = dict(normalized_controls)
    current_params = normalize_params(
        current_state["current_draft_params"],
        image_shape=(image.height, image.width),
    )
    computed_image, current_mask = _compute_mask(compute_draft, image, current_params)

    request_started = time.perf_counter()
    response, usage, cost, skill_version = vlm_call(
        image=computed_image,
        current_parameters=current_params,
        message=message,
        profile_mode=profile_mode,
        history=list(current_state.get("history") or [])[-6:],
        **vlm_options,
    )
    latency_seconds = time.perf_counter() - request_started

    recommended_params = normalize_params(
        response["parameters"],
        image_shape=(computed_image.height, computed_image.width),
    )
    # The VLM recommends parameters only. The 数据中台 host computes the full-resolution Draft.
    _, recommended_mask = _compute_mask(
        compute_draft,
        computed_image,
        recommended_params,
    )
    transition_report = validate_candidate_transition(
        current_mask,
        recommended_mask,
    )

    updated = copy.deepcopy(current_state)
    params_changed = (
        _params_signature(recommended_params) != _params_signature(current_params)
    )
    if params_changed:
        undo_stack = list(updated.get("undo_stack") or [])
        undo_stack.append(dict(current_params))
        updated["undo_stack"] = undo_stack[-6:]
        updated["current_draft_params"] = dict(recommended_params)
    updated["conversation_revision"] = int(updated.get("conversation_revision") or 0) + 1
    updated["turn_count"] = int(updated.get("turn_count") or 0) + 1
    updated["profile_mode"] = profile_mode
    updated["detected_profile"] = response["profile"]
    updated["confidence"] = response["confidence"]
    updated["skill_version"] = skill_version
    updated["last_usage"] = usage
    updated["last_cost"] = cost
    updated["last_latency_seconds"] = latency_seconds
    manual_review = bool(
        response["manual_review"] or transition_report["manual_review"]
    )
    history = list(updated.get("history") or [])
    history.append(
        {
            "kind": "recommendation",
            "user_message": message,
            "assistant_message": response["explanation"],
            "profile": response["profile"],
            "confidence": response["confidence"],
            "params": dict(updated["current_draft_params"]),
            "manual_review": manual_review,
        }
    )
    updated["history"] = history[-6:]
    preview_array = render_mask_delta(
        computed_image,
        current_mask,
        recommended_mask,
        manual_review=manual_review,
    )
    tokens = (usage or {}).get("total_tokens")
    status_parts = [
        f"profile={response['profile']} confidence={response['confidence']:.2f}",
        "manual review required" if manual_review else "topology checks passed",
    ]
    if tokens is not None:
        status_parts.append(f"tokens={tokens}")
    if cost is not None:
        status_parts.append(f"cost={cost:.6f}")
    status_parts.append(f"latency={latency_seconds:.1f}s")
    return {
        "state": updated,
        "chat": _history_chat(updated["history"]),
        "draft_preview": Image.fromarray(preview_array),
        "diff": format_parameter_diff(
            updated.get("baseline_params"),
            updated.get("current_draft_params"),
        ),
        "status": "; ".join(status_parts),
        "recommended_params": dict(updated["current_draft_params"]),
        "manual_review": manual_review,
    }


def undo_layout_mask_agent_draft(state, image, compute_draft):
    updated = copy.deepcopy(state or {})
    undo_stack = list(updated.get("undo_stack") or [])
    if not undo_stack:
        raise ValueError("No Agent Draft is available to undo")
    current = normalize_params(updated["current_draft_params"])
    previous = normalize_params(undo_stack.pop())
    updated["undo_stack"] = undo_stack
    updated["current_draft_params"] = previous
    updated["conversation_revision"] = int(updated.get("conversation_revision") or 0) + 1
    computed_image, current_mask = _compute_mask(compute_draft, image.convert("RGB"), current)
    _, previous_mask = _compute_mask(compute_draft, computed_image, previous)
    report = compare_masks(current_mask, previous_mask)
    preview = Image.fromarray(
        render_mask_delta(
            computed_image,
            current_mask,
            previous_mask,
            manual_review=report["manual_review"],
        )
    )
    return {
        "state": updated,
        "chat": _history_chat(updated.get("history")),
        "draft_preview": preview,
        "diff": format_parameter_diff(updated.get("baseline_params"), previous),
        "status": "Agent Draft restored locally; no VLM request was made.",
    }


def reset_layout_mask_agent(session_id, image, profile_mode="Auto"):
    state = _new_layout_mask_agent_state(session_id)
    if image is not None:
        state["image_sha256"] = layout_agent_image_sha256(image.convert("RGB"))
    state["profile_mode"] = profile_mode
    return state


def apply_layout_mask_agent_params(state):
    params = state.get("current_draft_params") if isinstance(state, dict) else None
    if not params:
        raise ValueError("No Agent Draft is available to apply")
    normalized = normalize_params(params)
    updated = copy.deepcopy(state)
    updated["applied_revision"] = int(updated.get("conversation_revision") or 0)
    return updated, dict(normalized)


def mark_layout_mask_agent_saved(state, controls):
    if not isinstance(state, dict):
        return state
    updated = copy.deepcopy(state)
    normalized = normalize_params(controls)
    updated["baseline_params"] = dict(normalized)
    updated["current_draft_params"] = dict(normalized)
    updated["undo_stack"] = []
    updated["applied_revision"] = int(updated.get("conversation_revision") or 0)
    return updated
