"""Deterministic candidate registry for the constrained layout-mask assistant."""

from __future__ import annotations

import re


PARAM_KEYS = (
    "threshold",
    "invert",
    "open_kernel",
    "close_kernel",
    "morph_pixels",
    "min_component_area",
    "region_mode",
)
PROFILE_NAMES = ("Auto", "ACT", "GE1", "GE2", "Unknown")
SKILL_VERSION = "1.0.0"

_INT_LIMITS = {
    "threshold": (0, 255),
    "open_kernel": (0, 31),
    "close_kernel": (0, 31),
    "morph_pixels": (-31, 31),
    "min_component_area": (0, None),
}
_EXPLICIT_PARAM_RE = re.compile(
    r"(?P<key>threshold|open(?:_kernel)?|close(?:_kernel)?|morph(?:_pixels)?|min(?:_component)?_area)\s*[=:\uFF1A]\s*(?P<value>[+-]?\d+)",
    re.IGNORECASE,
)


def params_from_controls(
    threshold,
    invert,
    open_kernel,
    close_kernel,
    min_component_area,
    region_mode,
    morph_pixels,
):
    return normalize_params(
        {
            "threshold": threshold,
            "invert": invert,
            "open_kernel": open_kernel,
            "close_kernel": close_kernel,
            "morph_pixels": morph_pixels,
            "min_component_area": min_component_area,
            "region_mode": region_mode,
        }
    )


def normalize_params(params, image_shape=None):
    source = dict(params or {})
    missing = [key for key in PARAM_KEYS if key not in source]
    if missing:
        raise ValueError(f"Missing layout preprocessing parameters: {', '.join(missing)}")
    result = {
        "threshold": int(source["threshold"]),
        "invert": bool(source["invert"]),
        "open_kernel": int(source["open_kernel"] or 0),
        "close_kernel": int(source["close_kernel"] or 0),
        "morph_pixels": int(source["morph_pixels"] or 0),
        "min_component_area": int(source["min_component_area"] or 0),
        "region_mode": str(source["region_mode"] or "all"),
    }
    for key, (lower, upper) in _INT_LIMITS.items():
        value = result[key]
        effective_upper = upper
        if key == "min_component_area" and image_shape is not None:
            effective_upper = int(image_shape[0]) * int(image_shape[1])
        if value < lower or (effective_upper is not None and value > effective_upper):
            raise ValueError(f"{key} is outside the UI range")
    if result["region_mode"] not in {"all", "largest"}:
        raise ValueError("region_mode must be all or largest")
    return result


def _with(base, **updates):
    result = dict(base)
    result.update(updates)
    return result


def _explicit_candidate(current, message, image_shape):
    updates = {}
    aliases = {
        "open": "open_kernel",
        "open_kernel": "open_kernel",
        "close": "close_kernel",
        "close_kernel": "close_kernel",
        "morph": "morph_pixels",
        "morph_pixels": "morph_pixels",
        "min_area": "min_component_area",
        "min_component_area": "min_component_area",
        "threshold": "threshold",
    }
    for match in _EXPLICIT_PARAM_RE.finditer(str(message or "")):
        updates[aliases[match.group("key").lower()]] = int(match.group("value"))
    if not updates:
        return None
    return normalize_params(_with(current, **updates), image_shape=image_shape)


def _effective_profile(profile_mode, detected_profile, message):
    if profile_mode in {"ACT", "GE1", "GE2"}:
        return profile_mode
    upper = str(message or "").upper()
    for profile in ("ACT", "GE1", "GE2"):
        if profile in upper:
            return profile
    if detected_profile in {"ACT", "GE1", "GE2"}:
        return detected_profile
    return "Unknown"


def build_candidates(
    current_params,
    *,
    profile_mode="Auto",
    detected_profile="Unknown",
    message="",
    previous_params=None,
    image_shape=None,
):
    """Return at most six validated, deduplicated parameter candidates."""
    if profile_mode not in PROFILE_NAMES:
        raise ValueError("Unknown profile mode")
    current = normalize_params(current_params, image_shape=image_shape)
    profile = _effective_profile(profile_mode, detected_profile, message)
    rows = [("Current controls", current, "Keep the current parameters")]
    explicit = _explicit_candidate(current, message, image_shape)
    if explicit is not None:
        rows.append(("Explicit user values", explicit, "Use user values within the UI range"))
    if previous_params:
        rows.append(("Previous draft", normalize_params(previous_params, image_shape=image_shape), "Restore the previous draft"))

    text = str(message or "")
    if profile == "ACT":
        current_close = int(current["close_kernel"])
        if "\u518d\u586b" in text or "\u586b\u8865" in text or "\u51f9\u5751" in text:
            rows.append(("ACT fill a little more", _with(current, close_kernel=min(31, max(3, current_close + 2)), morph_pixels=0), "Increase close without changing line width"))
        if "\u7c98\u8fde" in text or "\u51cf\u5c11" in text:
            rows.append(("ACT reduce bridges", _with(current, close_kernel=max(0, current_close - 2), morph_pixels=0), "Reduce close"))
        for close in (11, 13, 15, 17, 19):
            rows.append((f"ACT close={close}", _with(current, close_kernel=close, morph_pixels=0), "Fill small concavities without dilation"))
    elif profile == "GE1":
        for morph in (0, 1, 2):
            rows.append((f"GE1 morph={morph:+d}", _with(current, close_kernel=0, morph_pixels=morph), "Conservative line-width search"))
        if "\u4e0d\u52a0\u7c97" in text and ("\u65ad" in text or "\u4fee\u8865" in text):
            for close in (3, 5):
                rows.append((f"GE1 close={close}", _with(current, close_kernel=close, morph_pixels=0), "Repair gaps without thickening"))
    elif profile == "GE2":
        rows.append(("GE2 preserve holes", _with(current, close_kernel=0, morph_pixels=0), "Preserve square holes and separation gaps"))
        rows.append(("GE2 threshold-2", _with(current, threshold=max(0, current["threshold"] - 2), close_kernel=0, morph_pixels=0), "Only adjust threshold"))
        rows.append(("GE2 threshold+2", _with(current, threshold=min(255, current["threshold"] + 2), close_kernel=0, morph_pixels=0), "Only adjust threshold"))
    else:
        rows.extend(
            [
                ("ACT baseline", _with(current, threshold=12, open_kernel=0, close_kernel=15, morph_pixels=0), "ACT concavity baseline"),
                ("GE1 baseline", _with(current, threshold=12, open_kernel=0, close_kernel=0, morph_pixels=1), "GE1 line-width baseline"),
                ("GE2 baseline", _with(current, threshold=12, open_kernel=0, close_kernel=0, morph_pixels=0), "GE2 hole-preserving baseline"),
                ("Conservative denoise", _with(current, open_kernel=3), "Only add a light open"),
                ("threshold-2", _with(current, threshold=max(0, current["threshold"] - 2)), "Only reduce threshold"),
                ("threshold+2", _with(current, threshold=min(255, current["threshold"] + 2)), "Only increase threshold"),
            ]
        )
    if "\u6574\u4f53\u52a0\u7c97" in text or "\u52a0\u7c97" in text:
        rows.insert(1, ("Thicken globally", _with(current, morph_pixels=min(31, current["morph_pixels"] + 1)), "Add 1 px dilation"))
    if "\u4fdd\u7559\u5b54\u6d1e" in text:
        rows.insert(1, ("Preserve holes", _with(current, close_kernel=0, morph_pixels=min(0, current["morph_pixels"])), "Disable close and dilation"))

    candidates = []
    seen = set()
    for label, params, reason in rows:
        normalized = normalize_params(params, image_shape=image_shape)
        signature = tuple(normalized[key] for key in PARAM_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1}",
                "label": label,
                "params": normalized,
                "reason": reason,
            }
        )
        if len(candidates) == 6:
            break
    return candidates
