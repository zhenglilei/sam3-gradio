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

import cv2
import numpy as np
from PIL import Image

from sam3_demo.layout.preprocess_registry import PARAM_KEYS, normalize_params


ALLOWED_RESPONSE_KEYS = {
    "schema_version",
    "profile",
    "confidence",
    "period_total",
    "period_rows",
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


def _estimate_act_period_grid(image):
    """Estimate ACT rows/columns from repeated fork-cap geometry."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(
        gray,
        0,
        1,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    foreground_fraction = float(binary.mean())
    if not 0.005 <= foreground_fraction <= 0.65:
        binary = 1 - binary
        foreground_fraction = float(binary.mean())
    if not 0.005 <= foreground_fraction <= 0.65:
        return None

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    image_area = int(binary.shape[0] * binary.shape[1])
    min_main_area = max(16, int(round(image_area * 0.005)))
    main_components = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < min_main_area or width < max(3, round(height * 0.65)):
            continue
        cap_height = max(1, round(height * 0.30))
        cap = binary[y : y + cap_height, x : x + width]
        cap_count, _, cap_stats, _ = cv2.connectedComponentsWithStats(cap, 8)
        min_cap_area = max(4, round(area * 0.005))
        cap_lobes = sum(
            1
            for cap_index in range(1, cap_count)
            if int(cap_stats[cap_index, cv2.CC_STAT_AREA]) >= min_cap_area
        )
        if cap_lobes < 2:
            continue
        cycles = max(1, int(round(cap_lobes / 2.0)))
        main_components.append(
            {
                "center_y": y + height / 2.0,
                "height": height,
                "cycles": cycles,
            }
        )
    if not main_components:
        return None

    tolerance = max(2.0, float(np.median([row["height"] for row in main_components])) * 0.35)
    clusters = []
    for component in sorted(main_components, key=lambda row: row["center_y"]):
        if not clusters or abs(component["center_y"] - clusters[-1]["mean_y"]) > tolerance:
            clusters.append(
                {
                    "mean_y": component["center_y"],
                    "centers": [component["center_y"]],
                    "cycles": component["cycles"],
                }
            )
        else:
            cluster = clusters[-1]
            cluster["centers"].append(component["center_y"])
            cluster["mean_y"] = float(np.mean(cluster["centers"]))
            cluster["cycles"] += component["cycles"]
    row_cycles = [int(cluster["cycles"]) for cluster in clusters]
    if not row_cycles or max(row_cycles) - min(row_cycles) > 1:
        return None
    period_rows = len(row_cycles)
    period_columns = max(1, int(round(float(np.median(row_cycles)))))
    return {
        "period_total": period_columns * period_rows,
        "period_columns": period_columns,
        "period_rows": period_rows,
    }


def validate_vlm_response(content, profile_mode, image_shape=None, image=None):
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LayoutMaskVLMError("VLM response is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise LayoutMaskVLMError("VLM response must be a JSON object")
    if set(payload) != ALLOWED_RESPONSE_KEYS:
        raise LayoutMaskVLMError("VLM response fields do not match the canonical schema")
    if payload["schema_version"] != 4:
        raise LayoutMaskVLMError("Unsupported VLM response schema")
    if payload["profile"] not in ALLOWED_PROFILES:
        raise LayoutMaskVLMError("Unknown VLM profile")
    if profile_mode in {"ACT", "GE1", "GE2"} and payload["profile"] != profile_mode:
        raise LayoutMaskVLMError("VLM profile conflicts with the forced profile")
    for field in ("period_total", "period_rows"):
        if isinstance(payload[field], bool) or not isinstance(payload[field], int):
            raise LayoutMaskVLMError(f"VLM {field} must be an integer")
        if not 0 <= payload[field] <= 256:
            raise LayoutMaskVLMError(f"VLM {field} is outside [0, 256]")
    if payload["profile"] == "ACT":
        if payload["period_total"] < 1 or payload["period_rows"] < 1:
            raise LayoutMaskVLMError("ACT response must include positive period counts")
        if payload["period_total"] < payload["period_rows"]:
            raise LayoutMaskVLMError("ACT period_total cannot be smaller than period_rows")
    elif payload["period_total"] != 0 or payload["period_rows"] != 0:
        raise LayoutMaskVLMError("Non-ACT response must use zero period counts")
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
    if payload["profile"] == "ACT":
        verified_grid = _estimate_act_period_grid(image) if image is not None else None
        if verified_grid is not None:
            result["period_total"] = verified_grid["period_total"]
            result["period_rows"] = verified_grid["period_rows"]
            period_columns = verified_grid["period_columns"]
        else:
            period_columns = max(
                1,
                int(round(payload["period_total"] / payload["period_rows"])),
            )
        if period_columns <= 2:
            close_kernel = 15
        elif period_columns <= 4:
            close_kernel = 9
        elif period_columns <= 7:
            close_kernel = 5
        else:
            close_kernel = 3
        parameters = dict(parameters)
        parameters["close_kernel"] = close_kernel
        result["period_columns"] = period_columns
        result["explanation"] = (
            f"VLM识别为ACT，数据中台校验为{result['period_total']}个顶帽、"
            f"{result['period_rows']}行，约{period_columns}×{result['period_rows']}周期；"
            f"数据中台按周期尺度规则采用close={close_kernel}。"
        )
    else:
        result["period_columns"] = 0
        result["explanation"] = explanation.strip()
    result["parameters"] = parameters
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
    # History is parameter context only. Previous model classifications and
    # explanations must not anchor the next turn's image-only profile decision.
    allowed = (
        "user_message",
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
        "最高优先级：先定位一个最清晰的重复单元，单元内部拓扑优先于整图排列；"
        "多个 ACT 单元横向或多行重复后仍是 ACT，不能因为整图像长带或孔洞阵列而改判 GE1/GE2。"
        "ACT 的向上高竖指、宽主体、向下端脚和开口 U 槽是组合证据；"
        "开口 U 槽不是 GE2 的封闭小方孔。"
        "profile 不能依据用户修复词、缺陷类型、当前参数或历史对话判断；"
        "每轮必须从当前图像重新分类，禁止继承历史 profile 和历史分类解释。"
        "ACT、GE1、GE2 先验相同，禁止默认 ACT 或任何其他类别；"
        "Auto 模式两个清晰且重复的类别特有证据足以选择该类，不足两个证据时才返回 Unknown。"
        "解释若写符合某类，profile 必须与该类一致。"
        "识别为 ACT 后必须统计周期：period_total 只数全图位于主体上方、带双叉/Y形顶帽且穿过主体的高竖杆总数；"
        "主体下方向下伸出、末端为单头八角垫的端脚绝对不计，左右侧翼、U槽和主体分块也不得增加 period_total；"
        "period_rows 只数这些顶帽形成的水平行数；不要自行输出每行列数，数据中台 后端用 total/rows 确定计算；"
        "ACT 单元横向相连或开口槽因缩放看似闭合时，仍由高竖指和下端脚确定 ACT，不能改判 GE2。"
        "close_kernel 按原图中的 ACT 周期密度选择：周期越多、单元越小，kernel 越小；"
        "ACT 自动分析时严格使用周期档：1-2列取15，3-4列取9，5-7列取5，8列以上取3；"
        "不能根据主观坑深越档，只有会粘连或抹掉拓扑时才可向下调并设置 manual_review；"
        "不能按 ACT 类别固定使用 15。"
        "非 ACT 的 period_total 和 period_rows 必须都为 0。"
        "profile 不授权参数基线；没有直接可见缺陷时通常保持当前参数，"
        "但当前 kernel 明显超过单元尺度上限时应降低以避免粘连。"
        "不得用“可能、常见、通常”作为修改理由。"
        "在重复单元中一致出现的 U 槽、方孔和间隙是设计拓扑，不是缺陷。\n\n"
        + skill_text
        + "\n\n只返回一个原始 JSON 对象，不要使用 Markdown 代码块。"
        + "VLM 只推荐参数；数据中台 后端执行图像处理和 mask 生成。"
    )
    user_text = json.dumps(
        {
            "request": str(message or ""),
            "profile_mode": profile_mode,
            "classification_policy": (
                "先定位单个重复单元，再比较 ACT/GE1/GE2；单元内部拓扑优先于整图排列。"
                "横向或多行重复的 ACT 仍按其高竖指、宽主体、下端脚和开口U槽识别；"
                "开口U槽不是GE2封闭方孔；修复请求不影响profile，证据不足返回Unknown"
            ),
            "parameter_policy": (
                "参数修改必须有当前图像中的直接可见缺陷证据；"
                "但当前kernel超过单元尺度上限属于直接拓扑风险，应降低；"
                "可能、常见、通常不算证据"
            ),
            "kernel_scale_policy": (
                "仅在profile=ACT后计数。period_total只数全图主体上方带双叉/Y形顶帽且穿过主体的高竖杆总数；"
                "主体下方单头八角端脚绝对不计。period_rows只数这些顶帽形成的水平行数。"
                "把两个计数写入JSON；不要自行计算columns或按坑深选择close。"
                "数据中台后端用period_total/period_rows得到每行周期数，再确定性映射close："
                "1-2列取15，3-4列取9，5-7列取5，>=8列取3。"
                "这是通用尺度规则，不按文件或样例身份查答案"
            ),
            "history_policy": (
                "历史摘要只用于参数连续性；不得从历史继承profile或分类解释。"
                "本轮profile必须仅根据当前图像重新判断"
            ),
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
        image=image,
    )
    _validate_request_constraints(validated, message)
    usage, cost = _usage_summary(result)
    return validated, usage, cost, skill_version
