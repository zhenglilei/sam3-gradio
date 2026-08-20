"""Authoritative image caches and workspace identity helpers."""

from __future__ import annotations

import base64 as _sam3_base64
import gc
import io
import threading
import time
import uuid

import numpy as np
from PIL import Image

import image_crop_utils as _image_crop
import layout_transform_utils as _layout_tx
from sam3_demo.config import (
    _SOURCE_IMAGE_CACHE_MAX_ENTRIES,
    _SOURCE_IMAGE_CACHE_TTL_SECONDS,
    _WORKSPACE_CACHE_MAX_ENTRIES,
    _WORKSPACE_CACHE_TTL_SECONDS,
)


_WORKSPACE_CACHE = {}
_WORKSPACE_CACHE_LOCK = threading.RLock()
_SOURCE_IMAGE_CACHE = {}
_SOURCE_IMAGE_CACHE_LOCK = threading.RLock()


def _release_workspace_memory():
    gc.collect()


def _evict_workspace_images(image_ids):
    ids = [str(value) for value in image_ids or [] if value]
    if not ids:
        return
    from sam3_demo.model_supervisor import SUPERVISOR

    SUPERVISOR.evict(ids)

def _prune_workspace_cache(now=None, protected_image_id=None):
    now = time.monotonic() if now is None else float(now)
    protected = str(protected_image_id) if protected_image_id else None
    removed = []
    expired = [
        image_id
        for image_id, workspace in _WORKSPACE_CACHE.items()
        if now - float(workspace.get("last_accessed_at", now))
        > _WORKSPACE_CACHE_TTL_SECONDS
    ]
    for image_id in expired:
        if _WORKSPACE_CACHE.pop(image_id, None) is not None:
            removed.append(image_id)

    while len(_WORKSPACE_CACHE) > _WORKSPACE_CACHE_MAX_ENTRIES:
        candidates = [
            (image_id, workspace)
            for image_id, workspace in _WORKSPACE_CACHE.items()
            if image_id != protected
        ]
        if not candidates:
            break
        oldest_id, _ = min(
            candidates,
            key=lambda item: (
                float(item[1].get("last_accessed_at", now)),
                item[0],
            ),
        )
        if _WORKSPACE_CACHE.pop(oldest_id, None) is not None:
            removed.append(oldest_id)
    return removed

def _clear_workspace_cache(session_id=None):
    with _WORKSPACE_CACHE_LOCK:
        if session_id is None:
            removed = list(_WORKSPACE_CACHE)
            _WORKSPACE_CACHE.clear()
        else:
            session_id = str(session_id)
            removed = [
                image_id
                for image_id, workspace in _WORKSPACE_CACHE.items()
                if workspace.get("session_id") == session_id
            ]
            for image_id in removed:
                _WORKSPACE_CACHE.pop(image_id, None)
    if not removed:
        return []
    _release_workspace_memory()
    _evict_workspace_images(removed)
    return removed

def _prune_source_image_cache(now=None, protected_source_id=None):
    now = time.monotonic() if now is None else float(now)
    protected = str(protected_source_id) if protected_source_id else None
    expired = [
        source_id
        for source_id, item in _SOURCE_IMAGE_CACHE.items()
        if now - float(item.get("last_accessed_at", now)) > _SOURCE_IMAGE_CACHE_TTL_SECONDS
    ]
    for source_id in expired:
        _SOURCE_IMAGE_CACHE.pop(source_id, None)
    while len(_SOURCE_IMAGE_CACHE) > _SOURCE_IMAGE_CACHE_MAX_ENTRIES:
        candidates = [
            (source_id, item)
            for source_id, item in _SOURCE_IMAGE_CACHE.items()
            if source_id != protected
        ]
        if not candidates:
            break
        oldest_id, _ = min(
            candidates,
            key=lambda item: (float(item[1].get("last_accessed_at", now)), item[0]),
        )
        _SOURCE_IMAGE_CACHE.pop(oldest_id, None)

def _clear_source_image_cache(session_id=None):
    with _SOURCE_IMAGE_CACHE_LOCK:
        if session_id is None:
            _SOURCE_IMAGE_CACHE.clear()
            return
        session_id = str(session_id)
        for source_id in [
            key
            for key, item in _SOURCE_IMAGE_CACHE.items()
            if item.get("session_id") == session_id
        ]:
            _SOURCE_IMAGE_CACHE.pop(source_id, None)

def _source_image_cache_put(session_id, image):
    session_id = str(session_id)
    source = _pil_image(image)
    if source is None:
        raise ValueError("Upload a source image first")
    source_id = uuid.uuid4().hex
    source_hash = _layout_tx.image_pixel_sha256(source)
    now = time.monotonic()
    with _SOURCE_IMAGE_CACHE_LOCK:
        for old_id in [
            key
            for key, item in _SOURCE_IMAGE_CACHE.items()
            if item.get("session_id") == session_id
        ]:
            _SOURCE_IMAGE_CACHE.pop(old_id, None)
        _SOURCE_IMAGE_CACHE[source_id] = {
            "image": source.copy(),
            "session_id": session_id,
            "source_image_sha256": source_hash,
            "created_at": now,
            "last_accessed_at": now,
        }
        _prune_source_image_cache(now, protected_source_id=source_id)
    whole = list(_image_crop.whole_image_crop_box(source.width, source.height))
    return {
        "session_id": session_id,
        "source_image_id": source_id,
        "source_image_sha256": source_hash,
        "source_width": source.width,
        "source_height": source.height,
        "source_revision": 1,
        "crop_bbox_xyxy": whole,
        "pending_crop_bbox_xyxy": None,
        "workspace_image_id": None,
        "workspace_hash": None,
    }

def _source_image_cache_get(source_state):
    if not isinstance(source_state, dict) or not source_state.get("source_image_id"):
        raise ValueError("请先上传完整原图")
    source_id = str(source_state["source_image_id"])
    session_id = str(source_state.get("session_id") or "")
    source_hash = str(source_state.get("source_image_sha256") or "")
    now = time.monotonic()
    with _SOURCE_IMAGE_CACHE_LOCK:
        _prune_source_image_cache(now, protected_source_id=source_id)
        cached = _SOURCE_IMAGE_CACHE.get(source_id)
        if cached is not None:
            cached["last_accessed_at"] = now
    if cached is None:
        raise ValueError("完整原图缓存已过期，请重新上传")
    if cached.get("session_id") != session_id:
        raise ValueError("完整原图不属于当前会话")
    if not source_hash or cached.get("source_image_sha256") != source_hash:
        raise ValueError("完整原图 hash 不匹配，请重新上传")
    image = cached["image"]
    if image.width != int(source_state.get("source_width") or 0) or image.height != int(source_state.get("source_height") or 0):
        raise ValueError("完整原图尺寸不匹配，请重新上传")
    return image.copy()

def _image_gesture_payload(*, enabled, width, height, image_id, image_sha256, revision, interaction, crop_bbox_xyxy=None, selection_state="", status=""):
    payload = {
        "server_view": {
            "enabled": bool(enabled),
            "natural_width": int(width or 0),
            "natural_height": int(height or 0),
            "image_id": str(image_id or ""),
            "image_sha256": str(image_sha256 or ""),
            "revision": int(revision or 0),
            "interaction": str(interaction or "disabled"),
            "selection_state": str(selection_state or ""),
            "status": str(status or ""),
        },
        "client_intent": {},
    }
    if crop_bbox_xyxy and len(crop_bbox_xyxy) == 4:
        payload["client_intent"] = {
            "gesture": "drag",
            "start_xy": [float(crop_bbox_xyxy[0]), float(crop_bbox_xyxy[1])],
            "end_xy": [float(crop_bbox_xyxy[2]), float(crop_bbox_xyxy[3])],
            "expected_revision": int(revision or 0),
            "image_id": str(image_id or ""),
            "image_sha256": str(image_sha256 or ""),
        }
    return payload

def _source_gesture_payload(source_state, status="", retain_selection=True):
    state = source_state if isinstance(source_state, dict) else {}
    enabled = bool(state.get("source_image_id"))
    selection = None
    selection_state = ""
    if enabled and retain_selection:
        pending = state.get("pending_crop_bbox_xyxy")
        if isinstance(pending, (list, tuple)) and len(pending) == 4:
            selection = pending
            selection_state = "draft"
        else:
            applied = state.get("crop_bbox_xyxy")
            whole = list(
                _image_crop.whole_image_crop_box(
                    int(state.get("source_width") or 0),
                    int(state.get("source_height") or 0),
                )
            )
            if (
                isinstance(applied, (list, tuple))
                and len(applied) == 4
                and list(applied) != whole
            ):
                selection = applied
                selection_state = "applied"
    return _image_gesture_payload(
        enabled=enabled,
        width=state.get("source_width"),
        height=state.get("source_height"),
        image_id=state.get("source_image_id"),
        image_sha256=state.get("source_image_sha256"),
        revision=state.get("source_revision"),
        interaction="crop" if enabled else "disabled",
        crop_bbox_xyxy=selection,
        selection_state=selection_state,
        status=status,
    )

def _validate_gesture_intent(payload, identity_state, *, allowed_gestures):
    if not isinstance(payload, dict):
        raise ValueError("交互 payload 无效")
    gesture = str(payload.get("gesture") or "")
    if gesture not in set(allowed_gestures):
        raise ValueError("当前交互手势无效")
    expected_revision = payload.get("expected_revision")
    actual_revision = int(identity_state.get("interaction_revision") or identity_state.get("source_revision") or 0)
    if isinstance(expected_revision, bool) or expected_revision != actual_revision:
        raise ValueError("交互 revision 已过期，请重试")
    expected_id = str(identity_state.get("image_id") or identity_state.get("source_image_id") or "")
    expected_hash = str(identity_state.get("target_image_sha256") or identity_state.get("source_image_sha256") or "")
    if str(payload.get("image_id") or "") != expected_id:
        raise ValueError("交互图像 identity 已过期，请重试")
    if str(payload.get("image_sha256") or "") != expected_hash:
        raise ValueError("交互图像 hash 已过期，请重试")
    start = payload.get("start_xy")
    end = payload.get("end_xy")
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError("交互起点无效")
    if not isinstance(end, (list, tuple)) or len(end) != 2:
        raise ValueError("交互终点无效")
    values = np.asarray([*start, *end], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("交互坐标必须是有限数值")
    return gesture, values[:2].tolist(), values[2:].tolist()

def _pil_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")

def _data_url(image):
    image = _pil_image(image)
    if image is None:
        return ""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + _sam3_base64.b64encode(buf.getvalue()).decode("ascii")

def _workspace(image_state):
    if not image_state or not image_state.get("image_id"):
        raise ValueError("Load an image first")
    image_id = str(image_state["image_id"])
    session_id = str(image_state.get("session_id") or "")
    if not session_id:
        raise ValueError("Image state session is missing; reload the image")
    now = time.monotonic()
    with _WORKSPACE_CACHE_LOCK:
        removed = _prune_workspace_cache(now, protected_image_id=image_id)
        ws = _WORKSPACE_CACHE.get(image_id)
        if ws is not None:
            ws["last_accessed_at"] = now
    if removed:
        _release_workspace_memory()
        _evict_workspace_images(removed)
    if ws is None:
        raise ValueError("Image state expired; reload the image")
    if ws.get("session_id") != session_id:
        raise ValueError("Image state does not belong to the current session")
    expected_hash = str(image_state.get("target_image_sha256") or "")
    if not expected_hash or ws.get("target_image_sha256") != expected_hash:
        raise ValueError("Image state hash does not match the cached image")
    return ws

def _fresh_state(image_state):
    ws = _workspace(image_state)
    image = ws["image"]
    from sam3_demo.model_supervisor import SUPERVISOR

    return {
        "image_id": str(image_state.get("image_id") or ""),
        "session_id": str(image_state.get("session_id") or ""),
        "target_image_sha256": str(image_state.get("target_image_sha256") or ""),
        "original_width": int(image.width),
        "original_height": int(image.height),
        "generation": SUPERVISOR.generation,
    }


def _workspace_image_for_handle(handle):
    image_state = {
        "image_id": str((handle or {}).get("image_id") or ""),
        "session_id": str((handle or {}).get("session_id") or ""),
        "target_image_sha256": str((handle or {}).get("target_image_sha256") or ""),
    }
    ws = _workspace(image_state)
    image = ws["image"]
    expected = (
        int((handle or {}).get("original_width") or 0),
        int((handle or {}).get("original_height") or 0),
    )
    if expected != image.size:
        raise ValueError("Workspace image dimensions changed; reload the image")
    return image.copy()
