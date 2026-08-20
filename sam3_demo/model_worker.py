"""Isolated SAM3 model worker and its length-prefixed RPC protocol.

The controller imports this module only for the protocol helpers.  The heavy
torch/SAM3 imports happen inside :func:`main`, after stdout has been redirected
away from the binary protocol pipe.  This module is intentionally independent
of Gradio and of the controller's model runtime.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import struct
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Mapping


PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_BYTES = 512 * 1024 * 1024
_FRAME_HEADER = struct.Struct(">I")
_ERROR_CODES = {"MISSING_EMBEDDING", "OOM", "CONFIG", "INTERNAL"}
_OPS = {
    "ping",
    "load",
    "capabilities",
    "warmup",
    "encode",
    "predict_inst",
    "predict_pcs",
    "evict",
    "shutdown",
}


class WorkerProtocolError(ValueError):
    """Raised when a frame or message does not satisfy the wire contract."""


class WorkerRPCError(RuntimeError):
    """Controller-side representation of an error returned by a worker."""

    def __init__(self, code: str, message: str, error_type: str = "RuntimeError") -> None:
        self.code = str(code)
        self.error_type = str(error_type)
        super().__init__(str(message))


def max_frame_bytes() -> int:
    """Return the configured protocol frame limit."""

    raw = os.environ.get("SAM3_WORKER_MAX_FRAME_BYTES")
    if raw is None:
        return DEFAULT_MAX_FRAME_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("SAM3_WORKER_MAX_FRAME_BYTES must be an integer") from exc
    if value < 1024:
        raise WorkerProtocolError("SAM3_WORKER_MAX_FRAME_BYTES is too small")
    return value


def dumps_message(message: Mapping[str, Any], *, max_bytes: int | None = None) -> bytes:
    """Serialize one protocol message using pickle protocol 5.

    The pipe is local to the controller and worker; it is not an untrusted
    network endpoint.  The explicit size check prevents a bad state from
    allocating an unbounded frame in either process.
    """

    payload = pickle.dumps(dict(message), protocol=5)
    limit = max_frame_bytes() if max_bytes is None else int(max_bytes)
    if len(payload) > limit:
        raise WorkerProtocolError(f"RPC frame exceeds {limit} bytes")
    return _FRAME_HEADER.pack(len(payload)) + payload


def loads_message(payload: bytes) -> dict[str, Any]:
    """Deserialize and validate a single payload (without its 4-byte header)."""

    try:
        value = pickle.loads(payload)
    except Exception as exc:  # pragma: no cover - exact pickle exception varies by Python
        raise WorkerProtocolError(f"invalid pickle payload: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("RPC message must be a dictionary")
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    """Read exactly ``size`` bytes or return ``None`` on clean EOF."""

    if size < 0:
        raise WorkerProtocolError("negative read size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise EOFError("truncated RPC frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, *, max_bytes: int | None = None) -> dict[str, Any] | None:
    """Read one length-prefixed message from ``stream``.

    ``None`` means the peer closed the pipe before sending another header.
    A partial header or payload is an error, not a clean shutdown.
    """

    header = _read_exact(stream, _FRAME_HEADER.size)
    if header is None:
        return None
    (size,) = _FRAME_HEADER.unpack(header)
    limit = max_frame_bytes() if max_bytes is None else int(max_bytes)
    if size > limit:
        raise WorkerProtocolError(f"RPC frame exceeds {limit} bytes")
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("truncated RPC frame")
    return loads_message(payload)


def write_frame(stream: BinaryIO, message: Mapping[str, Any], *, max_bytes: int | None = None) -> None:
    """Write one complete framed message and flush it."""

    stream.write(dumps_message(message, max_bytes=max_bytes))
    stream.flush()


def _request_message(request_id: int, op: str, args: Mapping[str, Any] | None, generation: int | None) -> dict[str, Any]:
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise WorkerProtocolError("request id must be an integer")
    if op not in _OPS:
        raise WorkerProtocolError(f"unsupported RPC operation: {op!r}")
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise WorkerProtocolError("RPC args must be a dictionary")
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "id": request_id,
        "op": op,
        "args": dict(args),
    }
    if generation is not None:
        result["generation"] = int(generation)
    return result


def make_request(request_id: int, op: str, args: Mapping[str, Any] | None = None, *, generation: int | None = None) -> dict[str, Any]:
    """Build and validate a request dictionary for the controller."""

    return _request_message(request_id, op, args, generation)


def _validate_request(message: Mapping[str, Any]) -> tuple[int, str, dict[str, Any], int | None]:
    version = message.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise WorkerProtocolError(f"unsupported protocol version: {version!r}")
    request_id = message.get("id")
    op = message.get("op")
    args = message.get("args", {})
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise WorkerProtocolError("request id must be an integer")
    if not isinstance(op, str) or op not in _OPS:
        raise WorkerProtocolError(f"unsupported RPC operation: {op!r}")
    if not isinstance(args, dict):
        raise WorkerProtocolError("RPC args must be a dictionary")
    generation = message.get("generation")
    if generation is not None and (isinstance(generation, bool) or not isinstance(generation, int)):
        raise WorkerProtocolError("generation must be an integer")
    return request_id, op, args, generation


def success_response(request_id: int, result: Any, *, generation: int | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "id": request_id,
        "ok": True,
        "result": result,
        "error": None,
    }
    if generation is not None:
        response["generation"] = int(generation)
    return response


def error_response(request_id: int, code: str, exc: BaseException, *, generation: int | None = None) -> dict[str, Any]:
    if code not in _ERROR_CODES:
        code = "INTERNAL"
    response: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "id": request_id,
        "ok": False,
        "result": None,
        "error": {
            "code": code,
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    if generation is not None:
        response["generation"] = int(generation)
    return response


def validate_response(response: Mapping[str, Any], request_id: int) -> Any:
    """Validate a worker response and return its result or raise an RPC error."""

    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise WorkerProtocolError("unsupported response protocol version")
    if response.get("id") != request_id:
        raise WorkerProtocolError("response id does not match request")
    if response.get("ok"):
        return response.get("result")
    error = response.get("error")
    if not isinstance(error, Mapping):
        raise WorkerProtocolError("worker error response has no error object")
    code = str(error.get("code", "INTERNAL"))
    if code not in _ERROR_CODES:
        code = "INTERNAL"
    raise WorkerRPCError(code, str(error.get("message", "worker request failed")), str(error.get("type", "RuntimeError")))


def _redirect_stdout_for_protocol() -> BinaryIO:
    """Reserve fd 1 for protocol bytes and redirect accidental prints to fd 2."""

    sys.stdout.flush()
    channel_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(channel_fd, "wb", buffering=0)


def _as_numpy(value: Any, *, dtype: Any | None = None):
    """Convert a tensor/array-like result to a detached CPU NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    else:
        import numpy as np

        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def _finite_array(value: Any, name: str, *, dtype: Any):
    import numpy as np

    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def pack_masks(masks: Any) -> tuple[bytes, list[int]]:
    """Pack full-resolution masks as bits and retain their original shape."""

    import numpy as np

    array = _as_numpy(masks)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"masks must have shape KxHxW, got {array.shape}")
    binary = np.asarray(array > 0, dtype=np.bool_)
    pixels_per_mask = int(binary.shape[1]) * int(binary.shape[2])
    packed = np.packbits(binary.reshape(binary.shape[0], pixels_per_mask), axis=1, bitorder="big")
    return packed.tobytes(), [int(v) for v in binary.shape]


def unpack_masks(packed: bytes, shape: tuple[int, int, int] | list[int]):
    """Decode and validate a packbits mask response."""

    import numpy as np

    if len(shape) != 3:
        raise ValueError("mask shape must contain K,H,W")
    k, height, width = (int(v) for v in shape)
    if min(k, height, width) < 0:
        raise ValueError("mask shape must be non-negative")
    pixels_per_mask = height * width
    packed_per_mask = (pixels_per_mask + 7) // 8
    expected = k * packed_per_mask
    if len(packed) != expected:
        raise ValueError(f"packed mask length {len(packed)} != expected {expected}")
    if k == 0 or pixels_per_mask == 0:
        return np.zeros((k, height, width), dtype=np.bool_)
    packed_rows = np.frombuffer(packed, dtype=np.uint8).reshape(k, packed_per_mask)
    unpacked = np.unpackbits(packed_rows, axis=1, bitorder="big")[:, :pixels_per_mask]
    return unpacked.reshape((k, height, width)).astype(bool, copy=False)


@dataclass
class _Embedding:
    image_sha256: str
    width: int
    height: int
    state: dict[str, Any]
    last_used: float


class Sam3WorkerRuntime:
    """Lazy, single-threaded SAM3 runtime owned by the worker process."""

    def __init__(self, *, device: str | None = None, embedding_limit: int | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.device = device or os.environ.get("SAM3_WORKER_DEVICE", "cuda:0")
        self.embedding_limit = int(embedding_limit or os.environ.get("SAM3_WORKER_EMBED_CACHE", "2"))
        if self.embedding_limit < 1:
            raise ValueError("embedding cache limit must be positive")
        self.clock = clock
        self.model = None
        self.processor = None
        self._torch = None
        self._np = None
        self._capabilities: dict[str, Any] | None = None
        self._embeddings: OrderedDict[str, _Embedding] = OrderedDict()
        self._shutdown_requested = False

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def _import_runtime(self) -> None:
        if self._torch is not None:
            return
        import numpy as np
        import torch

        self._np = np
        self._torch = torch

    def _checkpoint_paths(self):
        from sam3_demo.config import current_dir

        return current_dir / "models" / "sam3.pt", current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    def _is_oom(self, exc: BaseException) -> bool:
        torch = self._torch
        if torch is not None and isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
            return True
        return "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower() and "memory" in str(exc).lower()

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError("worker model is not loaded")

    def load(self) -> dict[str, Any]:
        if self.loaded:
            return dict(self._capabilities or {})
        checkpoint_path, bpe_path = self._checkpoint_paths()
        if not checkpoint_path.exists():
            raise WorkerRPCError("CONFIG", f"model checkpoint not found: {checkpoint_path}", "FileNotFoundError")
        if not bpe_path.exists():
            raise WorkerRPCError("CONFIG", f"BPE vocabulary not found: {bpe_path}", "FileNotFoundError")
        try:
            self._import_runtime()
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            # The builder installs a CUDA autocast context. On torch 2.7,
            # spelling the pinned logical device as ``cuda:0`` leaves that
            # context mismatched; ``cuda`` still resolves exclusively to the
            # Worker's single visible logical device zero.
            build_device = "cuda" if self.device == "cuda:0" else self.device

            image_model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path),
                bpe_path=str(bpe_path),
                device=build_device,
                enable_inst_interactivity=True,
            )
            self.model = image_model
            self.processor = Sam3Processor(image_model, device=build_device)
            size = tuple(int(v) for v in image_model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
            gpu_name = str(self._torch.cuda.get_device_name(0)) if self._torch.cuda.is_available() else str(self.device)
            gpu_uuid = None
            try:
                gpu_uuid = str(self._torch.cuda.get_device_properties(0).uuid)
            except Exception:
                pass
            self._capabilities = {
                "mask_input_size": [size[0], size[1]],
                "gpu_name": gpu_name,
                "gpu_uuid": gpu_uuid,
                "device": str(self.device),
            }
            return dict(self._capabilities)
        except WorkerRPCError:
            raise
        except Exception as exc:
            self.model = None
            self.processor = None
            self._capabilities = None
            if self._is_oom(exc):
                raise WorkerRPCError("OOM", str(exc), type(exc).__name__) from exc
            raise

    def capabilities(self) -> dict[str, Any]:
        if self._capabilities is None:
            return {"loaded": False, "mask_input_size": None, "gpu_name": None, "gpu_uuid": None}
        result = dict(self._capabilities)
        result["loaded"] = bool(self.loaded)
        return result

    def _touch(self, image_id: str) -> _Embedding:
        entry = self._embeddings.get(image_id)
        if entry is None:
            raise WorkerRPCError("MISSING_EMBEDDING", f"embedding not found for image {image_id!r}", "KeyError")
        entry.last_used = self.clock()
        self._embeddings.move_to_end(image_id)
        return entry

    def _store_embedding(self, image_id: str, entry: _Embedding) -> None:
        self._embeddings.pop(image_id, None)
        self._embeddings[image_id] = entry
        while len(self._embeddings) > self.embedding_limit:
            self._embeddings.popitem(last=False)

    def _prediction_state(self, entry: _Embedding) -> dict[str, Any]:
        # Sam3Processor.set_text_prompt mutates backbone_out in place.  A top-
        # level copy gives every prediction a private prompt namespace while
        # retaining the nested feature tensors in the embedding cache.
        cached = entry.state
        return {
            "original_height": int(cached["original_height"]),
            "original_width": int(cached["original_width"]),
            "backbone_out": dict(cached["backbone_out"]),
        }

    def encode(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_loaded()
        image_id = str(args.get("image_id", ""))
        image_sha256 = str(args.get("image_sha256", ""))
        width = int(args.get("width", 0))
        height = int(args.get("height", 0))
        rgb_bytes = args.get("rgb_bytes")
        if not image_id or width <= 0 or height <= 0:
            raise ValueError("encode requires image_id, positive width and height")
        if not isinstance(rgb_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("rgb_bytes must be bytes")
        raw = bytes(rgb_bytes)
        expected = width * height * 3
        if len(raw) != expected:
            raise ValueError(f"rgb_bytes length {len(raw)} != expected {expected}")
        actual_hash = hashlib.sha256(raw).hexdigest()
        if image_sha256 and image_sha256 != actual_hash:
            raise ValueError("image_sha256 does not match rgb_bytes")
        image_sha256 = image_sha256 or actual_hash
        try:
            from PIL import Image

            image = Image.fromarray(self._np.frombuffer(raw, dtype=self._np.uint8).reshape(height, width, 3), mode="RGB")
            state = self.processor.set_image(image)
        except Exception as exc:
            if self._is_oom(exc):
                raise WorkerRPCError("OOM", str(exc), type(exc).__name__) from exc
            raise
        entry = _Embedding(image_sha256, width, height, state, self.clock())
        self._store_embedding(image_id, entry)
        return {"image_id": image_id, "image_sha256": image_sha256, "width": width, "height": height}

    def _common_inst_kwargs(self, args: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        kwargs: dict[str, Any] = {"multimask_output": True, "return_logits": True}
        box = args.get("box_xyxy_px")
        if box is not None:
            box_array = _finite_array(box, "box_xyxy_px", dtype=np.float32).reshape(-1)
            if box_array.shape != (4,):
                raise ValueError("box_xyxy_px must contain four values")
            kwargs["box"] = box_array
        point_coords = args.get("point_coords_px")
        if point_coords is not None:
            coords = _finite_array(point_coords, "point_coords_px", dtype=np.float32)
            if coords.ndim == 1:
                coords = coords[None, :]
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("point_coords_px must have shape Nx2")
            labels = np.ones((coords.shape[0],), dtype=np.int64) if args.get("point_labels") is None else np.asarray(args["point_labels"], dtype=np.int64).reshape(-1)
            if labels.shape[0] != coords.shape[0]:
                raise ValueError("point_labels length must match point_coords_px")
            kwargs["point_coords"] = coords
            kwargs["point_labels"] = labels
        mask_input = args.get("mask_input_lowres_logits")
        if mask_input is not None:
            logits = _finite_array(mask_input, "mask_input_lowres_logits", dtype=np.float32)
            if logits.ndim == 2:
                logits = logits[None, :, :]
            expected = tuple(int(v) for v in (self._capabilities or {}).get("mask_input_size", []))
            if logits.ndim != 3 or len(expected) != 2 or tuple(logits.shape[-2:]) != expected:
                raise ValueError(f"mask_input_lowres_logits must have shape 1x{expected[0]}x{expected[1]}")
            kwargs["mask_input"] = logits
        return kwargs

    def predict_inst(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_loaded()
        image_id = str(args.get("image_id", ""))
        image_sha256 = str(args.get("image_sha256", ""))
        entry = self._touch(image_id)
        if image_sha256 and image_sha256 != entry.image_sha256:
            raise WorkerRPCError("MISSING_EMBEDDING", f"embedding hash mismatch for image {image_id!r}", "ValueError")
        kwargs = self._common_inst_kwargs(args)
        try:
            masks, scores, lowres_logits = self.model.predict_inst(self._prediction_state(entry), **kwargs)
        except Exception as exc:
            if self._is_oom(exc):
                raise WorkerRPCError("OOM", str(exc), type(exc).__name__) from exc
            raise
        packed, shape = pack_masks(masks)
        scores_np = _finite_array(_as_numpy(scores), "scores", dtype=self._np.float32).reshape(-1)
        logits_np = _finite_array(_as_numpy(lowres_logits), "lowres_logits", dtype=self._np.float32)
        return {"masks_packed": packed, "masks_shape": shape, "scores": scores_np, "lowres_logits": logits_np}

    def predict_pcs(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_loaded()
        image_id = str(args.get("image_id", ""))
        image_sha256 = str(args.get("image_sha256", ""))
        entry = self._touch(image_id)
        if image_sha256 and image_sha256 != entry.image_sha256:
            raise WorkerRPCError("MISSING_EMBEDDING", f"embedding hash mismatch for image {image_id!r}", "ValueError")
        text = str(args.get("text", "") or "").strip()
        positive_boxes = args.get("positive_boxes_cxcywh")
        negative_boxes = args.get("negative_boxes_cxcywh")
        if positive_boxes is None:
            positive_boxes = []
        if negative_boxes is None:
            negative_boxes = []
        threshold = float(args.get("threshold", 0.5))
        if not self._np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        state = self._prediction_state(entry)
        try:
            if text:
                state = self.processor.set_text_prompt(text, state)
            for box in positive_boxes:
                values = _finite_array(box, "positive_boxes_cxcywh", dtype=self._np.float32).reshape(-1)
                if values.shape != (4,):
                    raise ValueError("positive box must contain four values")
                state = self.processor.add_geometric_prompt(values, True, state)
            for box in negative_boxes:
                values = _finite_array(box, "negative_boxes_cxcywh", dtype=self._np.float32).reshape(-1)
                if values.shape != (4,):
                    raise ValueError("negative box must contain four values")
                state = self.processor.add_geometric_prompt(values, False, state)
            state = self.processor.set_confidence_threshold(threshold, state)
        except Exception as exc:
            if self._is_oom(exc):
                raise WorkerRPCError("OOM", str(exc), type(exc).__name__) from exc
            raise
        masks = state.get("masks")
        height, width = entry.height, entry.width
        if masks is None:
            masks_np = self._np.zeros((0, height, width), dtype=self._np.bool_)
        else:
            masks_np = _as_numpy(masks)
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0]
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]
            masks_np = masks_np.astype(bool, copy=False)
            if masks_np.ndim != 3:
                raise ValueError(f"PCS masks must have shape NxHxW, got {masks_np.shape}")
        packed, shape = pack_masks(masks_np)
        probs = state.get("masks_logits")
        probs_np = None if probs is None else _as_numpy(probs, dtype=self._np.float16)
        if probs_np is not None and probs_np.ndim == 4:
            probs_np = probs_np[:, 0]
        if probs_np is not None:
            if probs_np.ndim != 3 or probs_np.shape[0] != masks_np.shape[0]:
                raise ValueError("PCS probability shape does not match masks")
            if not self._np.isfinite(probs_np).all():
                raise ValueError("PCS probabilities contain non-finite values")
        boxes = _as_numpy(state.get("boxes", self._np.zeros((0, 4), dtype=self._np.float32)), dtype=self._np.float32).reshape(-1, 4)
        scores = _finite_array(_as_numpy(state.get("scores", self._np.zeros((0,), dtype=self._np.float32))), "scores", dtype=self._np.float32).reshape(-1)
        if boxes.shape[0] != masks_np.shape[0] or scores.shape[0] != masks_np.shape[0]:
            raise ValueError("PCS output counts do not match masks")
        return {"masks_packed": packed, "masks_shape": shape, "probs_f16": probs_np, "boxes": boxes, "scores": scores}

    def warmup(self) -> dict[str, Any]:
        self._require_loaded()
        import numpy as np

        warm_id = f"__warmup__{os.getpid()}"
        started = self.clock()
        try:
            raw = np.zeros((64, 64, 3), dtype=np.uint8)
            self.encode({"image_id": warm_id, "image_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "width": 64, "height": 64, "rgb_bytes": raw.tobytes()})
            self.predict_inst({"image_id": warm_id, "image_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "box_xyxy_px": [8, 8, 56, 56]})
        finally:
            self._embeddings.pop(warm_id, None)
        return {"elapsed_seconds": float(self.clock() - started)}

    def evict(self, image_ids: Any) -> dict[str, Any]:
        ids = image_ids if isinstance(image_ids, (list, tuple, set)) else []
        removed = []
        for image_id in ids:
            key = str(image_id)
            if self._embeddings.pop(key, None) is not None:
                removed.append(key)
        return {"removed": removed}

    def shutdown(self) -> dict[str, Any]:
        self._shutdown_requested = True
        self._embeddings.clear()
        self.processor = None
        self.model = None
        return {"shutdown": True}

    def dispatch(self, op: str, args: Mapping[str, Any]) -> Any:
        if op == "ping":
            return {"pid": os.getpid()}
        if op == "load":
            return self.load()
        if op == "capabilities":
            return self.capabilities()
        if op == "warmup":
            return self.warmup()
        if op == "encode":
            return self.encode(args)
        if op == "predict_inst":
            return self.predict_inst(args)
        if op == "predict_pcs":
            return self.predict_pcs(args)
        if op == "evict":
            return self.evict(args.get("image_ids", []))
        if op == "shutdown":
            return self.shutdown()
        raise WorkerProtocolError(f"unsupported RPC operation: {op!r}")


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, WorkerRPCError):
        return exc.code
    if isinstance(exc, FileNotFoundError):
        return "CONFIG"
    return "INTERNAL"


def serve(stream_in: BinaryIO, stream_out: BinaryIO, runtime: Sam3WorkerRuntime | None = None, *, max_bytes: int | None = None) -> None:
    """Serve requests until EOF or a shutdown operation."""

    runtime = runtime or Sam3WorkerRuntime()
    while not runtime._shutdown_requested:
        try:
            message = read_frame(stream_in, max_bytes=max_bytes)
            if message is None:
                return
            request_id, op, args, generation = _validate_request(message)
            try:
                result = runtime.dispatch(op, args)
                response = success_response(request_id, result, generation=generation)
            except BaseException as exc:  # worker must always answer a valid frame
                response = error_response(request_id, _error_code(exc), exc, generation=generation)
            write_frame(stream_out, response, max_bytes=max_bytes)
            if op == "shutdown":
                return
        except BaseException as exc:
            # A malformed request has no trustworthy ID.  Report a protocol
            # error only when an ID can still be recovered; otherwise exit so
            # the controller treats EOF as a failed generation.
            print(f"worker protocol failure: {exc}", file=sys.stderr)
            return


def main() -> None:
    channel_out = _redirect_stdout_for_protocol()
    try:
        # Importing torch/SAM3 before the redirect would allow their startup
        # prints to corrupt the framed stdout stream.
        serve(sys.stdin.buffer, channel_out)
    finally:
        try:
            channel_out.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
