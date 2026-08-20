"""Lifecycle supervisor for the isolated SAM3 model worker."""

from __future__ import annotations

import atexit
import concurrent.futures
import json
import os
import pickle
import select
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from sam3_demo.config import current_dir, runtime_log_dir
from sam3_demo.model_worker import PROTOCOL_VERSION, max_frame_bytes


_PROTOCOL_VERSION = PROTOCOL_VERSION
_MAX_FRAME_BYTES = max_frame_bytes()
_FRAME_HEADER = struct.Struct(">I")

UNLOADED = "UNLOADED"
SEARCHING_GPU = "SEARCHING_GPU"
WAITING_GPU = "WAITING_GPU"
LOADING = "LOADING"
WARMUP = "WARMUP"
READY = "READY"
RUNNING = "RUNNING"
REHYDRATING = "REHYDRATING"
STOPPING = "STOPPING"
ERROR = "ERROR"


class ModelWorkerError(RuntimeError):
    """A structured failure reported by, or while communicating with, a worker."""

    def __init__(self, message, *, code="INTERNAL"):
        super().__init__(str(message))
        self.code = str(code or "INTERNAL")


def _env_float(name, default, *, minimum=0.0):
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value) or value < float(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_int(name, default, *, minimum=0):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _read_exact(file_obj, size, deadline):
    fd = file_obj.fileno()
    chunks = []
    remaining = int(size)
    while remaining:
        timeout = max(0.0, float(deadline) - time.monotonic())
        if timeout <= 0.0:
            raise TimeoutError("SAM3 worker response timed out")
        readable, _, _ = select.select([fd], [], [], timeout)
        if not readable:
            raise TimeoutError("SAM3 worker response timed out")
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("SAM3 worker closed its response pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(file_obj, timeout):
    deadline = time.monotonic() + float(timeout)
    header = _read_exact(file_obj, _FRAME_HEADER.size, deadline)
    (length,) = _FRAME_HEADER.unpack(header)
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError(f"invalid SAM3 worker frame length: {length}")
    payload = _read_exact(file_obj, length, deadline)
    return pickle.loads(payload)


def _write_all(file_obj, payload, timeout):
    fd = file_obj.fileno()
    view = memoryview(payload)
    deadline = time.monotonic() + float(timeout)
    while view:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            raise TimeoutError("SAM3 worker request timed out")
        _, writable, _ = select.select([], [fd], [], remaining)
        if not writable:
            raise TimeoutError("SAM3 worker request timed out")
        written = os.write(fd, view)
        if written <= 0:
            raise EOFError("SAM3 worker request pipe closed")
        view = view[written:]


def _write_frame(file_obj, value, timeout):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > _MAX_FRAME_BYTES:
        raise ValueError("SAM3 worker request frame is too large")
    _write_all(file_obj, _FRAME_HEADER.pack(len(payload)) + payload, timeout)


_GPU_PROBE_SOURCE = r"""
import json
import torch

rows = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    rows.append({
        "index": index,
        "uuid": str(getattr(props, "uuid", "") or ""),
        "name": str(props.name),
        "free": int(free),
        "total": int(total),
    })
print(json.dumps(rows))
"""


class ModelRuntimeSupervisor:
    """Own a single lazily started SAM3 worker without touching CUDA locally."""

    def __init__(
        self,
        *,
        clock=None,
        probe_fn=None,
        popen_factory=None,
        start_monitor=True,
    ):
        self._clock = clock or time.monotonic
        self._probe_fn = probe_fn or self._probe_gpus
        self._popen_factory = popen_factory or subprocess.Popen
        self._condition = threading.Condition(threading.RLock())
        self._transport_lock = threading.Lock()
        self._lease_local = threading.local()
        self._monitor_stop = threading.Event()

        self._state = UNLOADED
        self._worker = None
        self._worker_log = None
        self._worker_pid = None
        self._generation = 0
        self._load_future = None
        self._loader_thread = None
        self._active_requests = 0
        self._pending_requests = 0
        self._last_inference_completed_at = None
        self._selected_gpu_uuid = None
        self._last_error = ""
        self._last_start_failure_at = None
        self._request_id = 0
        self._mask_input_size = None
        self._encoded_generation = {}
        self._restart_requested = False
        self._closed = False

        self._min_free_gpu_mib = _env_int("SAM3_MIN_FREE_GPU_MIB", 8192, minimum=0)
        self._gpu_wait_seconds = _env_float("SAM3_GPU_WAIT_SECONDS", 120, minimum=0)
        self._idle_seconds = _env_float("SAM3_IDLE_RELEASE_SECONDS", 600, minimum=0)
        self._idle_check_seconds = _env_float("SAM3_IDLE_CHECK_SECONDS", 5, minimum=0.1)
        self._startup_timeout = _env_float("SAM3_WORKER_STARTUP_TIMEOUT", 180, minimum=1)
        self._rpc_timeout = _env_float("SAM3_RPC_TIMEOUT_SECONDS", 180, minimum=1)
        self._shutdown_grace = _env_float("SAM3_WORKER_SHUTDOWN_GRACE_SECONDS", 10, minimum=0)
        self._restart_backoff = _env_float("SAM3_RESTART_BACKOFF_SECONDS", 5, minimum=0)
        self._worker_embed_cache = _env_int("SAM3_WORKER_EMBED_CACHE", 2, minimum=1)

        selection = str(os.environ.get("SAM3_GPU_SELECTION", "auto") or "auto").strip().lower()
        if selection != "auto":
            raise ValueError("SAM3_GPU_SELECTION currently supports only 'auto'")

        self._monitor_thread = None
        if start_monitor:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="sam3-model-idle-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def _snapshot_locked(self):
        return {
            "state": self._state,
            "worker_pid": self._worker_pid,
            "generation": int(self._generation),
            "active_requests": int(self._active_requests),
            "pending_requests": int(self._pending_requests),
            "last_inference_completed_at": self._last_inference_completed_at,
            "selected_gpu_uuid": self._selected_gpu_uuid,
            "last_error": self._last_error,
            "mask_input_size": self._mask_input_size,
        }

    def snapshot(self):
        with self._condition:
            return self._snapshot_locked()

    @property
    def generation(self):
        with self._condition:
            return int(self._generation)

    def _probe_gpus(self):
        result = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "GPU probe failed").strip()
            raise ModelWorkerError(detail, code="CONFIG")
        try:
            rows = json.loads(result.stdout)
        except Exception as exc:
            raise ModelWorkerError("GPU probe returned invalid JSON", code="CONFIG") from exc
        if not isinstance(rows, list):
            raise ModelWorkerError("GPU probe returned an invalid device list", code="CONFIG")
        return rows

    def _candidate_gpus(self):
        minimum = int(self._min_free_gpu_mib) * 1024 * 1024
        rows = []
        for row in self._probe_fn() or []:
            if int(row.get("free") or 0) < minimum:
                continue
            uuid = str(row.get("uuid") or "").strip()
            rows.append({**row, "uuid": uuid})
        rows.sort(key=lambda row: int(row.get("free") or 0), reverse=True)
        if len(rows) > 1 and any(not row.get("uuid") for row in rows):
            raise ModelWorkerError(
                "Multiple GPUs are visible but UUID pinning is unavailable; restrict the container to one GPU",
                code="CONFIG",
            )
        return rows

    def _worker_environment(self, candidate):
        env = os.environ.copy()
        uuid = str(candidate.get("uuid") or "").strip()
        if uuid:
            env["CUDA_VISIBLE_DEVICES"] = uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
        elif len(self._probe_fn() or []) == 1:
            # Preserve the deployment-provided single-device visibility.
            pass
        else:
            raise ModelWorkerError("Unable to pin the selected GPU safely", code="CONFIG")
        return env

    def _open_worker_log(self, generation):
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_log_dir / f"sam3_worker_generation_{int(generation)}.log"
        return open(path, "ab", buffering=0)

    def _spawn_worker(self, candidate, generation):
        log_file = self._open_worker_log(generation)
        try:
            process = self._popen_factory(
                [sys.executable, "-m", "sam3_demo.model_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                cwd=str(current_dir),
                env=self._worker_environment(candidate),
                bufsize=0,
                close_fds=True,
            )
        except Exception:
            log_file.close()
            raise
        return process, log_file

    def _rpc_process(self, process, generation, op, args=None, *, timeout=None):
        timeout = self._rpc_timeout if timeout is None else float(timeout)
        with self._transport_lock:
            if process.poll() is not None:
                raise ModelWorkerError("SAM3 worker exited unexpectedly", code="INTERNAL")
            with self._condition:
                self._request_id += 1
                request_id = int(self._request_id)
            request = {
                "protocol_version": _PROTOCOL_VERSION,
                "id": request_id,
                "generation": int(generation),
                "op": str(op),
                "args": dict(args or {}),
            }
            try:
                _write_frame(process.stdin, request, timeout)
                response = _read_frame(process.stdout, timeout)
            except Exception as exc:
                raise ModelWorkerError(str(exc), code="INTERNAL") from exc
        if not isinstance(response, dict):
            raise ModelWorkerError("SAM3 worker returned a non-object response", code="INTERNAL")
        if response.get("protocol_version") != _PROTOCOL_VERSION:
            raise ModelWorkerError("SAM3 worker response protocol mismatch", code="INTERNAL")
        if int(response.get("id", -1)) != request_id:
            raise ModelWorkerError("SAM3 worker response ID mismatch", code="INTERNAL")
        response_generation = response.get("generation")
        if response_generation is None or int(response_generation) != int(generation):
            raise ModelWorkerError("SAM3 worker response generation mismatch", code="INTERNAL")
        if not response.get("ok"):
            error = response.get("error") or {}
            raise ModelWorkerError(error.get("message") or "SAM3 worker request failed", code=error.get("code") or "INTERNAL")
        return response.get("result")

    def _stop_process(self, process, *, graceful=True):
        if process is None:
            return
        if process.poll() is not None:
            return
        if graceful:
            try:
                self._rpc_process(process, self._generation, "shutdown", timeout=self._shutdown_grace or 1)
            except Exception:
                pass
            try:
                process.wait(timeout=self._shutdown_grace)
                return
            except subprocess.TimeoutExpired:
                pass
        try:
            process.terminate()
            process.wait(timeout=max(1.0, min(5.0, self._shutdown_grace or 1.0)))
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def _detach_worker_locked(self, process):
        if self._worker is not process:
            return None
        log_file = self._worker_log
        self._worker = None
        self._worker_log = None
        self._worker_pid = None
        self._selected_gpu_uuid = None
        self._encoded_generation.clear()
        return log_file

    def _discard_candidate(self, process, log_file):
        self._stop_process(process, graceful=False)
        with self._condition:
            owned_log = self._detach_worker_locked(process)
            self._condition.notify_all()
        for item in (owned_log, log_file):
            if item is not None and not item.closed:
                item.close()

    def _set_loading_state(self, state, *, gpu_uuid=None):
        with self._condition:
            self._state = state
            self._selected_gpu_uuid = gpu_uuid
            self._condition.notify_all()

    def _load_worker(self, future):
        deadline = self._clock() + self._gpu_wait_seconds
        last_error = None
        try:
            while True:
                self._set_loading_state(SEARCHING_GPU)
                candidates = self._candidate_gpus()
                if not candidates:
                    if self._clock() >= deadline:
                        raise ModelWorkerError(
                            f"No visible GPU has at least {self._min_free_gpu_mib} MiB free",
                            code="OOM",
                        )
                    self._set_loading_state(WAITING_GPU)
                    self._monitor_stop.wait(min(2.0, max(0.0, deadline - self._clock())))
                    continue

                attempted_oom = False
                for candidate in candidates:
                    current_rows = {str(row.get("uuid") or ""): row for row in self._candidate_gpus()}
                    uuid = str(candidate.get("uuid") or "")
                    if uuid and uuid not in current_rows:
                        continue
                    next_generation = self._generation + 1
                    process = log_file = None
                    try:
                        with self._condition:
                            if self._closed:
                                raise ModelWorkerError("SAM3 supervisor is shutting down", code="INTERNAL")
                        process, log_file = self._spawn_worker(candidate, next_generation)
                        with self._condition:
                            self._worker = process
                            self._worker_log = log_file
                            self._worker_pid = process.pid
                            self._selected_gpu_uuid = uuid or None
                            self._state = LOADING
                            self._condition.notify_all()
                        capabilities = self._rpc_process(
                            process,
                            next_generation,
                            "load",
                            timeout=self._startup_timeout,
                        ) or {}
                        reported_uuid = str(capabilities.get("gpu_uuid") or "").strip()
                        if uuid and reported_uuid != uuid:
                            raise ModelWorkerError("Worker loaded a different GPU than selected", code="CONFIG")
                        mask_size = tuple(int(v) for v in capabilities.get("mask_input_size") or ())
                        if len(mask_size) != 2 or min(mask_size) <= 0:
                            raise ModelWorkerError("Worker returned invalid mask_input_size", code="CONFIG")
                        self._set_loading_state(WARMUP, gpu_uuid=uuid or reported_uuid or None)
                        self._rpc_process(
                            process,
                            next_generation,
                            "warmup",
                            timeout=self._startup_timeout,
                        )
                        with self._condition:
                            if self._worker is not process:
                                raise ModelWorkerError("Worker ownership changed during startup", code="INTERNAL")
                            self._generation = next_generation
                            self._mask_input_size = mask_size
                            self._state = READY
                            self._last_error = ""
                            self._last_start_failure_at = None
                            self._last_inference_completed_at = self._clock()
                            self._load_future = None
                            self._condition.notify_all()
                        future.set_result(self.snapshot())
                        return
                    except ModelWorkerError as exc:
                        last_error = exc
                        if process is not None:
                            self._discard_candidate(process, log_file)
                        if exc.code == "OOM":
                            attempted_oom = True
                            continue
                        raise
                    except Exception as exc:
                        if process is not None:
                            self._discard_candidate(process, log_file)
                        raise ModelWorkerError(str(exc), code="INTERNAL") from exc

                if attempted_oom and self._clock() < deadline:
                    self._set_loading_state(WAITING_GPU)
                    self._monitor_stop.wait(min(2.0, max(0.0, deadline - self._clock())))
                    continue
                raise last_error or ModelWorkerError("Unable to start a SAM3 worker", code="INTERNAL")
        except Exception as exc:
            error = exc if isinstance(exc, ModelWorkerError) else ModelWorkerError(str(exc))
            with self._condition:
                self._state = UNLOADED if self._closed else ERROR
                self._last_error = "" if self._closed else str(error)
                self._last_start_failure_at = (
                    None if self._closed else self._clock()
                )
                self._load_future = None
                self._condition.notify_all()
            if not future.done():
                future.set_exception(error)

    def _start_future_locked(self):
        if self._closed:
            failed = concurrent.futures.Future()
            failed.set_exception(
                ModelWorkerError("SAM3 supervisor is shutting down", code="INTERNAL")
            )
            return failed

        if self._state in (READY, RUNNING, REHYDRATING) and self._worker is not None:
            completed = concurrent.futures.Future()
            completed.set_result(self._snapshot_locked())
            return completed
        if self._load_future is not None:
            return self._load_future
        if self._state == STOPPING:
            self._restart_requested = True
            return None
        if self._last_start_failure_at is not None:
            elapsed = self._clock() - self._last_start_failure_at
            if elapsed < self._restart_backoff:
                failed = concurrent.futures.Future()
                failed.set_exception(
                    ModelWorkerError(
                        f"Model restart is cooling down for {self._restart_backoff - elapsed:.1f}s",
                        code="INTERNAL",
                    )
                )
                return failed
        future = concurrent.futures.Future()
        self._load_future = future
        self._state = SEARCHING_GPU
        self._last_error = ""
        thread = threading.Thread(
            target=self._load_worker,
            args=(future,),
            name="sam3-model-loader",
            daemon=True,
        )
        self._loader_thread = thread
        thread.start()
        return future

    def request_start(self):
        with self._condition:
            if self._state == ERROR:
                # Explicit user action bypasses automatic crash-loop backoff;
                # inference-triggered restarts still respect it.
                self._last_start_failure_at = None
            self._start_future_locked()
            return self._snapshot_locked()

    def _wait_until_started(self):
        while True:
            with self._condition:
                while self._state == STOPPING:
                    self._condition.wait(timeout=0.5)
                future = self._start_future_locked()
            if future is None:
                continue
            future.result(timeout=self._gpu_wait_seconds + self._startup_timeout + 30)
            return

    @contextmanager
    def lease(self, *, reason):
        depth = int(getattr(self._lease_local, "depth", 0))
        if depth:
            self._lease_local.depth = depth + 1
            try:
                yield self.snapshot()
            finally:
                self._lease_local.depth -= 1
            return

        self._lease_local.depth = 1
        active = False
        with self._condition:
            self._pending_requests += 1
            self._condition.notify_all()
        try:
            self._wait_until_started()
            with self._condition:
                self._pending_requests -= 1
                self._active_requests += 1
                active = True
                self._state = RUNNING
                self._condition.notify_all()
            yield self.snapshot()
        finally:
            with self._condition:
                if active:
                    self._active_requests -= 1
                    self._last_inference_completed_at = self._clock()
                    if self._worker is not None and self._worker.poll() is None:
                        self._state = READY if self._active_requests == 0 else RUNNING
                else:
                    self._pending_requests = max(0, self._pending_requests - 1)
                self._condition.notify_all()
            self._lease_local.depth = 0

    def _current_worker(self):
        with self._condition:
            process = self._worker
            generation = self._generation
        if process is None or process.poll() is not None:
            raise ModelWorkerError("SAM3 worker is not running", code="INTERNAL")
        return process, generation

    def _rpc(self, op, args=None, *, timeout=None):
        process, generation = self._current_worker()
        try:
            return self._rpc_process(process, generation, op, args, timeout=timeout)
        except ModelWorkerError as exc:
            if exc.code == "MISSING_EMBEDDING":
                raise
            failure = exc
        except Exception as exc:
            failure = exc
        with self._condition:
            if self._worker is process:
                self._state = ERROR
                self._last_error = str(failure)
                self._last_start_failure_at = self._clock()
                log_file = self._detach_worker_locked(process)
            else:
                log_file = None
            self._condition.notify_all()
        self._stop_process(process, graceful=False)
        if log_file is not None and not log_file.closed:
            log_file.close()
        raise failure

    def mask_input_size(self):
        with self._condition:
            cached = self._mask_input_size
        if cached is not None:
            return tuple(cached)
        with self.lease(reason="mask_input_size"):
            with self._condition:
                if self._mask_input_size is None:
                    raise ModelWorkerError("Worker did not report mask_input_size", code="CONFIG")
                return tuple(self._mask_input_size)

    @staticmethod
    def _handle_key(handle):
        return (
            str(handle.get("image_id") or ""),
            str(handle.get("target_image_sha256") or handle.get("image_sha256") or ""),
        )

    def ensure_embedding(self, handle, image_supplier):
        key = self._handle_key(handle)
        if not all(key):
            raise ValueError("workspace handle is missing image identity")
        with self._condition:
            if self._encoded_generation.get(key) == self._generation:
                return
            self._state = REHYDRATING
        image = image_supplier()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("workspace image must be RGB")
        result = self._rpc(
            "encode",
            {
                "image_id": key[0],
                "image_sha256": key[1],
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "rgb_bytes": np.ascontiguousarray(rgb).tobytes(),
            },
        )
        if not isinstance(result, dict) or str(result.get("image_id") or "") != key[0]:
            raise ModelWorkerError("Worker returned an invalid encode response", code="INTERNAL")
        with self._condition:
            self._encoded_generation[key] = self._generation
            if self._active_requests:
                self._state = RUNNING

    @staticmethod
    def _unpack_masks(result):
        shape = tuple(int(v) for v in result.get("masks_shape") or ())
        if len(shape) != 3 or min(shape) < 0:
            raise ModelWorkerError("Worker returned invalid masks_shape", code="INTERNAL")
        packed = result.get("masks_packed")
        if not isinstance(packed, (bytes, bytearray)):
            raise ModelWorkerError("Worker returned invalid packed masks", code="INTERNAL")
        k, height, width = shape
        pixels_per_mask = height * width
        packed_per_mask = (pixels_per_mask + 7) // 8
        if len(packed) != k * packed_per_mask:
            raise ModelWorkerError("Worker packed mask length mismatch", code="INTERNAL")
        if k == 0 or pixels_per_mask == 0:
            return np.zeros(shape, dtype=bool)
        packed_rows = np.frombuffer(packed, dtype=np.uint8).reshape(k, packed_per_mask)
        bits = np.unpackbits(packed_rows, axis=1, bitorder="big")[:, :pixels_per_mask]
        return bits.astype(bool, copy=False).reshape(shape)

    def _predict_with_embedding(self, op, handle, image_supplier, args):
        key = self._handle_key(handle)
        self.ensure_embedding(handle, image_supplier)
        request_args = {
            "image_id": key[0],
            "image_sha256": key[1],
            **args,
        }
        try:
            return self._rpc(op, request_args)
        except ModelWorkerError as exc:
            if exc.code != "MISSING_EMBEDDING":
                raise
        with self._condition:
            self._encoded_generation.pop(key, None)
        self.ensure_embedding(handle, image_supplier)
        return self._rpc(op, request_args)

    def predict_inst(self, handle, image_supplier, **kwargs):
        def run():
            result = self._predict_with_embedding("predict_inst", handle, image_supplier, kwargs)
            if not isinstance(result, dict):
                raise ModelWorkerError("Worker returned invalid predict_inst response", code="INTERNAL")
            masks = self._unpack_masks(result)
            scores = np.asarray(result.get("scores"), dtype=np.float32).reshape(-1)
            logits = np.asarray(result.get("lowres_logits"), dtype=np.float32)
            expected_shape = (int(handle.get("original_height") or 0), int(handle.get("original_width") or 0))
            if tuple(masks.shape[1:]) != expected_shape:
                raise ModelWorkerError("Worker mask shape does not match workspace", code="INTERNAL")
            if masks.shape[0] != scores.size or logits.shape[0] != scores.size:
                raise ModelWorkerError("Worker candidate counts do not match", code="INTERNAL")
            expected_logits = tuple(self.mask_input_size())
            if logits.ndim not in (3, 4) or tuple(logits.shape[-2:]) != expected_logits:
                raise ModelWorkerError("Worker returned invalid low-res logits shape", code="INTERNAL")
            if not np.isfinite(scores).all() or not np.isfinite(logits).all():
                raise ModelWorkerError("Worker returned non-finite prediction values", code="INTERNAL")
            return {"masks": masks, "scores": scores, "lowres_logits": logits}

        if int(getattr(self._lease_local, "depth", 0)):
            return run()
        with self.lease(reason="predict_inst"):
            return run()

    def predict_pcs(self, handle, image_supplier, **kwargs):
        def run():
            result = self._predict_with_embedding("predict_pcs", handle, image_supplier, kwargs)
            if not isinstance(result, dict):
                raise ModelWorkerError("Worker returned invalid predict_pcs response", code="INTERNAL")
            masks = self._unpack_masks(result)
            scores = np.asarray(result.get("scores"), dtype=np.float32).reshape(-1)
            boxes = np.asarray(result.get("boxes"), dtype=np.float32)
            probs_value = result.get("probs_f16")
            probs = None if probs_value is None else np.asarray(probs_value, dtype=np.float16)
            expected_shape = (int(handle.get("original_height") or 0), int(handle.get("original_width") or 0))
            if tuple(masks.shape[1:]) != expected_shape:
                raise ModelWorkerError("Worker PCS mask shape does not match workspace", code="INTERNAL")
            if masks.shape[0] != scores.size or boxes.shape != (scores.size, 4):
                raise ModelWorkerError("Worker PCS candidate counts do not match", code="INTERNAL")
            if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
                raise ModelWorkerError("Worker returned non-finite PCS values", code="INTERNAL")
            if probs is not None and (probs.shape[0] != scores.size or not np.isfinite(probs).all()):
                raise ModelWorkerError("Worker returned invalid PCS probabilities", code="INTERNAL")
            if probs is not None and tuple(probs.shape[1:]) != expected_shape:
                raise ModelWorkerError("Worker PCS probability shape does not match workspace", code="INTERNAL")
            return {"masks": masks, "scores": scores, "boxes": boxes, "probs": probs}

        if int(getattr(self._lease_local, "depth", 0)):
            return run()
        with self.lease(reason="predict_pcs"):
            return run()

    def evict(self, image_ids):
        ids = [str(value) for value in image_ids or [] if value]
        if not ids:
            return
        with self._condition:
            for key in list(self._encoded_generation):
                if key[0] in ids:
                    self._encoded_generation.pop(key, None)
            ready = self._worker is not None and self._state in (READY, RUNNING, REHYDRATING)
        if not ready:
            return
        try:
            self._rpc("evict", {"image_ids": ids})
        except Exception:
            # LRU bounds Worker memory; eviction is an optimization, not a user operation.
            return

    def _monitor_loop(self):
        while not self._monitor_stop.wait(self._idle_check_seconds):
            process_to_stop = None
            log_to_close = None
            with self._condition:
                if self._worker is not None and self._worker.poll() is not None:
                    dead = self._worker
                    log_to_close = self._detach_worker_locked(dead)
                    self._state = ERROR
                    self._last_error = "SAM3 worker exited unexpectedly"
                    self._last_start_failure_at = self._clock()
                    self._condition.notify_all()
                elif (
                    self._state == READY
                    and self._worker is not None
                    and self._active_requests == 0
                    and self._pending_requests == 0
                    and self._load_future is None
                    and self._last_inference_completed_at is not None
                    and self._clock() - self._last_inference_completed_at >= self._idle_seconds
                ):
                    self._state = STOPPING
                    process_to_stop = self._worker
                    self._condition.notify_all()
            if log_to_close is not None and not log_to_close.closed:
                log_to_close.close()
            if process_to_stop is None:
                continue
            self._stop_process(process_to_stop, graceful=True)
            with self._condition:
                log_to_close = self._detach_worker_locked(process_to_stop)
                self._state = UNLOADED
                restart = not self._closed and (self._restart_requested or self._pending_requests > 0)
                self._restart_requested = False
                self._condition.notify_all()
                if restart:
                    self._start_future_locked()
            if log_to_close is not None and not log_to_close.closed:
                log_to_close.close()

    def shutdown(self, *, graceful_timeout=None):
        self._monitor_stop.set()
        with self._condition:
            self._closed = True
            process = self._worker
            if process is not None:
                self._state = STOPPING
            self._condition.notify_all()
        if process is not None:
            original = self._shutdown_grace
            if graceful_timeout is not None:
                self._shutdown_grace = max(0.0, float(graceful_timeout))
            try:
                self._stop_process(process, graceful=True)
            finally:
                self._shutdown_grace = original
            with self._condition:
                log_file = self._detach_worker_locked(process)
                self._state = UNLOADED
                self._condition.notify_all()
            if log_file is not None and not log_file.closed:
                log_file.close()


SUPERVISOR = ModelRuntimeSupervisor()
atexit.register(SUPERVISOR.shutdown)
