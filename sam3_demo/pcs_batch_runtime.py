"""Bounded PCS concurrency inside the model worker, never the web process."""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pickle

from sam3_demo.model_worker import max_frame_bytes


_LOG = logging.getLogger(__name__)
_GIB = 1024 ** 3


class _CheckedPCSModel:
    def __init__(self, model, torch):
        self._model = model
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward_grounding(self, *args, **kwargs):
        outputs = self._model.forward_grounding(*args, **kwargs)
        # Check before score filtering and mask thresholding can hide NaNs.
        for key in ("pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec"):
            if not self._torch.isfinite(outputs[key]).all().item():
                raise FloatingPointError(f"PCS {key} contains non-finite values")
        return outputs


class PCSBatchRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.limit = min(4, max(1, int(os.environ.get("SAM3_PCS_BATCH_CONCURRENCY", "4"))))
        self.executor = None
        self.streams = []

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.executor = None
        self.streams.clear()

    def _lanes(self, items, requested):
        maximum = min(len(items), self.limit, max(1, int(requested)))
        torch = self.runtime._torch
        if not self.runtime.device.startswith("cuda"):
            return 1
        free, _ = torch.cuda.mem_get_info(self.runtime.device)
        reusable = max(0, torch.cuda.memory_reserved(self.runtime.device) - torch.cuda.memory_allocated(self.runtime.device))
        candidates = int(self.runtime._capabilities["pcs_max_candidates"])
        pixels = max(int(item["width"]) * int(item["height"]) for item in items)
        # Leave headroom for other users and full-resolution mask interpolation.
        per_lane = int(1.5 * _GIB) + pixels * candidates * 6
        affordable = max(1, int((free + reusable - 2 * _GIB) // per_lane))
        return min(maximum, affordable)

    def _ensure_pool(self):
        torch = self.runtime._torch
        if self.executor is None:
            self.streams = [torch.cuda.Stream(device=self.runtime.device) for _ in range(4)]
            self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pcs-lane")
        # The builder and previous RPCs may have produced tensors on other streams.
        torch.cuda.synchronize(self.runtime.device)

    def _is_capacity_error(self, exc):
        checker = getattr(self.runtime, "_is_oom", None)
        if callable(checker) and checker(exc):
            return True
        text = str(exc).lower()
        return "out of memory" in text or ("cuda error" in text and "memory" in text)

    @staticmethod
    def _is_cuda_context_failure(exc):
        text = str(exc).lower()
        if "cuda" not in text:
            return False
        return any(marker in text for marker in (
            "illegal memory access",
            "device-side assert",
            "launch failure",
            "misaligned address",
            "unspecified launch failure",
        ))

    def _predict_one(self, item, lane=None):
        from contextlib import nullcontext
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor

        runtime = self.runtime
        torch = runtime._torch
        image = Image.frombytes("RGB", (int(item["width"]), int(item["height"])), item["rgb_bytes"])
        stream = self.streams[lane] if lane is not None else None
        stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
        is_cuda = runtime.device.startswith("cuda")
        with stream_context, torch.inference_mode(), torch.autocast(
            "cuda" if is_cuda else "cpu", dtype=torch.bfloat16,
            enabled=is_cuda, cache_enabled=False,
        ):
            # Both find_stage tensors and prompt state are private to this request.
            processor = Sam3Processor(_CheckedPCSModel(runtime.model, torch), device=runtime.device)
            state = processor.set_image(image)
            result = runtime._pcs_from_state(processor, state, item)
            if stream is not None:
                stream.synchronize()
            return result

    def _attempt(self, item, lane=None):
        try:
            return {"prediction": self._predict_one(item, lane), "error": None}
        except Exception as exc:
            # Leave a broken CUDA context so normal restart handling is not hidden.
            if self._is_cuda_context_failure(exc):
                raise
            _LOG.warning("PCS batch item failed: %s", type(exc).__name__)
            if self._is_capacity_error(exc):
                failure_kind = "capacity"
            elif isinstance(exc, FloatingPointError):
                failure_kind = "parallel"
            else:
                failure_kind = ""
            return {
                "prediction": None,
                "error": f"PCS inference failed ({type(exc).__name__})",
                "_failure_kind": failure_kind,
            }

    def predict(self, args):
        items = list(args.get("items") or [])
        if not 1 <= len(items) <= 4:
            raise ValueError("PCS batch requires one to four images")
        lanes = self._lanes(items, args.get("concurrency", 4))
        fallback = lanes < min(len(items), self.limit, int(args.get("concurrency", 4)))
        reason = "memory_limit" if fallback else ""
        results = []
        if lanes > 1:
            self._ensure_pool()
            for offset in range(0, len(items), lanes):
                wave = items[offset:offset + lanes]
                futures = [self.executor.submit(self._attempt, item, lane) for lane, item in enumerate(wave)]
                results.extend(future.result() for future in futures)
            recoverable = [
                i for i, result in enumerate(results)
                if result.get("_failure_kind") in {"capacity", "parallel"}
            ]
            if recoverable:
                # Retry only failures explained by capacity or parallel
                # numerical state. Ordinary item errors remain item errors.
                kinds = {results[index].get("_failure_kind") for index in recoverable}
                fallback = True
                reason = "memory_limit" if "capacity" in kinds else "parallel_failure"
                self.runtime._torch.cuda.synchronize(self.runtime.device)
                self.runtime._torch.cuda.empty_cache()
                for index in recoverable:
                    results[index] = self._attempt(items[index])
        else:
            results = [self._attempt(item) for item in items]
        for result in results:
            result.pop("_failure_kind", None)
        response = {"items": results, "concurrency": lanes, "fallback": fallback, "reason": reason}
        if len(pickle.dumps(response, protocol=5)) > max_frame_bytes() - 4096:
            if len(items) > 1:
                return {"items": [], "concurrency": 1, "fallback": True, "reason": "frame_limit"}
            response["items"] = [{"prediction": None, "error": "PCS result exceeds the worker transfer limit"}]
        _LOG.info("PCS batch: count=%d concurrency=%d fallback=%s reason=%s", len(items), lanes, fallback, reason)
        return response
