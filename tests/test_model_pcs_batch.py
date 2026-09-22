import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from sam3_demo import pcs_batch_runtime as batch_runtime
from sam3_demo.model_supervisor import ModelRuntimeSupervisor
from sam3_demo.model_worker import pack_masks


def prediction(width=3, height=2):
    masks = np.ones((1, height, width), dtype=bool)
    packed, shape = pack_masks(masks)
    return {"masks_packed": packed, "masks_shape": shape,
            "probs_f16": masks.astype(np.float16),
            "boxes": np.array([[0, 0, width, height]], dtype=np.float32),
            "scores": np.array([0.8], dtype=np.float32)}


def item():
    return {"image": Image.new("RGB", (3, 2)), "text": "circle",
            "positive_boxes_cxcywh": [], "negative_boxes_cxcywh": [], "threshold": 0.5}


class BatchRuntimeTests(unittest.TestCase):
    def runtime(self, free=20 * 1024 ** 3):
        cuda = SimpleNamespace(mem_get_info=lambda device: (free, 24 * 1024 ** 3),
                               memory_reserved=lambda device: 0, memory_allocated=lambda device: 0,
                               synchronize=Mock(), empty_cache=Mock())
        runtime = SimpleNamespace(device="cuda:0", _torch=SimpleNamespace(cuda=cuda),
                                  _capabilities={"pcs_max_candidates": 200})
        result = batch_runtime.PCSBatchRuntime(runtime)
        self.addCleanup(result.close)
        return result

    def test_cap_and_memory_fallback(self):
        engine = self.runtime()
        items = [{"width": 551, "height": 554}] * 4
        self.assertEqual(engine._lanes(items, 6), 4)
        engine.runtime._torch.cuda.mem_get_info = lambda device: (2 * 1024 ** 3, 24 * 1024 ** 3)
        self.assertEqual(engine._lanes(items, 4), 1)

    def test_order_and_retry_only_failed_after_join(self):
        engine = self.runtime()
        engine.executor = ThreadPoolExecutor(max_workers=4)
        engine._ensure_pool = Mock()
        seen = []
        def run(value, lane=None):
            seen.append((value["index"], lane))
            if value["index"] == 1 and lane is not None:
                raise FloatingPointError("non-finite")
            return value["index"]
        engine._predict_one = run
        items = [{"index": n, "width": 3, "height": 2} for n in range(4)]
        result = engine.predict({"items": items})
        self.assertEqual([entry["prediction"] for entry in result["items"]], [0, 1, 2, 3])
        self.assertEqual(seen[-1], (1, None))
        self.assertEqual(len(seen), 5)
        self.assertTrue(engine.disabled)
        self.assertTrue(result["fallback"])
        self.assertEqual(engine._lanes(items, 4), 1)

    def test_failure_preserves_other_results(self):
        engine = self.runtime()
        engine.disabled = True
        engine._predict_one = Mock(side_effect=[ValueError("bad"), prediction()])
        result = engine.predict({"items": [{"width": 3, "height": 2}] * 2})
        self.assertIsNotNone(result["items"][0]["error"])
        self.assertIsNone(result["items"][1]["error"])

    def test_raw_guard_catches_nan_before_filter(self):
        raw = {key: np.ones((2, 2)) for key in ("pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec")}
        raw["pred_masks"][0, 0] = np.nan
        model = SimpleNamespace(forward_grounding=lambda: raw)
        checked = batch_runtime._CheckedPCSModel(model, np)
        with self.assertRaises(FloatingPointError):
            checked.forward_grounding()

    def test_autocast_explicitly_disables_cache(self):
        source = Path(batch_runtime.__file__).read_text()
        calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "autocast"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(keyword.arg == "cache_enabled" and keyword.value.value is False
                            for keyword in calls[0].keywords))

    def test_oversized_response_does_not_break_protocol(self):
        engine = self.runtime()
        engine.disabled = True
        engine._predict_one = lambda value, lane=None: b"x" * 8000
        with patch.object(batch_runtime, "max_frame_bytes", return_value=8192):
            result = engine.predict({"items": [{"width": 3, "height": 2}] * 2})
            self.assertEqual(result["reason"], "frame_limit")
            result = engine.predict({"items": [{"width": 3, "height": 2}]})
            self.assertIsNotNone(result["items"][0]["error"])

    def test_reject_more_than_four(self):
        engine = self.runtime()
        with self.assertRaises(ValueError):
            engine.predict({"items": [{}] * 6})


class SupervisorBatchTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = ModelRuntimeSupervisor(start_monitor=False)
        self.addCleanup(self.supervisor.shutdown)
        self.supervisor.lease = lambda **kwargs: nullcontext()

    def test_shapes_order_and_blank_text_with_box(self):
        first = item()
        first.update(text="", positive_boxes_cxcywh=np.array([[0.5, 0.5, 0.2, 0.2]]))
        self.supervisor._rpc = Mock(return_value={"items": [
            {"prediction": prediction(), "error": None},
            {"prediction": None, "error": "failed"}], "concurrency": 2})
        result = self.supervisor.predict_pcs_batch([first, item()])
        self.assertEqual(result["items"][0]["prediction"]["masks"].shape, (1, 2, 3))
        self.assertEqual(result["items"][1]["error"], "failed")
        self.assertEqual(self.supervisor._rpc.call_args.args[1]["items"][0]["text"], "")

    def test_native_failure_retries_once_serial(self):
        response = {"items": [{"prediction": prediction(), "error": None}], "concurrency": 1}
        self.supervisor._rpc = Mock(side_effect=[EOFError(), response])
        self.supervisor._wait_until_started = Mock()
        result = self.supervisor.predict_pcs_batch([item()])
        self.assertTrue(result["fallback"])
        self.assertTrue(self.supervisor._pcs_parallel_disabled)
        self.assertEqual(self.supervisor._rpc.call_args.args[1]["concurrency"], 1)
        self.assertEqual(self.supervisor._rpc.call_count, 2)

    def test_serial_retry_failure_is_not_an_infinite_restart_loop(self):
        self.supervisor._rpc = Mock(side_effect=EOFError())
        self.supervisor._wait_until_started = Mock()
        with self.assertRaises(EOFError):
            self.supervisor.predict_pcs_batch([item()])
        self.assertEqual(self.supervisor._rpc.call_count, 2)

    def test_packet_limit_splits_without_reordering(self):
        def rpc(op, args):
            return {"items": [{"prediction": prediction(), "error": None} for _ in args["items"]], "concurrency": 1}
        self.supervisor._rpc = Mock(side_effect=rpc)
        with patch("sam3_demo.model_supervisor._MAX_FRAME_BYTES", 16 * 1024 * 1024 + 3000):
            result = self.supervisor.predict_pcs_batch([item()] * 4)
        self.assertEqual(len(result["items"]), 4)
        self.assertEqual(self.supervisor._rpc.call_count, 4)

    def test_no_prompt_or_nonfinite_inputs_never_call_worker(self):
        self.supervisor._rpc = Mock()
        invalid = item()
        invalid["text"] = ""
        with self.assertRaises(ValueError):
            self.supervisor.predict_pcs_batch([invalid])
        invalid = item()
        invalid["threshold"] = float("nan")
        with self.assertRaises(ValueError):
            self.supervisor.predict_pcs_batch([invalid])
        self.supervisor._rpc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
