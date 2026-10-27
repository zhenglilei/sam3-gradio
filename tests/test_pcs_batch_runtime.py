from concurrent.futures import ThreadPoolExecutor
import unittest

from sam3_demo.pcs_batch_runtime import PCSBatchRuntime


class _FakeCuda:
    def __init__(self):
        self.synchronize_calls = 0
        self.empty_cache_calls = 0

    def mem_get_info(self, _device):
        return 16 * 1024**3, 24 * 1024**3

    def memory_reserved(self, _device):
        return 0

    def memory_allocated(self, _device):
        return 0

    def synchronize(self, _device):
        self.synchronize_calls += 1

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class _CapacityError(RuntimeError):
    pass


class _Runtime:
    device = "cuda:0"

    def __init__(self):
        self._torch = _FakeTorch()
        self._capabilities = {"pcs_max_candidates": 200}

    @staticmethod
    def _is_oom(exc):
        return isinstance(exc, _CapacityError)


class PCSBatchRuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = _Runtime()
        self.engine = PCSBatchRuntime(self.runtime)
        self.engine.executor = ThreadPoolExecutor(max_workers=4)
        self.engine._ensure_pool = lambda: None
        self.addCleanup(self.engine.close)

    @staticmethod
    def items(count=2):
        return [{"id": index, "width": 16, "height": 16} for index in range(count)]

    def test_item_error_does_not_disable_next_batch(self):
        def predict(item, lane=None):
            del lane
            if item.get("fail"):
                raise ValueError("bad prompt")
            return item["id"]

        self.engine._predict_one = predict
        first_items = self.items(2)
        first_items[0]["fail"] = True
        first = self.engine.predict({"items": first_items, "concurrency": 4})

        self.assertEqual(first["concurrency"], 2)
        self.assertFalse(first["fallback"])
        self.assertEqual(first["reason"], "")
        self.assertIsNone(first["items"][0]["prediction"])
        self.assertIsNotNone(first["items"][0]["error"])
        self.assertEqual(set(first["items"][0]), {"prediction", "error"})

        healthy = self.engine.predict({"items": self.items(2), "concurrency": 4})
        self.assertEqual(healthy["concurrency"], 2)
        self.assertFalse(healthy["fallback"])
        self.assertEqual([item["prediction"] for item in healthy["items"]], [0, 1])

    def test_capacity_error_retries_this_call_and_recovers_next_batch(self):
        state = {"raised": False}

        def predict(item, lane=None):
            if item["id"] == 0 and lane is not None and not state["raised"]:
                state["raised"] = True
                raise _CapacityError("CUDA out of memory")
            return item["id"]

        self.engine._predict_one = predict
        first = self.engine.predict({"items": self.items(2), "concurrency": 4})

        self.assertEqual(first["concurrency"], 2)
        self.assertTrue(first["fallback"])
        self.assertEqual(first["reason"], "memory_limit")
        self.assertEqual([item["prediction"] for item in first["items"]], [0, 1])
        self.assertGreaterEqual(self.runtime._torch.cuda.empty_cache_calls, 1)

        healthy = self.engine.predict({"items": self.items(2), "concurrency": 4})
        self.assertEqual(healthy["concurrency"], 2)
        self.assertFalse(healthy["fallback"])
        self.assertEqual([item["prediction"] for item in healthy["items"]], [0, 1])


    def test_cuda_context_failure_is_not_converted_to_item_error(self):
        self.engine._predict_one = lambda item, lane=None: (_ for _ in ()).throw(
            RuntimeError("CUDA error: illegal memory access")
        )
        with self.assertRaises(RuntimeError):
            self.engine.predict({"items": self.items(2), "concurrency": 4})

if __name__ == "__main__":
    unittest.main()
