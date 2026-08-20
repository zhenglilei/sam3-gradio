"""Pure protocol/serialization tests for the isolated SAM3 worker.

These tests deliberately do not load torch or a checkpoint.  The subprocess
smoke only sends ``ping``/``shutdown`` and therefore verifies the stdout wire
channel without allocating a CUDA context.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest

import numpy as np

from sam3_demo import model_worker


class _FakeProcessor:
    def __init__(self) -> None:
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1
        return {
            "original_height": image.height,
            "original_width": image.width,
            "backbone_out": {"vision_features": object()},
        }

    def set_text_prompt(self, text, state):
        # This mirrors the in-place top-level update made by Sam3Processor.
        state["backbone_out"].update({"language_features": text})
        state["masks"] = np.ones((1, 1, 2, 3), dtype=np.bool_)
        state["masks_logits"] = np.ones((1, 1, 2, 3), dtype=np.float32)
        state["boxes"] = np.asarray([[0, 0, 3, 2]], dtype=np.float32)
        state["scores"] = np.asarray([0.8], dtype=np.float32)
        return state

    def add_geometric_prompt(self, box, label, state):
        return state

    def set_confidence_threshold(self, threshold, state):
        return state


class _FakeModel:
    def __init__(self) -> None:
        self.states = []

    def predict_inst(self, state, **kwargs):
        self.states.append(state)
        masks = np.asarray([[[1, 0, 0], [1, 1, 0]]], dtype=np.float32)
        scores = np.asarray([0.9], dtype=np.float32)
        logits = np.asarray([[[0.2, -0.2], [0.1, 0.4]]], dtype=np.float32)
        return masks, scores, logits


class _FakeBFloatTensor:
    def __init__(self):
        self.dtype = "torch.bfloat16"

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.dtype = "torch.float32"
        return self

    def numpy(self):
        if self.dtype == "torch.bfloat16":
            raise TypeError("Got unsupported ScalarType BFloat16")
        return np.asarray([1.25], dtype=np.float32)


class ProtocolTests(unittest.TestCase):
    def test_bfloat16_tensor_is_promoted_before_numpy_conversion(self):
        converted = model_worker._as_numpy(_FakeBFloatTensor(), dtype=np.float16)
        self.assertEqual(converted.dtype, np.float16)
        np.testing.assert_allclose(converted, np.asarray([1.25], dtype=np.float16))

    def test_frame_round_trip_and_partial_reads(self):
        request = model_worker.make_request(7, "ping", generation=3)
        framed = model_worker.dumps_message(request)

        class _Chunked(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 2))

        decoded = model_worker.read_frame(_Chunked(framed))
        self.assertEqual(decoded, request)

    def test_frame_limit_and_truncated_payload_are_rejected(self):
        with self.assertRaises(model_worker.WorkerProtocolError):
            model_worker.dumps_message({"x": "a" * 100}, max_bytes=16)
        with self.assertRaises(EOFError):
            model_worker.read_frame(io.BytesIO(b"\x00\x00\x00\x05abc"))

    def test_packbits_preserves_non_byte_aligned_masks(self):
        masks = np.asarray(
            [
                [[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]],
                [[0, 1, 0, 0, 1], [1, 0, 1, 1, 0]],
            ],
            dtype=np.uint8,
        )
        packed, shape = model_worker.pack_masks(masks)
        self.assertEqual(shape, [2, 2, 5])
        self.assertEqual(len(packed), 4)
        np.testing.assert_array_equal(model_worker.unpack_masks(packed, shape), masks.astype(bool))

    def test_packbits_accepts_zero_candidate_masks(self):
        masks = np.zeros((0, 7, 9), dtype=bool)
        packed, shape = model_worker.pack_masks(masks)
        self.assertEqual(packed, b"")
        self.assertEqual(shape, [0, 7, 9])
        decoded = model_worker.unpack_masks(packed, shape)
        self.assertEqual(decoded.shape, (0, 7, 9))

    def test_unpack_rejects_wrong_packed_length(self):
        with self.assertRaises(ValueError):
            model_worker.unpack_masks(b"\x00", [1, 2, 16])

    def test_response_validation_and_error_code(self):
        self.assertEqual(model_worker.validate_response(model_worker.success_response(4, {"ok": 1}), 4), {"ok": 1})
        error = model_worker.error_response(4, "MISSING_EMBEDDING", KeyError("missing"))
        with self.assertRaises(model_worker.WorkerRPCError) as ctx:
            model_worker.validate_response(error, 4)
        self.assertEqual(ctx.exception.code, "MISSING_EMBEDDING")
        with self.assertRaises(model_worker.WorkerProtocolError):
            model_worker.validate_response(model_worker.success_response(5, None), 4)

    def test_stdout_is_reserved_for_binary_protocol(self):
        root = os.path.dirname(os.path.dirname(__file__))
        proc = subprocess.Popen(
            [sys.executable, "-m", "sam3_demo.model_worker"],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(model_worker.dumps_message(model_worker.make_request(1, "ping")))
        proc.stdin.flush()
        response = model_worker.read_frame(proc.stdout)
        self.assertIsNotNone(response)
        self.assertEqual(model_worker.validate_response(response, 1)["pid"], proc.pid)
        proc.stdin.write(model_worker.dumps_message(model_worker.make_request(2, "shutdown")))
        proc.stdin.flush()
        response = model_worker.read_frame(proc.stdout)
        self.assertEqual(model_worker.validate_response(response, 2), {"shutdown": True})
        self.assertEqual(proc.wait(timeout=5), 0)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()

    def test_fake_runtime_lru_and_backbone_shallow_copy(self):
        runtime = model_worker.Sam3WorkerRuntime(embedding_limit=2)
        runtime._np = np
        runtime._capabilities = {"mask_input_size": [2, 2]}
        runtime.model = _FakeModel()
        runtime.processor = _FakeProcessor()

        def encode(image_id):
            raw = bytes([0, 0, 0] * 6)
            return runtime.encode({
                "image_id": image_id,
                "image_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                "width": 3,
                "height": 2,
                "rgb_bytes": raw,
            })

        encode("a")
        encode("b")
        encode("c")
        self.assertNotIn("a", runtime._embeddings)
        self.assertEqual(len(runtime._embeddings), 2)

        result = runtime.predict_pcs(
            {
                "image_id": "b",
                "text": "visual",
                "positive_boxes_cxcywh": np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
                "negative_boxes_cxcywh": np.empty((0, 4), dtype=np.float32),
                "threshold": 0.5,
            }
        )
        self.assertEqual(result["masks_shape"], [1, 2, 3])
        self.assertEqual(result["probs_f16"].dtype, np.float16)
        self.assertNotIn("language_features", runtime._embeddings["b"].state["backbone_out"])

    def test_missing_embedding_is_explicit(self):
        runtime = model_worker.Sam3WorkerRuntime()
        runtime.model = _FakeModel()
        runtime.processor = _FakeProcessor()
        runtime._capabilities = {"mask_input_size": [2, 2]}
        runtime._np = np
        with self.assertRaises(model_worker.WorkerRPCError) as ctx:
            runtime.predict_inst({"image_id": "missing"})
        self.assertEqual(ctx.exception.code, "MISSING_EMBEDDING")

    def test_request_rejects_unknown_op_and_protocol_version(self):
        unknown = model_worker.make_request(1, "ping")
        unknown["op"] = "shell"
        with self.assertRaises(model_worker.WorkerProtocolError):
            model_worker._validate_request(unknown)

        wrong_version = model_worker.make_request(2, "ping")
        wrong_version["protocol_version"] = model_worker.PROTOCOL_VERSION + 1
        with self.assertRaises(model_worker.WorkerProtocolError):
            model_worker._validate_request(wrong_version)



if __name__ == "__main__":
    unittest.main()
