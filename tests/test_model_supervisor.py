from __future__ import annotations

import concurrent.futures
from contextlib import nullcontext
from unittest import mock
import threading
import time
import unittest

import numpy as np

from sam3_demo.model_supervisor import (
    ERROR,
    READY,
    UNLOADED,
    ModelRuntimeSupervisor,
    ModelWorkerError,
)
from sam3_demo.model_worker import pack_masks


class _FakeProcess:
    def __init__(self, pid=1234):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode





class _FakeImage:
    width = 4
    height = 4

    def convert(self, _mode):
        return self

    def tobytes(self):
        return bytes(self.width * self.height * 3)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class ModelSupervisorTests(unittest.TestCase):
    def make_supervisor(self, **kwargs):
        return ModelRuntimeSupervisor(start_monitor=False, **kwargs)

    def test_twenty_start_requests_share_one_loader(self):
        supervisor = self.make_supervisor()
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def fake_loader(future):
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            process = _FakeProcess()
            with supervisor._condition:
                supervisor._worker = process
                supervisor._worker_pid = process.pid
                supervisor._generation = 1
                supervisor._mask_input_size = (288, 288)
                supervisor._state = READY
                supervisor._load_future = None
                supervisor._last_inference_completed_at = supervisor._clock()
                supervisor._condition.notify_all()
            future.set_result(supervisor.snapshot())

        supervisor._load_worker = fake_loader
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(supervisor.request_start) for _ in range(20)]
            self.assertTrue(entered.wait(timeout=5))
            release.set()
            [future.result(timeout=5) for future in futures]
        self.assertEqual(len(calls), 1)
        self.assertEqual(supervisor.snapshot()["state"], READY)

    def test_reentrant_lease_counts_only_outer_scope(self):
        supervisor = self.make_supervisor()
        process = _FakeProcess()
        with supervisor._condition:
            supervisor._worker = process
            supervisor._worker_pid = process.pid
            supervisor._state = READY
            supervisor._generation = 1
            supervisor._mask_input_size = (288, 288)
        with supervisor.lease(reason="outer"):
            self.assertEqual(supervisor.snapshot()["active_requests"], 1)
            with supervisor.lease(reason="inner"):
                self.assertEqual(supervisor.snapshot()["active_requests"], 1)
        snapshot = supervisor.snapshot()
        self.assertEqual(snapshot["active_requests"], 0)
        self.assertEqual(snapshot["pending_requests"], 0)
        self.assertEqual(snapshot["state"], READY)

    def test_missing_embedding_reencodes_once_without_killing_worker(self):
        supervisor = self.make_supervisor()
        process = _FakeProcess()
        with supervisor._condition:
            supervisor._worker = process
            supervisor._worker_pid = process.pid
            supervisor._state = READY
            supervisor._generation = 3
        handle = {"image_id": "i", "target_image_sha256": "h"}
        ensure_calls = []
        rpc_calls = []

        def ensure(_handle, _supplier):
            ensure_calls.append(1)

        def rpc(_op, _args):
            rpc_calls.append(1)
            if len(rpc_calls) == 1:
                raise ModelWorkerError("gone", code="MISSING_EMBEDDING")
            return {"ok": True}

        supervisor.ensure_embedding = ensure
        supervisor._rpc = rpc
        result = supervisor._predict_with_embedding("predict_inst", handle, lambda: None, {})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(ensure_calls), 2)
        self.assertEqual(len(rpc_calls), 2)
        self.assertIs(supervisor._worker, process)
        self.assertNotEqual(supervisor.snapshot()["state"], ERROR)

    def test_structured_missing_embedding_does_not_detach_worker(self):
        supervisor = self.make_supervisor()
        process = _FakeProcess()
        with supervisor._condition:
            supervisor._worker = process
            supervisor._worker_pid = process.pid
            supervisor._state = READY
            supervisor._generation = 2

        def missing(*_args, **_kwargs):
            raise ModelWorkerError("missing", code="MISSING_EMBEDDING")

        supervisor._rpc_process = missing
        with self.assertRaises(ModelWorkerError) as ctx:
            supervisor._rpc("predict_inst", {})
        self.assertEqual(ctx.exception.code, "MISSING_EMBEDDING")
        self.assertIs(supervisor._worker, process)
        self.assertEqual(supervisor.snapshot()["state"], READY)

    def test_non_byte_aligned_multi_mask_unpack(self):
        masks = np.asarray(
            [
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            ],
            dtype=np.uint8,
        )
        packed, shape = pack_masks(masks)
        decoded = ModelRuntimeSupervisor._unpack_masks(
            {"masks_packed": packed, "masks_shape": shape}
        )
        np.testing.assert_array_equal(decoded, masks.astype(bool))

    def test_multiple_visible_gpus_require_uuid_pinning(self):
        rows = [
            {"index": 0, "uuid": "", "free": 24 * 1024**3, "total": 24 * 1024**3},
            {"index": 1, "uuid": "u", "free": 23 * 1024**3, "total": 24 * 1024**3},
        ]
        supervisor = self.make_supervisor(probe_fn=lambda: rows)
        with self.assertRaises(ModelWorkerError) as ctx:
            supervisor._candidate_gpus()
        self.assertEqual(ctx.exception.code, "CONFIG")

    def test_user_activity_starts_unloaded_model_in_background(self):
        supervisor = self.make_supervisor(clock=_Clock())
        started = []
        supervisor._load_worker = lambda future: started.append(future)
        supervisor.note_user_activity()
        supervisor.note_user_activity()
        deadline = time.monotonic() + 1
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(started), 1)
        self.assertEqual(supervisor.snapshot()["last_activity_at"], 0.0)

    def test_user_activity_on_ready_model_only_records_time(self):
        clock = _Clock()
        supervisor = self.make_supervisor(clock=clock)
        supervisor._load_worker = lambda future: self.fail("must not reload a ready model")
        with supervisor._condition:
            supervisor._worker = _FakeProcess()
            supervisor._state = READY
        clock.value = 42.0
        supervisor.note_user_activity()
        self.assertEqual(supervisor.snapshot()["last_activity_at"], 42.0)

    def test_user_activity_respects_failed_start_backoff(self):
        clock = _Clock()
        supervisor = self.make_supervisor(clock=clock)
        supervisor._load_worker = lambda future: self.fail("backoff must hold")
        with supervisor._condition:
            supervisor._state = ERROR
            supervisor._last_start_failure_at = 0.0
        clock.value = supervisor._restart_backoff / 2
        supervisor.note_user_activity()
        self.assertIsNone(supervisor._load_future)

    def test_idle_release_counts_from_latest_interaction(self):
        clock = _Clock()
        supervisor = ModelRuntimeSupervisor(clock=clock, start_monitor=False)
        supervisor._idle_check_seconds = 0.01
        supervisor._idle_seconds = 10.0
        supervisor._monitor_thread = threading.Thread(
            target=supervisor._monitor_loop, name="test-sam3-activity-idle", daemon=True,
        )
        supervisor._monitor_thread.start()
        process = _FakeProcess()
        stopped = threading.Event()

        def stop(candidate, *, graceful=True):
            process.returncode = 0
            stopped.set()

        supervisor._stop_process = stop
        with supervisor._condition:
            supervisor._worker = process
            supervisor._worker_pid = process.pid
            supervisor._state = READY
            supervisor._last_inference_completed_at = 0.0
        clock.value = 8.0
        supervisor.note_user_activity()
        clock.value = 12.0
        time.sleep(0.05)
        self.assertFalse(stopped.is_set())
        clock.value = 18.0
        self.assertTrue(stopped.wait(timeout=1))
        supervisor.shutdown(graceful_timeout=0)

    def test_idle_monitor_stops_ready_worker_at_boundary(self):
        clock = _Clock()
        supervisor = ModelRuntimeSupervisor(clock=clock, start_monitor=False)
        supervisor._idle_check_seconds = 0.01
        supervisor._idle_seconds = 10.0
        supervisor._monitor_thread = threading.Thread(
            target=supervisor._monitor_loop,
            name="test-sam3-idle-monitor",
            daemon=True,
        )
        supervisor._monitor_thread.start()
        process = _FakeProcess()
        stopped = threading.Event()

        def stop(candidate, *, graceful=True):
            self.assertIs(candidate, process)
            process.returncode = 0
            stopped.set()

        supervisor._stop_process = stop
        with supervisor._condition:
            supervisor._worker = process
            supervisor._worker_pid = process.pid
            supervisor._state = READY
            supervisor._last_inference_completed_at = 0.0
        clock.value = 9.9
        time.sleep(0.04)
        self.assertFalse(stopped.is_set())
        clock.value = 10.0
        with supervisor._condition:
            supervisor._pending_requests = 1
        time.sleep(0.04)
        self.assertFalse(stopped.is_set())
        with supervisor._condition:
            supervisor._pending_requests = 0
            supervisor._condition.notify_all()
        self.assertTrue(stopped.wait(timeout=1))
        self.assertEqual(supervisor.snapshot()["state"], UNLOADED)
        supervisor.shutdown(graceful_timeout=0)

    def test_load_oom_tries_next_uuid_candidate(self):
        rows = [
            {"index": 0, "uuid": "u0", "free": 24 * 1024**3, "total": 24 * 1024**3},
            {"index": 1, "uuid": "u1", "free": 23 * 1024**3, "total": 24 * 1024**3},
        ]
        supervisor = self.make_supervisor(probe_fn=lambda: rows)
        spawned = []

        def spawn(candidate, generation):
            process = _FakeProcess(pid=2000 + len(spawned))
            process.gpu_uuid = candidate["uuid"]
            spawned.append(process)
            return process, None

        def rpc(process, _generation, op, args=None, timeout=None):
            if op == "load" and process.gpu_uuid == "u0":
                raise ModelWorkerError("oom", code="OOM")
            if op == "load":
                return {"gpu_uuid": "u1", "mask_input_size": [288, 288]}
            if op == "warmup":
                return {"elapsed_seconds": 0.1}
            raise AssertionError(op)

        supervisor._spawn_worker = spawn
        supervisor._rpc_process = rpc
        future = concurrent.futures.Future()
        supervisor._load_worker(future)
        self.assertEqual(future.result()["state"], READY)
        self.assertEqual([process.gpu_uuid for process in spawned], ["u0", "u1"])
        self.assertEqual(supervisor.snapshot()["selected_gpu_uuid"], "u1")
        supervisor.shutdown(graceful_timeout=0)

    def test_config_failure_does_not_try_another_gpu(self):
        rows = [
            {"index": 0, "uuid": "u0", "free": 24 * 1024**3, "total": 24 * 1024**3},
            {"index": 1, "uuid": "u1", "free": 23 * 1024**3, "total": 24 * 1024**3},
        ]
        supervisor = self.make_supervisor(probe_fn=lambda: rows)
        spawned = []

        def spawn(candidate, generation):
            process = _FakeProcess(pid=3000 + len(spawned))
            spawned.append(candidate["uuid"])
            return process, None

        supervisor._spawn_worker = spawn
        supervisor._rpc_process = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelWorkerError("missing checkpoint", code="CONFIG")
        )
        future = concurrent.futures.Future()
        supervisor._load_worker(future)
        with self.assertRaises(ModelWorkerError) as ctx:
            future.result()
        self.assertEqual(ctx.exception.code, "CONFIG")
        self.assertEqual(spawned, ["u0"])
        self.assertEqual(supervisor.snapshot()["state"], ERROR)

    def test_shutdown_during_probe_prevents_late_worker_spawn(self):
        entered = threading.Event()
        release = threading.Event()
        spawned = []

        def probe():
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return [
                {"index": 0, "uuid": "u0", "free": 24 * 1024**3, "total": 24 * 1024**3}
            ]

        supervisor = self.make_supervisor(probe_fn=probe)
        supervisor._spawn_worker = lambda *_args: spawned.append(1)
        supervisor.request_start()
        with supervisor._condition:
            future = supervisor._load_future
        self.assertTrue(entered.wait(timeout=5))
        supervisor.shutdown(graceful_timeout=0)
        release.set()
        with self.assertRaises(ModelWorkerError):
            future.result(timeout=5)
        self.assertEqual(spawned, [])
        self.assertEqual(supervisor.snapshot()["state"], UNLOADED)



    def test_batch_rpc_fault_is_serial_only_for_current_call(self):
        supervisor = self.make_supervisor()
        try:
            supervisor.lease = lambda *, reason: nullcontext()
            prediction = {
                "masks_packed": b"",
                "masks_shape": (0, 4, 4),
                "probs_f16": None,
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            }
            rpc_response = {
                "items": [{"prediction": prediction, "error": None}],
                "concurrency": 1,
                "fallback": False,
                "reason": "",
            }
            supervisor._rpc = mock.Mock(
                side_effect=[RuntimeError("transport failed"), rpc_response, rpc_response]
            )
            supervisor._wait_until_started = mock.Mock()
            item = {
                "image": _FakeImage(),
                "text": "target",
                "threshold": 0.5,
                "positive_boxes_cxcywh": [],
                "negative_boxes_cxcywh": [],
            }

            first = supervisor.predict_pcs_batch([item])
            second = supervisor.predict_pcs_batch([item])

            calls = supervisor._rpc.call_args_list
            self.assertEqual([call.args[1]["concurrency"] for call in calls], [4, 1, 4])
            self.assertTrue(first["fallback"])
            self.assertFalse(second["fallback"])
            supervisor._wait_until_started.assert_called_once_with()
        finally:
            supervisor.shutdown(graceful_timeout=0)

if __name__ == "__main__":
    unittest.main()
