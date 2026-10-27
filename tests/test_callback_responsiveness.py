"""Deterministic concurrency checks: no GPU, sleep-based speed assertions or UI changes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import TestCase, mock
import sys

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3_demo.session_guard import (
    guard_callback, inference_wait, record_handled_callback_failure,
)
from sam3_demo.session_runtime import RequestIdentity, SessionRegistry


class CallbackResponsivenessTest(TestCase):
    def setUp(self):
        self.cleaned = []
        self.registry = SessionRegistry(
            secret=b"test", start_sweeper=False,
            cleanup_callback=lambda record: self.cleaned.append(record.session_id),
        )
        self.identity = RequestIdentity("owner", "page", "127.0.0.1")
        self.state = self.registry.bind(self.identity)
        self.request = gr.Request(username="owner", session_hash="page",
                                  client=SimpleNamespace(host="127.0.0.1"), headers={})
        self.started = Event()
        self.release = Event()
        self.pool = ThreadPoolExecutor(max_workers=3)

    def tearDown(self):
        self.release.set()
        self.pool.shutdown(wait=True)
        self.registry.shutdown()

    def guarded(self, callback, **kwargs):
        return guard_callback(callback, registry=self.registry, **kwargs)

    def start_prediction(self, *, fail=False):
        def predict(state):
            try:
                with inference_wait():
                    self.started.set()
                    if not self.release.wait(5):
                        raise TimeoutError("test did not release prediction")
                    if fail:
                        raise RuntimeError("worker failed")
            except Exception:
                state["value"] = "incorrect error writeback"
                raise
            state["value"] = "prediction"
            return state

        result = self.pool.submit(self.guarded(predict), self.state, self.request)
        self.assertTrue(self.started.wait(3))
        return result

    def test_read_only_overlay_finishes_before_prediction_and_does_not_cancel_it(self):
        prediction = self.start_prediction()
        read = self.guarded(lambda state: state["session_id"], read_only=True)
        self.assertEqual(self.pool.submit(read, self.state, self.request).result(2),
                         self.state["session_id"])
        self.assertFalse(prediction.done())
        self.release.set()
        self.assertEqual(prediction.result(2)["value"], "prediction")

    def test_edit_finishes_before_prediction_and_old_result_is_not_written(self):
        prediction = self.start_prediction()

        def edit(state):
            state["value"] = "cleared"
            return state

        self.pool.submit(self.guarded(edit), self.state, self.request).result(2)
        self.assertFalse(prediction.done())
        self.release.set()
        self.assertEqual(prediction.result(2), gr.skip())
        self.assertEqual(self.state["value"], "cleared")

    def test_stale_worker_error_cannot_overwrite_new_state_via_business_catch(self):
        prediction = self.start_prediction(fail=True)
        self.pool.submit(self.guarded(lambda state: state), self.state, self.request).result(2)
        self.release.set()
        self.assertEqual(prediction.result(2), gr.skip())
        self.assertNotIn("value", self.state)

    def test_other_page_edit_does_not_invalidate_prediction(self):
        prediction = self.start_prediction()
        identity = RequestIdentity("other", "other-page", "127.0.0.1")
        state = self.registry.bind(identity)
        request = gr.Request(username="other", session_hash="other-page",
                             client=SimpleNamespace(host="127.0.0.1"), headers={})
        self.pool.submit(self.guarded(lambda state: state), state, request).result(2)
        self.release.set()
        self.assertEqual(prediction.result(2)["value"], "prediction")

    def test_close_during_prediction_defers_cleanup_and_drops_writeback(self):
        prediction = self.start_prediction()
        self.registry.close(self.state, self.identity)
        self.assertEqual(self.cleaned, [])
        self.release.set()
        self.assertEqual(prediction.result(2), gr.skip())
        self.assertEqual(self.cleaned, [self.state["session_id"]])
        self.assertNotIn("value", self.state)

    def test_nested_recovery_lease_is_fully_released_while_waiting(self):
        captured = {}

        def predict(state):
            captured["state"] = state
            with inference_wait():
                self.started.set()
                if not self.release.wait(5):
                    raise TimeoutError("test did not release prediction")
            return state

        callback = self.guarded(predict, recovery_factory=lambda state: {"state": state})
        prediction = self.pool.submit(callback, None, self.request)
        self.assertTrue(self.started.wait(3))
        read = self.guarded(lambda state: state["session_id"], read_only=True)
        self.pool.submit(read, captured["state"], self.request).result(2)
        self.release.set()
        self.assertEqual(prediction.result(2)["session_id"], self.state["session_id"])

    def test_model_startup_wait_does_not_hold_session_lock(self):
        from sam3_demo import app

        @contextmanager
        def loading(**kwargs):
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("test did not release startup")
            yield

        @app._model_lease_wrapper("test")
        def predict(state):
            return state

        with mock.patch.object(app.SUPERVISOR, "lease", loading):
            prediction = self.pool.submit(self.guarded(predict), self.state, self.request)
            self.assertTrue(self.started.wait(3))
            read = self.guarded(lambda state: True, read_only=True)
            self.assertTrue(self.pool.submit(read, self.state, self.request).result(2))
            self.release.set()
            prediction.result(2)

    def test_failed_wait_reacquires_lock_and_later_callback_still_works(self):
        prediction = self.start_prediction(fail=True)
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            prediction.result(2)
        self.assertEqual(self.guarded(lambda state: "ok")(self.state, self.request), "ok")

    def test_non_callback_prediction_can_use_wait_context(self):
        with inference_wait():
            self.assertTrue(True)

    def test_handled_failure_is_not_logged_as_success(self):
        observability = mock.Mock()
        observability.monotonic.return_value = 1.0
        observability.record_callback_failure.return_value = "error-123"

        def callback(state):
            try:
                raise RuntimeError("worker failed")
            except RuntimeError as exc:
                return "PCS failed: " + record_handled_callback_failure(exc)

        result = self.guarded(callback, observability=observability)(self.state, self.request)
        self.assertEqual(result, "PCS failed: error-123")
        observability.record_callback_failure.assert_called_once()
        observability.record_callback_success.assert_not_called()

    def test_real_pcs_callback_reports_worker_failure_and_preserves_output_contract(self):
        from PIL import Image
        from sam3_demo import app

        pcs, pvs = app._new_pcs_state(), app._new_pvs_state()
        for state in (pcs, pvs):
            state.update(session_id=self.state["session_id"],
                         owner_token=self.state["owner_token"])
        pcs["instances"] = {7: {"id": 7}}
        observability = mock.Mock()
        observability.monotonic.return_value = 1.0
        observability.record_callback_failure.return_value = "pcs-error-123"
        with (
            mock.patch.object(app.SUPERVISOR, "lease", return_value=nullcontext()),
            mock.patch.object(app, "_workspace", return_value={"image": Image.new("RGB", (8, 8))}),
            mock.patch.object(app, "_fresh_state", return_value={}),
            mock.patch.object(app, "_predict_pcs", side_effect=RuntimeError("worker died")),
            mock.patch.object(app, "_view", side_effect=lambda *args: (args[4],) * 8),
        ):
            result = self.guarded(app._run_pcs, observability=observability)(
                self.state, pcs, pvs, app.MODE_PCS, "object", 0.5, self.request)
        self.assertEqual(len(result), 9)
        self.assertEqual(pcs["instances"], {7: {"id": 7}})
        self.assertIn("pcs-error-123", result[1])
        observability.record_callback_failure.assert_called_once()
        observability.record_callback_success.assert_not_called()


class InferenceQueueTest(TestCase):
    def test_gpu_callbacks_do_not_share_the_light_interaction_queue(self):
        from sam3_demo import app

        demo = app.create_demo()
        functions = {fn.name: fn for fn in demo.fns.values()}
        heavy = ("_run_pcs", "_create_pvs_from_pending_boxes", "_pvs_point_prompt",
                 "_layout_point_refine", "_create_pvs_from_layout_selection",
                 "_finish_native_polygon")
        for name in heavy:
            with self.subTest(callback=name):
                self.assertEqual(functions[name].concurrency_id, "sam3-inference")
                self.assertEqual(functions[name].concurrency_limit, 1)
        for name in ("_workspace_gesture_payload", "_clear_prompt_selection",
                     "_switch_mode_with_layout_editor", "_source_upload_workspace"):
            with self.subTest(callback=name):
                self.assertNotEqual(functions[name].concurrency_id, "sam3-inference")
                self.assertGreater(functions[name].concurrency_limit, 1)

    def test_busy_worker_eviction_does_not_wait_for_transport(self):
        from sam3_demo.model_supervisor import ModelRuntimeSupervisor, RUNNING

        supervisor = ModelRuntimeSupervisor(start_monitor=False)
        supervisor._worker = mock.Mock()
        supervisor._state = RUNNING
        supervisor._active_requests = 1
        supervisor._encoded_generation[("old", "hash")] = 1
        with mock.patch.object(supervisor, "_rpc") as rpc:
            supervisor.evict(["old"])
        rpc.assert_not_called()
        self.assertEqual(supervisor._encoded_generation, {})
        supervisor._worker = None
        supervisor.shutdown()
