"""Regression contracts for EL batch responsiveness."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest import TestCase, mock

import gradio as gr
from PIL import Image

from sam3_demo import app, batch_workspace as batch, model_runtime
from sam3_demo.session_guard import guard_callback
from sam3_demo.session_runtime import RequestIdentity, SessionRegistry


class BatchResponsivenessTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = app.create_demo()

    def _function(self, name):
        functions = [fn for fn in self.demo.fns.values() if fn.name == name]
        self.assertEqual(len(functions), 1, name)
        return functions[0]

    def test_batch_events_use_separate_queue_contracts(self):
        step = self._function("batch_step")
        self.assertEqual(step.concurrency_id, "sam3-inference")
        self.assertEqual(step.concurrency_limit, 1)

        cancel = self._function("batch_cancel")
        self.assertFalse(cancel.queue)

        light_functions = [
            fn for fn in self.demo.fns.values()
            if fn.concurrency_id == "image-prepost-state"
        ]
        self.assertTrue(light_functions)
        for fn in light_functions:
            with self.subTest(name=fn.name):
                self.assertIsInstance(fn.concurrency_limit, int)
                self.assertGreaterEqual(fn.concurrency_limit, 8)

    def test_model_wait_releases_session_for_cancel(self):
        registry = SessionRegistry(secret=b"test", start_sweeper=False)
        identity = RequestIdentity("owner", "batch-page", "127.0.0.1")
        state = registry.bind(identity)
        state["running"] = True
        request = gr.Request(
            username="owner",
            session_hash="batch-page",
            client=SimpleNamespace(host="127.0.0.1"),
            headers={},
        )
        started = Event()
        release = Event()
        cancel_done = Event()
        pool = ThreadPoolExecutor(max_workers=2)

        def predict(_items):
            started.set()
            if not release.wait(5):
                raise AssertionError("test model was not released")
            return {"items": [{"prediction": object(), "error": None}], "concurrency": 1, "fallback": False, "reason": ""}

        def run_batch(current_state):
            model_runtime._predict_pcs_batch([{"image": Image.new("RGB", (2, 2))}])
            return current_state

        def cancel(current_state):
            current_state["running"] = False
            cancel_done.set()
            return current_state

        try:
            with mock.patch.object(
                model_runtime.SUPERVISOR,
                "predict_pcs_batch",
                side_effect=predict,
            ):
                batch_future = pool.submit(
                    guard_callback(run_batch, registry=registry),
                    state,
                    request,
                )
                self.assertTrue(started.wait(3))
                cancel_future = pool.submit(
                    guard_callback(cancel, registry=registry),
                    state,
                    request,
                )
                cancel_ready = cancel_done.wait(3)
                self.assertFalse(batch_future.done())
                release.set()
                cancel_future.result(timeout=3)
                batch_result = batch_future.result(timeout=3)
                self.assertTrue(
                    cancel_ready,
                    "cancel remained blocked while batch prediction was waiting",
                )
                self.assertEqual(batch_result, gr.skip())
            self.assertFalse(state["running"])
        finally:
            release.set()
            pool.shutdown(wait=True)
            registry.shutdown()

    def test_cancelled_step_keeps_pending_and_drops_stale_writeback(self):
        image = Image.new("RGB", (8, 8), (20, 30, 40))
        item = {
            "id": "item-1",
            "name": "tile.png",
            "original": image,
            "snapshot": {
                "image": image.copy(),
                "pcs": {
                    "positive_boxes": [[0, 0, 4, 4]],
                    "negative_boxes": [],
                },
                "pvs": {},
                "prompt": {},
                "layout": {},
                "mode": "PCS Auto",
                "tool": "bbox",
                "text": "",
                "threshold": 0.4,
            },
            "status": "pending",
            "error": "",
        }
        state = {
            "session_id": "a" * 32,
            "items": [item],
            "active_id": "item-1",
            "pending": ["item-1"],
            "running": True,
            "batch_text": "",
            "batch_threshold": 0.4,
        }
        registry = SessionRegistry(secret=b"test", start_sweeper=False)
        identity = RequestIdentity("owner", "batch-page", "127.0.0.1")
        session_state = registry.bind(identity)
        request = gr.Request(
            username="owner",
            session_hash="batch-page",
            client=SimpleNamespace(host="127.0.0.1"),
            headers={},
        )
        state["session_id"] = session_state["session_id"]
        fake_app = SimpleNamespace(
            _xyxy_to_cxcywh_norm=lambda box, _width, _height: box,
        )
        restored_values = (
            {"source_image_id": "source-1"},
            {"image_id": "image-1", "width": 8, "height": 8},
            {"positive_boxes": [], "instances": {}},
            {},
            {},
            {},
            "PCS Auto",
            "bbox",
            "",
            0.4,
        )
        started = Event()
        release = Event()
        cancel_done = Event()
        pool = ThreadPoolExecutor(max_workers=2)

        def predict(_items):
            started.set()
            if not release.wait(5):
                raise AssertionError("test model was not released")
            return {
                "items": [{"prediction": object(), "error": None}],
                "concurrency": 1,
                "fallback": False,
                "reason": "",
            }

        try:
            with (
                mock.patch.object(batch, "restore_item", return_value=restored_values) as restore_item,
                mock.patch.object(batch, "save_snapshot", return_value=True) as save_snapshot,
                mock.patch("sam3_demo.pcs_pvs_callbacks.apply_pcs_prediction"),
                mock.patch.object(
                    model_runtime.SUPERVISOR,
                    "predict_pcs_batch",
                    side_effect=predict,
                ),
            ):
                def step(session_state):
                    return batch.process_pcs_chunk(fake_app, state, session_state)

                def cancel(session_state):
                    state["running"] = False
                    state["pending"] = []
                    cancel_done.set()
                    return session_state

                step_callback = guard_callback(step, registry=registry)
                cancel_callback = guard_callback(cancel, registry=registry)
                batch_future = pool.submit(step_callback, session_state, request)
                self.assertTrue(started.wait(3))
                pending_during_wait = list(state["pending"])
                cancel_future = pool.submit(cancel_callback, session_state, request)
                cancel_ready = cancel_done.wait(3)
                self.assertFalse(batch_future.done())
                release.set()
                cancel_future.result(timeout=3)
                batch_result = batch_future.result(timeout=3)
                self.assertTrue(
                    cancel_ready,
                    "cancel remained blocked while batch prediction was waiting",
                )
                self.assertEqual(batch_result, gr.skip())

            restore_item.assert_not_called()
            save_snapshot.assert_not_called()
            self.assertFalse(state["running"])
            self.assertEqual(pending_during_wait, ["item-1"])
            self.assertEqual(state["pending"], [])
            self.assertEqual(item["status"], "pending")
            self.assertEqual(item["error"], "")
        finally:
            release.set()
            pool.shutdown(wait=True)
            registry.shutdown()


if __name__ == "__main__":
    import unittest

    unittest.main()
