from __future__ import annotations

import asyncio
import copy
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import gradio as gr
from gradio.state_holder import StateHolder
from PIL import Image

from sam3_demo import app
from sam3_demo.session_guard import (
    SessionGuardError,
    gradio_state_recovery,
    guard_callback,
)
from sam3_demo.session_runtime import RequestIdentity, SessionRegistry


def _request(session_hash, owner="cookie-owner", host="10.70.0.10"):
    return gr.Request(
        username=owner,
        session_hash=session_hash,
        client=SimpleNamespace(host=host),
        headers={},
    )


async def _process(blocks, fn, inputs, state, request):
    return await blocks.process_api(
        fn,
        inputs,
        state=state,
        request=request,
        session_hash=request.session_hash,
        explicit_call=True,
    )


def _find_fn(blocks, name, input_count=None):
    matches = [
        fn for fn in blocks.fns.values()
        if getattr(fn.fn, "__name__", None) == name
        and (input_count is None or len(fn.inputs) == input_count)
    ]
    if not matches:
        raise AssertionError(f"No Gradio function named {name!r}")
    return matches[0]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _MiniBlocks:
    """Small real Blocks/SessionState harness for recovery lifecycle checks."""

    def __init__(self):
        self.clock = _Clock()
        self.registry = SessionRegistry(
            clock=self.clock,
            idle_seconds=5,
            secret=b"session-recovery-lifecycle-tests",
            deployment_id="test",
            start_sweeper=False,
        )

        def factory(server_state):
            sid = server_state["session_id"]
            token = server_state["owner_token"]
            resume_id = server_state["resume_id"]

            def owned(**fields):
                return {
                    "session_id": sid,
                    "owner_token": token,
                    "resume_id": resume_id,
                    **fields,
                }

            return {
                "session_state": dict(server_state),
                "pcs_state": owned(instances={}, marker="fresh"),
                "image_state": owned(image_id="fresh-image", width=0, height=0),
                "layout_state": owned(layout_id="fresh-layout"),
                "pvs_state": owned(instances={}),
                "stitch_state": owned(images=[], mosaic=None),
            }

        self.factory = factory

        def echo(pcs_state, session_state):
            return pcs_state, session_state

        def batch_continue(batch, session_state):
            result = dict(batch)
            result.setdefault("items", []).append("continued")
            result["active_id"] = "continued"
            result["session_id"] = session_state["session_id"]
            return result, session_state

        def batch_only(batch):
            return batch

        self.destructive_calls = []

        def destructive_clear(pcs_state, pvs_state, session_state):
            self.destructive_calls.append(session_state["session_id"])
            pcs_state["instances"] = {}
            pvs_state["instances"] = {}
            return pcs_state, pvs_state, session_state

        def fail_after_mutating_identity(pcs_state, session_state):
            pcs_state["session_id"] = "mixed-session"
            pcs_state["owner_token"] = "mixed-token"
            session_state["session_id"] = "mixed-session"
            session_state["owner_token"] = "mixed-token"
            raise RuntimeError("callback failed")

        with gr.Blocks() as self.demo:
            self.session_state = gr.State(None)
            self.pcs_state = gr.State(None)
            self.image_state = gr.State(None)
            self.layout_state = gr.State(None)
            self.pvs_state = gr.State(None)
            self.stitch_state = gr.State(None)
            self.batch_state = gr.State({"items": [], "active_id": None})
            self.extension_state = gr.State({"items": [], "active_id": None})
            self.echo_inputs = [gr.JSON(), gr.JSON()]
            self.echo_button = gr.Button()
            self.batch_inputs = [gr.JSON(), gr.JSON()]
            self.batch_button = gr.Button()
            self.batch_only_input = gr.JSON()
            self.batch_only_button = gr.Button()
            self.destructive_inputs = [gr.JSON(), gr.JSON(), gr.JSON()]
            self.destructive_button = gr.Button()
            self.failure_button = gr.Button()

            self.echo_guard = guard_callback(
                echo,
                registry=self.registry,
                recovery_factory=self.factory,
            )
            self.batch_guard = guard_callback(
                batch_continue,
                registry=self.registry,
                recovery_factory=self.factory,
            )
            self.batch_only_guard = guard_callback(
                batch_only,
                registry=self.registry,
                recovery_factory=self.factory,
            )
            self.destructive_guard = guard_callback(
                destructive_clear,
                registry=self.registry,
                recovery_factory=self.factory,
            )
            self.failure_guard = guard_callback(
                fail_after_mutating_identity,
                registry=self.registry,
                recovery_factory=self.factory,
            )
            self.echo_button.click(
                self.echo_guard,
                inputs=self.echo_inputs,
                outputs=[self.pcs_state, self.session_state],
            )
            self.batch_button.click(
                self.batch_guard,
                inputs=self.batch_inputs,
                outputs=[self.batch_state, self.session_state],
            )
            self.batch_only_button.click(
                self.batch_only_guard,
                inputs=[self.batch_only_input],
                outputs=[self.batch_state],
            )
            self.destructive_button.click(
                self.destructive_guard,
                inputs=self.destructive_inputs,
                outputs=[self.pcs_state, self.pvs_state, self.session_state],
            )
            self.failure_button.click(
                self.failure_guard,
                inputs=self.echo_inputs,
                outputs=[self.pcs_state, self.session_state],
            )

        self.demo._sam3_session_recovery = gradio_state_recovery(
            self.demo,
            (SimpleNamespace(
                session_state=self.session_state,
                pcs_state=self.pcs_state,
                image_state=self.image_state,
                layout_state=self.layout_state,
                pvs_state=self.pvs_state,
                stitch_state=self.stitch_state,
                batch_state=self.batch_state,
                extension_state=self.extension_state,
            ),),
            registry=self.registry,
            factory=self.factory,
        )
        holder = StateHolder()
        holder.set_blocks(self.demo)

        self.echo_fn = _find_fn(self.demo, "echo")
        self.batch_fn = _find_fn(self.demo, "batch_continue")
        self.batch_only_fn = _find_fn(self.demo, "batch_only")
        self.destructive_fn = _find_fn(self.demo, "destructive_clear")
        self.failure_fn = _find_fn(self.demo, "fail_after_mutating_identity")
        for fn in (
            self.echo_fn,
            self.batch_fn,
            self.batch_only_fn,
            self.destructive_fn,
            self.failure_fn,
        ):
            fn.preprocess = False

    def close(self):
        self.registry.shutdown()

    def seed(self, session_hash, *, owner="cookie-owner", host="10.70.0.10"):
        request = _request(session_hash, owner, host)
        root = self.registry.bind(RequestIdentity(owner, session_hash, host))
        state = self.demo.state_holder[session_hash]
        values = self.factory(root)
        for name, component in (
            ("session_state", self.session_state),
            ("pcs_state", self.pcs_state),
            ("image_state", self.image_state),
            ("layout_state", self.layout_state),
            ("pvs_state", self.pvs_state),
            ("stitch_state", self.stitch_state),
        ):
            state[component._id] = copy.deepcopy(values[name])
        state[self.batch_state._id] = {
            "session_id": root["session_id"],
            "items": ["legacy"],
            "active_id": "legacy",
        }
        state[self.extension_state._id] = {
            "session_id": root["session_id"],
            "items": ["extension-old"],
            "active_id": "extension-old",
        }
        return request, state, root


class SessionRecoveryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._runner = asyncio.Runner()

    def _run(self, blocks, fn, inputs, state, request):
        return self._runner.run(_process(blocks, fn, inputs, state, request))

    def tearDown(self):
        harness = getattr(self, "harness", None)
        if harness is not None:
            harness.close()
        self._runner.close()

    def _new_harness(self):
        self.harness = _MiniBlocks()
        return self.harness

    def test_expired_thin_business_state_recovers_with_or_without_resume_id(self):
        for with_resume in (False, True):
            with self.subTest(with_resume=with_resume):
                harness = self._new_harness()
                request, state, old_root = harness.seed("thin-" + uuid.uuid4().hex)
                old_pcs = state[harness.pcs_state._id]
                old_pcs["marker"] = "old"
                old_pcs["instances"] = {"legacy": {"must_not_survive": True}}
                thin = {
                    "session_id": old_root["session_id"],
                    "owner_token": old_root["owner_token"],
                }
                if with_resume:
                    thin["resume_id"] = old_root["resume_id"]
                harness.clock.advance(6)

                self._run(harness.demo, harness.echo_fn, [thin, old_root], state, request)

                new_root = state[harness.session_state._id]
                new_pcs = state[harness.pcs_state._id]
                self.assertNotEqual(new_root["session_id"], old_root["session_id"])
                self.assertEqual(new_pcs["session_id"], new_root["session_id"])
                self.assertEqual(new_pcs["marker"], "fresh")
                self.assertEqual(new_pcs["instances"], {})
                self.assertEqual(new_pcs["owner_token"], new_root["owner_token"])
                harness.registry.close_state(new_root)
                harness.close()
                self.harness = None

    def test_missing_state_paired_with_thin_state_recovers_instead_of_stale_error(self):
        harness = self._new_harness()
        request, state, old_root = harness.seed("missing-" + uuid.uuid4().hex)
        thin = {
            "session_id": old_root["session_id"],
            "owner_token": old_root["owner_token"],
        }
        state[harness.session_state._id] = None
        harness.clock.advance(6)

        self._run(harness.demo, harness.echo_fn, [thin, None], state, request)

        self.assertNotEqual(
            state[harness.session_state._id]["session_id"], old_root["session_id"]
        )
        self.assertEqual(
            state[harness.pcs_state._id]["session_id"],
            state[harness.session_state._id]["session_id"],
        )
        harness.registry.close_state(state[harness.session_state._id])

    def test_expired_legacy_batch_requires_signed_root_and_resets_default(self):
        harness = self._new_harness()
        request, state, old_root = harness.seed("batch-" + uuid.uuid4().hex)
        old_batch = copy.deepcopy(state[harness.batch_state._id])
        harness.clock.advance(6)

        self._run(harness.demo, harness.batch_fn, [old_batch, old_root], state, request)

        new_root = state[harness.session_state._id]
        batch = state[harness.batch_state._id]
        self.assertNotEqual(new_root["session_id"], old_root["session_id"])
        self.assertEqual(batch["items"], ["continued"])
        self.assertEqual(batch["active_id"], "continued")
        self.assertEqual(batch["session_id"], new_root["session_id"])
        self.assertEqual(batch["owner_token"], new_root["owner_token"])
        self.assertEqual(batch["resume_id"], new_root["resume_id"])

        self._run(
            harness.demo,
            harness.batch_fn,
            [batch, new_root],
            state,
            request,
        )
        self.assertEqual(state[harness.batch_state._id]["items"], ["continued", "continued"])

        no_root_hash = "batch-no-root-" + uuid.uuid4().hex
        no_root_request, _, no_root = harness.seed(no_root_hash)
        harness.clock.advance(6)
        with self.assertRaises(SessionGuardError):
            harness.batch_only_guard(no_root, no_root_request)

        forged_root_hash = "batch-forged-root-" + uuid.uuid4().hex
        forged_request, _, signed_root = harness.seed(forged_root_hash)
        forged_root = dict(signed_root, owner_token="forged-token")
        harness.clock.advance(6)
        with self.assertRaises(SessionGuardError):
            harness.batch_guard(
                {"session_id": signed_root["session_id"], "items": ["legacy"]},
                forged_root,
                forged_request,
            )
        harness.registry.close_state(new_root)

    def test_expired_recovery_rejects_wrong_cookie_hash_token_and_resume_only(self):
        harness = self._new_harness()
        request, _, old_root = harness.seed("auth-" + uuid.uuid4().hex)
        thin = {
            "session_id": old_root["session_id"],
            "owner_token": old_root["owner_token"],
            "resume_id": old_root["resume_id"],
        }
        harness.clock.advance(6)

        bad_requests = (
            _request(request.session_hash, owner="different-cookie"),
            _request("different-hash-" + uuid.uuid4().hex),
        )
        for bad_request in bad_requests:
            with self.subTest(request=bad_request.username, session_hash=bad_request.session_hash):
                with self.assertRaises(SessionGuardError):
                    harness.echo_guard(thin, old_root, bad_request)

        forged_token = dict(thin, owner_token="not-the-owner-token")
        with self.assertRaises(SessionGuardError):
            harness.echo_guard(forged_token, old_root, request)

        resume_only = {
            "session_id": old_root["session_id"],
            "resume_id": old_root["resume_id"],
        }
        with self.assertRaises(SessionGuardError):
            harness.echo_guard(resume_only, None, request)

    def test_same_cookie_and_hash_can_recover_after_ip_change(self):
        harness = self._new_harness()
        request, state, old_root = harness.seed("ip-change-" + uuid.uuid4().hex)
        thin = {"session_id": old_root["session_id"], "owner_token": old_root["owner_token"]}
        harness.clock.advance(6)
        moved_request = _request(request.session_hash, host="10.70.0.99")

        self._run(harness.demo, harness.echo_fn, [thin, old_root], state, moved_request)

        new_root = state[harness.session_state._id]
        record = harness.registry.validate_owner(
            new_root["session_id"], RequestIdentity("cookie-owner", request.session_hash, "10.70.0.99")
        )
        self.assertNotEqual(new_root["session_id"], old_root["session_id"])
        self.assertEqual(record.client_ip, "10.70.0.99")
        harness.registry.close_state(new_root)

    def test_stale_queued_callback_preserves_recovered_state_and_other_page(self):
        harness = self._new_harness()
        request_a, state_a, old_root_a = harness.seed("page-a-" + uuid.uuid4().hex)
        harness.clock.advance(4)
        request_b, state_b, _ = harness.seed("page-b-" + uuid.uuid4().hex)
        page_b_before = copy.deepcopy(state_b.state_data)
        old_pcs = {
            "session_id": old_root_a["session_id"],
            "owner_token": old_root_a["owner_token"],
        }
        old_pvs = dict(old_pcs)
        harness.clock.advance(2)

        self._run(
            harness.demo,
            harness.destructive_fn,
            [old_pcs, old_pvs, old_root_a],
            state_a,
            request_a,
        )
        new_root = state_a[harness.session_state._id]
        self.assertEqual(harness.destructive_calls, [new_root["session_id"]])
        self.assertNotEqual(new_root["session_id"], old_root_a["session_id"])
        new_image = state_a[harness.image_state._id]
        new_image.update(image_id="current-image", owner_token=new_root["owner_token"])
        new_pcs = state_a[harness.pcs_state._id]
        new_pcs.update(instances={"current": {"id": "current-pcs"}})
        new_pvs = state_a[harness.pvs_state._id]
        new_pvs.update(instances={"current": {"id": "current-result"}})

        # A queued destructive request must be skipped, not replayed on new work.
        self._run(
            harness.demo,
            harness.destructive_fn,
            [old_pcs, old_pvs, old_root_a],
            state_a,
            request_a,
        )

        self.assertEqual(harness.destructive_calls, [new_root["session_id"]])
        self.assertEqual(state_a[harness.image_state._id]["image_id"], "current-image")
        self.assertIn("current", state_a[harness.pcs_state._id]["instances"])
        self.assertIn("current", state_a[harness.pvs_state._id]["instances"])
        self.assertEqual(state_a[harness.session_state._id]["session_id"], new_root["session_id"])
        self.assertEqual(state_b.state_data, page_b_before)
        harness.registry.close_state(new_root)

    def test_callback_exception_restores_identity_fields(self):
        harness = self._new_harness()
        request, state, old_root = harness.seed("callback-error-" + uuid.uuid4().hex)
        old_pcs = {
            "session_id": old_root["session_id"],
            "owner_token": old_root["owner_token"],
        }
        harness.clock.advance(6)

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            self._run(
                harness.demo,
                harness.failure_fn,
                [old_pcs, old_root],
                state,
                request,
            )

        new_root = state[harness.session_state._id]
        self.assertNotEqual(new_root["session_id"], old_root["session_id"])
        for component in (
            harness.session_state,
            harness.pcs_state,
            harness.image_state,
            harness.layout_state,
            harness.pvs_state,
            harness.stitch_state,
        ):
            value = state[component._id]
            self.assertEqual(value["session_id"], new_root["session_id"])
            self.assertEqual(value["owner_token"], new_root["owner_token"])
            self.assertEqual(value["resume_id"], new_root["resume_id"])
        self.assertEqual(state[harness.batch_state._id], {"items": [], "active_id": None})
        self.assertEqual(state[harness.extension_state._id], {"items": [], "active_id": None})

    def test_actual_app_upload_crop_recovers_all_gradio_states(self):
        clock = _Clock()
        registry = SessionRegistry(
            clock=clock,
            idle_seconds=5,
            secret=b"app-process-api-recovery-test",
            deployment_id="test",
            cleanup_callback=app._cleanup_session_record,
            start_sweeper=False,
        )
        with mock.patch.object(app, "_SESSION_REGISTRY", registry):
            demo = app.create_demo()
            holder = StateHolder()
            holder.set_blocks(demo)
            request = _request("app-recovery-" + uuid.uuid4().hex)
            state = demo.state_holder[request.session_hash]
            bootstrap_fn = _find_fn(demo, "_bootstrap_session")
            self._run(demo, bootstrap_fn, [], state, request)

            state_names = (
                "session_state",
                "image_state",
                "source_image_state",
                "pcs_state",
                "pvs_state",
                "template_match_state",
                "prompt_state",
                "layout_state",
                "layout_region_state",
                "layout_mask_agent_state",
                "stitch_state",
                "template_stitch_state",
            )
            state_components = dict(zip(state_names, bootstrap_fn.outputs))
            bootstrap_states = [
                component for component in bootstrap_fn.outputs
                if isinstance(component, gr.State)
            ]
            old_root = state[state_components["session_state"]._id]
            old_id = old_root["session_id"]
            for component in bootstrap_states:
                value = state[component._id]
                if isinstance(value, dict):
                    value["legacy_payload"] = component._id

            clock.advance(6)
            upload_fn = _find_fn(demo, "_source_upload_workspace", 4)
            upload_fn.preprocess = False
            upload_fn.inputs[0].data_model = None
            source_image = Image.new("RGB", (20, 14), (180, 90, 40))
            self._run(
                demo,
                upload_fn,
                [source_image, app.MODE_PVS, None, None],
                state,
                request,
            )

            new_root = state[state_components["session_state"]._id]
            new_id = new_root["session_id"]
            self.assertNotEqual(new_id, old_id)
            for component in bootstrap_states:
                value = state[component._id]
                if isinstance(value, dict) and value.get("session_id"):
                    self.assertEqual(value["session_id"], new_id, component._id)
                    self.assertNotIn("legacy_payload", value, component._id)
            for name in ("session_state", "layout_state", "pcs_state", "pvs_state", "stitch_state"):
                self.assertEqual(state[state_components[name]._id]["session_id"], new_id, name)

            source = state[state_components["source_image_state"]._id]
            gesture_fn = _find_fn(demo, "_record_source_crop_gesture", 2)
            gesture_fn.preprocess = False
            gesture = {
                "gesture": "drag",
                "expected_revision": source["source_revision"],
                "image_id": source["source_image_id"],
                "image_sha256": source["source_image_sha256"],
                "start_xy": [3, 2],
                "end_xy": [12, 10],
            }
            self._run(demo, gesture_fn, [None, gesture], state, request)
            self.assertEqual(
                state[state_components["source_image_state"]._id]["pending_crop_bbox_xyxy"],
                [3, 2, 12, 10],
            )

            crop_fn = _find_fn(demo, "_apply_source_crop", 4)
            self._run(demo, crop_fn, [None, app.MODE_PVS, None, None], state, request)
            cropped_source = state[state_components["source_image_state"]._id]
            cropped_image = state[state_components["image_state"]._id]
            self.assertEqual(cropped_source["crop_bbox_xyxy"], [3, 2, 12, 10])
            self.assertEqual((cropped_image["width"], cropped_image["height"]), (9, 8))
            for name in ("session_state", "layout_state", "pcs_state", "pvs_state", "stitch_state"):
                self.assertEqual(state[state_components[name]._id]["session_id"], new_id, name)

            registry.close_state(state[state_components["session_state"]._id])
        registry.shutdown()


if __name__ == "__main__":
    unittest.main()
