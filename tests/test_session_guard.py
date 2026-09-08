from __future__ import annotations

import inspect
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import gradio as gr
from gradio import helpers


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_demo.session_guard import SessionGuardError, guard_callback, request_identity
from sam3_demo.session_runtime import SessionRegistry


def _request(session_hash: str | None, host: str = "10.0.0.1", headers=None):
    return gr.Request(
        session_hash=session_hash,
        client=SimpleNamespace(host=host),
        headers=headers or {},
    )


class SessionGuardTest(unittest.TestCase):
    def setUp(self):
        self.registry = SessionRegistry(secret=b"test-secret", start_sweeper=False)
        self.state_a = self.registry.bind("browser-a", "10.0.0.1")
        self.state_b = self.registry.bind("browser-b", "10.0.0.1")

    def tearDown(self):
        self.registry.shutdown()

    def test_request_identity_and_trusted_proxy(self):
        session_hash, client_ip = request_identity(
            _request("browser-a", "10.0.0.2"),
            trusted_proxy_cidrs=["10.0.0.0/24"],
        )
        self.assertEqual((session_hash, client_ip), ("browser-a", "10.0.0.2"))
        session_hash, client_ip = request_identity(
            _request(
                "browser-a",
                "10.0.0.2",
                {"X-Forwarded-For": "198.51.100.8, 10.0.0.3"},
            ),
            trusted_proxy_cidrs=["10.0.0.0/24"],
        )
        self.assertEqual((session_hash, client_ip), ("browser-a", "198.51.100.8"))

    def test_missing_request_identity_is_rejected(self):
        with self.assertRaises(SessionGuardError):
            request_identity(None)
        with self.assertRaises(SessionGuardError):
            request_identity(_request(None))
        with self.assertRaises(SessionGuardError):
            request_identity(gr.Request(session_hash="browser-a"))

    def test_special_args_injects_request_without_component_input(self):
        calls = []

        def callback(state, value):
            calls.append((state["session_id"], value))
            return "<div>ok</div>"

        guarded = guard_callback(callback, registry=self.registry)
        original_count = len(inspect.signature(callback).parameters)
        guarded_parameters = list(inspect.signature(guarded).parameters.values())
        self.assertEqual(len(guarded_parameters), original_count + 1)
        self.assertIs(guarded_parameters[-1].annotation, gr.Request)
        self.assertIsNone(guarded_parameters[-1].default)
        injected, progress, event_data, props = helpers.special_args(
            guarded,
            inputs=[self.state_a, "value"],
            request=_request("browser-a"),
        )
        self.assertEqual(len(injected), original_count + 1)
        self.assertIsNone(progress)
        self.assertIsNone(event_data)
        self.assertEqual(props, [])
        self.assertEqual(guarded(*injected), "<div>ok</div>")
        self.assertEqual(calls, [(self.state_a["session_id"], "value")])

    def test_same_ip_different_browser_hash_is_isolated(self):
        calls = []

        def callback(state):
            calls.append(state["session_id"])
            return state["session_id"]

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(SessionGuardError):
            guarded(self.state_a, _request("browser-b"))
        self.assertEqual(calls, [])
        self.assertEqual(guarded(self.state_b, _request("browser-b")), self.state_b["session_id"])

    def test_cross_user_state_is_rejected_before_callback(self):
        calls = []

        def callback(state):
            calls.append(True)

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(SessionGuardError):
            guarded(self.state_a, _request("browser-b"))
        self.assertFalse(calls)

    def test_multiple_conflicting_states_are_rejected(self):
        calls = []

        def callback(first, second):
            calls.append(True)

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(SessionGuardError):
            guarded(self.state_a, self.state_b, _request("browser-a"))
        self.assertFalse(calls)

    def test_owner_only_resource_handle_is_supported(self):
        calls = []
        handle = {"session_id": self.state_a["session_id"], "region_id": 4}

        def callback(resource):
            calls.append(resource["region_id"])
            return "ok"

        guarded = guard_callback(callback, registry=self.registry)
        self.assertEqual(guarded(handle, _request("browser-a")), "ok")
        self.assertEqual(calls, [4])

    def test_business_state_requires_owner_token_and_is_never_forwarded_when_forged(self):
        calls = []

        def callback(session_state, pvs_state):
            calls.append(pvs_state)
            return pvs_state

        guarded = guard_callback(callback, registry=self.registry)
        valid = {
            "session_id": self.state_a["session_id"],
            "owner_token": self.state_a["owner_token"],
            "instances": {},
        }
        result = guarded(self.state_a, valid, _request("browser-a"))
        self.assertEqual(result["owner_token"], self.state_a["owner_token"])
        self.assertEqual(len(calls), 1)

        for forged in (
            {"session_id": self.state_a["session_id"], "instances": {"forged": True}},
            {
                "session_id": self.state_a["session_id"],
                "owner_token": self.state_b["owner_token"],
                "instances": {"forged": True},
            },
            {"owner_token": self.state_a["owner_token"], "instances": {"forged": True}},
        ):
            with self.subTest(forged=forged):
                with self.assertRaises(SessionGuardError):
                    guarded(self.state_a, forged, _request("browser-a"))
        self.assertEqual(len(calls), 1)

    def test_callback_stamps_new_business_state_output(self):
        def callback(session_state):
            return {"session_id": session_state["session_id"], "instances": {}}

        guarded = guard_callback(callback, registry=self.registry)
        result = guarded(self.state_a, _request("browser-a"))
        self.assertEqual(result["owner_token"], self.state_a["owner_token"])

    def test_missing_states_are_rebound_with_fresh_business_data(self):
        def callback(session_state, stitch_state):
            return stitch_state

        def recover(server_state):
            fresh = {
                "session_id": server_state["session_id"],
                "owner_token": server_state["owner_token"],
                "resume_id": server_state["resume_id"],
                "images": [],
            }
            return {"session_state": server_state, "stitch_state": fresh}

        guarded = guard_callback(
            callback,
            registry=self.registry,
            recovery_factory=recover,
        )
        result = guarded(None, {}, _request("new-browser"))
        self.assertEqual(result["images"], [])
        self.assertTrue(result["session_id"])

    def test_stale_same_browser_state_is_replaced_after_restart(self):
        old_registry = SessionRegistry(secret=b"old-secret")
        stale = old_registry.bind("browser-restart", "10.0.0.1")
        stale_business = dict(stale, instances={"must": "discard"})
        old_registry.shutdown()
        new_registry = SessionRegistry(secret=b"new-secret")

        def callback(session_state, pvs_state):
            return pvs_state

        def recover(server_state):
            return {
                "session_state": server_state,
                "pvs_state": {
                    "session_id": server_state["session_id"],
                    "owner_token": server_state["owner_token"],
                    "resume_id": server_state["resume_id"],
                    "instances": {},
                },
            }

        try:
            guarded = guard_callback(
                callback,
                registry=new_registry,
                recovery_factory=recover,
            )
            result = guarded(stale, stale_business, _request("browser-restart"))
            self.assertEqual(result["instances"], {})
            self.assertNotEqual(result["session_id"], stale["session_id"])
        finally:
            new_registry.shutdown()

    def test_callback_error_releases_lease(self):
        def callback(state):
            raise RuntimeError("boom")

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(RuntimeError):
            guarded(self.state_a, _request("browser-a"))
        record = self.registry.snapshot()["sessions"][0]
        self.assertEqual(record["in_flight"], 0)

    def test_callback_error_restores_business_state_owner_identity(self):
        business_state = {
            "session_id": self.state_a["session_id"],
            "owner_token": self.state_a["owner_token"],
            "instances": {},
        }

        def callback(session_state, pvs_state):
            pvs_state["session_id"] = self.state_b["session_id"]
            pvs_state.pop("owner_token")
            raise RuntimeError("after mutation")

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaisesRegex(RuntimeError, "after mutation"):
            guarded(self.state_a, business_state, _request("browser-a"))
        self.assertEqual(business_state["session_id"], self.state_a["session_id"])
        self.assertEqual(business_state["owner_token"], self.state_a["owner_token"])

    def test_base_exception_restores_complete_session_identity(self):
        original = dict(self.state_a)

        def callback(session_state):
            session_state["schema_version"] = 999
            session_state["generation"] = 999
            session_state["session_hash_digest"] = "forged"
            session_state["client_ip_digest"] = "forged"
            session_state.pop("owner_token")
            raise SystemExit("stop")

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaisesRegex(SystemExit, "stop"):
            guarded(self.state_a, _request("browser-a"))
        self.assertEqual(self.state_a, original)
        self.assertEqual(
            self.registry.validate(
                self.state_a,
                "browser-a",
                "10.0.0.1",
            ).session_id,
            self.state_a["session_id"],
        )

    def test_close_during_callback_rejects_stale_result(self):
        calls = []

        def callback(state):
            calls.append(True)
            self.assertTrue(self.registry.close(self.state_a, "browser-a", "10.0.0.1"))
            return "stale"

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(SessionGuardError):
            guarded(self.state_a, _request("browser-a"))
        self.assertEqual(calls, [True])
        remaining_ids = {item["session_id"] for item in self.registry.snapshot()["sessions"]}
        self.assertEqual(remaining_ids, {self.state_b["session_id"]})

    def test_no_session_state_never_calls_callback(self):
        calls = []

        def callback(value):
            calls.append(value)

        guarded = guard_callback(callback, registry=self.registry)
        with self.assertRaises(SessionGuardError):
            guarded("not state", _request("browser-a"))
        self.assertEqual(calls, [])

    def test_invalid_callback_shapes_are_rejected(self):
        def variadic(*args):
            return args

        def request_callback(state, request: gr.Request):
            return state, request

        def conflict_callback(state, __session_guard_request=None):
            return state

        with self.assertRaises(TypeError):
            guard_callback(variadic, registry=self.registry)
        with self.assertRaises(TypeError):
            guard_callback(request_callback, registry=self.registry)
        with self.assertRaises(TypeError):
            guard_callback(conflict_callback, registry=self.registry)


if __name__ == "__main__":
    unittest.main()
