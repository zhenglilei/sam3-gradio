from __future__ import annotations

import ast
import copy
import inspect
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from sam3_demo import app
from sam3_demo.session_guard import SessionGuardError


def _request(session_hash: str, host: str = "10.70.0.10"):
    return gr.Request(
        session_hash=session_hash,
        client=SimpleNamespace(host=host),
        headers={},
    )


class SessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.states = []

    def tearDown(self):
        for state in self.states:
            app._SESSION_REGISTRY.close_state(state)

    def _bootstrap(self, session_hash: str, host: str = "10.70.0.10"):
        bundle = app._bootstrap_session(_request(session_hash, host))
        self.states.append(bundle[0])
        return bundle

    def test_bootstrap_owns_every_business_state(self):
        before = app.SUPERVISOR.snapshot()
        bundle = self._bootstrap("browser-" + uuid.uuid4().hex)
        session_id = bundle[0]["session_id"]
        self.assertEqual(len(bundle), 10)
        for state in bundle[1:]:
            self.assertEqual(state["session_id"], session_id)
            self.assertEqual(state["owner_token"], bundle[0]["owner_token"])
        self.assertRegex(session_id, r"^[0-9a-f]{32}$")
        self.assertIn("owner_token", bundle[0])
        self.assertNotIn("client_ip", bundle[0])
        self.assertNotIn("session_hash", bundle[0])
        after = app.SUPERVISOR.snapshot()
        self.assertEqual(after["worker_pid"], before["worker_pid"])
        self.assertEqual(after["generation"], before["generation"])

    def test_same_ip_distinct_browser_hashes_and_ip_change_are_isolated(self):
        first_hash = "browser-a-" + uuid.uuid4().hex
        first = self._bootstrap(first_hash)
        second = self._bootstrap("browser-b-" + uuid.uuid4().hex)
        moved = self._bootstrap(first_hash, "10.70.0.11")
        ids = {first[0]["session_id"], second[0]["session_id"], moved[0]["session_id"]}
        self.assertEqual(len(ids), 3)

    def test_guard_rejects_cross_user_segmentation_state(self):
        first_hash = "browser-a-" + uuid.uuid4().hex
        first = self._bootstrap(first_hash)
        second = self._bootstrap("browser-b-" + uuid.uuid4().hex)
        callbacks = app._session_callback_registry()
        guarded = callbacks["_clear_pcs_instances"]
        before = copy.deepcopy(second[3])
        with self.assertRaises(SessionGuardError):
            guarded(
                first[1],
                second[3],
                first[4],
                app.MODE_PCS,
                _request(first_hash),
            )
        self.assertEqual(second[3], before)

    def test_guard_rejects_same_session_business_state_without_owner_token(self):
        browser_hash = "browser-a-" + uuid.uuid4().hex
        bundle = self._bootstrap(browser_hash)
        forged = {
            "session_id": bundle[0]["session_id"],
            "instances": {"forged": True},
        }
        guarded = app._session_callback_registry()["_clear_pcs_instances"]
        with self.assertRaises(SessionGuardError):
            guarded(
                bundle[1],
                forged,
                bundle[4],
                app.MODE_PCS,
                _request(browser_hash),
            )

    def test_guard_keeps_component_signature_and_injects_request(self):
        callbacks = app._session_callback_registry()
        original = app._run_pcs
        guarded = callbacks["_run_pcs"]
        original_parameters = list(inspect.signature(original).parameters.values())
        guarded_parameters = list(inspect.signature(guarded).parameters.values())
        self.assertEqual(len(guarded_parameters), len(original_parameters) + 1)
        self.assertIs(guarded_parameters[-1].annotation, gr.Request)

    def test_every_bound_stateful_callback_is_session_guarded(self):
        app_source = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
        bindings_path = Path(app.__file__).parent / "ui" / "bindings.py"
        bindings_source = ast.parse(bindings_path.read_text(encoding="utf-8"))
        parameters = {
            node.name: [argument.arg for argument in node.args.args]
            for node in app_source.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        bound_callbacks = {
            keyword.value.id
            for node in ast.walk(bindings_source)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "fn" and isinstance(keyword.value, ast.Name)
        }
        stateful_callbacks = {
            name
            for name in bound_callbacks
            if any(
                argument == "state" or argument.endswith("_state")
                for argument in parameters.get(name, ())
            )
        }
        self.assertEqual(
            stateful_callbacks - app._SESSION_GUARDED_CALLBACKS,
            set(),
        )

    def test_guarded_state_output_can_be_used_by_next_callback(self):
        browser_hash = "browser-chain-" + uuid.uuid4().hex
        bundle = self._bootstrap(browser_hash)
        guarded = app._session_callback_registry()["_clear_pcs_instances"]
        first_result = guarded(
            bundle[1],
            bundle[3],
            bundle[4],
            app.MODE_PCS,
            _request(browser_hash),
        )
        next_pcs_state = first_result[0]
        self.assertEqual(
            next_pcs_state["owner_token"],
            bundle[0]["owner_token"],
        )
        second_result = guarded(
            bundle[1],
            next_pcs_state,
            bundle[4],
            app.MODE_PCS,
            _request(browser_hash),
        )
        self.assertEqual(
            second_result[0]["owner_token"],
            bundle[0]["owner_token"],
        )

    def test_create_demo_has_private_nonqueued_bootstrap(self):
        demo = app.create_demo()
        load_dependencies = [
            item
            for item in demo.config["dependencies"]
            if any(target[1] == "load" for target in item.get("targets") or [])
        ]
        self.assertEqual(len(load_dependencies), 1)
        dependency = load_dependencies[0]
        self.assertEqual(len(dependency["inputs"]), 0)
        self.assertEqual(len(dependency["outputs"]), 10)
        self.assertFalse(dependency["queue"])
        self.assertEqual(dependency["show_progress"], "hidden")
        self.assertEqual(dependency["api_visibility"], "private")
        session_component = demo.blocks[dependency["outputs"][0]]
        self.assertEqual(session_component.time_to_live, float("inf"))

    def test_unload_request_does_not_close_or_rebind_session(self):
        browser_hash = "browser-unload-" + uuid.uuid4().hex
        bundle = self._bootstrap(browser_hash)
        app._close_request_session(_request(browser_hash))
        record = app._SESSION_REGISTRY.validate(
            bundle[0],
            browser_hash,
            "10.70.0.10",
        )
        self.assertEqual(record.session_id, bundle[0]["session_id"])

    def test_uninitialized_legacy_state_cannot_create_a_session(self):
        self.assertEqual(app._new_session_state()["session_id"], "")
        self.assertEqual(app._new_layout_state()["session_id"], "")
        with self.assertRaisesRegex(ValueError, "not initialized"):
            app._session_id_from_state(None)


if __name__ == "__main__":
    unittest.main()
