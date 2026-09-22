from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from gradio.queueing import Event
from starlette.middleware.sessions import SessionMiddleware
from starlette.testclient import TestClient

from sam3_demo.session_web import (
    _MAX_JSON_CONTROL_BODY,
    OwnerClaimRegistry,
    SessionClaimCapacityError,
    SessionClientDisconnected,
    SessionCookieSettings,
    SessionSecurityMiddleware,
    SessionWebError,
    _read_json_body,
    protect_gradio_state_holder,
)


ORIGIN = "http://session-lifecycle.test"
COOKIE_NAME = "session_lifecycle_owner"
DEPLOYMENT_ID = "session-lifecycle-test"


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SessionHttpLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.claims = self._build_app()

    def _build_app(self, *, max_hashes: int = 32, max_events: int | None = None):
        self.backend_states: set[str] = set()
        self.queue_hashes: set[str] = set()
        self.queue_events: set[str] = set()
        settings = SessionCookieSettings(
            deployment_id=DEPLOYMENT_ID,
            cookie_name=COOKIE_NAME,
            secret=b"synthetic-session-secret-for-http-tests-only",
            allowed_origins=frozenset({ORIGIN + ":80"}),
            secure_cookie=False,
        )
        claims = OwnerClaimRegistry(
            max_hashes=max_hashes,
            max_events=max_events,
            hash_live_probe=lambda session_hash: (
                session_hash in self.backend_states
                or session_hash in self.queue_hashes
                or any(
                    associated_hash == session_hash
                    for associated_hash in self.backend_events.values()
                )
            ),
            event_live_probe=lambda event_id: event_id in self.queue_events,
        )
        self.backend_events: dict[str, str] = {}
        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.get("/")
        async def home():
            return {"ok": True}

        @api.post("/gradio_api/run/{api_name}")
        async def run(api_name: str, request: Request):
            body = await request.json()
            if api_name == "missing":
                raise HTTPException(status_code=404)
            self.backend_states.add(body["session_hash"])
            return {"ok": True}

        @api.post("/gradio_api/queue/join")
        async def queue_join(request: Request):
            body = await request.json()
            session_hash = body["session_hash"]
            self.queue_hashes.add(session_hash)
            event_id = "event-" + uuid.uuid4().hex
            self.backend_events[event_id] = session_hash
            self.queue_events.add(event_id)
            return {"event_id": event_id}

        @api.get("/gradio_api/queue/data")
        async def queue_data(session_hash: str):
            return {"session_hash": session_hash}

        @api.get("/gradio_api/call/{api_name}/{event_id}")
        async def get_event(api_name: str, event_id: str):
            return {"event_id": event_id}

        middleware = SessionSecurityMiddleware(api, settings=settings, claims=claims)
        return SessionMiddleware(
            middleware, secret_key=settings.signer_secret,
            session_cookie=COOKIE_NAME, same_site="lax", https_only=False,
        ), claims

    def _client(self, ip: str = "198.51.100.10", *, cookies=None) -> TestClient:
        return TestClient(
            self.app,
            base_url=ORIGIN,
            client=(ip, 50000),
            cookies=cookies,
            raise_server_exceptions=False,
        )

    def _owner_client(self, ip: str = "198.51.100.10") -> TestClient:
        client = self._client(ip)
        self.assertEqual(client.get("/").status_code, 200)
        self.assertTrue(client.cookies.get(COOKIE_NAME))
        return client

    def _bootstrap(self, client: TestClient, session_hash: str):
        return client.post(
            "/gradio_api/run/bootstrap",
            json={"session_hash": session_hash},
            headers={"Origin": ORIGIN},
        )

    def test_unmatched_and_404_routes_do_not_retain_hash_claims(self):
        client = self._owner_client()
        try:
            unmatched = client.get("/not-a-route?session_hash=unmatched-hash")
            self.assertEqual(unmatched.status_code, 404)
            self.assertEqual(self.claims.snapshot()["hash_claims"], 0)

            missing_new = client.post(
                "/gradio_api/run/missing", json={"session_hash": "missing-new-hash"},
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(missing_new.status_code, 404)
            self.assertEqual(self.claims.snapshot()["hash_claims"], 0)

            response = self._bootstrap(client, "route-returns-404")
            self.assertEqual(response.status_code, 200)
            missing = client.post(
                "/gradio_api/run/missing",
                json={"session_hash": "route-returns-404"},
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(missing.status_code, 404)
            self.assertEqual(self.claims.snapshot()["hash_claims"], 1)
        finally:
            client.close()

    def test_bad_json_and_oversize_have_distinct_http_errors(self):
        client = self._owner_client()
        headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
        try:
            malformed = client.post(
                "/gradio_api/run/bootstrap",
                content=b"{",
                headers=headers,
            )
            self.assertEqual(malformed.status_code, 400)
            self.assertEqual(
                malformed.json()["error"]["code"], "SESSION_REQUEST_INVALID"
            )

            oversized = client.post(
                "/gradio_api/run/bootstrap",
                content=b" " * (_MAX_JSON_CONTROL_BODY + 1),
                headers=headers,
            )
            self.assertEqual(oversized.status_code, 413)
            self.assertEqual(
                oversized.json()["error"]["code"], "SESSION_REQUEST_TOO_LARGE"
            )
            self.assertEqual(self.claims.snapshot()["hash_claims"], 0)
        finally:
            client.close()

    def test_capacity_is_retryable_and_not_misreported_as_owner_mismatch(self):
        self.app, self.claims = self._build_app(max_hashes=1)
        first = self._owner_client("198.51.100.11")
        second = self._owner_client("198.51.100.12")
        try:
            self.assertEqual(self._bootstrap(first, "occupied-hash").status_code, 200)
            full = self._bootstrap(second, "another-hash")
            self.assertEqual(full.status_code, 503)
            self.assertEqual(full.headers["retry-after"], "5")
            self.assertEqual(
                full.json()["error"]["code"], "SESSION_CLAIM_CAPACITY"
            )
        finally:
            first.close()
            second.close()

    def test_json_media_types_cannot_bypass_hash_leases(self):
        client = self._owner_client()
        self.addCleanup(client.close)
        for index, content_type in enumerate((None, "application/vendor+json", "text/plain")):
            with self.subTest(content_type=content_type):
                headers = {"Origin": ORIGIN}
                if content_type is not None:
                    headers["Content-Type"] = content_type
                response = client.post(
                    "/gradio_api/run/bootstrap",
                    content=json.dumps({"session_hash": f"media-type-{index}"}),
                    headers=headers,
                )
                self.assertEqual(response.status_code, 400 if content_type == "text/plain" else 200)
        self.assertEqual(self.claims.snapshot()["hash_claims"], 2)

    def test_new_cookie_cannot_read_live_state_or_queue_event(self):
        first = self._owner_client("198.51.100.21")
        second = self._owner_client("198.51.100.22")
        session_hash = "live-state-and-event"
        try:
            self.assertEqual(self._bootstrap(first, session_hash).status_code, 200)
            joined = first.post(
                "/gradio_api/queue/join",
                json={"session_hash": session_hash},
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(joined.status_code, 200)
            event_id = joined.json()["event_id"]

            state_read = second.get(
                "/gradio_api/queue/data",
                params={"session_hash": session_hash},
            )
            event_read = second.get(f"/gradio_api/call/callback/{event_id}")
            self.assertEqual(state_read.status_code, 403)
            self.assertEqual(event_read.status_code, 403)
            self.assertEqual(
                state_read.json()["error"]["code"], "SESSION_OWNER_MISMATCH"
            )
            self.assertIn(session_hash, self.backend_states)
            self.assertIn(event_id, self.queue_events)
        finally:
            first.close()
            second.close()

    def test_same_cookie_remains_valid_after_ip_change(self):
        first = self._owner_client("198.51.100.31")
        session_hash = "ip-change-still-valid"
        moved = None
        try:
            self.assertEqual(self._bootstrap(first, session_hash).status_code, 200)
            moved = self._client("203.0.113.31", cookies=first.cookies)
            response = moved.get(
                "/gradio_api/queue/data",
                params={"session_hash": session_hash},
            )
            self.assertEqual(response.status_code, 200, response.text)
        finally:
            first.close()
            if moved is not None:
                moved.close()


class SessionClaimLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock()
        self.state_hashes: set[str] = set()
        self.queue_hashes: set[str] = set()
        self.live_events: set[str] = set()
        self.claims = self._registry()

    def _registry(self, *, max_hashes: int = 4, max_events: int = 4):
        return OwnerClaimRegistry(
            max_hashes=max_hashes,
            max_events=max_events,
            claim_ttl_seconds=10,
            hash_live_probe=lambda session_hash: (
                session_hash in self.state_hashes
                or session_hash in self.queue_hashes
            ),
            event_live_probe=lambda event_id: event_id in self.live_events,
            clock=self.clock,
        )

    def test_expired_hash_and_event_reclaim_only_after_backend_and_queue_are_gone(self):
        old_owner = "o" * 43
        new_owner = "n" * 43
        self.claims.claim_hash(old_owner, "reusable-hash")
        self.claims.bind_event(old_owner, "reusable-hash", "old-event")
        self.state_hashes.add("reusable-hash")
        self.live_events.add("old-event")

        self.clock.advance(11)
        self.claims.reap_expired_claims()
        self.assertEqual(self.claims.snapshot(), {"hash_claims": 1, "event_claims": 1})
        with self.assertRaises(SessionWebError):
            self.claims.claim_hash(new_owner, "reusable-hash")

        self.state_hashes.clear()
        self.queue_hashes.clear()
        self.live_events.clear()
        self.clock.advance(11)
        self.claims.claim_hash(new_owner, "reusable-hash")
        self.assertEqual(self.claims.snapshot(), {"hash_claims": 1, "event_claims": 0})
        with self.assertRaises(SessionWebError):
            self.claims.require_event(new_owner, "old-event")
        with self.assertRaises(SessionWebError):
            self.claims.require_hash(old_owner, "reusable-hash")

    def test_inflight_request_prevents_reassignment_after_ttl(self):
        old_owner = "o" * 43
        new_owner = "n" * 43
        lease = self.claims.begin_request(
            old_owner,
            session_hash="inflight-hash",
            claim_hash=True,
        )
        self.clock.advance(11)
        self.claims.reap_expired_claims()
        with self.assertRaises(SessionWebError):
            self.claims.claim_hash(new_owner, "inflight-hash")
        self.claims.end_request(lease, response_status=200)
        self.assertEqual(self.claims.snapshot()["hash_claims"], 1)

    def test_dead_event_is_reaped_while_its_state_hash_stays_owner_bound(self):
        old_owner = "o" * 43
        new_owner = "n" * 43
        self.claims.claim_hash(old_owner, "active-state")
        self.claims.bind_event(old_owner, "active-state", "finished-event")
        self.state_hashes.add("active-state")
        self.clock.advance(11)
        self.claims.reap_expired_claims()
        self.assertEqual(self.claims.snapshot(), {"hash_claims": 1, "event_claims": 0})
        with self.assertRaises(SessionWebError):
            self.claims.claim_hash(new_owner, "active-state")
        with self.assertRaises(SessionWebError):
            self.claims.require_event(new_owner, "finished-event")

    def test_state_holder_probe_is_wired_by_protection_helper(self):
        class StateHolder:
            def __init__(self):
                self.session_data = {"state-still-readable": object()}

            def __contains__(self, session_hash):
                return session_hash in self.session_data

        holder = StateHolder()
        queue = SimpleNamespace(
            pending_messages_per_session={},
            pending_event_ids_session={},
            event_ids_to_events={},
        )
        blocks = SimpleNamespace(state_holder=holder, _queue=queue)
        gradio_app = SimpleNamespace(
            state_holder=holder,
            iterators={},
            get_blocks=lambda: blocks,
        )
        claims = self._registry(max_hashes=1)
        protect_gradio_state_holder(gradio_app, claims)
        self.clock.advance(11)
        with self.assertRaises(SessionWebError):
            claims.claim_hash("n" * 43, "state-still-readable")
        self.assertIn("state-still-readable", holder.session_data)

    def test_event_capacity_raises_typed_retryable_error(self):
        owner = "o" * 43
        self.claims = self._registry(max_hashes=3, max_events=1)
        self.claims.claim_hash(owner, "hash-one")
        lease = self.claims.begin_request(
            owner,
            session_hash="hash-one",
            claim_hash=True,
            reserve_event=True,
        )
        self.claims.bind_reserved_event(lease, owner, "hash-one", "event-one")
        self.claims.end_request(lease, response_status=200)
        self.claims.claim_hash(owner, "hash-two")
        with self.assertRaises(SessionClaimCapacityError):
            self.claims.begin_request(
                owner,
                session_hash="hash-two",
                claim_hash=True,
                reserve_event=True,
            )

    def test_finished_event_is_reclaimed_without_resetting_active_workspace(self):
        with gr.Blocks() as demo:
            component = gr.State(None)
            button = gr.Button()
            button.click(lambda state: state, component, component)
        demo.queue()
        web = gr.routes.App.create_app(demo)
        holder, queue = web.state_holder, demo._queue
        clock = _FakeClock()
        claims = OwnerClaimRegistry(max_events=1, claim_ttl_seconds=10, clock=clock)
        protect_gradio_state_holder(web, claims)
        owner, page = "o" * 43, "active-workspace"
        claims.claim_hash(owner, page)
        state = holder[page]
        state[component._id] = {"private": "current-image"}
        event = Event(page, demo.fns[0], None, owner)
        queue.event_ids_to_events[event._id] = event
        queue.pending_event_ids_session[page] = {event._id}
        claims.bind_event(owner, page, event._id)
        clock.advance(6)
        claims.require_hash(owner, page)
        clock.advance(5)
        lease = claims.begin_request(owner, session_hash=page, reserve_event=True)
        try:
            self.assertEqual(claims.snapshot(), {"hash_claims": 1, "event_claims": 0})
            self.assertIs(holder[page], state)
            self.assertEqual(state[component._id], {"private": "current-image"})
            self.assertNotIn(event._id, queue.event_ids_to_events)
            self.assertNotIn(event._id, queue.pending_event_ids_session[page])
            with self.assertRaises(SessionWebError):
                claims.require_event(owner, event._id)
        finally:
            claims.end_request(lease, response_status=200)

    def test_real_gradio_cache_is_invalidated_before_releasing_claims(self):
        for busy in (None, "http", "queued", "running", "iterator"):
            with self.subTest(busy=busy):
                with gr.Blocks() as demo:
                    component = gr.State(None)
                    button = gr.Button()
                    button.click(lambda state: state, component, component)
                demo.queue()
                web = gr.routes.App.create_app(demo)
                holder = web.state_holder
                queue = demo._queue
                clock = _FakeClock()
                claims = OwnerClaimRegistry(max_hashes=1, claim_ttl_seconds=10, clock=clock)
                protect_gradio_state_holder(web, claims)
                old_owner, new_owner, page = "o" * 43, "n" * 43, "reclaim-page"
                claims.claim_hash(old_owner, page)
                old_state = holder[page]
                old_state[component._id] = {"private": "old-image"}
                event = Event(page, demo.fns[0], None, old_owner)
                queue.event_ids_to_events[event._id] = event
                queue.pending_event_ids_session[page] = {event._id}
                queue.pending_messages_per_session[page] = asyncio.Queue()
                claims.bind_event(old_owner, page, event._id)
                lease = None
                if busy == "http":
                    lease = claims.begin_request(old_owner, session_hash=page)
                elif busy == "queued":
                    queue.event_queue_per_concurrency_id[event.concurrency_id] = SimpleNamespace(queue=[event])
                elif busy == "running":
                    queue.active_jobs = [[event]]
                elif busy == "iterator":
                    web.iterators[event._id] = object()
                clock.advance(11)
                claims.reap_expired_claims()
                if busy:
                    self.assertIn(page, holder.session_data)
                    self.assertEqual(old_state[component._id], {"private": "old-image"})
                    with self.assertRaises(SessionWebError):
                        claims.claim_hash(new_owner, page)
                    with self.assertRaises(SessionClaimCapacityError):
                        claims.claim_hash(new_owner, "another-page")
                    if lease is not None:
                        claims.end_request(lease, response_status=200)
                    queue.active_jobs = []
                    queue.event_queue_per_concurrency_id.clear()
                    web.iterators.clear()
                    clock.advance(11)
                    claims.reap_expired_claims()
                self.assertEqual(claims.snapshot(), {"hash_claims": 0, "event_claims": 0})
                self.assertNotIn(page, holder.session_data)
                self.assertNotIn(page, holder.time_last_used)
                self.assertFalse(old_state.state_data)
                self.assertNotIn(page, queue.pending_messages_per_session)
                self.assertNotIn(event._id, queue.event_ids_to_events)
                claims.claim_hash(new_owner, page)
                self.assertIsNone(holder[page][component._id])
                with self.assertRaises(SessionWebError):
                    claims.require_event(new_owner, event._id)


class SessionBodyLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_disconnect_terminates_after_one_receive(self):
        for prefix in ([], [{"type": "http.request", "body": b"{", "more_body": True}]):
            messages = prefix + [{"type": "http.disconnect"}] * 2
            receive_count = 0

            async def receive() -> dict[str, Any]:
                nonlocal receive_count
                message = messages[receive_count]
                receive_count += 1
                return message

            with self.assertRaises(SessionClientDisconnected):
                await _read_json_body(receive)
            self.assertEqual(receive_count, len(prefix) + 1)


if __name__ == "__main__":
    unittest.main()
