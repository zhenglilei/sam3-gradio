"""Credential lifecycle contracts independent of model and UI implementation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from sam3_demo.session_credentials import BrowserCredentialRegistry, CredentialCapacityError
from sam3_demo.session_web import OwnerClaimRegistry, SessionCookieSettings, SessionSecurityMiddleware


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.live = set()
        self.registry = BrowserCredentialRegistry(
            "test-deploy", capacity=2, idle_seconds=10, clock=self.clock,
            owner_live_probe=lambda owner: owner in self.live,
        )

    def issue(self, registry=None):
        registry = registry or self.registry
        lease, token = registry.acquire(None, create=True)
        registry.release(lease, renew=True)
        return lease.owner_id, token

    def test_only_registered_tokens_resolve_and_registry_stores_no_bearer(self):
        owner, token = self.issue()
        self.assertRegex(token, r"^[A-Za-z0-9_-]{43}$")
        self.assertNotEqual(owner, token)
        self.assertNotIn(token, repr(self.registry._records))
        for invalid in (None, "", "x" * 43, token[:-1] + ".", "old.signed.cookie"):
            self.assertIsNone(self.registry.acquire(invalid))
        lease, same = self.registry.acquire(token, create=True)
        self.assertEqual(same, token)
        self.assertEqual(lease.owner_id, owner)
        self.registry.release(lease, renew=True)

    def test_sliding_expiration_requires_successful_activity(self):
        _, token = self.issue()
        self.clock.now += 9
        lease, _ = self.registry.acquire(token)
        self.registry.release(lease, renew=True)
        self.clock.now += 9
        lease, _ = self.registry.acquire(token)
        self.registry.release(lease, renew=False)
        self.clock.now += 1
        self.assertIsNone(self.registry.acquire(token))

    def test_capacity_never_evicts_active_browser_and_can_reclaim_idle(self):
        owner, token = self.issue()
        self.issue()
        with self.assertRaises(CredentialCapacityError):
            self.issue()
        self.live.add(owner)
        self.clock.now += 11
        new_owner, _ = self.issue()
        self.assertNotEqual(owner, new_owner)
        lease, _ = self.registry.acquire(token)
        self.assertEqual(lease.owner_id, owner)
        self.registry.release(lease, renew=True)

    def test_inflight_lease_and_queued_task_survive_concurrent_reaping(self):
        owner, token = self.issue()
        lease, _ = self.registry.acquire(token)
        self.clock.now += 11
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(lambda _: self.registry.reap(), range(20))), [0] * 20)
        self.live.add(owner)
        self.registry.release(lease, renew=False)
        self.assertEqual(self.registry.reap(), 0)
        self.live.clear()
        self.assertEqual(self.registry.reap(), 1)
        self.assertIsNone(self.registry.acquire(token))

    def test_unknown_liveness_retains_ownership(self):
        for probe in (None, lambda owner: None, lambda owner: 1 / 0):
            with self.subTest(probe=probe):
                registry = BrowserCredentialRegistry("test", idle_seconds=1, clock=self.clock, owner_live_probe=probe)
                self.issue(registry)
                self.clock.now += 2
                self.assertEqual(registry.reap(), 0)

    def test_simultaneous_admissions_respect_capacity(self):
        barrier = threading.Barrier(8)

        def admit(_):
            barrier.wait(timeout=5)
            try:
                self.issue()
                return True
            except CredentialCapacityError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(admit, range(8))), 2)
        self.assertEqual(self.registry.snapshot()["inflight"], 0)

    def test_restart_and_other_deployment_cannot_resolve_existing_token(self):
        _, token = self.issue()
        for deployment in ("test-deploy", "another-port"):
            restarted = BrowserCredentialRegistry(deployment)
            self.assertIsNone(restarted.acquire(token))


class CredentialHttpTests(unittest.TestCase):
    def build(self, name="browser_owner_7891", *, capacity=2, secure=False):
        clock = Clock()
        claims = OwnerClaimRegistry(max_hashes=capacity)
        credentials = BrowserCredentialRegistry("deployment-" + name, capacity=capacity,
                                                idle_seconds=10, clock=clock,
                                                owner_live_probe=claims.owner_is_live)
        api = FastAPI()

        @api.get("/")
        @api.get("/gradio_api/data")
        async def owner(request: Request):
            return {"owner": request.scope["session"]["owner_id"]}

        @api.get("/healthz")
        async def health():
            return {"ok": True}

        @api.get("/gradio_api/failure")
        async def fail():
            raise RuntimeError("synthetic callback failure")

        settings = SessionCookieSettings("deployment-" + name, name,
                                        frozenset({"http://testserver:80"}), secure)
        app = SessionSecurityMiddleware(api, settings=settings, claims=claims, credentials=credentials)
        return app, credentials, clock

    def test_only_homepage_creates_identity_and_full_capacity_is_503(self):
        app, registry, _ = self.build(capacity=1)
        with TestClient(app) as first, TestClient(app) as second:
            self.assertEqual(first.get("/gradio_api/data").status_code, 401)
            self.assertEqual(first.get("/healthz").status_code, 200)
            self.assertEqual(registry.snapshot()["count"], 0)
            home = first.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertEqual(first.get("/").json(), home.json())
            full = second.get("/")
            self.assertEqual(full.status_code, 503)
            self.assertEqual(full.json()["error"]["code"], "SESSION_CREDENTIAL_CAPACITY")
            self.assertEqual(full.headers["retry-after"], "5")
            self.assertEqual(first.get("/gradio_api/data").status_code, 200)

    def test_cookie_renewal_expiry_and_reopen(self):
        app, _, clock = self.build()
        with TestClient(app) as client:
            first = client.get("/").json()
            clock.now += 9
            response = client.get("/gradio_api/data")
            self.assertEqual(response.json(), first)
            self.assertIn("Max-Age=86400", response.headers["set-cookie"])
            clock.now += 11
            stale = client.get("/gradio_api/data")
            self.assertEqual(stale.status_code, 401)
            self.assertIn("重新打开首页", stale.json()["error"]["message"])
            self.assertEqual(stale.json()["detail"], stale.json()["error"]["message"])
            self.assertNotEqual(client.get("/").json(), first)

    def test_expired_config_gives_gradio_reopen_hint_without_a_login_prompt(self):
        app, _, _ = self.build()
        with TestClient(app) as client:
            response = client.get("/config")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["error"]["code"], "SESSION_COOKIE_REQUIRED")
            self.assertIn("重新打开首页", response.json()["detail"]["auth_message"])
            queued = client.post("/gradio_api/queue/join", json={})
            self.assertEqual(queued.status_code, 401)
            self.assertIn("重新打开首页", queued.json()["detail"])

    def test_independent_ports_even_if_both_cookies_are_sent(self):
        one, _, _ = self.build("browser_owner_7891")
        two, _, _ = self.build("browser_owner_7890")
        with TestClient(one) as first, TestClient(two) as second:
            owner = first.get("/").json()
            stolen = first.cookies.get("browser_owner_7891")
            second.cookies.set("browser_owner_7891", stolen)
            self.assertEqual(second.get("/gradio_api/data").status_code, 401)
            second.cookies.set("browser_owner_7890", stolen)
            self.assertEqual(second.get("/gradio_api/data").status_code, 401)
            second.cookies.clear()
            self.assertNotEqual(second.get("/").json(), owner)

    def test_restart_rejects_old_cookie_then_homepage_issues_new_identity(self):
        old, _, _ = self.build()
        new, _, _ = self.build()
        with TestClient(old) as before:
            old_owner = before.get("/").json()
            with TestClient(new, cookies=before.cookies) as after:
                self.assertEqual(after.get("/gradio_api/data").status_code, 401)
                self.assertNotEqual(after.get("/").json(), old_owner)

    def test_server_error_releases_request_lease(self):
        app, registry, _ = self.build()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.get("/")
            self.assertEqual(client.get("/gradio_api/failure").status_code, 500)
            self.assertEqual(registry.snapshot()["inflight"], 0)

    def test_cancelled_stream_releases_credential_lease(self):
        app, registry, _ = self.build()
        with TestClient(app) as client:
            client.get("/")
            token = client.cookies.get("browser_owner_7891")

        async def exercise():
            started = asyncio.Event()

            async def stream(scope, receive, send):
                await send({"type": "http.response.start", "status": 200, "headers": []})
                started.set()
                await asyncio.Event().wait()

            app.app = stream
            scope = {"type": "http", "method": "GET", "path": "/gradio_api/data", "query_string": b"",
                     "headers": [(b"cookie", ("browser_owner_7891=" + token).encode())]}

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                pass

            task = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(started.wait(), timeout=2)
            self.assertEqual(registry.snapshot()["inflight"], 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(registry.snapshot()["inflight"], 0)

        asyncio.run(exercise())

    def test_secure_cookie_has_host_only_attributes(self):
        app, _, _ = self.build(secure=True)
        with TestClient(app, base_url="https://testserver") as client:
            header = client.get("/").headers["set-cookie"].lower()
            for attribute in ("secure", "httponly", "samesite=lax", "path=/"):
                self.assertIn(attribute, header)
            self.assertNotIn("domain=", header)


if __name__ == "__main__":
    unittest.main()
