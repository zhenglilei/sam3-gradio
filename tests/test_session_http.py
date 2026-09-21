from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import uuid
from base64 import b64decode, b64encode
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from sam3_demo import app
from sam3_demo.session_web import (
    COOKIE_MAX_AGE_SECONDS,
    SessionWebError,
    load_session_cookie_settings,
)


ORIGIN = "http://testserver"
COOKIE_NAME = "sam3_el_http_test"


class SessionHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        secret_path = Path(cls.temp_dir.name) / "session-cookie.key"
        secret_path.write_bytes(b"session-cookie-http-test-secret-32-bytes-minimum")
        secret_path.chmod(0o600)
        cls.settings = load_session_cookie_settings(
            {
                "SAM3_SESSION_DEPLOYMENT_ID": "el-http-test",
                "SAM3_SESSION_COOKIE_NAME": COOKIE_NAME,
                "SAM3_SESSION_SECRET_FILE": str(secret_path),
                "SAM3_ALLOWED_ORIGINS": ORIGIN,
                "SAM3_SESSION_COOKIE_SECURE": "0",
                "SAM3_ALLOW_INSECURE_COOKIE": "1",
            }
        )
        with mock.patch.object(
            app.gr,
            "mount_gradio_app",
            wraps=app.gr.mount_gradio_app,
        ) as mount_gradio_app:
            cls.web = app.create_application(
                cls.settings,
                server_name="testserver",
                server_port=80,
            )
        cls.mount_gradio_app_kwargs = mount_gradio_app.call_args.kwargs
        cls.web_registry = app._SESSION_REGISTRY
        cls.lifespan_client = TestClient(
            cls.web,
            base_url=ORIGIN,
            client=("198.51.100.254", 50000),
            raise_server_exceptions=False,
        )
        cls.lifespan_client.__enter__()
        dependencies = cls.web.state.gradio_blocks.config["dependencies"]
        cls.bootstrap = next(
            item for item in dependencies if item.get("api_name") == "_bootstrap_session"
        )
        cls.clear_pcs = next(
            item for item in dependencies if item.get("api_name") == "_clear_pcs_instances"
        )
        cls.session_component_id = cls.bootstrap["outputs"][0]

    @classmethod
    def tearDownClass(cls):
        cls.lifespan_client.__exit__(None, None, None)
        cls.web_registry.shutdown()
        app._SESSION_REGISTRY = app._new_session_registry()
        cls.temp_dir.cleanup()

    def _client(self, ip: str, cookies=None, *, raise_exceptions=False):
        return TestClient(
            self.web,
            base_url=ORIGIN,
            client=(ip, 50000),
            cookies=cookies,
            raise_server_exceptions=raise_exceptions,
        )

    def _new_owner(self, ip="198.51.100.10"):
        client = self._client(ip)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        return client, response

    def _bootstrap(self, client, session_hash, *, headers=None):
        request_headers = {"Origin": ORIGIN}
        request_headers.update(headers or {})
        return client.post(
            "/gradio_api/run/_bootstrap_session",
            json={
                "data": [],
                "fn_index": self.bootstrap["id"],
                "trigger_id": self.bootstrap["id"],
                "session_hash": session_hash,
            },
            headers=request_headers,
        )

    def _signed_cookie(self, payload, *, timestamp=None):
        signer = TimestampSigner(self.settings.signer_secret)
        encoded = b64encode(json.dumps(payload).encode("utf-8"))
        if timestamp is None:
            return signer.sign(encoded).decode("ascii")
        with mock.patch("itsdangerous.timed.time.time", return_value=timestamp):
            return signer.sign(encoded).decode("ascii")

    def test_mount_preserves_demo_theme_and_css(self):
        self.assertEqual(
            self.mount_gradio_app_kwargs["css"],
            app.CUSTOM_CSS,
        )
        self.assertIsNotNone(self.mount_gradio_app_kwargs["theme"])

    def test_homepage_issues_minimal_host_only_http_cookie(self):
        client, response = self._new_owner()
        try:
            header = response.headers["set-cookie"]
            self.assertIn("Max-Age=86400", header)
            self.assertIn("httponly", header.lower())
            self.assertIn("samesite=lax", header.lower())
            self.assertNotIn("secure", header.lower())
            self.assertNotIn("domain=", header.lower())
            raw = client.cookies.get(COOKIE_NAME)
            unsigned = TimestampSigner(self.settings.signer_secret).unsign(
                raw,
                max_age=COOKIE_MAX_AGE_SECONDS,
            )
            payload = json.loads(b64decode(unsigned))
            self.assertEqual(
                set(payload),
                {"owner_id", "schema", "deployment_id"},
            )
            self.assertGreaterEqual(len(payload["owner_id"]), 43)
        finally:
            client.close()

    def test_missing_tampered_expired_and_cross_deployment_cookie_are_rejected(self):
        body = {"data": [], "fn_index": self.bootstrap["id"], "session_hash": "x"}
        variants = {
            "missing": None,
            "tampered": "not-a-valid-signed-cookie",
            "expired": self._signed_cookie(
                {
                    "owner_id": "o" * 43,
                    "schema": 1,
                    "deployment_id": self.settings.deployment_id,
                },
                timestamp=time.time() - COOKIE_MAX_AGE_SECONDS - 60,
            ),
            "cross-deployment": self._signed_cookie(
                {
                    "owner_id": "o" * 43,
                    "schema": 1,
                    "deployment_id": "pvs-other-deployment",
                }
            ),
        }
        for label, cookie in variants.items():
            with self.subTest(label=label):
                client = self._client("198.51.100.11")
                try:
                    if cookie is not None:
                        client.cookies.set(COOKIE_NAME, cookie)
                    response = client.post(
                        "/gradio_api/queue/join",
                        json=body,
                        headers={"Origin": ORIGIN},
                    )
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(
                        response.json()["error"]["code"],
                        "SESSION_COOKIE_REQUIRED",
                    )
                finally:
                    client.close()

    def test_same_owner_same_hash_survives_ip_change_and_guarded_callback_runs(self):
        session_hash = "ip-change-" + uuid.uuid4().hex
        first, _ = self._new_owner("198.51.100.20")
        try:
            existing_ids = {
                item["session_id"]
                for item in app._SESSION_REGISTRY.snapshot()["sessions"]
            }
            response = self._bootstrap(first, session_hash)
            self.assertEqual(response.status_code, 200, response.text)
            first_snapshot = app._SESSION_REGISTRY.snapshot()
            new_ids = {
                item["session_id"] for item in first_snapshot["sessions"]
            } - existing_ids
            self.assertEqual(len(new_ids), 1)
            first_session_id = next(iter(new_ids))

            moved = self._client("203.0.113.20", cookies=first.cookies)
            try:
                response = self._bootstrap(moved, session_hash)
                self.assertEqual(response.status_code, 200, response.text)
                moved_snapshot = app._SESSION_REGISTRY.snapshot()
                self.assertEqual(moved_snapshot["count"], first_snapshot["count"])
                self.assertIn(
                    first_session_id,
                    {item["session_id"] for item in moved_snapshot["sessions"]},
                )
                guarded = moved.post(
                    "/gradio_api/run/_clear_pcs_instances",
                    json={
                        "data": [None, None, None, app.MODE_PCS],
                        "fn_index": self.clear_pcs["id"],
                        "trigger_id": self.clear_pcs["id"],
                        "session_hash": session_hash,
                    },
                    headers={"Origin": ORIGIN},
                )
                self.assertEqual(guarded.status_code, 200, guarded.text)
                self.assertIn("已清空", guarded.text)
            finally:
                moved.close()
        finally:
            first.close()

    def test_different_owner_cannot_claim_or_write_existing_hash(self):
        session_hash = "owner-isolation-" + uuid.uuid4().hex
        first, _ = self._new_owner("198.51.100.30")
        second, _ = self._new_owner("198.51.100.31")
        try:
            self.assertEqual(self._bootstrap(first, session_hash).status_code, 200)
            before = app._SESSION_REGISTRY.snapshot()
            rejected = self._bootstrap(second, session_hash)
            self.assertEqual(rejected.status_code, 403)
            self.assertEqual(
                rejected.json()["error"]["code"],
                "SESSION_OWNER_MISMATCH",
            )
            self.assertEqual(app._SESSION_REGISTRY.snapshot(), before)
        finally:
            first.close()
            second.close()

    def test_queue_sse_event_query_and_cancel_are_owner_bound(self):
        session_hash = "queue-owner-" + uuid.uuid4().hex
        first, _ = self._new_owner("198.51.100.40")
        second, _ = self._new_owner("198.51.100.41")
        try:
            queue_response = first.post(
                "/gradio_api/queue/join",
                json={
                    "data": [],
                    "fn_index": self.bootstrap["id"],
                    "trigger_id": self.bootstrap["id"],
                    "session_hash": session_hash,
                },
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(queue_response.status_code, 200)
            event_id = queue_response.json()["event_id"]

            with first.stream(
                "GET",
                "/gradio_api/queue/data",
                params={"session_hash": session_hash},
            ) as stream:
                self.assertEqual(stream.status_code, 200)
                self.assertTrue(
                    stream.headers["content-type"].startswith("text/event-stream")
                )
                stream_text = "".join(stream.iter_text())
            self.assertIn("process_completed", stream_text)

            rejected_sse = second.get(
                "/gradio_api/queue/data",
                params={"session_hash": session_hash},
            )
            self.assertEqual(rejected_sse.status_code, 403)
            rejected_cancel = second.post(
                "/gradio_api/cancel",
                json={
                    "session_hash": session_hash,
                    "fn_index": self.bootstrap["id"],
                    "event_id": event_id,
                },
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(rejected_cancel.status_code, 403)

            call_response = first.post(
                "/gradio_api/call/_bootstrap_session",
                json={"data": [], "session_hash": session_hash + "-call"},
                headers={"Origin": ORIGIN},
            )
            self.assertEqual(call_response.status_code, 200)
            call_event = call_response.json()["event_id"]
            with first.stream(
                "GET",
                f"/gradio_api/call/_bootstrap_session/{call_event}",
            ) as event_stream:
                self.assertEqual(event_stream.status_code, 200)
                self.assertTrue(
                    event_stream.headers["content-type"].startswith(
                        "text/event-stream"
                    )
                )
            rejected_event = second.get(
                f"/gradio_api/call/_bootstrap_session/{call_event}"
            )
            self.assertEqual(rejected_event.status_code, 403)
        finally:
            first.close()
            second.close()

    def test_write_origin_requires_exact_origin_or_referer(self):
        client, _ = self._new_owner("198.51.100.50")
        try:
            session_hash = "origin-" + uuid.uuid4().hex
            body = {
                "data": [],
                "fn_index": self.bootstrap["id"],
                "trigger_id": self.bootstrap["id"],
                "session_hash": session_hash,
            }
            missing = client.post("/gradio_api/run/_bootstrap_session", json=body)
            self.assertEqual(missing.status_code, 403)
            bad = client.post(
                "/gradio_api/run/_bootstrap_session",
                json=body,
                headers={"Origin": "http://evil.example"},
            )
            self.assertEqual(bad.status_code, 403)
            referer = client.post(
                "/gradio_api/run/_bootstrap_session",
                json=body,
                headers={"Referer": ORIGIN + "/"},
            )
            self.assertEqual(referer.status_code, 200, referer.text)
        finally:
            client.close()


class SessionCookieConfigTests(unittest.TestCase):
    def test_http_requires_explicit_insecure_cookie_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "secret"
            secret.write_bytes(b"x" * 32)
            secret.chmod(0o600)
            with self.assertRaisesRegex(SessionWebError, "explicit"):
                load_session_cookie_settings(
                    {
                        "SAM3_SESSION_DEPLOYMENT_ID": "test-deploy",
                        "SAM3_SESSION_COOKIE_NAME": "test_cookie_name",
                        "SAM3_SESSION_SECRET_FILE": str(secret),
                        "SAM3_ALLOWED_ORIGINS": ORIGIN,
                        "SAM3_SESSION_COOKIE_SECURE": "0",
                    }
                )

    def test_secret_permissions_must_be_0600(self):
        if os.name == "nt":
            self.skipTest("POSIX permission assertion")
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "secret"
            secret.write_bytes(b"x" * 32)
            secret.chmod(0o644)
            with self.assertRaisesRegex(SessionWebError, "0600"):
                load_session_cookie_settings(
                    {
                        "SAM3_SESSION_DEPLOYMENT_ID": "test-deploy",
                        "SAM3_SESSION_COOKIE_NAME": "test_cookie_name",
                        "SAM3_SESSION_SECRET_FILE": str(secret),
                        "SAM3_ALLOWED_ORIGINS": ORIGIN,
                        "SAM3_SESSION_COOKIE_SECURE": "1",
                    }
                )


if __name__ == "__main__":
    unittest.main()
