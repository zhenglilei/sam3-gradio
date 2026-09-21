from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gradio as gr
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_demo.session_guard import guard_callback
from sam3_demo.session_runtime import RequestIdentity, SessionRegistry


REQUIRED_EVENT_FIELDS = {
    "deployment_id",
    "git_sha",
    "action",
    "result",
    "duration_ms",
}


class _ControlledClock:
    def __init__(self, value: dt.datetime):
        self.value = value

    def __call__(self):
        return self.value

    def now(self):
        return self.value

    def time(self):
        return self.value.timestamp()


def _request(
    session_hash: str,
    *,
    owner_id: str = "raw-owner-value",
    host: str = "198.51.100.77",
    headers=None,
):
    return gr.Request(
        username=owner_id,
        session_hash=session_hash,
        client=SimpleNamespace(host=host),
        headers=headers or {},
    )


def _load_observability_class():
    try:
        from sam3_demo.observability import Observability
    except (ImportError, ModuleNotFoundError) as exc:
        raise AssertionError(
            "P0 observability API is not present: "
            "expected sam3_demo.observability.Observability"
        ) from exc
    return Observability


def _make_observability(
    root: Path,
    *,
    clock=None,
    max_bytes=None,
    retention_days=None,
):
    """Construct the contract object without depending on path alias details."""

    cls = _load_observability_class()
    parameters = inspect.signature(cls).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {}

    def add(names, value, *, required=False):
        selected = next((name for name in names if name in parameters), None)
        if selected is None and accepts_kwargs:
            selected = names[0]
        if selected is not None:
            kwargs[selected] = value
        elif required:
            raise AssertionError(
                f"Observability constructor has no supported parameter from {names!r}"
            )

    if any(
        name in parameters
        for name in ("log_dir", "directory", "output_dir", "log_directory")
    ) or accepts_kwargs:
        add(("log_dir", "directory", "output_dir", "log_directory"), root, required=True)
    else:
        add(
            ("log_path", "path", "file_path", "output_path"),
            root / "events.jsonl",
            required=True,
        )
    add(("deployment_id",), "observability-test-deployment", required=True)
    add(("secret",), b"observability-test-secret-32-bytes", required=True)
    add(("git_sha",), "observability-test-sha", required=True)
    if clock is not None:
        add(("wall_clock", "clock", "now", "now_fn", "time_fn"), clock, required=True)
    if "monotonic" in parameters or accepts_kwargs:
        add(("monotonic",), lambda: 0.0)
    if max_bytes is not None:
        add(
            ("max_bytes", "max_size_bytes", "rotate_bytes", "max_log_bytes"),
            max_bytes,
            required=True,
        )
    if retention_days is not None:
        add(("retention_days", "log_retention_days"), retention_days, required=True)
    return cls(**kwargs)


def _event_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".jsonl", ".log"}
            or ".jsonl." in path.name
            or ".log." in path.name
        )
    )


def _read_events(root: Path):
    events = []
    for path in _event_files(root):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise AssertionError(f"JSONL event is not an object at {path}:{line_number}")
            events.append((path, event))
    return events


def _close(value):
    for name in ("close", "shutdown", "stop"):
        method = getattr(value, name, None)
        if callable(method):
            method()
            return


def _flush(value):
    method = getattr(value, "flush", None)
    if callable(method):
        method()


def _empty_demo():
    with gr.Blocks() as demo:
        pass
    return demo


class _SupervisorProbe:
    def __init__(self, state):
        self.state = state
        self.start_calls = 0

    def snapshot(self):
        return {
            "state": self.state,
            "last_error": "synthetic model error" if self.state == "ERROR" else "",
        }

    def set_observability(self, _observability):
        return None

    def request_start(self, *_args, **_kwargs):
        self.start_calls += 1
        raise AssertionError("health/readiness endpoint started the model")


@contextlib.contextmanager
def _application_client(observability, settings, *, supervisor_state, client_address):
    from sam3_demo import app

    web = None
    registry = None
    client = None
    probe = _SupervisorProbe(supervisor_state)
    with mock.patch.object(app, "create_demo", side_effect=_empty_demo), mock.patch.object(
        app, "SUPERVISOR", probe
    ):
        try:
            web = app.create_application(
                settings,
                server_name="testserver",
                server_port=80,
                observability=observability,
            )
            registry = app._SESSION_REGISTRY
            client = TestClient(
                web,
                base_url="http://testserver",
                client=client_address,
                raise_server_exceptions=False,
            )
            yield client, probe
        finally:
            if client is not None:
                client.close()
            if registry is not None:
                registry.shutdown()
            app._SESSION_REGISTRY = app._new_session_registry()


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._registries = []
        self._observations = []

    def tearDown(self):
        for observation in reversed(self._observations):
            _close(observation)
        for registry in reversed(self._registries):
            registry.shutdown()
        self.temp_dir.cleanup()

    def _observation(self, name="logs", **kwargs):
        root = self.root / name
        root.mkdir(parents=True, exist_ok=True)
        observation = _make_observability(root, **kwargs)
        self._observations.append(observation)
        return observation, root

    def _registry(self, owner="raw-owner-value", session_hash="raw-session-value"):
        registry = SessionRegistry(secret=b"observability-test-secret", start_sweeper=False)
        self._registries.append(registry)
        state = registry.bind(RequestIdentity(owner, session_hash, "198.51.100.77"))
        return registry, state

    def _settings(self):
        from sam3_demo.session_web import load_session_cookie_settings

        secret_path = self.root / "session-cookie.key"
        secret_path.write_bytes(b"observability-test-cookie-secret-32-bytes")
        secret_path.chmod(0o600)
        return load_session_cookie_settings(
            {
                "SAM3_SESSION_DEPLOYMENT_ID": "observability-http-test",
                "SAM3_SESSION_COOKIE_NAME": "observability_test_cookie",
                "SAM3_SESSION_SECRET_FILE": str(secret_path),
                "SAM3_ALLOWED_ORIGINS": "http://testserver",
                "SAM3_SESSION_COOKIE_SECURE": "0",
                "SAM3_ALLOW_INSECURE_COOKIE": "1",
            }
        )

    def test_jsonl_schema_and_sensitive_values_are_redacted(self):
        observation, root = self._observation()
        registry, state = self._registry()
        raw_prompt = "raw-prompt-value"
        raw_file_name = "raw-file-name-value.png"
        raw_cookie = "raw-cookie-value"
        calls = []

        def callback(session_state, prompt, file_name):
            calls.append((session_state["session_id"], prompt, file_name))
            return "ok"

        guarded = guard_callback(
            callback,
            registry=registry,
            observability=observation,
        )
        request = _request(
            "raw-session-value",
            headers={"cookie": f"session={raw_cookie}"},
        )
        self.assertEqual(guarded(state, raw_prompt, raw_file_name, request), "ok")
        self.assertEqual(len(calls), 1)
        _flush(observation)

        events = _read_events(root)
        self.assertTrue(events, "callback did not produce a JSONL event")
        serialized = "\n".join(json.dumps(event, sort_keys=True) for _, event in events)
        for path, event in events:
            self.assertTrue(REQUIRED_EVENT_FIELDS.issubset(event), path)
            self.assertIsInstance(event["action"], str)
            self.assertTrue(event["action"])
            self.assertIsInstance(event["duration_ms"], (int, float))
            self.assertGreaterEqual(event["duration_ms"], 0)
        for raw_value in (
            "raw-owner-value",
            "raw-session-value",
            "198.51.100.77",
            raw_cookie,
            raw_prompt,
            raw_file_name,
        ):
            self.assertNotIn(raw_value, serialized)

    def test_callback_success_error_metrics_and_identity_recovery(self):
        observation, root = self._observation()
        registry, state = self._registry()
        original_state = dict(state)
        success = guard_callback(
            lambda session_state: "success",
            registry=registry,
            observability=observation,
        )
        self.assertEqual(success(state, _request("raw-session-value")), "success")

        def failing(session_state):
            session_state["owner_token"] = "forged-owner-token"
            session_state["session_id"] = "forged-session-id"
            session_state.pop("resume_id", None)
            raise RuntimeError("synthetic callback failure")

        guarded_failure = guard_callback(
            failing,
            registry=registry,
            observability=observation,
        )
        with self.assertRaises(gr.Error) as raised:
            guarded_failure(state, _request("raw-session-value"))
        self.assertIn("错误编号", str(raised.exception))
        self.assertEqual(state, original_state)
        _flush(observation)

        events = _read_events(root)
        failure_events = [
            event
            for _, event in events
            if str(event.get("result", "")).lower()
            in {"error", "failure", "failed", "exception"}
        ]
        self.assertTrue(failure_events, "exception callback was not recorded")
        self.assertTrue(
            any(isinstance(event.get("error_id"), str) and event["error_id"] for event in failure_events)
        )
        self.assertTrue(
            any(isinstance(event.get("error_code"), str) and event["error_code"] for event in failure_events)
        )

        settings = self._settings()
        with _application_client(
            observation,
            settings,
            supervisor_state="UNLOADED",
            client_address=("127.0.0.1", 50000),
        ) as (client, _probe):
            response = client.get("/metrics")
        self.assertEqual(response.status_code, 200, response.text)
        metric_lines = [line.lower() for line in response.text.splitlines() if "callback" in line.lower()]
        self.assertTrue(any("success" in line or "ok" in line for line in metric_lines))
        self.assertTrue(any("error" in line or "failure" in line for line in metric_lines))
        self.assertNotIn("error_id", response.text.lower())

    def test_date_size_rotation_and_thirty_day_cleanup(self):
        size_observation, size_root = self._observation(
            "size-logs",
            max_bytes=512,
            retention_days=30,
        )
        registry, state = self._registry("size-owner", "size-session")
        guarded = guard_callback(
            lambda session_state, value: value,
            registry=registry,
            observability=size_observation,
        )
        for index in range(8):
            self.assertEqual(
                guarded(
                    state,
                    f"small-value-{index}",
                    _request("size-session", owner_id="size-owner"),
                ),
                f"small-value-{index}",
            )
        _flush(size_observation)
        self.assertGreaterEqual(
            len(_event_files(size_root)),
            2,
            "size limit did not create a rotated log",
        )

        clock = _ControlledClock(dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc))
        date_observation, date_root = self._observation(
            "date-logs",
            clock=clock,
            max_bytes=4096,
            retention_days=30,
        )
        date_registry, date_state = self._registry("date-owner", "date-session")
        date_guarded = guard_callback(
            lambda session_state, value: value,
            registry=date_registry,
            observability=date_observation,
        )
        date_guarded(
            date_state,
            "before-date-change",
            _request("date-session", owner_id="date-owner"),
        )
        _flush(date_observation)
        before_files = set(_event_files(date_root))
        self.assertTrue(before_files)

        clock.value += dt.timedelta(days=31)
        date_guarded(
            date_state,
            "after-date-change",
            _request("date-session", owner_id="date-owner"),
        )
        _flush(date_observation)
        after_files = set(_event_files(date_root))
        self.assertTrue(after_files)
        self.assertTrue(
            after_files - before_files,
            "date change did not create a new active log",
        )
        self.assertLessEqual(
            len(_read_events(date_root)),
            1,
            "events older than the 30-day retention window were not removed",
        )

    def test_health_and_readiness_do_not_start_the_model(self):
        observation, _root = self._observation()
        settings = self._settings()
        with _application_client(
            observation,
            settings,
            supervisor_state="UNLOADED",
            client_address=("127.0.0.1", 50000),
        ) as (client, probe):
            health = client.get("/healthz")
            ready = client.get("/readyz")
        self.assertEqual(health.status_code, 200, health.text)
        self.assertEqual(ready.status_code, 200, ready.text)
        self.assertEqual(probe.start_calls, 0)

        with _application_client(
            observation,
            settings,
            supervisor_state="ERROR",
            client_address=("127.0.0.1", 50000),
        ) as (client, probe):
            health = client.get("/healthz")
            ready = client.get("/readyz")
        self.assertEqual(health.status_code, 200, health.text)
        self.assertEqual(ready.status_code, 503, ready.text)
        self.assertEqual(probe.start_calls, 0)

    def test_metrics_are_local_only_prometheus_text_without_high_cardinality_labels(self):
        observation, _root = self._observation()
        settings = self._settings()
        with _application_client(
            observation,
            settings,
            supervisor_state="UNLOADED",
            client_address=("127.0.0.1", 50000),
        ) as (client, _probe):
            local = client.get("/metrics")
        self.assertEqual(local.status_code, 200, local.text)
        self.assertTrue(local.headers.get("content-type", "").startswith("text/plain"))
        self.assertRegex(local.text, r"(?m)^[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{|\s)")
        for line in local.text.splitlines():
            if line.startswith("#") or "{" not in line or "}" not in line:
                continue
            labels = line.split("{", 1)[1].split("}", 1)[0].lower()
            for forbidden in ("owner", "session", "ip", "cookie", "prompt", "file", "error_id"):
                self.assertNotIn(forbidden, labels, line)

        with _application_client(
            observation,
            settings,
            supervisor_state="UNLOADED",
            client_address=("198.51.100.99", 50000),
        ) as (client, _probe):
            remote = client.get("/metrics")
        self.assertEqual(remote.status_code, 403, remote.text)

    def test_callback_runs_when_observability_log_directory_fails(self):
        observation, root = self._observation()
        registry, state = self._registry("writer-owner", "writer-session")
        blocked_dir = root / "blocked"
        blocked_dir.mkdir()
        observation.log_dir = blocked_dir

        original_mkdir = Path.mkdir

        def fail_log_dir_mkdir(path, *args, **kwargs):
            if path == blocked_dir:
                raise OSError("synthetic log directory failure")
            return original_mkdir(path, *args, **kwargs)

        calls = []

        def callback(session_state):
            calls.append(session_state["session_id"])
            return "business-result"

        guarded = guard_callback(
            callback,
            registry=registry,
            observability=observation,
        )
        with mock.patch.object(Path, "mkdir", new=fail_log_dir_mkdir):
            self.assertEqual(
                guarded(state, _request("writer-session", owner_id="writer-owner")),
                "business-result",
            )
        self.assertEqual(calls, [state["session_id"]])
        metrics = observation.render_metrics()
        self.assertRegex(metrics, r"sam3_log_write_failures_total\s+[1-9][0-9]*")


if __name__ == "__main__":
    unittest.main()
