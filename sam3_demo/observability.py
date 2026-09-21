"""Low-dependency operational logging and metrics for SAM3 services."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_LOG_FILE_RE = re.compile(r"^app-(\d{4}-\d{2}-\d{2})\.(\d+)\.jsonl$")
_DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
_MODEL_STATES = (
    "UNLOADED",
    "SEARCHING_GPU",
    "WAITING_GPU",
    "LOADING",
    "WARMUP",
    "READY",
    "RUNNING",
    "REHYDRATING",
    "STOPPING",
    "ERROR",
)


def _safe_name(value: Any, *, fallback: str = "unknown") -> str:
    text = _SAFE_NAME_RE.sub("_", str(value or "").strip())[:96]
    return text or fallback


def _prometheus_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def resolve_git_sha(project_root: str | os.PathLike[str]) -> str:
    configured = os.environ.get("SAM3_BUILD_GIT_SHA", "").strip()
    if configured:
        return _safe_name(configured)
    try:
        value = subprocess.check_output(
            ["git", "-C", str(Path(project_root)), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return _safe_name(value)


class Observability:
    """Thread-safe JSONL event logger and in-process Prometheus registry."""

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        *,
        deployment_id: str,
        secret: bytes,
        git_sha: str | None = None,
        max_bytes: int = 20 * 1024 * 1024,
        retention_days: int = 30,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("observability secret must contain at least 16 bytes")
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        if int(retention_days) <= 0:
            raise ValueError("retention_days must be positive")
        self.log_dir = Path(log_dir)
        self.deployment_id = _safe_name(deployment_id)
        self.git_sha = _safe_name(git_sha or os.environ.get("SAM3_BUILD_GIT_SHA", "unknown"))
        self._secret = secret
        self.max_bytes = int(max_bytes)
        self.retention_days = int(retention_days)
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._started_at = self._monotonic()
        self._write_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._current_date: str | None = None
        self._current_index = 0
        self._current_path: Path | None = None
        self._current_file = None
        self._last_prune_date: str | None = None
        self._callback_total: dict[tuple[str, str], int] = defaultdict(int)
        self._callback_duration_count: dict[str, int] = defaultdict(int)
        self._callback_duration_sum: dict[str, float] = defaultdict(float)
        self._callback_duration_buckets: dict[tuple[str, float], int] = defaultdict(int)
        self._callback_errors: dict[tuple[str, str], int] = defaultdict(int)
        self._model_wait_count: dict[str, int] = defaultdict(int)
        self._model_wait_sum: dict[str, float] = defaultdict(float)
        self._session_rejections: dict[str, int] = defaultdict(int)
        self._model_restarts: dict[str, int] = defaultdict(int)
        self._log_write_failures = 0

    def now(self) -> datetime:
        value = self._wall_clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def monotonic(self) -> float:
        return float(self._monotonic())

    def uptime_seconds(self) -> float:
        return max(0.0, self.monotonic() - self._started_at)

    def digest(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return hmac.new(
            self._secret,
            str(value).encode("utf-8", "replace"),
            hashlib.sha256,
        ).hexdigest()[:16]

    @staticmethod
    def new_error_id() -> str:
        return f"err_{int(time.time() * 1000):x}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def error_code(exc: BaseException) -> str:
        explicit = getattr(exc, "code", None)
        return _safe_name(explicit or type(exc).__name__, fallback="INTERNAL")

    @staticmethod
    def _traceback_frames(exc: BaseException) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        current = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            tb = current.__traceback__
            while tb is not None:
                code = tb.tb_frame.f_code
                frames.append(
                    {
                        "file": str(Path(code.co_filename).resolve()),
                        "line": int(tb.tb_lineno),
                        "function": code.co_name,
                        "exception_type": type(current).__name__,
                    }
                )
                tb = tb.tb_next
            current = current.__cause__ or current.__context__
        return frames

    def _record_log_failure(self, exc: BaseException) -> None:
        with self._metrics_lock:
            self._log_write_failures += 1
        try:
            sys.stderr.write(f"observability log write failed: {type(exc).__name__}\n")
        except Exception:
            pass

    def _select_log_path(self, date_text: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[tuple[int, Path]] = []
        for path in self.log_dir.glob(f"app-{date_text}.*.jsonl"):
            match = _LOG_FILE_RE.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(2)), path))
        if candidates:
            index, path = max(candidates)
            if path.stat().st_size >= self.max_bytes:
                index += 1
                path = self.log_dir / f"app-{date_text}.{index}.jsonl"
        else:
            index = 0
            path = self.log_dir / f"app-{date_text}.0.jsonl"
        self._current_date = date_text
        self._current_index = index
        self._current_path = path
        self._current_file = path.open("a", encoding="utf-8", buffering=1)

    def _rotate_if_needed(self, date_text: str, encoded_size: int) -> None:
        if self._current_file is None or self._current_date != date_text:
            self.close()
            self._select_log_path(date_text)
            return
        current_size = self._current_path.stat().st_size if self._current_path else 0
        if current_size and current_size + encoded_size > self.max_bytes:
            self.close()
            self._current_date = date_text
            self._current_index += 1
            self._current_path = self.log_dir / f"app-{date_text}.{self._current_index}.jsonl"
            self._current_file = self._current_path.open("a", encoding="utf-8", buffering=1)

    def _prune_if_needed(self, now: datetime) -> None:
        date_text = now.strftime("%Y-%m-%d")
        if self._last_prune_date == date_text:
            return
        cutoff = now.date() - timedelta(days=self.retention_days)
        for path in self.log_dir.glob("app-*.jsonl"):
            match = _LOG_FILE_RE.fullmatch(path.name)
            if not match:
                continue
            try:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                if file_date < cutoff and path != self._current_path:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError):
                continue
        self._last_prune_date = date_text

    def emit(
        self,
        event: str,
        *,
        level: str = "INFO",
        exc: BaseException | None = None,
        **fields: Any,
    ) -> None:
        now = self.now()
        payload: dict[str, Any] = {
            "timestamp": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": _safe_name(level).upper(),
            "event": _safe_name(event),
            "deployment_id": self.deployment_id,
            "git_sha": self.git_sha,
        }
        payload.update({key: value for key, value in fields.items() if value is not None})
        if exc is not None:
            payload["exception_type"] = type(exc).__name__
            payload["traceback"] = self._traceback_frames(exc)
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                self._rotate_if_needed(now.strftime("%Y-%m-%d"), len(encoded))
                self._prune_if_needed(now)
                self._current_file.write(encoded.decode("utf-8"))
        except Exception as write_error:
            self._record_log_failure(write_error)

    def record_callback_success(
        self,
        action: str,
        duration_seconds: float,
        *,
        owner_id: Any = None,
        session_hash: Any = None,
    ) -> None:
        self._record_callback_metrics(action, "success", duration_seconds)
        self.emit(
            "callback_completed",
            action=_safe_name(action),
            result="success",
            duration_ms=round(max(0.0, duration_seconds) * 1000.0, 3),
            owner_digest=self.digest(owner_id),
            session_digest=self.digest(session_hash),
        )

    def record_callback_failure(
        self,
        action: str,
        duration_seconds: float,
        exc: BaseException,
        *,
        owner_id: Any = None,
        session_hash: Any = None,
        error_id: str | None = None,
    ) -> str:
        error_id = error_id or self.new_error_id()
        error_code = self.error_code(exc)
        self._record_callback_metrics(action, "error", duration_seconds, error_code=error_code)
        self.emit(
            "callback_failed",
            level="ERROR",
            exc=exc,
            action=_safe_name(action),
            result="error",
            duration_ms=round(max(0.0, duration_seconds) * 1000.0, 3),
            error_id=error_id,
            error_code=error_code,
            owner_digest=self.digest(owner_id),
            session_digest=self.digest(session_hash),
        )
        return error_id

    def _record_callback_metrics(
        self,
        action: str,
        result: str,
        duration_seconds: float,
        *,
        error_code: str | None = None,
    ) -> None:
        action = _safe_name(action)
        duration = max(0.0, float(duration_seconds))
        with self._metrics_lock:
            self._callback_total[(action, result)] += 1
            self._callback_duration_count[action] += 1
            self._callback_duration_sum[action] += duration
            for bucket in _DEFAULT_BUCKETS:
                if duration <= bucket:
                    self._callback_duration_buckets[(action, bucket)] += 1
            if error_code is not None:
                self._callback_errors[(action, _safe_name(error_code))] += 1

    def record_model_wait(self, action: str, duration_seconds: float) -> None:
        action = _safe_name(action)
        duration = max(0.0, float(duration_seconds))
        with self._metrics_lock:
            self._model_wait_count[action] += 1
            self._model_wait_sum[action] += duration
        self.emit(
            "model_lease_acquired",
            action=action,
            model_wait_ms=round(duration * 1000.0, 3),
        )

    def record_session_rejection(
        self,
        code: str,
        *,
        owner_id: Any = None,
        session_hash: Any = None,
        exc: BaseException | None = None,
    ) -> str:
        code = _safe_name(code)
        error_id = self.new_error_id()
        with self._metrics_lock:
            self._session_rejections[code] += 1
        self.emit(
            "session_rejected",
            level="WARNING",
            exc=exc,
            result="rejected",
            error_id=error_id,
            error_code=code,
            owner_digest=self.digest(owner_id),
            session_digest=self.digest(session_hash),
        )
        return error_id

    def record_model_restart(self, reason: str) -> None:
        reason = _safe_name(reason)
        with self._metrics_lock:
            self._model_restarts[reason] += 1
        self.emit("model_restart", reason=reason)

    def render_metrics(
        self,
        *,
        supervisor_snapshot: Mapping[str, Any] | None = None,
        session_snapshot: Mapping[str, Any] | None = None,
    ) -> str:
        supervisor = dict(supervisor_snapshot or {})
        sessions = dict(session_snapshot or {})
        lines: list[str] = []
        with self._metrics_lock:
            callback_total = dict(self._callback_total)
            duration_count = dict(self._callback_duration_count)
            duration_sum = dict(self._callback_duration_sum)
            duration_buckets = dict(self._callback_duration_buckets)
            callback_errors = dict(self._callback_errors)
            model_wait_count = dict(self._model_wait_count)
            model_wait_sum = dict(self._model_wait_sum)
            session_rejections = dict(self._session_rejections)
            model_restarts = dict(self._model_restarts)
            log_failures = int(self._log_write_failures)

        lines.extend(("# TYPE sam3_callback_total counter",))
        for (action, result), value in sorted(callback_total.items()):
            lines.append(f'sam3_callback_total{{action="{_prometheus_escape(action)}",result="{_prometheus_escape(result)}"}} {value}')
        lines.append("# TYPE sam3_callback_duration_seconds histogram")
        for action in sorted(duration_count):
            for bucket in _DEFAULT_BUCKETS:
                value = duration_buckets.get((action, bucket), 0)
                lines.append(f'sam3_callback_duration_seconds_bucket{{action="{_prometheus_escape(action)}",le="{bucket:g}"}} {value}')
            lines.append(f'sam3_callback_duration_seconds_bucket{{action="{_prometheus_escape(action)}",le="+Inf"}} {duration_count[action]}')
            lines.append(f'sam3_callback_duration_seconds_count{{action="{_prometheus_escape(action)}"}} {duration_count[action]}')
            lines.append(f'sam3_callback_duration_seconds_sum{{action="{_prometheus_escape(action)}"}} {duration_sum[action]:.9f}')
        lines.append("# TYPE sam3_callback_errors_total counter")
        for (action, code), value in sorted(callback_errors.items()):
            lines.append(f'sam3_callback_errors_total{{action="{_prometheus_escape(action)}",error_code="{_prometheus_escape(code)}"}} {value}')
        lines.append("# TYPE sam3_model_wait_seconds summary")
        for action in sorted(model_wait_count):
            lines.append(f'sam3_model_wait_seconds_count{{action="{_prometheus_escape(action)}"}} {model_wait_count[action]}')
            lines.append(f'sam3_model_wait_seconds_sum{{action="{_prometheus_escape(action)}"}} {model_wait_sum[action]:.9f}')
        lines.append("# TYPE sam3_model_state gauge")
        current_state = str(supervisor.get("state", "UNLOADED"))
        for state in _MODEL_STATES:
            lines.append(f'sam3_model_state{{state="{state}"}} {1 if state == current_state else 0}')
        lines.extend(
            (
                "# TYPE sam3_model_active_requests gauge",
                f'sam3_model_active_requests {int(supervisor.get("active_requests", 0) or 0)}',
                "# TYPE sam3_model_pending_requests gauge",
                f'sam3_model_pending_requests {int(supervisor.get("pending_requests", 0) or 0)}',
                "# TYPE sam3_active_sessions gauge",
                f'sam3_active_sessions {int(sessions.get("count", 0) or 0)}',
                "# TYPE sam3_session_rejections_total counter",
            )
        )
        for code, value in sorted(session_rejections.items()):
            lines.append(f'sam3_session_rejections_total{{code="{_prometheus_escape(code)}"}} {value}')
        lines.append("# TYPE sam3_model_restarts_total counter")
        for reason, value in sorted(model_restarts.items()):
            lines.append(f'sam3_model_restarts_total{{reason="{_prometheus_escape(reason)}"}} {value}')
        lines.extend(
            (
                "# TYPE sam3_log_write_failures_total counter",
                f"sam3_log_write_failures_total {log_failures}",
                "# TYPE sam3_process_uptime_seconds gauge",
                f"sam3_process_uptime_seconds {self.uptime_seconds():.3f}",
            )
        )
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        with self._write_lock:
            if self._current_file is not None:
                try:
                    self._current_file.close()
                finally:
                    self._current_file = None


__all__ = ["Observability", "resolve_git_sha"]
