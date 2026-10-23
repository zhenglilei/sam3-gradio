"""Load the packaged Xingzhou settings, check CUDA, then execute the single web worker."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback


STARTED_AT = time.monotonic()


class StartupError(ValueError):
    """An actionable failure whose message contains no credentials."""


def emit(check, status, **details):
    print(json.dumps({"event": "xingzhou_startup", "check": check, "status": status,
                      "pid": os.getpid(), "elapsed_s": round(time.monotonic() - STARTED_AT, 3), **details}),
          file=sys.stderr if status == "failed" else sys.stdout, flush=True)


def configure(root):
    # The platform cannot inject environment variables, so this file is the
    # deployment's configuration entry point. Explicit environment values win.
    profile = json.loads((root / "xingzhou-config.json").read_text())
    if not isinstance(profile, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in profile.items()
    ):
        raise StartupError("xingzhou-config.json must be a JSON object of string settings")
    sources = {name: "environment" if name in os.environ else "profile" for name in profile}
    for name, value in profile.items():
        os.environ.setdefault(name, value)
    manifest = root / "bundle-manifest.json"
    if manifest.is_file():
        # Version label for observability logs only; nothing is validated here.
        git_sha = json.loads(manifest.read_text()).get("git_sha")
        if isinstance(git_sha, str):
            os.environ.setdefault("SAM3_BUILD_GIT_SHA", git_sha)
    from sam3_demo.session_web import load_session_cookie_settings

    settings = load_session_cookie_settings()
    # A Secure cookie is never sent over HTTP; a non-Secure cookie works on both.
    if settings.secure_cookie and any(origin.startswith("http://") for origin in settings.allowed_origins):
        raise StartupError("SAM3_SESSION_COOKIE_SECURE=1 cannot be used with http:// SAM3_ALLOWED_ORIGINS")
    return {"deployment_id": settings.deployment_id, "origins": sorted(settings.allowed_origins),
            "secure_cookie": settings.secure_cookie, "configuration_sources": sources}


def check_gpu():
    started = time.monotonic()
    emit("torch_import", "begin")
    import torch

    emit("torch_import", "ok", duration_s=round(time.monotonic() - started, 3),
         version=torch.__version__, source=torch.__file__, cuda=torch.version.cuda)
    emit("cuda_init", "begin")
    if not torch.cuda.is_available():
        raise StartupError("CUDA is unavailable; check GPU allocation, driver and PyTorch in the selected image")
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if free < 8192 * 1024 * 1024:
        emit("gpu", "warning", message="Less than 8192 MiB free; model loading may wait for GPU memory")
    return {"device": properties.name, "cuda": torch.version.cuda,
            "free_mib": free // (1024 * 1024), "total_mib": total // (1024 * 1024),
            "bf16_supported": torch.cuda.is_bf16_supported()}


def main(argv=None):
    emit("entry", "begin", python=sys.version.split()[0], executable=sys.executable,
         cwd=str(Path.cwd()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run preflight without starting a web server")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    for name, check in (
        ("session", lambda: configure(root)),
        ("gpu", check_gpu),
    ):
        started = time.monotonic()
        emit(name, "begin")
        try:
            details = check()
        except Exception as exc:
            # Session validation can echo an invalid origin containing credentials.
            hide_value = name == "session" and isinstance(exc, ValueError) and not isinstance(exc, StartupError)
            message = "Check session configuration (invalid value omitted)" if hide_value else str(exc)
            emit(name, "failed", error_type=type(exc).__name__, message=message,
                 duration_s=round(time.monotonic() - started, 3))
            if hide_value:
                traceback.print_tb(exc.__traceback__, file=sys.stderr)
            else:
                traceback.print_exc()
            return 1
        emit(name, "ok", duration_s=round(time.monotonic() - started, 3), **details)
    emit("preflight", "ok", listen="0.0.0.0:8030", check_only=args.check)
    if not args.check:
        emit("app_exec", "begin", module="sam3_demo.app", git_sha=os.environ.get("SAM3_BUILD_GIT_SHA", "unknown"))
        os.execv(sys.executable, [sys.executable, "-m", "sam3_demo.app"])
    return 0


if __name__ == "__main__":
    faulthandler.enable()
    # Diagnose a stuck preflight without imposing another startup timeout.
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        raise SystemExit(main())
    finally:
        faulthandler.cancel_dump_traceback_later()
