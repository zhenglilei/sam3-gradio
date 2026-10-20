"""Load the packaged Xingzhou settings, check CUDA, then execute the single web worker."""

import argparse
import json
import os
from pathlib import Path
import sys


class StartupError(ValueError):
    """An actionable failure whose message contains no credentials."""


def emit(check, status, **details):
    print(json.dumps({"event": "xingzhou_startup", "check": check, "status": status, **details}),
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
    expected_scheme = "https://" if settings.secure_cookie else "http://"
    if not all(origin.startswith(expected_scheme) for origin in settings.allowed_origins):
        raise StartupError("SAM3_SESSION_COOKIE_SECURE must match every SAM3_ALLOWED_ORIGINS scheme")
    return {"deployment_id": settings.deployment_id, "origins": sorted(settings.allowed_origins),
            "secure_cookie": settings.secure_cookie, "configuration_sources": sources}


def check_gpu():
    import torch

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run preflight without starting a web server")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    for name, check in (
        ("session", lambda: configure(root)),
        ("gpu", check_gpu),
    ):
        try:
            details = check()
        except Exception as exc:
            message = str(exc) if isinstance(exc, StartupError) else "Check " + name + " configuration and runtime"
            emit(name, "failed", error_type=type(exc).__name__, message=message)
            return 1
        emit(name, "ok", **details)
    emit("preflight", "ok", listen="0.0.0.0:8030", check_only=args.check)
    if not args.check:
        os.execv(sys.executable, [sys.executable, "-m", "sam3_demo.app"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
