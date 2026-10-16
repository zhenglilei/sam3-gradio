"""Validate the extracted Xingzhou bundle before executing its single web worker."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile


PROFILE_KEYS = frozenset({
    "SAM3_SESSION_DEPLOYMENT_ID", "SAM3_SESSION_COOKIE_NAME", "SAM3_ALLOWED_ORIGINS",
    "SAM3_SESSION_COOKIE_SECURE", "SAM3_ALLOW_INSECURE_COOKIE",
})
COMPONENTS = ("image_gesture_overlay", "layout_transform_editor", "layout_region_annotator",
              "stitch_preview_canvas", "repair_mask_editor")
SKILL = "sam3_demo/layout/skills/layout-mask-preprocess/"
REQUIRED = (
    "run.sh", "xingzhou_startup.py", "xingzhou-config.json", "sam3_demo/app.py",
    "models/sam3.pt", "models/big-lama.pt", "assets/bpe_simple_vocab_16e6.txt.gz",
    *(SKILL + name for name in ("SKILL.md", "references/operation-catalog.md",
                               "references/profile-priors.md", "references/response-schema.json")),
    *(f"{name}/backend/gradio_{name}/templates/component/index.js" for name in COMPONENTS),
)


class StartupError(ValueError):
    """An actionable failure whose message contains no credentials."""


def emit(check, status, **details):
    print(json.dumps({"event": "xingzhou_startup", "check": check, "status": status, **details}),
          file=sys.stderr if status == "failed" else sys.stdout, flush=True)


def configure(root):
    profile = json.loads((root / "xingzhou-config.json").read_text())
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS or not all(
        isinstance(value, str) for value in profile.values()
    ):
        raise StartupError("xingzhou-config.json must contain only the five documented session settings")
    sources = {name: "environment" if name in os.environ else "profile" for name in profile}
    for name, value in profile.items():
        os.environ.setdefault(name, value)
        if any(marker in os.environ[name].upper() for marker in
               ("REPLACE_WITH", "CHANGE_ME", "CHANGEME", "YOUR_SERVICE_HOST")):
            raise StartupError(name + " contains a placeholder; remove the stale environment override "
                               "to use xingzhou-config.json, or provide the real deployment value")
    from sam3_demo.session_web import load_session_cookie_settings

    settings = load_session_cookie_settings()
    expected_scheme = "https://" if settings.secure_cookie else "http://"
    if not all(origin.startswith(expected_scheme) for origin in settings.allowed_origins):
        raise StartupError("SAM3_SESSION_COOKIE_SECURE must match every SAM3_ALLOWED_ORIGINS scheme")
    return {"deployment_id": settings.deployment_id, "origins": sorted(settings.allowed_origins),
            "secure_cookie": settings.secure_cookie, "configuration_sources": sources}


def check_bundle(root):
    manifest = json.loads((root / "bundle-manifest.json").read_text())
    sha = manifest.get("git_sha", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise StartupError("bundle-manifest.json must contain a full Git SHA")
    if (manifest.get("schema") != 2 or manifest.get("container_port") != 8030
            or manifest.get("entrypoint") != "/src/run.sh"):
        raise StartupError("bundle-manifest.json does not describe the /src/run.sh:8030 contract")
    configured_sha = os.environ.get("SAM3_BUILD_GIT_SHA", sha)
    if configured_sha != sha:
        raise StartupError("SAM3_BUILD_GIT_SHA conflicts with the packaged Git SHA")
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        raise StartupError("bundle-manifest.json has no file inventory")
    # External model mounts are intentionally not members of the source-only ZIP.
    for name in REQUIRED:
        if not name.startswith("models/") and name not in entries:
            raise StartupError("Missing required manifest entry: " + name)
    for name in set(entries) | set(REQUIRED):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise StartupError("Unsafe relative path in bundle-manifest.json")
        target = root / name
        if not target.is_file() or (name in REQUIRED and target.stat().st_size == 0):
            raise StartupError("Required file is missing or empty: " + name)
        if name in entries and target.stat().st_size != entries[name]["size"]:
            raise StartupError("Packaged file size differs from manifest: " + name)
    os.environ["SAM3_BUILD_GIT_SHA"] = sha
    return {"git_sha": sha, "files": len(entries), "model_files": "present"}


def check_dependencies(root):
    if sys.version_info < (3, 12):
        raise StartupError("Python 3.12 or newer is required; select the SAM3 runtime image")
    modules = ("torch", "torchvision", "gradio", "fastapi", "starlette", "pydantic",
               "numpy", "cv2", "PIL", "psutil", "requests", "sam3.model_builder")
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise StartupError("Cannot import " + name + "; check the selected image dependencies") from exc
    for name in COMPONENTS:
        sys.path.insert(0, str(root / name / "backend"))
        try:
            importlib.import_module("gradio_" + name)
        except Exception as exc:
            raise StartupError("Cannot import local component " + name) from exc
    return {"python": sys.version.split()[0], "versions": {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "gradio", "fastapi", "starlette", "pydantic", "numpy")
    }}


def check_directories(root):
    # Config creates directories on import, before the individual write probes.
    try:
        from sam3_demo import config
    except OSError as exc:
        raise StartupError("Cannot initialize runtime directory: " + str(exc.filename or root)
                           + "; grant the container user write access to the mounted runtime directories") from exc

    paths = {value for name, value in vars(config).items()
             if name.startswith("runtime_") and name.endswith("_dir") and isinstance(value, Path)}
    paths.update((config.runtime_dir, config.public_download_dir, config.qiyuan_cache_dir,
                  config.qiyuan_cache_dir / "huggingface/hub", config.qiyuan_cache_dir / "modelscope",
                  config.runtime_feedback_dir / "samples", root / ".gradio"))
    for path in sorted(paths):
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path) as probe:
                probe.write(b"startup write probe")
                probe.flush()
        except OSError as exc:
            raise StartupError("Runtime directory is not writable: " + str(path)) from exc
    return {"writable_directories": len(paths)}


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
        ("bundle", lambda: check_bundle(root)),
        ("directories", lambda: check_directories(root)),
        ("dependencies", lambda: check_dependencies(root)),
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
