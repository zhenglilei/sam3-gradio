#!/bin/sh
set -eu

STARTED_AT=$(date +%s)
STAGE=entry
boot_log() {
    printf '[xingzhou_boot] time=%s pid=%s elapsed_s=%s stage=%s status=%s %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$(( $(date +%s) - STARTED_AT ))" \
        "$STAGE" "$1" "${2:-}" >&2
}
trap 'code=$?; boot_log exit "exit_code=$code"' 0
boot_log begin

# The platform extracts the ZIP contents into /src before running this file.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export SAM3_SERVER_NAME=0.0.0.0
export SAM3_SERVER_PORT=8030
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export GRADIO_ANALYTICS_ENABLED=False
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.runtime/cache}"

PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [ -z "$PYTHON_BIN" ]; then
    printf '%s\n' 'Python interpreter not found: provide python3 or python on PATH.' >&2
    exit 127
fi
boot_log ok "python=$PYTHON_BIN cwd=$PWD port=$SAM3_SERVER_PORT"

# Report the base image (Python, OS, GPU/CUDA, preinstalled packages) before the
# offline overlay is added. Diagnostics only: a failure here never blocks startup.
STAGE=environment_report
boot_log begin
if "$PYTHON_BIN" "$PWD/xingzhou_envreport.py" >&2; then
    boot_log ok
else
    boot_log warning "exit_code=$? continuing=true"
fi

# Only application wheels go here; Torch/Torchvision/CUDA stay in the image.
STAGE=dependency_target
boot_log begin
DEPS_DIR="$("$PYTHON_BIN" -c 'import hashlib, pathlib, sys; root = pathlib.Path.cwd(); key = hashlib.sha256((root / "requirements-offline.txt").read_bytes()).hexdigest()[:16]; print(root / ".runtime" / ("python-deps-%d%d-" % sys.version_info[:2] + key))')"
boot_log ok "target=$DEPS_DIR"
STAGE=offline_install
if [ ! -f "$DEPS_DIR/.installed" ]; then
    boot_log begin "cache=miss"
    if "$PYTHON_BIN" -m pip --isolated install --disable-pip-version-check \
        --no-index --find-links "$PWD/wheels" --only-binary=:all: --no-deps --no-compile \
        --target "$DEPS_DIR" --upgrade -r "$PWD/requirements-offline.txt"; then
        boot_log ok "exit_code=0"
    else
        code=$?
        boot_log failed "exit_code=$code"
        exit "$code"
    fi
    STAGE=dependency_marker
    touch "$DEPS_DIR/.installed"
    boot_log ok
else
    boot_log skipped "cache=hit"
fi
export PYTHONPATH="$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Run Python as a child instead of exec so its exit code is logged even when
# it dies without output (e.g. 137 = SIGKILL/OOM, 139 = segmentation fault).
# xingzhou_startup.py execs the app in place, so this child is the server.
STAGE=python
boot_log begin
"$PYTHON_BIN" "$PWD/xingzhou_startup.py" "$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true' TERM INT
set +e
wait "$CHILD"
code=$?
while kill -0 "$CHILD" 2>/dev/null; do
    wait "$CHILD"
    code=$?
done
set -e
if [ "$code" -gt 128 ]; then
    boot_log failed "exit_code=$code signal=$((code - 128))"
elif [ "$code" -ne 0 ]; then
    boot_log failed "exit_code=$code"
else
    boot_log ok "exit_code=0"
fi
exit "$code"
