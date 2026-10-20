#!/bin/sh
set -eu

# The platform extracts the ZIP contents into /src before running this file.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export SAM3_SERVER_NAME=0.0.0.0
export SAM3_SERVER_PORT=8030
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.runtime/cache}"

PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [ -z "$PYTHON_BIN" ]; then
    printf '%s\n' 'Python interpreter not found: provide python3 or python on PATH.' >&2
    exit 127
fi

# Only application wheels go here; Torch/Torchvision/CUDA stay in the image.
DEPS_DIR="$("$PYTHON_BIN" -c 'import hashlib, pathlib, sys; root = pathlib.Path.cwd(); key = hashlib.sha256((root / "requirements-offline.txt").read_bytes()).hexdigest()[:16]; print(root / ".runtime" / ("python-deps-%d%d-" % sys.version_info[:2] + key))')"
if [ ! -f "$DEPS_DIR/.installed" ]; then
    "$PYTHON_BIN" -m pip --isolated install --disable-pip-version-check \
        --no-index --find-links "$PWD/wheels" --only-binary=:all: --no-deps \
        --target "$DEPS_DIR" --upgrade -r "$PWD/requirements-offline.txt"
    touch "$DEPS_DIR/.installed"
fi
export PYTHONPATH="$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$PWD/xingzhou_startup.py" "$@"
