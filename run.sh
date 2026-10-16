#!/bin/sh
set -eu

# The platform extracts the ZIP contents into /src before running this file.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export SAM3_SERVER_NAME=0.0.0.0
export SAM3_SERVER_PORT=8030
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.runtime/cache}"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" "$PWD/xingzhou_startup.py" "$@"
