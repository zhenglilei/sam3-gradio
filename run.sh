#!/bin/sh
set -eu

# The platform extracts the ZIP contents into /src before running this file.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export SAM3_SERVER_NAME=0.0.0.0
export SAM3_SERVER_PORT=8030
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.runtime/cache}"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_PATH="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
    if [ ! -f "$PYTHON_PATH" ] || [ ! -x "$PYTHON_PATH" ]; then
        printf '%s\n' 'Invalid PYTHON_BIN: set it to an executable Python interpreter path or command name.' >&2
        exit 127
    fi
    PYTHON_BIN="$PYTHON_PATH"
else
    PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    if [ -z "$PYTHON_BIN" ]; then
        printf '%s\n' 'Python interpreter not found: provide python3 or python on PATH, or set PYTHON_BIN.' >&2
        exit 127
    fi
fi
exec "$PYTHON_BIN" "$PWD/xingzhou_startup.py" "$@"
