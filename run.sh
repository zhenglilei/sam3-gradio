#!/bin/sh
set -eu

# The platform extracts the ZIP contents into /src before running this file.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export SAM3_SERVER_NAME=0.0.0.0
export SAM3_SERVER_PORT=8030
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.runtime/cache}"
export SAM3_SESSION_DEPLOYMENT_ID="${SAM3_SESSION_DEPLOYMENT_ID:-sam3-xingzhou}"
export SAM3_SESSION_COOKIE_NAME="${SAM3_SESSION_COOKIE_NAME:-sam3_xingzhou_owner}"

: "${SAM3_ALLOWED_ORIGINS:?Set the exact browser-facing service origin, not the platform management URL}"
: "${SAM3_SESSION_COOKIE_SECURE:?Set 1 for HTTPS, or 0 with explicit SAM3_ALLOW_INSECURE_COOKIE=1 for HTTP}"

for file in models/sam3.pt models/big-lama.pt assets/bpe_simple_vocab_16e6.txt.gz; do
    if [ ! -s "$file" ]; then
        printf 'Required file is missing or empty: %s/%s\n' "$PWD" "$file" >&2
        exit 1
    fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'from sam3_demo.session_web import load_session_cookie_settings; load_session_cookie_settings()'
printf 'Starting SAM3 on 0.0.0.0:8030 from %s\n' "$PWD"
exec "$PYTHON_BIN" -m sam3_demo.app
