#!/bin/sh
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
IMAGE="${SAM3_RUNTIME_IMAGE:-sam3-xingzhou-runtime:py312-cu126}"
docker info >/dev/null
docker build --tag "$IMAGE" "$HERE"
if [ "$#" -gt 0 ]; then
    if [ -e "$1" ]; then
        echo "Refusing to overwrite image archive: $1" >&2
        exit 1
    fi
    docker save --output "$1" "$IMAGE"
fi
printf 'Built image: %s\n' "$IMAGE"
