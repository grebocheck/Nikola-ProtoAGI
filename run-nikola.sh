#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

python -m protoagi telegram "$@"
