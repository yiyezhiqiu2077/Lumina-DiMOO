#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${DIMO_CONFIG:-}" ]]; then export TRAIN_CONFIG="${TRAIN_CONFIG:-$DIMO_CONFIG}"; fi
if [[ -n "${DIMO_PYTHON:-}" ]]; then export TRAIN_PYTHON="${TRAIN_PYTHON:-$DIMO_PYTHON}"; fi
exec bash "$ROOT/scripts/launch_full.sh" dimo "$@"
