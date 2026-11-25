#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${GPU_IDS+x}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_IDS"; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${INFER_PYTHON:-${TRAIN_PYTHON:-python3}}" -m inference.edit "$@"
