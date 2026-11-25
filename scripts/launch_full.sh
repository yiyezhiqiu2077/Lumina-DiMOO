#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PHASE="$1"; shift
PYTHON_BIN="${TRAIN_PYTHON:-python3}"
# Explicit GPU_IDS overrides CUDA_VISIBLE_DEVICES, as in LiveMGT.
if [[ -n "${GPU_IDS+x}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_IDS"; fi
VISIBLE_GPU_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
NUM_PROCESSES="${NUM_PROCESSES:-$VISIBLE_GPU_COUNT}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
 echo "Invalid NUM_PROCESSES=$NUM_PROCESSES; visible GPUs=$VISIBLE_GPU_COUNT" >&2; exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(--config "${TRAIN_CONFIG:-configs/formal/$PHASE.json}" --phase "$PHASE")
# Existing LoRA entry stays explicit; never used as full-mode fallback.
if [[ "${TRAIN_MODE:-full}" == lora ]]; then
 if (( NUM_PROCESSES != 1 )) || [[ "${NNODES:-1}" != 1 ]]; then echo "Legacy LoRA supports one process only; explicitly set NUM_PROCESSES=1 and NNODES=1" >&2; exit 1; fi
 exec "$PYTHON_BIN" -m livemgt_training.train --config "${TRAIN_CONFIG:-configs/lora/$PHASE.json}" "$@" --phase "$PHASE"
fi
if [[ "${TRAIN_MODE:-full}" != full ]]; then echo 'TRAIN_MODE must be full or lora' >&2; exit 1; fi
if [[ -n "${LOCAL_RANK+x}" && -n "${RANK+x}" && -n "${WORLD_SIZE+x}" ]]; then
 # Already launched by torchrun / scheduler: never spawn nested workers.
 exec "$PYTHON_BIN" -m livemgt_training.full.train "${ARGS[@]}" "$@"
fi
NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
EXPECTED_WORLD_SIZE=$(( NNODES * NUM_PROCESSES ))
if [[ -n "${WORLD_SIZE+x}" && "$WORLD_SIZE" != "$EXPECTED_WORLD_SIZE" ]]; then
 echo "WORLD_SIZE=$WORLD_SIZE disagrees with NNODES * NUM_PROCESSES=$EXPECTED_WORLD_SIZE" >&2; exit 1
fi
printf 'Full training: %s nodes x %s workers, CUDA_VISIBLE_DEVICES=%s\n' "$NNODES" "$NUM_PROCESSES" "${CUDA_VISIBLE_DEVICES:-<unset>}"
if [[ "${TRAIN_DRY_RUN:-0}" == 1 ]]; then
 printf 'python=%s phase=%s nodes=%s per_node=%s world=%s visible=%s CVD=%s backend/config supplied explicitly\n' "$PYTHON_BIN" "$PHASE" "$NNODES" "$NUM_PROCESSES" "$EXPECTED_WORLD_SIZE" "$VISIBLE_GPU_COUNT" "${CUDA_VISIBLE_DEVICES:-<unset>}"
 exit 0
fi
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
 --nproc_per_node="$NUM_PROCESSES" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
 --master_port="${MASTER_PORT:-25012}" -m livemgt_training.full.train "${ARGS[@]}" "$@"
