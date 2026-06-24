#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PY="${SCRIPT_DIR}/test_transport_paths.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29790}"
EXCHANGER_PORT_BASE="${EXCHANGER_PORT_BASE:-29800}"
GPU_IDS="${GPU_IDS:-6,7}"

TRANSPORTS=("${@}")
if [[ ${#TRANSPORTS[@]} -eq 0 ]]; then
  TRANSPORTS=("ipc" "tcp" "uccl")
fi

for i in "${!TRANSPORTS[@]}"; do
  transport="${TRANSPORTS[$i]}"
  exchanger_port="$((EXCHANGER_PORT_BASE + i))"
  echo "[transport paths suite] transport=${transport} exchanger_port=${exchanger_port}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  TRANSPORT="${transport}" \
  EXCHANGER_PORT="${exchanger_port}" \
  torchrun \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --nnodes 1 \
    --node-rank 0 \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    "${TEST_PY}"
done

echo "[transport paths suite] all selected transports passed"
