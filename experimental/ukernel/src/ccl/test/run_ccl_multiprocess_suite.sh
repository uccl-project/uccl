#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./test_multiprocess_collective}"

# Edit these values locally when you want to change the test shape.
NPROC_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
TORCHRUN_MASTER_PORT=29500
EXCHANGER_PORT_BASE=29600
GPU_IDS=0,1,2,3

# Use auto by default so same-node runs exercise the IPC path directly.
TRANSPORT=auto
BYTES_PER_RANK=$((1 * 1024 * 1024))
TILE_BYTES=$((64 * 1024))
NUM_FLOWS=2

run_case() {
  local collective="$1"
  local exchanger_port="$2"

  echo "[ccl suite] torchrun collective=${collective} nproc_per_node=${NPROC_PER_NODE} bytes_per_rank=${BYTES_PER_RANK} tile_bytes=${TILE_BYTES} num_flows=${NUM_FLOWS} transport=${TRANSPORT} torchrun_port=${TORCHRUN_MASTER_PORT} exchanger_port=${exchanger_port}"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    torchrun \
      --no-python \
      --nproc-per-node "${NPROC_PER_NODE}" \
      --nnodes "${NNODES}" \
      --node-rank "${NODE_RANK}" \
      --master-addr "${MASTER_ADDR}" \
      --master-port "${TORCHRUN_MASTER_PORT}" \
      "${BIN}" \
      --collective "${collective}" \
      --transport "${TRANSPORT}" \
      --bytes-per-rank "${BYTES_PER_RANK}" \
      --tile-bytes "${TILE_BYTES}" \
      --num-flows "${NUM_FLOWS}" \
      --exchanger-ip "${MASTER_ADDR}" \
      --exchanger-port "${exchanger_port}"
}

run_case allreduce "${EXCHANGER_PORT_BASE}"
run_case alltoall "$((EXCHANGER_PORT_BASE + 1))"

echo "[ccl suite] all multiprocess collective checks passed"
