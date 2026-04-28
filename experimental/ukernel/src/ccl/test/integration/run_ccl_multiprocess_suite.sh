#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./test_multiprocess_collective}"

# Minimal functional config for local correctness checks.
NPROC_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
TORCHRUN_MASTER_PORT=29500
EXCHANGER_PORT_BASE=29600
GPU_IDS=0,6,7

TRANSPORT=auto
TILE_BYTES=$((64 * 1024))
# 15 * 64KB = 983040, divisible by tile_bytes and by (3 * sizeof(float)=12).
BYTES_PER_RANK=$((15 * TILE_BYTES))
NUM_FLOWS=2
UHM_HOST_ID_OVERRIDE=${UHM_HOST_ID_OVERRIDE:-ccl-$(date +%s)-$$}

cleanup_ipc_shm() {
  local host_id_override="$1"
  rm -f /dev/shm/uk_t_oob_"${host_id_override}"_l*
}

run_case() {
  local collective="$1"
  local exchanger_port="$2"
  local host_id_override="${UHM_HOST_ID_OVERRIDE}-${collective}"
  local oob_namespace="ccl-oob-${host_id_override}-${exchanger_port}"

  cleanup() {
    cleanup_ipc_shm "${host_id_override}" || true
  }
  trap cleanup EXIT

  echo "[ccl suite] torchrun collective=${collective} nproc_per_node=${NPROC_PER_NODE} bytes_per_rank=${BYTES_PER_RANK} tile_bytes=${TILE_BYTES} num_flows=${NUM_FLOWS} transport=${TRANSPORT} torchrun_port=${TORCHRUN_MASTER_PORT} exchanger_port=${exchanger_port}"

  cleanup_ipc_shm "${host_id_override}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    UHM_HOST_ID_OVERRIDE="${host_id_override}" \
    UHM_OOB_NAMESPACE="${oob_namespace}" \
    UHM_OOB_CLEAN_START=0 \
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

  trap - EXIT
  cleanup
}

run_case allreduce "${EXCHANGER_PORT_BASE}"
run_case alltoall "$((EXCHANGER_PORT_BASE + 1))"

echo "[ccl suite] all multiprocess collective checks passed"
