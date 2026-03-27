#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./test_transport_backend}"

NPROC_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
TORCHRUN_MASTER_PORT=29750
EXCHANGER_PORT=29760
GPU_IDS=0,1
TRANSPORT=auto
BYTES=$((64 * 1024))
UHM_HOST_ID_OVERRIDE=${UHM_HOST_ID_OVERRIDE:-ccl-transport-$(date +%s)-$$}

cleanup_ipc_shm() {
  local host_id_override="$1"
  rm -f /dev/shm/uk_t_oob_"${host_id_override}"_l*
}

echo "[transport backend suite] torchrun nproc_per_node=${NPROC_PER_NODE} bytes=${BYTES} transport=${TRANSPORT} torchrun_port=${TORCHRUN_MASTER_PORT} exchanger_port=${EXCHANGER_PORT}"

cleanup_ipc_shm "${UHM_HOST_ID_OVERRIDE}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  UHM_HOST_ID_OVERRIDE="${UHM_HOST_ID_OVERRIDE}" \
  torchrun \
    --no-python \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${TORCHRUN_MASTER_PORT}" \
    "${BIN}" \
    --transport "${TRANSPORT}" \
    --bytes "${BYTES}" \
    --exchanger-ip "${MASTER_ADDR}" \
    --exchanger-port "${EXCHANGER_PORT}"

cleanup_ipc_shm "${UHM_HOST_ID_OVERRIDE}"

echo "[transport backend suite] transport backend checks passed"
