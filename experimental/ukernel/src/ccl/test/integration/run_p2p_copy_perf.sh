#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./test_perf_p2p_copy}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-6,7}"

GPU_A=$(echo "${GPU_IDS}" | cut -d',' -f1)
GPU_B=$(echo "${GPU_IDS}" | cut -d',' -f2)

EXCHANGER_IP="${EXCHANGER_IP:-127.0.0.1}"
EXCHANGER_PORT="${EXCHANGER_PORT:-6979}"

echo "[p2p copy perf] GPUs=${GPU_IDS} server uses GPU 0 (phys ${GPU_A}), client uses GPU 1 (phys ${GPU_B})"

# Launch server in background
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${BIN}" \
  --role=server \
  --gpu=0 \
  --exchanger-ip="${EXCHANGER_IP}" \
  --exchanger-port="${EXCHANGER_PORT}" \
  --transport=auto &
SERVER_PID=$!

sleep 1

# Launch client
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${BIN}" \
  --role=client \
  --gpu=1 \
  --exchanger-ip="${EXCHANGER_IP}" \
  --exchanger-port="${EXCHANGER_PORT}" \
  --transport=auto &
CLIENT_PID=$!

# Wait for both
wait $SERVER_PID; SERVER_RC=$?
wait $CLIENT_PID; CLIENT_RC=$?

echo "[p2p copy perf] done (server rc=${SERVER_RC}, client rc=${CLIENT_RC})"
exit $((SERVER_RC | CLIENT_RC))
