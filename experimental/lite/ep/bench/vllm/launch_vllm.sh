#!/bin/bash
# Multi-node vLLM with Expert Parallel (EP)
# Usage: launch_vllm.sh <head|worker> <NODE1_IP> [RPC_PORT] [MODEL] [BACKEND] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [LOCAL_TP_SIZE] [API_SERVERS_or_START_RANK]

set -e

# ============================================================================
# ROLE AND USAGE
# ============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <head|worker> <NODE1_IP> [RPC_PORT] [MODEL] [BACKEND] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [LOCAL_TP_SIZE] [API_SERVERS_or_START_RANK]"
    echo ""
    echo "  head   - Primary node (API servers). 9th arg = API_SERVERS (default: 1)."
    echo "  worker - Secondary node (headless). 9th arg = START_RANK (node 1: 1; node 2: 2; ...). Default: 1."
    echo ""
    echo "Defaults: RPC_PORT=13345, MODEL=deepseek-ai/DeepSeek-V3-0324, BACKEND=allgather_reducescatter,"
    echo "         TOTAL_DP_SIZE=2, LOCAL_DP_SIZE=1, LOCAL_TP_SIZE=8, 9th=1."
    echo ""
    echo "Example (Node 0): $0 head 172.31.41.55 13345 deepseek-ai/DeepSeek-V3-0324 deepep_high_throughput 2 1 8 1"
    echo "Example (Node 1): $0 worker 172.31.41.55 13345 deepseek-ai/DeepSeek-V3-0324 deepep_high_throughput 2 1 8 1"
    echo ""
    echo "NODE1_IP = IP of Node 0 (primary). Find on Node 0 with: hostname -I"
    exit 1
fi

ROLE="$1"
NODE1_IP="$2"
RPC_PORT="${3:-13345}"
MODEL="${4:-deepseek-ai/DeepSeek-V3-0324}"
BACKEND="${5:-allgather_reducescatter}"
TOTAL_DP_SIZE="${6:-2}"
LOCAL_DP_SIZE="${7:-1}"
LOCAL_TP_SIZE="${8:-8}"
NINTH="${9:-1}"

case "$ROLE" in
  head)
    API_SERVERS="$NINTH"
    ;;
  worker)
    START_RANK="$NINTH"
    ;;
  *)
    echo "Error: role must be 'head' or 'worker', got: $ROLE"
    exit 1
    ;;
esac

# ============================================================================
# ENVIRONMENT (shared)
# ============================================================================

export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))"):$LD_LIBRARY_PATH
export VLLM_USE_DEEP_GEMM=1

export GLOO_SOCKET_IFNAME=enp71s0
export NCCL_SOCKET_IFNAME=enp71s0
export TP_SOCKET_IFNAME=enp71s0

# AWS EFA: prefer lib path, fallback to x86_64-linux-gnu (older AMIs); exit if neither exists
if [ -f "/opt/amazon/ofi-nccl/lib/libnccl-net.so" ]; then
  export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/libnccl-net.so"
elif [ -f "/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so" ]; then
  export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"
else
  echo "Error: NCCL net plugin not found. Expected /opt/amazon/ofi-nccl/lib/libnccl-net.so or .../x86_64-linux-gnu/libnccl-net.so"
  exit 1
fi
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_DEBUG=INFO

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export DG_JIT_CACHE_DIR="/opt/dlami/nvme"

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

if [ "$ROLE" = "head" ]; then
  echo "🚀 Launching vLLM Node 0 (Primary) with Expert Parallel..."
  ROLE_DESC="Primary (handles API requests)"
else
  echo "🚀 Launching vLLM Secondary Node (Headless) with Expert Parallel..."
  ROLE_DESC="Secondary (headless worker)"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              vLLM Expert Parallel Configuration               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  • Role: ${ROLE_DESC}"
echo "  • Model: ${MODEL}"
echo "  • Node 0 IP: ${NODE1_IP}"
echo "  • RPC Port: ${RPC_PORT}"
echo "  • Backend: ${BACKEND}"
echo "  • Total DP: ${TOTAL_DP_SIZE}  Local DP: ${LOCAL_DP_SIZE}  Local TP: ${LOCAL_TP_SIZE}"
if [ "$ROLE" = "head" ]; then
  echo "  • API Servers: ${API_SERVERS}"
else
  echo "  • Start Rank: ${START_RANK}"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# LAUNCH vLLM
# ============================================================================

if [ "$ROLE" = "head" ]; then
  vllm serve "${MODEL}" \
    --enable-expert-parallel \
    --all2all-backend "${BACKEND}" \
    --tensor-parallel-size "${LOCAL_TP_SIZE}" \
    --data-parallel-size "${TOTAL_DP_SIZE}" \
    --data-parallel-size-local "${LOCAL_DP_SIZE}" \
    --data-parallel-address "${NODE1_IP}" \
    --data-parallel-rpc-port "${RPC_PORT}" \
    --gpu-memory-utilization 0.85 \
    --api-server-count="${API_SERVERS}"
else
  vllm serve "${MODEL}" \
    --enable-expert-parallel \
    --all2all-backend "${BACKEND}" \
    --tensor-parallel-size "${LOCAL_TP_SIZE}" \
    --data-parallel-size "${TOTAL_DP_SIZE}" \
    --data-parallel-size-local "${LOCAL_DP_SIZE}" \
    --data-parallel-start-rank "${START_RANK}" \
    --data-parallel-address "${NODE1_IP}" \
    --data-parallel-rpc-port "${RPC_PORT}" \
    --gpu-memory-utilization 0.85 \
    --headless
fi
