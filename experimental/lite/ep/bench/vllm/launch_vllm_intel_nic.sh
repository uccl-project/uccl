#!/bin/bash
# Multi-node vLLM with Expert Parallel (EP) on Intel irdma NICs
# Usage: launch_vllm_intel_nic.sh <head|worker> <NODE1_IP> [RPC_PORT] [MODEL] [BACKEND] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [LOCAL_TP_SIZE] [API_SERVERS_or_START_RANK]

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
    echo "Defaults: RPC_PORT=13345, MODEL=Qwen/Qwen3-30B-A3B-FP8, BACKEND=deepep_low_latency,"
    echo "         TOTAL_DP_SIZE=2, LOCAL_DP_SIZE=1, LOCAL_TP_SIZE=1, 9th=1."
    echo ""
    echo "Example (Node 0): $0 head 10.173.44.108 13345 Qwen/Qwen3-30B-A3B-FP8 deepep_low_latency 2 1 1 1"
    echo "Example (Node 1): $0 worker 10.173.44.108 13345 Qwen/Qwen3-30B-A3B-FP8 deepep_low_latency 2 1 1 1"
    echo ""
    echo "NODE1_IP = IP of Node 0 (primary). Find on Node 0 with: hostname -I"
    exit 1
fi

ROLE="$1"
NODE1_IP="$2"
RPC_PORT="${3:-13345}"
MODEL="${4:-Qwen/Qwen3-30B-A3B-FP8}"
BACKEND="${5:-deepep_low_latency}"
TOTAL_DP_SIZE="${6:-2}"
LOCAL_DP_SIZE="${7:-1}"
LOCAL_TP_SIZE="${8:-1}"
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
# ENVIRONMENT (Intel irdma NICs)
# ============================================================================

export VLLM_USE_DEEP_GEMM=1

# Intel RDMA (irdma) configuration
export NCCL_IB_HCA="irdma-mkp0:1"
export UCCL_IB_HCA="irdma-mkp0:1"
export NCCL_IB_GID_INDEX=1
export UCCL_IB_GID_INDEX=1

# Network interface for Intel NICs
export GLOO_SOCKET_IFNAME=eno0
export NCCL_SOCKET_IFNAME=eno0
export UCCL_SOCKET_IFNAME=eno0
export TP_SOCKET_IFNAME=eno0

export NCCL_DEBUG=INFO

# Tell UCCL-EP how many GPUs are local (per node).
# Without this, it defaults to NUM_MAX_NVL_PEERS=8 and tries to open
# CUDA IPC handles across nodes, which fails with "invalid resource handle".
GPUS_PER_NODE=$(( LOCAL_DP_SIZE * LOCAL_TP_SIZE ))
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
# Make all local GPUs visible (0,1,...,n-1)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

if [ "$ROLE" = "head" ]; then
  echo "Launching vLLM Node 0 (Primary) with Expert Parallel on Intel irdma NICs..."
  ROLE_DESC="Primary (handles API requests)"
else
  echo "Launching vLLM Secondary Node (Headless) with Expert Parallel on Intel irdma NICs..."
  ROLE_DESC="Secondary (headless worker)"
fi

echo ""
echo "+===============================================================+"
echo "|        vLLM Expert Parallel - Intel irdma NIC Config          |"
echo "+===============================================================+"
echo ""
echo "  * Role: ${ROLE_DESC}"
echo "  * Model: ${MODEL}"
echo "  * Node 0 IP: ${NODE1_IP}"
echo "  * RPC Port: ${RPC_PORT}"
echo "  * Backend: ${BACKEND}"
echo "  * Total DP: ${TOTAL_DP_SIZE}  Local DP: ${LOCAL_DP_SIZE}  Local TP: ${LOCAL_TP_SIZE}"
echo "  * IB HCA: ${NCCL_IB_HCA}"
echo "  * Socket IF: ${NCCL_SOCKET_IFNAME}"
if [ "$ROLE" = "head" ]; then
  echo "  * API Servers: ${API_SERVERS}"
else
  echo "  * Start Rank: ${START_RANK}"
fi
echo ""
echo "==============================================================="
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
