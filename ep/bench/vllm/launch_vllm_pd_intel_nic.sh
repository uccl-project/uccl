#!/bin/bash
# Multi-node vLLM with Expert Parallel (EP) on Intel irdma NICs
# EP all-to-all tested with UCCL-EP RDMA for inter-node communication.
# PD disaggregation uses NixlConnector with UCCL-P2P backend for KV cache transfer.
# Supports two modes:
#   1. Baseline EP-only: all nodes in a single EP group (no disaggregation)
#   2. PD Disaggregation: separate prefill and decode EP groups with KV transfer
#
# -- Baseline Mode (EP-only, no disaggregation) --
#
# Example (4-EP baseline):
#   Node 0: BASELINE_DP_SIZE=4 $0 baseline-head   <HEAD_IP> <unused>
#   Node 1: BASELINE_DP_SIZE=4 $0 baseline-worker <HEAD_IP> <unused> 1
#   Node 2: BASELINE_DP_SIZE=4 $0 baseline-worker <HEAD_IP> <unused> 2
#   Node 3: BASELINE_DP_SIZE=4 $0 baseline-worker <HEAD_IP> <unused> 3
#
# Example (2-EP baseline):
#   Node 0: BASELINE_DP_SIZE=2 $0 baseline-head   <HEAD_IP> <unused>
#   Node 1: BASELINE_DP_SIZE=2 $0 baseline-worker <HEAD_IP> <unused>
#
# -- PD Disaggregation Mode --
#
# Supports asymmetric prefill/decode configurations:
#   2P+2D (default): 2 prefill nodes + 2 decode nodes
#   2P+1D:           2 prefill nodes + 1 decode node
#   1P+2D:           1 prefill node  + 2 decode nodes
#   1P+1D:           1 prefill node  + 1 decode node (no EP)
#
# KV transfer: NixlConnector with UCCL P2P backend between prefill -> decode
#
# Example (2P+2D -- default, no env vars needed):
#   Node 0: $0 prefill-head   <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 1: $0 prefill-worker <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 2: $0 decode-head    <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 3: $0 decode-worker  <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#
# Example (2P+1D):
#   Node 0: PREFILL_DP_SIZE=2 $0 prefill-head   <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 1: PREFILL_DP_SIZE=2 $0 prefill-worker <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 2: DECODE_DP_SIZE=1  $0 decode-head    <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#
# Example (1P+2D):
#   Node 0: PREFILL_DP_SIZE=1 $0 prefill-head   <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 1: DECODE_DP_SIZE=2  $0 decode-head    <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 2: DECODE_DP_SIZE=2  $0 decode-worker  <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#
# Example (1P+1D):
#   Node 0: PREFILL_DP_SIZE=1 $0 prefill-head   <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#   Node 1: DECODE_DP_SIZE=1  $0 decode-head    <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#
# Proxy (run on any node, sends client requests through prefill -> decode):
#   $0 proxy <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
#
# A disagg proxy is required to route requests: prefill -> KV transfer -> decode.
# Use the "proxy" role to start it (can run on any node).
#
# Usage:
#   $0 <role> <HEAD_IP> <DECODE_HEAD_IP> [START_RANK]
#
# Arguments:
#   $1 = role: baseline-head | baseline-worker | prefill-head | prefill-worker |
#              decode-head | decode-worker | proxy
#   $2 = HEAD_IP (head node IP for EP coordination)
#   $3 = DECODE_HEAD_IP (decode head IP; unused for baseline roles)
#   $4 = START_RANK (default: 1, only for *-worker roles)
#
# Environment variables:
#   MODEL            -- HuggingFace model name (default: Qwen/Qwen3-30B-A3B-FP8)
#   BACKEND          -- EP all-to-all backend: deepep_low_latency | allgather_reducescatter
#                       (default: deepep_low_latency)
#   BASELINE_DP_SIZE -- number of nodes for baseline EP (default: 2)
#   PREFILL_DP_SIZE  -- number of prefill nodes (default: 2)
#   DECODE_DP_SIZE   -- number of decode  nodes (default: 2)
#   NIXL_BACKEND     -- NixlConnector backend: UCCL | UCX | Mooncake (default: UCCL)
#
# -- Role Descriptions --
#
# Baseline mode (EP-only, no disaggregation):
#   baseline-head   -- Rank-0 node that starts the vLLM API server and coordinates
#                     the EP group. All nodes serve both prefill and decode.
#   baseline-worker -- Additional EP node(s) that join the baseline-head's group.
#                     Requires START_RANK ($4) to set each worker's rank (1, 2, ...).
#
# PD disaggregation mode (separate prefill and decode EP groups):
#   prefill-head    -- Rank-0 of the prefill EP group. Runs the prefill API server
#                     (port 8100). Handles prompt encoding and produces KV cache,
#                     which is transferred to decode via RDMA (NixlConnector/UCCL).
#   prefill-worker  -- Additional EP node(s) in the prefill group (rank 1, 2, ...).
#                     Participates in expert-parallel all-to-all during prefill.
#   decode-head     -- Rank-0 of the decode EP group. Runs the decode API server
#                     (port 8000). Receives KV cache from prefill via RDMA and
#                     generates output tokens autoregressively.
#   decode-worker   -- Additional EP node(s) in the decode group (rank 1, 2, ...).
#                     Participates in expert-parallel all-to-all during decode.
#   proxy           -- Lightweight FastAPI proxy (port 9000) that routes client
#                     requests through the prefill->decode pipeline. Sends the
#                     prompt to prefill, extracts KV transfer metadata, then
#                     forwards the full request to decode for generation.

set -e

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <role> <PREFILL_HEAD_IP> <DECODE_HEAD_IP> [START_RANK]"
    echo ""
    echo "Roles:"
    echo "  baseline-head   -- EP-only head (API server, no disaggregation)"
    echo "  baseline-worker -- EP-only worker (needs BASELINE_DP_SIZE>=2)"
    echo "  prefill-head    -- prefill EP head (API server)"
    echo "  prefill-worker  -- prefill EP worker (needs PREFILL_DP_SIZE>=2)"
    echo "  decode-head     -- decode EP head (API server)"
    echo "  decode-worker   -- decode EP worker (needs DECODE_DP_SIZE>=2)"
    echo "  proxy           -- disagg proxy (routes requests: prefill -> decode)"
    echo ""
    echo "Environment variables:"
    echo "  MODEL            -- HuggingFace model (default: Qwen/Qwen3-30B-A3B-FP8)"
    echo "  BACKEND          -- EP all-to-all backend: deepep_low_latency|allgather_reducescatter (default: deepep_low_latency)"
    echo "  BASELINE_DP_SIZE -- number of nodes for baseline EP (default: 2)"
    echo "  PREFILL_DP_SIZE  -- number of prefill nodes (default: 2)"
    echo "  DECODE_DP_SIZE   -- number of decode  nodes (default: 2)"
    echo "  NIXL_BACKEND     -- NixlConnector backend: UCCL|UCX|Mooncake (default: UCCL)"
    echo ""
    echo "Examples:"
    echo "  4-EP baseline: BASELINE_DP_SIZE=4 $0 baseline-head 10.173.44.108 _"
    echo "  2P+2D: $0 prefill-head 10.173.44.108 10.173.44.104  (4 nodes)"
    echo "  2P+1D: DECODE_DP_SIZE=1 $0 decode-head 10.173.44.108 10.173.44.104"
    echo "  1P+2D: PREFILL_DP_SIZE=1 $0 prefill-head 10.173.44.108 10.173.44.104"
    exit 1
fi

ROLE="$1"
PREFILL_HEAD_IP="$2"
DECODE_HEAD_IP="$3"
START_RANK="${4:-1}"
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-FP8}

# EP configuration
BACKEND=${BACKEND:-deepep_low_latency}  # EP all-to-all backend
BASELINE_DP_SIZE=${BASELINE_DP_SIZE:-2}
PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-2}
DECODE_DP_SIZE=${DECODE_DP_SIZE:-2}
BASELINE_API_PORT=8100  # Baseline API server
LOCAL_DP_SIZE=1
LOCAL_TP_SIZE=1

# KV transfer backend (PD disaggregation only)
NIXL_BACKEND=${NIXL_BACKEND:-UCCL}  # NixlConnector backend: UCCL | UCX | Mooncake

# Ports
EP_RPC_PORT=13345       # vLLM DP/EP coordination
PREFILL_API_PORT=8100   # Prefill API server
DECODE_API_PORT=8000    # Decode API server (internal, used by proxy)
PROXY_PORT=9000         # Disagg proxy (user-facing)

# ============================================================================
# ENVIRONMENT
# ============================================================================

export VLLM_USE_DEEP_GEMM=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# Intel RDMA (irdma) configuration
export NCCL_IB_HCA="irdma-mkp0:1"
export UCCL_IB_HCA="irdma-mkp0:1"
export UCCL_P2P_RDMA_DEV="irdma-mkp0"

# Network interface for Intel NICs
export GLOO_SOCKET_IFNAME=eno0
export NCCL_SOCKET_IFNAME=eno0
export UCCL_SOCKET_IFNAME=eno0
export TP_SOCKET_IFNAME=eno0

#export NCCL_DEBUG=INFO

# Tell UCCL-EP how many GPUs are local (per node)
GPUS_PER_NODE=$(( LOCAL_DP_SIZE * LOCAL_TP_SIZE ))
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# NIXL side channel: each engine announces its RDMA metadata on this address.
# Must be set to a routable IP so remote nodes can reach this engine.
# We resolve it from the network interface used by GLOO/NCCL.
MY_IP=$(ip -4 addr show "${GLOO_SOCKET_IFNAME}" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
export VLLM_NIXL_SIDE_CHANNEL_HOST="${MY_IP}"

# ============================================================================
# KV TRANSFER CONFIG
# ============================================================================

# NixlConnector: RDMA-based KV transfer using NIXL.
# Backend: UCCL (UCCL-P2P RDMA, default), UCX, or Mooncake.
# Uses engine_id-based discovery -- each engine registers and discovers
# peers automatically. Works correctly with multi-node EP.
PREFILL_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["'"${NIXL_BACKEND}"'"]}}'
DECODE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["'"${NIXL_BACKEND}"'"]}}'

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

case "$ROLE" in
  baseline-head)
    DP_SIZE=$BASELINE_DP_SIZE
    ROLE_DESC="Baseline Head (EP rank 0, DP=${DP_SIZE}, API on :${BASELINE_API_PORT}, no disagg)"
    EP_ADDR="$PREFILL_HEAD_IP"
    API_PORT="$BASELINE_API_PORT"
    ;;
  baseline-worker)
    DP_SIZE=$BASELINE_DP_SIZE
    if [ "$DP_SIZE" -lt 2 ]; then
      echo "Error: baseline-worker role requires BASELINE_DP_SIZE>=2 (got ${DP_SIZE})"
      exit 1
    fi
    ROLE_DESC="Baseline Worker (EP rank ${START_RANK}, DP=${DP_SIZE}, no disagg)"
    EP_ADDR="$PREFILL_HEAD_IP"
    ;;
  prefill-head)
    DP_SIZE=$PREFILL_DP_SIZE
    ROLE_DESC="Prefill Head (EP rank 0, DP=${DP_SIZE}, API on :${PREFILL_API_PORT})"
    EP_ADDR="$PREFILL_HEAD_IP"
    KV_CONFIG="$PREFILL_KV_CONFIG"
    API_PORT="$PREFILL_API_PORT"
    ;;
  prefill-worker)
    DP_SIZE=$PREFILL_DP_SIZE
    if [ "$DP_SIZE" -lt 2 ]; then
      echo "Error: prefill-worker role requires PREFILL_DP_SIZE>=2 (got ${DP_SIZE})"
      exit 1
    fi
    ROLE_DESC="Prefill Worker (EP rank ${START_RANK}, DP=${DP_SIZE})"
    EP_ADDR="$PREFILL_HEAD_IP"
    KV_CONFIG="$PREFILL_KV_CONFIG"
    ;;
  decode-head)
    DP_SIZE=$DECODE_DP_SIZE
    ROLE_DESC="Decode Head (EP rank 0, DP=${DP_SIZE}, API on :${DECODE_API_PORT})"
    EP_ADDR="$DECODE_HEAD_IP"
    KV_CONFIG="$DECODE_KV_CONFIG"
    API_PORT="$DECODE_API_PORT"
    ;;
  decode-worker)
    DP_SIZE=$DECODE_DP_SIZE
    if [ "$DP_SIZE" -lt 2 ]; then
      echo "Error: decode-worker role requires DECODE_DP_SIZE>=2 (got ${DP_SIZE})"
      exit 1
    fi
    ROLE_DESC="Decode Worker (EP rank ${START_RANK}, DP=${DP_SIZE})"
    EP_ADDR="$DECODE_HEAD_IP"
    KV_CONFIG="$DECODE_KV_CONFIG"
    ;;
  proxy)
    ROLE_DESC="Disagg Proxy (prefill->decode, API on :${PROXY_PORT})"
    ;;
  *)
    echo "Error: role must be baseline-head|baseline-worker|prefill-head|prefill-worker|decode-head|decode-worker|proxy, got: $ROLE"
    exit 1
    ;;
esac

echo ""
echo "+===============================================================+"
echo "|    vLLM EP + PD Disaggregation - Intel irdma NIC Config       |"
echo "+===============================================================+"
echo ""
echo "  * Role: ${ROLE_DESC}"
echo "  * Model: ${MODEL}"
echo "  * Backend: ${BACKEND}"
if [[ "$ROLE" == baseline-* ]]; then
echo "  * Head IP: ${PREFILL_HEAD_IP}"
echo "  * Baseline DP: ${BASELINE_DP_SIZE}"
echo "  * This instance DP: ${DP_SIZE} (local_dp=${LOCAL_DP_SIZE}, TP=${LOCAL_TP_SIZE})"
echo "  * IB HCA: ${NCCL_IB_HCA}"
elif [ "$ROLE" = "proxy" ]; then
echo "  * Prefill URL: http://${PREFILL_HEAD_IP}:${PREFILL_API_PORT}"
echo "  * Decode URL:  http://${DECODE_HEAD_IP}:${DECODE_API_PORT}"
echo "  * Proxy Port:  ${PROXY_PORT}"
else
echo "  * Prefill Head IP: ${PREFILL_HEAD_IP}"
echo "  * Decode Head IP: ${DECODE_HEAD_IP}"
echo "  * Prefill DP: ${PREFILL_DP_SIZE}, Decode DP: ${DECODE_DP_SIZE}"
echo "  * This instance DP: ${DP_SIZE} (local_dp=${LOCAL_DP_SIZE}, TP=${LOCAL_TP_SIZE})"
echo "  * IB HCA: ${NCCL_IB_HCA}"
echo "  * NIXL Side Channel: ${MY_IP}:${VLLM_NIXL_SIDE_CHANNEL_PORT:-5600}"
echo "  * NIXL Backend: ${NIXL_BACKEND}"
echo "  * KV Config: ${KV_CONFIG}"
fi
echo ""
echo "==============================================================="
echo ""

# ============================================================================
# LAUNCH vLLM (EP + PD disaggregation)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common EP args shared by all vLLM roles (head and worker)
EP_ARGS=(
  --enable-expert-parallel
  --all2all-backend "${BACKEND}"
  --tensor-parallel-size "${LOCAL_TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --data-parallel-size-local "${LOCAL_DP_SIZE}"
  --data-parallel-address "${EP_ADDR}"
  --data-parallel-rpc-port "${EP_RPC_PORT}"
  --gpu-memory-utilization 0.85
)

# Head roles: serve API on a port
# Worker roles: headless with explicit start rank
case "$ROLE" in
  baseline-head|prefill-head|decode-head)
    EP_ARGS+=(--port "${API_PORT}") ;;
  baseline-worker|prefill-worker|decode-worker)
    EP_ARGS+=(--data-parallel-start-rank "${START_RANK}" --headless) ;;
esac

# PD disaggregation roles: add KV transfer config
case "$ROLE" in
  prefill-head|prefill-worker|decode-head|decode-worker)
    EP_ARGS+=(--kv-transfer-config "${KV_CONFIG}") ;;
esac

case "$ROLE" in
  proxy)
    PROXY_ARGS=(
      --prefill-url "http://${PREFILL_HEAD_IP}:${PREFILL_API_PORT}"
      --decode-url "http://${DECODE_HEAD_IP}:${DECODE_API_PORT}"
      --port "${PROXY_PORT}"
    )
    if [ "${VERBOSE:-0}" = "1" ]; then
      PROXY_ARGS+=(--verbose)
    fi
    python3 "${SCRIPT_DIR}/disagg_proxy.py" "${PROXY_ARGS[@]}"
    ;;

  *)
    vllm serve "${MODEL}" "${EP_ARGS[@]}"
    ;;
esac
