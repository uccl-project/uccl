#!/bin/bash
# Node 1+ (Secondary) - Multi-node vLLM with Expert Parallel (EP)
# This node runs in headless mode (no API server)
#
# Prerequisites: Same as Node 0
# 1. Install vLLM with EP support
# 2. Install DeepGEMM
# 3. Install EP kernels
# 4. For AWS EFA: Install AWS OFI-NCCL plugin
#
# IMPORTANT: All configuration must match Node 0!

# Example:
# bash launch_vllm_worker.sh 10.4.147.22 13345 deepseek-ai/DeepSeek-V3-0324 allgather_reducescatter 2 1 8 1

set -e

echo "🚀 Launching vLLM Secondary Node (Headless) with Expert Parallel..."

# Hugging Face cache (DPSK v3 weights in ~/efs/yzhou/hf_cache)
export HF_HOME="${HF_HOME:-$HOME/efs/yzhou/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

# Check if primary node IP is provided
if [ -z "$1" ]; then
    echo "❌ Error: Primary Node (Node 0) IP address is required!"
    echo ""
    echo "Usage: $0 <NODE1_IP> [RPC_PORT] [MODEL] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [START_RANK]"
    echo ""
    echo "Example:"
    echo "  $0 10.1.107.86 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8"
    echo ""
    echo "⚠️  Note: Use Node 0's IP address, not this node's IP!"
    echo "💡 To find Node 0's IP, run on Node 0: hostname -I"
    exit 1
fi

# PyTorch library path (required for DeepGEMM)
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))"):$LD_LIBRARY_PATH

export VLLM_USE_DEEP_GEMM=1

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# CRITICAL: Must match Node 0 exactly!
# For InfiniBand/EFA clusters
export GLOO_SOCKET_IFNAME=enp71s0         # Change to your primary network interface
export NCCL_SOCKET_IFNAME=enp71s0       # Uncomment if using NCCL
export TP_SOCKET_IFNAME=enp71s0         # Uncomment if using tensor parallel

# ============================================================================
# NCCL CONFIGURATION (Optional)
# ============================================================================
# CRITICAL: Must match Node 0 exactly!

# AWS EFA NCCL plugin (uncomment if using AWS EFA):
export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/libnccl-net.so"

# NCCL performance tuning (optional):
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608

# NCCL debugging (for diagnosing connection issues):
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

# https://github.com/vllm-project/vllm/pull/27444
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# Set to local non-shared disk like "/opt/dlami/nvme"
export DG_JIT_CACHE_DIR="/opt/dlami/nvme"

# ============================================================================
# ARGUMENTS PARSING
# ============================================================================

NODE1_IP="$1"                                      # Primary node IP (REQUIRED)
RPC_PORT="${2:-13345}"                             # Same RPC port as Node 0
MODEL="${3:-deepseek-ai/DeepSeek-V3-0324}"         # Same model as Node 0
BACKEND="${4:-allgather_reducescatter}"            # Backend to use
TOTAL_DP_SIZE="${5:-16}"                           # Same total DP as Node 0
LOCAL_DP_SIZE="${6:-8}"                            # Local DP on this node
LOCAL_TP_SIZE="${7:-1}"                            # Local TP on this node
START_RANK="${8:-8}"                               # Starting rank offset

# START_RANK calculation:
# - Node 1: LOCAL_DP_SIZE of Node 0 (e.g., 8)
# - Node 2: LOCAL_DP_SIZE of Node 0 + Node 1 (e.g., 16)
# - Node N: Sum of all previous nodes' LOCAL_DP_SIZE

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         vLLM Expert Parallel - Secondary Node Config          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Backend Configuration:"
echo "  • Backend: ${BACKEND}"
echo "  • DeepGEMM: Enabled"
echo ""
echo "Node Configuration:"
echo "  • Role: Secondary (headless worker)"
echo "  • Model: ${MODEL}"
echo "  • Primary Node IP: ${NODE1_IP}"
echo "  • RPC Port: ${RPC_PORT}"
echo ""
echo "Parallelism Configuration:"
echo "  • Total Data Parallel Size: ${TOTAL_DP_SIZE} (across all nodes)"
echo "  • Local Data Parallel Size: ${LOCAL_DP_SIZE} (this node)"
echo "  • Local Tensor Parallel Size: ${LOCAL_TP_SIZE} (this node)"
echo "  • Starting Rank: ${START_RANK}"
echo "  • Expert Parallel: Enabled"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# CONFIRM CUDA GRAPH IS ENABLED
# ============================================================================
# After the server starts, search the log output for:
#
#   ENABLED:  "Graph capturing finished in N secs, took X.XX GiB"
#   DISABLED: "Skipping CUDA graph capture..."
#
# Example (if logging to file): grep -E "Graph capturing finished|Skipping CUDA graph" <your_log>
#
# To force-enable and get the confirmation line, uncomment below:
#   -cc.cudagraph_mode=FULL_AND_PIECEWISE
# To log runtime CUDA graph usage (per interval), uncomment:
#   -observability-config.cudagraph_metrics=true

# ============================================================================
# PROFILING / TIMELINE TRACE
# ============================================================================
# Same as Node 0: VLLM_PROFILER_DIR default below; trigger from head:
# POST /start_profile, run workload, POST /stop_profile.
export VLLM_PROFILER_DIR="${VLLM_PROFILER_DIR:-$HOME/efs/ziming/uccl/ep/bench/vllm}"

# ============================================================================
# LAUNCH vLLM SERVER (HEADLESS MODE)
# ============================================================================

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
    --headless \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "'"${VLLM_PROFILER_DIR}"'"}'
    # -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    # -observability-config.cudagraph_metrics=true
    # -observability-config.enable_layerwise_nvtx_tracing=true

# Additional useful options (uncomment as needed, must match Node 0):
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.9 \
#   --dtype auto \
