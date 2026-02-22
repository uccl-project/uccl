#!/bin/bash
# Node 0 (Primary) - Multi-node vLLM with Expert Parallel (EP)
# This node handles incoming requests
#
# Prerequisites:
# 1. Install vLLM with EP support: https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html#architecture-overview
# 2. Install DeepGEMM: https://github.com/deepseek-ai/DeepGEMM#installation
# 3. Install EP kernels: Follow vLLM's EP installation guide
# 4. For AWS EFA: Install AWS OFI-NCCL plugin

#  Example: 
# bash launch_vllm_head.sh 10.4.147.22 13345 deepseek-ai/DeepSeek-V3-0324 deepep_low_latency 2 1 8 1

set -e

echo "🚀 Launching vLLM Node 0 (Primary) with Expert Parallel..."

# Hugging Face cache (DPSK v3 weights in ~/efs/yzhou/hf_cache)
export HF_HOME="${HF_HOME:-$HOME/efs/yzhou/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

# Check if IP is provided
if [ -z "$1" ]; then
    echo "❌ Error: Node IP address is required!"
    echo ""
    echo "Usage: $0 <NODE1_IP> [RPC_PORT] [MODEL] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [API_SERVERS]"
    echo ""
    echo "Example:"
    echo "  $0 10.1.107.86 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8"
    echo ""
    echo "💡 To find your IP address, run: hostname -I"
    exit 1
fi

# PyTorch library path (required for DeepGEMM)
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))"):$LD_LIBRARY_PATH

export VLLM_USE_DEEP_GEMM=1

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# For InfiniBand/EFA clusters: Prevent initialization hangs
# This ensures torch distributed uses Ethernet for initial setup
# Find your network interface: ip addr show | grep -E 'eth|enp'
export GLOO_SOCKET_IFNAME=enp71s0       # Change to your primary network interface
export NCCL_SOCKET_IFNAME=enp71s0       # Uncomment if using NCCL
export TP_SOCKET_IFNAME=enp71s0         # Uncomment if using tensor parallel

# ============================================================================
# NCCL CONFIGURATION (Optional - for advanced users)
# ============================================================================

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

NODE1_IP="$1"                                      # Node 0 IP address (REQUIRED)
RPC_PORT="${2:-13345}"                             # RPC communication port
MODEL="${3:-deepseek-ai/DeepSeek-V3-0324}"         # Model to serve
BACKEND="${4:-allgather_reducescatter}"            # Backend to use
TOTAL_DP_SIZE="${5:-16}"                           # Total DP size across all nodes
LOCAL_DP_SIZE="${6:-8}"                            # Local DP size on this node
LOCAL_TP_SIZE="${7:-1}"                            # Local TP size on this node
API_SERVERS="${8:-8}"                              # Number of API servers

# Recommendations:
# - TOTAL_DP_SIZE = LOCAL_DP_SIZE * NUMBER_OF_NODES
# - LOCAL_DP_SIZE = Number of GPUs per node (typically 8 for 8xGPU nodes)
# - API_SERVERS = LOCAL_DP_SIZE (one server per local DP process)

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              vLLM Expert Parallel Configuration               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Backend Configuration:"
echo "  • Backend: ${BACKEND}"
echo "  • DeepGEMM: Enabled"
echo ""
echo "Node Configuration:"
echo "  • Role: Primary (handles API requests)"
echo "  • Model: ${MODEL}"
echo "  • Node IP: ${NODE1_IP}"
echo "  • RPC Port: ${RPC_PORT}"
echo ""
echo "Parallelism Configuration:"
echo "  • Total Data Parallel Size: ${TOTAL_DP_SIZE} (across all nodes)"
echo "  • Local Data Parallel Size: ${LOCAL_DP_SIZE} (this node)"
echo "  • Local Tensor Parallel Size: ${LOCAL_TP_SIZE} (this node)"
echo "  • API Servers: ${API_SERVERS}"
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
# Example (after starting): grep -E "Graph capturing finished|Skipping CUDA graph" <your_log>
#
# To force-enable and get the confirmation line, uncomment below:
#   -cc.cudagraph_mode=FULL_AND_PIECEWISE
# To log runtime CUDA graph usage (per interval), uncomment:
#   -observability-config.cudagraph_metrics=true

# ============================================================================
# PROFILING / TIMELINE TRACE
# ============================================================================
# To get a timeline trace:
#
# 1) PyTorch profiler (Chrome trace → open at https://ui.perfetto.dev/):
#    - Traces go to VLLM_PROFILER_DIR (default: this directory). Profiler is ON by default below.
#    - For vllm bench: use ./run_bench_with_profile.sh ... so capture wraps the whole benchmark.
#    - Or manually: curl -X POST http://localhost:8000/start_profile; <workload>; curl -X POST .../stop_profile.
#    - Traces (head + workers) in VLLM_PROFILER_DIR; open .json/.json.gz at https://ui.perfetto.dev/
#
# 2) NVTX layerwise tracing (for Nsight Systems):
#    - Uncomment: -observability-config.enable_layerwise_nvtx_tracing=true
#    - Note: not compatible with CUDA graphs; disable cudagraph or use for nsys only.
#
# 3) Nsight Systems (nsys) for GPU timeline:
#    - Run with: VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile -t cuda,nvtx -o vllm_report -- vllm serve ...
#
# Default: under this repo; override with VLLM_PROFILER_DIR if needed
export VLLM_PROFILER_DIR="${VLLM_PROFILER_DIR:-$HOME/efs/ziming/uccl/ep/bench/vllm}"

# ============================================================================
# LAUNCH vLLM SERVER
# ============================================================================

vllm serve "${MODEL}" \
    --enable-expert-parallel \
    --all2all-backend "${BACKEND}" \
    --tensor-parallel-size "${LOCAL_TP_SIZE}" \
    --data-parallel-size "${TOTAL_DP_SIZE}" \
    --data-parallel-size-local "${LOCAL_DP_SIZE}" \
    --data-parallel-address "${NODE1_IP}" \
    --data-parallel-rpc-port "${RPC_PORT}" \
    --gpu-memory-utilization 0.85 \
    --api-server-count="${API_SERVERS}" \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "'"${VLLM_PROFILER_DIR}"'"}'
    # -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    # -observability-config.cudagraph_metrics=true
    # NVTX layerwise (for nsys; incompatible with CUDA graphs):
    # -observability-config.enable_layerwise_nvtx_tracing=true

# Additional useful options (uncomment as needed):
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.9 \
#   --dtype auto \
#   --enable-chunked-prefill \
#   --port 8000 \

