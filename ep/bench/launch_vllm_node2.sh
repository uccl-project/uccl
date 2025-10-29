#!/bin/bash
# Node 2+ (Secondary) - Multi-node vLLM with Expert Parallel (EP)
# This node runs in headless mode (no API server)
#
# Prerequisites: Same as Node 1
# 1. Install vLLM with EP support
# 2. Install DeepGEMM
# 3. Install EP kernels
# 4. For AWS EFA: Install AWS OFI-NCCL plugin
#
# IMPORTANT: All configuration must match Node 1!

set -e

echo "🚀 Launching vLLM Secondary Node (Headless) with Expert Parallel..."

# Check if primary node IP is provided
if [ -z "$1" ]; then
    echo "❌ Error: Primary Node (Node 1) IP address is required!"
    echo ""
    echo "Usage: $0 <NODE1_IP> [RPC_PORT] [MODEL] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [START_RANK]"
    echo ""
    echo "Example:"
    echo "  $0 10.1.107.86 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8"
    echo ""
    echo "⚠️  Note: Use Node 1's IP address, not this node's IP!"
    echo "💡 To find Node 1's IP, run on Node 1: hostname -I"
    exit 1
fi

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# IMPORTANT: These must match Node 1's configuration exactly!
# ============================================================================

# Python path configuration (adjust to your installation)
# Example paths - modify according to your setup:
export PYTHONPATH=/path/to/vllm:$PYTHONPATH
export PYTHONPATH=/path/to/DeepGEMM:$PYTHONPATH
export PYTHONPATH=/path/to/DeepEP:$PYTHONPATH
export PYTHONPATH=/path/to/pplx-kernels:$PYTHONPATH

# PyTorch library path (required for DeepGEMM)
export LD_LIBRARY_PATH=/path/to/python/site-packages/torch/lib:$LD_LIBRARY_PATH

# Example for conda/pip installation:
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))"):$LD_LIBRARY_PATH

# ============================================================================
# BACKEND CONFIGURATION
# ============================================================================
# CRITICAL: Must match Node 1 exactly!

export VLLM_ALL2ALL_BACKEND=allgather_reducescatter
export VLLM_USE_DEEP_GEMM=1

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================
# CRITICAL: Must match Node 1 exactly!

# For InfiniBand/EFA clusters
export GLOO_SOCKET_IFNAME=eth0         # Change to your primary network interface
export NCCL_SOCKET_IFNAME=eth0       # Uncomment if using NCCL
export TP_SOCKET_IFNAME=eth0         # Uncomment if using tensor parallel

# ============================================================================
# NCCL CONFIGURATION (Optional)
# ============================================================================
# CRITICAL: Must match Node 1 exactly!

# AWS EFA NCCL plugin (uncomment if using AWS EFA):
export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

# NCCL performance tuning (optional):
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608

# NCCL debugging (uncomment to diagnose connection issues):
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# ============================================================================
# ARGUMENTS PARSING
# ============================================================================

NODE1_IP="$1"                                      # Primary node IP (REQUIRED)
RPC_PORT="${2:-13345}"                             # Same RPC port as Node 1
MODEL="${3:-deepseek-ai/DeepSeek-V3-0324}"         # Same model as Node 1
TOTAL_DP_SIZE="${4:-16}"                           # Same total DP as Node 1
LOCAL_DP_SIZE="${5:-8}"                            # Local DP on this node
START_RANK="${6:-8}"                               # Starting rank offset

# START_RANK calculation:
# - Node 2: LOCAL_DP_SIZE of Node 1 (e.g., 8)
# - Node 3: LOCAL_DP_SIZE of Node 1 + Node 2 (e.g., 16)
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
echo "  • Backend: ${VLLM_ALL2ALL_BACKEND}"
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
echo "  • Starting Rank: ${START_RANK}"
echo "  • Expert Parallel: Enabled (automatically calculated)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# LAUNCH vLLM SERVER (HEADLESS MODE)
# ============================================================================

vllm serve "${MODEL}" \
    --tensor-parallel-size 1 \                          # TP size (must match Node 1)
    --enable-expert-parallel \                          # Enable Expert Parallel
    --data-parallel-size "${TOTAL_DP_SIZE}" \           # Total DP (must match Node 1)
    --data-parallel-size-local "${LOCAL_DP_SIZE}" \     # Local DP on this node
    --data-parallel-start-rank "${START_RANK}" \        # Starting rank offset
    --data-parallel-address "${NODE1_IP}" \             # Primary node IP
    --data-parallel-rpc-port "${RPC_PORT}" \            # RPC port (must match Node 1)
    --headless \                                        # No API server (worker only)
    --trust-remote-code                                 # Allow custom model code

# Additional useful options (uncomment as needed, must match Node 1):
#   --max-model-len 8192 \                              # Max sequence length
#   --gpu-memory-utilization 0.9 \                      # GPU memory usage (0.0-1.0)
#   --dtype auto \                                      # Data type (auto/float16/bfloat16)
