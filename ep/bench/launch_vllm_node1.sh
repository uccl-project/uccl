#!/bin/bash
# Node 1 (Primary) - Multi-node vLLM with Expert Parallel (EP)
# This node handles incoming requests
#
# Prerequisites:
# 1. Install vLLM with EP support: https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html#architecture-overview
# 2. Install DeepGEMM: https://github.com/deepseek-ai/DeepGEMM#installation
# 3. Install EP kernels: Follow vLLM's EP installation guide
# 4. For AWS EFA: Install AWS OFI-NCCL plugin

set -e

echo "🚀 Launching vLLM Node 1 (Primary) with Expert Parallel..."

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

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# IMPORTANT: Adjust these paths according to your installation
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
# Choose the appropriate backend based on your setup:
# - pplx: Single node deployment
# - deepep_low_latency: Multi-node, low-latency (decode-dominated workloads)
# - deepep_high_throughput: Multi-node, high-throughput (prefill-dominated)
# - allgather_reducescatter: Multi-node with NCCL (works well with InfiniBand/EFA)

export VLLM_ALL2ALL_BACKEND=allgather_reducescatter
export VLLM_USE_DEEP_GEMM=1

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# For InfiniBand/EFA clusters: Prevent initialization hangs
# This ensures torch distributed uses Ethernet for initial setup
# Find your network interface: ip addr show | grep -E 'eth|ib|enp'
#
# Common interfaces:
#   - eth0, eno1, enp0s3 (Ethernet)
#   - ib0, ib1 (InfiniBand)
#   - enp74s0, ens5 (Custom/AWS EFA)

export GLOO_SOCKET_IFNAME=eth0         # Change to your primary network interface
export NCCL_SOCKET_IFNAME=eth0       # Uncomment if using NCCL
export TP_SOCKET_IFNAME=eth0         # Uncomment if using tensor parallel

# ============================================================================
# NCCL CONFIGURATION (Optional - for advanced users)
# ============================================================================

# AWS EFA NCCL plugin (uncomment if using AWS EFA):
export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

# NCCL performance tuning (optional):
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608

# ============================================================================
# ARGUMENTS PARSING
# ============================================================================

NODE1_IP="$1"                                      # Node 1 IP address (REQUIRED)
RPC_PORT="${2:-13345}"                             # RPC communication port
MODEL="${3:-deepseek-ai/DeepSeek-V3-0324}"         # Model to serve
TOTAL_DP_SIZE="${4:-16}"                           # Total DP size across all nodes
LOCAL_DP_SIZE="${5:-8}"                            # Local DP size on this node
API_SERVERS="${6:-8}"                              # Number of API servers

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
echo "  • Backend: ${VLLM_ALL2ALL_BACKEND}"
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
echo "  • API Servers: ${API_SERVERS}"
echo "  • Expert Parallel: Enabled (automatically calculated)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# LAUNCH vLLM SERVER
# ============================================================================

vllm serve "${MODEL}" \
    --tensor-parallel-size 1 \                          # TP size (usually 1 for EP)
    --enable-expert-parallel \                          # Enable Expert Parallel
    --data-parallel-size "${TOTAL_DP_SIZE}" \           # Total DP across all nodes
    --data-parallel-size-local "${LOCAL_DP_SIZE}" \     # Local DP on this node
    --data-parallel-address "${NODE1_IP}" \             # Primary node IP
    --data-parallel-rpc-port "${RPC_PORT}" \            # RPC port for coordination
    --api-server-count="${API_SERVERS}" \               # Number of API servers
    --trust-remote-code                                  # Allow custom model code

# Additional useful options (uncomment as needed):
#   --max-model-len 8192 \                              # Max sequence length
#   --gpu-memory-utilization 0.9 \                      # GPU memory usage (0.0-1.0)
#   --dtype auto \                                      # Data type (auto/float16/bfloat16)
#   --enable-chunked-prefill \                          # Enable chunked prefill
#   --port 8000 \                                       # API server port
