#!/bin/bash
# Test EP low latency mode with Qwen3-30B on Intel irdma NICs
set -e

# Source common scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env_intel_nic.sh"
source "$SCRIPT_DIR/common_deepep_config.sh"
source "$SCRIPT_DIR/common_launch.sh"

# ============================
# Configuration
# ============================

# Tested with SGLang v0.5.9
# Install SGLang: pip install --force-reinstall "sglang[all]==0.5.9"
#
# Optional installation commands:
# python setup.py install
# pip install ../../../../wheelhouse-cu12/uccl-0.1.0.post6-cp312-abi3-manylinux_2_34_x86_64.whl

# Generate DeepEP config
DEEPEP_CFG=$(generate_deepep_config)

# Usage: bash Qwen3-30B_uep_intel_nic.sh <node_rank> [nnodes] [gpus_per_node]
#   node_rank:     Rank of this node (0-indexed, required)
#   nnodes:        Number of nodes (default: 2)
#   gpus_per_node: GPUs per node (default: 1)
MODEL_PATH="Qwen/Qwen3-30B-A3B-FP8"
DIST_ADDR="10.173.44.108:5000"   # Node 0 master

NODE_RANK=${1:?Usage: $0 <node_rank> [nnodes] [gpus_per_node]}
NNODES=${2:-2}
GPUS_PER_NODE=${3:-1}

# With DeepEP enabled, SGLang forces EP=TP.
# TP = total GPUs across all nodes. EP = TP. DP = 1.
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
TP_SIZE=$TOTAL_GPUS
EP_SIZE=$TOTAL_GPUS
DP_SIZE=1

# Low-latency mode environment
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128

export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
# Make all local GPUs visible (0,1,...,n-1)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# ============================
# Launch
# ============================

launch_sglang_server "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
  --attention-backend flashinfer \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-config "$DEEPEP_CFG" \
  --chunked-prefill-size 65536
