#!/bin/bash
set -e

# Source common scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
source "$SCRIPT_DIR/common_launch.sh"

# ============================
# Configuration
# ============================

# Optional installation commands:
# python setup.py install
# pip install ../../wheelhouse-cuda/uccl-0.0.1.post4-py3-none-any.whl

# Model and cluster configuration
MODEL_PATH="Qwen/Qwen3-30B-A3B-FP8"
DIST_ADDR="172.31.36.62:5000"   # Node 0 master
NODE_RANK=$1
NNODES=4
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32  # DP=EP=TP

# ============================
# Launch
# ============================

launch_nccl "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
  --chunked-prefill-size 65536