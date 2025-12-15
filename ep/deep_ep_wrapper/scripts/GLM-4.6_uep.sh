#!/bin/bash
set -e

# Source common scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
source "$SCRIPT_DIR/common_deepep_config.sh"
source "$SCRIPT_DIR/common_launch.sh"

# ============================
# Configuration
# ============================

# Optional installation commands:
# python setup.py install
# pip install ../../wheelhouse-cuda/uccl-0.0.1.post4-py3-none-any.whl

# Generate DeepEP config
DEEPEP_CFG=$(generate_deepep_config)

# Model and cluster configuration
MODEL_PATH="zai-org/GLM-4.6-FP8"
DIST_ADDR="172.31.36.62:5000"   # Node 0 master
NODE_RANK=$1
NNODES=4
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32  # DP=EP=TP

# ============================
# Launch
# ============================

# Note: GLM-4.6 uses fa3 backend and has specific parsers
launch_uep "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" "$DEEPEP_CFG" \
  --disable-radix-cache \
  --disable-chunked-prefix-cache \
  --tool-call-parser glm45 \
  --reasoning-parser glm45
  # If you see hangs in cuda graph capture, add:
  # --disable-cuda-graph
