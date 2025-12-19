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
MODEL_PATH="moonshotai/Kimi-K2-Instruct-0905"
DIST_ADDR="172.31.36.62:5000"   # Node 0 master
NODE_RANK=$1
NNODES=4
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32  # DP=EP=TP

# ============================
# Launch
# ============================

# Note: Kimi uses fa3 backend and has specific parser
# DeepEP backend is commented out - uncomment if needed
launch_sglang_server "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-chunked-prefix-cache \
  --tool-call-parser kimi_k2 \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm dynamic \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --deepep-config "$DEEPEP_CFG"
  # --moe-a2a-backend deepep \
  # --deepep-mode normal \
  # If you see hangs in cuda graph capture, add:
  # --disable-cuda-graph
