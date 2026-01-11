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

# Generate DeepEP config (for reference, though NCCL backend won't use it)
DEEPEP_CFG=$(generate_deepep_config)

# ---------- Model path ----------
# Option A: use BF16 checkpoint from HuggingFace (remote)
MODEL_PATH="unsloth/DeepSeek-R1-0528-BF16"

# Option B (recommended for stability): download once and use local dir
# huggingface-cli download unsloth/DeepSeek-R1-0528-BF16 \
#   --local-dir /mnt/models/DeepSeek-R1-0528-BF16 \
#   --local-dir-use-symlinks False
# MODEL_PATH="/mnt/models/DeepSeek-R1-0528-BF16"

# Model and cluster configuration
DIST_ADDR="172.31.36.62:5000"   # Node 0 master (host:port)
NODE_RANK=$1                    # pass as first argument
NNODES=4                        # 4 nodes * 8 GPUs = 32 GPUs

# Parallelism layout: DP=EP=TP (all equal)
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32  # DP=EP=TP

# ============================
# Launch
# ============================

echo "DeepEP config: $DEEPEP_CFG"

launch_nccl "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
  --mem-fraction-static 0.8 \
  --deepep-config "$DEEPEP_CFG"
