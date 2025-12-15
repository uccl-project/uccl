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

# Generate DeepEP config
DEEPEP_CFG=$(generate_deepep_config)

# Model and cluster configuration
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528"
DIST_ADDR="172.31.36.62:5000"     # decode master
NODE_RANK=$1
NNODES=4
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32  # DP=EP=TP

# Additional environment for low-latency mode
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# ============================
# Launch
# ============================

echo ">>> Launching SGLang server (LOW LATENCY MODE)"

# Using base launch_sglang_server with custom args for low_latency mode
launch_sglang_server "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
  --host 0.0.0.0 \
  --port 30000 \
  --watchdog-timeout 1000000 \
  --mem-fraction-static 0.7 \
  --attention-backend flashinfer \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm dynamic \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-config "$DEEPEP_CFG" \
  --cuda-graph-bs 128
  # --disaggregation-transfer-backend nixl \
  # --disaggregation-mode decode \
  # --chunked-prefill-size 65536
  # Optional flags you may enable:
  # --prefill-round-robin-balance
  # --dp-size 16
  # --page-size 1
  # --disable-radix-cache
  # --max-running-requests 4096
  # --context-length 4500
  # --decode-log-interval 1
  # --enable-two-batch-overlap
  # --init-expert-location PATH
  # --num-reserved-decode-tokens VALUE
