#!/bin/bash
# Common SGLang launch function
# Builds and executes python -m sglang.launch_server with common and custom arguments

launch_sglang_server() {
  local MODEL_PATH="$1"
  local TP_SIZE="$2"
  local EP_SIZE="$3"
  local DP_SIZE="$4"
  local NNODES="$5"
  local NODE_RANK="$6"
  local DIST_ADDR="$7"
  shift 7
  
  # Remaining arguments are passed as additional flags to launch_server
  local EXTRA_ARGS=("$@")
  
  echo ">>> Launching SGLang server"
  echo "Model path:  $MODEL_PATH"
  echo "TP=${TP_SIZE}, EP=${EP_SIZE}, DP=${DP_SIZE}, NNODES=${NNODES}"
  echo "Node rank:   $NODE_RANK"
  echo "Dist addr:   $DIST_ADDR"
  
  python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --dp-size "$DP_SIZE" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_ADDR" \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --moe-dense-tp-size 1 \
    --chunked-prefill-size 32768 \
    --cuda-graph-bs 256 \
    --page-size 256 \
    "${EXTRA_ARGS[@]}"
}

# Helper function for launching with NCCL backend (no DeepEP)
launch_nccl() {
  local MODEL_PATH="$1"
  local TP_SIZE="$2"
  local EP_SIZE="$3"
  local DP_SIZE="$4"
  local NNODES="$5"
  local NODE_RANK="$6"
  local DIST_ADDR="$7"
  shift 7
  
  launch_sglang_server \
    "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
    --attention-backend flashinfer \
    --enable-eplb \
    --eplb-algorithm deepseek \
    --ep-num-redundant-experts 0 \
    --ep-dispatch-algorithm dynamic \
    --enable-dp-attention \
    --enable-dp-lm-head \
    "$@"
}

# Helper function for launching with DeepEP backend
launch_uep() {
  local MODEL_PATH="$1"
  local TP_SIZE="$2"
  local EP_SIZE="$3"
  local DP_SIZE="$4"
  local NNODES="$5"
  local NODE_RANK="$6"
  local DIST_ADDR="$7"
  local DEEPEP_CFG="$8"
  shift 8
  
  echo "DeepEP config: $DEEPEP_CFG"
  
  launch_sglang_server \
    "$MODEL_PATH" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$NNODES" "$NODE_RANK" "$DIST_ADDR" \
    --attention-backend fa3 \
    --ep-num-redundant-experts 0 \
    --ep-dispatch-algorithm dynamic \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --deepep-config "$DEEPEP_CFG" \
    "$@"
}
