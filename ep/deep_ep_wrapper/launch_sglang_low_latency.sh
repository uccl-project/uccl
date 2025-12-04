#!/bin/bash
set -e

# ============================
# Configuration
# ============================

NODE_RANK=$1
NNODES=2
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528"
DIST_ADDR="172.31.36.62:5000"     # decode master


# Local path for DeepEP config
DEEPEP_CFG="$(pwd)/deepep_config.json"

# Generate DeepEP config file if not exists
cat > "$DEEPEP_CFG" <<'EOF'
{
  "normal_dispatch": {
    "num_sms": 24,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 512,
    "num_max_rdma_chunked_send_tokens": 16,
    "num_max_rdma_chunked_recv_tokens": 512
  },
  "normal_combine": {
    "num_sms": 24,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 512,
    "num_max_rdma_chunked_send_tokens": 16,
    "num_max_rdma_chunked_recv_tokens": 512
  }
}
EOF

export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^lo,docker"
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# ============================
# Launch
# ============================

echo ">>> Launching SGLang server (DECODE MODE)"
echo "Node rank: $NODE_RANK"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --dist-init-addr "$DIST_ADDR" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --tp-size 16 \
  --ep-size 16 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --watchdog-timeout 1000000 \
  --mem-fraction-static 0.7 \
  --attention-backend flashinfer \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --ep-num-redundant-experts 16 \
  --ep-dispatch-algorithm dynamic \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-config "$DEEPEP_CFG" \
  --cuda-graph-bs 128 \
  --moe-dense-tp-size 1 \
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
  # --cuda-graph-bs 128
  # --decode-log-interval 1
  # --enable-two-batch-overlap
  # --init-expert-location PATH
  # --num-reserved-decode-tokens VALUE
