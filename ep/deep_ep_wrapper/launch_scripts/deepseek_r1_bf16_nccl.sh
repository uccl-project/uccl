#!/bin/bash
set -e

# ============================
# Configuration
# ============================

# Optional: install your local UCCL wheel, etc.
# python setup.py install
# pip install ../../wheelhouse-cuda/uccl-0.0.1.post4-py3-none-any.whl

# ---------- Model path ----------
# Option A: use BF16 checkpoint from HuggingFace (remote)
MODEL_PATH="unsloth/DeepSeek-R1-0528-BF16"

# Option B (recommended for stability): download once and use local dir
# huggingface-cli download unsloth/DeepSeek-R1-0528-BF16 \
#   --local-dir /mnt/models/DeepSeek-R1-0528-BF16 \
#   --local-dir-use-symlinks False
# MODEL_PATH="/mnt/models/DeepSeek-R1-0528-BF16"

# ---------- DeepEP config ----------
DEEPEP_CFG="$(pwd)/deepep_config.json"

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

# ============================
# Environment setup
# ============================

export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^lo,docker"
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

# JIT GEMM for DeepSeek kernels
export SGLANG_ENABLE_JIT_DEEPGEMM=1

# ============================
# Distributed parameters
# ============================

DIST_ADDR="172.31.36.62:5000"   # Node 0 master (host:port)
NODE_RANK=$1                    # pass as first argument
NNODES=4                        # 4 nodes * 8 GPUs = 32 GPUs

# Parallelism layout:
#   32 GPUs total
#   TP = 1, EP = 32, DP = 1  -> single EP=32 group across all GPUs
TP_SIZE=32
EP_SIZE=32
DP_SIZE=32

# ============================
# Launch
# ============================

echo ">>> Launching SGLang server"
echo "Node rank: $NODE_RANK"
echo "DeepEP config: $DEEPEP_CFG"
echo "Model path: $MODEL_PATH"
echo "TP=${TP_SIZE}, EP=${EP_SIZE}, DP=${DP_SIZE}, NNODES=${NNODES}"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size $TP_SIZE \
  --ep-size $EP_SIZE \
  --dp-size $DP_SIZE \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_ADDR" \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --attention-backend flashinfer \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm dynamic \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --page-size 256 \
  --moe-dense-tp-size 1 \
  --chunked-prefill-size 32768 \
  --cuda-graph-bs 256 \
  --deepep-config "$DEEPEP_CFG"
