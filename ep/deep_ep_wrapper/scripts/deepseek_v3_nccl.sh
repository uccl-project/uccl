#!/bin/bash
set -e

# ============================
# Configuration
# ============================

# python setup.py install
# pip install ../../wheelhouse-cuda/uccl-0.0.1.post4-py3-none-any.whl


# Environment setup
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^lo,docker"
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1

# Parameters
MODEL_PATH="deepseek-ai/DeepSeek-V3-0324"
DIST_ADDR="172.31.36.62:5000"   # Node 0 master
NODE_RANK=$1
NNODES=4

# ============================
# Launch
# ============================

echo ">>> Launching SGLang server"
echo "Node rank: $NODE_RANK"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size 32 \
  --ep-size 32 \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_ADDR" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm dynamic \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --chunked-prefill-size 65536 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --page-size 256 \
  --moe-dense-tp-size 1 \
  --chunked-prefill-size 32768 \
  --cuda-graph-bs 256 \
  --dp-size 32 \