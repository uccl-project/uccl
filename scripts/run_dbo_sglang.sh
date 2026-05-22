#!/bin/bash
# Two-node sglang+DBO repro using the ziming uv env.
# Adapted from the pastebin Qwen3-30B + --enable-two-batch-overlap script.
# Usage: ./run_dbo_sglang.sh <node_rank>   # 0=local(172.31.73.10), 1=remote(p5en_1)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Activate ziming uv env (has torch 2.9.1+cu128, sglang editable, uccl.ep, deep_ep)
source "$REPO_DIR/ziming/bin/activate"

# ---- LD_LIBRARY_PATH (EFA + cuda; no custom libfabric)
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# Shared HF cache via EFS
export HF_HUB_CACHE=/home/ubuntu/efs/xingyu/hf_cache/hub

# NVSHMEM (for the LL deepep paths over EFA)
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa

# NCCL / UCCL bootstrap
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp71s0
export UCCL_SOCKET_IFNAME=enp71s0
export NCCL_IB_GID_INDEX=0
export UCCL_IB_GID_INDEX=0
# Cap NCCL channels so they don't fight uccl-ep for EFA QPs/CQs.
# Without this, NCCL DP-attention all-gathers + uccl-ep RDMA dispatch
# can deadlock at the second dispatch_b (CPU timeout on counters).
export NCCL_MAX_NCHANNELS=8
# Bump uccl-ep's CPU-side dispatch timeout to 10 min so we can attach py-spy
# while the dispatch is genuinely stuck instead of seeing it die at 100s.
export UCCL_EP_CPU_TIMEOUT_SECS=600

# EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

# CUDA + sglang JIT
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SG_DEEPGEMM_JIT=1

# ---- DeepEP config (shape from pastebin)
DEEPEP_CFG="$SCRIPT_DIR/deepep_config_2node.json"
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

# ---- Cluster config
MODEL_PATH="Qwen/Qwen3-30B-A3B-FP8"     # pre-quantized FP8 (already cached)
NODE_RANK="${1:?node_rank required (0=local, 1=remote)}"
NNODES=2
TP_SIZE=16
EP_SIZE=16
DP_SIZE=16
DIST_ADDR="172.31.73.10:5001"
SGLANG_PORT="30001"

echo ">>> Launching sglang"
echo "    Model:     $MODEL_PATH"
echo "    TP/EP/DP:  $TP_SIZE / $EP_SIZE / $DP_SIZE"
echo "    NNODES:    $NNODES, node_rank=$NODE_RANK"
echo "    Master:    $DIST_ADDR"

# IMPORTANT: cd away from the uccl repo so the local ./uccl/ source dir
# does not shadow the site-packages `uccl` (which actually has ep.abi3.so).
# Without this, `from deep_ep import Buffer, Config` fails -> sglang thinks
# DeepEP is uninstalled.
cd /tmp

exec python -m sglang.launch_server \
  --port "$SGLANG_PORT" \
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
  --cuda-graph-max-bs 128 \
  --page-size 256 \
  --attention-backend fa3 \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm dynamic \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "$DEEPEP_CFG" \
  --skip-server-warmup \
  --enable-two-batch-overlap
