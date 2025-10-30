#!/bin/bash
# Fail loud on any error, undefined var, or pipe failure

set -euo pipefail
cd /workspace/uccl/ep/deep_ep_wrapper
python setup.py install
pip install ../../wheelhouse-cuda/uccl-0.0.1.post4-py3-none-manylinux_2_34_x86_64.whl

# ============================
# Configuration
# ============================

# Local path for DeepEP config
DEEPEP_CFG="$(pwd)/deepep_config.json"

# Generate DeepEP config file (idempotent)
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

# --- CUDA / Hopper (H200) toolchain pinning ---
# Prefer the system CUDA 12.8 toolchain everywhere (nvcc/ptxas/nvrtc).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/nccl/build/lib:$CUDA_HOME/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}"

# Force JIT/compile for Hopper arch-accelerated features (sm_90a).
export TORCH_CUDA_ARCH_LIST="9.0a"
export CUDAARCHS="90a"

# Helpful verbosity to catch JIT issues
export PTXAS_OPTIONS="-v"
export CUDA_VERBOSE_PTXAS=1

# --- Avoid /tmp noexec for CUDA/NVRTC caches ---
CUDA_CACHE_DIR="$(pwd)/.cuda_cache"
TMPDIR_SAFE="$(pwd)/.tmp_exec"
mkdir -p "$CUDA_CACHE_DIR" "$TMPDIR_SAFE"
export CUDA_CACHE_PATH="$CUDA_CACHE_DIR"
export CUDA_CACHE_MAXSIZE=2147483647
export TMPDIR="$TMPDIR_SAFE"

# --- NCCL / NVSHMEM / EFA ---
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="^lo,docker"
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export FI_LOG_LEVEL=warn

# ============================
# Parameters
# ============================

MODEL_PATH="/workspace/models/DeepSeek-R1-0528"
DIST_ADDR="172.31.36.62:5000"   # Node 0 master
NODE_RANK=${1:? "Usage: $0 <node_rank:int>"}
NNODES=2

# ============================
# Launch
# ============================

# Quick sanity logs to make JIT/toolchain issues obvious
echo ">>> Launching SGLang server"
echo "Node rank: $NODE_RANK"
echo "DeepEP config: $DEEPEP_CFG"
echo "CUDA_HOME=$CUDA_HOME"

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not found"
fi

if command -v ptxas >/dev/null 2>&1; then
  ptxas --version | head -n 3 || true
else
  echo "ptxas not found"
fi

python - <<'PY' || true
import torch, os, ctypes, sys
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("arch list:", os.getenv("TORCH_CUDA_ARCH_LIST"))
try:
  ctypes.CDLL('libnvrtc.so')
  print("nvrtc: loaded OK")
except OSError as e:
  print("nvrtc load error:", e, file=sys.stderr)
PY

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size 16 \
  --ep-size 16 \
  --dp-size 16 \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_ADDR" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --enable-eplb \
  --eplb-algorithm deepseek \
  --ep-num-redundant-experts 16 \
  --ep-dispatch-algorithm dynamic \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "$DEEPEP_CFG" \
  --enable-dp-attention \
  --enable-dp-lm-head

echo ">>> Done."
