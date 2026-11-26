#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
compat=${UCCL_LITE_COMPAT_SO:-"$here/build/libuccl_lite_nccl_compat.so"}
model_path=${MODEL_PATH:-/data/wxzheng/cache/huggingface/models--zai-org--GLM-4.5-Air-FP8/snapshots/f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc}
sglang_python=${SGLANG_PYTHON:-/data/wxzheng/kt-venv/bin/python}
port=${PORT:-30089}

test -f "$compat"
test -f "$model_path/config.json"
export PATH="$(dirname "$sglang_python"):/usr/local/cuda/bin:$PATH"
export LD_PRELOAD="$compat${LD_PRELOAD:+:$LD_PRELOAD}"
export SGLANG_NCCL_SO_PATH="$compat"
export UCCL_LITE_NCCL_ENABLE=${UCCL_LITE_NCCL_ENABLE:-1}
export UCCL_LITE_NCCL_MAX_BYTES=${UCCL_LITE_NCCL_MAX_BYTES:-33554432}
if [[ ${UCCL_LITE_NCCL_NO_OVERRIDES:-0} != 1 ]]; then
  export NCCL_NET=${UCCL_LITE_NCCL_NET:-Socket}
  export NCCL_NET_PLUGIN=${UCCL_LITE_NCCL_NET_PLUGIN:-none}
  export NCCL_CUMEM_ENABLE=0
  export NCCL_NVLS_ENABLE=0
fi
export MAX_JOBS=${MAX_JOBS:-2}

exec "$sglang_python" -m sglang.launch_server \
  --model-path "$model_path" --tp-size 8 --dp-size 8 --ep-size 8 \
  --enable-dp-attention --moe-dense-tp-size 1 --moe-a2a-backend none \
  --moe-runner-backend triton --attention-backend triton \
  --mem-fraction-static 0.9 --context-length 4096 --max-running-requests 32 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-radix-cache \
  --port "$port" "$@"
