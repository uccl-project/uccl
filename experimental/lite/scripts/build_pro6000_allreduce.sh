#!/usr/bin/env bash
set -euo pipefail
# Source the existing isolated environment; do not install or alter system tools.
uccl_lite_env=${1:-/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910/runtime_env.sh}
uccl_lite_experiment=$(cd "$(dirname "$uccl_lite_env")" && pwd -P)
source "$uccl_lite_env"
cd "$(dirname "$0")/.."
make -j2 pro6000-ar pro6000-ar-reference CUDA_ARCH=sm_120 \
  MSCCLPP_HOME="$uccl_lite_experiment/deps/mscclpp" \
  NCCL_HOME="$uccl_lite_experiment/env/.venv/lib/python3.12/site-packages/nvidia/nccl"
