#!/usr/bin/env bash
set -euo pipefail
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv
nvidia-smi topo -m
lscpu
/usr/local/cuda/bin/nvcc --version
sha256sum integration/l4_e2e/benchmark.py integration/l4_e2e/collectives.cc src/collectives/peer_collectives.cu build/l4_e2e/libuccl_lite_peer_e2e.so "$HOME/third_party/nccl/build/lib/libnccl.so.2" "$HOME/.cache/uccl-lite-gpt2-large/model.safetensors"
