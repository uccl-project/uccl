#!/usr/bin/env bash
set -euo pipefail
exec torchrun --nnodes="${CI_NODES:?}" --nproc_per_node="${CI_GPUS:?}" \
  --node_rank="${CI_NODE:?}" --master_addr="${MASTER_ADDR:?}" \
  --master_port="${MASTER_PORT:?}" "$@"
