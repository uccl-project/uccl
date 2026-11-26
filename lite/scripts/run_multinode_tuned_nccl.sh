#!/usr/bin/env bash
# Recheck only the default NCCL LL spike sizes with an explicit RING+SIMPLE
# control. Other sizes retain the complete default-control paired sweep.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root_output="${OUTPUT_DIR:-$root/build/multinode-tuned-nccl}"
mkdir -p "$root_output"

cases=(
  allreduce:262144 allreduce:524288 allreduce:1048576
  allgather:262144 allgather:524288
  reducescatter:131072 reducescatter:262144 reducescatter:524288
)
for entry in "${cases[@]}"; do
  IFS=: read -r operation bytes <<< "$entry"
  OUTPUT_DIR="$root_output/${operation}_${bytes}" \
  OPERATIONS="$operation" MIN_BYTES="$bytes" MAX_BYTES="$bytes" \
  WARMUPS=20 ITERATIONS=10 REPEATS=3 CHECK_EACH=1 TIMING_MODE=host \
  AUTO_VARIANTS=1 NCCL_CONTROL_ALGO=Ring NCCL_CONTROL_PROTO=Simple \
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}" \
    bash "$root/scripts/run_multinode_peer_compare.sh"
done
