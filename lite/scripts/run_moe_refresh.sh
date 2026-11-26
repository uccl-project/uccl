#!/usr/bin/env bash
# Full-layer DeepSeek-shaped EP benchmark with the shipped settled,
# round-robin power-stability protocol. Keep the two complete-operation paths
# and one unfused UCCL Lite control.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${MOE_BIN:-$root/build/uccl_lite_fused_moe_layer}"
output="${OUTPUT_LOG:-$root/build/moe-refresh.log}"
if [[ -e "$output" ]]; then
  printf 'Refusing to overwrite %s\n' "$output" >&2
  exit 1
fi
timeout "${TIMEOUT_SECONDS:-3600}" "$binary" \
  -b 8 -e 32768 -f 2 -w 2 -n 4 -r 12 -S 2500 -c 1 \
  -X gemm-only -X cutlass-only -X disp-only -X comm-only -X ours+cutlass \
  > "$output" 2>&1
