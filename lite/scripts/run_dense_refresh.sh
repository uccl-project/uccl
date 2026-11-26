#!/usr/bin/env bash
# Three independent, shape-order-alternated runs of the shipped dense harness.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${DENSE_BIN:-$root/build/uccl_lite_dense_fusion}"
output="${OUTPUT_DIR:-$root/build/dense-refresh}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
mkdir -p "$output"

# name:M:AG-N:AG-K:RS-N:RS-K. The AR shape equals the RS shape.
shapes=(
  llama8-m128:128:28672:4096:4096:14336
  llama8-m256:256:28672:4096:4096:14336
  llama8-m512:512:28672:4096:4096:14336
  llama8-m1024:1024:28672:4096:4096:14336
  llama8-m2048:2048:28672:4096:4096:14336
  qwen32-m2048:2048:55296:5120:5120:27648
  llama70-m2048:2048:57344:8192:8192:28672
  reference-m4096:4096:49152:12288:12288:49152
)

for repeat in 1 2 3; do
  if ((repeat == 2)); then
    order=(7 6 5 4 3 2 1 0)
  else
    order=(0 1 2 3 4 5 6 7)
  fi
  for index in "${order[@]}"; do
    IFS=: read -r name m ag_n ag_k rs_n rs_k <<< "${shapes[index]}"
    log="$output/${name}_run${repeat}.log"
    if [[ -e "$log" ]]; then
      printf 'Refusing to overwrite %s\n' "$log" >&2
      exit 1
    fi
    timeout "$timeout_seconds" "$binary" \
      --ag-m "$m" --ag-n "$ag_n" --ag-k "$ag_k" \
      --rs-m "$m" --rs-n "$rs_n" --rs-k "$rs_k" \
      -w 5 -n 20 -c 1 > "$log" 2>&1
  done
done
