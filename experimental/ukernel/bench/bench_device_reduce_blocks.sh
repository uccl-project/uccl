#!/usr/bin/env bash
# bench_device_reduce_blocks.sh — measure device reduce-kernel throughput vs
# block count (and threads/block), using the persistent-worker dispatch bench.
#
# Usage (from experimental/ukernel):
#   bash bench/bench_device_reduce_blocks.sh
#
# Env overrides:
#   BLOCKS   space-separated block counts    (default "1 2 4 8 16 32 64 128")
#   THREADS  space-separated threads/block   (default "128 256 512")
#   SIZES    space-separated payload sizes   (default "16M 64M 256M"; K/M/G suffixes ok)
#   ROUNDS / WARMUP                          (default 100 / 20)
#   SMEM     reduce shared memory            (default 4096)
#
# Output rows: blocks|threads|bytes|task_us|GB/s
# GB/s = payload bytes reduced per second (read+write traffic is 2x that).
#
# Use these numbers to pick blocks_per_worker so the reduce kernel keeps up
# with the IPC put bandwidth measured in bench_shim_param_sweep.sh.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$UK_DIR" || exit 1

BLOCKS="${BLOCKS:-1 2 4 8 16 32 64 128}"
THREADS="${THREADS:-128 256 512}"
SIZES="${SIZES:-16M 64M 256M}"
ROUNDS="${ROUNDS:-100}"
WARMUP="${WARMUP:-20}"
SMEM="${SMEM:-4096}"
BENCH="src/device/benchmarks/bench_device_launch_vs_worker"

if [[ ! -x "$BENCH" ]]; then
  echo "# building device bench..."
  make -j"$(nproc)" device_bench >/dev/null || {
    echo "error: make device_bench failed" >&2; exit 1; }
fi
[[ -x "$BENCH" ]] || { echo "error: $BENCH missing after build" >&2; exit 1; }

bytes_of() {
  local v="$1"
  case "$v" in
    *[Kk]) echo $(( ${v%[Kk]} * 1024 )) ;;
    *[Mm]) echo $(( ${v%[Mm]} * 1024 * 1024 )) ;;
    *[Gg]) echo $(( ${v%[Gg]} * 1024 * 1024 * 1024 )) ;;
    *) echo "$v" ;;
  esac
}

echo "# device reduce throughput: persistent worker, 1 task/round, smem=$SMEM"
echo "# blocks|threads|bytes|task_us|GB_s"

for sz in $SIZES; do
  bytes="$(bytes_of "$sz")"
  for b in $BLOCKS; do
    for t in $THREADS; do
      out="$("$BENCH" 1 "$ROUNDS" "$WARMUP" "$bytes" "$b" "$t" "$SMEM" 2>/dev/null)"
      task_us="$(printf '%s\n' "$out" | awk '
        /=== reduce ===/ { inr = 1; next }
        inr && /single enqueue/ { single = 1; next }
        single && /Task latency/ {
          match($0, /[0-9.]+/); print substr($0, RSTART, RLENGTH); exit }')"
      if [[ -z "$task_us" ]]; then
        echo "# blocks=$b threads=$t bytes=$bytes : parse failed (bench output changed?)" >&2
        continue
      fi
      gbps="$(awk -v u="$task_us" -v n="$bytes" 'BEGIN { printf "%.1f", n / u / 1000.0 }')"
      printf '%s|%s|%s|%s|%s\n' "$b" "$t" "$bytes" "$task_us" "$gbps"
    done
  done
done
