#!/usr/bin/env bash
# bench_shim_param_sweep.sh — sweep the shim's env knobs against nccl-tests
# to find the best (LARGE_TILES, IPC_BATCH, TILE_MIN_BYTES) combo on THIS
# machine.
#
# Usage (from experimental/ukernel):
#   CUDA_VISIBLE_DEVICES=6,7 bash bench/bench_shim_param_sweep.sh [all_reduce|all_gather|reduce_scatter]
#
# Env overrides:
#   SWEEP_MIN / SWEEP_MAX  size range           (default 1M / 256M, factor-2 steps)
#   ITERS / WARMUP         perftest iterations  (default 20 / 5)
#   LARGE_TILES_VALS       tile-count targets, "-" = unset (default "64 32 16 8 4")
#   IPC_BATCH_VALS         IPC in-flight sizes, "-" = unset (default "16 24 32 48")
#   TILE_MIN_VALS          UK_CCL_TILE_MIN_BYTES, "-" = unset (default "- 2097152 4194304 8388608")
#   DEV_BLOCKS_VALS        UK_CCL_DEV_BLOCKS sweep (default "8 16 32 64";
#                          128+ is unrealistic for the shim's low-SM goal),
#                          run with BASE_LT / BASE_TM / BASE_IB fixed
#   BASE_LT / BASE_TM / BASE_IB   config used for the DEV_BLOCKS pass
#                                 (default 8 / 8388608 / 16)
#
# Output: one line per (config, size):
#   label|size|oop_time_us|oop_algbw|oop_wrong|ip_time_us|ip_algbw|ip_wrong
# followed by a best-per-size summary (max OOP algbw).
#
# NOTE: mpirun must run with CPU binding disabled (--mca hwloc_base_binding_policy
# none) or small-message latency explodes; env knobs are propagated with -x.

set -uo pipefail

COLL="${1:-all_reduce}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$UK_DIR/../.." && pwd)"
PERF="$REPO_ROOT/thirdparty/nccl-tests/build/${COLL}_perf"

[[ -x "$PERF" ]] || { echo "error: $PERF not found — build nccl-tests first" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export LD_LIBRARY_PATH="$UK_DIR/build/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MIN="${SWEEP_MIN:-1M}"
MAX="${SWEEP_MAX:-256M}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
LARGE_TILES_VALS="${LARGE_TILES_VALS:-64 32 16 8 4}"
IPC_BATCH_VALS="${IPC_BATCH_VALS:-16 24 32 48}"
TILE_MIN_VALS="${TILE_MIN_VALS:-- 2097152 4194304 8388608}"
DEV_BLOCKS_VALS="${DEV_BLOCKS_VALS:-8 16 32 64}"
BASE_LT="${BASE_LT:-8}"
BASE_TM="${BASE_TM:-8388608}"
BASE_IB="${BASE_IB:-16}"

echo "# collective=$COLL range=$MIN..$MAX iters=$ITERS warmup=$WARMUP gpus=$CUDA_VISIBLE_DEVICES"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# run_one LABEL [VAR=VAL ...]  — "-" value means "leave unset"
run_one() {
  local label="$1"; shift
  local -a envs=() xflags=(-x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES)
  local kv name val
  for kv in "$@"; do
    name="${kv%%=*}"
    val="${kv#*=}"
    [[ "$val" == "-" ]] && continue
    envs+=("$kv")
    xflags+=("-x" "$name")
  done

  local out
  out="$(env "${envs[@]}" mpirun --mca hwloc_base_binding_policy none -np 2 \
      "${xflags[@]}" "$PERF" -b "$MIN" -e "$MAX" -f 2 -g 1 -c 1 \
      -n "$ITERS" -w "$WARMUP" 2>/dev/null)"

  local n=0 sz t_o a_o w_o t_i a_i w_i
  while read -r sz t_o a_o w_o t_i a_i w_i; do
    [[ -n "$sz" ]] || continue
    n=$((n + 1))
    printf '%s|%s|%.1f|%.2f|%s|%.1f|%.2f|%s\n' \
      "$label" "$sz" "$t_o" "$a_o" "$w_o" "$t_i" "$a_i" "$w_i" >> "$TMP"
    printf '%s|%s|%.1f|%.2f|%s|%.1f|%.2f|%s\n' \
      "$label" "$sz" "$t_o" "$a_o" "$w_o" "$t_i" "$a_i" "$w_i"
  done < <(printf '%s\n' "$out" | awk '$1 ~ /^[0-9]+$/ && NF >= 13 {
             print $1, $6, $7, $9, $10, $11, $13 }')
  [[ $n -gt 0 ]] || echo "# $label : NO DATA (run failed?)" >&2
}

echo "# --- pass 1: LARGE_TILES x IPC_BATCH (TILE_MIN default) ---"
for lt in $LARGE_TILES_VALS; do
  for ib in $IPC_BATCH_VALS; do
    run_one "LT=$lt IB=$ib TM=-" \
      "UK_CCL_LARGE_TILES=$lt" "UK_CCL_IPC_BATCH=$ib"
  done
done

echo "# --- pass 2: TILE_MIN sweep (LARGE_TILES/IPC_BATCH default) ---"
for tm in $TILE_MIN_VALS; do
  run_one "LT=- IB=- TM=$tm" "UK_CCL_TILE_MIN_BYTES=$tm"
done

echo "# --- pass 3: DEV_BLOCKS sweep (LT=$BASE_LT TM=$BASE_TM IB=$BASE_IB fixed) ---"
for blk in $DEV_BLOCKS_VALS; do
  run_one "BLK=$blk LT=$BASE_LT TM=$BASE_TM IB=$BASE_IB" \
    "UK_CCL_DEV_BLOCKS=$blk" "UK_CCL_LARGE_TILES=$BASE_LT" \
    "UK_CCL_TILE_MIN_BYTES=$BASE_TM" "UK_CCL_IPC_BATCH=$BASE_IB"
done

echo
echo "# --- BEST OOP algbw per size ---"
awk -F'|' '
  { sz = $2; if ($4 + 0 > best[sz] + 0) {
      best[sz] = $4 + 0;
      blab[sz] = $1 "  oop=" $4 " GB/s (" $3 "us, wrong=" $5 ")  ip=" $7 " GB/s (" $6 "us, wrong=" $8 ")";
    }
  }
  END { for (s in best) printf "%s  %s\n", s, blab[s] }
' "$TMP" | sort -n
