#!/bin/bash
# sweep_ar_blocks.sh - shim AllReduce blocks-ladder saturation sweep.
#
# Runs all_reduce_perf (1M..256M, f4) for a fine blocks ladder against
# the shim, plus a native NCCL baseline, for each rank count. The point:
# find the smallest blocks count that saturates the shim's reduce path,
# and compare it with the number of channels native needs (fewer blocks
# for the same bandwidth = fewer SMs spent).
#
# Output: CSV ($LOGDIR/sweep_ar_blocks.csv) with rows
#   rank,blocks,size_bytes,busbw_gbps,wrong
# plus a printed summary: per (rank,size) max shim busbw, the blocks
# value that first reaches >= 95% of that max, and the native busbw.
# Any nonzero wrong aborts the sweep.
#
# Env overrides (same conventions as measure_single_node.sh):
#   RANKS / BLOCKS / ITERS / WARMUP / LOGDIR / WAIT_IDLE
#   UCCL_ROOT / NCCTESTS_DIR / CUDA_LIBDIR / SHIM_LD / NATIVE_LD
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UCCL_ROOT=${UCCL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO_ROOT=$(cd "$UCCL_ROOT/../.." && pwd)
NCCTESTS_DIR=${NCCTESTS_DIR:-$REPO_ROOT/thirdparty/nccl-tests/build}
LOGDIR=${LOGDIR:-/tmp/uk_single_node/logs}
RANKS=${RANKS:-"2 4 8"}
BLOCKS=${BLOCKS:-"2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 44 48 52 56 60 64"}
ITERS=${ITERS:-10}
WARMUP=${WARMUP:-2}
WAIT_IDLE=${WAIT_IDLE:-1}

SHIM_LD=${SHIM_LD:-$UCCL_ROOT/build/nccl/lib:/usr/local/lib}
NATIVE_LD=${NATIVE_LD:-/usr/lib/x86_64-linux-gnu:/usr/local/lib}
if [ -n "${CUDA_LIBDIR:-}" ]; then
  SHIM_LD="$SHIM_LD:$CUDA_LIBDIR"
  NATIVE_LD="$NATIVE_LD:$CUDA_LIBDIR"
fi

mkdir -p "$LOGDIR"
RUNLOG=$LOGDIR/sweep_run.log
CSV=$LOGDIR/sweep_ar_blocks.csv

log() { echo "[$(date +%F_%T)] $*" | tee -a "$RUNLOG"; }

gpu_idle() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '$1 != 0' | wc -l)
  [ "${busy:-1}" -eq 0 ]
}

wait_idle() {
  [ "$WAIT_IDLE" != 1 ] && return 0
  while ! gpu_idle; do
    log "GPUs busy, waiting 60s..."
    sleep 60
  done
}

run_ar() {
  local np=$1 b=$2 ld=$3
  local envs=(-x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1)
  [ "$b" != "native" ] && envs+=(-x UK_CCL_DEV_BLOCKS="$b")
  wait_idle
  local f="$LOGDIR/sweep_ar_np${np}_b${b}.txt"
  log "== sweep allreduce np=$np blocks=$b =="
  env LD_LIBRARY_PATH="$ld" \
    mpirun --bind-to none -np "$np" "${envs[@]}" \
      "$NCCTESTS_DIR/all_reduce_perf" -b 1M -e 256M -f 4 -g 1 -c 1 \
      -n "$ITERS" -w "$WARMUP" \
      > "$f" 2>&1
  local rc=$?
  local wcols
  wcols=$(awk '/^#/ && /busbw/ {
               sub(/^#/, "", $0)
               for (i = 1; i <= NF; i++)
                 if ($i ~ /wrong/) printf "%s ", i
               exit
             }' "$f")
  awk -v np="$np" -v b="$b" -v c="$wcols" '
    BEGIN { split(c, a, " "); for (i in a) ci[a[i]] = 1 }
    /^ *[0-9]+/ && NF >= 13 {
      w = 0
      for (i in ci) if (i + 0 <= NF && $i != 0) w = 1
      printf "%s,%s,%s,%.2f,%d\n", np, b, $1, $8, w
    }
  ' "$f" >> "$CSV"
  local bad
  bad=$(awk -F, -v np="$np" -v b="$b" '$1==np && $2==b && $5!=0 { n++ } END { print n+0 }' "$CSV")
  log "  exit=$rc wrong_lines=$bad"
  if [ "$rc" -ne 0 ] || [ "$bad" -gt 0 ]; then
    log "ABORT: sweep np=$np blocks=$b failed"
    exit 1
  fi
}

summary() {
  log "== saturation summary =="
  awk -F, 'NR > 1 {
      key = $1 "," $3
      if ($2 == "native") nat[key] = $4
      else {
        if (!(key in max) || $4 > max[key]) { max[key] = $4; maxb[key] = $2 }
        rows[key, cnt[key]++] = $2 " " $4
      }
    }
    END {
      for (key in max) {
        split(key, k, ","); np = k[1]; s = k[2]
        sat = "-"
        for (i = 0; i < cnt[key]; i++) {
          split(rows[key, i], r, " ")
          if (r[2] >= 0.95 * max[key]) { sat = r[1]; break }
        }
        ratio = nat[key] ? max[key] / nat[key] : 0
        printf "%d %d %.1f %s %.1f %.2f\n", np, s, max[key], sat, nat[key], ratio
      }
    }' "$CSV" | sort -n -k1,1 -k2,2 | tee -a "$RUNLOG"
}

main() {
  log "start sweep $(hostname) $(date)"
  log "shim  LD: $SHIM_LD"
  log "native LD: $NATIVE_LD"
  : > "$CSV"
  echo "rank,blocks,size_bytes,busbw_gbps,wrong" > "$CSV"

  local np b
  for np in $RANKS; do
    for b in $BLOCKS; do
      run_ar "$np" "$b" "$SHIM_LD"
    done
    run_ar "$np" native "$NATIVE_LD"
  done

  summary
  log "done $(date)"
}

main "$@"
