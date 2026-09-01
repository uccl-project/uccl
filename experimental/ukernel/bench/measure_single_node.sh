#!/bin/bash
# measure_single_node.sh - shim vs native NCCL, single-node measurements.
#
# Same plan as docs/l40s_measurements.md (same-node columns only):
#   AllReduce: ranks 2/4/8, sizes 1M..256M (f4), shim blocks ladder
#   1/8/32 plus blocks == native coll channels, and native NCCL.
#   AllToAll:  ranks 2/4/8, sizes 1M..256M, shim (b8) + native.
# AllReduce runs are checked for 0 wrong; AllToAll runs verify the
# exchange (same-node) unless --skip-verify=1 is in A2A_EXTRA.
#
# If WAIT_IDLE=1 (default) the script waits until every GPU reports
# 0 MiB used before each run, so co-tenant jobs cannot pollute numbers.
#
# Env overrides:
#   UCCL_ROOT     ukernel root (default: script's ../..)
#   NCCTESTS_DIR  nccl-tests build dir
#   A2A_BIN       alltoall_perf binary
#   CUDA_LIBDIR   extra lib dir appended to both LD_LIBRARY_PATHs
#                 (cudart location when the system lacks /usr/local/cuda)
#   SHIM_LD / NATIVE_LD   explicit LD_LIBRARY_PATHs
#   RANKS / SIZES / AR_BLOCKS / WAIT_IDLE
#   ITERS / WARMUP (allreduce), A2A_ITERS / A2A_WARMUP / A2A_EXTRA
#   LOGDIR        output dir (default /tmp/uk_single_node/logs)
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UCCL_ROOT=${UCCL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO_ROOT=$(cd "$UCCL_ROOT/../.." && pwd)
NCCTESTS_DIR=${NCCTESTS_DIR:-$REPO_ROOT/thirdparty/nccl-tests/build}
A2A_BIN=${A2A_BIN:-/tmp/alltoall_perf}
LOGDIR=${LOGDIR:-/tmp/uk_single_node/logs}
RANKS=${RANKS:-"2 4 8"}
SIZES=${SIZES:-"1M 4M 16M 64M 256M"}
AR_BLOCKS=${AR_BLOCKS:-"1 8 32"}
WAIT_IDLE=${WAIT_IDLE:-1}
ITERS=${ITERS:-10}
WARMUP=${WARMUP:-2}
A2A_ITERS=${A2A_ITERS:-5}
A2A_WARMUP=${A2A_WARMUP:-2}
A2A_EXTRA=${A2A_EXTRA:-}

SHIM_LD=${SHIM_LD:-$UCCL_ROOT/build/nccl/lib:/usr/local/lib}
NATIVE_LD=${NATIVE_LD:-/usr/lib/x86_64-linux-gnu:/usr/local/lib}
if [ -n "${CUDA_LIBDIR:-}" ]; then
  SHIM_LD="$SHIM_LD:$CUDA_LIBDIR"
  NATIVE_LD="$NATIVE_LD:$CUDA_LIBDIR"
fi

mkdir -p "$LOGDIR"
RUNLOG=$LOGDIR/run.log

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

bytes_of() {
  case "$1" in
    1M)  echo 1048576 ;;
    4M)  echo 4194304 ;;
    16M) echo 16777216 ;;
    64M) echo 67108864 ;;
    256M) echo 268435456 ;;
    *) echo "$1" ;;
  esac
}

capture_channels() {
  local np=$1 ch
  wait_idle
  log "== native channels np=$np =="
  env LD_LIBRARY_PATH="$NATIVE_LD" NCCL_DEBUG=INFO \
    mpirun --bind-to none -np "$np" \
      -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO \
      "$NCCTESTS_DIR/all_reduce_perf" -b 1M -e 1M -f 4 -g 1 -c 1 -n 1 \
      > "$LOGDIR/channels_np${np}.txt" 2>&1
  ch=$(grep -oE 'nchannels [0-9]+' "$LOGDIR/channels_np${np}.txt" \
         | head -1 | awk '{print $2}')
  if [ -z "$ch" ]; then
    log "  FAILED to capture channels (exit=$?)"
    ch=0
  fi
  printf '%s %s\n' "$np" "$ch" >> "$LOGDIR/channels.txt"
  log "  native channels(np=$np)=$ch"
}

get_ch() { grep "^$1 " "$LOGDIR/channels.txt" | awk '{print $2}'; }

run_ar() {
  local label=$1 ld=$2 np=$3 blocks=$4
  local envs=(-x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1)
  [ "$blocks" != "native" ] && envs+=(-x UK_CCL_DEV_BLOCKS="$blocks")
  wait_idle
  log "== allreduce $label np=$np blocks=$blocks =="
  env LD_LIBRARY_PATH="$ld" \
    mpirun --bind-to none -np "$np" \
      "${envs[@]}" \
      "$NCCTESTS_DIR/all_reduce_perf" -b 1M -e 256M -f 4 -g 1 -c 1 \
      -n "$ITERS" -w "$WARMUP" \
      > "$LOGDIR/ar_${label}_np${np}_b${blocks}.txt" 2>&1
  log "  exit=$?  $(tail -1 "$LOGDIR/ar_${label}_np${np}_b${blocks}.txt")"
}

run_a2a() {
  local label=$1 ld=$2 np=$3 bytes=$4
  local envs=(-x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1)
  [ "$label" = "shim" ] && envs+=(-x UK_CCL_DEV_BLOCKS=8)
  wait_idle
  log "== alltoall $label np=$np bytes=$bytes =="
  # shellcheck disable=SC2086
  env LD_LIBRARY_PATH="$ld" \
    mpirun --bind-to none -np "$np" \
      "${envs[@]}" \
      "$A2A_BIN" --bytes="$bytes" --iters="$A2A_ITERS" --warmup="$A2A_WARMUP" \
      $A2A_EXTRA \
      > "$LOGDIR/a2a_${label}_np${np}_${bytes}.txt" 2>&1
  log "  exit=$?  $(grep -m1 'busbw' "$LOGDIR/a2a_${label}_np${np}_${bytes}.txt" || tail -1 "$LOGDIR/a2a_${label}_np${np}_${bytes}.txt")"
}

main() {
  log "start $(hostname) $(date)"
  log "shim  LD_LIBRARY_PATH: $SHIM_LD"
  log "native LD_LIBRARY_PATH: $NATIVE_LD"
  log "nccl-tests: $NCCTESTS_DIR  alltoall: $A2A_BIN"

  rm -f "$LOGDIR/channels.txt"
  local np ch
  for np in $RANKS; do
    capture_channels "$np"
  done

  for np in $RANKS; do
    ch=$(get_ch "$np")
    for b in $AR_BLOCKS; do
      run_ar shim "$SHIM_LD" "$np" "$b"
    done
    [ -n "$ch" ] && [ "$ch" != "0" ] && run_ar shim "$SHIM_LD" "$np" "$ch"
    run_ar native "$NATIVE_LD" "$np" native
  done

  local s b
  for np in $RANKS; do
    for s in $SIZES; do
      b=$(bytes_of "$s")
      run_a2a shim "$SHIM_LD" "$np" "$b"
      run_a2a native "$NATIVE_LD" "$np" "$b"
    done
  done

  log "done $(date)"
}

main "$@"
