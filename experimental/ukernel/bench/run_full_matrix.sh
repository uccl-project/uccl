#!/bin/bash
# run_full_matrix.sh — shim vs native matrix, one host (B300: 8 GPUs).
#
# Collectives: AllReduce (nccl-tests all_reduce_perf, ring/fp32/OOP) and
# AllToAll (bench/alltoall_perf, ncclAllToAll). Sizes 1M..256M (f4),
# ranks 2/4/8, 3 reps each. Configs:
#   shim-unfused (UK_CCL_DEV_BLOCKS=32)
#   shim-fused   (b32 + UK_CCL_FUSE_REDUCE_COPY/AG + LT16/TM8M/IB16)
#   native       (system NCCL)
# Median OOP busbw per (size, rank, config) is written to a CSV.
#
# Env: UCCL_ROOT / LOGDIR / ITERS / WARMUP / SHIM_LD / NATIVE_LD
#      A2A_SIZES (space list) / AR_EXTRA (extra -x flags for all configs)
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UCCL_ROOT=${UCCL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO_ROOT=$(cd "$UCCL_ROOT/../.." && pwd)
NCCTESTS_DIR=${NCCTESTS_DIR:-$REPO_ROOT/thirdparty/nccl-tests/build}
A2A_BIN=${A2A_BIN:-$UCCL_ROOT/bench/alltoall_perf}
LOGDIR=${LOGDIR:-/tmp/uk_full_matrix}
ITERS=${ITERS:-5}
WARMUP=${WARMUP:-1}
SIZES=${SIZES:-"1M 4M 16M 64M 256M"}
A2A_SIZES=${A2A_SIZES:-"1M 4M 16M 64M 256M"}
RANKS=${RANKS:-"2 4 8"}
SHIM_LD=${SHIM_LD:-$UCCL_ROOT/build/nccl/lib:/usr/local/lib}
NATIVE_LD=${NATIVE_LD:-/usr/lib/x86_64-linux-gnu:/usr/local/lib}
mkdir -p "$LOGDIR"
CSV="$LOGDIR/matrix.csv"
: > "$CSV"
echo "collective,size_bytes,ranks,config,rep,busbw,algbw,wrong" > "$CSV"

log() { echo "[$(date +%F_%T)] $*" >> "$LOGDIR/run.log"; }

gpu_ids() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    2>/dev/null | awk -F, -v n="$1" '{ if ($2+0 == 0 && c < n) { out = out (c ? "," : "") $1; ++c } }
      END { print out }'
}

wait_idle() {
  [ "${WAIT_IDLE:-1}" = "0" ] && return 0
  while true; do
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'int($1) != 0' | wc -l)
    [ "${busy:-1}" -eq 0 ] && return 0
    log "GPUs busy ($busy), waiting 60s"; sleep 60
  done
}

bytes_of() {
  case "$1" in
    1M) echo 1048576;; 4M) echo 4194304;; 16M) echo 16777216;;
    64M) echo 67108864;; 256M) echo 268435456;;
    *) echo "$1";;
  esac
}

run_ar() { # config ld np extra...
  local cfg=$1 ld=$2 np=$3; shift 3
  local gpus
  gpus=$(gpu_ids "$np")
  [ -n "$gpus" ] || { log "no idle gpus np=$np"; return 1; }
  env LD_LIBRARY_PATH="$ld" CUDA_VISIBLE_DEVICES="$gpus" \
    mpirun --bind-to none -np "$np" \
      -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_UNBIND=1 \
      "$@" "$NCCTESTS_DIR/all_reduce_perf" \
      -b 1M -e 256M -f 4 -g 1 -c 1 -n "$ITERS" -w "$WARMUP"
}

run_a2a() { # config ld np bytes extra...
  local cfg=$1 ld=$2 np=$3 bytes=$4; shift 4
  # alltoall_perf picks its GPU via --dev (default rank); with
  # CUDA_VISIBLE_DEVICES the ordinal must be the *relative* index, so use
  # the wrapper that passes ${OMPI_COMM_WORLD_LOCAL_RANK}. Runs on all
  # physical GPUs (no device filtering) so relative == physical index.
  local bin=${A2A_WRAP:-$A2A_BIN}
  env LD_LIBRARY_PATH="$ld" \
    mpirun --bind-to none -np "$np" \
      -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
      "$@" "$bin" --bytes="$bytes" --iters="$ITERS" --warmup="$WARMUP"
}

parse_ar() { # file np cfg rep
  awk -v np="$2" -v cfg="$3" -v rep="$4" '
    /^ *[0-9]+/ && NF >= 13 {
      printf "allreduce,%s,%s,%s,%s,%s,%s,%s\n", $1, np, cfg, rep, $8, $7, $9
    }' "$1" >> "$CSV"
}

parse_a2a() { # file np cfg rep bytes
  # [r0] ... algbw=X GB/s busbw=Y GB/s
  local line bw aw
  line=$(grep -a "\[r0\]" "$1" | grep -a "busbw=" | head -1)
  [ -n "$line" ] || { log "a2a parse miss $1"; return; }
  aw=$(echo "$line" | sed -E 's/.*algbw=([0-9.]+).*/\1/')
  bw=$(echo "$line" | sed -E 's/.*busbw=([0-9.]+).*/\1/')
  # header: collective,size_bytes,ranks,config,rep,busbw,algbw,wrong
  printf "alltoall,%s,%s,%s,%s,%s,%s,%s\n" "$5" "$2" "$3" "$4" "$bw" "$aw" 0 >> "$CSV"
}

log "start $(hostname) $(date)"
log "shim: $SHIM_LD  native: $NATIVE_LD"

for np in $RANKS; do
  for rep in 1 2 3; do
    wait_idle
    f="$LOGDIR/ar_unfused_np${np}_r${rep}.txt"
    run_ar unfused "$SHIM_LD" "$np" -x UK_CCL_DEV_BLOCKS=32 > "$f" 2>&1
    parse_ar "$f" "$np" unfused "$rep"
    wait_idle
    f="$LOGDIR/ar_fused_np${np}_r${rep}.txt"
    run_ar fused "$SHIM_LD" "$np" -x UK_CCL_DEV_BLOCKS=32 \
      -x UK_CCL_FUSE_REDUCE_COPY=1 -x UK_CCL_FUSE_AG_COPY=1 \
      -x UK_CCL_LARGE_TILES=16 -x UK_CCL_TILE_MIN_BYTES=8388608 \
      -x UK_CCL_IPC_BATCH=16 > "$f" 2>&1
    parse_ar "$f" "$np" fused "$rep"
    wait_idle
    f="$LOGDIR/ar_native_np${np}_r${rep}.txt"
    run_ar native "$NATIVE_LD" "$np" > "$f" 2>&1
    parse_ar "$f" "$np" native "$rep"
  done
done

for np in $RANKS; do
  for sz in $A2A_SIZES; do
    b=$(bytes_of "$sz")
    for rep in 1 2 3; do
      wait_idle
      f="$LOGDIR/a2a_shim_np${np}_${sz}_r${rep}.txt"
      run_a2a shim "$SHIM_LD" "$np" "$b" -x UK_CCL_DEV_BLOCKS=32 > "$f" 2>&1
      parse_a2a "$f" "$np" shim "$rep" "$b"
      wait_idle
      f="$LOGDIR/a2a_native_np${np}_${sz}_r${rep}.txt"
      run_a2a native "$NATIVE_LD" "$np" "$b" > "$f" 2>&1
      parse_a2a "$f" "$np" native "$rep" "$b"
    done
  done
done

log "done $(date)"
