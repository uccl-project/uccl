#!/bin/bash
# run_l40s_matrix.sh — L40S two-node shim vs native matrix.
#
# Same-node (node5 only): ranks 2/4/8, sizes 1M..256M.
# Cross-node (node5+node6): ranks 4 (2+2), 8 (4+4), 16 (8+8).
# Configs: shim (default blocks; cross-node adds RDMA fused proxy env)
# and native (system NCCL). AllReduce + AllToAll, 3 reps each.
#
# Env: UCCL_ROOT / LOGDIR / ITERS / WARMUP / SHIM_LD / NATIVE_LD
#      NODE5 / NODE6 / CUDA_LIBDIR
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UCCL_ROOT=${UCCL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO_ROOT=$(cd "$UCCL_ROOT/../.." && pwd)
NCCTESTS_DIR=${NCCTESTS_DIR:-$REPO_ROOT/thirdparty/nccl-tests/build}
A2A_BIN=${A2A_BIN:-$UCCL_ROOT/bench/alltoall_perf}
A2A_WRAP=${A2A_WRAP:-/tmp/a2a_rank.sh}
LOGDIR=${LOGDIR:-/tmp/uk_l40s_matrix}
ITERS=${ITERS:-5}
WARMUP=${WARMUP:-1}
SIZES=${SIZES:-"1M 4M 16M 64M 256M"}
A2A_SIZES=${A2A_SIZES:-"1M 4M 16M 64M 256M"}
LOCAL_RANKS=${LOCAL_RANKS:-"2 4 8"}
CROSS_RANKS=${CROSS_RANKS:-"4 8 16"}
NODE5=${NODE5:-10.31.154.11}
NODE6=${NODE6:-10.31.154.12}
SHIM_LD=${SHIM_LD:-/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib}
NATIVE_LD=${NATIVE_LD:-/usr/local/cuda/lib64:/usr/lib64}
mkdir -p "$LOGDIR"
CSV="$LOGDIR/matrix.csv"
: > "$CSV"
echo "collective,size_bytes,ranks,config,mode,rep,busbw,algbw,wrong" > "$CSV"

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOGDIR/run.log"; }

bytes_of() {
  case "$1" in
    1M) echo 1048576;; 4M) echo 4194304;; 16M) echo 16777216;;
    64M) echo 67108864;; 256M) echo 268435456;;
    *) echo "$1";;
  esac
}

run_mpi() { # np hostspec extra... -- cmd...
  local np=$1 host=$2; shift 2
  local -a args=()
  while [ "$1" != "--" ]; do args+=("$1"); shift; done
  shift
  if [ -n "$host" ]; then
    mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE \
      -np "$np" --host "$host" "${args[@]}" "$@"
  else
    mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE \
      -np "$np" "${args[@]}" "$@"
  fi
}

run_ar() { # config mode np hostspec ld extra...
  local cfg=$1 mode=$2 np=$3 host=$4 ld=$5; shift 5
  LD_LIBRARY_PATH="$ld" run_mpi "$np" "$host" \
    -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 "$@" -- \
    "$NCCTESTS_DIR/all_reduce_perf" \
    -b 1M -e 256M -f 4 -g 1 -c 1 -n "$ITERS" -w "$WARMUP"
}

run_a2a() { # config mode np hostspec ld bytes extra...
  local cfg=$1 mode=$2 np=$3 host=$4 ld=$5 bytes=$6; shift 6
  local bin=$A2A_BIN
  # Per-GPU device assignment: alltoall_perf defaults --dev=rank, which
  # breaks on multi-node runs (node-local ranks must map to 0..7). The
  # wrapper injects --dev=${OMPI_COMM_WORLD_LOCAL_RANK}. Same-node runs
  # also use it for consistency.
  if [ -n "$A2A_WRAP" ] && [ -x "$A2A_WRAP" ]; then bin=$A2A_WRAP; fi
  LD_LIBRARY_PATH="$ld" run_mpi "$np" "$host" \
    -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 "$@" -- \
    "$bin" --bytes="$bytes" --iters="$ITERS" --warmup="$WARMUP"
}

parse_ar() { # file np cfg mode rep
  awk -v np="$2" -v cfg="$3" -v mode="$4" -v rep="$5" '
    /^ *[0-9]+/ && NF >= 13 {
      printf "allreduce,%s,%s,%s,%s,%s,%s,%s,%s\n", $1, np, cfg, mode, rep, $8, $7, $9
    }' "$1" >> "$CSV"
}

parse_a2a() { # file np cfg mode rep bytes
  local line bw aw
  line=$(grep -a "\[r0\]" "$1" | grep -a "busbw=" | head -1)
  [ -n "$line" ] || { log "a2a parse miss $1"; return; }
  aw=$(echo "$line" | sed -E 's/.*algbw=([0-9.]+).*/\1/')
  bw=$(echo "$line" | sed -E 's/.*busbw=([0-9.]+).*/\1/')
  # header: collective,size_bytes,ranks,config,mode,rep,busbw,algbw,wrong
  printf "alltoall,%s,%s,%s,%s,%s,%s,%s,%s\n" "$6" "$2" "$3" "$4" "$5" "$bw" "$aw" 0 >> "$CSV"
}

log "start $(hostname) $(date)"
log "shim: $SHIM_LD  native: $NATIVE_LD"

# Same-node
for np in $LOCAL_RANKS; do
  for rep in 1 2 3; do
    f="$LOGDIR/ar_shim_local_np${np}_r${rep}.txt"
    run_ar shim local "$np" "" "$SHIM_LD" > "$f" 2>&1
    parse_ar "$f" "$np" shim local "$rep"
    f="$LOGDIR/ar_native_local_np${np}_r${rep}.txt"
    run_ar native local "$np" "" "$NATIVE_LD" > "$f" 2>&1
    parse_ar "$f" "$np" native local "$rep"
  done
done

for np in $LOCAL_RANKS; do
  for sz in $A2A_SIZES; do
    b=$(bytes_of "$sz")
    for rep in 1 2 3; do
      f="$LOGDIR/a2a_shim_local_np${np}_${sz}_r${rep}.txt"
      run_a2a shim local "$np" "" "$SHIM_LD" "$b" > "$f" 2>&1
      parse_a2a "$f" "$np" shim local "$rep" "$b"
      f="$LOGDIR/a2a_native_local_np${np}_${sz}_r${rep}.txt"
      run_a2a native local "$np" "" "$NATIVE_LD" "$b" > "$f" 2>&1
      parse_a2a "$f" "$np" native local "$rep" "$b"
    done
  done
done

# Cross-node
for np in $CROSS_RANKS; do
  half=$((np / 2))
  host="$NODE5:$half,$NODE6:$half"
  for rep in 1 2 3; do
    f="$LOGDIR/ar_shim_xnode_np${np}_r${rep}.txt"
    run_ar shim xnode "$np" "$host" "$SHIM_LD" \
      -x UK_CCL_RDMA_FUSED_MODE=proxy > "$f" 2>&1
    parse_ar "$f" "$np" shim xnode "$rep"
    f="$LOGDIR/ar_native_xnode_np${np}_r${rep}.txt"
    run_ar native xnode "$np" "$host" "$NATIVE_LD" > "$f" 2>&1
    parse_ar "$f" "$np" native xnode "$rep"
  done
done

for np in $CROSS_RANKS; do
  half=$((np / 2))
  host="$NODE5:$half,$NODE6:$half"
  for sz in $A2A_SIZES; do
    b=$(bytes_of "$sz")
    for rep in 1 2 3; do
      f="$LOGDIR/a2a_shim_xnode_np${np}_${sz}_r${rep}.txt"
      run_a2a shim xnode "$np" "$host" "$SHIM_LD" "$b" \
        -x UK_CCL_RDMA_FUSED_MODE=proxy > "$f" 2>&1
      parse_a2a "$f" "$np" shim xnode "$rep" "$b"
      f="$LOGDIR/a2a_native_xnode_np${np}_${sz}_r${rep}.txt"
      run_a2a native xnode "$np" "$host" "$NATIVE_LD" "$b" > "$f" 2>&1
      parse_a2a "$f" "$np" native xnode "$rep" "$b"
    done
  done
done

log "done $(date)"
