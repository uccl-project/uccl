#!/bin/bash
# retest_g1.sh — rerun G1 cells that were contaminated by co-tenant
# driver resets / livelocks during the first full matrix (or that were
# never completed). Cells are appended to LOGDIR/g1_medians.csv in the
# same schema as run_g1.sh so results can be diffed/merged.
#
# Cell spec lines: PLACEMENT|SCENARIO|COMM|BYTES|K|LIBS
# LIBS = "shim native" or subset.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOGDIR=${LOGDIR:-/tmp/g1_retest}
BIN=${BIN:-/tmp/stream_concurrent}
NODE5=${NODE5:-10.31.154.11}
NODE6=${NODE6:-10.31.154.12}
SHIM_LD=${SHIM_LD:-/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib}
REPS=${REPS:-3}
mkdir -p "$LOGDIR"
RAW="$LOGDIR/raw.csv"
CSV="$LOGDIR/g1_medians.csv"
: > "$RAW"
: > "$CSV"

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOGDIR/run.log"; }

CELLS=${CELLS:-"X16|seqfsdp|shared|268435456|1|shim native
X16|seqfsdp|shared|268435456|30|shim native
X16|fsdp2|per-op|1048576|1|shim
X16|fsdp2|per-op|1048576|30|shim
X16|fsdp2|per-op|268435456|1|shim
X16|fsdp2|per-op|268435456|30|shim
X16|fsdp2|shared|268435456|1|shim
X16|fsdp2|shared|268435456|30|shim
S8|fsdp2|per-op|1048576|1|shim
S8|fsdp2|per-op|1048576|30|shim
S8|seqfsdp|shared|1048576|30|shim"}

rs_b() { # placement bytes
  case "$1:$2" in
    S2:*) echo 4;;
    S4:1048576) echo 4;;
    S4:*) echo 2;;
    S8:1048576) echo 2;;
    S8:*) echo 1;;
    X8:1048576) echo 2;;
    X8:*) echo 1;;
    X16:*) echo 2;;
    *) echo 1;;
  esac
}

run_mpi() { # np hostspec xargs... -- cmd...
  local np=$1 host=$2; shift 2
  local -a xargs=()
  while [ "$1" != "--" ]; do xargs+=("$1"); shift; done
  shift
  if [ -n "$host" ]; then
    mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE \
      -np "$np" --host "$host" "${xargs[@]}" "$@"
  else
    mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE \
      -np "$np" "${xargs[@]}" "$@"
  fi < /dev/null
}

run_cell() {
  local placement=$1 scenario=$2 comm=$3 W=$4 K=$5 lib=$6 rep=$7
  local np host
  case "$placement" in
    S2) np=2; host="";;
    S4) np=4; host="";;
    S8) np=8; host="";;
    X8) np=8; host="$NODE5:4,$NODE6:4";;
    X16) np=16; host="$NODE5:8,$NODE6:8";;
  esac
  local -a xargs=()
  if [ "$lib" = shim ]; then
    export LD_LIBRARY_PATH="$SHIM_LD"
    export UK_CCL_UNBIND=1
    export UK_CCL_DEV_BLOCKS=$(rs_b "$placement" "$W")
    xargs=(-x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 -x UK_CCL_DEV_BLOCKS)
    if [ "$placement" = X8 ] || [ "$placement" = X16 ]; then
      export UK_CCL_RDMA_FUSED_MODE=proxy
      xargs+=(-x UK_CCL_RDMA_FUSED_MODE)
    fi
  else
    unset LD_LIBRARY_PATH UK_CCL_UNBIND UK_CCL_DEV_BLOCKS UK_CCL_RDMA_FUSED_MODE
    xargs=()
  fi
  local f="$LOGDIR/${placement}_${scenario}_${comm}_${W}_K${K}_${lib}_r${rep}.txt"
  run_mpi "$np" "$host" "${xargs[@]}" -- \
    "$BIN" --scenario "$scenario" --comm-mode "$comm" \
      --layer-bytes "$W" --iters 30 --warmup 5 --sync-every "$K" \
    > "$f" 2>&1
  local rc=$?
  local line
  line=$(grep -a "\[r0\]" "$f" | head -1)
  if [ $rc -ne 0 ] || [ -z "$line" ]; then
    log "FAIL $placement $scenario $comm $W K$K $lib r$rep rc=$rc"
    return 1
  fi
  local wall bw
  wall=$(echo "$line" | sed -E 's/.*wall_us=([0-9.]+).*/\1/')
  bw=$(echo "$line" | sed -E 's/.*agg_busbw=([0-9.]+).*/\1/')
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$placement" "$scenario" "$comm" "$W" "$K" "$lib" "$rep" "$wall" "$bw" \
    >> "$RAW"
  log "ok $placement $scenario $comm $W K$K $lib r$rep wall=${wall}us bw=${bw}GB/s"
}

median() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

log "start $(hostname) $(date) logdir=$LOGDIR"
while IFS='|' read -r placement scenario comm W K libs; do
  [ -z "$placement" ] && continue
  for lib in $libs; do
    ok=0
    for rep in $(seq 1 "$REPS"); do
      if run_cell "$placement" "$scenario" "$comm" "$W" "$K" "$lib" "$rep"; then
        ok=$((ok + 1))
      fi
    done
    if [ "$ok" -gt 0 ]; then
      med_w=$(awk -F, -v p="$placement" -v s="$scenario" -v c="$comm" \
        -v w="$W" -v k="$K" -v l="$lib" \
        '$1==p && $2==s && $3==c && $4==w && $5==k && $6==l {print $8}' \
        "$RAW" | median)
      med_b=$(awk -F, -v p="$placement" -v s="$scenario" -v c="$comm" \
        -v w="$W" -v k="$K" -v l="$lib" \
        '$1==p && $2==s && $3==c && $4==w && $5==k && $6==l {print $9}' \
        "$RAW" | median)
      printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$placement" "$scenario" "$comm" "$W" "$K" "$lib" "$med_w" "$med_b" \
        >> "$CSV"
      log "MEDIAN $placement $scenario $comm $W K$K $lib wall=${med_w}us bw=${med_b}GB/s"
    fi
  done
done <<< "$CELLS"
log "done $(date) csv=$CSV"
