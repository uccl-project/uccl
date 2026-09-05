#!/bin/bash
# run_g1.sh — G1 validation: concurrent collectives across CUDA streams
# (FSDP backward-prefetch shape: AllGather next-layer params +
# ReduceScatter current-layer grads).
#
# Cells:
#   placement S2/S8 (node5) and X16 (node5+node6)
#   W = 1M / 256M
#   scenario = fsdp2 shared / fsdp2 per-op / seqfsdp
#   syncK = 1 (host waits every batch) / 30 (fully pipelined)
#   lib = shim (SM-budget b per L40S tables) / native NCCL 2.31.2
# 3 reps each; output medians to LOGDIR/g1_medians.csv.
#
# Usage: bash bench/run_g1.sh  (runs on this machine; assumes both
# nodes reachable for X16).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOGDIR=${LOGDIR:-/tmp/g1_$(date +%m%d_%H%M%S)}
BIN=${BIN:-/tmp/stream_concurrent}
NODE5=${NODE5:-10.31.154.11}
NODE6=${NODE6:-10.31.154.12}
SHIM_LD=${SHIM_LD:-/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib}
REPS=${REPS:-3}
PLACEMENTS=${PLACEMENTS:-"S2 S8 X16"}
SCENARIOS=${SCENARIOS:-"fsdp2 shared fsdp2 per-op seqfsdp shared"}
SIZES=${SIZES:-"1048576 268435456"}
SYNCKS=${SYNCKS:-"1 30"}
mkdir -p "$LOGDIR"
CSV="$LOGDIR/g1_medians.csv"
RAW="$LOGDIR/raw.csv"
: > "$CSV"
: > "$RAW"

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOGDIR/run.log"; }

run_mpi() { # np hostspec envlist -- cmd...
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
  fi
}

# shim RS/AR worker blocks per placement+size (SM budget rule, L40S).
rs_b() { # placement bytes -> blocks
  case "$1:$2" in
    S2:*) echo 4;;
    S8:1048576) echo 2;;
    S8:*) echo 1;;
    X16:1048576) echo 2;;
    X16:*) echo 2;;
    *) echo 1;;
  esac
}

run_cell() { # placement scenario comm W K lib rep
  local placement=$1 scenario=$2 comm=$3 W=$4 K=$5 lib=$6 rep=$7
  local np host b extra=()
  case "$placement" in
    S2) np=2; host="";;
    S8) np=8; host="";;
    X16) np=16; host="$NODE5:8,$NODE6:8";;
  esac
  b=$(rs_b "$placement" "$W")
  if [ "$lib" = shim ]; then
    export LD_LIBRARY_PATH="$SHIM_LD"
    export UK_CCL_UNBIND=1
    export UK_CCL_DEV_BLOCKS="$b"
    extra=(-x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 -x UK_CCL_DEV_BLOCKS)
    if [ "$placement" = X16 ]; then
      export UK_CCL_RDMA_FUSED_MODE=proxy
      extra+=(-x UK_CCL_RDMA_FUSED_MODE)
    fi
  else
    unset LD_LIBRARY_PATH UK_CCL_UNBIND UK_CCL_DEV_BLOCKS UK_CCL_RDMA_FUSED_MODE
    extra=()
  fi
  local f="$LOGDIR/${placement}_${scenario}_${comm}_${W}_K${K}_${lib}_r${rep}.txt"
  run_mpi "$np" "$host" "${extra[@]}" -- \
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
  # parse wall_us / agg_busbw
  local wall bw
  wall=$(echo "$line" | sed -E 's/.*wall_us=([0-9.]+).*/\1/')
  bw=$(echo "$line" | sed -E 's/.*agg_busbw=([0-9.]+).*/\1/')
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$placement" "$scenario" "$comm" "$W" "$K" "$lib" "$rep" "$wall" "$bw" \
    >> "$RAW"
  log "ok $placement $scenario $comm $W K$K $lib r$rep wall=${wall}us bw=${bw}GB/s"
}

median() { # print median of column $1 from stdin rows
  sort -n | awk -v c="$1" '{a[NR]=$c} END {print a[int((NR+1)/2)]}'
}

log "start $(hostname) $(date) logdir=$LOGDIR"

SCENARIO_LIST=($SCENARIOS)
for placement in $PLACEMENTS; do
  for W in $SIZES; do
    for ((si = 0; si < ${#SCENARIO_LIST[@]}; si += 2)); do
      sc=${SCENARIO_LIST[si]}
      cm=${SCENARIO_LIST[si + 1]}
      for K in $SYNCKS; do
        for lib in shim native; do
          for rep in $(seq 1 "$REPS"); do
            run_cell "$placement" "$sc" "$cm" "$W" "$K" "$lib" "$rep" \
              || true
          done
          # median wall/bw over reps
          med_w=$(awk -F, -v p="$placement" -v s="$sc" -v c="$cm" -v w="$W" \
            -v k="$K" -v l="$lib" \
            '$1==p && $2==s && $3==c && $4==w && $5==k && $6==l {print $8}' \
            "$RAW" | median 1)
          med_b=$(awk -F, -v p="$placement" -v s="$sc" -v c="$cm" -v w="$W" \
            -v k="$K" -v l="$lib" \
            '$1==p && $2==s && $3==c && $4==w && $5==k && $6==l {print $9}' \
            "$RAW" | median 1)
          printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$placement" "$sc" "$cm" "$W" "$K" "$lib" "$med_w" "$med_b" \
            >> "$CSV"
          log "MEDIAN $placement $sc $cm $W K$K $lib wall=${med_w}us bw=${med_b}GB/s"
        done
      done
    done
  done
done

log "done $(date) csv=$CSV"
