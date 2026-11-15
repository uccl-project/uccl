#!/usr/bin/env bash
# Fixed-config comparisons of single/dual RDMA fan-out and matched P2P.
set -euo pipefail
: "${HOSTS:?Set two hosts with four slots each}"
: "${SOCKET_IFNAME:?Set the bootstrap interface}"
: "${BUILD_DIR:?Set identical staged binaries on both hosts}"
: "${OUTPUT_DIR:?Set a new output directory}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"
for round in 1 2 3; do
 modes=(auto fanout_sm fanout_ce dual_sm dual_ce matched_sm matched_ce)
 if [[ $round == 2 ]]; then modes=(matched_ce matched_sm dual_ce dual_sm fanout_ce fanout_sm auto); fi
 for mode in "${modes[@]}"; do
  log="$OUTPUT_DIR/${mode}_${round}.log"
  [[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
  opts=(--ag-piece-bytes 4194304 --ag-depth 4 -B 1 -T 256)
  case "$mode" in
   auto) opts=(--ag-auto 1) ;;
   fanout_sm) opts+=(--ag-pipeline sm) ;;
   fanout_ce) opts+=(--ag-pipeline ce) ;;
   dual_sm) opts+=(--ag-pipeline sm --ag-dual-rdma 1) ;;
   dual_ce) opts+=(--ag-pipeline ce --ag-dual-rdma 1) ;;
   matched_sm) opts+=(--ag-pipeline matched-sm --ag-first-mask 2 -B "${MATCHED_BLOCKS:-16}") ;;
   matched_ce) opts+=(--ag-pipeline matched --ag-first-mask 2 -B "${MATCHED_BLOCKS:-16}") ;;
  esac
  printf 'round=%s mode=%s\n' "$round" "$mode"
  timeout "${TIMEOUT_SECONDS:-900}" bash "$root/scripts/run_multinode.sh" multinode_allreduce \
   --op allgather "${opts[@]}" -b "${MIN_BYTES:-1024}" -e "${MAX_BYTES:-1073741824}" \
   -w "${WARMUPS:-10}" -n "${ITERATIONS:-10}" --check-each 1 --gpu-timing 0 -c 1 > "$log" 2>&1
 done
done
