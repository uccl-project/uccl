#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/ag_pipeline_20261115"
out=benchmarks/data/ag_pipeline_2x4l4_20261115/checks
mkdir -p "$out"
for mode in sm ce ce-pair; do
 for blocks in 1 4; do
  timeout 150 bash scripts/run_multinode.sh multinode_allreduce --op allgather --ag-pipeline "$mode" --ag-piece-bytes 65552 --ag-depth 3 -B "$blocks" -T 256 -b 1536 -e 3145728 -w 3 -n 10 --check-each 1 -c 1 > "$out/${mode}_b${blocks}_tails.log" 2>&1
 done
 for depth in 1 4; do
  timeout 150 bash scripts/run_multinode.sh multinode_allreduce --op allgather --ag-pipeline "$mode" --ag-piece-bytes 262144 --ag-depth "$depth" -B 1 -T 256 -b 1024 -e 16777216 -w 5 -n 50 --check-each 0 -c 1 > "$out/${mode}_d${depth}_consecutive.log" 2>&1
 done
done
for op in allreduce reducescatter; do
 short=ar; if [[ $op == reducescatter ]]; then short=rs; fi
 timeout 150 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "--${short}-auto" 1 -b 1024 -e 1048576 -w 3 -n 5 --check-each 1 -c 1 > "$out/${op}_regression.log" 2>&1
done
for case in wrong_op bad_alignment bad_depth too_many_pieces; do
 opts=(--op allgather --ag-pipeline ce -b 1024 -e 1048576)
 case "$case" in
 wrong_op) opts+=(--op allreduce) ;;
 bad_alignment) opts+=(--ag-piece-bytes 513) ;;
 bad_depth) opts+=(--ag-depth 0) ;;
 too_many_pieces) opts+=(--ag-piece-bytes 512) ;;
 esac
 set +e
 timeout 20 "$MPIRUN" --oversubscribe -np 8 "$BUILD_DIR/uccl_lite_multinode_allreduce" "${opts[@]}" > "$out/${case}.log" 2>&1
 code=$?
 set -e
 [[ $code -eq 2 ]] || { echo "Expected validation exit 2, got $code for $case" >&2; exit 1; }
done
