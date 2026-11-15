#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
record="$PWD/benchmarks/data/rs_direct_medium_2x4l4_20261115"
for round in 1 2 3; do
  modes=(baseline candidate)
  if [[ $round == 2 ]]; then modes=(candidate baseline); fi
  for mode in "${modes[@]}"; do
    export BUILD_DIR="$PWD/build/ar_fix"
    if [[ $mode == candidate ]]; then export BUILD_DIR="$PWD/build/rs_direct_medium"; fi
    log="$record/raw/${mode}_${round}.log"; [[ ! -e "$log" ]]
    timeout 900 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
      --rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512 \
      -b 1024 -e 1073741824 -w 10 -n 10 --check-each 1 -c 1 > "$log" 2>&1
    echo "Completed $mode round $round"
  done
done
export BUILD_DIR="$PWD/build/rs_direct_medium"
for size in 8388096 8388608 8389120 12583424 16776704 16777216 16777728 25165824; do
  timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
    --rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512 \
    -b "$size" -e "$size" -w 2 -n 10 --check-each 1 -c 1 > "$record/raw/edge_${size}.log" 2>&1
done
timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
  --rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512 \
  -b 8389120 -e 8389120 -w 5 -n 100 --check-each 0 -c 1 > "$record/raw/repeated.log" 2>&1
for op in allreduce allgather; do
  opts=(--ar-auto 1)
  if [[ $op == allgather ]]; then opts=(--ag-auto 1); fi
  timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "${opts[@]}" \
    -b 1024 -e 67108864 -w 2 -n 3 --check-each 1 -c 1 > "$record/raw/regression_${op}.log" 2>&1
done
timeout 180 bash scripts/run_multinode_hybrid.sh --op allreduce --host-repeat-check 3 \
  -b 1024 -e 67108864 -w 2 -n 3 --check-each 1 -c 1 > "$record/raw/regression_hybrid.log" 2>&1
