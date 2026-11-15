#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
export BUILD_DIR="$PWD/build/ar_fix"
record="$PWD/benchmarks/data/ar_small_fix_2x4l4_20261115"
old=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/ar_latency_2x4l4_20261115
for round in 1 2 3; do
  for mode in native previous fixed; do
    log="$record/raw/${mode}_${round}.log"
    [[ ! -e "$log" ]]
    if [[ $mode == fixed ]]; then
      timeout 900 bash scripts/run_multinode_hybrid.sh --op allreduce -b 1024 -e 1073741824 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
    elif [[ $mode == previous ]]; then
      timeout 180 env BUILD_DIR="$old" bash scripts/run_multinode_hybrid.sh --op allreduce -b 1024 -e 1048576 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
    else
      timeout 180 env BUILD_DIR="$old" bash scripts/run_multinode.sh multinode_allreduce --op allreduce --ar-auto 1 -b 1024 -e 1048576 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
    fi
    echo "Completed $mode round $round"
  done
done
# Small-only setup, cutoff boundaries, tails, alternating algorithms and in-place reuse.
for size in 1024 1536 261632 262144 262656 524288 1048576 6291968 25165824; do
  timeout 180 bash scripts/run_multinode_hybrid.sh --op allreduce --host-repeat-check 3 \
    --host-piece-bytes 131200 -b "$size" -e "$size" -w 2 -n 3 --check-each 1 -c 1 > "$record/raw/edge_${size}.log" 2>&1
done
for op in allreduce allgather reducescatter; do
  opts=(--ar-auto 1)
  if [[ $op == allgather ]]; then opts=(--ag-auto 1); fi
  if [[ $op == reducescatter ]]; then opts=(--rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512); fi
  timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "${opts[@]}" \
    -b 1024 -e 67108864 -w 2 -n 3 --check-each 1 -c 1 > "$record/raw/regression_${op}.log" 2>&1
done
