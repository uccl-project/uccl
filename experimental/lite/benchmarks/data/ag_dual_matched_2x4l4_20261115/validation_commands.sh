#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/ag_dual_matched_20261115"
out=benchmarks/data/ag_dual_matched_2x4l4_20261115/checks
mkdir -p "$out"
for mode in dual-sm dual-ce matched matched-sm; do
 opts=(--ag-pipeline "$mode")
 if [[ $mode == dual-* ]]; then opts=(--ag-pipeline "${mode#dual-}" --ag-dual-rdma 1); fi
 firsts=(2)
 if [[ $mode == matched* ]]; then firsts=(1 2); fi
 for first in "${firsts[@]}"; do
  timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op allgather "${opts[@]}" --ag-first-mask "$first" --ag-piece-bytes 65552 --ag-depth 3 -B 4 -T 256 -b 1536 -e 3145728 -w 3 -n 10 --check-each 1 -c 1 > "$out/${mode}_first${first}_tails.log" 2>&1
 done
 timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op allgather "${opts[@]}" --ag-piece-bytes 262144 --ag-depth 1 -B 4 -T 256 -b 1024 -e 16777216 -w 5 -n 50 --check-each 0 -c 1 > "$out/${mode}_consecutive.log" 2>&1
done
for op in allreduce reducescatter; do
 short=ar; if [[ $op == reducescatter ]]; then short=rs; fi
 timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "--${short}-auto" 1 -b 1024 -e 1048576 -w 3 -n 5 --check-each 1 -c 1 > "$out/${op}_regression.log" 2>&1
done
