#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/ag_dual_matched_20261115"
out=benchmarks/data/ag_dual_matched_2x4l4_20261115/geometry_control
mkdir -p "$out"
for round in 1 2 3; do
 modes=(ce matched sm matched-sm)
 if [[ $round == 2 ]]; then modes=(matched-sm sm matched ce); fi
 for mode in "${modes[@]}"; do
  printf 'geometry round=%s mode=%s\n' "$round" "$mode"
  timeout 300 bash scripts/run_multinode.sh multinode_allreduce --op allgather --ag-pipeline "$mode" --ag-first-mask 2 --ag-piece-bytes 4194304 --ag-depth 4 -B 16 -T 256 -b 1073741824 -e 1073741824 -w 5 -n 10 --check-each 1 -c 1 > "$out/${mode}_${round}.log" 2>&1
 done
done
