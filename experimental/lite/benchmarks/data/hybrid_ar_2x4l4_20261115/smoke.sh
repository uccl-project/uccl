#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/hybrid_ar_20261115"
for op in allgather allreduce; do
 timeout 180 bash scripts/run_multinode_hybrid.sh --op "$op" -b 1024 -e 67108864 -w 2 -n 3 --check-each 1 -c 1 > "$BUILD_DIR/smoke_${op}.log" 2>&1
 done
