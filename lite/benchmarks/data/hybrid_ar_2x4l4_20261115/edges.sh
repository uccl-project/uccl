#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/hybrid_ar_20261115"
for op in allgather allreduce; do
 opts=(); if [[ $op == allreduce ]]; then opts+=(--host-repeat-check 3); fi
 timeout 180 bash scripts/run_multinode_hybrid.sh --op "$op" --host-piece-bytes 131200 -b 6291968 -e 6291968 -w 5 -n 20 --check-each 1 -c 1 "${opts[@]}" > "$BUILD_DIR/tail_${op}.log" 2>&1
 timeout 180 bash scripts/run_multinode_hybrid.sh --op "$op" --host-piece-bytes 128 -b 1024 -e 32768 -w 2 -n 3 --check-each 1 -c 1 "${opts[@]}" > "$BUILD_DIR/fine_${op}.log" 2>&1
 done
# Regression of ordinary GPU paths: the new host allocation remains opt-in.
for op in allgather reducescatter allreduce; do
 opts=(--ar-auto 1)
 if [[ $op == allgather ]]; then opts=(--ag-auto 1); fi
 if [[ $op == reducescatter ]]; then opts=(--rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512); fi
 timeout 180 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "${opts[@]}" -b 1024 -e 67108864 -w 2 -n 3 --check-each 1 -c 1 > "$BUILD_DIR/regression_${op}.log" 2>&1
 done
