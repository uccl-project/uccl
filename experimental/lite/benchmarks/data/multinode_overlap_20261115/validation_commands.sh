#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$PWD/build/multinode_pull_20261115"
out=benchmarks/data/multinode_overlap_20261115/validated_checks
for mode in overlap pull; do
 for blocks in 1 4; do
  timeout 120 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter "--rs-$mode-local" 1 -B "$blocks" -T 256 -b 1536 -e 1572864 -w 3 -n 10 --check-each 1 > "$out/${mode}_b${blocks}_tails.log" 2>&1
  timeout 120 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter "--rs-$mode-local" 1 -B "$blocks" -T 256 -b 1024 -e 4194304 -w 5 -n 50 --check-each 0 > "$out/${mode}_b${blocks}_consecutive.log" 2>&1
 done
done
for op in allreduce allgather; do
 short=ar; if [[ $op == allgather ]]; then short=ag; fi
 timeout 120 bash scripts/run_multinode.sh multinode_allreduce --op "$op" "--${short}-auto" 1 -b 1024 -e 1048576 -w 2 -n 3 --check-each 1 > "$out/${op}_regression.log" 2>&1
done
