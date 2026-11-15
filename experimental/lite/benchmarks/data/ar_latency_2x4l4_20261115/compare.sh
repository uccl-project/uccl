#!/usr/bin/env bash
# Rerun UCCL Lite only; reuse the recorded NCCL samples.
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
export BUILD_DIR="$PWD/build/ar_latency_2x4l4_20261115"
record="$PWD/benchmarks/data/ar_latency_2x4l4_20261115"
for round in 1 2 3; do
  log="$record/raw/hybrid_${round}.log"
  [[ ! -e "$log" ]]
  timeout 900 bash scripts/run_multinode_hybrid.sh --op allreduce \
    -b 1024 -e 1073741824 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
done
