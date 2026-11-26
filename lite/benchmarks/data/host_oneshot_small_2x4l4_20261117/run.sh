#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
record="$PWD/benchmarks/data/host_oneshot_small_2x4l4_20261117"
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
export BUILD_DIR="$PWD/build"
for round in 1 2 3; do
  case $round in
    1) modes=(baseline p2p host) ;;
    2) modes=(host baseline p2p) ;;
    3) modes=(p2p host baseline) ;;
  esac
  for mode in "${modes[@]}"; do
    log="$record/raw/${mode}_${round}.log"
    [[ ! -e "$log" ]]
    opts=()
    if [[ $mode == p2p ]]; then opts=(--ar-oneshot 1); fi
    if [[ $mode == host ]]; then opts=(--host-oneshot 1); fi
    timeout 120 bash scripts/run_multinode_hybrid.sh "${opts[@]}" \
      -b 1024 -e 262144 -w 20 -n 30 --check-each 1 -c 1 > "$log" 2>&1
    echo "Completed $mode round $round"
  done
done
# In-place reuse, non-power-of-two input, packet tails, and three-slot reuse.
for spec in 1024:262144 1536:262144 262144:262144 1048576:131200; do
  size=${spec%:*}; piece=${spec#*:}
  log="$record/raw/repeat_${size}_${piece}.log"
  [[ ! -e "$log" ]]
  timeout 120 bash scripts/run_multinode_hybrid.sh --host-oneshot 1 \
    --host-repeat-check 8 --host-piece-bytes "$piece" -b "$size" -e "$size" \
    -w 2 -n 3 --check-each 1 -c 1 > "$log" 2>&1
  echo "Completed host repeat/tail check $spec"
done
# The shared transport's original gather consumer must still work.
for op in allgather allreduce; do
  log="$record/raw/regression_${op}.log"
  [[ ! -e "$log" ]]
  timeout 120 bash scripts/run_multinode_hybrid.sh --op "$op" \
    -b 1048576 -e 4194304 -w 2 -n 3 --check-each 1 -c 1 > "$log" 2>&1
  echo "Completed original hybrid $op regression"
done
