#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
record="$PWD/benchmarks/data/oneshot_small_2x4l4_20261117"
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
export BUILD_DIR="$PWD/build"
for round in 1 2 3; do
  modes=(baseline oneshot)
  if [[ $round == 2 ]]; then modes=(oneshot baseline); fi
  for mode in "${modes[@]}"; do
    log="$record/raw/${mode}_${round}.log"
    [[ ! -e "$log" ]]
    opts=(--ar-serial-local 1)
    if [[ $mode == oneshot ]]; then opts=(--ar-oneshot 1); fi
    timeout 120 bash scripts/run_multinode.sh multinode_allreduce \
      "${opts[@]}" -b 1024 -e 262144 -w 20 -n 30 --check-each 1 -c 1 \
      > "$log" 2>&1
    echo "Completed $mode round $round"
  done
done
# Exercise the candidate's declared upper size bound separately from the
# <=256-KiB small-message baseline comparison.
log="$record/raw/upper_bound_check.log"
[[ ! -e "$log" ]]
timeout 120 bash scripts/run_multinode.sh multinode_allreduce \
  --ar-oneshot 1 -b 524288 -e 1048576 -w 2 -n 3 --check-each 1 -c 1 \
  > "$log" 2>&1
# Existing full-output check for repeated in-place calls without input refill.
# --host-gather selects the small-message GPU path at all these sizes.
for mode in baseline oneshot; do
  for size in 1024 1536 262144; do
    log="$record/raw/repeat_${mode}_${size}.log"
    [[ ! -e "$log" ]]
    opts=(--host-gather 1)
    if [[ $mode == oneshot ]]; then opts+=(--ar-oneshot 1); fi
    timeout 120 bash scripts/run_multinode.sh multinode_allreduce \
      "${opts[@]}" --host-repeat-check 8 -b "$size" -e "$size" \
      -w 2 -n 3 --check-each 1 -c 1 > "$log" 2>&1
    echo "Completed repeated in-place $mode $size"
  done
done
