#!/usr/bin/env bash
# Paired four-L4 FP32 AllReduce sweep. Both sides report host enqueue+completion per
# invocation and validate every warmup and timed output.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${OUTPUT_DIR:-$root/build/single-node-refresh}"
begin="${MIN_BYTES:-1024}"
end="${MAX_BYTES:-1073741824}"
timeout_seconds="${TIMEOUT_SECONDS:-3600}"
mkdir -p "$output"

run_uccl() {
  timeout "$timeout_seconds" "$root/build/uccl_lite_allreduce" \
    -b "$begin" -e "$end" -w 5 -n 10 -c 1
}
run_nccl() {
  timeout "$timeout_seconds" "${MPIRUN:-mpirun}" \
    --host localhost:4 -np 4 --map-by ppr:4:node --bind-to none \
    --mca pml ob1 --mca btl self,vader,tcp \
    "$root/build/uccl_lite_nccl_4l4_mpi_baseline" \
    --op allreduce -b "$begin" -e "$end" -w 5 -n 10
}

for repeat in 1 2 3; do
  uccl_log="$output/allreduce_uccl_run${repeat}.log"
  nccl_log="$output/allreduce_nccl_run${repeat}.log"
  if [[ -e "$uccl_log" || -e "$nccl_log" ]]; then
    printf 'Refusing to overwrite paired logs for AllReduce run %d\n' \
      "$repeat" >&2
    exit 1
  fi
  if ((repeat % 2)); then
    run_uccl > "$uccl_log" 2>&1
    run_nccl > "$nccl_log" 2>&1
  else
    run_nccl > "$nccl_log" 2>&1
    run_uccl > "$uccl_log" 2>&1
  fi
done
