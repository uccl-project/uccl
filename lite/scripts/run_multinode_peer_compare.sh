#!/usr/bin/env bash
# Paired eight-rank AG/RS comparison. Build and stage the same executables on
# both hosts before running this script.
set -euo pipefail

: "${HOSTS:?Set HOSTS to two hosts with four GPU slots each}"
: "${SOCKET_IFNAME:?Set SOCKET_IFNAME to the MPI TCP bootstrap interface}"
if [[ -z "${NCCL_BASELINE_BIN:-}" ]]; then
  : "${NCCL_TESTS_BIN:?Set NCCL_TESTS_BIN or NCCL_BASELINE_BIN}"
fi
: "${NCCL_LIB:?Set NCCL_LIB to the NCCL lib directory on both hosts}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${OUTPUT_DIR:-$root/build/multinode-peer-comparison}"
min_bytes="${MIN_BYTES:-1024}"
max_bytes="${MAX_BYTES:-67108864}"
warmups="${WARMUPS:-5}"
iterations="${ITERATIONS:-10}"
repeats="${REPEATS:-3}"
timeout_seconds="${TIMEOUT_SECONDS:-180}"
operations="${OPERATIONS:-allgather reducescatter}"
timing_mode="${TIMING_MODE:-cuda-event}"
check_each="${CHECK_EACH:-0}"
auto_variants="${AUTO_VARIANTS:-0}"
case "$timing_mode" in
  cuda-event) gpu_timing=1 ;;
  host) gpu_timing=0 ;;
  *) printf 'TIMING_MODE must be cuda-event or host\n' >&2; exit 2 ;;
esac
mkdir -p "$output"

run_uccl() {
  local op="$1"
  local opts=(--op "$op" -b "$min_bytes" -e "$max_bytes" -w "$warmups"
              -n "$iterations" -c 1 --gpu-timing "$gpu_timing")
  if [[ "$check_each" == 1 ]]; then opts+=(--check-each 1); fi
  if [[ "$auto_variants" == 1 ]]; then
    case "$op" in
      allreduce) opts+=(--ar-auto 1) ;;
      allgather) opts+=(--ag-auto 1) ;;
      reducescatter)
        opts+=(--rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512) ;;
    esac
  else
    if [[ "$op" == allgather ]]; then
      opts+=(--ag-ring 1)
    elif [[ "$op" == reducescatter ]]; then
      opts+=(--rs-ring 1 --rs-direct-ring 1 --rs-piece-bytes 8388608
             -B 96 -T 512)
    fi
  fi
  timeout "${timeout_seconds}s" bash "$root/scripts/run_multinode.sh" \
    multinode_allreduce "${opts[@]}"
}

run_nccl() {
  local op="$1"
  local nccl_env=()
  if [[ -n "${NCCL_CONTROL_ALGO:-}" ]]; then
    nccl_env+=(-x "NCCL_ALGO=$NCCL_CONTROL_ALGO")
  fi
  if [[ -n "${NCCL_CONTROL_PROTO:-}" ]]; then
    nccl_env+=(-x "NCCL_PROTO=$NCCL_CONTROL_PROTO")
  fi
  if [[ -n "${NCCL_BASELINE_BIN:-}" ]]; then
    local library_path="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    timeout "${timeout_seconds}s" "${MPIRUN:-mpirun}" \
      --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
      --mca pml ob1 --mca btl self,vader,tcp \
      --mca btl_tcp_if_include "$SOCKET_IFNAME" \
      -x "LD_LIBRARY_PATH=$library_path" \
      -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" \
      -x "NCCL_IB_HCA=${HCA_DEVICES:-mlx5_0,mlx5_1}" \
      "${nccl_env[@]}" \
      "$NCCL_BASELINE_BIN" --op "$op" -b "$min_bytes" \
      -e "$max_bytes" -w "$warmups" -n "$iterations"
    return
  fi
  local test_name
  case "$op" in
    allgather) test_name=all_gather ;;
    reducescatter) test_name=reduce_scatter ;;
    allreduce) test_name=all_reduce ;;
  esac
  local binary="$NCCL_TESTS_BIN/${test_name}_perf_mpi"
  if [[ ! -x "$binary" ]]; then
    printf 'Missing MPI-enabled nccl-tests binary: %s\n' "$binary" >&2
    exit 1
  fi
  local library_path="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  local blocking_args=()
  if [[ -n "${NCCL_BLOCKING:-}" ]]; then
    blocking_args=(-z "$NCCL_BLOCKING")
  fi
  timeout "${timeout_seconds}s" "${MPIRUN:-mpirun}" \
    --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
    --mca pml ob1 --mca btl self,vader,tcp \
    --mca btl_tcp_if_include "$SOCKET_IFNAME" \
    -x "LD_LIBRARY_PATH=$library_path" \
    -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" \
    -x "NCCL_IB_HCA=${HCA_DEVICES:-mlx5_0,mlx5_1}" \
    "${nccl_env[@]}" \
    "$binary" -b "$min_bytes" -e "$max_bytes" -f 2 -g 1 -d float \
    -w "$warmups" -n "$iterations" -c "$iterations" -a 3 \
    "${blocking_args[@]}"
}

for op in $operations; do
  if [[ "$op" != allgather && "$op" != reducescatter &&
        "$op" != allreduce ]]; then
    printf 'Unknown operation: %s\n' "$op" >&2
    exit 2
  fi
  for ((repeat = 1; repeat <= repeats; ++repeat)); do
    uccl_log="$output/${op}_uccl_run${repeat}.txt"
    nccl_log="$output/${op}_nccl_run${repeat}.txt"
    if [[ -e "$uccl_log" || -e "$nccl_log" ]]; then
      printf 'Refusing to overwrite paired logs for %s run %d\n' "$op" \
        "$repeat" >&2
      exit 1
    fi
    if ((repeat % 2)); then
      run_uccl "$op" > "$uccl_log" 2>&1
      run_nccl "$op" > "$nccl_log" 2>&1
    else
      run_nccl "$op" > "$nccl_log" 2>&1
      run_uccl "$op" > "$uccl_log" 2>&1
    fi
  done
done
