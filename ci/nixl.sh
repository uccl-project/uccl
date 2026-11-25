#!/usr/bin/env bash
# ASIO coordinates peers directly: no NFS, shared prefix or etcd daemon.
set -euo pipefail
prefix="$CI_REPO/.ci-nixl"
export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$CI_DEPS/abseil-d7aaad83/lib:$LD_LIBRARY_PATH"
export NIXL_PLUGIN_DIR="$prefix/lib/plugins"
export UCCL_P2P_TRANSPORT=${1:?rdma or nccl required}
export UCCL_P2P_DISABLE_IPC=${2:-0}
export NCCL_IB_DISABLE=1
args=(--runtime_type ASIO --asio_address "$MASTER_ADDR" --asio_port "$MASTER_PORT"
  --backend UCCL --initiator_seg_type DRAM --target_seg_type DRAM
  --total_buffer_size 80000000 --start_block_size 16384 --max_block_size 16384
  --start_batch_size 4 --max_batch_size 4 --num_iter 16 --warmup_iter 4 --check_consistency)
if [[ $CI_NODES == 1 ]]; then
  nixlbench "${args[@]}" & first=$!
  nixlbench "${args[@]}" & second=$!
  status=0
  wait "$first" || status=1
  wait "$second" || status=1
  exit "$status"
else
  # ASIO's client has no startup retry. Retry only its pre-handshake refusal,
  # not transfer errors or consistency failures.
  if [[ $CI_NODE == 0 ]]; then exec nixlbench "${args[@]}"; fi
  for attempt in {1..30}; do
    log="$CI_ARTIFACTS/$CI_TEST_ID.connect-$attempt.txt"
    if nixlbench "${args[@]}" > "$log" 2>&1; then cat "$log"; exit 0; else status=$?; fi
    cat "$log"
    if ! grep -q 'ASIO Connection failed: Connection refused' "$log"; then exit "$status"; fi
    sleep 1
  done
  exit 1
fi
