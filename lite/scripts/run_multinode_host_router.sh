#!/usr/bin/env bash
# Paired end-to-end host-router candidates, their four-process NCCL controls,
# and the existing eight-process GPU policy. Binaries must be staged identically.
set -euo pipefail
: "${HOSTS:?Set two hosts with four GPU slots each}"
: "${SOCKET_IFNAME:?Set bootstrap interface}"
: "${BUILD_DIR:?Set the host-binary directory on both hosts}"
: "${GPU_BUILD_DIR:?Set the GPU-binary directory on both hosts}"
: "${OUTPUT_DIR:?Set a new result directory}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=, read -r host0 host1 <<< "$HOSTS"
groups="${host0%:*}:2,${host1%:*}:2"
mkdir -p "$OUTPUT_DIR"
for round in 1 2 3; do
 modes=(gpu legacy early shared shared_ready shared_rdma)
 if [[ $round == 2 ]]; then modes=(shared_rdma shared_ready shared early legacy gpu); fi
 for mode in "${modes[@]}"; do
  log="$OUTPUT_DIR/${mode}_${round}.log"
  [[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
  echo "round=$round mode=$mode"
  if [[ $mode == gpu ]]; then
   BUILD_DIR="$GPU_BUILD_DIR" timeout 900 bash "$root/scripts/run_multinode.sh" multinode_allreduce --op allgather --ag-auto 1 \
    -b 1048576 -e 1073741824 -w 10 -n 10 -c 1 --check-each 1 --gpu-timing 0 > "$log" 2>&1
  else
   binary="$BUILD_DIR/uccl_lite_multinode_host_allgather"
   opts=(-k 262144 --download sm --upload dma --local-input self --early-consume 1 --gpu-workers 0 --progress steps --local-exchange copy)
   case "$mode" in
    legacy) opts=(-k 1048576 --download sm --upload dma --local-input host --early-consume 0 --gpu-workers 0 --progress steps --local-exchange copy) ;;
    early) ;;
    shared) opts+=(--local-exchange shared) ;;
    shared_ready) opts+=(--local-exchange shared --progress ready) ;;
    shared_rdma) opts+=(--local-exchange shared --network rdma); binary="$BUILD_DIR/uccl_lite_multinode_host_rdma_allgather" ;;
   esac
   timeout 900 "${MPIRUN:-mpirun}" --host "$groups" -np 4 --map-by ppr:2:node --bind-to none --mca pml ucx --mca coll '^hcoll' \
    -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" -x NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
    -x "MSCCLPP_SOCKET_IFNAME=$SOCKET_IFNAME" -x MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1 \
    "$binary" -b 1048576 -e 1073741824 -w 10 -n 10 -r 1 --nccl 1 "${opts[@]}" > "$log" 2>&1
  fi
 done
done
