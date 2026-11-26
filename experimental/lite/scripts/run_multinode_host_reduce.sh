#!/usr/bin/env bash
# Explicit host reduction experiment; two GPUs per MPI rank.
set -euo pipefail
: "${HOSTS:?Set node0:4,node1:4}"
: "${SOCKET_IFNAME:?Set the bootstrap interface}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=, read -r host0 host1 <<< "$HOSTS"
exec "${MPIRUN:-mpirun}" --host "${host0%:*}:2,${host1%:*}:2" -np 4 \
  --map-by ppr:2:node --bind-to none --mca pml ucx --mca coll '^hcoll' \
  -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" -x NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
  "${BUILD_DIR:-$root/build}/uccl_lite_multinode_host_reduce" "$@"
