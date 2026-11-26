#!/usr/bin/env bash
# Explicit eight-rank host-packet AG / GPU-RS + host-AG AllReduce experiment.
set -euo pipefail
: "${HOSTS:?Set two hosts with four GPU slots each}"
: "${SOCKET_IFNAME:?Set the bootstrap interface}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${MPIRUN:-mpirun}" --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
  --mca pml ucx --mca coll '^hcoll' -x PATH \
  -x "MSCCLPP_SOCKET_IFNAME=$SOCKET_IFNAME" -x MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1 \
  -x MSCCLPP_IBV_MODE=host-no-atomic \
  "${BUILD_DIR:-$root/build}/uccl_lite_multinode_allreduce" --host-gather 1 "$@"
