#!/usr/bin/env bash
# Explicit two-host/four-rank correctness check for the network shared library.
set -euo pipefail
: "${HOSTS:?Set HOSTS=node0:4,node1:4}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suffix=""
if [[ ${1:-} == --static ]]; then suffix=_static; shift; fi
if [[ $# != 0 ]]; then echo 'Usage: run_multinode_library.sh [--static]' >&2; exit 2; fi
binary="${BUILD_DIR:-$root/build}/multinode_communicator_check$suffix"
args=(--host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none
      --mca pml ob1 --mca btl self,vader,tcp -x PATH)
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then args+=(-x LD_LIBRARY_PATH); fi
if [[ -n ${SOCKET_IFNAME:-} ]]; then
  args+=(--mca btl_tcp_if_include "$SOCKET_IFNAME"
         -x "MSCCLPP_SOCKET_IFNAME=$SOCKET_IFNAME")
fi
args+=(-x "MSCCLPP_HCA_DEVICES=${HCA_DEVICES:-mlx5_0,mlx5_1}"
       -x "MSCCLPP_IBV_MODE=${IBV_MODE:-host-no-atomic}")
exec timeout "${TIMEOUT_SECONDS:-180}s" "${MPIRUN:-mpirun}" "${args[@]}" "$binary"
