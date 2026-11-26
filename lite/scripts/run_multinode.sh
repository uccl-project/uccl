#!/usr/bin/env bash
# Open MPI launcher for two nodes with four GPUs each.
set -euo pipefail

if [[ $# -lt 1 || "$1" == --help ]]; then
  printf '%s\n' \
    'Usage: HOSTS=node0:4,node1:4 bash scripts/run_multinode.sh BACKEND [options]' \
    'Backends: multinode_allreduce, multinode_cpu_router,' \
    '          host_allreduce, host_hierarchical_allreduce' \
    'multinode_allreduce accepts --op allreduce|allgather|reducescatter.' \
    'Optional: MPIRUN, BUILD_DIR, SOCKET_IFNAME, HCA_DEVICES, IBV_MODE.'
  exit 0
fi

backend="$1"
shift
: "${HOSTS:?Set HOSTS to two hosts with four GPU slots each, e.g. node0:4,node1:4}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${BUILD_DIR:-$root/build}/uccl_lite_$backend"
args=(--host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none)
args+=(-x PATH)
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then args+=(-x LD_LIBRARY_PATH); fi

case "$backend" in
  multinode_allreduce)
    # MPI bootstraps the ranks; MSCCL++ owns the IB data path.
    args+=(--mca pml ob1 --mca btl self,vader,tcp)
    if [[ -n "${SOCKET_IFNAME:-}" ]]; then
      args+=(--mca btl_tcp_if_include "$SOCKET_IFNAME")
    fi
    ;;
  multinode_cpu_router|host_allreduce|host_hierarchical_allreduce)
    args+=(--mca pml ucx)
    ;;
  *) printf 'Unknown backend: %s\n' "$backend" >&2; exit 2 ;;
esac

if [[ -n "${SOCKET_IFNAME:-}" ]]; then
  args+=(-x "MSCCLPP_SOCKET_IFNAME=$SOCKET_IFNAME")
fi
args+=(-x "MSCCLPP_HCA_DEVICES=${HCA_DEVICES:-mlx5_0,mlx5_1}")
args+=(-x "MSCCLPP_IBV_MODE=${IBV_MODE:-host-no-atomic}")
if [[ ! -x "$binary" ]]; then
  printf 'Missing executable: %s\nBuild with make %s first.\n' "$binary" "$backend" >&2
  exit 1
fi
exec "${MPIRUN:-mpirun}" "${args[@]}" "$binary" "$@"
