#!/usr/bin/env bash
# Explicit experimental RS comparison; stage matching binaries on both hosts.
set -euo pipefail
: "${HOSTS:?Set two hosts with four slots each}"
: "${SOCKET_IFNAME:?Set bootstrap interface}"
: "${BUILD_DIR:?Set matching binary directory on both hosts}"
: "${OUTPUT_DIR:?Set a new results directory}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"
for round in 1 2 3; do
  modes=(serial auto overlap nccl_default nccl_simple)
  if [[ $round == 2 ]]; then modes=(nccl_simple nccl_default overlap auto serial); fi
  for mode in "${modes[@]}"; do
    log="$OUTPUT_DIR/${mode}_${round}.log"
    [[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
    opts=(--op reducescatter -b "${MIN_BYTES:-1024}" -e "${MAX_BYTES:-4194304}" -w 20 -n 20)
    if [[ $mode == nccl_* ]]; then
      protocol=()
      if [[ $mode == nccl_simple ]]; then protocol=(-x NCCL_ALGO=Ring -x NCCL_PROTO=Simple); fi
      timeout 180 "${MPIRUN:-mpirun}" --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
        --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include "$SOCKET_IFNAME" \
        -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" -x "NCCL_IB_HCA=mlx5_0,mlx5_1" \
        "${protocol[@]}" \
        "$BUILD_DIR/uccl_lite_nccl_2x4l4_baseline" "${opts[@]}" > "$log" 2>&1
    else
      local_opts=("--rs-${mode}-local" 1 -B "${LOCAL_BLOCKS:-1}" -T 256)
      if [[ $mode == auto ]]; then
        local_opts=(--rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512)
      fi
      timeout 180 bash "$root/scripts/run_multinode.sh" multinode_allreduce \
        "${opts[@]}" "${local_opts[@]}" \
        --gpu-timing 0 --check-each 1 -c 1 > "$log" 2>&1
    fi
  done
done
