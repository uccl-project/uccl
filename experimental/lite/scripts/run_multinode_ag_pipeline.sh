#!/usr/bin/env bash
# Paired, checked AllGather comparison. Stage identical binaries on both hosts.
set -euo pipefail
: "${HOSTS:?Set two hosts with four GPU slots each}"
: "${SOCKET_IFNAME:?Set the bootstrap interface}"
: "${BUILD_DIR:?Set the same absolute binary directory on both hosts}"
: "${OUTPUT_DIR:?Set a new output directory}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"
read -r -a requested_modes <<< "${MODES:-serial ring flat sm4096 ce1024 ce4096 pair4096 nccl_default nccl_simple}"
for mode in "${requested_modes[@]}"; do
  case "$mode" in
    serial|ring|flat|sm4096|ce1024|ce4096|pair4096|nccl_default|nccl_simple) ;;
    *) echo "Unknown comparison mode: $mode" >&2; exit 2 ;;
  esac
done
for round in 1 2 3; do
  modes=("${requested_modes[@]}")
  if [[ $round == 2 ]]; then
    modes=()
    for ((i=${#requested_modes[@]}-1; i>=0; --i)); do
      modes+=("${requested_modes[i]}")
    done
  fi
  for mode in "${modes[@]}"; do
    log="$OUTPUT_DIR/${mode}_${round}.log"
    [[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
    printf "round=%s mode=%s\n" "$round" "$mode"
    common=(--op allgather -b "${MIN_BYTES:-1024}" -e "${MAX_BYTES:-67108864}" -w 20 -n 20)
    if [[ $mode == nccl_* ]]; then
      protocol=()
      if [[ $mode == nccl_simple ]]; then protocol=(-x NCCL_ALGO=Ring -x NCCL_PROTO=Simple); fi
      timeout "${TIMEOUT_SECONDS:-240}" env -u NCCL_ALGO -u NCCL_PROTO "${MPIRUN:-mpirun}" \
        --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
        --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include "$SOCKET_IFNAME" \
        -x "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" -x "NCCL_IB_HCA=${HCA_DEVICES:-mlx5_0,mlx5_1}" \
        "${protocol[@]}" "$BUILD_DIR/uccl_lite_nccl_2x4l4_baseline" "${common[@]}" > "$log" 2>&1
    else
      case "$mode" in
        serial) opts=(--ag-serial-local 1) ;;
        ring) opts=(--ag-ring 1) ;;
        flat) opts=() ;;
        sm4096) opts=(--ag-pipeline sm --ag-piece-bytes 4194304 --ag-depth 4 -B 1 -T 256) ;;
        ce1024) opts=(--ag-pipeline ce --ag-piece-bytes 1048576 --ag-depth 4 -B 1 -T 256) ;;
        ce4096) opts=(--ag-pipeline ce --ag-piece-bytes 4194304 --ag-depth 4 -B 1 -T 256) ;;
        pair4096) opts=(--ag-pipeline ce-pair --ag-piece-bytes 4194304 --ag-depth 4 -B 1 -T 256) ;;
      esac
      timeout "${TIMEOUT_SECONDS:-240}" bash "$root/scripts/run_multinode.sh" multinode_allreduce \
        "${common[@]}" "${opts[@]}" --gpu-timing 0 --check-each 1 -c 1 > "$log" 2>&1
    fi
  done
done
