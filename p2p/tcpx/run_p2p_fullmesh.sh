#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") server
  $(basename "$0") client <server_ip>

Description:
  Launches TCPX P2P perf runs for all 8 GPUs per node (16 GPUs total across 2 nodes).
  Each GPU is paired with the NIC on its PCIe root: {0,1}->eth1, {2,3}->eth2,
  {4,5}->eth3, {6,7}->eth4. A unique bootstrap port is derived from
  UCCL_TCPX_BOOTSTRAP_PORT_BASE + gpu_id to avoid collisions.

Environment overrides:
  GPU_LIST                       Space-separated GPU ids (default: 0 1 2 3 4 5 6 7)
  UCCL_TCPX_BOOTSTRAP_PORT_BASE  Base port (default: 20000)
  UCCL_TCPX_PERF_SIZE            Bytes per iteration (default: 67108864)
  UCCL_TCPX_PERF_ITERS           Iterations (default: 20)
  UCCL_TCPX_CHUNK_BYTES          Chunk size (default: 524288)
  LOG_DIR                        Output log directory (default: p2p/tcpx/logs)
USAGE
}

ROLE=${1:-}
if [[ -z "${ROLE}" ]]; then
  usage; exit 1;
fi
shift || true

case "${ROLE}" in
  server)
    ;;
  client)
    SERVER_IP=${1:-}
    [[ -z "${SERVER_IP}" ]] && { echo "[ERROR] Missing <server_ip>" >&2; usage; exit 1; }
    shift || true
    ;;
  *)
    usage; exit 1;
    ;;
esac

# Defaults (can be overridden by environment)
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
BOOTSTRAP_BASE=${UCCL_TCPX_BOOTSTRAP_PORT_BASE:-20000}
PERF_SIZE=${UCCL_TCPX_PERF_SIZE:-67108864}
PERF_ITERS=${UCCL_TCPX_PERF_ITERS:-20}
CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-524288}
LOG_DIR=${LOG_DIR:-"$(dirname "$0")/logs"}
mkdir -p "${LOG_DIR}"

map_gpu_to_iface() {
  local gpu=$1
  case ${gpu} in
    0|1) echo "eth1" ;;
    2|3) echo "eth2" ;;
    4|5) echo "eth3" ;;
    6|7) echo "eth4" ;;
    *) echo "" ;;
  esac
}

# Shared environment (mirrors bench_p2p.sh / run_nccl_test_tcpx.sh)
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_GPUDIRECTTCPX_CTRL_DEV="eth0"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ENV}

run_instance() {
  local role=$1 gpu=$2 iface=$3
  local timestamp
  timestamp=$(date +%Y%m%d_%H%M%S)
  local log_file="${LOG_DIR}/fullmesh_${role}_gpu${gpu}_${timestamp}.log"

  (
    set -euo pipefail
    export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${iface}"
    export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"
    export UCCL_TCPX_BOOTSTRAP_PORT_BASE="${BOOTSTRAP_BASE}"
    export UCCL_TCPX_NUM_CHANNELS=1
    export UCCL_TCPX_PERF_SIZE="${PERF_SIZE}"
    export UCCL_TCPX_PERF_ITERS="${PERF_ITERS}"
    export UCCL_TCPX_CHUNK_BYTES="${CHUNK_BYTES}"

    if [[ "${role}" == "server" ]]; then
      exec ./tests/test_tcpx_perf_multi server "${gpu}"
    else
      exec ./tests/test_tcpx_perf_multi client "${SERVER_IP}" "${gpu}"
    fi
  ) &>"${log_file}" &
}

for gpu in ${GPU_LIST}; do
  iface=$(map_gpu_to_iface "${gpu}")
  if [[ -z "${iface}" ]]; then
    echo "[WARN] No NIC mapping for GPU ${gpu}; skipping" >&2
    continue
  fi
  echo "[INFO] Launching ${ROLE} GPU ${gpu} on ${iface} (port base ${BOOTSTRAP_BASE})"
  run_instance "${ROLE}" "${gpu}" "${iface}"
  sleep 0.2
done

wait

echo "[INFO] All ${ROLE} processes completed. Logs in ${LOG_DIR}."
