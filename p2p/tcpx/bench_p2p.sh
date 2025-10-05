#!/usr/bin/env bash
set -euo pipefail

# TCPX P2P benchmark harness
# - Runs server/client roles
# - Configures multi-NIC (default eth1,eth2,eth3,eth4)
# - Optional UNIX client flow steering (requires dp-manager)
# - Parses logs and reports steady-state BW (skips warmup iters)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") server <gpu_id> [options]
  $(basename "$0") client <server_ip> <gpu_id> [options]

Options:
  --ifaces=LIST           Comma-separated NIC list (default: eth1,eth2,eth3,eth4)
  --ctrl=DEV              Control NIC (default: eth1)
  --size=BYTES            Total bytes per iter (default: 67108864 = 64MB)
  --iters=N               Iterations (default: 20)
  --chunk=BYTES           Chunk bytes (default: 524288 = 512KB)
  --nsocks=N              NCCL_NSOCKS_PERTHREAD (default: 4)
  --nthreads=N            NCCL_SOCKET_NTHREADS (default: 1)
  --unix-prefix=PATH      Enable UNIX client flow steering with prefix PATH (requires dp-manager)
  --no-unix               Disable UNIX client flow steering (default: off)
  --impl=kernel|d2d       Unpack implementation (default: kernel)
  --host-recv             Server receive on host (debug) (default: off)
  --skip-warmup=N         Skip first N iter(s) in steady-state stats (default: 1)

Examples:
  # Node1 (Server: 10.65.74.150)
  ./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --unix-prefix=/tmp/uccl_perf
  # Node2 (Client: 10.64.113.77)
  ./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --unix-prefix=/tmp/uccl_perf
EOF
}

ROLE=${1:-}
if [[ -z "${ROLE}" ]]; then usage; exit 1; fi
shift || true

SERVER_IP=""
GPU_ID=""

case "${ROLE}" in
  server)
    GPU_ID=${1:-}
    [[ -z "${GPU_ID}" ]] && { echo "Missing <gpu_id>"; usage; exit 1; }
    shift || true
    ;;
  client)
    SERVER_IP=${1:-}
    GPU_ID=${2:-}
    { [[ -z "${SERVER_IP}" ]] || [[ -z "${GPU_ID}" ]]; } && { echo "Missing <server_ip> and/or <gpu_id>"; usage; exit 1; }
    shift 2 || true
    ;;
  *)
    usage; exit 1;
    ;;
esac

# Defaults
IFACES="eth1,eth2,eth3,eth4"
CTRL_DEV="eth0"  # Control network (eth0), data networks (eth1-4)
SIZE=$((64*1024*1024))
ITERS=20
CHUNK=$((512*1024))
NSOCKS=4
NTHREADS=1
UNIX_PREFIX=""  # empty by default (explicit opt-in)
USE_UNIX=0
IMPL="kernel"
HOST_RECV=0
SKIP_WARMUP=1

# Parse options
for arg in "$@"; do
  case "$arg" in
    --ifaces=*) IFACES="${arg#*=}" ;;
    --ctrl=*) CTRL_DEV="${arg#*=}" ;;
    --size=*) SIZE="${arg#*=}" ;;
    --iters=*) ITERS="${arg#*=}" ;;
    --chunk=*) CHUNK="${arg#*=}" ;;
    --nsocks=*) NSOCKS="${arg#*=}" ;;
    --nthreads=*) NTHREADS="${arg#*=}" ;;
    --unix-prefix=*) UNIX_PREFIX="${arg#*=}"; USE_UNIX=1 ;;
    --no-unix) USE_UNIX=0; UNIX_PREFIX="" ;;
    --impl=*) IMPL="${arg#*=}" ;;
    --host-recv) HOST_RECV=1 ;;
    --skip-warmup=*) SKIP_WARMUP="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; usage; exit 1;;
  esac
done

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_BASE="logs/bench_${ROLE}_${TS}"
LOG_FILE="${LOG_BASE}.log"

# Env for TCPX (adapted from run_nccl_test_tcpx.sh)
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${IFACES}"
export NCCL_GPUDIRECTTCPX_CTRL_DEV="${CTRL_DEV}"
export NCCL_NSOCKS_PERTHREAD="${NSOCKS}"
export NCCL_SOCKET_NTHREADS="${NTHREADS}"

# TCPX-specific optimizations from NCCL test script
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608

# TCPX TX/RX CPU bindings (H100 specific, from GCP best practices)
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# TCPX flow steering and performance tuning
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_PROGRAM_CONNECT_TIMEOUT_MS=30000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# NCCL general settings
export NCCL_SOCKET_IFNAME=eth0  # Control network
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0

# NCCL debug output (to verify TCPX configuration)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

if [[ ${USE_UNIX} -eq 1 ]]; then
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="${UNIX_PREFIX}"
  if [[ ! -d "${UNIX_PREFIX}" ]]; then
    echo "[WARN] UNIX prefix '${UNIX_PREFIX}' directory does not exist. dp-manager may not be running."
  fi
fi

# Test program config
export UCCL_TCPX_PERF_SIZE="${SIZE}"
export UCCL_TCPX_CHUNK_BYTES="${CHUNK}"
export UCCL_TCPX_PERF_ITERS="${ITERS}"
export UCCL_TCPX_UNPACK_IMPL="${IMPL}"
export UCCL_TCPX_HOST_RECV_DEBUG="${HOST_RECV}"

# Show config
echo "=== TCPX P2P Benchmark ===" | tee "${LOG_FILE}"
echo "Role        : ${ROLE}" | tee -a "${LOG_FILE}"
echo "Server IP   : ${SERVER_IP:-N/A}" | tee -a "${LOG_FILE}"
echo "GPU ID      : ${GPU_ID}" | tee -a "${LOG_FILE}"
echo "Ifaces      : ${IFACES}" | tee -a "${LOG_FILE}"
echo "Ctrl dev    : ${CTRL_DEV}" | tee -a "${LOG_FILE}"
echo "Size        : ${SIZE} bytes" | tee -a "${LOG_FILE}"
echo "Iters       : ${ITERS} (skip warmup ${SKIP_WARMUP})" | tee -a "${LOG_FILE}"
echo "Chunk bytes : ${CHUNK}" | tee -a "${LOG_FILE}"
echo "nsocks/thrds: ${NSOCKS}/${NTHREADS}" | tee -a "${LOG_FILE}"
echo "UNIX prefix : ${UNIX_PREFIX:-OFF}" | tee -a "${LOG_FILE}"
echo "Impl        : ${IMPL} (host-recv=${HOST_RECV})" | tee -a "${LOG_FILE}"
echo "Log file    : ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Run
if [[ "${ROLE}" == "server" ]]; then
  ./tests/test_tcpx_perf server "${GPU_ID}" | tee -a "${LOG_FILE}"
else
  ./tests/test_tcpx_perf client "${SERVER_IP}" "${GPU_ID}" | tee -a "${LOG_FILE}"
fi

# Post-process steady-state BW (skip warmup)
# Extract lines like: "[PERF] Iter X time=Yms"
ITER_LINES=$(grep -E "\[PERF\] Iter [0-9]+ time=([0-9]+\.?[0-9]*)ms" -n "${LOG_FILE}" || true)
if [[ -z "${ITER_LINES}" ]]; then
  echo "[WARN] No per-iteration timings found in ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit 0
fi

# Compute steady-state avg time
SS_AVG_MS=$(echo "${ITER_LINES}" | awk -v skip="${SKIP_WARMUP}" '
  {
    # line format: N:[PERF] Iter i time=valms
    split($0,a,":");
    line=a[2];
    match(line, /Iter ([0-9]+) time=([0-9]+\.?[0-9]*)ms/, m);
    iter=m[1]+0; val=m[2]+0.0;
    if (iter>=skip) { sum+=val; n+=1; }
  }
  END { if (n>0) printf "%.6f", sum/n; else print ""; }
')

if [[ -z "${SS_AVG_MS}" ]]; then
  echo "[WARN] Not enough iterations for steady-state (skip=${SKIP_WARMUP})" | tee -a "${LOG_FILE}"
  exit 0
fi

SS_BW_GBPS=$(python3 -c "
size_gb = float(${SIZE})/(1024**3)
avg_ms = float(${SS_AVG_MS})
print(f'{size_gb / (avg_ms/1000.0):.3f}')
")

echo "[STEADY-STATE] Avg time: ${SS_AVG_MS} ms, BW: ${SS_BW_GBPS} GB/s" | tee -a "${LOG_FILE}"

