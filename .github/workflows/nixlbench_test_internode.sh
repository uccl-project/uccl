#!/usr/bin/env bash
# Usage:
#   nixlbench_test_internode.sh <NIXL_INSTALL_DIR> [--host-etcd]
#
# Runs inter-node nixlbench tests (RDMA + NCCL) across two nodes coordinating via etcd.
#
#   --host-etcd: This node (spark0) starts etcd on 10.0.0.1:12379
#   Without flag: This node (spark1) connects to etcd on spark0

set -euo pipefail

NIXL_INSTALL_DIR="${1:?Usage: $0 <NIXL_INSTALL_DIR> [--host-etcd]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UCCL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MINICONDA_DIR="${HOME}/nfs/miniconda3"
CONDA_ENV="uccl-ci-sandbox"

HOST_ETCD=false
ETCD_PORT=12379
ETCD_PEER_PORT=12380
ETCD_HOST_IP="10.0.0.1"
ETCD_ENDPOINT="http://${ETCD_HOST_IP}:${ETCD_PORT}"

ETCD_PID=""

# Parse flags
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-etcd) HOST_ETCD=true ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

cleanup() {
    echo "=== Cleanup ==="

    # Kill any remaining nixlbench processes
    pkill -f nixlbench 2>/dev/null || true

    # Stop etcd if this node started it
    if [ -n "${ETCD_PID}" ]; then
        if kill -0 "${ETCD_PID}" 2>/dev/null; then
            echo "Stopping etcd (PID ${ETCD_PID})..."
            kill "${ETCD_PID}" 2>/dev/null || true
        fi
        wait "${ETCD_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Activate conda env ────────────────────────────────────────────────────────
echo "=== Activating conda env: ${CONDA_ENV} ==="
# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Set runtime environment ───────────────────────────────────────────────────
echo "=== Configuring runtime environment ==="
ARCH=$(uname -m)
[ "${ARCH}" = "arm64" ] && ARCH="aarch64"

export PATH="${NIXL_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${NIXL_INSTALL_DIR}/lib:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins:/usr/local/lib:/usr/lib/${ARCH}-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins"
export UCCL_SOCKET_IFNAME="enP2p1s0f0np0"

# NCCL environment variables for debugging and inter-node communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="enP2p1s0f0np0"

echo "NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "ETCD_ENDPOINT=${ETCD_ENDPOINT}"
nvidia-smi topo -m || true

# ── Start etcd (if this is the etcd host node) ─────────────────────────────────
if $HOST_ETCD; then
    echo "=== Starting etcd on ${ETCD_HOST_IP}:${ETCD_PORT} ==="
    etcd \
        --listen-client-urls    "http://${ETCD_HOST_IP}:${ETCD_PORT}" \
        --advertise-client-urls "http://${ETCD_HOST_IP}:${ETCD_PORT}" \
        --listen-peer-urls      "http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}" \
        --initial-advertise-peer-urls "http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}" \
        --initial-cluster       "default=http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}" \
        --log-level warn \
        &
    ETCD_PID=$!
fi

# ── Wait for etcd to be ready ──────────────────────────────────────────────────
wait_etcd() {
    echo "=== Waiting for etcd to be ready at ${ETCD_ENDPOINT} ==="
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if ETCDCTL_API=3 etcdctl --endpoints="${ETCD_ENDPOINT}" endpoint health &>/dev/null; then
            echo "etcd is ready"
            return 0
        fi
        echo "Waiting for etcd... (attempt $((attempt + 1))/$max_attempts)"
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "ERROR: etcd did not become ready within ${max_attempts} attempts"
    return 1
}

wait_etcd

# ── Helper function to run a single test ───────────────────────────────────────
run_test() {
    local transport=$1
    local benchmark_group="internode-${transport}"

    echo ""
    echo "=== Test: transport=${transport}, benchmark_group=${benchmark_group} ==="
    echo "Environment:"
    echo "  UCCL_P2P_TRANSPORT=${transport}"
    echo "  NCCL_DEBUG=${NCCL_DEBUG:-unset}"
    echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}"
    echo ""

    NIXLBENCH_ARGS=(
        --etcd_endpoints "${ETCD_ENDPOINT}"
        --backend UCCL
        --benchmark_group "${benchmark_group}"
        --initiator_seg_type DRAM
        --target_seg_type DRAM
        --total_buffer_size 80000000
        --start_block_size 16384
        --max_block_size 16384
        --start_batch_size 4
        --max_batch_size 4
        --check_consistency
    )

    # Explicit timeout of 120s for NCCL
    timeout 120 env UCCL_P2P_TRANSPORT="${transport}" \
        NCCL_DEBUG="${NCCL_DEBUG}" \
        NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
        NCCL_COMM_TIMEOUT=30 \
        nixlbench "${NIXLBENCH_ARGS[@]}" 2>&1

    local status=$?
    if [ $status -eq 124 ]; then
        echo "ERROR: Test ${transport} timed out after 120 seconds (likely deadlock)"
        return 1
    fi
    echo "Test ${transport} exit code: ${status}"
    return $status
}

# ── Run tests sequentially ─────────────────────────────────────────────────────
echo ""
echo "=== Running inter-node nixlbench tests ==="
echo "NOTE: This script must run on BOTH spark0 and spark1 simultaneously!"
echo ""

OVERALL_STATUS=0

run_test "rdma" || OVERALL_STATUS=1

if [ $OVERALL_STATUS -ne 0 ]; then
    echo "=== RDMA test FAILED ==="
    exit 1
fi

# sleep 2

# run_test "nccl" || OVERALL_STATUS=1

# if [ $OVERALL_STATUS -ne 0 ]; then
#     echo "=== NCCL test FAILED ==="
#     exit 1
# fi

echo ""
echo "=== All inter-node nixlbench tests PASSED ==="
