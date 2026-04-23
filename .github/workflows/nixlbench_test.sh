# Usage:
#   .gitlab/nixlbench_test.sh <NIXL_INSTALL_DIR>
#
#   NIXL_INSTALL_DIR  - Where libuccl_p2p, NIXL, and nixlbench are installed
#                       (e.g. ~/nfs/nixl_install)
#

set -euo pipefail

NIXL_INSTALL_DIR="${1:?Usage: $0 <NIXL_INSTALL_DIR>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UCCL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MINICONDA_DIR="${HOME}/nfs/miniconda3"
CONDA_ENV="uccl-ci-sandbox"

NIXL_SRC_DIR="/tmp/nixl_ci_$$"
ABSEIL_DIR="/tmp/abseil_ci_$$"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
ABSEIL_TAG="20240116.2"

ARCH=$(uname -m)
[ "${ARCH}" = "arm64" ] && ARCH="aarch64"

ETCD_PID=""
ETCD_PORT=12379
ETCD_PEER_PORT=12380

cleanup() {
    echo "=== Cleanup ==="
    if [ -n "${ETCD_PID}" ]; then
        echo "Stopping etcd (PID ${ETCD_PID})..."
        kill "${ETCD_PID}" 2>/dev/null || true
        wait "${ETCD_PID}" 2>/dev/null || true
    fi
    rm -rf "${NIXL_SRC_DIR}" "${ABSEIL_DIR}"
}
trap cleanup EXIT

# ── Activate conda env ────────────────────────────────────────────────────────
echo "=== Activating conda env: ${CONDA_ENV} ==="
# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Build UCCL p2p and install all transport backends ────────────────────────
echo "=== Building UCCL p2p (all transports) ==="
cd "${UCCL_ROOT}/p2p"
make clean PYTHON=python3 || true
make all-transport PYTHON=python3
make install PYTHON=python3 LIBDIR="${NIXL_INSTALL_DIR}/lib" PREFIX="${NIXL_INSTALL_DIR}"

export LIBRARY_PATH="${NIXL_INSTALL_DIR}/lib:${LIBRARY_PATH:-}"

# ── Build and install Abseil (NIXL needs absl_log; Ubuntu 24.04's Abseil is too old) ─
echo "=== Building Abseil (required by NIXL) ==="
git clone --depth 1 --branch "${ABSEIL_TAG}" https://github.com/abseil/abseil-cpp.git "${ABSEIL_DIR}"
cd "${ABSEIL_DIR}"
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${NIXL_INSTALL_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DABSL_BUILD_TESTING=OFF \
    -DABSL_ENABLE_INSTALL=ON
cmake --build . -j"$(nproc)"
cmake --install .
cd "${UCCL_ROOT}"

# Prefer our Abseil over system (Ubuntu's lacks absl_log)
export PKG_CONFIG_PATH="${NIXL_INSTALL_DIR}/lib/pkgconfig:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

# ── Clone NIXL ────────────────────────────────────────────────────────────────
echo "=== Cloning latest NIXL ==="
git clone --depth 1 "${NIXL_REPO}" "${NIXL_SRC_DIR}"

# ── Build NIXL with UCCL plugin ───────────────────────────────────────────────
echo "=== Building NIXL (UCCL plugin only) ==="
cd "${NIXL_SRC_DIR}"
rm -rf build
meson setup build \
    --prefix="${NIXL_INSTALL_DIR}" \
    --buildtype=release \
    -Dbuild_tests=false \
    -Dbuild_examples=false \
    "-Dcpp_args=-I${NIXL_INSTALL_DIR}/include"
cd build
ninja
ninja install

# ── Build nixlbench ───────────────────────────────────────────────────────────
echo "=== Building nixlbench ==="
cd "${NIXL_SRC_DIR}/benchmark/nixlbench"
rm -rf build
meson setup build \
    -Dnixl_path="${NIXL_INSTALL_DIR}" \
    --prefix="${NIXL_INSTALL_DIR}" \
    --buildtype=release
cd build
ninja
ninja install

# ── Runtime environment ───────────────────────────────────────────────────────
echo "=== Configuring runtime environment ==="
export PATH="${NIXL_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${NIXL_INSTALL_DIR}/lib:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins:/usr/local/lib:/usr/lib/${ARCH}-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins"
export UCCL_SOCKET_IFNAME="${UCCL_SOCKET_IFNAME:-lo}"
#export UCCL_DEBUG="${UCCL_DEBUG:-WARN}"

echo "NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
nvidia-smi topo -m || true

# ── Start etcd ────────────────────────────────────────────────────────────────
echo "=== Starting etcd ==="
etcd \
    --listen-client-urls    "http://127.0.0.1:${ETCD_PORT}" \
    --advertise-client-urls "http://127.0.0.1:${ETCD_PORT}" \
    --listen-peer-urls      "http://127.0.0.1:${ETCD_PEER_PORT}" \
    --initial-advertise-peer-urls "http://127.0.0.1:${ETCD_PEER_PORT}" \
    --initial-cluster       "default=http://127.0.0.1:${ETCD_PEER_PORT}" \
    --log-level warn \
    &
ETCD_PID=$!
sleep 2

ETCDCTL_API=3 etcdctl --endpoints="http://127.0.0.1:${ETCD_PORT}" endpoint health

# ── Run nixlbench across transport configurations ────────────────────────────
#
# Test matrix (single node, 2 processes each):
#   1. RDMA + IPC        (default intra-node path)
#   2. RDMA without IPC  (forced RDMA loopback)
#   3. NCCL + IPC        (TCP control plane + IPC data)
#   4. NCCL without IPC  (forced NCCL data path)

NIXLBENCH_ARGS=(
    --etcd_endpoints "http://127.0.0.1:${ETCD_PORT}"
    --backend UCCL
    --initiator_seg_type DRAM
    --target_seg_type DRAM
    --total_buffer_size 80000000
    --start_block_size 16384
    --max_block_size 16384
    --start_batch_size 4
    --max_batch_size 4
    --check_consistency
)

TOTAL_PASSED=0
TOTAL_FAILED=0

run_nixlbench_test() {
    local test_name="$1"
    local transport="$2"
    local disable_ipc="$3"

    echo ""
    echo "=== Test: ${test_name} (transport=${transport}, disable_ipc=${disable_ipc}) ==="

    local env_vars="UCCL_P2P_TRANSPORT=${transport}"
    if [ "${disable_ipc}" = "1" ]; then
        env_vars="${env_vars} UCCL_P2P_DISABLE_IPC=1"
    fi

    env ${env_vars} nixlbench "${NIXLBENCH_ARGS[@]}" 2>&1 &
    local pid1=$!
    sleep 1
    env ${env_vars} nixlbench "${NIXLBENCH_ARGS[@]}" 2>&1 &
    local pid2=$!

    local s1=0 s2=0
    wait "${pid1}" || s1=$?
    wait "${pid2}" || s2=$?

    echo "  Process 1 exit: ${s1}, Process 2 exit: ${s2}"

    if [ "${s1}" -ne 0 ] || [ "${s2}" -ne 0 ]; then
        echo "  FAILED: ${test_name}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        return 1
    else
        echo "  PASSED: ${test_name}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
        return 0
    fi
}

OVERALL_RC=0

run_nixlbench_test "RDMA + IPC"         rdma 0 || OVERALL_RC=1
run_nixlbench_test "RDMA without IPC"   rdma 1 || OVERALL_RC=1
run_nixlbench_test "NCCL + IPC"         nccl 0 || OVERALL_RC=1
run_nixlbench_test "NCCL without IPC"   nccl 1 || OVERALL_RC=1

echo ""
echo "================================================================"
echo "  RESULTS: ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed"
echo "================================================================"

if [ "${OVERALL_RC}" -ne 0 ]; then
    echo "=== NIXLBench UCCL benchmark FAILED ==="
    exit 1
fi

echo "=== NIXLBench UCCL benchmark PASSED ==="
