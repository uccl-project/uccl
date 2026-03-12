#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 UCCL Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# .gitlab/nixlbench_test.sh
#
# Builds UCCL p2p from the current tree, clones latest NIXL, builds it with
# the UCCL plugin, builds nixlbench, then runs a single-node benchmark
# (two nixlbench processes on the same host).
#
# Usage:
#   UCCL_SOCKET_IFNAME=<iface> .gitlab/nixlbench_test.sh <NIXL_INSTALL_DIR>
#
#   NIXL_INSTALL_DIR  - Where libuccl_p2p, NIXL, and nixlbench are installed
#                       (e.g. ~/nfs/nixl_install)
#
# Run from the root of the uccl repo so that ./p2p/Makefile is reachable.

set -euo pipefail

NIXL_INSTALL_DIR="${1:?Usage: $0 <NIXL_INSTALL_DIR>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UCCL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MINICONDA_DIR="${HOME}/nfs/miniconda3"
CONDA_ENV="uccl-ci-sandbox"

NIXL_SRC_DIR="/tmp/nixl_ci_$$"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"

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
    rm -rf "${NIXL_SRC_DIR}"
}
trap cleanup EXIT

# ── Activate conda env ────────────────────────────────────────────────────────
echo "=== Activating conda env: ${CONDA_ENV} ==="
# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Build UCCL p2p and install libuccl_p2p.so ────────────────────────────────
echo "=== Building UCCL p2p ==="
cd "${UCCL_ROOT}/p2p"
make -j"$(nproc)" PYTHON=python3
make install PYTHON=python3 LIBDIR="${NIXL_INSTALL_DIR}/lib" PREFIX="${NIXL_INSTALL_DIR}"

# Make libuccl_p2p.so visible to meson's cc.find_library() during NIXL build
export LIBRARY_PATH="${NIXL_INSTALL_DIR}/lib:${LIBRARY_PATH:-}"

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
    -Dbuild_examples=false
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
export LD_LIBRARY_PATH="${NIXL_INSTALL_DIR}/lib:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu:${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${NIXL_INSTALL_DIR}/lib/${ARCH}-linux-gnu/plugins"
export UCCL_SOCKET_IFNAME="${UCCL_SOCKET_IFNAME:-lo}"
export UCCL_DEBUG="${UCCL_DEBUG:-WARN}"

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

# ── Run nixlbench (two processes, single node) ────────────────────────────────
echo "=== Running nixlbench UCCL benchmark (single node, 2 processes) ==="

NIXLBENCH_ARGS=(
    --etcd_endpoints "http://127.0.0.1:${ETCD_PORT}"
    --backend UCCL
    --initiator_seg_type VRAM
    --target_seg_type VRAM
    --total_buffer_size 80000000
    --start_block_size 16384
    --max_block_size 16384
    --start_batch_size 4
    --max_batch_size 4
    --check_consistency
)

nixlbench "${NIXLBENCH_ARGS[@]}" 2>&1 &
PID1=$!

sleep 1

nixlbench "${NIXLBENCH_ARGS[@]}" 2>&1 &
PID2=$!

wait "${PID1}"; STATUS1=$?
wait "${PID2}"; STATUS2=$?

echo "Process 1 exit code: ${STATUS1}"
echo "Process 2 exit code: ${STATUS2}"

if [ "${STATUS1}" -ne 0 ] || [ "${STATUS2}" -ne 0 ]; then
    echo "=== nixlbench UCCL benchmark FAILED ==="
    exit 1
fi

echo "=== nixlbench UCCL benchmark PASSED ==="
