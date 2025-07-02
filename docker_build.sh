#!/bin/bash
set -e

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) back-ends
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker image derived from Ubuntu 22.04.
#
# Usage:
#   ./docker_build.sh [cuda|rocm|all]
#
# The wheels are written to ./wheelhouse/
# -----------------------

TARGET=${1:-cuda}

if [[ $TARGET != "cuda" && $TARGET != "rocm" && $TARGET != "all" ]]; then
  echo "Usage: $0 [cuda|rocm|all]" >&2
  exit 1
fi

rm -r uccl.egg-info || true
rm -r dist || true
rm -r uccl/lib || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -r "${WHEEL_DIR}" || true
mkdir -p "${WHEEL_DIR}"

# Determine host UID/GID once so we can use it everywhere (must come before
# the early 'all' branch).
HOST_UID=$(id -u)
HOST_GID=$(id -g)

# If TARGET=all, orchestrate both builds
if [[ $TARGET == "all" ]]; then
  # Build both backend-specific wheels first
  "$0" cuda
  "$0" rocm

  echo "### Packaging $TARGET wheel (contains both libs) ###"
  docker run --rm --user "${HOST_UID}:${HOST_GID}" \
    -e TARGET="${TARGET}" \
    -v "$(pwd)":/io \
    -w /io \
    uccl-builder-rocm /bin/bash -c '
      set -euo pipefail
      ls -lh uccl/lib
      python3 -m build
      auditwheel repair dist/uccl-*.whl --exclude libibverbs.so.1 -w /io/wheelhouse-${TARGET}
      auditwheel show /io/wheelhouse-${TARGET}/*.whl
    '

  echo "Done. $TARGET wheel is in wheelhouse-${TARGET}/."
  exit 0
fi

DOCKERFILE="Dockerfile.${TARGET}"
IMAGE_NAME="uccl-builder-${TARGET}"

# Build the builder image (contains toolchain + CUDA/ROCm)
echo "[1/3] Building Docker image ${IMAGE_NAME} using ${DOCKERFILE}..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

echo "[2/3] Running build inside container..."
docker run --rm --user "${HOST_UID}:${HOST_GID}" \
  -e TARGET="${TARGET}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -v "$(pwd)":/io \
  -w /io \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail
    echo "[container] Backend: $TARGET"
    echo "[container] Compiling native library…"
    cd rdma
    if [[ "$TARGET" == cuda ]]; then
        make clean && make -j$(nproc)
        TARGET_SO=libnccl-net-uccl.so
    else
        make clean -f Makefile_hip
        make -j$(nproc) -f Makefile_hip "CXXFLAGS=-DBROADCOM_NIC"
        TARGET_SO=librccl-net-uccl.so
    fi
    cd ..
    echo "[container] Packaging uccl..."
    mkdir -p uccl/lib
    cp rdma/${TARGET_SO} uccl/lib/
    python3 -m build
    echo "[container] Running auditwheel..."
    auditwheel repair dist/*.whl --exclude libibverbs.so.1 -w /io/${WHEEL_DIR}
    auditwheel show /io/${WHEEL_DIR}/*.whl
  '

# 3. Done
echo "[3/3] Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl-*.whl || true
