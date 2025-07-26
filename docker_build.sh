#!/bin/bash
set -e

# -----------------------
# Build uccl wheels for CUDA (NVIDIA), ROCm (AMD), and CUDA (Grace-Hopper) backends
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker image derived from Ubuntu 22.04.
#
# Usage:
#   ./docker_build.sh [cuda|rocm|gh|efa|all] [3.13]
#
# The wheels are written to ./wheelhouse-*/
# -----------------------

TARGET=${1:-cuda}
PY_VER=${2:-3.13}
ARCH="$(uname -m)"

if [[ $TARGET != "cuda" && $TARGET != "rocm" && $TARGET != "gh" && $TARGET != "efa" && $TARGET != "all" ]]; then
  echo "Usage: $0 [cuda|rocm|gh|efa|all]" >&2
fi

if [[ $TARGET == "gh" && "$ARCH" != "aarch64" ]]; then
  echo "Skipping ARM build on x86 host. You need an ARM host to build ARM wheels."
  exit 0
fi

rm -r uccl.egg-info >/dev/null 2>&1 || true
rm -r dist >/dev/null 2>&1 || true
rm -r uccl/lib >/dev/null 2>&1 || true
rm -r build >/dev/null 2>&1 || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -r "${WHEEL_DIR}" >/dev/null 2>&1 || true
mkdir -p "${WHEEL_DIR}"


#####################################################################################
############ Build for each backend and package **all** shared libraries ############
#####################################################################################
if [[ $TARGET == "all" ]]; then
  # Temporary directory to accumulate .so files from each backend build
  TEMP_LIB_DIR="uccl/.tmp"
  rm -r "${TEMP_LIB_DIR}" >/dev/null 2>&1 || true
  mkdir -p "${TEMP_LIB_DIR}"

  echo "### Building CUDA backend and collecting its shared library ###"
  "$0" cuda
  cp uccl/lib/*.so "${TEMP_LIB_DIR}/"

  echo "### Building ROCm backend and collecting its shared library ###"
  if [[ "$ARCH" == "aarch64" ]]; then
    echo "Skipping ROCm build on Arm64."
  else
    "$0" rocm
    cp uccl/lib/*.so "${TEMP_LIB_DIR}/"
  fi

  echo "### Building EFA backend and collecting its shared library ###"
  if [[ "$ARCH" == "aarch64" ]]; then
    echo "Skipping EFA build on Arm64."
  else
    "$0" efa
    cp uccl/lib/*.so "${TEMP_LIB_DIR}/"
  fi

  echo "### Building Grace Hopper backend and collecting its shared library ###"
  "$0" gh
  cp uccl/lib/*.so "${TEMP_LIB_DIR}/"

  echo "### Preparing combined library directory ###"
  rm -r uccl/lib >/dev/null 2>&1 || true
  mkdir -p uccl/lib
  cp "${TEMP_LIB_DIR}"/*.so uccl/lib/

  echo "### Packaging $TARGET wheel (contains all libs) ###"
  docker run --rm --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $HOME:$HOME \
    -v "$(pwd)":/io \
    -e TARGET="${TARGET}" \
    -w /io \
    uccl-builder-cuda /bin/bash -c '
      set -euo pipefail
      ls -lh uccl/lib
      python3 -m build
      auditwheel repair dist/uccl-*.whl --exclude libibverbs.so.1 -w /io/wheelhouse-${TARGET}
      auditwheel show /io/wheelhouse-${TARGET}/*.whl
    '

  rm -r "${TEMP_LIB_DIR}" >/dev/null 2>&1 || true

  echo "Done. $TARGET wheel is in wheelhouse-${TARGET}/."
  exit 0
fi
#####################################################################################


#####################################################################################
############ Build for a single backend #############################################
#####################################################################################
DOCKERFILE="docker/Dockerfile.${TARGET}"
IMAGE_NAME="uccl-builder-${TARGET}"

# Function to build target-specific libraries and package
build_target() {
    local TARGET="$1"
    local WHEEL_DIR="$2"
    
    set -euo pipefail
    echo "[container] Backend: $TARGET"
    echo "[container] Compiling native library…"
    
    if [[ "$TARGET" == cuda ]]; then
        cd rdma && make clean && make -j$(nproc) && cd ..
        TARGET_SO=rdma/libnccl-net-uccl.so
    elif [[ "$TARGET" == rocm ]]; then
        # Unlike CUDA, ROCM does not include nccl.h. So we need to build rccl to get nccl.h.
        if [[ ! -f "thirdparty/rccl/build/release/include/nccl.h" ]]; then
          cd thirdparty/rccl
          # Just to get nccl.h, not the whole library
          CXX=/opt/rocm/bin/hipcc cmake -B build/release -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF >/dev/null 2>&1 || true
          cd ../..
        fi
        cd rdma && make clean -f MakefileHip && make -j$(nproc) -f MakefileHip && cd ..
        TARGET_SO=rdma/librccl-net-uccl.so
    elif [[ "$TARGET" == efa ]]; then
        cd efa && make clean && make -j$(nproc) && cd ..
        TARGET_SO=efa/libnccl-net-efa.so
        # EFA requires a custom NCCL.
        cd thirdparty/nccl-sg
        make src.build -j NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
        cd ../..
    elif [[ "$TARGET" == gh ]]; then
        cd rdma && make clean && make -j$(nproc) && cd ..
        TARGET_SO=rdma/libnccl-net-uccl.so
    else 
        echo "Unsupported target: $TARGET"
        exit 1
    fi

    echo "[container] Packaging uccl..."
    mkdir -p uccl/lib
    cp ${TARGET_SO} uccl/lib/

    echo "[container] Building uccl.p2p Python binding..."
    rm uccl/p2p*.so >/dev/null 2>&1 || true
    # Not supporting p2p for EFA now.
    if [[ "$TARGET" == cuda || "$TARGET" == gh ]]; then
      cd p2p
      make clean && make -j$(nproc)
      mv p2p*.so ../uccl
      cd ../
    elif [[ "$TARGET" == rocm ]]; then
      cd p2p
      make clean -f MakefileHip && make -j$(nproc) -f MakefileHip
      mv p2p*.so ../uccl
      cd ../
    fi

    if [[ "$TARGET" == efa ]]; then
      cp thirdparty/nccl-sg/build/lib/libnccl.so uccl/lib/libnccl-efa.so
    fi

    ls -lh uccl/lib
    python3 -m build

    echo "[container] Running auditwheel..."
    auditwheel repair dist/*.whl --exclude libibverbs.so.1 -w /io/${WHEEL_DIR}
    for whl in /io/${WHEEL_DIR}/*.whl; do
      auditwheel show "$whl"
    done
}

# Build the builder image (contains toolchain + CUDA/ROCm)
echo "[1/3] Building Docker image ${IMAGE_NAME} using ${DOCKERFILE}..."
echo "Python version: ${PY_VER}"
if [[ "$TARGET" == "gh" ]]; then
  docker build --platform=linux/arm64 --build-arg PY_VER="${PY_VER}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  docker build --build-arg PY_VER="${PY_VER}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

echo "[2/3] Running build inside container..."
docker run --rm --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v "$(pwd)":/io \
  -e TARGET="${TARGET}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e FUNCTION_DEF="$(declare -f build_target)" \
  -w /io \
  "$IMAGE_NAME" /bin/bash -c '
    eval "$FUNCTION_DEF"
    build_target "$TARGET" "$WHEEL_DIR"
  '

# 3. Done
echo "[3/3] Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl-*.whl || true
#####################################################################################
