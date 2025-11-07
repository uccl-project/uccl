#!/bin/bash
set -e

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) backends/targets.
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker image derived from Ubuntu 22.04.
#
# Usage:
#   ./build.sh [cuda|rocm|therock] [all|rdma|p2p|efa|ep] [py_version] [rocm_index_url] [therock_base_image]
#
# The wheels are written to wheelhouse-[cuda|rocm|therock]
# -----------------------

TARGET=${1:-cuda}
BUILD_TYPE=${2:-all}
PY_VER=${3:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
ARCH="$(uname -m)"
# The default for ROCM_IDX_URL depends on the gfx architecture of your GPU and the index URLs may change.
ROCM_IDX_URL=${4:-https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu}
# The default for THEROCK_BASE_IMAGE is current, but may change. Make sure to track TheRock's dockerfile.
THEROCK_BASE_IMAGE=${5:-quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017}
IS_EFA=$( [ -d "/sys/class/infiniband/" ] && ls /sys/class/infiniband/ 2>/dev/null | grep -q rdmap && echo "EFA support: true" ) || echo "EFA support: false"


if [[ $TARGET != cuda* && $TARGET != rocm* && $TARGET != "therock" ]]; then
  echo "Usage: $0 [cuda|rocm|therock] [all|rdma|p2p|efa|ep|eccl] [py_version] [rocm_index_url]" >&2
  exit 1
fi

if [[ $ARCH == "aarch64" && ( $TARGET == rocm* || $TARGET == "therock" ) ]]; then
  echo "Skipping ROCm build on Arm64 (no ROCm toolchain)."
  exit 1
fi

rm -r uccl.egg-info >/dev/null 2>&1 || true
rm -r dist >/dev/null 2>&1 || true
rm -r build >/dev/null 2>&1 || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -r "${WHEEL_DIR}" >/dev/null 2>&1 || true
mkdir -p "${WHEEL_DIR}"

build_rccl_nccl_h() {
  # Unlike CUDA, ROCM does not include nccl.h. So we need to build rccl to get nccl.h.
  if [[ ! -f "thirdparty/rccl/build/release/include/nccl.h" ]]; then
    cd thirdparty/rccl
    # Just to get nccl.h, not the whole library
    CXX=/opt/rocm/bin/hipcc cmake -B build/release -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF >/dev/null 2>&1 || true
    cd ../..
  fi
}

build_rdma() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_rdma Target: $TARGET"

  if [[ "$TARGET" == cuda* ]]; then
    cd collective/rdma && make clean && make -j$(nproc) && cd ../../
    TARGET_SO=collective/rdma/libnccl-net-uccl.so
  elif [[ "$TARGET" == rocm* ]]; then
    if [[ "$ARCH" == "aarch64" ]]; then
      echo "Skipping ROCm build on Arm64 (no ROCm toolchain)."
      return
    fi
    cd collective/rdma && make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm && cd ../../
    TARGET_SO=collective/rdma/librccl-net-uccl.so
  elif [[ "$TARGET" == "therock" ]]; then
    if [[ "$ARCH" == "aarch64" ]]; then
      echo "Skipping ROCm build on Arm64 (no ROCm toolchain)."
      return
    fi
    # Unlike CUDA, ROCM does not include nccl.h. So we need to build rccl to get nccl.h.
    if [[ ! -f "thirdparty/rccl/build/release/include/nccl.h" ]]; then
      cd thirdparty/rccl
      # Just to get nccl.h, not the whole library
      CXX=hipcc cmake -B build/release -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DCMAKE_PREFIX_PATH=$(rocm-sdk path --cmake) -DROCM_PATH=$(rocm-sdk path --root) -DHIP_PLATFORM=amd >/dev/null 2>&1 || true
      cd ../..
    fi
    cd collective/rdma && make clean -f Makefile.therock && make -j$(nproc) -f Makefile.therock HIP_HOME=$(rocm-sdk path --root) CONDA_LIB_HOME=$VIRTUAL_ENV/lib && cd ../../
    TARGET_SO=collective/rdma/librccl-net-uccl.so
  fi

  echo "[container] Copying RDMA .so to uccl/lib/"
  mkdir -p uccl/lib
  cp ${TARGET_SO} uccl/lib/
}

build_efa() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_efa Target: $TARGET"

  if [[ "$ARCH" == "aarch64" || "$TARGET" == rocm* || "$TARGET" == "therock" ]]; then
    echo "Skipping EFA build on Arm64 (no EFA installer) or ROCm (no CUDA)."
    return
  fi
  cd collective/efa && make clean && make -j$(nproc) && cd ../../

  # EFA requires a custom NCCL.
  cd thirdparty/nccl-sg
  make src.build -j$(nproc) NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
  cd ../..

  echo "[container] Copying EFA .so to uccl/lib/"
  mkdir -p uccl/lib
  cp collective/efa/libnccl-net-efa.so uccl/lib/
  cp thirdparty/nccl-sg/build/lib/libnccl.so uccl/lib/libnccl-efa.so
}

build_p2p() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_p2p Target: $TARGET"

  cd p2p
  if [[ "$TARGET" == cuda* ]]; then
    make clean && make -j$(nproc)
  elif [[ "$TARGET" == rocm* ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  elif [[ "$TARGET" == "therock" ]]; then
    make clean -f Makefile.therock && make -j$(nproc) -f Makefile.therock HIP_HOME=$(rocm-sdk path --root) CONDA_LIB_HOME=$VIRTUAL_ENV/lib
  fi
  cd ..

  echo "[container] Copying P2P .so, collective.py and utils.py to uccl/"
  mkdir -p uccl
  mkdir -p uccl/lib
  if [[ -z "${USE_TCPX:-}" || "$USE_TCPX" != "1" ]]; then
    cp p2p/p2p.*.so uccl/
    cp p2p/collective.py uccl/
    cp p2p/transfer.py uccl/
    cp p2p/utils.py uccl/
  else
    echo "[container] USE_TCPX=1, skipping copying p2p runtime files"
  fi
}

build_ep() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ep Target: $TARGET"

  if [[ "$TARGET" == "therock" ]]; then
    echo "Skipping GPU-driven build on therock (no GPU-driven support yet)."
  elif [[ "$TARGET" == rocm* ]]; then
    cd ep
    python3 setup.py build
    cd ..
    echo "[container] Copying GPU-driven .so to uccl/"
    mkdir -p uccl/lib
    cp ep/build/**/*.so uccl/
  elif [[ "$TARGET" == cuda* ]]; then
    cd ep
    make clean && make -j$(nproc) all
    cd ..
    echo "[container] Copying GPU-driven .so to uccl/"
    mkdir -p uccl/lib
    cp ep/*.so uccl/
  fi
}

build_eccl() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_eccl Target: $TARGET"

  cd eccl
  if [[ "$TARGET" == cuda* ]]; then
    echo "Skipping eccl build on Cuda."
    return
  elif [[ "$TARGET" == rocm* ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  fi
  cd ..

  echo "[container] Copying eccl .so to uccl/"
  # mkdir -p uccl/lib
  # cp eccl/eccl.*.so uccl/
}

# Determine the Docker image to use based on the target and architecture
if [[ $TARGET == "cuda" ]]; then
  # default is cuda 12 from `nvidia/cuda:12.3.2-devel-ubuntu22.04`/`nvidia/cuda:12.4.1-devel-ubuntu22.04`
  if [[ "$ARCH" == "aarch64" ]]; then
    DOCKERFILE="docker/Dockerfile.gh"
    IMAGE_NAME="uccl-builder-gh"
  elif [[ -n "$IS_EFA" ]]; then
    DOCKERFILE="docker/Dockerfile.efa"
    IMAGE_NAME="uccl-builder-efa"
  else
    DOCKERFILE="docker/Dockerfile.cuda"
    IMAGE_NAME="uccl-builder-cuda"
  fi
elif [[ $TARGET == "cuda13" ]]; then
  BASE_IMAGE="nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04"
  if [[ "$ARCH" == "aarch64" ]]; then
    DOCKERFILE="docker/Dockerfile.gh"
    IMAGE_NAME="uccl-builder-gh"
  elif [[ -n "$IS_EFA" ]]; then
    DOCKERFILE="docker/Dockerfile.efa"
    IMAGE_NAME="uccl-builder-efa"
  else
    DOCKERFILE="docker/Dockerfile.cuda"
    IMAGE_NAME="uccl-builder-cuda"
  fi
elif [[ $TARGET == "rocm" ]]; then
  # default is latest rocm version from `rocm/dev-ubuntu-22.04`
  DOCKERFILE="docker/Dockerfile.rocm"
  IMAGE_NAME="uccl-builder-rocm"
elif [[ $TARGET == "rocm6" ]]; then
  DOCKERFILE="docker/Dockerfile.rocm"
  BASE_IMAGE="rocm/dev-ubuntu-22.04:6.4.3-complete"
  IMAGE_NAME="uccl-builder-rocm"
elif [[ $TARGET == "therock" ]]; then
  DOCKERFILE="docker/Dockerfile.therock"
  BASE_IMAGE="${THEROCK_BASE_IMAGE}"
  IMAGE_NAME="uccl-builder-therock"
fi

# Build the builder image (contains toolchain + CUDA/ROCm)
echo "[1/3] Building Docker image ${IMAGE_NAME} using ${DOCKERFILE}..."
echo "Python version: ${PY_VER}"
if [[ "$TARGET" == "therock" ]]; then
  echo "ROCm index URL: ${ROCM_IDX_URL}"
fi
BUILD_ARGS="--build-arg PY_VER=${PY_VER}"
if [[ -n "${BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=" --build-arg BASE_IMAGE=${BASE_IMAGE}"
fi

if [[ "$ARCH" == "aarch64" ]]; then
  echo 'docker build --platform=linux/arm64 $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .'
  # docker build --platform=linux/arm64 $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  echo "docker build $BUILD_ARGS -t '$IMAGE_NAME' -f '$DOCKERFILE' ."
  # docker build $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

echo "[2/3] Building..."

export USE_TCPX="${USE_TCPX:-0}"
export MAKE_NORMAL_MODE="${MAKE_NORMAL_MODE:-}"
export FUNCTION_DEF="$(declare -f build_rccl_nccl_h build_rdma build_efa build_p2p build_ep build_eccl)"

set -euo pipefail

eval "$FUNCTION_DEF"

echo "BUILD_TYPE : ${BUILD_TYPE}"

if [[ $TARGET == "cuda" && "$ARCH" == "x86" ]]; then

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin

# install dependencies
apt-get install libelf-dev

# defaul to BUILD_TYPE all
build_rdma "$TARGET" "$ARCH" "$IS_EFA"
build_efa "$TARGET" "$ARCH" "$IS_EFA"
build_p2p "$TARGET" "$ARCH" "$IS_EFA"
build_ep "$TARGET" "$ARCH" "$IS_EFA"
NOTE (yiakwy) : eccl is skpipped on CUDA platform
build_eccl "$TARGET" "$ARCH" "$IS_EFA"

else

echo "$TARGET is not supported yet."
exit 1

fi

python${PY_VER} -m build

auditwheel repair dist/uccl-*.whl --exclude "libtorch*.so" --exclude "libc10*.so" --exclude "libibverbs.so.1" --exclude "libcudart.so.12" --exclude "libamdhip64.so.*" --exclude "libcuda.so.1" -w `pwd`/${WHEEL_DIR}
auditwheel show `pwd`/${WHEEL_DIR}/*.whl

# 3. Done
echo "[3/3] Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "`pwd`/${WHEEL_DIR}"/uccl-*.whl || true
