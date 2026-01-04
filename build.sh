#!/bin/bash
set -e

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) backends/targets.
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker image derived from Ubuntu 22.04.
#
# Usage:
#   ./build.sh [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image]
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
  echo "Usage: $0 [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image]" >&2
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

build_ccl_rdma() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ccl_rdma Target: $TARGET"
  
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

build_ccl_efa() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ccl_efa Target: $TARGET"

  if [[ "$ARCH" == "aarch64" || "$TARGET" == rocm* || "$TARGET" == "therock" ]]; then
    echo "Skipping EFA build on Arm64 (no EFA installer) or ROCm (no CUDA)."
    return
  fi
  cd collective/efa && make clean && make -j$(nproc) && cd ../../

  # EFA requires a custom NCCL.
  cd thirdparty/nccl-sg
  make src.build -j$(nproc) NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
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
    cp p2p/libuccl_p2p.so uccl/lib/
    cp p2p/librdma_plugin.a uccl/lib/
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
  elif [[ "$TARGET" == rocm* || "$TARGET" == cuda* ]]; then
    cd ep
    # This may be needed if you traverse through different git commits
    # make clean && rm -r build || true
    python3 setup.py build
    cd ..
    echo "[container] Copying GPU-driven .so to uccl/"
    mkdir -p uccl/lib
    cp ep/build/**/*.so uccl/
  fi
}

build_eccl() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_eccl Target: $TARGET"

  cd experimental/eccl
  if [[ "$TARGET" == cuda* ]]; then
    make clean -f Makefile && make -j$(nproc) -f Makefile
  elif [[ "$TARGET" == rocm* ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  fi
  cd ../..

  echo "[container] Copying eccl .so to uccl/"
  mkdir -p uccl/lib # mkdir anyway
  cp experimental/eccl/*eccl*.so uccl/lib
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

# Detect stale builder image
# If a builder image exists...
hash_image=$(docker images -q ${IMAGE_NAME})
if [[ "${hash_image}" != "" ]]; then

  # Get its and its dockerfile's timestamps
  ts_dockerfile=$(date -r ${DOCKERFILE} --iso-8601=seconds)
  ts_image=$(docker inspect -f '{{ .Created }}' ${IMAGE_NAME})

  # If image is stale, suggest deleting & purging it
  if [[ "${ts_dockerfile}" > "${ts_image}" ]]; then
      echo "WARNING: builder image '${IMAGE_NAME}' is older than its source (${DOCKERFILE})" >&2
      echo "Please consider removing it, pruning the builder cache, and retrying the build to regenerate it." >&2
      echo " " >&2
      echo "  $ docker image rm '${IMAGE_NAME}'" >&2
      echo "  $ docker buildx prune -f" >&2
      echo " " >&2
      echo "NOTE: this may also prune unrelated builder cache images!" >&2
      sleep 1
  fi
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
  docker build --platform=linux/arm64 $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  docker build $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

echo "[2/3] Running build inside container..."

# Auto-detect CUDA architecture for ep build
DETECTED_GPU_ARCH=""
if [[ "$BUILD_TYPE" =~ (ep|all) ]];then
  if [[ "$TARGET" == cuda* ]] && command -v nvidia-smi &> /dev/null; then
    DETECTED_GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    if [[ -n "$DETECTED_GPU_ARCH" ]]; then
      echo "Auto-detected CUDA compute capability: ${DETECTED_GPU_ARCH}"
    fi
  elif [[ "$TARGET" == rocm* ]] && command -v amd-smi &> /dev/null; then
    # Check if jq is installed, install via pip if not
    if ! command -v jq &> /dev/null; then
      echo "jq not found, installing via pip..."
      pip install jq
    fi
    DETECTED_GPU_ARCH=$(amd-smi static -g 0 --asic --json | jq -r '.[].asic.target_graphics_version')
    if [[ -n "$DETECTED_GPU_ARCH" ]]; then
      echo "Auto-detected ROCm architecture: ${DETECTED_GPU_ARCH}"
    fi
  fi
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_GPU_ARCH}}"

docker run --rm --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v "$(pwd)":/io \
  -e TARGET="${TARGET}" \
  -e PY_VER="${PY_VER}" \
  -e ARCH="${ARCH}" \
  -e ROCM_IDX_URL="${ROCM_IDX_URL}" \
  -e IS_EFA="${IS_EFA}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e BUILD_TYPE="${BUILD_TYPE}" \
  -e USE_TCPX="${USE_TCPX:-0}" \
  -e USE_EFA="${USE_EFA:-0}" \
  -e USE_IB="${USE_IB:-0}" \
  -e USE_TCP="${USE_TCP:-0}" \
  -e MAKE_NORMAL_MODE="${MAKE_NORMAL_MODE:-}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
  -e FUNCTION_DEF="$(declare -f build_rccl_nccl_h build_ccl_rdma build_ccl_efa build_p2p build_ep build_eccl)" \
  -w /io \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail

    if [[ "$TARGET" == "therock" ]]; then

      # Setup requested Python (PyPA images have all versions pre-installed)
      PY_V=$(echo ${PY_VER} | tr -d .)
      export PATH=/opt/python/cp${PY_V}-cp${PY_V}/bin:$PATH

      # Python environment with ROCm from TheRock
      python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
      pip3 install --no-cache-dir --upgrade pip
      pip3 install --no-cache-dir build auditwheel pybind11
      pip3 install --no-cache-dir rocm[libraries,devel] --index-url ${ROCM_IDX_URL}
    fi

    eval "$FUNCTION_DEF"

    if [[ "$TARGET" == rocm* ]]; then
      build_rccl_nccl_h
    fi

    if [[ "$BUILD_TYPE" == "ccl_rdma" ]]; then
      build_ccl_rdma "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "ccl_efa" ]]; then
      build_ccl_efa "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "p2p" ]]; then
      build_p2p "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "ep" ]]; then
      build_ep "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "eccl" ]]; then
      build_eccl "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "all" ]]; then
      build_ccl_rdma "$TARGET" "$ARCH" "$IS_EFA"
      build_ccl_efa "$TARGET" "$ARCH" "$IS_EFA"
      build_p2p "$TARGET" "$ARCH" "$IS_EFA"
      # build_ep "$TARGET" "$ARCH" "$IS_EFA"
      # build_eccl "$TARGET" "$ARCH" "$IS_EFA"
    fi

    ls -lh uccl/
    ls -lh uccl/lib/

    # Emit TheRock init code
    if [[ "$TARGET" == "therock" ]]; then
      echo "
def initialize():
  import rocm_sdk
  rocm_sdk.initialize_process(preload_shortnames=[
    \"amd_comgr\",
    \"amdhip64\",
    \"roctx64\",
    \"hiprtc\",
    \"hipblas\",
    \"hipfft\",
    \"hiprand\",
    \"hipsparse\",
    \"hipsolver\",
    \"rccl\",
    \"hipblaslt\",
    \"miopen\",
  ],
  check_version=\"$(rocm-sdk version)\")
" > uccl/_rocm_init.py

      # Back-up setup.py and emit UCCL package dependence on TheRock
      BACKUP_FN=$(mktemp -p . -t setup.py.XXXXXX)
      cp ./setup.py ${BACKUP_FN}
      sed -i "s/\"rocm\": \[\],/\"rocm\": \[\"rocm\[libraries\]==$(rocm-sdk version)\"\, \"torch\", \"numpy\"],/;" setup.py

      export PIP_EXTRA_INDEX_URL=${ROCM_IDX_URL}
    fi

    python3 -m build

    if [[ "$TARGET" == "therock" ]]; then
      # Undo UCCL package dependence on TheRock wheels after the build is done
      mv ${BACKUP_FN} setup.py
    fi

    auditwheel repair dist/uccl-*.whl \
      --exclude "libtorch*.so" \
      --exclude "libc10*.so" \
      --exclude "libibverbs.so.1" \
      --exclude "libcudart.so.12" \
      --exclude "libamdhip64.so.*" \
      --exclude "libcuda.so.1" \
      -w /io/${WHEEL_DIR}

    # Add backend tag to wheel filename using local version identifier
    if [[ "$TARGET" == rocm* || "$TARGET" == "therock" ]]; then
      # Adjust TARGET to the preferred wheel name suffix for python-packaged ROCm, e.g. "rocm7.9.0rc1"
      if [[ "$TARGET" == "therock" ]]; then
        TARGET="rocm$(rocm-sdk version)"
      fi
      cd /io/${WHEEL_DIR}
      for wheel in uccl-*.whl; do
        if [[ -f "$wheel" ]]; then
          # Extract wheel name components: uccl-version-python-abi-platform.whl
          if [[ "$wheel" =~ ^(uccl-)([^-]+)-([^-]+-[^-]+-.+)(\.whl)$ ]]; then
            name="${BASH_REMATCH[1]}"
            version="${BASH_REMATCH[2]}"
            python_abi_platform="${BASH_REMATCH[3]}"
            suffix="${BASH_REMATCH[4]}"
            
            # Add backend to version using local identifier: uccl-version+backend-python-abi-platform.whl
            new_wheel="${name}${version}+${TARGET}-${python_abi_platform}${suffix}"
            
            echo "Renaming wheel: $wheel -> $new_wheel"
            mv "$wheel" "$new_wheel"
          else
            echo "Warning: Could not parse wheel filename: $wheel"
          fi
        fi
      done
      cd /io
    fi
    
    auditwheel show /io/${WHEEL_DIR}/*.whl
  '

# 3. Done
echo "[3/3] Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl-*.whl || true
