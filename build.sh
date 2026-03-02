#!/bin/bash

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) backends/targets.
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker/Podman image derived from Ubuntu 22.04.
#
# Usage:
#   ./build.sh [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image] [--install]
#
# Environment Variables:
#   CONTAINER_ENGINE=podman Use podman instead of docker.
#                          Example: CONTAINER_ENGINE=podman ./build.sh cuda all
#   USE_INTEL_RDMA_NIC=1   Enable Intel RDMA NIC support (irdma driver, vendor 0x8086)
#                          Example: USE_INTEL_RDMA_NIC=1 ./build.sh cuda ccl_efa
#
# The wheels are written to wheelhouse-[cuda|rocm|therock]
# -----------------------

set -euo pipefail
source shared.sh

########################################################
# 1. Parse arguments: positional args first, then flags
########################################################
DO_INSTALL=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
  --install) DO_INSTALL=1 ;;
  *) POSITIONAL_ARGS+=("$arg") ;;
  esac
done

TARGET=${POSITIONAL_ARGS[0]:-cuda}
BUILD_TYPE=${POSITIONAL_ARGS[1]:-all}
PY_VER=${POSITIONAL_ARGS[2]:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
# The default for ROCM_IDX_URL depends on the gfx architecture of your GPU and the index URLs may change.
ROCM_IDX_URL=${POSITIONAL_ARGS[3]:-https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu}
# The default for THEROCK_BASE_IMAGE is current, but may change. Make sure to track TheRock's dockerfile.
THEROCK_BASE_IMAGE=${POSITIONAL_ARGS[4]:-quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017}

if [[ $TARGET != cuda* && $TARGET != rocm* && $TARGET != "therock" ]]; then
  echo "Usage: $0 [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image] [--install]" >&2
  exit 1
fi

########################################################
# 2. Detect host architecture, container engine, EFA support, etc.
########################################################
ARCH="$(uname -m)"
if [[ $ARCH == "aarch64" && ($TARGET == rocm* || $TARGET == "therock") ]]; then
  echo "Skipping ROCm build on Arm64 (no ROCm toolchain)."
  exit 1
fi

# Container engine: "docker" (default) or "podman"
CONTAINER_ENGINE=${CONTAINER_ENGINE:-docker}
if [[ "$CONTAINER_ENGINE" != "docker" && "$CONTAINER_ENGINE" != "podman" ]]; then
  echo "Error: CONTAINER_ENGINE must be 'docker' or 'podman', got '${CONTAINER_ENGINE}'" >&2
  exit 1
fi
IS_EFA=$([ -d "/sys/class/infiniband/" ] && ls /sys/class/infiniband/ 2>/dev/null | grep -q rdmap && echo "EFA support: true") || echo "EFA support: false"

# Auto-detect CUDA architecture for ep build, auto-detect ROCm architecture for ep build
DETECTED_GPU_ARCH=""
if [[ "$BUILD_TYPE" =~ (ep|all|p2p) ]]; then
  if [[ "$TARGET" == cuda* ]] && command -v nvidia-smi &>/dev/null; then
    DETECTED_GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"

    if [[ -n "$DETECTED_GPU_ARCH" ]]; then
      echo "Auto-detected CUDA compute capability: ${DETECTED_GPU_ARCH}"
    fi
  elif [[ "$TARGET" == rocm* ]] && command -v amd-smi &>/dev/null; then
    # Check if jq is installed, install via pip if not
    if ! command -v jq &>/dev/null; then
      echo "jq not found, installing via pip..."
      pip install jq
    fi
    DETECTED_GPU_ARCH="$(
      PYTHONWARNINGS=ignore \
        amd-smi static -g 0 --asic --json 2>/dev/null |
        jq -r '
          if type == "array" then
            if length > 0 then .[0].asic.target_graphics_version else empty end
          elif .gpu_data and (.gpu_data | length > 0) then
            .gpu_data[0].asic.target_graphics_version
          else
            empty
          end
        ' ||
        true
    )"
    if [[ -n "$DETECTED_GPU_ARCH" ]]; then
      echo "Auto-detected ROCm architecture: ${DETECTED_GPU_ARCH}"
    fi
  else
    echo "No compatible GPU detection tool found, skipping auto-detect"
  fi
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_GPU_ARCH}}"

# The build container (Ubuntu 22.04, glibc 2.35) produces wheels tagged
# manylinux_2_35 by default.  Rocky Linux 9.x only ships glibc 2.34, so
# those wheels won't install there.  Detect the host distro and, when
# running on Rocky 9, retag the wheel to manylinux_2_34 after auditwheel
# repair.  Note: this does not verify glibc symbol compatibility -- the
# binaries are built against glibc 2.35 and may use 2.35-only symbols.
# UCCL collectives has been tested and working on Rocky Linux 9.4 with this
# retagged wheel.
UCCL_WHEEL_PLAT=""
if [[ -f /etc/os-release ]]; then
  HOST_ID=$(. /etc/os-release && echo "${ID:-}")
  HOST_VERSION_ID=$(. /etc/os-release && echo "${VERSION_ID:-}")
  if [[ "$HOST_ID" == "rocky" && "$HOST_VERSION_ID" == 9* ]]; then
    UCCL_WHEEL_PLAT="manylinux_2_34_${ARCH}"
    echo "Rocky Linux 9 detected wheel will be tagged ${UCCL_WHEEL_PLAT}"
  fi
fi

########################################################
# 3. Clean up previous builds
########################################################
rm -rf uccl.egg-info >/dev/null 2>&1 || true
rm -rf dist >/dev/null 2>&1 || true
rm -rf build >/dev/null 2>&1 || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -rf "${WHEEL_DIR}" >/dev/null 2>&1 || true
mkdir -p "${WHEEL_DIR}"

########################################################
# 4. Determine the Docker image to use based on the target and architecture
########################################################
if [[ $TARGET == "cuda" ]]; then
  # default is cuda 12.8 from `nvidia/cuda:12.8.0-devel-ubuntu22.04`
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
  # default is latest rocm 7 version from `rocm/dev-ubuntu-22.04`
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

########################################################
# 5. Detect stale builder image, if a builder image exists
########################################################
hash_image=$(${CONTAINER_ENGINE} images -q ${IMAGE_NAME})
if [[ "${hash_image}" != "" ]]; then

  # Get its and its dockerfile's timestamps
  ts_dockerfile=$(date -r ${DOCKERFILE} --iso-8601=seconds)
  ts_image=$(${CONTAINER_ENGINE} inspect -f '{{ .Created }}' ${IMAGE_NAME})

  # If image is stale, suggest deleting & purging it
  if [[ "${ts_dockerfile}" > "${ts_image}" ]]; then
    echo "WARNING: builder image '${IMAGE_NAME}' is older than its source (${DOCKERFILE})" >&2
    echo "Please consider removing it, pruning the builder cache, and retrying the build to regenerate it." >&2
    echo " " >&2
    echo "  $ ${CONTAINER_ENGINE} image rm '${IMAGE_NAME}'" >&2
    echo "  $ ${CONTAINER_ENGINE} buildx prune -f" >&2
    echo " " >&2
    echo "NOTE: this may also prune unrelated builder cache images!" >&2
    sleep 1
  fi
fi

########################################################
# 6. Build the builder image (contains toolchain + CUDA/ROCm)
########################################################
echo "Building container image ${IMAGE_NAME} using ${DOCKERFILE} (engine: ${CONTAINER_ENGINE})..."
echo "Python version: ${PY_VER}"
if [[ "$TARGET" == "therock" ]]; then
  echo "ROCm index URL: ${ROCM_IDX_URL}"
fi
BUILD_ARGS="--build-arg PY_VER=${PY_VER}"
if [[ -n "${BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=" --build-arg BASE_IMAGE=${BASE_IMAGE}"
fi
if [[ "$ARCH" == "aarch64" ]]; then
  sudo ${CONTAINER_ENGINE} build --platform=linux/arm64 $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  sudo ${CONTAINER_ENGINE} build $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

########################################################
# 7. Run build inside container
########################################################
echo "Running build inside container..."

# Build container run command – podman runs as --user root since it works well
# with NFS volume permissions and is suitable for development and testing;
# docker uses the host uid:gid.
CONTAINER_RUN_ARGS=(run --rm)
if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
  CONTAINER_RUN_ARGS+=(--user root)
else
  CONTAINER_RUN_ARGS+=(--user "$(id -u):$(id -g)")
fi

sudo ${CONTAINER_ENGINE} "${CONTAINER_RUN_ARGS[@]}" \
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
  -e USE_DIETGPU="${USE_DIETGPU:-0}" \
  -e USE_IB="${USE_IB:-0}" \
  -e USE_TCP="${USE_TCP:-0}" \
  -e USE_INTEL_RDMA_NIC="${USE_INTEL_RDMA_NIC:-0}" \
  -e MAKE_NORMAL_MODE="${MAKE_NORMAL_MODE:-}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
  -e DISABLE_AGGRESSIVE_ATOMIC="${DISABLE_AGGRESSIVE_ATOMIC:-0}" \
  -e UCCL_WHEEL_PLAT="${UCCL_WHEEL_PLAT:-}" \
  -e FUNCTION_DEF="$(declare -f rename_to_abi3 build_rccl_nccl_header build_ccl_rdma build_ccl_efa build_p2p build_ep build_ukernel)" \
  -w /io \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail
    eval "$FUNCTION_DEF"

    if [[ "$TARGET" == "therock" ]]; then

      # Setup requested Python (PyPA images have all versions pre-installed)
      PY_V=$(echo ${PY_VER} | tr -d .)
      export PATH=/opt/python/cp${PY_V}-cp${PY_V}/bin:$PATH

      # Python environment with ROCm from TheRock
      python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
      pip3 install --no-cache-dir --upgrade pip
      pip3 install --no-cache-dir build auditwheel pybind11 nanobind
      pip3 install --no-cache-dir rocm[libraries,devel] --index-url ${ROCM_IDX_URL}
    fi

    if [[ "$TARGET" == rocm* ]]; then
      build_rccl_nccl_header
    fi

    if [[ "$BUILD_TYPE" == "ccl_rdma" ]]; then
      build_ccl_rdma "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "ccl_efa" ]]; then
      build_ccl_efa "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "p2p" ]]; then
      build_p2p "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "ep" ]]; then
      build_ep "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "ukernel" ]]; then
      build_ukernel "$TARGET" "$ARCH" "$IS_EFA"
    elif [[ "$BUILD_TYPE" == "all" ]]; then
      build_ccl_rdma "$TARGET" "$ARCH" "$IS_EFA"
      build_ccl_efa "$TARGET" "$ARCH" "$IS_EFA"
      build_p2p "$TARGET" "$ARCH" "$IS_EFA"
      # build_ep "$TARGET" "$ARCH" "$IS_EFA"
      # build_ukernel "$TARGET" "$ARCH" "$IS_EFA"
    fi

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

    ls -lh uccl/
    ls -lh uccl/lib/
    python3 -m build

    if [[ "$TARGET" == "therock" ]]; then
      # Undo UCCL package dependence on TheRock wheels after the build is done
      mv ${BACKUP_FN} setup.py
    fi

    # The _abi3_stub.so is too minimal for auditwheel to detect libc
    # on its own, so we must supply --plat explicitly.
    if [[ -z "${UCCL_WHEEL_PLAT:-}" ]]; then
      GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])")
      UCCL_WHEEL_PLAT="manylinux_${GLIBC_VER//./_}_$(uname -m)"
    fi

    auditwheel repair dist/uccl-*.whl \
      --plat "${UCCL_WHEEL_PLAT}" \
      --exclude "libtorch*.so" \
      --exclude "libc10*.so" \
      --exclude "libibverbs.so.1" \
      --exclude "libcudart.so.12" \
      --exclude "libamdhip64.so.*" \
      --exclude "libcuda.so.1" \
      --exclude "libefa.so.1" \
      -w /io/${WHEEL_DIR}

    # auditwheel may emit compressed dual tags (e.g. manylinux_2_34.manylinux_2_35).
    # Collapse to the single requested platform tag via simple rename.
    cd /io/${WHEEL_DIR}
    for whl in uccl-*.whl; do
      new="${whl%%abi3-*}abi3-${UCCL_WHEEL_PLAT}.whl"
      [[ "$whl" != "$new" ]] && mv "$whl" "$new"
    done
    cd /io

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

########################################################
# 8. Print the built wheel
########################################################
echo "Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl-*.whl || true

########################################################
# 9. Optionally install the built wheel
########################################################
if [[ "$DO_INSTALL" == "1" ]]; then
  # Auto-detect uv vs pip
  if command -v uv &>/dev/null && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PIP_CMD="uv pip"
  else
    PIP_CMD="pip"
  fi
  echo "Installing uccl wheel (using ${PIP_CMD})..."
  ${PIP_CMD} install -r requirements.txt
  ${PIP_CMD} uninstall uccl -y 2>/dev/null || true
  if [[ "$TARGET" != "therock" ]]; then
    ${PIP_CMD} install "${WHEEL_DIR}"/uccl-*.whl --no-deps
  else
    ${PIP_CMD} install --extra-index-url "${ROCM_IDX_URL}" "$(ls "${WHEEL_DIR}"/uccl-*.whl)[rocm]"
  fi

  UCCL_INSTALL_PATH=$(${PIP_CMD} show uccl 2>/dev/null | grep "^Location:" | cut -d' ' -f2 || echo "")
  if [[ -n "$UCCL_INSTALL_PATH" && -d "$UCCL_INSTALL_PATH" ]]; then
    UCCL_PACKAGE_PATH="$UCCL_INSTALL_PATH/uccl"
    if [[ -d "$UCCL_PACKAGE_PATH" ]]; then
      echo "UCCL installed at: $UCCL_PACKAGE_PATH"
      echo "Set LIBRARY_PATH: export LIBRARY_PATH=\"$UCCL_PACKAGE_PATH/lib:\$LIBRARY_PATH\""
    else
      echo "UCCL package directory not found at: $UCCL_PACKAGE_PATH"
    fi
  else
    echo "Warning: Could not detect UCCL installation path"
  fi
fi
