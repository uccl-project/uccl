#!/bin/bash
set -e

# -----------------------
# Build uccl_ep wheel for CUDA (NVIDIA) and ROCm (AMD) backends.
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker/Podman image derived from Ubuntu 22.04.
#
# This script reuses the shared Dockerfiles from the parent uccl repo
# (../docker/).
#
# Usage:
#   ./build.sh [cuda|rocm] [py_version] [--install]
#
# Environment Variables:
#   CONTAINER_ENGINE=podman  Use podman instead of docker.
#   USE_INTEL_RDMA_NIC=1     Enable Intel RDMA NIC support.
#
# The wheels are written to wheelhouse-[cuda|rocm]
# -----------------------

# Parse arguments: positional args first, then flags
DO_INSTALL=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    *) POSITIONAL_ARGS+=("$arg") ;;
  esac
done

TARGET=${POSITIONAL_ARGS[0]:-cuda}
PY_VER=${POSITIONAL_ARGS[1]:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
ARCH="$(uname -m)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UCCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Container engine: "docker" (default) or "podman"
CONTAINER_ENGINE=${CONTAINER_ENGINE:-docker}
if [[ "$CONTAINER_ENGINE" != "docker" && "$CONTAINER_ENGINE" != "podman" ]]; then
  echo "Error: CONTAINER_ENGINE must be 'docker' or 'podman', got '${CONTAINER_ENGINE}'" >&2
  exit 1
fi

IS_EFA=$( [ -d "/sys/class/infiniband/" ] && ls /sys/class/infiniband/ 2>/dev/null | grep -q rdmap && echo "EFA support: true" ) || echo "EFA support: false"

if [[ $TARGET != cuda* && $TARGET != rocm* ]]; then
  echo "Usage: $0 [cuda|rocm] [py_version] [--install]" >&2
  exit 1
fi

if [[ $ARCH == "aarch64" && $TARGET == rocm* ]]; then
  echo "Skipping ROCm build on Arm64 (no ROCm toolchain)."
  exit 1
fi

# Clean previous build artifacts
cd "${SCRIPT_DIR}"
rm -rf build dist *.egg-info wheelhouse 2>/dev/null || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -rf "${WHEEL_DIR}" 2>/dev/null || true
mkdir -p "${WHEEL_DIR}"

# Determine the Docker image (shared Dockerfiles live in ../docker/)
if [[ $TARGET == "cuda" ]]; then
  if [[ "$ARCH" == "aarch64" ]]; then
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.gh"
    IMAGE_NAME="uccl-ep-builder-gh"
  elif [[ -n "$IS_EFA" ]]; then
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.efa"
    IMAGE_NAME="uccl-ep-builder-efa"
  else
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.cuda"
    IMAGE_NAME="uccl-ep-builder-cuda"
  fi
elif [[ $TARGET == "cuda13" ]]; then
  BASE_IMAGE="nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04"
  if [[ "$ARCH" == "aarch64" ]]; then
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.gh"
    IMAGE_NAME="uccl-ep-builder-gh"
  elif [[ -n "$IS_EFA" ]]; then
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.efa"
    IMAGE_NAME="uccl-ep-builder-efa"
  else
    DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.cuda"
    IMAGE_NAME="uccl-ep-builder-cuda"
  fi
elif [[ $TARGET == "rocm" ]]; then
  DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.rocm"
  IMAGE_NAME="uccl-ep-builder-rocm"
elif [[ $TARGET == "rocm6" ]]; then
  DOCKERFILE="${UCCL_ROOT}/docker/Dockerfile.rocm"
  BASE_IMAGE="rocm/dev-ubuntu-22.04:6.4.3-complete"
  IMAGE_NAME="uccl-ep-builder-rocm"
fi

# Detect stale builder image
hash_image=$(${CONTAINER_ENGINE} images -q ${IMAGE_NAME} 2>/dev/null || true)
if [[ -n "${hash_image}" ]]; then
  ts_dockerfile=$(date -r ${DOCKERFILE} --iso-8601=seconds)
  ts_image=$(${CONTAINER_ENGINE} inspect -f '{{ .Created }}' ${IMAGE_NAME})
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

# Build the builder image (contains toolchain + CUDA/ROCm)
echo "[1/3] Building container image ${IMAGE_NAME} using ${DOCKERFILE} (engine: ${CONTAINER_ENGINE})..."
echo "Python version: ${PY_VER}"
BUILD_ARGS="--build-arg PY_VER=${PY_VER}"
if [[ -n "${BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=" --build-arg BASE_IMAGE=${BASE_IMAGE}"
fi
if [[ "$ARCH" == "aarch64" ]]; then
  ${CONTAINER_ENGINE} build --platform=linux/arm64 $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" "${UCCL_ROOT}"
else
  ${CONTAINER_ENGINE} build $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE" "${UCCL_ROOT}"
fi

echo "[2/3] Running build inside container..."

# Auto-detect GPU architecture
DETECTED_GPU_ARCH=""
if [[ "$TARGET" == cuda* ]] && command -v nvidia-smi &> /dev/null; then
  DETECTED_GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
  if [[ -n "$DETECTED_GPU_ARCH" ]]; then
    echo "Auto-detected CUDA compute capability: ${DETECTED_GPU_ARCH}"
  fi
elif [[ "$TARGET" == rocm* ]] && command -v amd-smi &> /dev/null; then
  if ! command -v jq &> /dev/null; then
    echo "jq not found, installing via pip..."
    pip install jq
  fi
  DETECTED_GPU_ARCH="$(
    PYTHONWARNINGS=ignore \
    amd-smi static -g 0 --asic --json 2>/dev/null \
    | jq -r '
        if .gpu_data and (.gpu_data | length > 0) then
          .gpu_data[0].asic.target_graphics_version
        else
          empty
        end
      ' \
    || true
  )"
  if [[ -n "$DETECTED_GPU_ARCH" ]]; then
    echo "Auto-detected ROCm architecture: ${DETECTED_GPU_ARCH}"
  fi
else
  echo "[INFO] No compatible GPU detection tool found, skipping auto-detect"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_GPU_ARCH}}"

# Rocky Linux 9.x glibc 2.34 compat: retag the wheel after build
UCCL_WHEEL_PLAT=""
if [[ -f /etc/os-release ]]; then
  HOST_ID=$(. /etc/os-release && echo "${ID:-}")
  HOST_VERSION_ID=$(. /etc/os-release && echo "${VERSION_ID:-}")
  if [[ "$HOST_ID" == "rocky" && "$HOST_VERSION_ID" == 9* ]]; then
    UCCL_WHEEL_PLAT="manylinux_2_34_${ARCH}"
    echo "[INFO] Rocky Linux 9 detected, wheel will be tagged ${UCCL_WHEEL_PLAT}"
  fi
fi

# Build container run command
CONTAINER_RUN_ARGS=(run --rm)
if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
  CONTAINER_RUN_ARGS+=(--user root)
else
  CONTAINER_RUN_ARGS+=(--user "$(id -u):$(id -g)")
fi

# Mount the whole uccl repo so ep can access ../include/ headers
${CONTAINER_ENGINE} "${CONTAINER_RUN_ARGS[@]}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v "${UCCL_ROOT}":/io \
  -e TARGET="${TARGET}" \
  -e PY_VER="${PY_VER}" \
  -e ARCH="${ARCH}" \
  -e IS_EFA="${IS_EFA}" \
  -e USE_INTEL_RDMA_NIC="${USE_INTEL_RDMA_NIC:-0}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
  -e DISABLE_AGGRESSIVE_ATOMIC="${DISABLE_AGGRESSIVE_ATOMIC:-0}" \
  -e UCCL_WHEEL_PLAT="${UCCL_WHEEL_PLAT:-}" \
  -w /io/ep \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail

    echo "[container] Building uccl_ep wheel, TARGET=$TARGET"

    if [[ "${USE_INTEL_RDMA_NIC:-0}" == "1" ]]; then
      echo "[container] Building with Intel RDMA NIC support (USE_INTEL_RDMA_NIC=1)"
    fi

    if [[ "$TARGET" == "rocm" || "$TARGET" == "rocm6" ]]; then
      echo "[container] ROCm target"
    fi

    # bdist_wheel runs the extension build + auditwheel repair
    USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} python3 setup.py bdist_wheel

    # Copy wheels to output directory
    cp wheelhouse/uccl_ep-*.whl /io/ep/wheelhouse-${TARGET}/ 2>/dev/null || \
      cp dist/uccl_ep-*.whl /io/ep/wheelhouse-${TARGET}/

    # Rocky Linux retag
    if [[ -n "${UCCL_WHEEL_PLAT:-}" ]]; then
      echo "[container] Retagging wheel platform to ${UCCL_WHEEL_PLAT}"
      cd /io/ep/wheelhouse-${TARGET}
      for whl in uccl_ep-*.whl; do
        if [[ -f "$whl" ]]; then
          python3 -m wheel tags --platform-tag "${UCCL_WHEEL_PLAT}" --remove "$whl"
        fi
      done
      cd /io/ep
    fi

    # Add backend tag for ROCm
    if [[ "$TARGET" == rocm* ]]; then
      cd /io/ep/wheelhouse-${TARGET}
      for wheel in uccl_ep-*.whl; do
        if [[ -f "$wheel" ]]; then
          if [[ "$wheel" =~ ^(uccl_ep-)([^-]+)-([^-]+-[^-]+-.+)(\.whl)$ ]]; then
            name="${BASH_REMATCH[1]}"
            version="${BASH_REMATCH[2]}"
            python_abi_platform="${BASH_REMATCH[3]}"
            suffix="${BASH_REMATCH[4]}"
            new_wheel="${name}${version}+${TARGET}-${python_abi_platform}${suffix}"
            echo "Renaming wheel: $wheel -> $new_wheel"
            mv "$wheel" "$new_wheel"
          else
            echo "Warning: Could not parse wheel filename: $wheel"
          fi
        fi
      done
      cd /io/ep
    fi

    echo "[container] uccl_ep wheel build complete"
    ls -lh /io/ep/wheelhouse-${TARGET}/
  '

# 3. Done
echo "[3/3] uccl_ep wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl_ep-*.whl || true

# 4. Optionally install the built wheel
if [[ "$DO_INSTALL" == "1" ]]; then
  # Auto-detect uv vs pip
  if command -v uv &> /dev/null && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PIP_CMD="uv pip"
  else
    PIP_CMD="pip"
  fi
  echo "[4/4] Installing uccl_ep wheel (using ${PIP_CMD})..."
  ${PIP_CMD} uninstall uccl_ep -y 2>/dev/null || true
  ${PIP_CMD} install "${WHEEL_DIR}"/uccl_ep-*.whl --no-deps
  echo "uccl_ep installed successfully. Test with: python -c 'import uccl_ep'"
fi
