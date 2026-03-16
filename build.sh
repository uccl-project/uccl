#!/bin/bash

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) backends/targets.
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker/Podman image derived from Ubuntu 22.04.
#
# Usage:
#   ./build.sh [cu12|cu13|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image] [--install]
#
# Environment Variables:
#   CONTAINER_ENGINE=podman         Use podman instead of docker.
#   CONTAINER_ENGINE=apptainer      Use apptainer instead of docker/podman. 
#                                       It will create and remove 2 temp files in /tmp
#                          Example: CONTAINER_ENGINE=podman ./build.sh cu12 all
#   USE_INTEL_RDMA_NIC=1   Enable Intel RDMA NIC support (irdma driver, vendor 0x8086)
#                          Example: USE_INTEL_RDMA_NIC=1 ./build.sh cu12 ccl_efa
#   UCCL_WHEEL_ENABLE_FORCE_RETAG=1  Allow retagging the wheel to the host's
#                          glibc version when it differs from the container's.
#                          By default the wheel keeps the container's glibc tag.
#                          WARNING: the wheel is still built against the
#                          container's glibc and may use symbols not present in
#                          an older host glibc.
#                          Example: UCCL_WHEEL_ENABLE_FORCE_RETAG=1 ./build.sh cu12 all
#
# The wheels are written to wheelhouse-[cu12|cu13|rocm|therock]
# -----------------------

set -euo pipefail
source shared.sh

########################################################
# utilities
########################################################

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'  # Orange-ish yellow
RED='\033[0;31m'
NC='\033[0m' # No Color

# Info message (green)
msg_info() { 
    echo -e "${GREEN}[build.sh] INFO: $*${NC}"
}

# Warning message (yellow/orange)
msg_warning() { 
    echo -e "${YELLOW}[build.sh] WARNING: $*${NC}"
}

# Error message (red)
msg_error() { 
    echo -e "${RED}[build.sh] ERROR: $*${NC}" >&2; exit 1;
}

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

TARGET=${POSITIONAL_ARGS[0]:-cu12}
BUILD_TYPE=${POSITIONAL_ARGS[1]:-all}
PY_VER=${POSITIONAL_ARGS[2]:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
# The default for ROCM_IDX_URL depends on the gfx architecture of your GPU and the index URLs may change.
ROCM_IDX_URL=${POSITIONAL_ARGS[3]:-https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu}
# The default for THEROCK_BASE_IMAGE is current, but may change. Make sure to track TheRock's dockerfile.
THEROCK_BASE_IMAGE=${POSITIONAL_ARGS[4]:-quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017}

if [[ $TARGET != cu* && $TARGET != rocm* && $TARGET != "therock" ]]; then
  msg_error "Usage: $0 [cu12|cu13|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url] [therock_base_image] [--install]" >&2
fi

########################################################
# 2. Detect host architecture, container engine, EFA support, etc.
########################################################
ARCH="$(uname -m)"
if [[ $ARCH == "aarch64" && ($TARGET == rocm* || $TARGET == "therock") ]]; then
  msg_error "Skipping ROCm build on Arm64 (no ROCm toolchain)."
fi

# Container engine: `docker` (default), `podman`, or `apptainer`.
CONTAINER_ENGINE=${CONTAINER_ENGINE:-docker}
VALID_ENGINES=("docker" "podman" "apptainer")

if [[ ! " ${VALID_ENGINES[*]} " =~ " ${CONTAINER_ENGINE} " ]]; then
  msg_error "Invalid CONTAINER_ENGINE: ${CONTAINER_ENGINE}"
fi

# Auto-detect CUDA architecture for ep build, auto-detect ROCm architecture for ep build
DETECTED_GPU_ARCH=""
if [[ "$BUILD_TYPE" =~ (ep|all|p2p) ]]; then
  if [[ "$TARGET" == cu* ]] && command -v nvidia-smi &>/dev/null; then
    DETECTED_GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"

    if [[ -n "$DETECTED_GPU_ARCH" ]]; then
      msg_info "Auto-detected CUDA compute capability: ${DETECTED_GPU_ARCH}"
    fi
  elif [[ "$TARGET" == rocm* ]] && command -v amd-smi &>/dev/null; then
    # Check if jq is installed, install via pip if not
    if ! command -v jq &>/dev/null; then
      msg_info "jq not found, installing via pip..."
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
      msg_info "Auto-detected ROCm architecture: ${DETECTED_GPU_ARCH}"
    fi
  else
    msg_info "No compatible GPU detection tool found, skipping auto-detect"
  fi
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_GPU_ARCH}}"
IS_EFA="${IS_EFA:-$([ -d "/sys/class/infiniband/" ] && ls /sys/class/infiniband/ 2>/dev/null | grep -q rdmap && echo "EFA support: true")}" || echo "EFA support: false"

# Auto-derive UCCL_LOCAL_VERSION from target when not explicitly set.
# Default cu12 (no EFA) gets no local version (published to PyPI).
# TheRock computes its local version inside the container from rocm-sdk.
if [[ -z "${UCCL_LOCAL_VERSION+x}" ]]; then
  if [[ "$TARGET" == "cu12" && -z "$IS_EFA" ]]; then
    UCCL_LOCAL_VERSION=""
  elif [[ "$TARGET" == cu* ]]; then
    if [[ -n "$IS_EFA" ]]; then
      UCCL_LOCAL_VERSION="${TARGET}.efa"
    else
      UCCL_LOCAL_VERSION="${TARGET}"
    fi
  elif [[ "$TARGET" == rocm* || "$TARGET" == "therock" ]]; then
    UCCL_LOCAL_VERSION="rocm"
  fi
fi

# The build container produces wheels tagged with the container's glibc
# version by default.  If the host has an older glibc (e.g. Rocky Linux 9.x
# ships glibc 2.34), the wheel won't install there.  Set
# UCCL_WHEEL_ENABLE_FORCE_RETAG=1 to retag the wheel to the host glibc.
# Note: this does not verify glibc symbol compatibility -- the binaries are
# built against the container's glibc and may use newer symbols.  UCCL
# collectives and ep has been tested and working on Rocky Linux 9.4 with
# this retagged wheel.
HOST_GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])")

########################################################
# 3. Clean up previous builds
########################################################
rm -r uccl.egg-info >/dev/null 2>&1 || true
rm -r dist >/dev/null 2>&1 || true
rm -r build >/dev/null 2>&1 || true
rm collective/rdma/*.so >/dev/null 2>&1 || true
rm collective/efa/*.so >/dev/null 2>&1 || true
rm p2p/*.so >/dev/null 2>&1 || true
rm ep/*.so >/dev/null 2>&1 || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -r "${WHEEL_DIR}" >/dev/null 2>&1 || true
mkdir -p "${WHEEL_DIR}"

########################################################
# 4. Determine the Docker image to use based on the target and architecture
#    Override IMAGE_NAME and/or DOCKERFILE via env vars to force a specific
#    image (e.g. EFA on a non-EFA host, or pre-pulled images in CI).
#
#    Override IMAGE_NAME and/or DOCKERFILE via env vars to force a specific
#    image (e.g. EFA on a non-EFA host, or pre-pulled images in CI).
########################################################
if [[ $TARGET == "cu12" ]]; then
  # default is CUDA 12.8 from `nvidia/cuda:12.8.0-devel-ubuntu22.04`
  if [[ "$ARCH" == "aarch64" ]]; then
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=12
        : "${DOCKERFILE:=containers/apptainer/gh.def}"
        : "${BASE_IMAGE:=nvidia/cuda:12.8.0-devel-ubuntu22.04}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.gh}"
    fi
    : "${IMAGE_NAME:=uccl-builder-gh}"
  elif [[ -n "$IS_EFA" ]]; then
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=12
        : "${DOCKERFILE:=containers/apptainer/efa.def}"
        : "${BASE_IMAGE:=nvidia/cuda:12.8.0-devel-ubuntu22.04}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.efa}"
    fi
    : "${IMAGE_NAME:=uccl-builder-efa}"
  else
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=12
        : "${DOCKERFILE:=containers/apptainer/cuda.def}"
        : "${BASE_IMAGE:=nvidia/cuda:12.8.0-devel-ubuntu22.04}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.cuda}"
    fi
    : "${IMAGE_NAME:=uccl-builder-cuda}"
  fi
elif [[ $TARGET == "cu13" ]]; then
  : "${BASE_IMAGE:=nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04}"
  if [[ "$ARCH" == "aarch64" ]]; then
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=13
        : "${DOCKERFILE:=containers/apptainer/gh.def}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.gh}"
    fi
    : "${IMAGE_NAME:=uccl-builder-gh13}"
  elif [[ -n "$IS_EFA" ]]; then
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=13
        : "${DOCKERFILE:=containers/apptainer/efa.def}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.efa}"
    fi
    : "${IMAGE_NAME:=uccl-builder-efa13}"
  else
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=13
        : "${DOCKERFILE:=containers/apptainer/cuda.def}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.cuda}"
    fi
    : "${IMAGE_NAME:=uccl-builder-cuda13}"
  fi
elif [[ $TARGET == "rocm" ]]; then
  # default is latest rocm 7 version from `rocm/dev-ubuntu-22.04`
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=""
        : "${DOCKERFILE:=containers/apptainer/rocm.def}"
        : "${BASE_IMAGE:=rocm/dev-ubuntu-22.04}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.rocm}"
    fi  
    : "${IMAGE_NAME:=uccl-builder-rocm}"
elif [[ $TARGET == "rocm6" ]]; then
    : "${BASE_IMAGE:=rocm/dev-ubuntu-22.04:6.4.3-complete}"
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=6
        : "${DOCKERFILE:=containers/apptainer/rocm.def}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.rocm}"
    fi
    : "${IMAGE_NAME:=uccl-builder-rocm6}"
elif [[ $TARGET == "therock" ]]; then
    BASE_IMAGE="${THEROCK_BASE_IMAGE}"
    if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
        # IMAGE_VERSION=""
        : "${DOCKERFILE:=containers/apptainer/therock.def}"
    else
        : "${DOCKERFILE:=containers/docker/Dockerfile.therock}"
    fi
    : "${IMAGE_NAME:=uccl-builder-therock}"
fi

# IMAGE_NAME, IMAGE_VERSION
if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
    # Add extension: IMAGE_NAME-IMAGE_VERSION.sif if IMAGE_VERSION is set, otherwise just IMAGE_NAME.sif
    # IMAGE_NAME="${IMAGE_NAME}${IMAGE_VERSION:+-${IMAGE_VERSION}}.sif"
    IMAGE_NAME="${IMAGE_NAME}.sif"
fi

########################################################
# 5. Detect stale builder image, if a builder image exists
########################################################
if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
    if [[ -f "${IMAGE_NAME}" ]]; then
        msg_warning "Apptainer image ${IMAGE_NAME}.sif already exists. Please consider removing it if you want to rebuild the image from ${DOCKERFILE}."
        sleep 5
    fi
else
    hash_image=$(${CONTAINER_ENGINE} images -q ${IMAGE_NAME})
    if [[ "${hash_image}" != "" ]]; then

        # Get its and its dockerfile's timestamps
        ts_dockerfile=$(date -r ${DOCKERFILE} --iso-8601=seconds)
        ts_image=$(${CONTAINER_ENGINE} inspect -f '{{ .Created }}' ${IMAGE_NAME})

        # If image is stale, suggest deleting & purging it
        if [[ "${ts_dockerfile}" > "${ts_image}" ]]; then
            msg_warning "WARNING: builder image '${IMAGE_NAME}' is older than its source (${DOCKERFILE})" >&2
            msg_warning "Please consider removing it, pruning the builder cache, and retrying the build to regenerate it." >&2
            msg_warning " " >&2
            msg_warning "  $ ${CONTAINER_ENGINE} image rm '${IMAGE_NAME}'" >&2
            msg_warning "  $ ${CONTAINER_ENGINE} buildx prune -f" >&2
            msg_warning " " >&2
            msg_warning "NOTE: this may also prune unrelated builder cache images!" >&2
            sleep 1
        fi
    fi
fi

########################################################
# 6. Build the builder image (contains toolchain + CUDA/ROCm)
# Set SKIP_DOCKER_BUILD=1 to use a pre-pulled/tagged image (e.g. from GHCR in CI)
########################################################

if [[ "$TARGET" == "therock" ]]; then
  msg_info "ROCm index URL: ${ROCM_IDX_URL}"
fi

if [[ "${SKIP_DOCKER_BUILD:-0}" != "1" ]]; then
  msg_info "[1/3] Building container image ${IMAGE_NAME} using ${DOCKERFILE} (engine: ${CONTAINER_ENGINE})... (Python version: ${PY_VER})"

  BUILD_ARGS="--build-arg PY_VER=${PY_VER}"

  if [[ -n "${BASE_IMAGE:-}" ]]; then
    BUILD_ARGS+=" --build-arg BASE_IMAGE=${BASE_IMAGE}"
  fi

  if [[ "${CONTAINER_ENGINE}" == "apptainer" ]]; then
    msg_info "Using Apptainer, base image: ${BASE_IMAGE}, definition file: ${DOCKERFILE}"

    # If the image already exists, skip the build
    if [[ -f "$IMAGE_NAME" ]]; then
      msg_warning "Apptainer image ${IMAGE_NAME} already exists. Recreating..."
      # TODO: rm -f "$IMAGE_NAME"
    fi

    # msg_info "Building Apptainer image ${IMAGE_NAME}"
  
    # # Ensure definition file
    # if [[ "$DOCKERFILE" != *.def && "$DOCKERFILE" != *.def.template ]]; then
    #   msg_error "Apptainer requires a .def or .def.template file"
    # fi

    # BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"
    # PY_VER="${PY_VER:-3.10}"

    # apptainer build --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg PY_VER=$PY_VER $IMAGE_NAME $DOCKERFILE
    
  else

    if [[ "$ARCH" == "aarch64" ]]; then
      ${CONTAINER_ENGINE} build \
        --platform=linux/arm64 \
        $BUILD_ARGS \
        -t "$IMAGE_NAME" \
        -f "$DOCKERFILE" .
    else
      ${CONTAINER_ENGINE} build \
        $BUILD_ARGS \
        -t "$IMAGE_NAME" \
        -f "$DOCKERFILE" .
    fi

  fi
else
  msg_info "[1/3] Skipping Docker build (SKIP_DOCKER_BUILD=1), using existing image: ${IMAGE_NAME}"
fi

########################################################
# 7. Run build inside container
########################################################
msg_info "Building inside container ${IMAGE_NAME} (engine: ${CONTAINER_ENGINE})..."

INNER_SCRIPT=$(cat << 'INNER_SCRIPT'
set -euo pipefail

# Source the function definitions if they were passed as a file
if [[ -f /tmp/function_def.sh ]]; then
  source /tmp/function_def.sh
fi

# ADD THIS LINE - source shared.sh which contains the build functions
source /io/shared.sh

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
elif [[ "$BUILD_TYPE" == "p2p_ep" ]]; then
  build_p2p "$TARGET" "$ARCH" "$IS_EFA"
  build_ep "$TARGET" "$ARCH" "$IS_EFA"
elif [[ "$BUILD_TYPE" == "ukernel" ]]; then
  build_ukernel "$TARGET" "$ARCH" "$IS_EFA"
elif [[ "$BUILD_TYPE" == "all" ]]; then
  if [[ -n "$IS_EFA" ]]; then
    build_ccl_efa "$TARGET" "$ARCH" "$IS_EFA"
  else
    build_ccl_rdma "$TARGET" "$ARCH" "$IS_EFA"
  fi
  build_p2p "$TARGET" "$ARCH" "$IS_EFA"
  build_ep "$TARGET" "$ARCH" "$IS_EFA"
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
# Always use the *container* glibc for auditwheel repair (symbol
# validation), then retag the wheel to the desired host platform
# afterwards if UCCL_WHEEL_ENABLE_FORCE_RETAG=1.
CONTAINER_GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])")
AUDIT_PLAT="manylinux_${CONTAINER_GLIBC_VER//./_}_$(uname -m)"

# Decide the final wheel platform tag
if [[ "${UCCL_WHEEL_ENABLE_FORCE_RETAG}" == "1" ]]; then
  UCCL_WHEEL_PLAT="manylinux_${HOST_GLIBC_VER//./_}_$(uname -m)"
  if [[ "${UCCL_WHEEL_PLAT}" != "${AUDIT_PLAT}" ]]; then
    echo "WARNING: UCCL_WHEEL_ENABLE_FORCE_RETAG is set." >&2
    echo "  The wheel will be retagged from ${AUDIT_PLAT} to ${UCCL_WHEEL_PLAT}." >&2
    echo "  The binaries are built against the container glibc (${CONTAINER_GLIBC_VER})." >&2
    echo "  If the host glibc is older, the wheel may fail at runtime" >&2
    echo "  due to missing versioned symbols." >&2
  fi
  echo "Host glibc ${HOST_GLIBC_VER}, container glibc ${CONTAINER_GLIBC_VER} -> wheel tagged ${UCCL_WHEEL_PLAT} (force-retag enabled)"
else
  UCCL_WHEEL_PLAT="${AUDIT_PLAT}"
  echo "Container glibc ${CONTAINER_GLIBC_VER} -> wheel tagged ${UCCL_WHEEL_PLAT}"
  if [[ "${HOST_GLIBC_VER}" != "${CONTAINER_GLIBC_VER}" ]]; then
    echo "  Note: host glibc (${HOST_GLIBC_VER}) differs from container glibc (${CONTAINER_GLIBC_VER})."
    echo "  Tip: set UCCL_WHEEL_ENABLE_FORCE_RETAG=1 to retag to host glibc ${HOST_GLIBC_VER}."
  fi
fi

auditwheel repair dist/uccl-*.whl \
  --plat "${AUDIT_PLAT}" \
  --exclude "libtorch*.so" \
  --exclude "libc10*.so" \
  --exclude "libibverbs.so.1" \
  --exclude "libcudart.so.12" \
  --exclude "libamdhip64.so.*" \
  --exclude "libcuda.so.1" \
  --exclude "libefa.so.1" \
  --exclude "libglog.so.0" \
  -w /io/${WHEEL_DIR}

# auditwheel may emit compressed dual tags (e.g. manylinux_2_34.manylinux_2_35).
# Collapse to the single requested platform tag via simple rename.
cd /io/${WHEEL_DIR}
for whl in uccl*.whl; do
  if [[ "$whl" == *-abi3-* ]]; then
    new="${whl%%abi3-*}abi3-${UCCL_WHEEL_PLAT}.whl"
  else
    new="${whl%%-manylinux*}-${UCCL_WHEEL_PLAT}.whl"
  fi
  [[ "$whl" != "$new" ]] && mv "$whl" "$new"
done
cd /io

# Add local version identifier to wheel filename (PEP 440, vLLM-style).
# UCCL_LOCAL_VERSION is set by the caller (e.g. "cu13", "cu12.efa", "rocm").
# TheRock auto-computes it from rocm-sdk version.
# If empty, the wheel keeps its base version (default cu12 for PyPI).
if [[ "$TARGET" == "therock" ]]; then
  UCCL_LOCAL_VERSION="rocm$(rocm-sdk version)"
fi
if [[ -n "${UCCL_LOCAL_VERSION:-}" ]]; then
  cd /io/${WHEEL_DIR}
  for wheel in uccl*.whl; do
    if [[ -f "$wheel" ]]; then
      if [[ "$wheel" =~ ^(uccl[^-]*-)([^-]+)-([^-]+-[^-]+-.+)(\.whl)$ ]]; then
        name="${BASH_REMATCH[1]}"
        version="${BASH_REMATCH[2]}"
        python_abi_platform="${BASH_REMATCH[3]}"
        suffix="${BASH_REMATCH[4]}"
        new_wheel="${name}${version}+${UCCL_LOCAL_VERSION}-${python_abi_platform}${suffix}"
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
INNER_SCRIPT
)

# Build container run command with appropriate arguments for each engine
if [[ "$CONTAINER_ENGINE" == "apptainer" ]]; then
  # Apptainer run arguments
  CONTAINER_RUN_ARGS=(
    exec
    --cleanenv
    --pwd /io
    --bind "$(pwd):/io"
  )
  
  # Add GPU support if needed
  if [[ "$TARGET" == cuda* ]]; then
    # Check if we have nvidia-container-cli or if --nv is supported
    if $CONTAINER_ENGINE exec --help | grep -q -- --nv; then
      CONTAINER_RUN_ARGS+=(--nv)  # Enable NVIDIA GPU support
      msg_info "Enabling NVIDIA GPU support with --nv flag"
    else
      msg_warning "Warning: --nv flag not supported, GPU may not be available"
    fi
  elif [[ "$TARGET" == rocm* || "$TARGET" == "therock" ]]; then
    # Check if --rocm flag is supported
    if $CONTAINER_ENGINE exec --help | grep -q -- --rocm; then
      CONTAINER_RUN_ARGS+=(--rocm)  # Enable ROCm support
      msg_info "Enabling ROCm GPU support with --rocm flag"
    else
      msg_warning "Warning: --rocm flag not supported, GPU may not be available"
    fi
  fi
  
  # Set environment variables for apptainer
  for env_var in TARGET PY_VER ARCH ROCM_IDX_URL IS_EFA WHEEL_DIR BUILD_TYPE \
               USE_TCPX USE_EFA USE_IB USE_TCP USE_DIETGPU USE_INTEL_RDMA_NIC \
               PER_EXPERT_BATCHING MAKE_NORMAL_MODE TORCH_CUDA_ARCH_LIST \
               DISABLE_AGGRESSIVE_ATOMIC HOST_GLIBC_VER UCCL_WHEEL_ENABLE_FORCE_RETAG; do
    # Use variable with default empty string
    value="${!env_var-}"
    CONTAINER_RUN_ARGS+=(--env "$env_var=$value")
  done
  
  # Add function definitions as a file for apptainer (can't pass via env easily due to size)
  FUNCTION_DEF_FILE="/tmp/function_def_$$.sh"
  trap "rm -f $FUNCTION_DEF_FILE" EXIT
  declare -f rename_to_abi3 build_rccl_nccl_header build_ccl_rdma build_ccl_efa build_p2p build_ep build_ukernel > "$FUNCTION_DEF_FILE"
  
  # Write the inner script to a temporary file
  INNER_SCRIPT_FILE="/tmp/inner_script_$$.sh"
  trap "rm -f $INNER_SCRIPT_FILE $FUNCTION_DEF_FILE" EXIT
  printf '%s\n' "$INNER_SCRIPT" > "$INNER_SCRIPT_FILE"

  
  # Run the build command
  $CONTAINER_ENGINE "${CONTAINER_RUN_ARGS[@]}" \
    --bind "$FUNCTION_DEF_FILE":/tmp/function_def.sh:ro \
    --bind "$INNER_SCRIPT_FILE":/tmp/inner_script.sh:ro \
    "$IMAGE_NAME" \
    /bin/bash /tmp/inner_script.sh
  
  # Clean up temp files
  rm -f "$FUNCTION_DEF_FILE" "$INNER_SCRIPT_FILE"
  
else
  # Docker/Podman run command
  CONTAINER_RUN_ARGS=(run --rm)
  if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
    CONTAINER_RUN_ARGS+=(--user root)
  else
    CONTAINER_RUN_ARGS+=(--user "$(id -u):$(id -g)")
  fi

  ${CONTAINER_ENGINE} "${CONTAINER_RUN_ARGS[@]}" \
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
    -e USE_DIETGPU="${USE_DIETGPU:-0}" \
    -e USE_INTEL_RDMA_NIC="${USE_INTEL_RDMA_NIC:-0}" \
    -e PER_EXPERT_BATCHING="${PER_EXPERT_BATCHING:-0}" \
    -e MAKE_NORMAL_MODE="${MAKE_NORMAL_MODE:-}" \
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
    -e DISABLE_AGGRESSIVE_ATOMIC="${DISABLE_AGGRESSIVE_ATOMIC:-0}" \
    -e HOST_GLIBC_VER="${HOST_GLIBC_VER}" \
    -e UCCL_WHEEL_ENABLE_FORCE_RETAG="${UCCL_WHEEL_ENABLE_FORCE_RETAG:-0}" \
    -e UCCL_WHEEL_PLAT="${UCCL_WHEEL_PLAT:-}" \
    -e UCCL_LOCAL_VERSION="${UCCL_LOCAL_VERSION:-}" \
    -e UCCL_PACKAGE_NAME="${UCCL_PACKAGE_NAME:-uccl-${TARGET}}" \
    -e UCCL_SKIP_LOCAL_VERSION="${UCCL_SKIP_LOCAL_VERSION:-0}" \
    -e FUNCTION_DEF="$(declare -f rename_to_abi3 build_rccl_nccl_header build_ccl_rdma build_ccl_efa build_p2p build_ep build_ukernel)" \
    -w /io \
    "$IMAGE_NAME" /bin/bash -c "$INNER_SCRIPT"
fi

########################################################
# 8. Print the built wheel
########################################################
msg_info "Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl*.whl || true

########################################################
# 9. Optionally install the built wheel
########################################################
if [[ "$DO_INSTALL" == "1" ]]; then
  # Install for the default "python" so "python -c 'from uccl import ep'" works.
  PYTHON_CMD="python"
  if ! command -v python &>/dev/null; then
    PYTHON_CMD="python3"
  fi
  if command -v uv &>/dev/null && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PIP_CMD="uv pip"
  else
    PIP_CMD="${PYTHON_CMD} -m pip"
  fi
  
  msg_info "Installing uccl wheel for ${PYTHON_CMD} (using ${PIP_CMD})..."
  ${PIP_CMD} install -r requirements.txt
  # Uninstall any previous uccl so pip doesn't skip with "already installed".
  ${PIP_CMD} uninstall uccl -y 2>/dev/null || true
  UCCL_CLEANUP_DIR="$(${PYTHON_CMD} -c "import site; print(site.getsitepackages()[0])")/uccl"
  if [[ -d "$UCCL_CLEANUP_DIR" ]]; then
    msg_info "Cleaning up stale files in $UCCL_CLEANUP_DIR"
    rm -r "$UCCL_CLEANUP_DIR"
  fi
  if [[ "$TARGET" != "therock" ]]; then
    ${PIP_CMD} install "${WHEEL_DIR}"/uccl*.whl --no-deps
  else
    ${PIP_CMD} install --extra-index-url "${ROCM_IDX_URL}" "$(ls "${WHEEL_DIR}"/uccl*.whl)[rocm]"
  fi


  UCCL_INSTALL_PATH=$(${PIP_CMD} show uccl 2>/dev/null | grep "^Location:" | cut -d' ' -f2 || true)
  if [[ -n "$UCCL_INSTALL_PATH" && -d "$UCCL_INSTALL_PATH" ]]; then
    UCCL_PACKAGE_PATH="$UCCL_INSTALL_PATH/uccl"
    if [[ -d "$UCCL_PACKAGE_PATH" ]]; then
      msg_info "UCCL installed at: $UCCL_PACKAGE_PATH"
      msg_info "Set LIBRARY_PATH: export LIBRARY_PATH=\"$UCCL_PACKAGE_PATH/lib:\$LIBRARY_PATH\""
    else
      msg_warning "UCCL package directory not found at: $UCCL_PACKAGE_PATH"
    fi
  else
    msg_warning "Warning: Could not detect UCCL installation path"
  fi
fi