#!/bin/bash

# -----------------------
# build_native.sh — compile uccl native modules.
#
# Replaces the top-level Makefile.  Invoked by setup.py's
# ShellBuildExtension; can also be run standalone for in-tree builds.
# The compile logic is lifted (almost verbatim) from build_inner.sh's
# pre-Makefile incarnation.
#
# Output layout:
#   ${UCCL_PY_DIR}/lib/   library .so / .a files (libnccl-net-uccl.so, ...)
#   ${UCCL_PY_DIR}/       p2p*.so + collective.py + utils.py
#   ${UCCL_EP_DIR}/       ep_cpp*.so   (derived from UCCL_PY_DIR, see below)
#
# By default the script writes in-tree (UCCL_PY_DIR=./uccl), matching the
# source layout that the editable install relies on.  setup.py overrides
# UCCL_PY_DIR to point at the wheel-staging dir (build_lib/uccl) for
# non-inplace builds.  The matching ``uccl.ep`` target is derived
# automatically:
#   * source layout (UCCL_PY_DIR == <project>/uccl) -> ./ep/python/uccl_ep
#                                                     (matches package_dir)
#   * any other UCCL_PY_DIR (wheel staging)         -> ${UCCL_PY_DIR}/ep
# so the same script serves all three modes (editable / install / wheel)
# with a single env var from setup.py.
#
# Usage:
#   ./build_native.sh [BUILD_TYPE]
#     BUILD_TYPE := all (default) | ccl_rdma | ccl_efa | p2p | ep | p2p_ep | ukernel | clean
#     When no positional argument is supplied the value of $BUILD_TYPE is used.
#
# Environment variables consumed:
#   TARGET                Build target: cu12, cu13, roc7, roc6, therock (default cu12)
#   ARCH                  Host architecture: x86_64 or aarch64 (default $(uname -m))
#   IS_EFA                Non-empty when EFA is detected (swaps ccl_rdma -> ccl_efa)
#   BUILD_TYPE            Default value when no positional arg is given (default ``all``)
#
#   Output staging (set by setup.py during wheel/install builds):
#     UCCL_PY_DIR         Target dir for the ``uccl`` package (default ./uccl)
#     UCCL_EP_DIR         Target dir for the ``uccl.ep`` package
#                         (auto-derived from UCCL_PY_DIR; override only
#                         if you really know what you're doing)
#
#   Feature flags:
#     USE_DIETGPU         Enable DietGPU compression (default 0)
#     USE_INTEL_RDMA_NIC  Enable Intel RDMA NIC / irdma driver (default 0)
#     USE_DMABUF          Enable EP DMA-BUF GPU memory registration (default 0)
#     TORCH_CUDA_ARCH_LIST CUDA compute capabilities for dietgpu (default 9.0)
# -----------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

TARGET="${TARGET:-cu12}"
ARCH="${ARCH:-$(uname -m)}"
IS_EFA="${IS_EFA:-}"
BUILD_TYPE="${BUILD_TYPE:-all}"


UCCL_PY_DIR="${UCCL_PY_DIR:-${PROJECT_ROOT}/uccl}"

if [[ -z "${UCCL_EP_DIR:-}" ]]; then
  if [[ "$(realpath -m "${UCCL_PY_DIR}")" == "$(realpath -m "${PROJECT_ROOT}/uccl")" ]]; then
    # for build mode
    UCCL_EP_DIR="${PROJECT_ROOT}/ep/python/uccl_ep"
  else
    # for install/wheel mode
    UCCL_EP_DIR="${UCCL_PY_DIR}/ep"
  fi
fi
UCCL_LIB_DIR="${UCCL_PY_DIR}/lib"

# Positional argument overrides BUILD_TYPE.
if [[ $# -gt 0 ]]; then
  BUILD_TYPE="$1"
fi

mkdir -p "${UCCL_PY_DIR}" "${UCCL_LIB_DIR}" "${UCCL_EP_DIR}"

########################################################
# Build helper functions
########################################################

# Rename cpython-versioned .so files to .abi3.so for stable ABI compatibility.
# Only applies on Python >= 3.12 where nanobind stable ABI is enabled.
rename_to_abi3() {
  local dir="$1"
  local py_stable_abi_ok
  py_stable_abi_ok=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
  if [[ "$py_stable_abi_ok" != "1" ]]; then
    echo "Python < 3.12 detected, skipping abi3 rename (nanobind stable ABI not supported)"
    return
  fi
  for f in "$dir"/*.cpython-*.so; do
    if [[ -f "$f" ]]; then
      local newname
      newname=$(echo "$f" | sed 's/\.cpython-[^.]*-[^.]*-[^.]*\.so/.abi3.so/')
      echo "Renaming $(basename "$f") -> $(basename "$newname")"
      mv "$f" "$newname"
    fi
  done
}

build_rccl_nccl_header() {
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

  if [[ "${USE_INTEL_RDMA_NIC:-0}" == "1" ]]; then
    echo "[container] Building with Intel RDMA NIC support (USE_INTEL_RDMA_NIC=1)"
  fi

  if [[ "$TARGET" == cu* ]]; then
    cd collective/rdma && make clean && make -j$(nproc) USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} && cd ../../
    TARGET_SO=collective/rdma/libnccl-net-uccl.so
  elif [[ "$TARGET" == roc[67] ]]; then
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

  echo "[container] Copying RDMA .so to ${UCCL_LIB_DIR}"
  mkdir -p "${UCCL_LIB_DIR}"
  cp ${TARGET_SO} "${UCCL_LIB_DIR}/"
}

build_ccl_efa() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ccl_efa Target: $TARGET"

  if [[ "$ARCH" == "aarch64" || "$TARGET" == roc[67] || "$TARGET" == "therock" ]]; then
    echo "Skipping EFA build on Arm64 (no EFA installer) or ROCm (no CUDA)."
    return
  fi

  if [[ "${USE_INTEL_RDMA_NIC:-0}" == "1" ]]; then
    echo "[container] Building with Intel RDMA NIC support (USE_INTEL_RDMA_NIC=1)"
  fi

  cd collective/efa && make clean && make -j$(nproc) USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} && cd ../../

  # EFA requires a custom NCCL.
  cd thirdparty/nccl-sg
  make src.build -j$(nproc) NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0}
  cd ../..

  echo "[container] Copying EFA .so to ${UCCL_LIB_DIR}"
  mkdir -p "${UCCL_LIB_DIR}"
  cp collective/efa/libnccl-net-efa.so "${UCCL_LIB_DIR}/"
  cp thirdparty/nccl-sg/build/lib/libnccl.so "${UCCL_LIB_DIR}/libnccl-efa.so"
}

build_p2p() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_p2p Target: $TARGET"

  if [[ "${USE_DIETGPU:-0}" == "1" ]]; then
    cd thirdparty/dietgpu
    if [[ "$TARGET" == cu* ]]; then
      cd dietgpu/float
      CUDA_GPU_ARCH="sm_$(echo "${TORCH_CUDA_ARCH_LIST:-9.0}" | awk '{print $1}' | sed 's/+PTX//; s/\.//')"
      echo "Building dietgpu float for CUDA: $CUDA_GPU_ARCH"
      make clean -f Makefile.cuda && make -j$(nproc) -f Makefile.cuda GPU_ARCH=$CUDA_GPU_ARCH
    else
      rm -rf build/
      python3 setup.py build
      cd dietgpu/float
      echo $TORCH_CUDA_ARCH_LIST
      make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm GPU_ARCH=$TORCH_CUDA_ARCH_LIST
    fi
    cd ../../../..
    cp thirdparty/dietgpu/dietgpu/float/libdietgpu_float.so "${UCCL_LIB_DIR}/"
  fi

  cd p2p
  if [[ "$TARGET" == cu* ]]; then
    make clean && make -j$(nproc)
  elif [[ "$TARGET" == roc[67] ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  elif [[ "$TARGET" == "therock" ]]; then
    make clean -f Makefile.therock && make -j$(nproc) -f Makefile.therock HIP_HOME=$(rocm-sdk path --root) CONDA_LIB_HOME=$VIRTUAL_ENV/lib
  fi
  cd ..

  echo "[container] Copying P2P .so, collective.py and utils.py to ${UCCL_PY_DIR}"
  mkdir -p "${UCCL_PY_DIR}" "${UCCL_LIB_DIR}"
  cp p2p/libuccl_p2p.so "${UCCL_LIB_DIR}/"
  cp p2p/p2p.*.so "${UCCL_PY_DIR}/"
  cp p2p/collective.py "${UCCL_PY_DIR}/"
  cp p2p/utils.py "${UCCL_PY_DIR}/"
  rename_to_abi3 "${UCCL_PY_DIR}"
}

build_ep() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ep Target: $TARGET"

  if [[ "${USE_INTEL_RDMA_NIC:-0}" == "1" ]]; then
    echo "[container] Building EP with Intel RDMA NIC support (USE_INTEL_RDMA_NIC=1)"
  fi
  if [[ "${USE_DMABUF:-0}" == "1" ]]; then
    echo "[container] Building EP with DMA-BUF GPU memory registration (USE_DMABUF=1)"
  fi

  if [[ "$TARGET" == "roc6" ]]; then
    echo "ERROR: EP requires roc7 (ROCm 7) for HIP code transformation; roc6 is not supported." >&2
    exit 1
  elif [[ "$TARGET" == roc[67] || "$TARGET" == cu* || "$TARGET" == "therock" ]]; then
    cd ep
    # This may be needed if you traverse through different git commits
    # make clean && rm -r build || true
    extra_env=()
    if [[ "$TARGET" == "therock" ]]; then
      # On TheRock, ROCm comes from a pip-installed rocm-sdk wheel; expose its
      # root to ep/setup.py via HIP_HOME/ROCM_HOME so hipcc can find headers
      # and libraries. The IBGDA (GPU-driven RDMA) code path in
      # ep/src/internode_ll.cu is already gated by __HIP_PLATFORM_AMD__ guards,
      # so no extra flag is needed to keep the AMD build clean.
      ROCM_ROOT="$(rocm-sdk path --root)"
      extra_env+=(
        "HIP_HOME=${ROCM_ROOT}"
        "ROCM_HOME=${ROCM_ROOT}"
        "ROCM_PATH=${ROCM_ROOT}"
      )
    fi
    env "${extra_env[@]}" \
      USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} \
      USE_DMABUF=${USE_DMABUF:-0} \
      python3 setup.py build
    cd ..
    echo "[container] Copying EP .so to ${UCCL_EP_DIR}"
    mkdir -p "${UCCL_EP_DIR}"
    cp ep/build/**/*.so "${UCCL_EP_DIR}/"
  fi
  rename_to_abi3 "${UCCL_EP_DIR}"
}

build_ukernel() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ukernel Target: $TARGET"

  cd experimental/ukernel
  if [[ "$TARGET" == cu* ]]; then
    make clean -f Makefile && make -j$(nproc) -f Makefile
  elif [[ "$TARGET" == roc[67] ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  fi
  cd ../..

  echo "[container] Copying ukernel .so to ${UCCL_LIB_DIR}"
  mkdir -p "${UCCL_LIB_DIR}"
  cp experimental/ukernel/*ukernel*.so "${UCCL_LIB_DIR}/"
}

clean_all() {
  # Best-effort clean of every per-module Makefile flavour; ignore missing
  # files / missing toolchains.
  for f in Makefile Makefile.rocm Makefile.therock; do
    make -C collective/rdma -f "$f" clean 2>/dev/null || true
    make -C p2p -f "$f" clean 2>/dev/null || true
    make -C experimental/ukernel -f "$f" clean 2>/dev/null || true
  done
  make -C collective/efa clean 2>/dev/null || true
  rm -rf ep/build
  rm -f "${UCCL_LIB_DIR}"/*.so "${UCCL_LIB_DIR}"/*.a
  rm -f "${UCCL_PY_DIR}"/p2p*.so
  rm -f "${UCCL_PY_DIR}"/collective.py "${UCCL_PY_DIR}"/utils.py
  rm -f "${UCCL_EP_DIR}"/*.so
}

########################################################
# Main build logic
########################################################

if [[ "$BUILD_TYPE" == "clean" ]]; then
  clean_all
  exit 0
fi

if [[ "$TARGET" == roc[67] ]]; then
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
else
  echo "build_native: unknown BUILD_TYPE '$BUILD_TYPE'" >&2
  exit 1
fi
