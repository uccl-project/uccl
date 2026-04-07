#!/bin/bash

# -----------------------
# build_inner.sh — runs *inside* the build container.
# Invoked by build.sh via docker/podman/apptainer; not intended for direct
# execution on the host.
#
# Environment variables consumed (set by build.sh before container launch):
#
#   Required:
#     TARGET                        Build target: cu12, cu13, roc7, roc6, therock
#     PY_VER                        Python version, e.g. "3.10"
#     ARCH                          Host architecture: x86_64 or aarch64
#     BUILD_TYPE                    What to build: all, ccl_rdma, ccl_efa, p2p, ep, p2p_ep, ukernel
#     IS_EFA                        Non-empty string when EFA is detected on the host
#     WHEEL_DIR                     Output directory for built wheels (relative to /io)
#     HOST_GLIBC_VER                Host glibc version string (e.g. "2.35")
#
#   Optional (with defaults):
#     ROCM_IDX_URL                  ROCm package index URL (used by therock target)
#     UCCL_RETAG_TO_HOST_GLIBC      Retag wheel to host glibc version (default "0")
#     UCCL_LOCAL_VERSION            Local version suffix appended to wheel filename (PEP 440)
#
#   Build feature flags:
#     USE_TCPX                      Enable TCPX transport (default "0")
#     USE_EFA                       Enable EFA transport (default "0")
#     USE_IB                        Enable InfiniBand transport (default "0")
#     USE_TCP                       Enable TCP transport (default "0")
#     USE_DIETGPU                   Enable DietGPU compression (default "0")
#     USE_INTEL_RDMA_NIC            Enable Intel RDMA NIC / irdma driver (default "0")
#     PER_EXPERT_BATCHING           Enable per-expert batching (default "0")
#     MAKE_NORMAL_MODE              Make normal mode flag
#     TORCH_CUDA_ARCH_LIST          CUDA compute capabilities to compile for
# -----------------------

set -euo pipefail

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
    cp thirdparty/dietgpu/dietgpu/float/libdietgpu_float.so uccl/lib
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

  echo "[container] Copying P2P .so, collective.py and utils.py to uccl/"
  mkdir -p uccl
  mkdir -p uccl/lib
  if [[ -z "${USE_TCPX:-}" || "$USE_TCPX" != "1" ]]; then
    cp p2p/libuccl_p2p.so uccl/lib/
    cp p2p/p2p.*.so uccl/
    cp p2p/collective.py uccl/
    cp p2p/utils.py uccl/
  else
    echo "[container] USE_TCPX=1, skipping copying p2p runtime files"
  fi
  rename_to_abi3 uccl
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

  if [[ "$TARGET" == "roc6" ]]; then
    echo "ERROR: EP requires roc7 (ROCm 7) for HIP code transformation; roc6 is not supported." >&2
    exit 1
  elif [[ "$TARGET" == "therock" ]]; then
    echo "Skipping GPU-driven build on therock (no GPU-driven support yet)."
  elif [[ "$TARGET" == roc[67] || "$TARGET" == cu* ]]; then
    cd ep
    # This may be needed if you traverse through different git commits
    # make clean && rm -r build || true
    USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} python3 setup.py build
    cd ..
    echo "[container] Copying GPU-driven .so to uccl/"
    mkdir -p uccl/lib
    cp ep/build/**/*.so uccl/
  fi
  rename_to_abi3 uccl
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

  echo "[container] Copying ukernel .so to uccl/"
  mkdir -p uccl/lib # mkdir anyway
  cp experimental/ukernel/*ukernel*.so uccl/lib
}

########################################################
# Main build logic
########################################################

if [[ "$TARGET" == "therock" ]]; then
  PY_V=$(echo ${PY_VER} | tr -d .)
  export PATH=/opt/python/cp${PY_V}-cp${PY_V}/bin:$PATH

  python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
  pip3 install --no-cache-dir --upgrade pip
  pip3 install --no-cache-dir build auditwheel pybind11 nanobind
  pip3 install --no-cache-dir rocm[libraries,devel] --index-url ${ROCM_IDX_URL}
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
fi

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
" >uccl/_rocm_init.py

  BACKUP_FN=$(mktemp -p . -t setup.py.XXXXXX)
  cp ./setup.py ${BACKUP_FN}
  sed -i "s/\"rocm\": \[\],/\"rocm\": \[\"rocm\[libraries\]==$(rocm-sdk version)\"\, \"torch\", \"numpy\"],/;" setup.py

  export PIP_EXTRA_INDEX_URL=${ROCM_IDX_URL}
fi

ls -lh uccl/
ls -lh uccl/lib/
python3 -m build

if [[ "$TARGET" == "therock" ]]; then
  mv ${BACKUP_FN} setup.py
fi

# Always use the *container* glibc for auditwheel repair (symbol validation),
# then retag the wheel to the desired host platform afterwards if requested.
CONTAINER_GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])")
AUDIT_PLAT="manylinux_${CONTAINER_GLIBC_VER//./_}_$(uname -m)"

if [[ "${UCCL_RETAG_TO_HOST_GLIBC}" == "1" ]]; then
  UCCL_WHEEL_PLAT="manylinux_${HOST_GLIBC_VER//./_}_$(uname -m)"
  if [[ "${UCCL_WHEEL_PLAT}" != "${AUDIT_PLAT}" ]]; then
    echo "WARNING: UCCL_RETAG_TO_HOST_GLIBC is set." >&2
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
    echo "  Tip: set UCCL_RETAG_TO_HOST_GLIBC=1 to retag to host glibc ${HOST_GLIBC_VER}."
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

# Collapse dual platform tags to the single requested tag.
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

# Add local version identifier to wheel filename (PEP 440).
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
