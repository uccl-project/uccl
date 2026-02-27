#!/bin/bash

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

  if [[ "$TARGET" == cuda* ]]; then
    cd collective/rdma && make clean && make -j$(nproc) USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} && cd ../../
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
    cp p2p/p2p.*.so uccl/
    cp p2p/collective.py uccl/
    cp p2p/transfer.py uccl/
    cp p2p/utils.py uccl/
  else
    echo "[container] USE_TCPX=1, skipping copying p2p runtime files"
  fi
  if [[ "$TARGET" == rocm* ]]; then
    cd thirdparty/dietgpu
    rm -rf build/
    python3 setup.py build
    cd ../..
    cp thirdparty/dietgpu/build/**/*.so uccl/
  fi
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

  if [[ "$TARGET" == "therock" ]]; then
    echo "Skipping GPU-driven build on therock (no GPU-driven support yet)."
  elif [[ "$TARGET" == rocm* || "$TARGET" == cuda* ]]; then
    cd ep
    # This may be needed if you traverse through different git commits
    # make clean && rm -r build || true
    USE_INTEL_RDMA_NIC=${USE_INTEL_RDMA_NIC:-0} python3 setup.py build
    cd ..
    echo "[container] Copying GPU-driven .so to uccl/"
    mkdir -p uccl/lib
    cp ep/build/**/*.so uccl/
  fi
}

build_ukernel() {
  local TARGET="$1"
  local ARCH="$2"
  local IS_EFA="$3"

  set -euo pipefail
  echo "[container] build_ukernel Target: $TARGET"

  cd experimental/ukernel
  if [[ "$TARGET" == cuda* ]]; then
    make clean -f Makefile && make -j$(nproc) -f Makefile
  elif [[ "$TARGET" == rocm* ]]; then
    make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
  fi
  cd ../..

  echo "[container] Copying ukernel .so to uccl/"
  mkdir -p uccl/lib # mkdir anyway
  cp experimental/ukernel/*ukernel*.so uccl/lib
}
