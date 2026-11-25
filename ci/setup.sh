#!/usr/bin/env bash
set -euo pipefail
cd "${CI_REPO:?}"
source ci/env.sh
component=${1:?component required}
if [[ $component == environment ]]; then
  if [[ ! -f $CI_ENV_DIR/.complete ]]; then
    python3 -m venv "$CI_ENV_DIR"
    index_args=()
    case $CI_LOCK in
      *cu128.lock) index_args=(--extra-index-url https://download.pytorch.org/whl/cu128);;
      *cu130.lock) index_args=(--extra-index-url https://download.pytorch.org/whl/cu130);;
    esac
    "$CI_ENV_DIR/bin/python" -m pip install --no-cache-dir --require-hashes "${index_args[@]}" -r "$CI_LOCK"
    "$CI_ENV_DIR/bin/python" -m pip check
    touch "$CI_ENV_DIR/.complete"
  fi
  "$CI_ENV_DIR/bin/python" -m pip freeze --all > "$CI_ARTIFACTS/python-packages.txt"
  exit 0
fi
case $component in
  cpu-cuda-headers)
    sdk="$CI_DEPS/cpu-cuda-12.8"
    if [[ ! -f $sdk/.complete ]]; then
      python3 -m pip install --require-hashes --no-deps --target "$sdk" -r ci/requirements/cpu-cuda.lock
      ln -sfn lib "$sdk/nvidia/cuda_runtime/lib64"
      ln -sfn libcudart.so.12 "$sdk/nvidia/cuda_runtime/lib/libcudart.so"
      touch "$sdk/.complete"
    fi
    ;;
  p2p)
    make -C p2p clean
    make -C p2p -j"$CI_JOBS" PYTHON=python3 ENABLE_EFA=0 ENABLE_CXI=1 \
      INSTALL_DIR="$CI_REPO/uccl" PREFIX="$CI_REPO/.ci-install" LIBDIR="$CI_REPO/.ci-install/lib" install
    python3 -c 'import torch; from uccl import p2p; print(torch.__version__, p2p.__file__)'
    ;;
  ep)
    # ep/tests also writes src/common.o, without the extension's PIC/ABI flags.
    make -C ep clean
    make -C ep -j"$CI_JOBS" SM="$SM" NUM_MAX_NVL_PEERS=1 PYTHON=python3 INSTALL_DIR="$CI_REPO/uccl" install
    python3 -c 'import torch; import uccl.ep; print(uccl.ep.__file__)'
    ;;
  nccl-headers)
    if [[ ! -f $NCCL_INCLUDE_DIR/nccl_device.h ]]; then
      python3 -m pip install --require-hashes --no-deps --target "$CI_DEPS/nccl-2.30.4" -r ci/requirements/nccl.lock
    fi
    ln -sfn libnccl.so.2 "$NCCL_ROOT/lib/libnccl.so"
    ;;
  collective)
    make -C collective/rdma -j"$CI_JOBS" librdma.a
    ;;
  collective-plugin)
    bash ci/setup.sh nccl-headers
    make -C collective/rdma -j"$CI_JOBS" libnccl-net-uccl.so \
      NCCL_INC="-I$NCCL_INCLUDE_DIR -I$CI_REPO/thirdparty/nccl-sg/src/include -I$CUDA_HOME/include"
    ;;
  compression)
    make -C p2p clean
    make -C thirdparty/dietgpu/dietgpu/float -f Makefile.cuda -j"$CI_JOBS" GPU_ARCH="sm_$SM"
    make -C p2p -j"$CI_JOBS" PYTHON=python3 ENABLE_EFA=0 ENABLE_CXI=1 USE_DIETGPU=1 \
      INSTALL_DIR="$CI_REPO/uccl" PREFIX="$CI_REPO/.ci-install" LIBDIR="$CI_REPO/.ci-install/lib" install
    ;;
  ukernel)
    bash ci/setup.sh gdrcopy-lib
    make -C experimental/ukernel -j"$CI_JOBS" SM="$SM" libukernel.a libukernel.so
    ;;
  ukernel-python)
    bash ci/setup.sh gdrcopy-lib
    bash ci/setup.sh collective
    (cd experimental/ukernel/py; python3 setup.py build_ext --inplace)
    ;;
  gin-local)
    bash ci/setup.sh nccl-headers
    make -C experimental/uccl_gin/tests/local -j"$CI_JOBS" SM="$SM" \
      EXTRA_DEVFLAGS=-DNCCL_GIN_GDAKI_ENABLE=0 all coop-tests adapter-tests \
      compile-fail adapter-compile-fail ht-tests model-tests perf-controls
    ;;
  lite-ep)
    bash ci/setup.sh nccl-headers
    make -C experimental/lite/lite-ep -j"$CI_JOBS" SM="$SM" PYTHON=python3 \
      NCCL_ROOT="$NCCL_ROOT" EP_JIT_CACHE_DIR="$EP_JIT_CACHE_DIR" TORCH_NVSHMEM_STUB=
    ;;
  lite-collective)
    make -C experimental/lite/lite-collective -j"$CI_JOBS" CUDA_ARCH="$SM"
    make -C experimental/lite/lite-collective/nccl -j"$CI_JOBS" CUDA_ARCH="$SM"
    ;;
  nccl-tests)
    bash ci/setup.sh nccl-headers
    bash ci/fetch.sh nccl-tests "$CI_REPO/.ci-nccl-tests"
    make -C "$CI_REPO/.ci-nccl-tests" -j"$CI_JOBS" MPI=0 \
      NCCL_HOME="$NCCL_ROOT" CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_$SM,code=sm_$SM"
    ;;
  nixl)
    bash ci/setup_nixl.sh
    ;;
  gdrcopy-lib)
    if [[ ! -f $GDRCOPY_LIBDIR/libgdrapi.so ]]; then
      bash ci/fetch.sh gdrcopy "$CI_REPO/.ci-gdrcopy"
      make -C "$CI_REPO/.ci-gdrcopy" -j"$CI_JOBS" lib_install prefix="$CI_DEPS/gdrcopy-2.5.1"
    fi
    ;;
  *) echo "Unknown setup component: $component" >&2; exit 2;;
esac
