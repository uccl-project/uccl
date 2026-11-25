#!/usr/bin/env bash
# Sourced by each suite. Everything points at this checkout or local CI storage.
export PATH="${CI_ENV_DIR}/bin:${CUDA_HOME:-/usr}/bin:$PATH"
export PYTHONPATH="${CI_REPO}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr}/lib64:${CI_REPO}/p2p:${CI_REPO}/collective/rdma${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${CI_DEPS}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export NCCL_ROOT="${CI_DEPS}/nccl-2.30.4/nvidia/nccl"
export NCCL_INCLUDE_DIR="${NCCL_ROOT}/include"
export EP_NCCL_ROOT_DIR="${NCCL_ROOT}"
export EP_JIT_CACHE_DIR="${CI_REPO}/.ci-jit"
export MAX_JOBS="${CI_JOBS}"
export GDRCOPY_INCLUDE_DIR="$CI_DEPS/gdrcopy-2.5.1/include"
export GDRCOPY_INCLUDEDIR="$GDRCOPY_INCLUDE_DIR"
export GDRCOPY_LIBDIR="$CI_DEPS/gdrcopy-2.5.1/lib"
export CPATH="$GDRCOPY_INCLUDE_DIR${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$GDRCOPY_LIBDIR:$LD_LIBRARY_PATH"

# Native C++ fixtures dlopen NCCL without importing torch first.
if [[ -n ${CUDA_HOME:-} ]]; then
  export LD_LIBRARY_PATH="$CI_ENV_DIR/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
fi
