#!/bin/bash
set -e
# Build NCCL-EP with UCCL-GIN adapter.
# Set NCCL_EP_USE_UCCL_GIN=1 to enable UCCL path (default: NCCL GIN only).

NCCL_DIR=${NCCL_DIR:-/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/nvidia/nccl}
MPI_DIR=${MPI_DIR:-/opt/amazon/openmpi5}
CUDA=${CUDA:-/usr/local/cuda-13.0}
CUPTI=${CUPTI:-/usr/local/cuda-13.0/extras/CUPTI}
UCCL_GIN=${UCCL_GIN:-$HOME/efs/yzhou/playground/daniel/uccl-danyang/uccl-danyang/experimental/uccl_gin}
USE_UCCL=${NCCL_EP_USE_UCCL_GIN:-0}

BUILD_DIR=build_uccl
mkdir -p $BUILD_DIR

DEVFLAGS="-DNCCL_CHECK_CUDACC=1 -D_NCCL_EP_LSA_TEAM_SIZE_MIN=4 -D_NCCL_EP_LSA_TEAM_SIZE_MAX=8 -D_NCCL_EP_NUM_LSA_TEAMS_2=1"
if [ "$USE_UCCL" = "1" ]; then
  DEVFLAGS="$DEVFLAGS -DNCCL_EP_USE_UCCL_GIN=1"
fi

NVCC_BASE="$CUDA/bin/nvcc -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -Xcompiler -fPIC"
INCS="-I. -I./include -I./device -I$NCCL_DIR/include -I$CUDA/include -I$MPI_DIR/include -I$CUPTI/include"
if [ "$USE_UCCL" = "1" ]; then
  INCS="$INCS -I$UCCL_GIN -I$UCCL_GIN/transport"
  UCCL_OBJS="$BUILD_DIR/uccl_proxy.o $BUILD_DIR/uccl_rdma.o $BUILD_DIR/uccl_common.o $BUILD_DIR/uccl_fifo.o $BUILD_DIR/uccl_adapter.o $BUILD_DIR/adapter_context.o"
else
  UCCL_OBJS=""
fi

echo "=== Building NCCL-EP (UCCL=$USE_UCCL) ==="

# UCCL-GIN transport objects (from experimental/uccl_gin)
if [ "$USE_UCCL" = "1" ]; then
  echo "--- UCCL transport ---"
  $NVCC_BASE $INCS $DEVFLAGS -c $UCCL_GIN/transport/proxy.cpp -o $BUILD_DIR/uccl_proxy.o
  $NVCC_BASE $INCS $DEVFLAGS -c $UCCL_GIN/transport/rdma.cpp -o $BUILD_DIR/uccl_rdma.o
  $NVCC_BASE $INCS $DEVFLAGS -c $UCCL_GIN/transport/common.cpp -o $BUILD_DIR/uccl_common.o
  $NVCC_BASE $INCS $DEVFLAGS -c $UCCL_GIN/transport/fifo.cpp -o $BUILD_DIR/uccl_fifo.o
  $NVCC_BASE $INCS $DEVFLAGS -c $UCCL_GIN/transport/adaptive_sleeper.cc -o $BUILD_DIR/uccl_adapter.o

  # Adapter host-side context
  NVCC_HOST="$CUDA/bin/nvcc -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -Xcompiler -fPIC"
  $NVCC_HOST $INCS -DNCCL_HOSTLIB_ONLY -c adapter/uccl_gin_context.cpp -o $BUILD_DIR/adapter_context.o
fi

# NCCL-EP objects
echo "--- NCCL-EP ---"
$NVCC_BASE $INCS $DEVFLAGS -c device/hybridep_adapter.cu -o $BUILD_DIR/hyb.o
$NVCC_BASE $INCS $DEVFLAGS -c device/low_latency.cu -o $BUILD_DIR/ll.o
$NVCC_BASE $INCS -DNCCL_HOSTLIB_ONLY -D_NCCL_EP_LSA_TEAM_SIZE_MIN=4 -D_NCCL_EP_LSA_TEAM_SIZE_MAX=8 -D_NCCL_EP_NUM_LSA_TEAMS_2=1 -c nccl_ep.cc -o $BUILD_DIR/nccl_ep.o
$NVCC_BASE $INCS $DEVFLAGS -c ep_test.cu -o $BUILD_DIR/test.o

# Link
echo "=== Linking ep_test ==="
$CUDA/bin/nvcc -std=c++17 -arch=sm_90 \
  $BUILD_DIR/test.o $BUILD_DIR/nccl_ep.o $BUILD_DIR/hyb.o $BUILD_DIR/ll.o $UCCL_OBJS \
  -L$NCCL_DIR/lib -lnccl -L$CUDA/lib64 -lcudart -lcuda -L$MPI_DIR/lib -lmpi \
  -L/opt/amazon/efa/lib -lefa -libverbs \
  -Xlinker -rpath -Xlinker $NCCL_DIR/lib -Xlinker -rpath -Xlinker $CUDA/lib64 -Xlinker -rpath -Xlinker $MPI_DIR/lib \
  -o $BUILD_DIR/ep_test

echo "=== Done ==="
ls -la $BUILD_DIR/ep_test
