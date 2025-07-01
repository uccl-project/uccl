#!/bin/bash
set -e

# Configuration
IMAGE_NAME="uccl-builder"
CUDA_PATH="/usr/local/cuda"

# 1. Build the Docker image
echo "[1/3] Building Docker image..."
docker build -t $IMAGE_NAME .

# 2. Run the container and build inside it
echo "[2/3] Running build inside container..."
docker run --rm \
  -v $(pwd):/io \
  -v ${CUDA_PATH}:${CUDA_PATH} \
  -e CUDA_HOME=${CUDA_PATH} \
  $IMAGE_NAME /bin/bash -c "
    set -e
    echo '[inside container] Building libnccl-net-uccl.so...'
    cd rdma && make -j$(nproc) && cd ..

    echo '[inside container] Packaging uccl...'
    mkdir -p uccl/lib
    cp rdma/libnccl-net-uccl.so uccl/lib/
    python3 -m build

    echo '[inside container] Running auditwheel...'
    auditwheel repair dist/uccl-*.whl -w /io/wheelhouse
  "

# 3. Done
echo "[3/3] Wheel built successfully:"
ls -lh wheelhouse/*.whl
