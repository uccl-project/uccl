#!/bin/bash
set -e

# -----------------------
# Build uccl wheels for CUDA (NVIDIA) and ROCm (AMD) back-ends
# The host machine does *not* need CUDA or ROCm – everything lives inside
# a purpose-built Docker image derived from Ubuntu 22.04.
#
# Usage:
#   ./docker_build.sh [cuda|rocm|efa|all] [-it]
#
# The wheels are written to ./wheelhouse-*/
# -----------------------

TARGET=${1:-cuda}

if [[ $TARGET != "cuda" && $TARGET != "rocm" && $TARGET != "efa" && $TARGET != "all" ]]; then
  echo "Usage: $0 [cuda|rocm|efa|all]" >&2
  exit 1
fi

rm -r uccl.egg-info || true
rm -r dist || true
rm -r uccl/lib || true
rm -r build || true
WHEEL_DIR="wheelhouse-${TARGET}"
rm -r "${WHEEL_DIR}" || true
mkdir -p "${WHEEL_DIR}"

# If TARGET=all, orchestrate builds for each backend and package **all** shared libraries
if [[ $TARGET == "all" ]]; then
  # Temporary directory to accumulate .so files from each backend build
  TEMP_LIB_DIR="uccl/lib_all"
  rm -rf "${TEMP_LIB_DIR}" || true
  mkdir -p "${TEMP_LIB_DIR}"

  echo "### Building CUDA backend and collecting its shared library ###"
  "$0" cuda
  cp uccl/lib/*.so "${TEMP_LIB_DIR}/" || true

  echo "### Building ROCm backend and collecting its shared library ###"
  "$0" rocm
  cp uccl/lib/*.so "${TEMP_LIB_DIR}/" || true

  echo "### Building EFA backend and collecting its shared library ###"
  "$0" efa
  cp uccl/lib/*.so "${TEMP_LIB_DIR}/" || true

  # Prepare combined library directory
  rm -rf uccl/lib || true
  mkdir -p uccl/lib
  cp "${TEMP_LIB_DIR}"/*.so uccl/lib/

  echo "### Packaging $TARGET wheel (contains all libs) ###"
  docker run --rm --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $HOME:$HOME \
    -v "$(pwd)":/io \
    -e TARGET="${TARGET}" \
    -w /io \
    uccl-builder-cuda /bin/bash -c '
      set -euo pipefail
      ls -lh uccl/lib
      python3 -m build
      auditwheel repair dist/uccl-*.whl --exclude libibverbs.so.1 -w /io/wheelhouse-${TARGET}
      auditwheel show /io/wheelhouse-${TARGET}/*.whl
    '

  echo "Done. $TARGET wheel is in wheelhouse-${TARGET}/."
  exit 0
fi

DOCKERFILE="docker/Dockerfile.${TARGET}"
IMAGE_NAME="uccl-builder-${TARGET}"

# Build the builder image (contains toolchain + CUDA/ROCm)
echo "[1/3] Building Docker image ${IMAGE_NAME} using ${DOCKERFILE}..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

echo "[2/3] Running build inside container..."
if [[ $2 == "-it" ]]; then
  docker run -it --rm --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $HOME:$HOME \
    -v "$(pwd)":/io \
    -e TARGET="${TARGET}" \
    -e WHEEL_DIR="${WHEEL_DIR}" \
    -w /io \
    "$IMAGE_NAME" /bin/bash
else
  docker run --rm --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $HOME:$HOME \
    -v "$(pwd)":/io \
    -e TARGET="${TARGET}" \
    -e WHEEL_DIR="${WHEEL_DIR}" \
    -w /io \
    "$IMAGE_NAME" /bin/bash -c '
      set -euo pipefail
      echo "[container] Backend: $TARGET"
      echo "[container] Compiling native library…"
      
      if [[ "$TARGET" == cuda ]]; then
          cd rdma && make clean && make -j$(nproc) && cd ..
          TARGET_SO=rdma/libnccl-net-uccl.so
      elif [[ "$TARGET" == rocm ]]; then
          # Unlike CUDA, ROCM does not include nccl.h. So we need to build rccl to get nccl.h.
          if [[ ! -f "thirdparty/rccl/build/release/include/nccl.h" ]]; then
            cd thirdparty/rccl
            # Just to get nccl.h, not the whole library
            CXX=/opt/rocm/bin/hipcc cmake -B build/release -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF >/dev/null 2>&1 || true
            cd ../..
          fi
          cd rdma && make clean -f Makefile_hip && make -j$(nproc) -f Makefile_hip && cd ..
          TARGET_SO=rdma/librccl-net-uccl.so
      elif [[ "$TARGET" == efa ]]; then
          cd efa && make clean && make -j$(nproc) && cd ..
          TARGET_SO=efa/libnccl-net-efa.so
      fi

      echo "[container] Packaging uccl..."
      mkdir -p uccl/lib
      cp ${TARGET_SO} uccl/lib/
      python3 -m build

      echo "[container] Running auditwheel..."
      auditwheel repair dist/*.whl --exclude libibverbs.so.1 -w /io/${WHEEL_DIR}
      auditwheel show /io/${WHEEL_DIR}/*.whl
    '
  fi

# 3. Done
echo "[3/3] Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/uccl-*.whl || true
