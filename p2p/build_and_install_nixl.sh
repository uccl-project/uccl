#!/bin/bash

TARGET=${1:-rocm}
PY_VER=${2:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}

ARCH="$(uname -m)"
DOCKERFILE="docker/dockerfile.${TARGET}"
IMAGE_NAME="nixl-builder-${TARGET}"
WHEEL_DIR="wheelhouse-${TARGET}"

rm -r "${WHEEL_DIR}" >/dev/null 2>&1 || true
mkdir -p "${WHEEL_DIR}"

build_nixl_rocm () {
  local PY_VER="$1"
  local WHEEL_DIR="$2"
  
  cd /io/thirdparty/ucx &&        \
    ./autogen.sh && ./configure     \
    --prefix=/usr/local/ucx         \
    --enable-shared                 \
    --disable-static                \
    --disable-doxygen-doc           \
    --enable-optimizations          \
    --enable-cma                    \
    --enable-devel-headers          \
    --with-cuda=/usr/local/cuda     \
    --with-gdrcopy=/usr/local       \
    --with-verbs                    \
    --with-efa                      \
    --with-dm                       \
    --enable-mt &&                  \
    make -j &&                      \
    sudo make -j install-strip &&   \
    sudo ldconfig

  cd /io/thirdparty/nixl && \
    pip install meson -U && \
    mkdir build && \
    meson setup build/ --prefix=/usr/local/nixl -Ddisable_gds_backend=true -Ducx_path=/usr/local/ucx && \
    cd build/ && \
    ninja && \
    yes | ninja install

  cd /io/thirdparty/nixl
  python3 -m build
  auditwheel repair dist/nixl-*.whl --exclude libibverbs.so.1 --exclude libcudart.so.12 --exclude libamdhip64.so.6 -w /io/p2p/${WHEEL_DIR}
}

if [ "$ARCH" == "aarch64" ]; then
  docker build --platform=linux/arm64 --build-arg PY_VER="${PY_VER}" --build-arg USER="${USER}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  docker build --build-arg PY_VER="${PY_VER}" --build-arg USER="${USER}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

docker run --rm --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v "$(pwd)/..":/io \
  -e TARGET="${TARGET}" \
  -e PY_VER="${PY_VER}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e FUNCTION_DEF="$(declare -f build_nixl_cuda build_nixl_rocm)" \
  -w /io \
  # -it "$IMAGE_NAME" /bin/bash
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail

    eval "$FUNCTION_DEF"
    if [ "$TARGET" == "rocm" ]; then
      build_nixl_rocm "$PY_VER" "$WHEEL_DIR"
    fi
  '

echo "Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/*.whl || true

# pip install ${WHEEL_DIR}/*.whl --force-reinstall
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
