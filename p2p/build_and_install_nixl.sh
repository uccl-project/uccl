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

  # This is inconsistent with the host one. We need to fix this.
  sudo mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.bak
  
  cd /io/thirdparty/ucx &&          \
    ./autogen.sh && ./configure     \
    --prefix=/usr/local/ucx         \
    --enable-shared                 \
    --disable-static                \
    --disable-doxygen-doc           \
    --enable-optimizations          \
    --enable-cma                    \
    --enable-devel-headers          \
    --with-rocm=/opt/rocm           \
    --with-verbs                    \
    --with-efa                      \
    --with-dm                       \
    --enable-mt &&                  \
    make -j &&                      \
    sudo make -j install-strip &&   \
    sudo ldconfig

  cd /io/thirdparty/nixl && \
    pip3 install meson -U && \
    rm -r build && \
    mkdir build && \
    meson setup build/ --prefix=/usr/local/nixl -Ddisable_gds_backend=true -Ducx_path=/usr/local/ucx && \
    cd build/ && \
    ninja && \
    yes | ninja install

  export LD_LIBRARY_PATH="/usr/local/nixl/lib/`uname -m`-linux-gnu/plugins:/usr/local/ucx/lib:$LD_LIBRARY_PATH"
  cd /io/thirdparty/nixl
  python3 -m build
  auditwheel repair dist/nixl-*.whl --exclude libibverbs.so.1 --exclude libcudart.so.12 --exclude libamdhip64.so.6 -w /io/p2p/${WHEEL_DIR}
}

_UID=$(id -u)
_GID=$(id -g)

if [ "$ARCH" == "aarch64" ]; then
  docker build --platform=linux/arm64 --build-arg PY_VER="${PY_VER}" --build-arg UID="${_UID}" --build-arg GID="${_GID}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
else
  docker build --build-arg PY_VER="${PY_VER}" --build-arg UID="${_UID}" --build-arg GID="${_GID}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

docker run --rm \
  -v $HOME:$HOME \
  -v "$(pwd)/..":/io \
  -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/local/lib/libbnxt_re-rdmav34.so:ro \
  -e TARGET="${TARGET}" \
  -e PY_VER="${PY_VER}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e FUNCTION_DEF="$(declare -f build_nixl_rocm)" \
  -w /io \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail

    eval "$FUNCTION_DEF"
    if [ "$TARGET" == "rocm" ]; then
      build_nixl_rocm "$PY_VER" "$WHEEL_DIR"
    fi
  '

  # -it "$IMAGE_NAME" /bin/bash

echo "Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/*.whl || true

pip uninstall nixl -y || true
pip install ${WHEEL_DIR}/*.whl
