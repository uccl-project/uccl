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

  cd /tmp
  git clone https://github.com/google/googletest.git
  cd googletest
  cmake -S . -B build \
    -DBUILD_GMOCK=ON -DBUILD_GTEST=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
  cmake --build build -j
  sudo cmake --install build

  cd /io/thirdparty/nixl && \
    pip3 install meson -U && \
    rm -rf build || true && \
    mkdir build && \
    meson setup build/ --prefix=/usr/local/nixl -Ddisable_gds_backend=true -Ducx_path=/usr/local/ucx && \
    cd build/ && \
    ninja && \
    yes | ninja install
  cd ..
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
  pip3 install pyzmq
  pip3 install .

  cd /io/p2p/benchmarks
  export LD_LIBRARY_PATH="/usr/local/nixl/lib/`uname -m`-linux-gnu/plugins:/usr/local/ucx/lib:$LD_LIBRARY_PATH"

  UCX_MAX_RMA_LANES=4 UCX_NET_DEVICES=rdma3:1 UCX_TLS=rocm,rc python benchmark_nixl.py --role server
  UCX_MAX_RMA_LANES=4 UCX_NET_DEVICES=rdma3:1 UCX_TLS=rocm,rc python benchmark_nixl.py --role client --remote-ip <Server IP>
  
  # cd /io/thirdparty/nixl
  # python3 -m build
  # auditwheel repair dist/nixl-*.whl --exclude libibverbs.so.1 --exclude libcudart.so.12 --exclude libamdhip64.so.6 -w /io/p2p/${WHEEL_DIR}
}

_UID=$(id -u)
_GID=$(id -g)

PLATFORM_OPT=""
if [ "$ARCH" == "aarch64" ]; then
  PLATFORM_OPT="--platform=linux/arm64"
fi

# Check if image already exists
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Image $IMAGE_NAME already exists, skipping build"
else
  docker build $PLATFORM_OPT --build-arg PY_VER="${PY_VER}" --build-arg UID="${_UID}" --build-arg GID="${_GID}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

docker run --rm \
  --device /dev/dri \
  --device /dev/kfd \
  --device /dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --group-add render \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --cap-add=IPC_LOCK \
  --shm-size 64G \
  -v /opt/rocm:/opt/rocm:ro \
  -v $HOME:$HOME \
  -v "$(pwd)/..":/io \
  -v /usr/local/lib/libbnxt_re-rdmav34.so:/usr/local/lib/libbnxt_re-rdmav34.so:ro \
  -e TARGET="${TARGET}" \
  -e PY_VER="${PY_VER}" \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e FUNCTION_DEF="$(declare -f build_nixl_rocm)" \
  -w /io \
  -it "$IMAGE_NAME" /bin/bash
  # "$IMAGE_NAME" /bin/bash -c '
  #   set -euo pipefail

  #   eval "$FUNCTION_DEF"
  #   if [ "$TARGET" == "rocm" ]; then
  #     build_nixl_rocm "$PY_VER" "$WHEEL_DIR"
  #   fi
  # '


echo "Wheel built successfully (stored in ${WHEEL_DIR}):"
ls -lh "${WHEEL_DIR}"/*.whl || true

pip uninstall nixl -y || true
pip install ${WHEEL_DIR}/*.whl
