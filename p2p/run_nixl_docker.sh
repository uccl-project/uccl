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

  # Unfortuantely, the wheel built in the container is not compatible with the host, so nixl cannot detect RDMA devices.
  cd /tmp/nixl
  python3 -m build
  auditwheel repair dist/nixl-*.whl --exclude libibverbs.so.1 --exclude libcudart.so.12 --exclude libamdhip64.so.6 -w /io/p2p/${WHEEL_DIR}
}

_UID=$(id -u)
_GID=$(id -g)

PLATFORM_OPT=""
if [ "$ARCH" == "aarch64" ]; then
  PLATFORM_OPT="--platform=linux/arm64"
fi

docker build $PLATFORM_OPT --build-arg PY_VER="${PY_VER}" --build-arg UID="${_UID}" --build-arg GID="${_GID}" -t "$IMAGE_NAME" -f "$DOCKERFILE" .

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
  -w /io/p2p/benchmarks \
  -it "$IMAGE_NAME" /bin/bash

  # "$IMAGE_NAME" /bin/bash -c '
  #   set -euo pipefail
  #   eval "$FUNCTION_DEF"
  #   if [ "$TARGET" == "rocm" ]; then
  #     build_nixl_rocm "$PY_VER" "$WHEEL_DIR"
  #   fi

# echo "Wheel built successfully (stored in ${WHEEL_DIR}):"
# ls -lh "${WHEEL_DIR}"/*.whl || true

# pip uninstall nixl -y || true
# pip install ${WHEEL_DIR}/*.whl
