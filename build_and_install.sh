#!/bin/bash

TARGET=${1:-cuda}
BUILD_TYPE=${2:-all}
PY_VER=${3:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
# The default for ROCM_IDX_URL depends on the gfx architecture of your GPU and the index URLs may change.
ROCM_IDX_URL=${4:-https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu}
# The default for THEROCK_BASE_IMAGE is current, but may change. Make sure to track TheRock's dockerfile.
THEROCK_BASE_IMAGE=${5:-quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017}

echo "TARGET : $TARGET"

if [[ $TARGET != cuda* && $TARGET != rocm* && $TARGET != "therock" ]]; then
  echo "Usage: $0 [cuda|rocm|therock] [all|rdma|p2p|efa|ep] [py_version] [rocm_index_url] [therock_base_image]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)

echo "ARCH_SUFFIX : $ARCH_SUFFIX"

is_docker_container() {
    [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null
}

# Check if docker command is available and working
has_docker_command() {
    command -v docker &> /dev/null && docker info &> /dev/null
}

if is_docker_container || !has_docker_command; then
  echo "Running inside the docker : ./build_insider_docker.sh $TARGET $BUILD_TYPE $PY_VER $ROCM_IDX_URL $THEROCK_BASE_IMAGE"
  ./build_insider_docker.sh $TARGET $BUILD_TYPE $PY_VER $ROCM_IDX_URL $THEROCK_BASE_IMAGE
else
  echo "Running with the docker"
  ./build.sh $TARGET $BUILD_TYPE $PY_VER $ROCM_IDX_URL $THEROCK_BASE_IMAGE
fi

pip install -r requirements.txt
pip uninstall uccl -y || true
if [[ $TARGET != "therock" ]]; then
  pip install wheelhouse-$TARGET/uccl-*.whl --no-deps
else
  # TheRock packages ROCm dependences through python packaging
  # That (currently) requires --extra-index-url
  pip install --extra-index-url ${ROCM_IDX_URL} $(ls wheelhouse-$TARGET/uccl-*.whl)[rocm]
fi
