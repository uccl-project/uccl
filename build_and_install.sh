#!/bin/bash

TARGET=${1:-cuda}
BUILD_TYPE=${2:-all}
PY_VER=${3:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
# The default for ROCM_IDX_URL depends on the gfx architecture of your GPU and the index URLs may change.
ROCM_IDX_URL=${4:-https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu}
# The default for THEROCK_BASE_IMAGE is current, but may change. Make sure to track TheRock's dockerfile.
THEROCK_BASE_IMAGE=${5:-quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017}

if [[ $TARGET != cuda* && $TARGET != rocm* && $TARGET != "therock" ]]; then
  echo "Usage: $0 [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep|eccl] [py_version] [rocm_index_url] [therock_base_image]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)
./build.sh $TARGET $BUILD_TYPE $PY_VER $ROCM_IDX_URL $THEROCK_BASE_IMAGE
pip install -r requirements.txt
pip uninstall uccl -y || true
if [[ $TARGET != "therock" ]]; then
  pip install wheelhouse-$TARGET/uccl-*.whl --no-deps
else
  # TheRock packages ROCm dependences through python packaging
  # That (currently) requires --extra-index-url
  pip install --extra-index-url ${ROCM_IDX_URL} $(ls wheelhouse-$TARGET/uccl-*.whl)[rocm]
fi

UCCL_INSTALL_PATH=$(pip show uccl 2>/dev/null | grep "^Location:" | cut -d' ' -f2 || echo "")
if [[ -n "$UCCL_INSTALL_PATH" && -d "$UCCL_INSTALL_PATH" ]]; then
  UCCL_PACKAGE_PATH="$UCCL_INSTALL_PATH/uccl"
  if [[ -d "$UCCL_PACKAGE_PATH" ]]; then
    echo "UCCL installed at: $UCCL_PACKAGE_PATH"
    echo "Set LIBRARY_PATH: export LIBRARY_PATH=\"$UCCL_PACKAGE_PATH/lib:\$LIBRARY_PATH\""
  else
    echo "UCCL package directory not found at: $UCCL_PACKAGE_PATH"
  fi
else
  echo "Warning: Could not detect UCCL installation path"
fi
