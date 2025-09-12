#!/bin/bash

TARGET=${1:-cuda}
BUILD_TYPE=${2:-all}
PY_VER=${3:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
GFX_VER=${4:-gfx94X-dcgpu}

if [[ $TARGET != "cuda" && $TARGET != "rocm" && $TARGET != "therock" ]]; then
  echo "Usage: $0 [cuda|rocm|therock] [all|rdma|p2p|efa|ep] [py_version] [gfx_version]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)
./build.sh $TARGET $BUILD_TYPE $PY_VER $GFX_VER
pip install -r requirements.txt
pip uninstall uccl -y || true
if [[ $TARGET != "therock" ]]; then
  pip install wheelhouse-$TARGET/uccl-*.whl --no-deps
else
  # TheRock packages ROCm dependences through python packaging
  # That (currently) requires --extra-index-url
  pip install --extra-index-url https://rocm.nightlies.amd.com/v2/${GFX_VER} $(ls wheelhouse-$TARGET/uccl-*.whl)[rocm]
fi
