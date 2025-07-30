#!/bin/bash

TARGET=${1:-cuda}
BUILD_TYPE=${2:-all}
PY_VER=${3:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}

if [[ $TARGET != "cuda" && $TARGET != "rocm" ]]; then
  echo "Usage: $0 [cuda|rocm] [all|rdma|p2p|efa] [PY_VER]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)
./build.sh $TARGET $BUILD_TYPE $PY_VER
pip install wheelhouse-$TARGET/uccl-*.whl --force-reinstall
