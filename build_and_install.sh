#!/bin/bash

TARGET=${1:-cuda}
PY_VER=${2:-$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}

if [[ $TARGET != "cuda" && $TARGET != "rocm" ]]; then
  echo "Usage: $0 [cuda|rocm] [PY_VER]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)
./build.sh $TARGET $PY_VER
pip install wheelhouse-$TARGET/uccl-*.whl --force-reinstall
