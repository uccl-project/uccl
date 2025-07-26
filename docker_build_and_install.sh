#!/bin/bash

TARGET=${1:-cuda}
PY_VER=${2:-3.13}

if [[ $TARGET != "cuda" && $TARGET != "rocm" ]]; then
  echo "Usage: $0 [cuda|rocm] [PY_VER]" >&2
  exit 1
fi

ARCH_SUFFIX=$(uname -m)
./docker_build.sh $TARGET $PY_VER
pip install wheelhouse-$TARGET/uccl-*.whl --force-reinstall
