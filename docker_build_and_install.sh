#!/bin/bash

TARGET=${1:-cuda}
PY_VER=${2:-3.13}
PY_TAG="cp${PY_VER/./}"

if [[ $TARGET != "cuda" && $TARGET != "rocm" && $TARGET != "gh" && $TARGET != "efa" && $TARGET != "all" ]]; then
  echo "Usage: $0 [cuda|rocm|gh|efa|all] [PY_VER]" >&2
  exit 1
fi

./docker_build.sh $TARGET $PY_VER
pip install wheelhouse-$TARGET/uccl-0.0.1.post2-${PY_TAG}-${PY_TAG}-manylinux_2_35_aarch64.whl --force-reinstall