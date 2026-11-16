#!/usr/bin/env bash
# Build the MSCCL++ topology-aware AllReduce benchmark for this host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"
MSCCLPP="${MSCCLPP:-$SCRIPT_DIR/../.deps/mscclpp/install}"
ARCH="${ARCH:-sm_120}"
OUT="${OUT:-$(dirname "$0")/mscclpp_allreduce}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc not found under CUDA_HOME=$CUDA_HOME" >&2
  exit 1
fi

INC=""
LIBDIR=""
for d in "$MSCCLPP/include" "$MSCCLPP/build/include"; do
  [[ -f "$d/mscclpp/core.hpp" ]] && INC="$d" && break
done
for d in "$MSCCLPP/lib" "$MSCCLPP/lib64" "$MSCCLPP/build/lib"; do
  if [[ -f "$d/libmscclpp.so" || -f "$d/mscclpp.so" ]]; then LIBDIR="$d"; break; fi
done
if [[ -z "$INC" || -z "$LIBDIR" ]]; then
  echo "MSCCL++ headers/library not found under MSCCLPP=$MSCCLPP" >&2
  echo "run bench/setup_mscclpp.sh or point MSCCLPP at an installed prefix" >&2
  exit 1
fi

if [[ -f "$LIBDIR/libmscclpp.so" ]]; then
  MSCCLPP_LINK="-lmscclpp"
else
  MSCCLPP_LINK="-l:mscclpp.so"
fi

"$CUDA_HOME/bin/nvcc" -O3 -std=c++20 -arch="$ARCH" \
  -I"$INC" -o "$OUT" "$SCRIPT_DIR/mscclpp_allreduce.cu" \
  -L"$LIBDIR" "$MSCCLPP_LINK" -lcudart -lpthread \
  -Xlinker -rpath="$LIBDIR"

echo "built -> $OUT"
