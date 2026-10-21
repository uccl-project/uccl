#!/usr/bin/env bash
set -euo pipefail
: "${ROOT:?Set ROOT to the prepared two-arm build directory}"
: "${LAB:?Set LAB to the existing dependency installation}"

# Recorded build commands, with paths normalized. Run inside the P800 container.
# ROOT contains baseline/ and candidate/. LAB contains the existing headers and Python environment.

# baseline-build
env CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/usr/local/xcudart/lib "LIBRARY_PATH=${LAB}/deps/lib" nice -n 15 make -C "${ROOT}/baseline/p2p" -j2 libuccl_p2p.so "PYTHON=${LAB}/.venv/bin/python" CUDA_HOME=/usr/local/cuda "EFA_HOME=${LAB}/deps" "LIBFABRIC_HOME=${LAB}/deps" PYTHON_LDFLAGS= PYTHON_LIB= CUDA_LIB=/usr/local/xcudart/lib 'LIBS=-L/usr/local/xcudart/lib -lcudart -lcuda -lpthread -lz -lelf -ldl'

# candidate-build
env CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/usr/local/xcudart/lib "LIBRARY_PATH=${LAB}/deps/lib" nice -n 15 make -C "${ROOT}/candidate/p2p" -j2 libuccl_p2p.so "PYTHON=${LAB}/.venv/bin/python" CUDA_HOME=/usr/local/cuda "EFA_HOME=${LAB}/deps" "LIBFABRIC_HOME=${LAB}/deps" PYTHON_LDFLAGS= PYTHON_LIB= CUDA_LIB=/usr/local/xcudart/lib 'LIBS=-L/usr/local/xcudart/lib -lcudart -lcuda -lpthread -lz -lelf -ldl'

# capi-probe-build
env CUDA_VISIBLE_DEVICES= g++ -std=c++17 -O2 "-I${ROOT}/baseline/p2p" "-I${ROOT}/baseline/p2p/util" "-I${LAB}/deps/include" "${ROOT}/uccl_capi_probe.cc" "-L${ROOT}/baseline/p2p" -luccl_p2p -L/usr/local/xcudart/lib -lcudart -lcuda "-Wl,-rpath,${ROOT}/baseline/p2p" -Wl,-rpath,/usr/local/xcudart/lib -L/root/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib -lpython3.10 -Wl,-rpath,/root/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib -o "${ROOT}/uccl_capi_probe"

# runtime-probe-build
env CUDA_VISIBLE_DEVICES= g++ -std=c++17 -O2 -I/usr/local/cuda/include "${ROOT}/runtime_probe.cc" -L/usr/local/xcudart/lib -lcudart -lcuda -Wl,-rpath,/usr/local/xcudart/lib -o "${ROOT}/runtime_probe"

# isolated topology probes (same flags as the recorded builds)
for ARM in baseline candidate; do
  env CUDA_VISIBLE_DEVICES= g++ -std=c++17 -O2 \
    "-I${ROOT}/${ARM}/include" -I/usr/local/cuda/include \
    "${ROOT}/topology_probe.cc" -L/usr/local/xcudart/lib \
    -lcudart -lcuda -lpthread -ldl -Wl,-rpath,/usr/local/xcudart/lib \
    -o "${ROOT}/topology-${ARM}"
done
