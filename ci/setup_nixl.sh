#!/usr/bin/env bash
set -euo pipefail
source ci/env.sh
prefix="$CI_REPO/.ci-nixl"
mkdir -p "$prefix" "$CI_DEPS"
if [[ ! -f $CI_DEPS/meson-1.9.1/mesonbuild/__init__.py ]]; then
  python3 -m pip install --require-hashes --no-deps --target "$CI_DEPS/meson-1.9.1" -r ci/requirements-meson.lock
fi
export PYTHONPATH="$CI_DEPS/meson-1.9.1:$PYTHONPATH"
meson() { python3 -m mesonbuild.mesonmain "$@"; }
fetch() { bash ci/fetch.sh "$@"; }

absl="$CI_DEPS/abseil-d7aaad83"
if [[ ! -f $absl/.complete ]]; then
  fetch abseil "$CI_REPO/.ci-abseil"
  cmake -S "$CI_REPO/.ci-abseil" -B "$CI_REPO/.ci-abseil/build" \
    -DCMAKE_INSTALL_PREFIX="$absl" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17 \
    -DABSL_BUILD_TESTING=OFF -DABSL_ENABLE_INSTALL=ON
  cmake --build "$CI_REPO/.ci-abseil/build" -j "$CI_JOBS"
  cmake --install "$CI_REPO/.ci-abseil/build"
  touch "$absl/.complete"
fi
make -C p2p -j"$CI_JOBS" PYTHON=python3 ENABLE_EFA=0 ENABLE_CXI=1 \
  PREFIX="$prefix" LIBDIR="$prefix/lib" INSTALL_DIR="$CI_REPO/uccl" install
export CMAKE_PREFIX_PATH="$absl:$prefix:$CMAKE_PREFIX_PATH"
export PKG_CONFIG_PATH="$absl/lib/pkgconfig:$prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LIBRARY_PATH="$prefix/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$prefix/lib:$absl/lib:$LD_LIBRARY_PATH"
fetch nixl "$CI_REPO/.ci-nixl-source"
meson setup "$CI_REPO/.ci-nixl-source/build" "$CI_REPO/.ci-nixl-source" \
  --prefix="$prefix" --libdir=lib --buildtype=release \
  -Dbuild_tests=false -Dbuild_examples=false -Denable_plugins=UCCL \
  -Dwith_trace=false "-Dcpp_args=-I$prefix/include"
meson compile -C "$CI_REPO/.ci-nixl-source/build" -j "$CI_JOBS"
meson install -C "$CI_REPO/.ci-nixl-source/build"
meson setup "$CI_REPO/.ci-nixl-source/benchmark/nixlbench/build" \
  "$CI_REPO/.ci-nixl-source/benchmark/nixlbench" --prefix="$prefix" --libdir=lib \
  --buildtype=release -Dnixl_path="$prefix" -Dbuild_tests=false
meson compile -C "$CI_REPO/.ci-nixl-source/benchmark/nixlbench/build" -j "$CI_JOBS"
meson install -C "$CI_REPO/.ci-nixl-source/benchmark/nixlbench/build"
