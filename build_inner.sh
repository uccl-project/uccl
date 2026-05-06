#!/bin/bash

# -----------------------
# build_inner.sh — runs *inside* the build container.
# Invoked by build.sh via docker/podman/apptainer; not intended for direct
# execution on the host.
#
# Environment variables consumed (set by build.sh before container launch):
#
#   Required:
#     TARGET                        Build target: cu12, cu13, roc7, roc6, therock
#     PY_VER                        Python version, e.g. "3.10"
#     ARCH                          Host architecture: x86_64 or aarch64
#     BUILD_TYPE                    What to build: all, ccl_rdma, ccl_efa, p2p, ep, p2p_ep, ukernel
#     IS_EFA                        Non-empty string when EFA is detected on the host
#     WHEEL_DIR                     Output directory for built wheels (relative to /io)
#     HOST_GLIBC_VER                Host glibc version string (e.g. "2.35")
#
#   Optional (with defaults):
#     ROCM_IDX_URL                  ROCm package index URL (used by therock target)
#     UCCL_RETAG_TO_HOST_GLIBC      Retag wheel to host glibc version (default "0")
#     UCCL_LOCAL_VERSION            Local version suffix appended to wheel filename (PEP 440)
#
#   Build feature flags:
#     USE_DIETGPU                   Enable DietGPU compression (default "0")
#     USE_INTEL_RDMA_NIC            Enable Intel RDMA NIC / irdma driver (default "0")
#     PER_EXPERT_BATCHING           Enable per-expert batching (default "0")
#     MAKE_NORMAL_MODE              Make normal mode flag
#     TORCH_CUDA_ARCH_LIST          CUDA compute capabilities to compile for
# -----------------------

set -euo pipefail

########################################################
# Main build logic
########################################################

if [[ "$TARGET" == "therock" ]]; then
  PY_V=$(echo ${PY_VER} | tr -d .)
  export PATH=/opt/python/cp${PY_V}-cp${PY_V}/bin:$PATH

  python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
  pip3 install --no-cache-dir --upgrade pip
  pip3 install --no-cache-dir build auditwheel pybind11 nanobind
  pip3 install --no-cache-dir rocm[libraries,devel] --index-url ${ROCM_IDX_URL}
fi


if [[ "$TARGET" == "therock" ]]; then
  echo "
def initialize():
  import rocm_sdk
  rocm_sdk.initialize_process(preload_shortnames=[
    \"amd_comgr\",
    \"amdhip64\",
    \"roctx64\",
    \"hiprtc\",
    \"hipblas\",
    \"hipfft\",
    \"hiprand\",
    \"hipsparse\",
    \"hipsolver\",
    \"rccl\",
    \"hipblaslt\",
    \"miopen\",
  ],
  check_version=\"$(rocm-sdk version)\")
" >uccl/_rocm_init.py

  BACKUP_FN=$(mktemp -p . -t setup.py.XXXXXX)
  cp ./setup.py ${BACKUP_FN}
  sed -i "s/\"rocm\": \[\],/\"rocm\": \[\"rocm\[libraries\]==$(rocm-sdk version)\"\, \"torch\", \"numpy\"],/;" setup.py

  export PIP_EXTRA_INDEX_URL=${ROCM_IDX_URL}
fi

# All native build logic lives in the top-level Makefile, driven by setup.py's
# MakeBuildExtension. ``--no-isolation`` reuses the container's setuptools/wheel.
pip3 install $(python3 -c "
import tomllib
print(' '.join(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))")
python3 -m build --wheel --no-isolation

# Restore the original setup.py if we patched it.
if [[ "$TARGET" == "therock" ]]; then
  mv ${BACKUP_FN} setup.py
fi

# Always use the *container* glibc for auditwheel repair (symbol validation),
# then retag the wheel to the desired host platform afterwards if requested.
CONTAINER_GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])")
AUDIT_PLAT="manylinux_${CONTAINER_GLIBC_VER//./_}_$(uname -m)"

if [[ "${UCCL_RETAG_TO_HOST_GLIBC}" == "1" ]]; then
  UCCL_WHEEL_PLAT="manylinux_${HOST_GLIBC_VER//./_}_$(uname -m)"
  if [[ "${UCCL_WHEEL_PLAT}" != "${AUDIT_PLAT}" ]]; then
    echo "WARNING: UCCL_RETAG_TO_HOST_GLIBC is set." >&2
    echo "  The wheel will be retagged from ${AUDIT_PLAT} to ${UCCL_WHEEL_PLAT}." >&2
    echo "  The binaries are built against the container glibc (${CONTAINER_GLIBC_VER})." >&2
    echo "  If the host glibc is older, the wheel may fail at runtime" >&2
    echo "  due to missing versioned symbols." >&2
  fi
  echo "Host glibc ${HOST_GLIBC_VER}, container glibc ${CONTAINER_GLIBC_VER} -> wheel tagged ${UCCL_WHEEL_PLAT} (force-retag enabled)"
else
  UCCL_WHEEL_PLAT="${AUDIT_PLAT}"
  echo "Container glibc ${CONTAINER_GLIBC_VER} -> wheel tagged ${UCCL_WHEEL_PLAT}"
  if [[ "${HOST_GLIBC_VER}" != "${CONTAINER_GLIBC_VER}" ]]; then
    echo "  Note: host glibc (${HOST_GLIBC_VER}) differs from container glibc (${CONTAINER_GLIBC_VER})."
    echo "  Tip: set UCCL_RETAG_TO_HOST_GLIBC=1 to retag to host glibc ${HOST_GLIBC_VER}."
  fi
fi

auditwheel repair dist/uccl-*.whl \
  --plat "${AUDIT_PLAT}" \
  --exclude "libtorch*.so" \
  --exclude "libc10*.so" \
  --exclude "libibverbs.so.1" \
  --exclude "libcudart.so.12" \
  --exclude "libamdhip64.so.*" \
  --exclude "libcuda.so.1" \
  --exclude "libefa.so.1" \
  --exclude "libglog.so.0" \
  -w /io/${WHEEL_DIR}

# Collapse dual platform tags to the single requested tag.
cd /io/${WHEEL_DIR}
for whl in uccl*.whl; do
  if [[ "$whl" == *-abi3-* ]]; then
    new="${whl%%abi3-*}abi3-${UCCL_WHEEL_PLAT}.whl"
  else
    new="${whl%%-manylinux*}-${UCCL_WHEEL_PLAT}.whl"
  fi
  [[ "$whl" != "$new" ]] && mv "$whl" "$new"
done
cd /io

# Add local version identifier to wheel filename (PEP 440).
if [[ "$TARGET" == "therock" ]]; then
  UCCL_LOCAL_VERSION="rocm$(rocm-sdk version)"
fi
if [[ -n "${UCCL_LOCAL_VERSION:-}" ]]; then
  cd /io/${WHEEL_DIR}
  for wheel in uccl*.whl; do
    if [[ -f "$wheel" ]]; then
      if [[ "$wheel" =~ ^(uccl[^-]*-)([^-]+)-([^-]+-[^-]+-.+)(\.whl)$ ]]; then
        name="${BASH_REMATCH[1]}"
        version="${BASH_REMATCH[2]}"
        python_abi_platform="${BASH_REMATCH[3]}"
        suffix="${BASH_REMATCH[4]}"
        new_wheel="${name}${version}+${UCCL_LOCAL_VERSION}-${python_abi_platform}${suffix}"
        echo "Renaming wheel: $wheel -> $new_wheel"
        mv "$wheel" "$new_wheel"
      else
        echo "Warning: Could not parse wheel filename: $wheel"
      fi
    fi
  done
  cd /io
fi

auditwheel show /io/${WHEEL_DIR}/*.whl
