#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/home/yangz/nfs/zhongjie/nccl-tests}"
EXTERNAL_NCCL_DIR="${EXTERNAL_NCCL_DIR:-/home/yangz/nfs/zhongjie/nccl}"
MPI_HOME="${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.7rc1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

SDK_DIR="${SDK_DIR:-.tmp/mint-nccl-sdk}"
BUILD_DIR="${BUILD_DIR:-.tmp/mint-nccl-tests-mpi-build}"
RUNTIME_ROOT="${RUNTIME_ROOT:-.tmp/mint-nccl-tests-runtime}"

BACKEND="mscclpp"
GPU_LIST="0,1"
MIN_BYTES="8"
MAX_BYTES="256M"
STEP_FACTOR="2"
ITERS="20"
WARMUP_ITERS="5"
REBUILD_TESTS=0
REBUILD_MSCCLPP=0
REAL_NCCL_LIB=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra nccl-tests args]

Build and run MPI-mode nccl-tests all_reduce_perf on a single node with 2 GPUs.
The binary is compiled against standard external NCCL headers, then run with:
  - backend=nccl: real libnccl.so
  - backend=mscclpp: this repo's libmscclpp_nccl.so

Options:
  --backend <nccl|mscclpp>   Backend to use. Default: mscclpp
  --gpus <csv>               Visible GPU list. Default: 0,1,2,3
  --min-bytes <size>         nccl-tests -b value. Default: 8
  --max-bytes <size>         nccl-tests -e value. Default: 256M
  --step-factor <n>          nccl-tests -f value. Default: 2
  --iters <n>                nccl-tests -n value. Default: 20
  --warmup-iters <n>         nccl-tests -w value. Default: 5
  --nccl-lib <path>          Explicit real NCCL shared library path
  --rebuild-tests            Rebuild MPI nccl-tests before running
  --rebuild-mscclpp          Rebuild this repo's mscclpp NCCL shim before running
  -h, --help                 Show this help

Examples:
  bash scripts/run_nccl_tests_allreduce_mpi.sh --backend nccl
  bash scripts/run_nccl_tests_allreduce_mpi.sh --backend mscclpp
  bash scripts/run_nccl_tests_allreduce_mpi.sh --backend mscclpp --max-bytes 1G
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "[run_nccl_tests_allreduce_mpi] $*" >&2
}

count_csv_items() {
  local csv="$1"
  local old_ifs="$IFS"
  local -a items
  IFS=',' read -r -a items <<<"$csv"
  IFS="$old_ifs"
  echo "${#items[@]}"
}

pick_real_nccl_lib() {
  if [[ -n "${REAL_NCCL_LIB}" ]]; then
    [[ -f "${REAL_NCCL_LIB}" ]] || die "--nccl-lib does not exist: ${REAL_NCCL_LIB}"
    echo "${REAL_NCCL_LIB}"
    return
  fi

  if [[ -n "${NCCL_LIB_PATH:-}" && -f "${NCCL_LIB_PATH}" ]]; then
    echo "${NCCL_LIB_PATH}"
    return
  fi

  local candidate
  candidate="$(find "${EXTERNAL_NCCL_DIR}" -maxdepth 4 -type f \
    \( -name 'libnccl.so' -o -name 'libnccl.so.*' \) | sort -V | tail -n 1)"
  [[ -n "${candidate}" ]] || die "unable to find libnccl.so; pass --nccl-lib or set NCCL_LIB_PATH"
  echo "${candidate}"
}

ensure_mscclpp_nccl_lib() {
  local lib_path="${ROOT_DIR}/nccl/build/libmscclpp_nccl.so"
  if [[ ! -f "${lib_path}" || "${REBUILD_MSCCLPP}" -eq 1 ]]; then
    info "building mscclpp NCCL shim"
    make -C "${ROOT_DIR}/nccl"
  fi
  [[ -f "${lib_path}" ]] || die "missing ${lib_path} after build"
  echo "${lib_path}"
}

prepare_external_nccl_sdk() {
  local real_nccl_lib="$1"

  mkdir -p "${SDK_DIR}/include" "${SDK_DIR}/lib"

  [[ -f "${EXTERNAL_NCCL_DIR}/build/include/nccl.h" ]] || \
    die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl.h"
  [[ -f "${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h" ]] || \
    die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h"
  [[ -d "${EXTERNAL_NCCL_DIR}/build/include/nccl_device" ]] || \
    die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl_device"

  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl.h" "${SDK_DIR}/include/nccl.h"
  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h" "${SDK_DIR}/include/nccl_device.h"
  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl_device" "${SDK_DIR}/include/nccl_device"
  ln -sfn "${real_nccl_lib}" "${SDK_DIR}/lib/libnccl.so"
  ln -sfn "${real_nccl_lib}" "${SDK_DIR}/lib/libnccl.so.2"
}

ensure_mpi_nccl_tests_binary() {
  local bin_path="${BUILD_DIR}/all_reduce_perf_mpi"
  if [[ -x "${bin_path}" && "${REBUILD_TESTS}" -eq 0 ]]; then
    echo "${bin_path}"
    return
  fi

  local real_nccl_lib
  real_nccl_lib="$(pick_real_nccl_lib)"
  prepare_external_nccl_sdk "${real_nccl_lib}"

  info "building MPI nccl-tests all_reduce_perf_mpi"
  make -C "${NCCL_TESTS_DIR}/src" \
    "${bin_path}" \
    BUILDDIR="${BUILD_DIR}" \
    MPI=1 \
    MPI_HOME="${MPI_HOME}" \
    NCCL_HOME="${SDK_DIR}" \
    NAME_SUFFIX=_mpi >&2

  [[ -x "${bin_path}" ]] || die "missing ${bin_path} after build"
  echo "${bin_path}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --gpus)
      GPU_LIST="${2:-}"
      shift 2
      ;;
    --min-bytes)
      MIN_BYTES="${2:-}"
      shift 2
      ;;
    --max-bytes)
      MAX_BYTES="${2:-}"
      shift 2
      ;;
    --step-factor)
      STEP_FACTOR="${2:-}"
      shift 2
      ;;
    --iters)
      ITERS="${2:-}"
      shift 2
      ;;
    --warmup-iters)
      WARMUP_ITERS="${2:-}"
      shift 2
      ;;
    --nccl-lib)
      REAL_NCCL_LIB="${2:-}"
      shift 2
      ;;
    --rebuild-tests)
      REBUILD_TESTS=1
      shift
      ;;
    --rebuild-mscclpp)
      REBUILD_MSCCLPP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${BACKEND}" == "nccl" || "${BACKEND}" == "mscclpp" ]] || \
  die "--backend must be nccl or mscclpp"

GPU_COUNT="$(count_csv_items "${GPU_LIST}")"
[[ "${GPU_COUNT}" -eq 2 ]] || \
  die "this helper expects exactly 2 GPUs; got ${GPU_COUNT} from --gpus ${GPU_LIST}"

[[ -x "${MPI_HOME}/bin/mpirun" || -x "$(command -v mpirun 2>/dev/null || true)" ]] || \
  die "mpirun not found"

BIN_PATH="$(ensure_mpi_nccl_tests_binary)"
RUNTIME_DIR="${RUNTIME_ROOT}/${BACKEND}"
mkdir -p "${RUNTIME_DIR}"

if [[ "${BACKEND}" == "mscclpp" ]]; then
  ACTIVE_LIB="$(ensure_mscclpp_nccl_lib)"
  EXTRA_LD_PATH="${ROOT_DIR}/build:${ROOT_DIR}/nccl/build"
else
  ACTIVE_LIB="$(pick_real_nccl_lib)"
  EXTRA_LD_PATH=""
fi

ln -sfn "${ACTIVE_LIB}" "${RUNTIME_DIR}/libnccl.so"
ln -sfn "${ACTIVE_LIB}" "${RUNTIME_DIR}/libnccl.so.2"

LD_LIBRARY_PATH_VALUE="${RUNTIME_DIR}:${MPI_HOME}/lib:${CUDA_HOME}/lib64"
if [[ -n "${EXTRA_LD_PATH}" ]]; then
  LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${EXTRA_LD_PATH}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${LD_LIBRARY_PATH}"
fi

info "backend=${BACKEND}"
info "gpus=${GPU_LIST}"
info "binary=${BIN_PATH}"
info "libnccl=$(readlink -f "${RUNTIME_DIR}/libnccl.so.2")"

exec mpirun -np "${GPU_COUNT}" \
  --bind-to none \
  -x CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  -x LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" \
  -x NCCL_DEBUG=INFO \
  -x NCCL_P2P_LEVEL=SYS \
  "${BIN_PATH}" \
  -g 1 \
  -b "${MIN_BYTES}" \
  -e "${MAX_BYTES}" \
  -f "${STEP_FACTOR}" \
  -w "${WARMUP_ITERS}" \
  -n "${ITERS}" \
  "${EXTRA_ARGS[@]}"
