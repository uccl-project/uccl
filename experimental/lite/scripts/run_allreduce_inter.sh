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
# Read default hosts from ip.txt (one IP per line)
IP_FILE="${ROOT_DIR}/ip.txt"
if [[ -f "$IP_FILE" ]]; then
  HOSTS="$(paste -sd',' "$IP_FILE")"
else
  echo "warning: ${IP_FILE} not found, set --hosts or create ip.txt with one IP per line" >&2
  HOSTS=""
fi
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
NCCL_SOCKET_IFNAME="eno8303"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra nccl-tests args]

Build and run MPI-mode nccl-tests all_reduce_perf across 2 nodes with 4 GPUs total.
Default hosts are read from ip.txt (one IP per line).

The binary is compiled against standard external NCCL headers, then run with:
  - backend=nccl: real libnccl.so
  - backend=mscclpp: this repo's libmscclpp_nccl.so

Options:
  --backend <nccl|mscclpp>   Backend to use. Default: mscclpp
  --hosts <csv>              Two hosts for MPI launch. Default: from ip.txt
  --gpus <csv>               Visible GPU list on each host. Default: 0,1
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
  bash scripts/run_allreduce_inter_4GPU.sh --backend nccl
  bash scripts/run_allreduce_inter_4GPU.sh --backend mscclpp
  bash scripts/run_allreduce_inter_4GPU.sh --hosts HOST1,HOST2 --gpus 0,1
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "[run_allreduce_inter_4GPU] $*" >&2
}

count_csv_items() {
  local csv="$1"
  local old_ifs="$IFS"
  local -a items
  IFS=',' read -r -a items <<<"$csv"
  IFS="$old_ifs"
  echo "${#items[@]}"
}

csv_to_lines() {
  local csv="$1"
  echo "${csv}" | tr ',' '\n'
}

build_host_spec() {
  local csv="$1"
  local slots="$2"
  local old_ifs="$IFS"
  local -a hosts
  local -a host_spec
  local host

  IFS=',' read -r -a hosts <<<"$csv"
  IFS="$old_ifs"

  for host in "${hosts[@]}"; do
    host_spec+=("${host}:${slots}")
  done

  local joined=""
  local idx
  for idx in "${!host_spec[@]}"; do
    if [[ "${idx}" -gt 0 ]]; then
      joined+=","
    fi
    joined+="${host_spec[${idx}]}"
  done

  echo "${joined}"
}

is_local_host() {
  local target="$1"
  local current

  for current in \
    "localhost" \
    "127.0.0.1" \
    "$(hostname)" \
    "$(hostname -s)" \
    "$(hostname -f 2>/dev/null || true)"; do
    if [[ -n "${current}" && "${target}" == "${current}" ]]; then
      return 0
    fi
  done

  local ip
  for ip in $(hostname -I 2>/dev/null || true); do
    if [[ "${target}" == "${ip}" ]]; then
      return 0
    fi
  done

  return 1
}

ensure_ssh_access() {
  local host

  for host in $(csv_to_lines "${HOSTS}"); do
    if is_local_host "${host}"; then
      continue
    fi

    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" true >/dev/null 2>&1; then
      die "passwordless ssh to ${host} is required by mpirun; test with: ssh -o BatchMode=yes ${host} true"
    fi
  done
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
    --hosts)
      HOSTS="${2:-}"
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

HOST_COUNT="$(count_csv_items "${HOSTS}")"
GPU_COUNT="$(count_csv_items "${GPU_LIST}")"
[[ "${HOST_COUNT}" -eq 2 ]] || \
  die "this helper expects exactly 2 hosts; got ${HOST_COUNT} from --hosts ${HOSTS}"

[[ -x "${MPI_HOME}/bin/mpirun" || -x "$(command -v mpirun 2>/dev/null || true)" ]] || \
  die "mpirun not found"

ensure_ssh_access

TOTAL_RANKS="$((HOST_COUNT * GPU_COUNT))"
HOST_SPEC="$(build_host_spec "${HOSTS}" "${GPU_COUNT}")"
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
info "hosts=${HOSTS}"
info "gpus=${GPU_LIST}"
info "ranks=${TOTAL_RANKS}"
info "binary=${BIN_PATH}"
info "libnccl=$(readlink -f "${RUNTIME_DIR}/libnccl.so.2")"

MPI_ENV_ARGS=(
  -x "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_VALUE}"
)

for env_name in \
  NCCL_SOCKET_IFNAME \
  NCCL_IB_HCA \
  NCCL_IB_GID_INDEX \
  NCCL_DEBUG \
  NCCL_DEBUG_SUBSYS \
  NCCL_IB_DISABLE \
  NCCL_P2P_DISABLE \
  MSCCLPP_LOG_LEVEL \
  MSCCLPP_LOG_SUBSYS; do
  if [[ -n "${!env_name:-}" ]]; then
    MPI_ENV_ARGS+=(-x "${env_name}=${!env_name}")
  fi
done

exec mpirun -np "${TOTAL_RANKS}" \
  --host "${HOST_SPEC}" \
  --bind-to none \
  -x NCCL_P2P_LEVEL=SYS \
  "${MPI_ENV_ARGS[@]}" \
  "${BIN_PATH}" \
  -g 1 \
  -b "${MIN_BYTES}" \
  -e "${MAX_BYTES}" \
  -f "${STEP_FACTOR}" \
  -w "${WARMUP_ITERS}" \
  -n "${ITERS}" \
  "${EXTRA_ARGS[@]}"
