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

BACKEND=""
GPU_LIST=""
TEST_PROGRAM=""
TOPOLOGY=""  # "inter" or "intra"
MIN_BYTES="8"
MAX_BYTES="256M"
STEP_FACTOR="2"
ITERS="20"
WARMUP_ITERS="10"
REBUILD_TESTS=0
REBUILD_MSCCLPP=0
REAL_NCCL_LIB=""
EXTRA_ARGS=()
NCCL_SOCKET_IFNAME="eno8303"
NCCL_BUFFSIZE=1073741824
NCCL_NET_GDR_LEVEL=

# Read default hosts from ip.txt (one IP per line)
IP_FILE="${ROOT_DIR}/ip.txt"
if [[ -f "$IP_FILE" ]]; then
  HOSTS="$(paste -sd',' "$IP_FILE")"
else
  echo "warning: ${IP_FILE} not found, set --hosts or create ip.txt with one IP per line" >&2
  HOSTS=""
fi

# ---------- Presets: GPU configurations per topology ----------
declare -A GPU_PRESETS_INTER=(
  [2GPU]="0"        # 1 GPU/node × 2 nodes
  [4GPU]="0,1"      # 2 GPUs/node × 2 nodes
  [8GPU]="0,1,2,3"  # 4 GPUs/node × 2 nodes
)
PRESET_ORDER_INTER=(2GPU 4GPU 8GPU)

declare -A GPU_PRESETS_INTRA=(
  [2GPU]="0,1"        # 2 GPUs on 1 node
  [4GPU]="0,1,2,3"    # 4 GPUs on 1 node
)
PRESET_ORDER_INTRA=(2GPU 4GPU)

# ---------- Test programs from nccl-tests ----------
# Programs marked with * are natively supported by uccl-lite (no NCCL fallback).
# Others will run via the dlopen NCCL fallback path when using mscclpp backend.
declare -A TEST_PROGRAMS=(
  [sendrecv]="sendrecv_perf"
  [all_reduce]="all_reduce_perf"
  [all_gather]="all_gather_perf"
  [broadcast]="broadcast_perf"
  [reduce_scatter]="reduce_scatter_perf"
  [reduce]="reduce_perf"
  [alltoall]="alltoall_perf"
  [scatter]="scatter_perf"
  [gather]="gather_perf"
  [hypercube]="hypercube_perf"
)
TEST_PROGRAM_ORDER=(sendrecv all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather hypercube)
# Native support in uccl-lite (no fallback needed with mscclpp backend):
NATIVE_PROGRAMS="sendrecv"

# ---------- Helpers ----------

die()  { echo "error: $*" >&2; exit 1; }
info() { echo "[run_p2p] $*" >&2; }

count_csv_items() {
  local csv="$1" old_ifs="$IFS"
  local -a items
  IFS=',' read -r -a items <<<"$csv"
  IFS="$old_ifs"
  echo "${#items[@]}"
}

csv_to_lines() { echo "$1" | tr ',' '\n'; }

build_host_spec() {
  local csv="$1" slots="$2" old_ifs="$IFS"
  local -a hosts host_spec
  IFS=',' read -r -a hosts <<<"$csv"
  IFS="$old_ifs"
  for host in "${hosts[@]}"; do host_spec+=("${host}:${slots}"); done
  local joined="" idx
  for idx in "${!host_spec[@]}"; do
    [[ "$idx" -gt 0 ]] && joined+=","
    joined+="${host_spec[$idx]}"
  done
  echo "$joined"
}

is_local_host() {
  local target="$1" current
  for current in \
    "localhost" "127.0.0.1" "$(hostname)" "$(hostname -s)" \
    "$(hostname -f 2>/dev/null || true)"; do
    [[ -n "$current" && "$target" == "$current" ]] && return 0
  done
  local ip
  for ip in $(hostname -I 2>/dev/null || true); do
    [[ "$target" == "$ip" ]] && return 0
  done
  return 1
}

ensure_ssh_access() {
  local host
  for host in $(csv_to_lines "$HOSTS"); do
    is_local_host "$host" && continue
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" true >/dev/null 2>&1 \
      || die "passwordless ssh to $host is required by mpirun"
  done
}

pick_real_nccl_lib() {
  if [[ -n "$REAL_NCCL_LIB" ]]; then
    [[ -f "$REAL_NCCL_LIB" ]] || die "--nccl-lib does not exist: $REAL_NCCL_LIB"
    echo "$REAL_NCCL_LIB"; return
  fi
  if [[ -n "${NCCL_LIB_PATH:-}" && -f "$NCCL_LIB_PATH" ]]; then
    echo "$NCCL_LIB_PATH"; return
  fi
  local candidate
  candidate="$(find "$EXTERNAL_NCCL_DIR" -maxdepth 4 -type f \
    \( -name 'libnccl.so' -o -name 'libnccl.so.*' \) | sort -V | tail -n 1)"
  [[ -n "$candidate" ]] || die "unable to find libnccl.so; pass --nccl-lib or set NCCL_LIB_PATH"
  echo "$candidate"
}

ensure_mscclpp_nccl_lib() {
  local lib_path="${ROOT_DIR}/nccl/build/libmscclpp_nccl.so"
  if [[ ! -f "$lib_path" || "$REBUILD_MSCCLPP" -eq 1 ]]; then
    info "building mscclpp NCCL shim"
    make -C "${ROOT_DIR}/nccl"
  fi
  [[ -f "$lib_path" ]] || die "missing $lib_path after build"
  echo "$lib_path"
}

prepare_external_nccl_sdk() {
  local real_nccl_lib="$1"
  mkdir -p "${SDK_DIR}/include" "${SDK_DIR}/lib"
  [[ -f "${EXTERNAL_NCCL_DIR}/build/include/nccl.h" ]] \
    || die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl.h"
  [[ -f "${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h" ]] \
    || die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h"
  [[ -d "${EXTERNAL_NCCL_DIR}/build/include/nccl_device" ]] \
    || die "missing ${EXTERNAL_NCCL_DIR}/build/include/nccl_device"
  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl.h"        "${SDK_DIR}/include/nccl.h"
  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl_device.h"  "${SDK_DIR}/include/nccl_device.h"
  ln -sfn "${EXTERNAL_NCCL_DIR}/build/include/nccl_device"     "${SDK_DIR}/include/nccl_device"
  ln -sfn "$real_nccl_lib" "${SDK_DIR}/lib/libnccl.so"
  ln -sfn "$real_nccl_lib" "${SDK_DIR}/lib/libnccl.so.2"
}

ensure_mpi_nccl_tests_binary() {
  local perf_name="$1"  # e.g. "sendrecv_perf"
  local bin_path="${BUILD_DIR}/${perf_name}_mpi"
  if [[ -x "$bin_path" && "$REBUILD_TESTS" -eq 0 ]]; then
    echo "$bin_path"; return
  fi
  local real_nccl_lib
  real_nccl_lib="$(pick_real_nccl_lib)"
  prepare_external_nccl_sdk "$real_nccl_lib"
  info "building MPI nccl-tests ${perf_name}_mpi"
  make -C "${NCCL_TESTS_DIR}/src" \
    "$bin_path" BUILDDIR="$BUILD_DIR" MPI=1 \
    MPI_HOME="$MPI_HOME" NCCL_HOME="$SDK_DIR" NAME_SUFFIX=_mpi >&2
  [[ -x "$bin_path" ]] || die "missing $bin_path after build"
  echo "$bin_path"
}

# ---------- Interactive menu (only when a choice wasn't given via CLI) ----------

prompt_choice() {
  local prompt="$1"
  shift
  local -a options=("$@")
  local i

  echo "" >&2
  echo "$prompt" >&2
  for i in "${!options[@]}"; do
    echo "  $((i + 1))) ${options[$i]}" >&2
  done
  while true; do
    read -rp "> " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] \
       && (( choice >= 1 && choice <= ${#options[@]} )); then
      echo "${options[$((choice - 1))]}"
      return
    fi
    echo "  invalid selection, enter 1-${#options[@]}" >&2
  done
}

# ---------- Usage ----------

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra nccl-tests args]

Build and run MPI-mode nccl-tests.
If --test, --backend, --topology, or --gpus are not given, an interactive menu
is shown.

Options:
  --test <name>              Test program to run (or choose interactively).
                             Available: ${TEST_PROGRAM_ORDER[*]}
                             Native uccl-lite support: ${NATIVE_PROGRAMS}
  --backend <nccl|mscclpp>   Backend to use (or choose interactively)
  --topology <inter|intra>   inter-node (2 hosts) or intra-node (1 host)
  --hosts <csv>              Hosts for MPI launch. Default: from ip.txt
  --gpus <csv>               GPU list per host (e.g., 0  0,1  0,1,2,3)
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
  bash scripts/run_p2p.sh                                       # fully interactive
  bash scripts/run_p2p.sh --test sendrecv --backend mscclpp --topology inter --gpus 0,1
  bash scripts/run_p2p.sh --test all_reduce --backend nccl --topology intra --gpus 0,1,2,3
EOF
}

# ---------- Parse CLI args ----------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)          TEST_PROGRAM="${2:-}";   shift 2 ;;
    --topology)      TOPOLOGY="${2:-}";      shift 2 ;;
    --backend)       BACKEND="${2:-}";       shift 2 ;;
    --hosts)         HOSTS="${2:-}";         shift 2 ;;
    --gpus)          GPU_LIST="${2:-}";      shift 2 ;;
    --min-bytes)     MIN_BYTES="${2:-}";     shift 2 ;;
    --max-bytes)     MAX_BYTES="${2:-}";     shift 2 ;;
    --step-factor)   STEP_FACTOR="${2:-}";   shift 2 ;;
    --iters)         ITERS="${2:-}";         shift 2 ;;
    --warmup-iters)  WARMUP_ITERS="${2:-}";  shift 2 ;;
    --nccl-lib)      REAL_NCCL_LIB="${2:-}"; shift 2 ;;
    --rebuild-tests)    REBUILD_TESTS=1;     shift ;;
    --rebuild-mscclpp)  REBUILD_MSCCLPP=1;   shift ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; EXTRA_ARGS=("$@"); break ;;
    *)               die "unknown argument: $1" ;;
  esac
done

# ---------- Interactive prompts for missing choices ----------

if [[ -z "$TEST_PROGRAM" ]]; then
  # Build display labels: mark native programs with *
  test_labels=()
  for t in "${TEST_PROGRAM_ORDER[@]}"; do
    if [[ " $NATIVE_PROGRAMS " == *" $t "* ]]; then
      test_labels+=("$t  (native)")
    else
      test_labels+=("$t")
    fi
  done
  selection="$(prompt_choice "Select test program (* = native uccl-lite support):" "${test_labels[@]}")"
  # Strip the label suffix to get the bare program name.
  TEST_PROGRAM="${selection%%  *}"
fi

[[ -n "${TEST_PROGRAMS[$TEST_PROGRAM]+x}" ]] \
  || die "unknown test program: $TEST_PROGRAM (available: ${TEST_PROGRAM_ORDER[*]})"
PERF_BINARY_NAME="${TEST_PROGRAMS[$TEST_PROGRAM]}"

if [[ -z "$BACKEND" ]]; then
  BACKEND="$(prompt_choice "Select backend:" mscclpp nccl)"
fi

if [[ -z "$TOPOLOGY" ]]; then
  TOPOLOGY="$(prompt_choice "Select topology:" "inter-node  (2 hosts)" "intra-node  (1 host)")"
  TOPOLOGY="${TOPOLOGY%%  *}"
  # Normalize
  case "$TOPOLOGY" in
    inter-node) TOPOLOGY="inter" ;;
    intra-node) TOPOLOGY="intra" ;;
  esac
fi

[[ "$TOPOLOGY" == "inter" || "$TOPOLOGY" == "intra" ]] \
  || die "--topology must be inter or intra"

if [[ -z "$GPU_LIST" ]]; then
  if [[ "$TOPOLOGY" == "inter" ]]; then
    preset="$(prompt_choice "Select GPU configuration (per node × 2 nodes):" "${PRESET_ORDER_INTER[@]}")"
    GPU_LIST="${GPU_PRESETS_INTER[$preset]}"
  else
    preset="$(prompt_choice "Select GPU configuration (single node):" "${PRESET_ORDER_INTRA[@]}")"
    GPU_LIST="${GPU_PRESETS_INTRA[$preset]}"
  fi
fi

[[ "$BACKEND" == "nccl" || "$BACKEND" == "mscclpp" ]] \
  || die "--backend must be nccl or mscclpp"

# ---------- Validate hosts ----------

GPU_COUNT="$(count_csv_items "$GPU_LIST")"

if [[ "$TOPOLOGY" == "inter" ]]; then
  HOST_COUNT="$(count_csv_items "$HOSTS")"
  [[ "$HOST_COUNT" -eq 2 ]] \
    || die "inter-node mode expects exactly 2 hosts; got ${HOST_COUNT} from --hosts ${HOSTS}"
  [[ -x "${MPI_HOME}/bin/mpirun" || -x "$(command -v mpirun 2>/dev/null || true)" ]] \
    || die "mpirun not found"
  ensure_ssh_access
  TOTAL_RANKS="$((HOST_COUNT * GPU_COUNT))"
  HOST_SPEC="$(build_host_spec "$HOSTS" "$GPU_COUNT")"
else
  # Intra-node: single host, no MPI host spec needed.
  HOST_COUNT=1
  [[ -x "${MPI_HOME}/bin/mpirun" || -x "$(command -v mpirun 2>/dev/null || true)" ]] \
    || die "mpirun not found"
  TOTAL_RANKS="$GPU_COUNT"
  HOST_SPEC=""
fi

# ---------- Prepare runtime ----------

BIN_PATH="$(ensure_mpi_nccl_tests_binary "$PERF_BINARY_NAME")"
RUNTIME_DIR="${RUNTIME_ROOT}/${BACKEND}"
mkdir -p "$RUNTIME_DIR"

if [[ "$BACKEND" == "mscclpp" ]]; then
  ACTIVE_LIB="$(ensure_mscclpp_nccl_lib)"
  EXTRA_LD_PATH="${ROOT_DIR}/build:${ROOT_DIR}/nccl/build"
else
  ACTIVE_LIB="$(pick_real_nccl_lib)"
  EXTRA_LD_PATH=""
  # Tune NCCL P2P for best host-staged RDMA throughput on this hardware.
  : "${NCCL_P2P_NET_CHUNKSIZE:=2097152}"
  export NCCL_P2P_NET_CHUNKSIZE
fi

ln -sfn "$ACTIVE_LIB" "${RUNTIME_DIR}/libnccl.so"
ln -sfn "$ACTIVE_LIB" "${RUNTIME_DIR}/libnccl.so.2"

LD_LIBRARY_PATH_VALUE="${RUNTIME_DIR}:${MPI_HOME}/lib:${CUDA_HOME}/lib64"
[[ -n "$EXTRA_LD_PATH" ]] && LD_LIBRARY_PATH_VALUE+=":${EXTRA_LD_PATH}"
[[ -n "${LD_LIBRARY_PATH:-}" ]] && LD_LIBRARY_PATH_VALUE+=":${LD_LIBRARY_PATH}"

info "test=${TEST_PROGRAM}"
info "backend=${BACKEND}"
info "topology=${TOPOLOGY}"
if [[ "$TOPOLOGY" == "inter" ]]; then
  info "hosts=${HOSTS}"
fi
info "gpus=${GPU_LIST}"
info "ranks=${TOTAL_RANKS}"
info "binary=${BIN_PATH}"
info "libnccl=$(readlink -f "${RUNTIME_DIR}/libnccl.so.2")"

# ---------- Build MPI environment ----------

MPI_ENV_ARGS=(
  -x "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_VALUE}"
)

[[ -z "${MSCCLPP_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]] \
  && MSCCLPP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
[[ -z "${MSCCLPP_HCA_DEVICES:-}" && -n "${NCCL_IB_HCA:-}" ]] \
  && MSCCLPP_HCA_DEVICES="${NCCL_IB_HCA}"

for env_name in \
  NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
  NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_P2P_NET_CHUNKSIZE \
  NCCL_NET_GDR_LEVEL NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_BUFFSIZE \
  MSCCLPP_DEBUG MSCCLPP_DEBUG_SUBSYS MSCCLPP_SOCKET_IFNAME \
  MSCCLPP_HCA_DEVICES MSCCLPP_LOG_LEVEL MSCCLPP_LOG_SUBSYS \
  MSCCLPP_NCCL_LIB_PATH MSCCLPP_NCCL_SENDRECV_STAGING_BYTES; do
  [[ -n "${!env_name:-}" ]] && MPI_ENV_ARGS+=(-x "${env_name}=${!env_name}")
done

# ---------- Launch ----------

MPI_HOST_ARGS=()
if [[ -n "$HOST_SPEC" ]]; then
  MPI_HOST_ARGS=(--host "$HOST_SPEC")
fi

exec mpirun -np "${TOTAL_RANKS}" \
  "${MPI_HOST_ARGS[@]}" \
  --bind-to none \
  "${MPI_ENV_ARGS[@]}" \
  "${BIN_PATH}" \
  -g 1 \
  -b "${MIN_BYTES}" \
  -e "${MAX_BYTES}" \
  -f "${STEP_FACTOR}" \
  -w "${WARMUP_ITERS}" \
  -n "${ITERS}" \
  "${EXTRA_ARGS[@]}"
