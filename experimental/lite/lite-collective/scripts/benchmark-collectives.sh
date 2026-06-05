#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NCCL_TESTS="${ROOT_DIR}/scripts/run-nccl-tests.sh"

COLLECTIVES_CSV="allreduce,allgather,reducescatter,alltoall"
TOPOLOGIES_CSV="1nx2g,2nx1g,2nx4g"
BACKENDS_CSV="nccl,mscclpp"
HOSTS="${HOSTS:-10.10.55.1,10.10.55.2}"
OUTPUT_DIR="${ROOT_DIR}/.tmp/collective-benchmarks/$(date +%Y%m%d-%H%M%S)"
PROFILE="smoke"
MIN_BYTES=""
MAX_BYTES=""
STEP_FACTOR="2"
ITERS=""
WARMUP_ITERS=""
CHECK_ITERS="1"
NATIVE_ONLY=1
DRY_RUN=0
REBUILD_TESTS=0
REBUILD_MSCCLPP=0
ALLTOALL_OPT="off"
EXTRA_ARGS=()

declare -A TEST_BY_COLLECTIVE=(
  [allreduce]="all_reduce"
  [allgather]="all_gather"
  [reducescatter]="reduce_scatter"
  [alltoall]="alltoall"
)

declare -A TOPOLOGY_MODE=(
  [1nx2g]="intra"
  [2nx1g]="inter"
  [2nx4g]="inter"
)

declare -A TOPOLOGY_GPUS=(
  [1nx2g]="0,1"
  [2nx1g]="0"
  [2nx4g]="0,1,2,3"
)

declare -A TOPOLOGY_RANKS=(
  [1nx2g]="2"
  [2nx1g]="2"
  [2nx4g]="8"
)

die() { echo "error: $*" >&2; exit 1; }
info() { echo "[benchmark-collectives] $*" >&2; }

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra nccl-tests args]

Run a lite-collective benchmark matrix against NCCL.

Options:
  --collectives <csv>       Default: ${COLLECTIVES_CSV}
                             Names: allreduce,allgather,reducescatter,alltoall
  --topologies <csv>        Default: ${TOPOLOGIES_CSV}
                             Names: 1nx2g,2nx1g,2nx4g
  --backends <csv>          Default: ${BACKENDS_CSV}
                             Names: nccl,mscclpp
  --hosts <csv>             Two hosts for inter-node runs.
                             Default: ${HOSTS}
  --output-dir <path>       Default: ${OUTPUT_DIR}
  --profile <name>          smoke, latency, throughput, or full.
                             Default: ${PROFILE}
  --min-bytes <size>        Override profile -b value.
  --max-bytes <size>        Override profile -e value.
  --step-factor <n>         Default: ${STEP_FACTOR}
  --iters <n>               Override profile -n value.
  --warmup-iters <n>        Override profile -w value.
  --check-iters <n>         Pass nccl-tests correctness check: -c <n>.
                             Use 0 to disable. Default: ${CHECK_ITERS}
  --allow-fallback          Allow mscclpp to dlopen NCCL fallback.
                             Default is native-only for mscclpp runs.
  --alltoall-opt <mode>     off, numa, or node. Default: ${ALLTOALL_OPT}
  --rebuild-tests           Rebuild nccl-tests MPI binaries.
  --rebuild-mscclpp         Rebuild lite-collective NCCL shim.
  --dry-run                 Print commands and write run metadata only.
  -h, --help                Show this help.

Examples:
  bash scripts/benchmark-collectives.sh --dry-run
  bash scripts/benchmark-collectives.sh --profile latency --topologies 1nx2g
  bash scripts/benchmark-collectives.sh --profile full --collectives alltoall
EOF
}

csv_to_array() {
  local csv="$1" old_ifs="$IFS"
  local -a items
  IFS=',' read -r -a items <<<"$csv"
  IFS="$old_ifs"
  printf '%s\n' "${items[@]}"
}

fill_csv_array() {
  local csv="$1"
  local -n out="$2"
  local old_ifs="$IFS"
  IFS=',' read -r -a out <<<"$csv"
  IFS="$old_ifs"
}

size_to_bytes() {
  local size="$1"
  local number unit
  if [[ "$size" =~ ^([0-9]+)([KkMmGg]?)$ ]]; then
    number="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "$unit" in
      K|k) echo "$((number * 1024))" ;;
      M|m) echo "$((number * 1024 * 1024))" ;;
      G|g) echo "$((number * 1024 * 1024 * 1024))" ;;
      *) echo "$number" ;;
    esac
  else
    echo ""
  fi
}

effective_min_bytes() {
  local collective="$1" topology="$2"
  local min_bytes="$MIN_BYTES"
  if [[ "$collective" == "alltoall" || "$collective" == "allgather" ||
        "$collective" == "reducescatter" ]]; then
    local parsed min_nonzero
    parsed="$(size_to_bytes "$MIN_BYTES")"
    min_nonzero="$((TOPOLOGY_RANKS[$topology] * 16))"
    if [[ -n "$parsed" && "$parsed" -lt "$min_nonzero" ]]; then
      min_bytes="$min_nonzero"
    fi
  fi
  echo "$min_bytes"
}

effective_max_bytes() {
  local min_bytes="$1"
  local max_bytes="$MAX_BYTES"
  local min_parsed max_parsed
  min_parsed="$(size_to_bytes "$min_bytes")"
  max_parsed="$(size_to_bytes "$MAX_BYTES")"
  if [[ -n "$min_parsed" && -n "$max_parsed" && "$max_parsed" -lt "$min_parsed" ]]; then
    max_bytes="$min_bytes"
  fi
  echo "$max_bytes"
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

set_profile_defaults() {
  case "$PROFILE" in
    smoke)
      : "${MIN_BYTES:=8}"
      : "${MAX_BYTES:=8}"
      : "${ITERS:=1}"
      : "${WARMUP_ITERS:=1}"
      ;;
    latency)
      : "${MIN_BYTES:=8}"
      : "${MAX_BYTES:=64K}"
      : "${ITERS:=1000}"
      : "${WARMUP_ITERS:=100}"
      ;;
    throughput)
      : "${MIN_BYTES:=64K}"
      : "${MAX_BYTES:=1G}"
      : "${ITERS:=50}"
      : "${WARMUP_ITERS:=20}"
      ;;
    full)
      : "${MIN_BYTES:=8}"
      : "${MAX_BYTES:=1G}"
      : "${ITERS:=50}"
      : "${WARMUP_ITERS:=20}"
      ;;
    *)
      die "--profile must be smoke, latency, throughput, or full"
      ;;
  esac
}

validate_inputs() {
  [[ -x "$RUN_NCCL_TESTS" ]] || die "missing executable $RUN_NCCL_TESTS"

  local collective
  while IFS= read -r collective; do
    [[ -n "$collective" ]] || continue
    [[ -n "${TEST_BY_COLLECTIVE[$collective]+x}" ]] \
      || die "unknown collective: $collective"
  done < <(csv_to_array "$COLLECTIVES_CSV")

  local topology
  while IFS= read -r topology; do
    [[ -n "$topology" ]] || continue
    [[ -n "${TOPOLOGY_MODE[$topology]+x}" ]] \
      || die "unknown topology: $topology"
  done < <(csv_to_array "$TOPOLOGIES_CSV")

  local backend
  while IFS= read -r backend; do
    [[ "$backend" == "nccl" || "$backend" == "mscclpp" ]] \
      || die "unknown backend: $backend"
  done < <(csv_to_array "$BACKENDS_CSV")

  [[ "$ALLTOALL_OPT" == "off" || "$ALLTOALL_OPT" == "numa" || "$ALLTOALL_OPT" == "node" ]] \
    || die "--alltoall-opt must be off, numa, or node"
}

append_run_csv() {
  local csv_path="$1" collective="$2" backend="$3" topology="$4" status="$5"
  local exit_code="$6" log_path="$7" command_text="$8"
  {
    printf '%s,%s,%s,%s,%s,' \
      "$collective" "$backend" "$topology" "$status" "$exit_code"
    csv_escape "$log_path"
    printf ','
    csv_escape "$command_text"
    printf '\n'
  } >>"$csv_path"
}

run_case() {
  local collective="$1" backend="$2" topology="$3" runs_csv="$4"
  local test_name="${TEST_BY_COLLECTIVE[$collective]}"
  local topology_mode="${TOPOLOGY_MODE[$topology]}"
  local gpus="${TOPOLOGY_GPUS[$topology]}"
  local case_min_bytes case_max_bytes
  case_min_bytes="$(effective_min_bytes "$collective" "$topology")"
  case_max_bytes="$(effective_max_bytes "$case_min_bytes")"
  local log_dir="${OUTPUT_DIR}/logs/${collective}/${topology}"
  local log_path="${log_dir}/${backend}.log"
  mkdir -p "$log_dir"

  local -a cmd=(
    bash "$RUN_NCCL_TESTS"
    --test "$test_name"
    --backend "$backend"
    --topology "$topology_mode"
    --hosts "$HOSTS"
    --gpus "$gpus"
    --min-bytes "$case_min_bytes"
    --max-bytes "$case_max_bytes"
    --step-factor "$STEP_FACTOR"
    --iters "$ITERS"
    --warmup-iters "$WARMUP_ITERS"
  )

  [[ "$REBUILD_TESTS" -eq 1 ]] && cmd+=(--rebuild-tests)
  [[ "$REBUILD_MSCCLPP" -eq 1 ]] && cmd+=(--rebuild-mscclpp)

  local -a nccl_test_args=()
  nccl_test_args+=(-c "$CHECK_ITERS")
  nccl_test_args+=("${EXTRA_ARGS[@]}")
  if [[ "${#nccl_test_args[@]}" -gt 0 ]]; then
    cmd+=(-- "${nccl_test_args[@]}")
  fi

  local -a env_cmd=(env)
  if [[ "$backend" == "mscclpp" && "$NATIVE_ONLY" -eq 1 ]]; then
    env_cmd+=(
      -u MSCCLPP_NCCL_LIB_PATH
      MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION=
      MSCCLPP_NCCL_LOCAL_P2P_FALLBACK=0
    )
  fi

  if [[ "$backend" == "mscclpp" && "$collective" == "alltoall" && "$ALLTOALL_OPT" != "off" ]]; then
    env_cmd+=(MSCCLPP_NCCL_ALLTOALL_OPT="$ALLTOALL_OPT")
  fi

  local command_text
  command_text="$(printf '%q ' "${env_cmd[@]}" "${cmd[@]}")"
  info "${collective}/${topology}/${backend}: ${command_text}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' "$command_text" >"$log_path"
    append_run_csv "$runs_csv" "$collective" "$backend" "$topology" \
      "DRY_RUN" "0" "$log_path" "$command_text"
    return 0
  fi

  set +e
  "${env_cmd[@]}" "${cmd[@]}" >"$log_path" 2>&1
  local exit_code=$?
  set -e

  local status="PASS"
  if [[ "$exit_code" -ne 0 ]]; then
    status="FAIL"
  fi
  append_run_csv "$runs_csv" "$collective" "$backend" "$topology" \
    "$status" "$exit_code" "$log_path" "$command_text"
  return "$exit_code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collectives) COLLECTIVES_CSV="${2:-}"; shift 2 ;;
    --topologies) TOPOLOGIES_CSV="${2:-}"; shift 2 ;;
    --backends) BACKENDS_CSV="${2:-}"; shift 2 ;;
    --hosts) HOSTS="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --min-bytes) MIN_BYTES="${2:-}"; shift 2 ;;
    --max-bytes) MAX_BYTES="${2:-}"; shift 2 ;;
    --step-factor) STEP_FACTOR="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="${2:-}"; shift 2 ;;
    --check-iters) CHECK_ITERS="${2:-}"; shift 2 ;;
    --allow-fallback) NATIVE_ONLY=0; shift ;;
    --alltoall-opt) ALLTOALL_OPT="${2:-}"; shift 2 ;;
    --rebuild-tests) REBUILD_TESTS=1; shift ;;
    --rebuild-mscclpp) REBUILD_MSCCLPP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "unknown argument: $1" ;;
  esac
done

set_profile_defaults
validate_inputs

mkdir -p "$OUTPUT_DIR"
runs_csv="${OUTPUT_DIR}/runs.csv"
printf 'collective,backend,topology,status,exit_code,log_path,command\n' >"$runs_csv"

declare -a COLLECTIVES TOPOLOGIES BACKENDS
fill_csv_array "$COLLECTIVES_CSV" COLLECTIVES
fill_csv_array "$TOPOLOGIES_CSV" TOPOLOGIES
fill_csv_array "$BACKENDS_CSV" BACKENDS

info "output=${OUTPUT_DIR}"
info "profile=${PROFILE} min=${MIN_BYTES} max=${MAX_BYTES} iters=${ITERS} warmup=${WARMUP_ITERS}"
if [[ "$NATIVE_ONLY" -eq 1 ]]; then
  info "mscclpp native-only mode is enabled"
fi

overall_status=0
for collective in "${COLLECTIVES[@]}"; do
  [[ -n "$collective" ]] || continue
  for topology in "${TOPOLOGIES[@]}"; do
    [[ -n "$topology" ]] || continue
    for backend in "${BACKENDS[@]}"; do
      [[ -n "$backend" ]] || continue
      if ! run_case "$collective" "$backend" "$topology" "$runs_csv"; then
        overall_status=1
      fi
    done
  done
done

info "run metadata: ${runs_csv}"
exit "$overall_status"
