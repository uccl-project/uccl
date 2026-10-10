#!/usr/bin/env bash

# Exercise build.sh through its public CLI while replacing only the external
# Apptainer executable. This keeps the command-line construction under test
# without building an image or requiring a GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/uccl-apptainer-test.XXXXXX")"
if [[ "${TEMP_ROOT}" != /* || "${TEMP_ROOT}" != */uccl-apptainer-test.* ]]; then
  echo "Unexpected temporary-directory path: ${TEMP_ROOT}" >&2
  exit 1
fi
trap 'rm -rf "${TEMP_ROOT}"' EXIT

BUILD_DIR="${TEMP_ROOT}/build"
FAKE_BIN="${TEMP_ROOT}/bin"
CAPTURE_FILE="${TEMP_ROOT}/apptainer-args"
mkdir -p "${BUILD_DIR}" "${FAKE_BIN}"
cp "${REPO_ROOT}/build.sh" "${BUILD_DIR}/build.sh"

cat >"${FAKE_BIN}/apptainer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "exec" && "${2:-}" == "--help" ]]; then
  printf '%s\n' '--nv' '--rocm'
  exit 0
fi

printf '%s\n' "$@" >"${UCCL_TEST_CAPTURE:?}"
EOF
chmod +x "${FAKE_BIN}/apptainer"

# build.sh only queries the libc version in this harness; avoid depending on
# the host's Python launcher when the test runs under Git Bash on Windows.
cat >"${FAKE_BIN}/python3" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' '2.35'
EOF
chmod +x "${FAKE_BIN}/python3"

run_target() {
  local target="$1"
  local expected_flag="$2"
  local unexpected_flag="$3"
  local output_file="${TEMP_ROOT}/${target}.log"

  rm -f "${CAPTURE_FILE}"
  if ! (
    cd "${BUILD_DIR}"
    PATH="${FAKE_BIN}:${PATH}" \
      CONTAINER_ENGINE=apptainer \
      SKIP_DOCKER_BUILD=1 \
      UCCL_TEST_CAPTURE="${CAPTURE_FILE}" \
      bash ./build.sh "${target}" ccl_rdma 3.12
  ) >"${output_file}" 2>&1; then
    cat "${output_file}" >&2
    return 1
  fi

  grep -Fqx -- "${expected_flag}" "${CAPTURE_FILE}"
  if grep -Fqx -- "${unexpected_flag}" "${CAPTURE_FILE}"; then
    echo "${target} unexpectedly passed ${unexpected_flag} to Apptainer" >&2
    return 1
  fi
}

run_target cu12 --nv --rocm
run_target cu13 --nv --rocm
run_target roc6 --rocm --nv
run_target roc7 --rocm --nv
run_target therock --rocm --nv

# Keep the existing usage failure independent of any container runtime.
if (
  cd "${BUILD_DIR}"
  bash ./build.sh unsupported-target ccl_rdma 3.12
) >"${TEMP_ROOT}/usage.log" 2>&1; then
  echo "build.sh unexpectedly accepted an unsupported target" >&2
  exit 1
fi
grep -Fq 'Usage:' "${TEMP_ROOT}/usage.log"

echo "Apptainer GPU flag regression test passed"
