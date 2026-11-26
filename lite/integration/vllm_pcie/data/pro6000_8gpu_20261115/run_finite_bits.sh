#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd -P)"
source "$R/runtime_env.sh"
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v | sed -n '/^NCCL_/p; /^VLLM_/p')
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1 NCCL_NET=Socket OMP_NUM_THREADS=1
PYTHON="$R/env/.venv/bin/python"
test -f "$R/quality-repeats/COMPLETE.json"
test ! -e "$R/finite-bitwise"
restore_best_on_error() {
  diagnostic_status=$?
  if (( diagnostic_status != 0 )); then
    "$PYTHON" "$R/controller/snapshot.py" restore c03 || true
  fi
  exit "$diagnostic_status"
}
trap restore_best_on_error EXIT
for diagnostic_snapshot in initial c03; do
  "$PYTHON" "$R/controller/snapshot.py" restore "$diagnostic_snapshot"
  "$PYTHON" -u "$R/controller/run_alias.py" "$diagnostic_snapshot" "$R/finite-bitwise/$diagnostic_snapshot" --test finite-bits
done
"$PYTHON" "$R/controller/snapshot.py" restore c03
