#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd -P)"
source "$R/runtime_env.sh"
BEST_SNAPSHOT="${1:?Pass the retained snapshot after all ten loops}"
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v | sed -n '/^NCCL_/p; /^VLLM_/p')
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1 NCCL_NET=Socket OMP_NUM_THREADS=1
PYTHON="$R/env/.venv/bin/python"
restore_on_error() {
  suite_status=$?
  if (( suite_status != 0 )); then
    "$PYTHON" "$R/controller/snapshot.py" restore "$BEST_SNAPSHOT" || true
  fi
  exit "$suite_status"
}
trap restore_on_error EXIT
"$PYTHON" "$R/controller/snapshot.py" restore "$BEST_SNAPSHOT"
"$PYTHON" "$R/controller/final_fingerprint.py" "$BEST_SNAPSHOT" "$R/controller/final-nccl.toml" "$R/final-audit/best-before.json"
"$PYTHON" -u "$R/controller/reproduce.py" "$R/controller/final-nccl.toml" "$R/controller/final-pcieccl.toml" --quality
"$PYTHON" "$R/controller/final_fingerprint.py" "$BEST_SNAPSHOT" "$R/controller/final-nccl.toml" "$R/final-audit/best-after.json" --compare "$R/final-audit/best-before.json"
"$PYTHON" "$R/controller/snapshot.py" restore initial
"$PYTHON" "$R/controller/final_fingerprint.py" initial "$R/controller/final-original-pcieccl.toml" "$R/final-audit/original-before.json"
"$PYTHON" -u "$R/controller/reproduce.py" "$R/controller/final-original-pcieccl.toml"
"$PYTHON" "$R/controller/final_fingerprint.py" initial "$R/controller/final-original-pcieccl.toml" "$R/final-audit/original-after.json" --compare "$R/final-audit/original-before.json"
"$PYTHON" "$R/controller/snapshot.py" restore "$BEST_SNAPSHOT"
