#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd -P)"
source "$R/runtime_env.sh"
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v | sed -n '/^NCCL_/p; /^VLLM_/p')
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1 NCCL_NET=Socket OMP_NUM_THREADS=1
PYTHON="$R/env/.venv/bin/python"
for mode in p2p default; do
  "$PYTHON" "$R/controller/final_fingerprint.py" c03 "$R/forced-p2p/$mode.toml" "$R/forced-p2p/$mode-before.json"
  "$PYTHON" -u "$R/forced-p2p/run.py" "$mode"
  "$PYTHON" "$R/controller/final_fingerprint.py" c03 "$R/forced-p2p/$mode.toml" "$R/forced-p2p/$mode-after.json" --compare "$R/forced-p2p/$mode-before.json"
done
