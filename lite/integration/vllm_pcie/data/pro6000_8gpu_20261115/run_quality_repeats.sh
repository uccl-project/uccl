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
"$PYTHON" - "$R" <<'PY'
import json, sys
from pathlib import Path
if sys.flags.optimize:
    raise RuntimeError('Optimized Python disables required quality preflight assertions')
r = Path(sys.argv[1])
assert json.loads((r / 'final-original/COMPLETE.json').read_text())['complete']
q = json.loads((r / 'final-validation/quality/gsm8k-pcieccl.json').read_text())
original = json.loads((r / 'reproduction/quality/gsm8k-pcieccl.json').read_text())
assert round(original['accuracy'] * 1319) == 1222
assert q['invalid_rate'] == 0 and round(q['accuracy'] * 1319) < 1222
assert not (r / 'quality-repeats').exists(), 'Refusing prior repeat directory'
PY
restore_best() {
  repeat_status=$?
  if (( repeat_status != 0 )); then
    "$PYTHON" "$R/controller/snapshot.py" restore c03 || true
  fi
  exit "$repeat_status"
}
trap restore_best EXIT
repeat_index=0
for repeat_snapshot in initial c03 initial c03; do
  repeat_index=$((repeat_index + 1))
  "$PYTHON" "$R/controller/snapshot.py" restore "$repeat_snapshot"
  "$PYTHON" -u "$R/controller/quality_repeat.py" "$repeat_snapshot" "$R/quality-repeats/repeat-$repeat_index-$repeat_snapshot"
done
"$PYTHON" "$R/controller/snapshot.py" restore c03
"$PYTHON" - "$R" <<'PY'
import json, sys, time
from pathlib import Path
r = Path(sys.argv[1]) / 'quality-repeats'
(r / 'COMPLETE.json').write_text(json.dumps({'complete': True, 'order': ['initial', 'c03', 'initial', 'c03'], 'finished': time.time()}) + '\n')
PY
