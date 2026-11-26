#!/usr/bin/env bash
set -euo pipefail
# Fresh calibration and five independent validation rounds. Probe/profile times
# never select the final per-size policy. The Python runner owns the GPU lock.
cd "$(dirname "$0")/.."
uccl_lite_output=${1:-build/pro6000-large-run}
uccl_lite_env=${2:-/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910/runtime_env.sh}
if [[ -e "$uccl_lite_output" ]]; then
  echo "Refusing to overwrite $uccl_lite_output" >&2
  exit 2
fi
case "$uccl_lite_output" in
  build/*) ;;
  *) echo "Output must be under build/" >&2; exit 2 ;;
esac
bash scripts/build_pro6000_allreduce.sh "$uccl_lite_env"
mkdir -p "$uccl_lite_output"
uccl_lite_experiment=$(cd "$(dirname "$uccl_lite_env")" && pwd -P)
uccl_lite_nccl=$(python3 -B -c 'import sys; sys.path.insert(0,"scripts"); from run_pro6000_allreduce import NCCL; print(",".join(NCCL))')
uccl_lite_ours=ll-1-256,ll-1-1024,ll-4-256,ll-4-1024,ll-16-256,ll-16-1024,ll-32-256,ll-32-1024
uccl_lite_ours+=,put-4,put-16,put-64,drained-ce-128,staged-ce-32,staged-ce-64,staged-ce-128
uccl_lite_ours+=,ring-ce-1-256m-t1024,ring-ce-1-1024m-w64-t512,ring-ce-2-32m,ring-ce-2-32m-t1024,ring-ce-2-64m-t512,ring-ce-2-256m
uccl_lite_ours+=,ring-host-ce-1-1024m-w128-t1024,ring-host-ce-2-256m,ring-host-ce-4-128m
uccl_lite_ours+=,ring-ce-2-32m-s2,ring-ce-2-32m-t512-s2,ring-ce-2-32m-t1024-s2,ring-ce-2-64m-t512-s2,ring-ce-2-256m-s2,ring-ce-4-32m-s2
uccl_lite_ours+=,ring-ce-2-8m-t512-s2,ring-ce-2-8m-t1024-s2,ring-ce-2-16m-t512-s2,ring-ce-2-16m-t1024-s2
python3 -B scripts/run_pro6000_allreduce.py calibrate \
  --output "$uccl_lite_output/calibration" --experiment "$uccl_lite_experiment" \
  --backend both --configs "$uccl_lite_nccl,$uccl_lite_ours"
python3 -B scripts/run_pro6000_allreduce.py freeze \
  --output "$uccl_lite_output/validation" --experiment "$uccl_lite_experiment" \
  --calibration "$uccl_lite_output/calibration"
python3 -B scripts/run_pro6000_allreduce.py validate \
  --output "$uccl_lite_output/validation" --experiment "$uccl_lite_experiment"
