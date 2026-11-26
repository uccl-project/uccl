#!/usr/bin/env bash
set -euo pipefail
# Use the existing isolated PRO6000 environment; never install system tools.
uccl_lite_env=${2:-/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910/runtime_env.sh}
source "$uccl_lite_env"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64"
cd "$(dirname "$0")/.."
uccl_lite_output=$(realpath -m "${1:?supply a new output directory}")
if [[ -e "$uccl_lite_output" ]]; then
  echo "Refusing to overwrite an existing experiment: $uccl_lite_output" >&2
  exit 2
fi
mkdir -p "$uccl_lite_output"
python3 -B scripts/prepare_pro6000_ag_rs.py --output "$uccl_lite_output/calibration-jobs.json"
python3 -B scripts/run_pro6000_ag_rs.py --jobs "$uccl_lite_output/calibration-jobs.json" --output "$uccl_lite_output/calibration"
python3 -B scripts/summarize_pro6000_ag_rs.py freeze --calibration "$uccl_lite_output/calibration" --output "$uccl_lite_output/policy.json"
python3 -B scripts/summarize_pro6000_ag_rs.py validation-jobs --policy "$uccl_lite_output/policy.json" --output "$uccl_lite_output/validation-jobs.json"
python3 -B scripts/run_pro6000_ag_rs.py --jobs "$uccl_lite_output/validation-jobs.json" --output "$uccl_lite_output/validation"
python3 -B scripts/summarize_pro6000_ag_rs.py summarize --calibration "$uccl_lite_output/calibration" --validation "$uccl_lite_output/validation" --policy "$uccl_lite_output/policy.json" --output "$uccl_lite_output/report"
