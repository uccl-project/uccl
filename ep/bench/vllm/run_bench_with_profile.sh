#!/bin/bash
# Run vllm bench serve with PyTorch profiler capture: start_profile before bench, stop_profile after.
# Prerequisite: vLLM server already running with torch profiler (e.g. launch_vllm_head.sh).
#
# Usage: ./run_bench_with_profile.sh
# Override server URL: VLLM_BENCH_URL=http://HOST:PORT ./run_bench_with_profile.sh
#
# Tuned for profiling: fewer prompts and lower concurrency so the trace is
# short and readable. For throughput benchmarking, increase num-prompts and max-concurrency.

set -e

BASE_URL="${VLLM_BENCH_URL:-http://127.0.0.1:8000}"

BENCH_ARGS=(
  --backend openai-chat
  --host 127.0.0.1
  --port 8000
  --endpoint /v1/chat/completions
  --model deepseek-ai/DeepSeek-V3-0324
  --dataset-name random
  --random-input-len 1024
  --random-output-len 256
  --num-prompts 10
  --request-rate 1
  --max-concurrency 32
  --seed 42
  --ignore-eos
  --save-result
  --result-dir ./results
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,95,99
)

echo "Profiler: start capture at $BASE_URL"
curl -s -X POST "$BASE_URL/start_profile" || { echo "Warning: start_profile failed (is server up with profiler enabled?)"; }

echo "Running: vllm bench serve ${BENCH_ARGS[*]}"
vllm bench serve "${BENCH_ARGS[@]}"
ret=$?

echo "Profiler: stop capture"
curl -s -X POST "$BASE_URL/stop_profile" || true

echo "Traces written to VLLM_PROFILER_DIR (default: \$HOME/efs/ziming/uccl/ep/bench/vllm). Open at https://ui.perfetto.dev/"
exit $ret
