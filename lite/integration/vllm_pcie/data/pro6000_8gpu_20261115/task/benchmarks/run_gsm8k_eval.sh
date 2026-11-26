#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {nccl|pcieccl} PORT" >&2
  exit 2
fi

backend="$1"
port="$2"
if [[ "$backend" != "nccl" && "$backend" != "pcieccl" ]]; then
  echo "backend must be nccl or pcieccl" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]]; then
  echo "port must be numeric" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p results/quality

uv run --project environments/vllm-pcieccl --no-sync \
  python vllm-0.26.0/tests/evals/gsm8k/gsm8k_eval.py \
  --num-questions 1319 \
  --num-shots 5 \
  --max-tokens 256 \
  --temperature 0 \
  --seed 42 \
  --max-concurrency 64 \
  --port "$port" \
  --save-results "results/quality/gsm8k-${backend}.json"
