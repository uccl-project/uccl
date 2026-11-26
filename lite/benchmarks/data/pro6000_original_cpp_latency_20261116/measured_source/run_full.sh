#!/usr/bin/env bash
# Full correctness/performance sweep. This is intentionally not run by setup.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p results

./bench/mscclpp_allreduce \
  -b 128 -e 1073741824 -n 20 -w 10 -m all -c 1 \
  | tee results/mscclpp_allreduce_sm120.txt

awk '
  $1 ~ /^[0-9]+$/ {
    rows++
    sizes[$1] = 1
    modes[$3]++
    if ($7 != "0.0e+00") {
      printf "PCIeCCL correctness failure: %s\n", $0 > "/dev/stderr"
      bad = 1
    }
    if ($1 == 1073741824) saw_max = 1
  }
  END {
    if (rows != 81 || length(sizes) != 24 || modes["ll"] != 9 ||
        modes["put"] != 24 || modes["ce"] != 24 || modes["auto"] != 24 ||
        !saw_max) {
      print "PCIeCCL sweep is incomplete" > "/dev/stderr"
      exit 1
    }
    exit bad
  }
' results/mscclpp_allreduce_sm120.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 all_reduce_perf \
  -b 128 -e 1073741824 -f 2 -g 8 -c 1 -n 20 -w 10 \
  | tee results/nccl_allreduce_sm120.txt

awk '
  $1 ~ /^[0-9]+$/ && $3 == "float" {
    rows++
    sizes[$1] = 1
    if ($9 != 0 || $13 != 0) {
      printf "NCCL correctness failure: %s\n", $0 > "/dev/stderr"
      bad = 1
    }
    if ($1 == 1073741824) saw_max = 1
  }
  END {
    if (rows != 24 || length(sizes) != 24 || !saw_max) {
      print "NCCL sweep is incomplete" > "/dev/stderr"
      exit 1
    }
    exit bad
  }
' results/nccl_allreduce_sm120.txt

echo "full PCIeCCL and NCCL correctness gates passed"
