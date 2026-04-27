---
name: bench-lite-ep-intra
description: "Run the DeepEP-Lite single-node/intra-node benchmark in experimental/lite/ep. Use when benchmarking lite EP, intra-node, single-node, torchrun, test_low_latency, or L4 GPUs; accepts an optional GPU count prompt and defaults to 2 GPUs."
argument-hint: "[GPU count, default: 2]"
---

# DeepEP-Lite Intra-node Benchmark

Run the single-node benchmark from `ep/AGENTS.md`.

## Inputs

- Use the user's extra prompt text to infer the GPU count, for example `4 GPUs`
  or `use 1 GPU`.
- If no GPU count is specified, use `GPU_COUNT=2`.
- `--num-experts=64` must be divisible by `GPU_COUNT`. If it is not, ask the
  user for another GPU count before running.

## Procedure

1. Work from `/home/yangz/nfs/zhongjie/uccl/experimental/lite/ep`.
2. Activate the environment for the current node before rebuilding:

```bash
# l40
conda activate uccl

# l41
source ~/zhongjie/zj_py/bin/activate
```

3. Rebuild and install:

```bash
make -j SM=89
make -j install
```

4. Run the benchmark, replacing `GPU_COUNT=2` only when the user requested a
   different GPU count:

```bash
GPU_COUNT=2
GPU_LIST="$(seq -s, 0 $((GPU_COUNT - 1)))"
CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --nproc_per_node="$GPU_COUNT" \
  bench/test_low_latency.py --num-tokens=128 --hidden=7168 \
  --num-topk=8 --num-experts=64 --disable-nvlink
```
