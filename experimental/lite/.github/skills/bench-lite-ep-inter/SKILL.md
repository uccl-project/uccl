---
name: bench-lite-ep-inter
description: "Run the DeepEP-Lite multi-node/inter-node benchmark across l40 and l41 in experimental/lite/ep. Use when benchmarking lite EP, inter-node, multi-node, RDMA, or run_multinode; accepts an optional GPU count prompt and defaults to 2 total GPUs."
argument-hint: "[GPU count, default: 2 total GPUs]"
---

# DeepEP-Lite Inter-node Benchmark

Run the multi-node benchmark from `ep/AGENTS.md` across `l40` and `l41`.

## Inputs

- Use the user's extra prompt text to infer the GPU count.
- If the prompt says `N GPUs per node`, set `GPUS_PER_NODE=N`.
- If the prompt only says `N GPUs`, treat `N` as the total GPU count across
  `l40` and `l41`, and set `GPUS_PER_NODE=N/2`.
- If no GPU count is specified, use `TOTAL_GPUS=2`, so `GPUS_PER_NODE=1`.
- The total GPU count must be even because this workflow uses the same GPU
  count on both nodes.
- `--num-experts=64` must be divisible by the total GPU count. If either
  constraint is not met, ask the user for another GPU count before running.

## Procedure

1. Run this workflow from `l40` in
   `/home/yangz/nfs/zhongjie/uccl/experimental/lite/ep`.
2. Activate the environment for the node before rebuilding:

```bash
# l40
conda activate uccl

# l41, before any manual rebuild there
source ~/zhongjie/zj_py/bin/activate
```

3. On `l40`, rebuild and install:

```bash
make -j SM=89
make -j install
```

4. Run the benchmark, replacing `GPUS_PER_NODE=1` only when the user requested
   another valid GPU count:

```bash
GPUS_PER_NODE=1
bash run_multinode.sh --gpus-per-node "$GPUS_PER_NODE" \
  --test-args "--num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64"
```
