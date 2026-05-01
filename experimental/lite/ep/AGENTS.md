# DeepEP-Lite

`experimental/lite/ep` is the DeepEPv2 Lite `ElasticBuffer` codebase. In this
branch the active EP data transport is the UCCL-EP CPU proxy path
(`EP_USE_UCCL_PROXY=1`) for both GPUDirect RDMA and no-GPUDirect-RDMA runs.
NCCL may still be loaded for communicator/bootstrap and DeepEP interface
compatibility, but NCCL GIN is not the EP data path.

Detailed migration notes are in `doc/uccl-ep-deepepv2-migration.md`.

Target environment:

- NVIDIA L4 / `sm_89`
- no NVLink
- DeepEPv2 dispatch/combine kernels
- UCCL CPU proxy over ibverbs; same-node peers use host memcpy, remote-node
  peers use RDMA writes
- GDR mode uses GPU RDMA windows
- no-GDR mode uses host-pinned CUDA-mapped windows

## Runtime modes

Common environment for the current 2-node x 1-GPU L40/L41 setup:

```bash
EP_DIR=/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-upgrade-lite-ep-to-deepepv2/experimental/lite/ep
PYTHONPATH=$EP_DIR
EP_USE_UCCL_PROXY=1
EP_FORCE_NO_NVLINK=1
NCCL_GIN_TYPE=2
DISABLE_SM90_FEATURES=1
EP_SUPPRESS_NCCL_CHECK=1
EP_TEST_DISABLE_FP8=1
EP_FORCE_PROCESS_EXIT=1
UCCL_IB_HCA=mlx5_1
CUDA_VISIBLE_DEVICES=3
EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:/home/yangz/nfs/miniconda3/envs/uccl/lib:${LD_LIBRARY_PATH:-}
EP_JIT_CACHE_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/jit-cache
```

GDR-enabled mode:

```bash
UCCL_FORCE_NO_GDR=0
EP_FORCE_HOST_WINDOW=0
NCCL_NET_GDR_LEVEL=5
```

No-GDR mode:

```bash
UCCL_FORCE_NO_GDR=1
EP_FORCE_HOST_WINDOW=1
NCCL_NET_GDR_LEVEL=0
```

Notes:

- `NCCL_GIN_TYPE=2` keeps DeepEP on the Gin-compatible interface path; UCCL
  proxy still handles EP traffic when `EP_USE_UCCL_PROXY=1`.
- `EP_FORCE_NO_NVLINK=1` preserves the conservative `Ranks: 2 x 1` mode used
  for L4 validation.
- Do not use `EP_TEST_BASIC_ONLY` for final validation. The full built-in path
  must include dispatch, expanded dispatch, cached dispatch, combine, and
  reduced combine.

## Build and test

Build from this directory:

```bash
conda activate uccl
make -j SM=89 PYTHON=/home/yangz/nfs/miniconda3/envs/uccl/bin/python
```

Use DeepEPv2's built-in benchmark for the current target shape:

```bash
EP_BENCH_NUM_TESTS=5 timeout 15s \
  /home/yangz/nfs/miniconda3/envs/uccl/bin/python \
  tests/elastic/test_ep.py \
  --num-processes 1 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check --num-gpu-timeout-secs=3
```

For 2-node x 1-GPU runs, launch rank 1 on `l41` and rank 0 on `l40` with
`WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, matching `MASTER_ADDR`/`MASTER_PORT`, and
`RANK=1` or `RANK=0`. The first run after a JIT-visible change may spend the
15-second timeout compiling; rerun with the same `EP_JIT_CACHE_DIR` after warmup.

## Current UCCL-EP benchmark data

Shape: 2 nodes x 1 GPU, L4 GPU3 on each node, BF16
`128 tok x 7168 hid x top-8 x 64 exp`, `do_handle_copy=1`,
`expert_alignment=128`, no NVLink, `EP_BENCH_NUM_TESTS=5`, DeepEPv2
`tests/elastic/test_ep.py --test-first-only --skip-check`.

Cells are `DeepEPv2 SU BW / legacy BW @ latency`. Legacy BW uses the old
DeepEP low-latency numerator (`valid_topk * hidden * 2` for BF16).

| Mode | Rank | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDR on | l40/r0 | 6 / 23 GB/s @ 636.607 us | 6 / 23 GB/s @ 638.283 us | 6 / 24 GB/s @ 618.700 us | 5 / 21 GB/s @ 693.329 us | 4 / 4 GB/s @ 3714.000 us |
| GDR on | l41/r1 | 5 / 21 GB/s @ 684.153 us | 5 / 22 GB/s @ 667.457 us | 6 / 22 GB/s @ 654.573 us | 5 / 20 GB/s @ 724.166 us | 4 / 4 GB/s @ 3750.000 us |
| no-GDR | l40/r0 | 3 / 12 GB/s @ 1186.000 us | 3 / 12 GB/s @ 1191.000 us | 3 / 13 GB/s @ 1153.000 us | 1 / 5 GB/s @ 3107.000 us | 1 / 1 GB/s @ 16026.000 us |
| no-GDR | l41/r1 | 3 / 12 GB/s @ 1247.000 us | 3 / 12 GB/s @ 1237.000 us | 3 / 13 GB/s @ 1154.000 us | 1 / 5 GB/s @ 3111.000 us | 1 / 1 GB/s @ 16033.000 us |

Interpretation:

- GDR mode now reaches the expected 20+ GB/s legacy bandwidth for dispatch and
  combine on 2n x 1g.
- no-GDR dispatch, expanded dispatch, and cached dispatch reach 12-13 GB/s
  legacy bandwidth.
- no-GDR combine remains the open bottleneck at about 5 GB/s legacy bandwidth;
  reduced combine is about 1 GB/s. Lightweight attempts such as increasing
  proxy threads or direct-read shortcuts were unstable or too small to keep.
  Treat further no-GDR combine work as a larger host-window/proxy pipeline
  redesign.

## UCCL-EP correctness notes

- UCCL Gin-compatible barrier signal/shadow storage must be indexed by barrier
  tag and rank. Reusing one counter per rank causes different DeepEP phases to
  advance the same counter and can hang cached dispatch or combine.
- `ElasticBuffer::sync_uccl_peers()` must wait until every UCCL proxy reports
  ready after `start_dual()`. Without this, GDR kernels can enqueue commands
  before all QPs and ACK paths are ready; debug logging can mask the race.
- Bump `EP_UCCL_PROXY_TRANSPORT_VERSION` whenever JIT-visible device headers,
  barrier layout, or transport semantics change.

## Modification guidelines

- Keep UCCL transport changes scoped to `experimental/lite/ep`.
- Do not modify top-level `ep/` for DeepEPv2 work; it is legacy DeepEP v1
  UCCL-EP reference code.
- Keep the UCCL path gated by `EP_USE_UCCL_PROXY=1`; use
  `UCCL_FORCE_NO_GDR=0/1` and `EP_FORCE_HOST_WINDOW=0/1` to select GDR vs
  no-GDR.
- Do not re-enable NVLink/LSA peer bypass, multiple QPs, or multiple-reduction
  combine in this mode without rerunning correctness and performance on
  l40+l41.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
