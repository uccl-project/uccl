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
- GDR mode uses GPU RDMA windows for single-local-rank runs; in no-NVLink
  multi-GPU-per-node runs it defaults to the shared host window to avoid slow
  same-node PCIe GPU-window traffic
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

For `local_world_size > 1` with `EP_FORCE_NO_NVLINK=1`, UCCL proxy still
defaults to the POSIX shared host window even in GDR-enabled mode. Use
`EP_UCCL_FORCE_GPU_WINDOW=1` only for diagnostics; the same-node GPU-window path
is not the robust L4/PCIe fast path.

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

Cells are `DeepEPv2 SU BW / legacy BW @ latency`. The SU/SO bandwidth is based
on the bytes that the benchmark attributes to scale-up/scale-out traffic. Legacy
BW uses the old DeepEP low-latency numerator (`valid_topk * hidden * 2` for
BF16), so ordinary combine legacy bandwidth is about 4x the attributed SU/SO
traffic for this shape and must not be read as physical link bandwidth.

| Mode | Rank | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDR on | l40/r0 | 6 / 22 GB/s @ 665.775 us | 6 / 23 GB/s @ 647.869 us | 6 / 24 GB/s @ 621.661 us | 8 / 34 GB/s @ 433.005 us | 7 / 7 GB/s @ 2195.000 us |
| GDR on | l41/r1 | 6 / 23 GB/s @ 640.006 us | 6 / 23 GB/s @ 640.288 us | 6 / 24 GB/s @ 618.087 us | 8 / 34 GB/s @ 431.433 us | 7 / 7 GB/s @ 2193.000 us |
| no-GDR | l40/r0 | 3 / 12 GB/s @ 1177.000 us | 3 / 12 GB/s @ 1187.000 us | 3 / 13 GB/s @ 1160.000 us | 7 / 30 GB/s @ 495.903 us | 6 / 6 GB/s @ 2494.000 us |
| no-GDR | l41/r1 | 3 / 12 GB/s @ 1266.000 us | 3 / 12 GB/s @ 1258.000 us | 3 / 12 GB/s @ 1186.000 us | 7 / 29 GB/s @ 505.365 us | 6 / 6 GB/s @ 2498.000 us |

Interpretation:

- GDR mode now reaches the expected 20+ GB/s legacy bandwidth for dispatch and
  ordinary combine on 2n x 1g.
- no-GDR dispatch, expanded dispatch, and cached dispatch reach 12-13 GB/s
  legacy bandwidth.
- UCCL ordinary combine now bypasses the old two-stage forwarding path for 2n x
  1g single-scaleup runs in both GDR and no-GDR mode. The earlier no-GDR-only
  `29 GB/s legacy` result was misleading because GDR was still on the old path;
  with the same direct path, GDR ordinary combine is about `34 GB/s legacy`
  (`8 GB/s` SU/SO) and no-GDR is about `29-30 GB/s legacy` (`7 GB/s` SU/SO).
- no-GDR reduced combine also improves, but remains lower at about 6 GB/s legacy
  bandwidth because it moves much more expanded-layout data.

## Additional scale benchmark data

Same target shape (`128 tok x 7168 hid x top-8 x 64 exp`) and built-in
`tests/elastic/test_ep.py --test-first-only --skip-check`.

1 node x 2 GPUs on `l40`, `CUDA_VISIBLE_DEVICES=0,1`, `--num-processes=2`,
`EP_BENCH_NUM_TESTS=5`:

| Mode | Rank | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDR on | r0 | 3 / 11 GB/s @ 1306.000 us | 3 / 11 GB/s @ 1312.000 us | 3 / 11 GB/s @ 1304.000 us | 3 / 14 GB/s @ 1064.000 us | 4 / 4 GB/s @ 3598.000 us |
| GDR on | r1 | 3 / 11 GB/s @ 1308.000 us | 3 / 11 GB/s @ 1352.000 us | 3 / 11 GB/s @ 1314.000 us | 3 / 13 GB/s @ 1101.000 us | 4 / 4 GB/s @ 3653.000 us |
| no-GDR | r0 | 3 / 12 GB/s @ 1244.000 us | 3 / 12 GB/s @ 1248.000 us | 3 / 12 GB/s @ 1240.000 us | 7 / 29 GB/s @ 501.961 us | 6 / 6 GB/s @ 2507.000 us |
| no-GDR | r1 | 3 / 11 GB/s @ 1303.000 us | 3 / 11 GB/s @ 1288.000 us | 3 / 12 GB/s @ 1266.000 us | 7 / 27 GB/s @ 538.794 us | 6 / 6 GB/s @ 2555.000 us |

1 node x 4 GPUs on `l40`, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`--num-processes=4`:

| Mode | Iterations | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDR on | 1 | 3 / 6 GB/s @ 2268-2360 us | not rerun | not rerun | 6-7 / 12-15 GB/s @ 985-1160 us | basic-only |
| no-GDR | 1 | 3 / 6 GB/s @ 2262-2431 us | not rerun | not rerun | 6 / 13-14 GB/s @ 1023-1091 us | basic-only |

2 nodes x 4 GPUs (`l40` + `l41`, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`--num-processes=4` on each node) now completes the basic built-in
dispatch/combine benchmark in GDR-enabled UCCL mode. With the no-NVLink
multi-local-rank default shared host window, representative one-iteration
results are:

| Mode | Dispatch | Combine | Notes |
| --- | ---: | ---: | --- |
| GDR-enabled default | 2-3 / 3-4 GB/s @ 4033-4924 us | 5 / 7-8 GB/s @ 1829-1893 us | basic-only, BF16 |

The previous GPU-window path hung in combine for 2n x 2g and 2n x 4g because
same-node CUDA IPC signal writes were not a reliable completion path while all
ranks were inside cooperative kernels. The robust default is now shared host
window for any no-NVLink multi-GPU node; `EP_UCCL_FORCE_GPU_WINDOW=1` remains a
diagnostic override.

## UCCL-EP correctness notes

- UCCL Gin-compatible barrier signal/shadow storage must be indexed by barrier
  tag and rank. Reusing one counter per rank causes different DeepEP phases to
  advance the same counter and can hang cached dispatch or combine.
- `ElasticBuffer::sync_uccl_peers()` must wait until every UCCL proxy reports
  ready after `start_dual()`. Without this, GDR kernels can enqueue commands
  before all QPs and ACK paths are ready; debug logging can mask the race.
- In UCCL `kNumScaleupRanks == 1` combine, avoid the generic hybrid two-stage
  path (`scaleup_buffer -> forward warp -> scaleout buffer`). Directly write
  from the scaleup warp into the scaleout send/recv buffer and issue UCCL puts;
  otherwise the no-GDR path performs an extra GPU PCIe read from host-window
  memory and the GDR/no-GDR benchmark comparison is unfair.
- Direct combine completion signals must be posted by one elected lane after
  payload puts so the UCCL D2H ring preserves data-before-signal ordering.
- In UCCL/no-NVLink mode with more than one local rank, prefer the shared host
  window even when `UCCL_FORCE_NO_GDR=0`. Same-node GPU-window traffic over PCIe
  is slower and less robust than host memcpy on L4.
- Bump `EP_UCCL_PROXY_TRANSPORT_VERSION` whenever JIT-visible device headers,
  barrier layout, or transport semantics change.

## Modification guidelines

- Keep UCCL transport changes scoped to `experimental/lite/ep`.
- Do not modify top-level `ep/` for DeepEPv2 work; it is legacy DeepEP v1
  UCCL-EP reference code.
- Keep the UCCL path gated by `EP_USE_UCCL_PROXY=1`; use
  `UCCL_FORCE_NO_GDR=0/1` and `EP_FORCE_HOST_WINDOW=0/1` to select GDR vs
  no-GDR for single-local-rank runs. Multi-local-rank no-NVLink runs default to
  shared host windows unless `EP_UCCL_FORCE_GPU_WINDOW=1` is explicitly set.
- Do not re-enable NVLink/LSA peer bypass, multiple QPs, or multiple-reduction
  combine in this mode without rerunning correctness and performance on
  l40+l41.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
