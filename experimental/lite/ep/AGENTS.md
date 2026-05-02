# DeepEP-Lite

`experimental/lite/ep` is the DeepEPv2 Lite `ElasticBuffer` codebase. In this
branch the active EP data transport is the UCCL-EP CPU proxy path
(`EP_USE_UCCL_PROXY=1`) for both GPUDirect RDMA and no-GPUDirect-RDMA runs.
NCCL may still be loaded for communicator/bootstrap and DeepEP interface
compatibility, but NCCL GIN is not the EP data path.

Detailed migration notes are in `doc/uccl-ep-deepepv2-migration.md`.

Primary goal:

- Run Lite-EP, meaning UCCL-EP integrated with DeepEPv2, efficiently on
  consumer or low-end GPU systems where NVLink and GPUDirect RDMA are not
  available.
- Treat no-NVLink/no-GDR as the target path, not a fallback. GDR-enabled runs are
  useful for comparison and diagnostics, but correctness and performance work
  should optimize the host-staged UCCL proxy path.
- Keep inter-node payload movement staged through host memory and same-node
  no-NVLink multi-GPU traffic on the POSIX shared host window unless a future
  change proves a robust faster path.

Target environment:

- Consumer/no-NVLink/no-GDR GPUs first; NVIDIA L4 / `sm_89` is the current
  validation proxy
- no NVLink
- DeepEPv2 dispatch/combine kernels
- UCCL CPU proxy over ibverbs; same-node peers use host memcpy, remote-node
  peers use RDMA writes
- GDR mode uses GPU RDMA windows for single-local-rank runs; in no-NVLink
  multi-GPU-per-node runs it defaults to the shared host window to avoid slow
  same-node PCIe GPU-window traffic
- no-GDR mode uses host-pinned CUDA-mapped windows

## Runtime modes

Common environment template for local validation. Keep site-local absolute paths
in the shell environment or session notes, not in committed docs.

```bash
EP_DIR=$(pwd)
PYTHONPATH=$EP_DIR
PYTHON_BIN=${PYTHON_BIN:-python}
EP_USE_UCCL_PROXY=1
EP_FORCE_NO_NVLINK=1
NCCL_GIN_TYPE=2
DISABLE_SM90_FEATURES=1
EP_SUPPRESS_NCCL_CHECK=1
EP_TEST_DISABLE_FP8=1
EP_FORCE_PROCESS_EXIT=1
UCCL_IB_HCA=${UCCL_IB_HCA:-<ib-device>}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<gpu-list>}
EP_NCCL_ROOT_DIR=${EP_NCCL_ROOT_DIR:-<local-nccl-root>}
EP_TORCH_NVSHMEM_STUB=${EP_TORCH_NVSHMEM_STUB:-}
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2${EP_TORCH_NVSHMEM_STUB:+:$EP_TORCH_NVSHMEM_STUB}
LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:${CONDA_PREFIX:-<python-env>}/lib:${LD_LIBRARY_PATH:-}
EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-<local-jit-cache-dir>}
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
- Current scale tables use `EP_TEST_BASIC_ONLY=1` for BF16 dispatch/combine
  transport validation. Full first-test validation also runs without that filter
  and includes expanded dispatch, cached dispatch, and reduced combine.

## Build and test

Build from this directory:

```bash
conda activate uccl
make -j SM=89 PYTHON=$PYTHON_BIN
```

Use DeepEPv2's built-in benchmark for the current target shape:

```bash
EP_BENCH_NUM_TESTS=5 timeout 15s \
  $PYTHON_BIN \
  tests/elastic/test_ep.py \
  --num-processes 1 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check --num-gpu-timeout-secs=3
```

For 2-node x 1-GPU runs, launch rank 0 on `NODE0` and rank 1 on `NODE1` with
`WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, matching `MASTER_ADDR`/`MASTER_PORT`, and
`RANK=0` or `RANK=1`. The first run after a JIT-visible change may spend the
15-second timeout compiling; rerun with the same `EP_JIT_CACHE_DIR` after warmup.

## Current UCCL-EP benchmark data

All rows below use DeepEPv2's built-in benchmark with
`128 tok x 7168 hid x top-8 x 64 exp`, BF16 basic dispatch/combine only:
`EP_TEST_BASIC_ONLY=1`, `EP_TEST_DO_HANDLE_COPY=0`,
`EP_TEST_DISABLE_FP8=1`, `EP_FORCE_NO_NVLINK=1`,
`tests/elastic/test_ep.py --test-first-only --skip-check`.

Cells are rank averages in the form `SO/SU GB/s, legacy GB/s @ latency`. SO/SU
bandwidth uses the bytes that DeepEPv2 attributes to scale-out/scale-up traffic.
Legacy bandwidth uses the old low-latency numerator, so it is useful for
comparing runs with the same topology but is not physical link bandwidth.

### 1n x 2g

Single node, `CUDA_VISIBLE_DEVICES=0,1`, `--num-processes=2`,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 3/3 GB/s, 12.5 GB/s @ 1180 us | 7.5/7.5 GB/s, 30.0 GB/s @ 488 us |
| no-GDR | 3/3 GB/s, 12.0 GB/s @ 1238 us | 8.0/8.0 GB/s, 30.5 GB/s @ 482 us |

### 1n x 4g

Single node, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `--num-processes=4`,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 3/3 GB/s, 6.8 GB/s @ 2222 us | 6.3/6.3 GB/s, 14.0 GB/s @ 1031 us |
| no-GDR | 3/3 GB/s, 7.0 GB/s @ 2084 us | 6.3/6.3 GB/s, 14.3 GB/s @ 1015 us |

The 1n x 4g legacy dispatch number is lower than 1n x 2g because top-8 routing
touches about twice as many scaleout destinations per token when ranks increase
from 2 to 4. The actual attributed dispatch throughput stays at about 3 GB/s in
both cases. GDR and no-GDR are intentionally similar here because no-NVLink
multi-local-rank UCCL runs use the shared host window in both modes. Forcing the
diagnostic GPU-window path with `EP_UCCL_FORCE_GPU_WINDOW=1` times out for this
1n x 4g benchmark and remains unsupported as a fast path on L4/PCIe.

### 2n x 1g

Two nodes, `CUDA_VISIBLE_DEVICES=3`, `--num-processes=1` per node,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 6/6 GB/s, 23.0 GB/s @ 646 us | 9/9 GB/s, 35.0 GB/s @ 423 us |
| no-GDR | 3/3 GB/s, 12.5 GB/s @ 1179 us | 7/7 GB/s, 29.5 GB/s @ 498 us |

### 2n x 4g

Two nodes, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `--num-processes=4` per node,
`EP_BENCH_NUM_TESTS=1` to stay within the 15-second script timeout.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 2/2 GB/s, 3.1 GB/s @ 4345 us | 5.3/5.3 GB/s, 8.0 GB/s @ 1842 us |
| no-GDR | 2/2 GB/s, 3.0 GB/s @ 4698 us | 5.0/5.0 GB/s, 7.1 GB/s @ 1950 us |

The previous 2n x 4g blocked case is fixed for this BF16 basic built-in
benchmark. GDR-enabled runs can still exceed the 15-second wrapper during
teardown after all ranks print results; use the printed benchmark lines and keep
the wrapper timeout at 15 seconds.

## Current full-path validation

Full-path validation removes `EP_TEST_BASIC_ONLY` and uses
`do_handle_copy=1`, expanded dispatch, cached dispatch, ordinary combine, and
reduced combine:

```bash
timeout 15s $PYTHON_BIN \
  tests/elastic/test_ep.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check --skip-perf-test
```

Validated rows:

| Topology | GDR | no-GDR |
| --- | --- | --- |
| 1n x 2g | pass | pass |
| 1n x 4g | pass | pass |
| 2n x 1g | pass | pass |
| 2n x 4g | startup can exceed 15 s | pass |

The target consumer path is the no-GDR column. 2n x 4g no-GDR reaches `after
reduced combine` on all eight ranks. 2n x 4g GDR is still diagnostic-only for
this workstream; repeated 8-process startup can consume the 15-second wrapper
before the first test prints.

Additional no-GDR full-path 1n x 4g data with `EP_BENCH_NUM_TESTS=1`:

| Stage | Rank-average result |
| --- | ---: |
| Dispatch | ~3/3 GB/s SO/SU, ~6 GB/s legacy @ ~2.4 ms |
| Expanded dispatch | ~3/3 GB/s SO/SU, ~7 GB/s legacy @ ~2.1 ms |
| Cached dispatch | ~3/3 GB/s SO/SU, ~6 GB/s legacy @ ~2.3 ms |
| Combine | ~6 GB/s SO/SU, ~14 GB/s legacy @ ~1.0 ms |
| Reduced combine | ~4 GB/s SO/SU, ~4 GB/s legacy @ ~3.4 ms |

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
  the two-node validation setup.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
