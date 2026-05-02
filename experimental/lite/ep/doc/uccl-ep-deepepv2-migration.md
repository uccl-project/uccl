# UCCL-EP to DeepEPv2 Lite migration

This document records the current UCCL-EP transport integration in
`experimental/lite/ep`, which is the DeepEPv2 Lite `ElasticBuffer` codebase.
The active EP data path is UCCL-EP through `EP_USE_UCCL_PROXY=1`; NCCL remains
available for communicator/bootstrap and DeepEP interface compatibility, but
NCCL GIN is not used for EP data movement.

## Scope and target mode

Target validation environment:

- 2 nodes x 1 GPU: `l40` rank 0 and `l41` rank 1
- NVIDIA L4 / `sm_89`, GPU3 on each node
- no NVLink: `EP_FORCE_NO_NVLINK=1`
- DeepEPv2 built-in benchmark:
  `tests/elastic/test_ep.py --test-first-only`
- target shape: `128 tok x 7168 hid x top-8 x 64 exp`
- BF16 path: `EP_TEST_DISABLE_FP8=1`

Two UCCL-EP modes are supported:

| Mode | Key env | Window type |
| --- | --- | --- |
| GDR on | `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0`, `NCCL_NET_GDR_LEVEL=5` | GPU RDMA windows for single-local-rank runs; shared host windows for no-NVLink multi-GPU nodes |
| no-GDR | `UCCL_FORCE_NO_GDR=1`, `EP_FORCE_HOST_WINDOW=1`, `NCCL_NET_GDR_LEVEL=0` | host-pinned CUDA-mapped windows |

Common runtime environment:

```bash
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

## Main migration changes

### 1. Imported and adapted the UCCL proxy runtime

The UCCL-EP runtime was added under `csrc/uccl/` and wired into the DeepEPv2
extension build.

Main files:

- `csrc/uccl/include/*`
- `csrc/uccl/src/{common.cpp,proxy.cpp,rdma.cpp,uccl_proxy.cpp}`
- `setup.py`
- `csrc/jit/compiler.hpp`

What changed:

- Added D2H ring queues, proxy context, RDMA setup, shared-buffer helpers, and
  proxy thread management.
- Built UCCL proxy sources into the DeepEPv2 Lite extension.
- Added JIT flags for `EP_USE_UCCL_PROXY` and a transport-version cache key.
- Current JIT-visible UCCL transport version is `51`.

### 2. Added UCCL proxy activation and metadata exchange to ElasticBuffer

DeepEPv2's `ElasticBuffer` now decides whether to activate UCCL proxy mode,
allocates the correct workspace windows, exchanges peer metadata, and starts
the CPU proxy threads.

Main files:

- `deep_ep/buffers/elastic.py`
- `csrc/elastic/buffer.hpp`
- `csrc/uccl/include/uccl_proxy.hpp`
- `csrc/uccl/src/uccl_proxy.cpp`

What changed:

- `EP_USE_UCCL_PROXY=1` requests UCCL-EP transport.
- `UCCL_FORCE_NO_GDR=1` also requests UCCL-EP and forces host-window mode.
- `EP_FORCE_HOST_WINDOW=0/1` selects GPU RDMA windows vs host-pinned mapped
  windows for single-local-rank runs when UCCL is active.
- With `EP_FORCE_NO_NVLINK=1` and `local_world_size > 1`, UCCL proxy defaults
  to the POSIX shared host window even when `UCCL_FORCE_NO_GDR=0`. This avoids
  slow and fragile same-node GPU-window traffic on L4/PCIe. Use
  `EP_UCCL_FORCE_GPU_WINDOW=1` only for diagnostics.
- Python bootstrap uses `all_gather_object` to exchange UCCL peer metadata.
- Runtime creates per-proxy D2H queues and exports their GPU-visible addresses
  to JIT kernels.
- `sync_uccl_peers()` starts all proxies and waits for readiness before kernels
  can enqueue commands.

### 3. Replaced EP data movement with UCCL D2H commands

DeepEPv2 device transport calls are routed through a UCCL-compatible handle when
`EP_USE_UCCL_PROXY` is compiled into the JIT kernel.

Main files:

- `deep_ep/include/deep_ep/common/handle.cuh`
- `csrc/uccl/include/uccl_device_runtime.cuh`
- `deep_ep/include/deep_ep/impls/dispatch.cuh`
- `deep_ep/include/deep_ep/impls/combine.cuh`
- `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
- `deep_ep/include/deep_ep/impls/hybrid_combine.cuh`
- `csrc/kernels/elastic/{dispatch.hpp,combine.hpp,barrier.hpp}`

What changed:

- Implemented UCCL-backed `put`, `put_value`, and `quiet/flush` behavior.
- GPU kernels post WRITE and PUT_VALUE commands into D2H rings instead of using
  NCCL GIN for EP payload movement.
- CPU proxies consume those commands and execute host memcpy for same-node peers
  or ibverbs RDMA writes for remote-node peers.
- Transport puts were made safe for DeepEPv2 divergent top-k branches; the code
  no longer assumes a full warp participates in every transport operation.
- Active deduplicated lanes can each post payload commands, which is required
  for `topk > 1`.

### 4. Added GDR/no-GDR workspace selection

The migration keeps one UCCL-EP data path while allowing two backing memory
modes.

Main files:

- `csrc/elastic/buffer.hpp`
- `csrc/uccl/src/rdma.cpp`
- `csrc/uccl/include/shared_buffer.hpp`
- `csrc/uccl/include/proxy_ctx.hpp`

What changed:

- GDR mode uses GPU buffers registered for RDMA when there is one local rank per
  node.
- Multi-GPU no-NVLink nodes use the shared host window by default in both
  GDR-enabled and no-GDR modes.
- no-GDR mode uses POSIX shared host-pinned windows mapped into CUDA.
- Same-node peers use shared-memory/host memcpy rather than RDMA loopback.
- Same-rank commands are handled locally instead of posting RDMA to self.
- Shared-memory IDs are unique per run, and non-creators wait for initialized
  sizes to avoid stale or partially initialized mappings.

### 5. Fixed DeepEPv2 barrier semantics on UCCL

DeepEPv2 uses multiple Gin barrier tags across dispatch, cached dispatch,
combine, and reduced combine. UCCL's Gin-compatible barrier storage therefore
must be tag-aware.

Main files:

- `deep_ep/include/deep_ep/common/layout.cuh`
- `deep_ep/include/deep_ep/common/comm.cuh`
- `csrc/elastic/buffer.hpp`

What changed:

- Added `kNumGinBarrierTags = 16`.
- Expanded Gin barrier signal/shadow memory from per-rank to
  `tag x rank`.
- `get_gin_barrier_signal_ptr(rank, tag)` now maps each tag to its own signal
  slot.
- UCCL barrier waits use the matching tag/rank shadow slot.

Without this fix, different DeepEP phases advanced the same per-rank counter
and could hang with target drift in cached dispatch or combine.

### 6. Fixed UCCL proxy startup ordering

GDR mode exposed a race where the first kernel could enqueue D2H commands before
all proxy threads had completed QP/ACK setup. Debug logging slowed startup
enough to hide the race.

Main files:

- `csrc/uccl/include/proxy.hpp`
- `csrc/uccl/include/uccl_proxy.hpp`
- `csrc/uccl/src/proxy.cpp`
- `csrc/uccl/src/uccl_proxy.cpp`
- `csrc/elastic/buffer.hpp`

What changed:

- Added a per-proxy readiness flag.
- Proxy threads mark ready only after initialization needed to consume commands
  is complete.
- `UcclProxy::wait_until_ready()` waits with a deadline and reports startup
  failures explicitly.
- `ElasticBuffer::sync_uccl_peers()` waits for all proxies after `start_dual()`.

### 7. Preserved DeepEPv2 built-in benchmark coverage

The migration is validated against DeepEPv2's own test program rather than only
custom mini-benchmarks.

Main files:

- `tests/elastic/test_ep.py`
- `run_multinode.sh`

What changed:

- Added environment controls useful for UCCL validation, including benchmark
  iteration control and targeted path filters.
- Final validation does not require `EP_TEST_BASIC_ONLY`.
- The full first test case covers dispatch, expanded dispatch, cached dispatch,
  combine, and reduced combine.

### 8. Optimized UCCL single-scaleup combine

For 2n x 1g UCCL runs, `kNumScaleupRanks == 1`. The generic hybrid combine
path still wrote each token into a scale-up buffer, then had forward warps copy
it into scale-out send/recv buffers. In no-GDR mode that also made the GPU read
host-window memory back over PCIe. An initial no-GDR-only version made no-GDR
ordinary combine look faster than GDR because GDR was still on the old two-stage
path. The direct single-scaleup path is now compiled for both UCCL GDR and
no-GDR modes so the comparison is fair.

Main files:

- `deep_ep/include/deep_ep/impls/hybrid_combine.cuh`
- `csrc/jit/compiler.hpp`

What changed:

- For `kNumScaleupRanks == 1 && !kAllowMultipleReduction`, scale-up warps write
  directly into the scale-out send/recv buffer and issue the UCCL put to the
  source scaleout rank.
- Forward warps return early in this specialized path, eliminating the
  scaleup-buffer-to-forward-buffer copy.
- Final scale-out completion signals are issued by one elected lane after all
  payload puts for the channel, preserving FIFO ordering in the UCCL D2H ring.
- The path is gated by `EP_USE_UCCL_PROXY`, not by no-GDR mode.

### 9. Fixed no-NVLink multi-GPU scaling

2n x 2g and 2n x 4g GDR-enabled runs originally reached dispatch but hung in
ordinary combine. The failing path used GPU RDMA windows for same-node peers and
then tried to complete same-node `WRITE`/`PUT_VALUE` commands through CUDA IPC
while all ranks were inside cooperative DeepEP kernels. On L4/PCIe this path is
both slower than host memcpy and not a reliable completion mechanism for combine
signals.

Main files:

- `csrc/elastic/buffer.hpp`
- `csrc/uccl/src/proxy.cpp`
- `csrc/uccl/src/uccl_proxy.cpp`
- `csrc/uccl/include/proxy.hpp`
- `csrc/uccl/include/shared_buffer.hpp`

What changed:

- UCCL/no-NVLink runs with more than one local rank now prefer the POSIX shared
  host window by default, even when `UCCL_FORCE_NO_GDR=0`.
- Same-node QPs are skipped whenever shared-memory or CUDA-IPC intra-node
  transport is selected, avoiding RDMA loopback setup for local peers.
- CUDA current device is captured when the proxy is constructed and restored in
  proxy threads before CUDA IPC operations.
- CUDA IPC handles are cached per process/device/handle to avoid repeated
  `cudaIpcOpenMemHandle` calls from every proxy thread. This path is retained
  for `EP_UCCL_FORCE_GPU_WINDOW=1` diagnostics, but it is not the default
  robust path for L4/no-NVLink scaling.
- Shared-window sizing was reduced to the actual channel/token bound for the
  selected SM/channel count, and the benchmark now passes the requested
  `num_topk` into `ElasticBuffer`.

## Current benchmark results

Command shape:

```bash
EP_BENCH_NUM_TESTS=5 timeout 15s \
  /home/yangz/nfs/miniconda3/envs/uccl/bin/python \
  tests/elastic/test_ep.py \
  --num-processes 1 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check --num-gpu-timeout-secs=3
```

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

Summary:

- GDR mode reaches the expected 20+ GB/s legacy bandwidth for dispatch and
  ordinary combine on 2n x 1g.
- no-GDR dispatch, expanded dispatch, and cached dispatch reach 12-13 GB/s
  legacy bandwidth.
- UCCL ordinary combine now bypasses the generic two-stage forwarding path for
  single-scaleup runs in both GDR and no-GDR mode. The earlier no-GDR-only
  `29 GB/s legacy` result was misleading because GDR was still on the old path;
  with the same direct path, GDR ordinary combine is about `34 GB/s legacy`
  (`8 GB/s` SU/SO) and no-GDR is about `29-30 GB/s legacy` (`7 GB/s` SU/SO).
- no-GDR reduced combine improves to about 6 GB/s legacy bandwidth, but remains
  below ordinary combine due to expanded-layout traffic.

## Additional scale benchmark results

Same target shape (`128 tok x 7168 hid x top-8 x 64 exp`) and built-in
`tests/elastic/test_ep.py --test-first-only --skip-check`.

### 1 node x 2 GPUs

Environment: `l40`, `CUDA_VISIBLE_DEVICES=0,1`, `--num-processes=2`,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Rank | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDR on | r0 | 3 / 11 GB/s @ 1306.000 us | 3 / 11 GB/s @ 1312.000 us | 3 / 11 GB/s @ 1304.000 us | 3 / 14 GB/s @ 1064.000 us | 4 / 4 GB/s @ 3598.000 us |
| GDR on | r1 | 3 / 11 GB/s @ 1308.000 us | 3 / 11 GB/s @ 1352.000 us | 3 / 11 GB/s @ 1314.000 us | 3 / 13 GB/s @ 1101.000 us | 4 / 4 GB/s @ 3653.000 us |
| no-GDR | r0 | 3 / 12 GB/s @ 1244.000 us | 3 / 12 GB/s @ 1248.000 us | 3 / 12 GB/s @ 1240.000 us | 7 / 29 GB/s @ 501.961 us | 6 / 6 GB/s @ 2507.000 us |
| no-GDR | r1 | 3 / 11 GB/s @ 1303.000 us | 3 / 11 GB/s @ 1288.000 us | 3 / 12 GB/s @ 1266.000 us | 7 / 27 GB/s @ 538.794 us | 6 / 6 GB/s @ 2555.000 us |

### 1 node x 4 GPUs

Environment: `l40`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `--num-processes=4`.

| Mode | Iterations | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDR on | 1 | 3 / 6 GB/s @ 2268-2360 us | not rerun | not rerun | 6-7 / 12-15 GB/s @ 985-1160 us | basic-only |
| no-GDR | 1 | 3 / 6 GB/s @ 2262-2431 us | not rerun | not rerun | 6 / 13-14 GB/s @ 1023-1091 us | basic-only |

The slow 1n x 4g GDR path was traced to same-node peers using UCCL proxy RDMA
loopback against GPU memory. In UCCL proxy mode, single-node multi-GPU runs now
prefer the existing POSIX shared host window unless `EP_UCCL_FORCE_GPU_WINDOW=1`
is set, so GDR-enabled and no-GDR runs use the same intra-node path.

### 2 nodes x 4 GPUs

Environment: `l40` + `l41`, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`--num-processes=4` per node.

The previous blocked case is fixed for the basic BF16 built-in benchmark. With
`UCCL_FORCE_NO_GDR=0` and the new default shared host window for no-NVLink
multi-local-rank runs, all ranks complete dispatch and combine:

| Mode | Rank group | Dispatch | Combine |
| --- | --- | ---: | ---: |
| GDR-enabled default | l40/r0-r3 | 2-3 / 3-4 GB/s @ 4033-4560 us | 5 / 8 GB/s @ 1829-1890 us |
| GDR-enabled default | l41/r4-r7 | 2 / 3 GB/s @ 4394-4924 us | 5 / 7-8 GB/s @ 1884-1893 us |

The run remains basic-only/BF16 for communication debugging:
`EP_TEST_BASIC_ONLY=1`, `EP_TEST_DISABLE_FP8=1`,
`EP_TEST_DO_HANDLE_COPY=0`, `EP_BENCH_NUM_TESTS=1`. Full FP8 can still stall in
PyTorch `@torch.compile` preprocessing before UCCL communication.

## Known limitation and next performance work

Ordinary no-GDR combine exceeds the 10+ GB/s legacy target, but its attributed
SU/SO bandwidth is about 7 GB/s. The remaining performance gap is reduced
combine:

- Reduced combine is still about 6 GB/s legacy bandwidth in the 2n x 1g target
  shape.
- It moves expanded-layout data and still performs more host-window traffic than
  ordinary combine.
- Increasing proxy threads from 4 to 8 was previously unstable and should not be
  retried as a simple fix.

Further reduced-combine work should focus on expanded-layout traffic and
host-window/proxy pipeline overlap.

## Validation checklist

After UCCL transport, JIT-visible headers, or barrier layout changes:

```bash
conda activate uccl
make -j SM=89 PYTHON=/home/yangz/nfs/miniconda3/envs/uccl/bin/python
```

Then rerun the built-in 2n x 1g benchmark in both modes:

- GDR on: `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0`,
  `NCCL_NET_GDR_LEVEL=5`
- no-GDR: `UCCL_FORCE_NO_GDR=1`, `EP_FORCE_HOST_WINDOW=1`,
  `NCCL_NET_GDR_LEVEL=0`

For no-NVLink multi-GPU-per-node validation, do not set
`EP_UCCL_FORCE_GPU_WINDOW=1` unless specifically debugging the experimental
same-node GPU-window path.

Keep test-script timeouts at 15 seconds or less. If JIT compilation consumes
the timeout after a transport-version bump, rerun with the same
`EP_JIT_CACHE_DIR` after warmup.
