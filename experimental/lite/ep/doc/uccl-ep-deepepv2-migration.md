# UCCL-EP to DeepEPv2 Lite migration

This document records the current UCCL-EP transport integration in
`experimental/lite/ep`, which is the DeepEPv2 Lite `ElasticBuffer` codebase.
The active EP data path is UCCL-EP through `EP_USE_UCCL_PROXY=1`; NCCL remains
available for communicator/bootstrap and DeepEP interface compatibility, but
NCCL GIN is not used for EP data movement.

## Scope and target mode

Target validation environment:

- 2 nodes x 1 GPU: `NODE0` rank 0 and `NODE1` rank 1
- NVIDIA L4 / `sm_89`, with the selected local GPU on each node
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

## Design notes

### Communication window

DeepEP kernels and the UCCL proxy communicate through a shared communication
workspace. This document calls that workspace a *window*, matching the NCCL
symmetric-window terminology used by `ncclCommWindowRegister(...)`.

In UCCL proxy mode, the window is the memory region that contains DeepEP
payload buffers, counters, and completion signals. GPU kernels write commands
and payload metadata relative to this window; CPU proxy threads interpret those
addresses and perform same-node memcpy or remote-node RDMA writes.

There are two backing choices:

| Window | Backing memory | Current use |
| --- | --- | --- |
| host window | CPU host memory, either `cudaHostAlloc` or POSIX shared memory registered with CUDA | no-GDR runs, and all no-NVLink multi-GPU-per-node UCCL runs |
| GPU window | `cudaMalloc` GPU memory registered for RDMA, with CUDA IPC available for same-node diagnostics | single-local-rank GDR runs; diagnostic only for multi-local-rank no-NVLink |

### Why multi-local-rank no-NVLink defaults to host window

On L4/PCIe without NVLink, same-node GPU-window traffic is not a reliable fast
path. Direct GPU-window operation either uses PCIe peer/CUDA IPC behavior for
same-node peers or GPU RDMA-loopback style access, both of which were slower and
less robust than CPU proxy memcpy through shared host memory during 1n x 4g,
2n x 2g, and 2n x 4g validation.

Therefore, when `EP_USE_UCCL_PROXY=1`, `EP_FORCE_NO_NVLINK=1`, and
`local_world_size > 1`, the runtime chooses a POSIX shared host window even if
`UCCL_FORCE_NO_GDR=0`. `EP_UCCL_FORCE_GPU_WINDOW=1` only disables that policy for
diagnostics; it is not a supported benchmark or production path for L4/PCIe
multi-GPU-per-node runs.

The important code paths are:

- `csrc/elastic/buffer.hpp`: selects `uccl_use_host_window`, allocates the
  window, and passes its addresses to UCCL proxy threads.
- `csrc/uccl/include/shared_buffer.hpp`: implements the POSIX shared host
  window used by local ranks on the same node.
- `csrc/uccl/src/proxy.cpp`: uses `shared_rdma_base != nullptr` to route
  same-node `WRITE`, `ATOMIC`, and `PUT_VALUE` commands through host-memory
  operations instead of RDMA loopback.
- `csrc/kernels/backend/nccl.cu`: contains the original NCCL symmetric-window
  registration path, which UCCL proxy mode bypasses for EP payload movement.

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
- Current JIT-visible UCCL transport version is `52`.

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
custom mini-benchmarks. The scale matrix still reports the BF16 basic
dispatch/combine path for transport comparisons, and the full first-test path is
validated separately for the no-GDR consumer target.

Main files:

- `tests/elastic/test_ep.py`
- `run_multinode.sh`

What changed:

- Added environment controls useful for UCCL validation, including benchmark
  iteration control and targeted path filters.
- Current scale validation uses `EP_TEST_BASIC_ONLY=1`,
  `EP_TEST_DO_HANDLE_COPY=0`, and `EP_TEST_DISABLE_FP8=1`.
- Full first-test validation removes the basic-only filter and covers expanded
  dispatch, cached dispatch, combine, and reduced combine.

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

All current scale data uses the DeepEPv2 built-in benchmark, not custom
mini-benchmarks:

```bash
timeout 15s $PYTHON_BIN \
  tests/elastic/test_ep.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check
```

Common env:

```bash
EP_USE_UCCL_PROXY=1
EP_FORCE_NO_NVLINK=1
EP_TEST_BASIC_ONLY=1
EP_TEST_DO_HANDLE_COPY=0
EP_TEST_DISABLE_FP8=1
EP_FORCE_PROCESS_EXIT=1
```

Cells are rank averages in the form `SO/SU GB/s, legacy GB/s @ latency`. SO/SU
bandwidth uses the traffic that DeepEPv2 attributes to scale-out/scale-up.
Legacy bandwidth uses the historical low-latency numerator, so it is useful for
same-topology comparisons but must not be treated as physical link bandwidth.

### 1n x 2g

Environment: single node, `CUDA_VISIBLE_DEVICES=0,1`, `--num-processes=2`,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 3/3 GB/s, 12.5 GB/s @ 1180 us | 7.5/7.5 GB/s, 30.0 GB/s @ 488 us |
| no-GDR | 3/3 GB/s, 12.0 GB/s @ 1238 us | 8.0/8.0 GB/s, 30.5 GB/s @ 482 us |

### 1n x 4g

Environment: single node, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `--num-processes=4`,
`EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 3/3 GB/s, 6.8 GB/s @ 2222 us | 6.3/6.3 GB/s, 14.0 GB/s @ 1031 us |
| no-GDR | 3/3 GB/s, 7.0 GB/s @ 2084 us | 6.3/6.3 GB/s, 14.3 GB/s @ 1015 us |

1n x 4g looks poor if only the legacy number is read, but the attributed
throughput is not lower than 1n x 2g. With top-8 and 64 experts, increasing from
2 to 4 scaleout ranks raises the expected number of touched destinations per
token from about 2.0 to about 3.6, which matches the measured scaleout bytes
increase from about 3.7 MB/rank to about 6.6 MB/rank. The path is therefore doing
more real work at about the same 3 GB/s dispatch rate. GDR and no-GDR are close
because both intentionally use the shared host window for no-NVLink multi-local
ranks. `EP_UCCL_FORCE_GPU_WINDOW=1` is a diagnostic-only path and times out for
this 1n x 4g benchmark on L4/PCIe.

### 2n x 1g

Environment: two nodes, `CUDA_VISIBLE_DEVICES=3`, `--num-processes=1` per
node, `EP_BENCH_NUM_TESTS=5`.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 6/6 GB/s, 23.0 GB/s @ 646 us | 9/9 GB/s, 35.0 GB/s @ 423 us |
| no-GDR | 3/3 GB/s, 12.5 GB/s @ 1179 us | 7/7 GB/s, 29.5 GB/s @ 498 us |

### 2n x 4g

Environment: two nodes, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`--num-processes=4` per node, `EP_BENCH_NUM_TESTS=1` to stay within the
15-second script timeout.

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| GDR | 2/2 GB/s, 3.1 GB/s @ 4345 us | 5.3/5.3 GB/s, 8.0 GB/s @ 1842 us |
| no-GDR | 2/2 GB/s, 3.0 GB/s @ 4698 us | 5.0/5.0 GB/s, 7.1 GB/s @ 1950 us |

The previous blocked 2n x 4g case is fixed for the BF16 basic built-in
benchmark. GDR-enabled runs can still exceed the 15-second wrapper during
teardown after all ranks have printed benchmark lines; the printed dispatch and
combine rows are valid for comparison.

### Full-path validation

Full-path validation removes `EP_TEST_BASIC_ONLY` and runs the first built-in
case with `do_handle_copy=1`, expanded dispatch, cached dispatch, ordinary
combine, and reduced combine:

```bash
timeout 15s $PYTHON_BIN \
  tests/elastic/run_full_path_bench.py \
  --transport nogdr \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --validate-only --trace-steps
```

For profiling under the same full data path, use the same runner with
`--measure-stages=all` or a comma-separated subset of `dispatch`,
`expanded_dispatch`, `cached_dispatch`, `combine`, and `reduced_combine`.
Unselected stages still execute once and print `nan` placeholder bandwidths.

| Topology | GDR | no-GDR |
| --- | --- | --- |
| 1n x 2g | pass | pass |
| 1n x 4g | pass | pass |
| 2n x 1g | pass | pass |
| 2n x 4g | startup can exceed 15 s | pass |

The target consumer path is no-GDR. In 2n x 4g no-GDR, all eight ranks reach
`after reduced combine` within the 15-second wrapper once the UCCL JIT cache is
warm. 2n x 4g GDR is retained as diagnostics; repeated 8-process GDR startup can
consume the wrapper before the first test prints.

For the no-GDR target, 1n x 4g full-path performance with `EP_BENCH_NUM_TESTS=1`
is:

| Stage | Rank-average result |
| --- | ---: |
| Dispatch | ~3/3 GB/s SO/SU, ~6 GB/s legacy @ ~2.4 ms |
| Expanded dispatch | ~3/3 GB/s SO/SU, ~7 GB/s legacy @ ~2.1 ms |
| Cached dispatch | ~3/3 GB/s SO/SU, ~6 GB/s legacy @ ~2.3 ms |
| Combine | ~6 GB/s SO/SU, ~14 GB/s legacy @ ~1.0 ms |
| Reduced combine | ~4 GB/s SO/SU, ~4 GB/s legacy @ ~3.4 ms |

## Known limitation and next performance work

The four-table matrix above is BF16 basic-only. The full-path no-GDR target now
completes the first built-in test case, but reduced combine still moves
expanded-layout data and more host-window traffic than ordinary combine, so it
is the next performance bottleneck to improve.

- Use `tests/elastic/run_full_path_bench.py --measure-stages=<stage>` for short
  focused profiling when all-stage 2n x 4g runs approach the 15-second wrapper.
- Increasing proxy threads from 4 to 8 was previously unstable and should not be
  retried as a simple fix.

## Validation checklist

After UCCL transport, JIT-visible headers, or barrier layout changes:

```bash
conda activate uccl
make -j SM=89 PYTHON=$PYTHON_BIN
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
