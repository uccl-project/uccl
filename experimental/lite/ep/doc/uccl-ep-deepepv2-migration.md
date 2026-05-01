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
| GDR on | `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0`, `NCCL_NET_GDR_LEVEL=5` | GPU RDMA windows |
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
- Current JIT-visible UCCL transport version is `48`.

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
  windows when UCCL is active.
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

- GDR mode uses GPU buffers registered for RDMA.
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

Cells are `DeepEPv2 SU BW / legacy BW @ latency`. Legacy BW uses the old
DeepEP low-latency numerator (`valid_topk * hidden * 2` for BF16).

| Mode | Rank | Dispatch | Expanded dispatch | Cached dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDR on | l40/r0 | 6 / 23 GB/s @ 636.607 us | 6 / 23 GB/s @ 638.283 us | 6 / 24 GB/s @ 618.700 us | 5 / 21 GB/s @ 693.329 us | 4 / 4 GB/s @ 3714.000 us |
| GDR on | l41/r1 | 5 / 21 GB/s @ 684.153 us | 5 / 22 GB/s @ 667.457 us | 6 / 22 GB/s @ 654.573 us | 5 / 20 GB/s @ 724.166 us | 4 / 4 GB/s @ 3750.000 us |
| no-GDR | l40/r0 | 3 / 12 GB/s @ 1186.000 us | 3 / 12 GB/s @ 1191.000 us | 3 / 13 GB/s @ 1153.000 us | 1 / 5 GB/s @ 3107.000 us | 1 / 1 GB/s @ 16026.000 us |
| no-GDR | l41/r1 | 3 / 12 GB/s @ 1247.000 us | 3 / 12 GB/s @ 1237.000 us | 3 / 13 GB/s @ 1154.000 us | 1 / 5 GB/s @ 3111.000 us | 1 / 1 GB/s @ 16033.000 us |

Summary:

- GDR mode reaches the expected 20+ GB/s legacy bandwidth for dispatch and
  combine on 2n x 1g.
- no-GDR dispatch, expanded dispatch, and cached dispatch reach 12-13 GB/s
  legacy bandwidth.
- no-GDR combine remains the main unresolved bottleneck at about 5 GB/s legacy
  bandwidth; reduced combine is about 1 GB/s.

## Known limitation and next performance work

no-GDR combine is still below the desired 10+ GB/s target. Lightweight changes
were tried and reverted:

- Increasing proxy threads from 4 to 8 caused dispatch/barrier instability.
- A single-scaleup direct-read combine shortcut gave only about 5% improvement
  and risked GDR/full-path regressions.

Further no-GDR combine work should be treated as a larger host-window/proxy
pipeline redesign, not a small localized patch.

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

Keep test-script timeouts at 15 seconds or less. If JIT compilation consumes
the timeout after a transport-version bump, rerun with the same
`EP_JIT_CACHE_DIR` after warmup.
