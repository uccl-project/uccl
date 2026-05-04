# UCCL-EP to DeepEPv2 Lite migration

This document describes how UCCL-EP is integrated into the DeepEPv2 Lite
`ElasticBuffer` codebase under `experimental/lite/ep`, and records the current
benchmark numbers. Operational instructions (build, run, env) live in
`AGENTS.md`.

## Scope

- Active EP data path: UCCL CPU proxy via `EP_USE_UCCL_PROXY=1`.
- NCCL is used only for communicator/bootstrap and DeepEP interface
  compatibility; NCCL GIN does not move EP payload.
- Validation hardware: NVIDIA L4 (`sm_89`), no NVLink, ConnectX over ibverbs.
- Validation shape: `128 tok × 7168 hid × top-8 × 64 exp`, BF16
  (`--fp8-dispatch-modes=0`), `do_handle_copy=0`, `expert_alignment=128`.

Two transport modes coexist:

| Mode | Settings | Backing window |
| --- | --- | --- |
| GDR | `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0`, `NCCL_NET_GDR_LEVEL=5` | GPU RDMA window for single-local-rank runs; shared host window otherwise |
| no-GDR | `UCCL_FORCE_NO_GDR=1`, `EP_FORCE_HOST_WINDOW=1`, `NCCL_NET_GDR_LEVEL=0` | host-pinned CUDA-mapped window |

## Window concept

DeepEP kernels and the UCCL proxy share a *window* — the memory region
holding payload buffers, counters, and completion signals. GPU kernels write
window-relative commands; CPU proxies execute them as host memcpy (same node)
or RDMA write (remote node).

| Window | Backing | Use |
| --- | --- | --- |
| host window | `cudaHostAlloc` or POSIX shared memory registered with CUDA | no-GDR runs; all no-NVLink multi-GPU-per-node UCCL runs |
| GPU window | `cudaMalloc` registered for RDMA, with CUDA IPC for same-node | single-local-rank GDR runs; diagnostics only otherwise |

On L4/PCIe without NVLink, same-node GPU-window traffic is slower and less
reliable than CPU proxy memcpy through shared host memory. So UCCL +
`EP_FORCE_NO_NVLINK=1` + `local_world_size > 1` always picks a POSIX shared
host window even with `UCCL_FORCE_NO_GDR=0`. `EP_UCCL_FORCE_GPU_WINDOW=1`
exists for diagnostics only.

Key code paths:

- `csrc/elastic/buffer.hpp` — selects `uccl_use_host_window`, allocates
  windows, hands addresses to the proxy.
- `csrc/uccl/include/shared_buffer.hpp` — POSIX shared host window for local
  ranks on the same node.
- `csrc/uccl/src/proxy.cpp` — routes same-node `WRITE`/`ATOMIC`/`PUT_VALUE`
  through host memcpy when `shared_rdma_base != nullptr`.

## Migration summary

The integration touches five areas:

1. **UCCL proxy runtime** (`csrc/uccl/`): D2H ring queues, RDMA setup, proxy
   thread management, JIT flags. JIT-visible transport version is `52`;
   bump it whenever device headers or transport semantics change.
2. **`ElasticBuffer` activation** (`deep_ep/buffers/elastic.py`,
   `csrc/elastic/buffer.hpp`): selects window type, exchanges UCCL peer
   metadata via `all_gather_object`, starts proxies, and waits for readiness
   in `sync_uccl_peers()` before kernels enqueue commands.
3. **EP data movement** (`deep_ep/include/deep_ep/impls/{dispatch,combine,
   hybrid_*}.cuh`, `csrc/kernels/elastic/*.hpp`): kernels post WRITE /
   PUT_VALUE into D2H rings instead of NCCL GIN. Transport puts are safe for
   divergent top-k branches; active deduplicated lanes can each post payload.
4. **Tag-aware Gin barriers** (`deep_ep/include/deep_ep/common/{layout,
   comm}.cuh`): `kNumGinBarrierTags = 16`, signal/shadow storage indexed by
   `(tag, rank)`. Without this, dispatch and combine phases collide on the
   same per-rank counter.
5. **Single-scaleup combine optimization**
   (`deep_ep/include/deep_ep/impls/hybrid_combine.cuh`): for
   `kNumScaleupRanks == 1 && !kAllowMultipleReduction`, the scaleup warp
   writes directly into the scaleout send/recv buffer and issues the UCCL
   put. The forward warp returns early. Compiled for both GDR and no-GDR so
   the comparison stays fair.

## Benchmark results

All numbers are from DeepEPv2's built-in benchmark with the validation shape
above and CLI:

```
--test-first-only --skip-check --num-gpu-timeout-secs=3 \
--do-handle-copy-modes=0 --expert-alignment-modes=128 \
--fp8-dispatch-modes=0 --num-bench-tests=<N>
```

Cells are rank averages, format `SO/SU GB/s, legacy GB/s @ latency`. SO/SU
uses bytes DeepEPv2 attributes to scale-out/scale-up. Legacy uses the
historical low-latency numerator and is comparable only between rows of the
same topology.

| Topology | Mode | Dispatch | Combine |
| --- | --- | ---: | ---: |
| 1n × 2g (`N=5`) | GDR    | 3/3, 13 GB/s @ 1144 µs | 8/8, 31 GB/s @ 472 µs |
| 1n × 2g (`N=5`) | no-GDR | 3/3, 12 GB/s @ 1210 µs | 7/7, 29 GB/s @ 507 µs |
| 1n × 4g (`N=5`) | GDR    | 3/3,  6 GB/s @ 2344 µs | 6/6, 13 GB/s @ 1085 µs |
| 1n × 4g (`N=5`) | no-GDR | 3/3,  6 GB/s @ 2218 µs | 6/6, 14 GB/s @ 1050 µs |
| 2n × 1g (`N=5`) | GDR    | 5/5, 22 GB/s @  678 µs | 8/8, 33 GB/s @ 444 µs |
| 2n × 1g (`N=5`) | no-GDR | 3/3, 13 GB/s @ 1158 µs | 7/7, 28 GB/s @ 528 µs |
| 2n × 4g (`N=1`) | GDR    | fail (CUDA launch error during dispatch) | — |
| 2n × 4g (`N=1`) | no-GDR | fail (Gin barrier timeout during dispatch) | — |

Notes:

- 1n × 4g shows lower legacy throughput than 1n × 2g because top-8 over 4
  scaleout ranks raises the touched destinations per token from ~2.0 to ~3.6
  (measured scaleout bytes per rank rises from ~3.7 MB to ~6.6 MB). Attributed
  SO/SU stays at ~3 GB/s; the path is doing more real work at the same rate.
- 1n × 4g GDR ≈ no-GDR because both use the shared host window for
  no-NVLink multi-local ranks.
- 2n × 4g currently fails during dispatch on this branch on L4. no-GDR hits
  a Gin barrier timeout (`tag=6/8`); GDR hits `CUDA_ERROR_LAUNCH_FAILED`.
  This is a regression from earlier numbers and is the next item to debug.

## Full-path validation

Beyond the basic benchmark, `tests/elastic/run_full_path_bench.py` exercises
expanded dispatch, cached dispatch, ordinary combine, and reduced combine
with `do_handle_copy=1`:

```bash
timeout 15s $PYTHON_BIN tests/elastic/run_full_path_bench.py \
  --transport nogdr \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --validate-only --trace-steps
```

Use `--measure-stages=all` or a comma-separated subset
(`dispatch,expanded_dispatch,cached_dispatch,combine,reduced_combine`) for
profiling. Unselected stages still run once and report `nan` bandwidth.

## Open issues

- **2n × 4g failure** (above) — the basic benchmark previously ran with
  `--num-bench-tests=1` on this topology. Current branch fails at dispatch in
  both GDR and no-GDR modes; needs investigation before reporting numbers.
- **Reduced combine** still moves expanded-layout data and more host-window
  traffic than ordinary combine, so it is the per-stage performance
  bottleneck on no-GDR. Use
  `run_full_path_bench.py --measure-stages=reduced_combine` for focused work.
- Increasing UCCL proxy threads from 4 to 8 was previously unstable; do not
  retry as a simple fix.

## Validation checklist

After UCCL transport, JIT-visible header, or barrier layout changes:

```bash
make -j SM=89 PYTHON=$PYTHON_BIN
```

Then rerun the 2n × 1g benchmark in both GDR and no-GDR modes (the smallest
multi-node configuration covering both window types). Keep test wrapper
timeouts at 15 seconds; if the JIT cache is cold, rerun with the same
`EP_JIT_CACHE_DIR` after warmup.
