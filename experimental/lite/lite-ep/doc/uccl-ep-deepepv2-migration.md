# UCCL-EP on DeepEPv2 Lite

UCCL-EP integrated into the DeepEPv2 `ElasticBuffer` codebase under
`experimental/lite/ep`. Operational instructions (build, run, env) live in
`AGENTS.md`.

## Scope

- Active EP datapath: UCCL CPU proxy via `EP_USE_UCCL_PROXY=1`.
- NCCL is used for communicator/bootstrap only; NCCL Gin does not move EP payload.
- Validation HW: NVIDIA L4 (`sm_89`), no NVLink, ConnectX over ibverbs.
- Validation shape: `128 tok × 7168 hid × top-8 × 64 exp`, BF16,
  `do_handle_copy=0`, `expert_alignment=128`.

Two transport modes:

| Mode | Settings | Window |
| --- | --- | --- |
| GDR | `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0` | GPU RDMA window for single-local-rank; shared host window otherwise |
| no-GDR | `UCCL_FORCE_NO_GDR=1`, `EP_FORCE_HOST_WINDOW=1` | host-pinned CUDA-mapped window |

On L4/PCIe without NVLink, same-node GPU-window traffic is slower than
CPU-proxy memcpy through shared host memory, so UCCL +
`EP_FORCE_NO_NVLINK=1` + `local_world_size > 1` always picks a POSIX
shared host window. `EP_UCCL_FORCE_GPU_WINDOW=1` exists for diagnostics.

Key code paths: `csrc/elastic/buffer.hpp` (window selection),
`csrc/uccl/include/shared_buffer.hpp` (POSIX shared window),
`csrc/uccl/src/proxy.cpp` (same-node memcpy short-circuit).

## Migration summary

1. **UCCL proxy runtime** (`csrc/uccl/`): D2H ring queues, RDMA setup,
   proxy threads, JIT flags. JIT transport version is `52`; bump on
   device-header / transport-semantics changes.
2. **`ElasticBuffer` activation** (`deep_ep/buffers/elastic.py`,
   `csrc/elastic/buffer.hpp`): selects window type, exchanges UCCL peer
   metadata via `all_gather_object`, starts proxies, waits for readiness
   in `sync_uccl_peers()` before enqueue.
3. **EP data movement** (`deep_ep/include/deep_ep/impls/{dispatch,
   combine,hybrid_*}.cuh`, `csrc/kernels/elastic/*.hpp`): kernels post
   WRITE / PUT_VALUE into D2H rings instead of NCCL Gin. Safe for
   divergent top-k branches.
4. **Tag-aware Gin barriers** (`deep_ep/include/deep_ep/common/{layout,
   comm}.cuh`): `kNumGinBarrierTags = 16`, signal/shadow indexed by
   `(tag, rank)`. Without this, dispatch and combine collide on the
   same per-rank counter.
5. **No-NVLink fallback** (`deep_ep/buffers/elastic.py`,
   `deep_ep/include/deep_ep/common/handle.cuh`,
   `csrc/kernels/backend/nccl.cu`, `csrc/jit/compiler.hpp`,
   `csrc/elastic/buffer.hpp`): on PCIe-only GPUs (L4/L40), upstream's
   NVLink scaleup path (`is_nvlink_accessible`, LSA team direct
   `ld.acquire.sys` cross-GPU loads) hangs or segfaults. Setting
   `EP_FORCE_NO_NVLINK=1` forces `kNumScaleupRanks=1` in Python, gates
   out the NVLink helpers in device headers via JIT macro, and skips
   NVLink-requiring NCCL symmetric-memory segment types — so all
   cross-GPU traffic goes through the UCCL proxy. Consequence:
   `kNumScaleupRanks=1` makes the `hybrid_*` kernels collapse into
   their direct-mode branch (no intra-node NVLink aggregation), so we
   only support direct EP — DeepEPv2's hybrid traffic-deduplication
   advantage is unavailable in this deployment.
6. **SM89 vectorized TMA fallback**
   (`deep_ep/include/deep_ep/common/ptx.cuh` and call sites in
   `{hybrid_,}{dispatch,combine}.cuh`, the two epilogue kernels, and
   `pp_send_recv.cuh`): on SM89 the upstream `tma_load_1d` /
   `tma_store_1d` fall back to a single-thread byte loop (one lane
   copies 14 KB while 31 lanes idle), which is a perf cliff. Replaced
   with warp-cooperative variants that vectorize the copy across all
   32 lanes (`cp.async` + `ld.shared.v4` + `st.global.v4`), and made
   the SM89 `tma_store_commit()` emit a system-scope release fence so
   GPU stores are flushed past L2 before the proxy fires the RDMA
   doorbell — without it, GDR mode (NIC reads GPU memory over PCIe
   peer-to-peer with no CPU in the path) can read stale or
   partially-written bytes. SM90 paths are untouched.

## Benchmark results

CLI:

```
--test-first-only --skip-check \
--do-handle-copy-modes=0 --expert-alignment-modes=128 \
--fp8-dispatch-modes=0 --num-bench-tests=<N>
```

Cells are rank averages, format `SO/SU GB/s @ latency µs ± stddev`. SO
counts inter-node bytes (RDMA); SU counts intra-node bytes (host SHM
memcpy). Same byte formulas as upstream `tests/elastic/test_ep.py`
(self-rank traffic counted), with the only substitution being
`NVLink domain → physical node (LOCAL_WORLD_SIZE)` because under
`EP_FORCE_NO_NVLINK=1` the logical scaleup domain `kNumScaleupRanks=1`
does not match what data physically traverses.

| Topology | Mode | Dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: |
| 1n × 2g | no-GDR | 0/7 GB/s @ 552 ± 5 µs | 0/16 GB/s @ 227 ± 8 µs | 0/21 GB/s @ 686 ± 33 µs |
| 1n × 4g | no-GDR | 0/6 GB/s @ 1185 ± 18 µs | 0/13 GB/s @ 510 ± 17 µs | 0/17 GB/s @ 859 ± 65 µs |
| 2n × 1g | no-GDR | 7/3 GB/s @ 546 ± 8 µs | 17/8 GB/s @ 220 ± 12 µs | 6/11 GB/s @ 676 ± 32 µs |
| 2n × 1g | GDR    | 18/9 GB/s @ 201 ± 19 µs | 19/9 GB/s @ 199 ± 22 µs | 5/10 GB/s @ 779 ± 230 µs |
| 2n × 4g | no-GDR | 2/3 GB/s @ 1848 ± 83 µs | 5/6 GB/s @ 797 ± 86 µs | 3/6 GB/s @ 1141 ± 176 µs |
| 2n × 4g | GDR    | 2/3 GB/s @ 1863 ± 43 µs | 5/6 GB/s @ 764 ± 68 µs | 4/7 GB/s @ 1098 ± 154 µs |

DeepEP README's "Bottleneck Bandwidth" is just the bottleneck-link column
of the same `(SO)/(SU) GB/s` upstream prints: SO for multi-node (RDMA
limit), SU for single-node (NVLink limit). Pick the relevant column from
the table above; no separate sub-table needed.

### NCCL all-to-all reference

`tests/elastic/bench_nccl_a2a.py` runs `dist.all_to_all_single` (NCCL
backend, `NCCL_NET_GDR_LEVEL=0`, `NCCL_P2P_DISABLE=1`) with the same
`legacy_bytes = num_tokens × num_topk × hidden × 2` as the EP bench, so
its `legacy GB/s` is directly comparable to ours. Steady-state, 30 iters:

| Topology | NCCL a2a | EP dispatch (legacy) | EP reduced combine (legacy) |
| --- | ---: | ---: | ---: |
| 1n × 2g | 25 GB/s @ 583 µs | 26 GB/s | 21 GB/s |
| 1n × 4g | 17 GB/s @ 868 µs | 12 GB/s | 17 GB/s |
| 2n × 1g | 23 GB/s @ 628 µs | 27 GB/s | 22 GB/s |
| 2n × 4g | 15 GB/s @ 998 µs |  6 GB/s | 12 GB/s |

Take-aways: 2n × 1g, 1n × 2g already match (or exceed, on the small
configs) NCCL's pure a2a; 1n × 4g sits ~30% under and 2n × 4g ~2.4×
under. The 1n × 4g gap is the intra-node SHM memcpy path being
DDR-bandwidth-bound (proxy threads observed idle during dispatch under
`top -H`); the 2n × 4g gap is per-WR overhead in the proxy
ibv_post_send loop, not NIC bandwidth (see Open issues).

## Open issues

- **Single-scaleup combine fast path** (optional optimization, not
  required for correctness): in `hybrid_combine.cuh`, for
  `kNumScaleupRanks == 1 && !kAllowMultipleReduction`, the scaleup warp
  writes directly into the scaleout send/recv buffer and the forward
  warp returns early, saving one intermediate copy. Targets 2n × 1g
  specifically; do not generalize without revisiting the
  `kAllowMultipleReduction` and expanded-layout cases.
- **Reduced combine** moves expanded-layout data and more host-window
  traffic than ordinary combine; per-stage bottleneck on no-GDR.
- **Cold-start**: first put_value on a fresh UCCL connection can take
  several seconds. Keep `--num-gpu-timeout-secs` at default (100 s) and
  reuse `EP_JIT_CACHE_DIR`.
- **Proxy threads 4 → 8** was previously unstable; do not retry as a
  simple fix.
- **2n × 4g dispatch** ~2 ms after the channel-cap fix (commit
  `c1f893c3`, was ~3 ms after SM89 TMA fix, ~4 ms before). Remaining
  cost is per-WR overhead (`mlx5_post_send` spinlock + proxy
  `post_gpu_command`), not NIC bandwidth. Real fix needs kernel-level
  per-peer bulk-WRITE coalescing inside `hybrid_dispatch.cuh` — current
  one-WR-per-token layout interleaves remote slots so multi-SGE merging
  in the proxy doesn't help.
- **Intra-node ceiling on 1n × 4g.** Dispatch ~1180 µs corresponds to
  ~22 MB/iter intra-node SHM traffic (4 ranks × 5.5 MB) at ~25 GB/s
  effective host DDR bandwidth on this Xeon Silver 4410Y dual-channel
  DDR4. The `top -H` sample of proxy threads during dispatch shows them
  idle (CPU not the bottleneck), so the bench is hitting raw DDR
  throughput. Tested optimizations (kept as opt-in code paths, not
  default) for future hardware where DDR isn't the limiter:
  - **NT-store memcpy** (`UCCL_EP_SHM_USE_NT_MEMCPY=1`, kept reverted):
    AVX2 `vmovntdq` for the SHM WRITE batch. A/B showed −13% / −64%
    regression on 2n × 4g (NT stores force DRAM round-trip; cached
    stores let receiver-GPU PCIe DMA snoop the writer's L3).
  - **THP + page pre-touch** (`UCCL_EP_SHM_HUGEPAGE=1` /
    `UCCL_EP_SHM_PRETOUCH=1`): no measurable change vs. run-to-run noise
    (~10% on this hardware), kept off by default.
  - **Intra-node direct host-window write**: enabled by default
    (`EP_UCCL_INTRANODE_DIRECT=1`) since it doesn't regress and frees
    DDR bandwidth + proxy CPU for co-located workloads. The latency
    improvement on the bench itself is 0 µs (the proxy memcpy is fully
    overlapped with PCIe), but ~22 MB/iter of host DDR traffic and the
    proxy thread's memcpy work are eliminated. Set
    `EP_UCCL_INTRANODE_DIRECT=0` to fall back to the proxy-memcpy path
    for diagnostics.

  Real fix would require either NVLink (changes intra-node from PCIe-
  bound to NVLink-bound, ~10× faster) or a different algorithm that
  cuts intra-node bytes per token.

## Validation checklist

After UCCL transport, JIT-visible header, or barrier layout changes:

```bash
make -j SM=89 PYTHON=$PYTHON_BIN
```

Then rerun 2n × 1g in both GDR and no-GDR (smallest multi-node
configuration covering both window types). Test wrapper timeout 15 s.

For changes that may shift work between the kernel hot path and the
proxy thread, or alter PCIe traffic patterns, run the full six-config
matrix above and compare against a clean-tree baseline. Aim for ≥ 8
samples on 2n × 4g (stddev 3–7%); 2 samples suffice elsewhere.

## Full-path validation

`tests/elastic/run_full_path_bench.py` exercises expanded dispatch,
cached dispatch, combine, and reduced combine with `do_handle_copy=1`:

```bash
timeout 120s $PYTHON_BIN tests/elastic/run_full_path_bench.py \
  --transport nogdr \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --validate-only --trace-steps
```

Use `--measure-stages=all` or a comma-separated subset for profiling.
