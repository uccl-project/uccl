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
--test-first-only --skip-check \
--do-handle-copy-modes=0 --expert-alignment-modes=128 \
--fp8-dispatch-modes=0 --num-bench-tests=<N>
```

Cells are rank averages, format `SO/SU GB/s @ latency µs ± stddev`. **SO
counts inter-node bytes (RDMA traffic); SU counts intra-node bytes (host
shared-memory memcpy).** This mirrors upstream DeepEPv2's bench
methodology exactly — same formulas (unique `(token, dst-domain)` pairs
for SO, `num_recv_tokens` from `<scaleup-domain>` peers for SU,
self-domain included by default and excluded by `--ignore-local-traffic`)
— with the **only substitution** being the boundary unit:
`NVLink domain` (upstream) → `physical node` (LOCAL_WORLD_SIZE here),
because under `EP_FORCE_NO_NVLINK=1` the logical scaleup domain
`kNumScaleupRanks=1` does not match what the data physically traverses.
Numbers are steady-state (`--num-bench-tests=10 --skip-check`); n
samples ≈ ranks × num_bench_tests × warmup.

| Topology | Mode | Dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: |
| 1n × 2g | no-GDR | 0/7 GB/s @ 552 ± 5 µs | 0/16 GB/s @ 227 ± 8 µs | 0/21 GB/s @ 686 ± 33 µs |
| 1n × 4g | no-GDR | 0/6 GB/s @ 1185 ± 18 µs | 0/13 GB/s @ 510 ± 17 µs | 0/17 GB/s @ 859 ± 65 µs |
| 2n × 1g | no-GDR | 7/3 GB/s @ 546 ± 8 µs | 17/8 GB/s @ 220 ± 12 µs | 6/11 GB/s @ 676 ± 32 µs |
| 2n × 1g | GDR    | 18/9 GB/s @ 201 ± 19 µs | 19/9 GB/s @ 199 ± 22 µs | 5/10 GB/s @ 779 ± 230 µs |
| 2n × 4g | no-GDR | 2/3 GB/s @ 1848 ± 83 µs | 5/6 GB/s @ 797 ± 86 µs | 3/6 GB/s @ 1141 ± 176 µs |
| 2n × 4g | GDR    | 2/3 GB/s @ 1863 ± 43 µs | 5/6 GB/s @ 764 ± 68 µs | 4/7 GB/s @ 1098 ± 154 µs |

Sanity checks per row:
- **1n configs** have **SO = 0** (single node has no inter-node traffic)
  and SU > 0 (all cross-GPU traffic is intra-node SHM).
- **2n × 1g** has SU > 0 because upstream's `num_recv_tokens` includes
  self-recv bytes; with only 1 GPU per node the "intra-node domain" is
  just self, so SU reflects the bytes the GPU writes for itself.
- **2n × 4g** has both > 0 (mixed RDMA inter-node + SHM intra-node).

These reflect the SM89 warp-cooperative TMA fallback fix (commits
`38093df5`, `2f958b9b`, `44d1b976`, `3cce0ee8`), the proxy `getenv`
cache (commit `9255edf9`), and the channel-count cap on multi-node
multi-GPU UCCL proxy (commit `c1f893c3`). See the corresponding entries
under "Open issues" for context. The intra-kernel "copy GB/s" / "reduce
GB/s" numbers reported by the benchmark are L2-cache-resident (L40 has
96 MB L2 vs ~1.8 MB working set, and the dispatch/combine kernel just
wrote the buffer), not GDDR or PCIe bandwidth, so they should not be
quoted as a sustained throughput.

For reference, the pre-`c1f893c3` baseline on 2n × 4g was: dispatch
~3050 µs / ~3164 µs (no-GDR / GDR), combine ~1034 µs / ~1060 µs.
The fix delivers −34 to −38% on dispatch and −11 to −24% on combine
with a +9% regression on 2n × 4g GDR reduced combine (see Open issues).

Notes:

- 1n × 4g shows lower legacy throughput than 1n × 2g because top-8 over 4
  scaleout ranks raises the touched destinations per token from ~2.0 to ~3.6
  (measured scaleout bytes per rank rises from ~3.7 MB to ~6.6 MB). Attributed
  SO/SU stays at ~3 GB/s; the path is doing more real work at the same rate.
- 1n × 4g GDR ≈ no-GDR because both use the shared host window for
  no-NVLink multi-local ranks.
- 2n × 4g uses `NCCL_IB_HCA=mlx5_0,mlx5_1` (both NICs) via the default in
  `run_multinode.sh`. UCCL's PCIe-distance + NUMA-aware NIC selection in
  `csrc/uccl/src/rdma.cpp` automatically pins each GPU to its node-local NIC:
  GPU 0,1 → mlx5_0 (NUMA 0); GPU 2,3 → mlx5_1 (NUMA 1). Confirmed via
  `EP_UCCL_DEBUG=1` output. The remaining bottleneck at ~4 ms is proxy CPU
  overhead (D2H ring drain + RDMA submission), not NIC bandwidth.

## Full-path validation

Beyond the basic benchmark, `tests/elastic/run_full_path_bench.py` exercises
expanded dispatch, cached dispatch, ordinary combine, and reduced combine
with `do_handle_copy=1`:

```bash
timeout 120s $PYTHON_BIN tests/elastic/run_full_path_bench.py \
  --transport nogdr \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --validate-only --trace-steps
```

Use `--measure-stages=all` or a comma-separated subset
(`dispatch,expanded_dispatch,cached_dispatch,combine,reduced_combine`) for
profiling. Unselected stages still run once and report `nan` bandwidth.

## Open issues

- **Reduced combine** still moves expanded-layout data and more host-window
  traffic than ordinary combine, so it is the per-stage performance
  bottleneck on no-GDR. Use
  `run_full_path_bench.py --measure-stages=reduced_combine` for focused work.
- The first put_value on a cold UCCL connection (especially across nodes)
  can take a few seconds; keep `--num-gpu-timeout-secs` at the default
  (100 s) for cold-start runs and rely on `EP_JIT_CACHE_DIR` reuse for
  subsequent runs.
- Increasing UCCL proxy threads from 4 to 8 was previously unstable; do not
  retry as a simple fix.
- 2n × 4g dispatch latency was originally ~3 ms (after the SM89 warp-coop
  TMA fix; was ~4 ms pre-fix). After commit `c1f893c3` it is **~2 ms**
  (−34 to −38%, see steady-state table above). The remaining cost is
  per-WR overhead, not NIC bandwidth.

  **Root cause analysis (commit `c1f893c3`)**: kineto traces showed the
  dispatch hot kernel is `hybrid_dispatch_impl`, even when the host code
  is reached via `--allow-hybrid-mode 0`. With `EP_FORCE_NO_NVLINK=1`
  the logical domain becomes `(num_scaleout_ranks=8, num_scaleup_ranks=1)`,
  so the host-side `num_scaleout_ranks > 1` branch picks the hybrid
  kernel. That kernel decides `num_channels_per_sm = 3` from smem budget
  (`58 SMs × 3 = 174 channels`) and fires one "finish" PUT_VALUE per
  scaleout peer per channel at the end of the dispatch loop, regardless
  of how many tokens that channel actually carried. With
  `num_max_tokens_per_rank = 128` and 174 channels, ~88% of channels
  process 0 or 1 tokens but each still contributes 7 PUT_VALUEs (one per
  cross-rank peer in 8-rank world): **~1218 PUT_VALUEs / rank / dispatch
  just for finish signals**, on top of ~640 actual RDMA WRITEs for
  payload. Across 4 proxy threads × ~1 µs of `ibv_post_send` per WR
  (mlx5 driver per-QP `pthread_spin_lock`), that is ~465 µs of pure
  proxy CPU time per dispatch — exactly matching the 1640 µs gap
  between 1n × 4g (`~1100 µs`, no NIC) and 2n × 4g (`~2740 µs`)
  pre-fix.

  **Fix**: in `csrc/elastic/buffer.hpp`, when `use_uccl_proxy &&
  local_world_size > 1 && (num_ranks / local_world_size) > 1`, cap
  `num_channels_per_sm` to 1 (channels: 174 → 58, finish PUT_VALUEs:
  1218 → 406, −66%). The kernel's TMA warps were already idle most of
  the time on under-loaded channels, so reducing channel count does
  not lose GPU-side parallelism on small num_tokens. Single-node
  configs do not pay the cross-node finish-signal cost (intra-node
  PUT_VALUE goes through host-shm memcpy, not the NIC) and keep the
  wider channel layout to maximise GPU parallelism. `EP_NUM_CHANNELS_PER_SM`
  env var is provided as an explicit override (clamped to
  `[1, kNumMaxChannelsPerSM]`).

  **Things that previously did not help** (kept for future reference):

  - `UCCL_IB_MAX_INFLIGHT_NORMAL` (8 → 16, 64, 256): no change.
  - `EP_UCCL_FORCE_GPU_WINDOW=1` (skip host-staging): crashes with
    `CUDA IPC PUT_VALUE failed` on no-NVLink PCIe-only systems.
  - Per-GPU NIC pinning: already automatic via PCIe-distance + NUMA
    tie-breaker (verified with `EP_UCCL_DEBUG=1`).
  - Larger token counts (`--num-tokens=1024`): bandwidth doubles to
    ~4 GB/s, confirming per-message overhead is the bottleneck. The
    channel-cap fix targets the per-finish-PUT_VALUE component of
    that overhead specifically; the remaining per-token-WRITE cost
    is what the kernel-level coalescing fix below would attack.
  - **All-RDMA baseline** (intra-node also via NIC loopback): tried
    via a temporary `EP_UCCL_FORCE_ALL_RDMA=1` toggle that disables
    the shared host window so same-node peers also get an RDMA QP.
    2n × 4g result: dispatch **1 GB/s @ ~13 ms (3× slower)**,
    combine 1 GB/s @ ~11 ms. Two reasons it loses to the SHM path:
    (a) NIC loopback adds two PCIe root-complex hops vs a single
    host memcpy; (b) intra-node traffic now competes for NIC WR-rate
    budget with cross-node traffic, amplifying the existing per-WR
    bottleneck. DeepEP hybrid-ep PCIe kernel uses 1D RDMA loopback
    successfully because it bundles many tokens per WR; lite-ep's
    one-token-per-WR pattern can't amortize the loopback cost.

  **Remaining headroom** beyond `c1f893c3`: per-WR `mlx5_post_send`
  spinlock + `Proxy::post_gpu_command` are the next hot spots
  (`perf record` post-fix shows ~28% and ~37% respectively; getenv
  was eliminated in commit `9255edf9`). They are fundamental to the
  one-WR-per-token pattern. Real fix would coalesce per-token
  WRITEs into per-peer bulk WRITEs inside `hybrid_dispatch.cuh`
  (kernel-level change). Multi-SGE coalescing in the proxy was
  attempted but does not help: the `scaleout_recv_buffer` layout
  interleaves slots by (channel, slot, rank) so consecutive cmds to
  the same destination land 48 token-widths apart in the remote
  address space — never contiguous, never SGE-mergeable.
- **SM89 (Ada / L40) TMA fallback was a perf cliff** that was fixed in
  commits `38093df5`, `2f958b9b`, and `44d1b976`. DeepEPv2's
  `DISABLE_SM90_FEATURES` fallback for `tma_load_1d` / `tma_store_1d`
  (in `deep_ep/include/deep_ep/common/ptx.cuh`) was a single-thread byte
  loop — only 1/32 lanes did the 14 336-byte token copy, while 31 lanes
  idled. We added warp-cooperative variants (`tma_load_1d_warp` /
  `tma_store_1d_warp`) using `cp.async.cg` (gmem→smem) and
  `ld.shared.v4` + `st.global.v4` (smem→gmem), and converted all 23
  hot TMA call sites in `hybrid_dispatch.cuh`, `hybrid_combine.cuh`,
  `dispatch_copy_epilogue.cuh`, and `combine_reduce_epilogue.cuh` to
  use them. A follow-up commit (`44d1b976`) made the SM89
  `tma_store_commit()` emit `fence.acq_rel.sys` instead of being a
  no-op: parallel `st.global.v4` writes complete from the SM perspective
  immediately, but may still be buffered in L2 / PCIe write FIFO when
  the elected lane subsequently fires an IBGDA doorbell, so without
  the system fence cross-node GDR traffic suffered NIC RNR-retry stalls
  (2n × 4g GDR dispatch ballooned from 4400 µs pre-fix to 20000 µs
  before the fence was added). Result: dispatch 1.4–2.6× faster,
  combine 1.8–2.7× faster, reduced combine up to 3.4× faster — see
  the post-fix benchmark table. The treatment leaves SM90 paths
  untouched under `#ifndef DISABLE_SM90_FEATURES` guards. The pending
  conversions for non-hybrid kernels (`dispatch.cuh`, `combine.cuh`,
  `pp_send_recv.cuh`) were completed in commit `3cce0ee8`:
  - `dispatch.cuh`: per-token data-load TMA inside the dispatch warp.
  - `combine.cuh`: no-reduce, multi-reduce, and expanded-send
    load+store pairs (three sites).
  - `pp_send_recv.cuh::tma_copy`: rewrote the internal helper to be
    warp-cooperative on SM89 (takes `lane_id`); both `pp_send_impl` and
    `pp_recv_impl` now invoke it outside `elect_one_sync` so all 32
    lanes participate. Mbarrier init / arrive / wait remain
    single-lane via inner `elect_one_sync` + `__syncwarp`.

  After `3cce0ee8`, no `ptx::tma_*_1d(...)` call site outside an
  `#ifndef DISABLE_SM90_FEATURES` branch remains in
  `deep_ep/include/deep_ep/impls/`. SM89 builds always take the
  warp-cooperative path. SM90 behavior is unchanged because
  `tma_*_1d_warp` delegates to the elected lane there.
- **Intra-node direct host-window write (prototype, NOT committed).**
  Goal: bypass the UCCL CPU proxy memcpy for intra-node WORLD-team /
  LSA-team WRITE and PUT_VALUE in the no-NVLink shared-host-window mode
  by having the sender warp TMA-store directly into the destination
  peer's slice (each rank `cudaHostRegister`-maps the same shared POSIX
  region, so peer slices have a sender-resolvable device VA via
  `lsa_base_ptr + (dst_local_rank - my_local_rank) * per_rank_bytes`).

  The prototype was implemented end-to-end in the working tree and
  validated for correctness on 1n × 4g (test_ep.py without
  `--skip-check` passes across all variants). It is **not committed**
  because of the 2n × 4g regression below. The full diff is preserved
  in `/tmp/refactor_full.patch` for follow-up.

  Touched files in the prototype (kept in working tree only):
  - `deep_ep/include/deep_ep/common/handle.cuh`: `NCCLGin` gains
    `uccl_shared_per_rank_bytes`, `uccl_intranode_local_world_size`,
    `uccl_intranode_my_local_rank`, `uccl_intranode_node_idx`. Helper
    `uccl_intranode_dst_local_rank<team_t>` returns the dst local rank
    if the peer is on the same node and the shared window is enabled,
    else `-1`. `get_sym_ptr<World|LSA>` resolves the shifted device VA
    for intra-node peers; cross-node still returns `nullptr` (RDMA
    fallback). `is_nvlink_accessible<World|LSA>` mirrors so the
    `nvlink_bypass` branch in `combine.cuh` fires on intra-node peers.
  - `deep_ep/include/deep_ep/impls/{dispatch,combine,barrier}.cuh`:
    kernel signatures take the four scalar params and forward them to
    the `NCCLGin` constructor. Dispatch gates the local send-buffer
    write on `warp_needs_local = ptx::reduce_or(dst_ptr == nullptr ?
    1u : 0u)` so warps with all-intra-node peers skip the local copy.
    The post-`get_sym_ptr` peer-direct TMA store uses parallel
    `cp.async.bulk` on SM90 and a per-active-lane `tma_store_1d_warp`
    serial loop on SM89 to preserve the dedup'd per-lane destinations.
  - `csrc/kernels/elastic/{dispatch,combine,barrier}.hpp`: `Args` and
    `launch_*` accept and forward the four params.
  - `csrc/elastic/buffer.hpp`: accessors return descriptor values
    (`0/-1` when `uccl_use_shared_window == false`), threaded to all
    three `launch_*` call sites. Auto-fallback intact: with
    `uccl_use_shared_window == false`, all four are zero/-1 and
    `get_sym_ptr` reverts to nullptr-for-non-self.
  - Kept on the legacy proxy-memcpy path: `hybrid_dispatch.cuh`,
    `hybrid_combine.cuh`, `pp_send_recv.cuh`, `engram_fetch.cuh`. They
    keep ABI compatibility with the ctor's defaulted-arg path.

  Multi-config benchmark (32-sample mean ± stddev for 2n configs,
  2-4 sample for 1n; `--num-tokens=128 --hidden=7168 --num-topk=8
  --num-experts=64`, BF16, do_handle_copy=0, `--allow-hybrid-mode 0`,
  L4 / L40 + L41, two CX7 NICs, NVLink off via `EP_FORCE_NO_NVLINK=1`):

  | Topology | Stage | baseline µs | prototype µs | Δ% |
  | --- | --- | ---: | ---: | ---: |
  | 1n × 2g no-GDR | dispatch | 539±12 | 555±1 | +3% |
  | 1n × 2g no-GDR | combine | 209±2 | 211±2 | +1% |
  | 1n × 2g no-GDR | reduced combine | 642±6 | 652±18 | +2% |
  | **1n × 4g** no-GDR | **dispatch** | **1200±2** | **1080±21** | **−10%** |
  | 1n × 4g no-GDR | expanded dispatch | 1186±1 | 1081±17 | −9% |
  | 1n × 4g no-GDR | cached dispatch | 1040±6 | 1102±7 | +6% |
  | 1n × 4g no-GDR | combine | 464±4 | 471±11 | +2% |
  | 2n × 1g no-GDR | (all stages) | identical | identical | ±1% |
  | 2n × 1g GDR | (all stages) | identical | identical | ±1% |
  | **2n × 4g** no-GDR | **dispatch** | **2822±83** | **2940±173** | **+4%** |
  | 2n × 4g no-GDR | combine | 1002±78 | 1071±55 | +7% |
  | 2n × 4g no-GDR | reduced combine | 1208±101 | 1288±31 | +7% |
  | 2n × 4g GDR | dispatch | 2792±198 | 2878±106 | +3% |
  | 2n × 4g GDR | combine | 1022±117 | 1047±59 | +3% |

  Why the 2n × 4g regression: the prototype changes *where* the GPU
  writes intra-node payload, not *how much* PCIe traffic it generates.
  Both the baseline and prototype consume the same 14 336 B of PCIe
  upstream per intra-node token (sender writes its own slice in the
  baseline, peer slice in the prototype) plus the same 14 336 B of
  PCIe downstream when the receiver loads. The host-DDR memcpy that
  the proxy used to do was overlapped with GPU work and ran at ~30
  GB/s; eliminating it freed CPU but did not free PCIe.

  What did regress on 2n × 4g: dispatch warps that have *both*
  intra-node and cross-node lanes now run a serial per-lane TMA-store
  loop on SM89 (one `tma_store_1d_warp` per active intra-node peer)
  instead of the previous "one local store + delegate everything to
  proxy" pattern. With ≤ 8 active dedup'd peers this adds up to
  ~5 µs/token of additional GPU store work on the dispatch hot path,
  while contending for the same PCIe upstream lane that the NIC
  concurrently uses to read RDMA payloads on the sender side. On 1n
  configs the NIC is idle so the trade pays off; on 2n × 4g it does
  not.

  Decision: keep the prototype in the working tree as a reference for
  future kernel-level work (e.g. moving to a per-peer bulk-WRITE
  layout or gating on `num_nodes == 1`). Do not commit until either
  (a) the dispatch hot path is restructured to avoid the per-lane
  serial loop, or (b) the path is gated to fire only when
  `num_nodes == 1`.

## Validation checklist

After UCCL transport, JIT-visible header, or barrier layout changes:

```bash
make -j SM=89 PYTHON=$PYTHON_BIN
```

Then rerun the 2n × 1g benchmark in both GDR and no-GDR modes (the smallest
multi-node configuration covering both window types). Keep test wrapper
timeouts at 15 seconds; if the JIT cache is cold, rerun with the same
`EP_JIT_CACHE_DIR` after warmup.

For changes that may affect scaling characteristics (e.g. anything that
shifts work between the kernel hot path and the proxy thread, or alters
PCIe traffic patterns), run the full six-config matrix and compare against
a clean-tree baseline:

| Config | nodes × GPUs | Mode | Why |
| --- | --- | --- | --- |
| 1n × 2g no-GDR | 1 × 2 | shared host window | small intra-node sanity |
| 1n × 4g no-GDR | 1 × 4 | shared host window | dense intra-node, no NIC |
| 2n × 1g no-GDR | 2 × 1 | host window per node | pure cross-node, no shared window |
| 2n × 1g GDR | 2 × 1 | GPU window per node | pure cross-node GDR |
| 2n × 4g no-GDR | 2 × 4 | shared host + RDMA | mixed traffic, NIC contention |
| 2n × 4g GDR | 2 × 4 | shared host + RDMA GDR | mixed + GDR contention |

Aim for ≥ 8 samples per cell on the 2n × 4g configs (run-to-run stddev
is typically 3–7%); 2 samples are usually enough on the others.

A reusable harness lives at `/tmp/run_all_configs.sh` (created during
the intra-node-direct prototype review) and writes per-config logs to
`$EP_DIR/.bench/<label>/`. Pair with `/tmp/parse_bench2.py` for an
aggregated comparison table.
