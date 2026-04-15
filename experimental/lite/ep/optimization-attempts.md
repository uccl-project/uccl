# DeepEP-Lite Optimization Attempts

Low-latency mode on L4 GPUs (sm_89, PCIe Gen4 x16, no NVLink, no GDR).

Baseline: Phase 7 shared-memory intra-node (1n×2g 128×2048 = **8.98 GB/s**).

## Committed Fix

| # | Change | Result |
|---|--------|--------|
| 9 | Remove redundant `__threadfence_system()` before D2H ring buffer puts | **+9.1%** (8.98 → 9.81 GB/s) |

`ring_buffer.cuh::atomic_set_and_commit()` already contains `__threadfence_system()`
(line 439). The explicit fence in `internode_ll.cu` before each
`nvshmemi_ibgda_put_nbi_warp()` call was redundant — `membar.sys` flushes ALL
pending writes from the SM, so the ring buffer's internal fence already covers
the preceding data writes. This is a bug in the original DeepEP code, invisible
on H100 (NVLink latency hides the ~1 μs fence cost) but measurable on L4 PCIe.

## Failed Optimization Attempts

19 ideas attempted exhaustively. Only #9 above was effective.
Plus 4 additional attempts (#20–#23), none effective.

| # | Idea | Status | Result |
|---|------|--------|--------|
| 1 | Dual buffer + PER_EXPERT_BATCHING | Tried, reverted | −3% to +7%. Grid sync overhead (45 μs) cancels fence savings (1.5 μs). |
| 2 | Split send/recv kernels (hook=True) | Tested | Fused (349 μs) faster than split (430 μs). Launch overhead > grid sync. |
| 3 | Write-Combined (WC) host memory | Tried, reverted | 8.90 vs 8.91 GB/s = zero effect. Coalesced writes already generate efficient PCIe TLPs. |
| 4 | Combine recv prefetch | Tried, reverted | 108 vs 101 μs = 7% worse. Register pressure from `int4[9]` reduces occupancy. |
| 5 | Proxy memcpy coalescing | Analyzed | Dead end: source addresses scattered (different token_idx per memcpy). |
| 6 | Cross-destination WR chaining | Analyzed | Dead end: different destinations use different QPs; `ibv_post_send` is per-QP. |
| 7 | L2 prefetch / cache hints for recv | Analyzed | Host memory mapped as uncacheable in GPU PTEs on sm_89. |
| 8 | FP8 compute reduction | Analyzed | Already uses `__nv_cvt_float2_to_fp8x2` hardware intrinsic. |
| 10 | Fence-batched D2H replay | Tried | Ring buffer livelock — can't re-acquire already committed slots. |
| 11 | Non-cooperative send kernel | Tested | Performance-neutral. Send uses only 8/58 SMs — not SM-bound. |
| 12 | Pipeline overlap (return_recv_hook) | Validated (app-level) | +11–13.5%. Python-level reordering: dispatch_send → attention → dispatch_recv. Not a library change. |
| 13 | Copy engine DMA for host transfer | Validated | Only 10% advantage at 256 KB vs SM stores. Cooperative launch blocks CE concurrency. |
| 14 | Gate speculation agreement | Validated | 93.5% exact match. Enables idea 12 but not a perf change itself. |
| 15 | PER_EXPERT_BATCHING (no grid_sync) | Tested | −42%. Batching defers ALL D2H → breaks GPU-proxy pipelining. |
| 16 | PER_EXPERT_BATCHING (with grid_sync) | Tested | −43%. Same root cause as #15. |
| 17 | PER_EXPERT_BATCHING (with GDR) | Tested | −42%. Not a host-memory issue — pipelining is the problem. |
| 18 | Fence coalescing (deferred commit) | Tested | DS 2× slower (335 vs 170 μs). Deferred D2H function generates worse machine code. Barriers themselves have zero overhead. |
| 19 | Outer head/tail loop removal | Tested | 9.86 vs 9.84 GB/s = no effect. Outer loop reads are L2-cached. |
| 20 | CUDA IPC P2P for intra-node (NVL_PEERS=4) | Tested, reverted | **−80%** (7.80 → 1.54 GB/s 4-GPU). GPU kernel individual PCIe stores are 5× slower than CPU proxy batch memcpy. |
| 21 | VRAM staging (bulk D2H before RDMA) | Tested, reverted | **−35%** (9.79 → 6.41 GB/s 2n×1g). Breaks GPU-proxy pipelining (same as #15). |
| 22 | Fence batching (amortize threadfence_system) | Tested | **−2 to −5%**. 1.07 μs fence cost hidden by 4 μs RDMA latency. |
| 23 | Proxy multi-SGE RDMA coalescing | Tested, reverted | **0%** (10.16 vs 10.13 GB/s = noise). Merged consecutive same-expert WRITEs into fewer WRs with up to 16 sge entries. Sorting + merging overhead cancels WR reduction benefit. |

## Key Insights

1. **Per-token D2H commands enable GPU-proxy pipelining**: The proxy starts
   RDMA while the GPU processes the next token. ANY approach that defers D2H
   commands (batching, expert-level grouping) serializes GPU and proxy work,
   causing 42%+ regression.

2. **`__threadfence_system()` is irreducible on PCIe**: It serves dual purpose —
   memory ordering AND PCIe write buffer flushing. Cannot batch, skip, or
   replace with cheaper primitives.

3. **The system is PCIe-bandwidth-limited**: FP8 cast uses HW intrinsics,
   PCIe writes are coalesced, ring buffer atomics are pipelined. The 60%
   "overhead" in D+C total is fundamental to the MoE dispatch/combine protocol
   (FP8 compute, atomic signaling, command queue latency, grid sync).

4. **GPU kernel PCIe P2P writes are 5× slower than CPU proxy memcpy**: Each
   GPU `st.global` to remote GPU memory generates an individual PCIe TLP with
   high per-access latency (~2 μs). CPU bulk memcpy coalesces into large DMA
   bursts. NVLink hides this (450 GB/s, ~100 ns latency), PCIe does not.

5. **Proxy WR count reduction (multi-SGE coalescing) has zero effect**: Merging
   256 WRs into ~16 coalesced WRs with multiple scatter-gather entries does not
   help. The NIC's WQE processing rate is not the bottleneck — the proxy's
   per-command polling + GPU-proxy signaling loop is.

## Performance Summary

| Setup | D+C BW (GB/s) | D BW | C BW | D+C Latency |
|-------|:-------------:|:----:|:----:|:-----------:|
| 1n×2g 128×2048 | 9.81 | 6.23 | 9.97 | 318 μs |
| 1n×4g 128×2048 | 7.82 | 5.82 | 7.09 | 400 μs |
| 2n×1g 128×2048 | 10.13 | 6.42 | 10.43 | 308 μs |
| 2n×4g 128×2048 | 4.45 | 3.70 | 4.89 | 702 μs |
