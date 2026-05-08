# UCCL-EP intra-node + inter-node gap vs NCCL — open work

Latest baseline (commit `d74a9b5b`, no EP path changes), 4 rounds A/B
(warmup discard) on L4 PCIe-only / CX7 ×2 / dual-channel DDR4-3200,
no-GDR, validation shape `128 tok × 7168 hid × top-8`:

| Topology | Stage | EP latency | NCCL a2a ref | Gap |
| --- | --- | ---: | ---: | ---: |
| 1n × 4g | dispatch | 1183 µs | 868 µs |  ~1.4× |
| 1n × 4g | combine  |  520 µs | (≈ a2a) |  — |
| 2n × 4g | dispatch | 2148 µs | 998 µs |  ~2.2× |
| 2n × 4g | combine  | 1007 µs | (≈ a2a) |  — |

## Gap source breakdown (revised after empirical test)

| Gap source | 1n × 4g | 2n × 4g |
| --- | ---: | ---: |
| ~~Extra host DDR R + W vs NCCL's 1× GPU↔host pass~~ **REFUTED — see below** | 0 µs | 0 µs |
| GPU kernel inherent cost (SHM write fan-out + setup + barriers) | ~1150 µs | ~1150 µs |
| Per-WR posting × ~400 × 1.5 µs (mlx5 doorbell + spinlock) | — | ~600 µs |
| CPU-proxy path latency vs GPU-driving NIC | — | ~150 µs |
| Channel-finish PUT_VALUEs | — | ~100 µs |

### Why "extra host DDR R + W" turned out to cost 0 µs

Initial analysis estimated the proxy memcpy (sender slice → receiver
slice) added ~300 µs of DDR work to the critical path on 1n × 4g.
Empirical test (6-round interleaved A/B with `EP_UCCL_INTRANODE_DIRECT`
toggle, std ≈ 30 µs):

| Mode | Dispatch | Combine |
| --- | ---: | ---: |
| Proxy memcpy (default) | 1152 ± 38 µs | 506 ± 24 µs |
| Direct sender-to-peer-slice write | 1152 ± 31 µs | 493 ± 27 µs |
| Δ | **0 µs (0%)** | -13 µs (-2.6%) |

So the proxy memcpy is **fully overlapped with concurrent GPU PCIe and
sender/receiver activity** — it's not on the critical path. The DDR
round-trip estimate ignored PCIe/DDR pipelining.

This rules out **all intra-node memcpy / DDR-bandwidth optimizations**
as a source of measurable improvement on this hardware:

- ~~NT-store memcpy~~: regresses 2n × 4g, 0 µs win on 1n × 4g.
- ~~THP madvise + page pre-touch~~: 0 µs win, kept opt-in only.
- ~~Sender direct-write to peer slice~~ (eliminates proxy memcpy):
  0 µs win on 1n × 4g (within 30 µs noise band).

**Real intra-node ceiling on 1n × 4g is the GPU dispatch kernel
itself**, not the host-side memcpy. NCCL beats us by ~285 µs because
its GPU-side ring code is leaner (no token-count all-to-all + layout
discovery + per-expert atomic counters).

## Open optimization ideas (ranked by expected payoff, post-refute)

1. **Coalesce per-token RDMA WRITEs into per-peer bulk WRITEs** (2n × 4g
   only).  EP currently posts ~400 WRs/iter on 2n × 4g (one per token).
   Real fix needs the kernel to lay out remote slots contiguously so
   multiple tokens share a single SGE — current layout interleaves
   slots by `(channel, slot, rank)` which makes proxy-side multi-SGE
   merging a no-op (already documented + tested).  Expected payoff:
   up to ~600 µs on 2n × 4g.  Significant kernel-layout change.

2. **Eliminate dispatch setup phase (token-count all-to-all + layout
   discovery)**.  Either run-ahead the count exchange a few iterations
   in advance, or fold it into the previous combine's barrier traffic.
   Expected payoff: ~100-150 µs on every config.  Touches
   `hybrid_dispatch.cuh` non-trivially.

3. **Inspect remaining SM89 single-lane sites in dispatch hot path**.
   The fan-out write loop for ~5.5 MB SHM payload should hit ~16 GB/s
   per GPU (PCIe upstream limit) ≈ 350 µs; if the kernel takes longer
   than that, vectorization is incomplete.  Diagnostic via kineto
   trace of `hybrid_dispatch_impl` then targeted PTX inspection.

## Refuted hypotheses (kept as opt-in env vars or reverted)

- **AVX2 NT-store memcpy + cross-iter prefetch**: regresses 2n × 4g
  combine +64% (cache-snoop loss on receiver-GPU PCIe DMA).  Reverted.
- **THP `madvise(MADV_HUGEPAGE)` + page pre-touch on shared window**:
  no measurable change, within run-to-run noise.  Reverted.
- **Heap-thrash elimination** (recycle proxy `std::vector` scratch
  across calls): no real win in steady-state interleaved A/B.  Reverted.
- **Proxy threads 4 → 8**: previously unstable.  Documented dead end.
- **Intra-node direct host-window write (eliminate proxy memcpy)**:
  0 µs improvement on 1n × 4g end-to-end latency under 6-round
  interleaved A/B (noise std ~30 µs).  The proxy memcpy is fully
  overlapped with concurrent GPU PCIe traffic and is not on the
  critical path.  However, it does eliminate ~22 MB/iter of host DDR
  traffic and frees the proxy thread's CPU time, so the path is
  enabled by default (`EP_UCCL_INTRANODE_DIRECT=1`) for the benefit of
  co-located workloads (e.g. KV cache prefetch, mixed RDMA traffic).
  Set `=0` to fall back to proxy memcpy for diagnostics.

## Validation harness

- `tests/elastic/bench_nccl_a2a.py` — NCCL a2a ref using same
  `legacy_bytes` as EP bench; comparable directly to EP `legacy GB/s`.
- `.bench/run_ab.sh <label> [env=val ...]` — runs 1n × 4g + 2n × 4g
  with given env, dumps logs to `.bench/<label>_{1n4g,2n4g}/`.
- `.bench/run_reps.sh <label> <reps> [env=val ...]` — N reps, drops
  rep 0 as warmup, prints per-rep + mean.
- `.bench/run_interleaved.sh <A_label> <A_envs> <B_label> <B_envs> <rounds>`
  — interleaved A/B, randomizes JIT/thermal effects; the only
  methodology that gives reproducible signal under run-to-run noise of
  ~10%.
