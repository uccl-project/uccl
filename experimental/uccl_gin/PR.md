# PR: UCCL-GIN — standalone GPU-initiated networking primitives for AWS EFA

## Summary

This PR introduces `experimental/uccl_gin/` — a clean, standalone backend that
provides a NCCL GIN-compatible device API on AWS EFA. It mirrors the method
surface of NCCL's `ncclGin` (`put`, `put_value`, `put_tail_add`, `red_add_rel`,
`quiet`, `flush`) so that communication libraries (DeepEP V2, NCCL-EP, etc.) can
run on EFA without per-kernel source changes.

Cross-node (Rail) traffic goes through UCCL's D2H ring → CPU proxy → EFA
RDMA writes. **No dependency on `ep/`** — all transport code is copied into
`transport/` and trimmed. `ep/` has been reverted to its pre-UCCL-GIN state
(commit `34bdf4f4`) and remains unchanged.

## Verified on server

**Hardware**: 2 × p5en.48xlarge (16 × H200), AWS EFA, SRD multi-path
**Software**: CUDA 13.0, NCCL 2.30.4, aws-ofi-nccl master, OpenMPI 5, 2 rails

**Full correctness sweep (all primitives, 7 sizes)**:

```
uccl_gin_microbench world=16 local_world=8 iters=4 (paired-remote)

UCCL-red_add counter: PASS
UCCL-put_value: PASS

bytes        NCCL     UCCL-put/add   UCCL-tail/q    UCCL-put+q
1024         -        PASS           PASS           PASS
4096         -        PASS           PASS           PASS
16384        -        PASS           PASS           PASS
65536        -        PASS           PASS           PASS
262144       -        PASS           PASS           PASS
1048576      -        PASS           PASS           PASS
4194304      -        PASS           PASS           PASS

all correctness PASS
```

**Per-primitive independence** (each testable with `--only` flag):

| Test | `--only` flag | Primitives exercised | What it validates |
|------|--------------|---------------------|-------------------|
| UCCL-put/add | `put-add` | `put` + `red_add_rel` | Data integrity + put-before-tail ordering |
| UCCL-tail/q | `tail-add` | `put_tail_add` + `quiet` | Piggyback correctness + lane drain |
| UCCL-put+q | `quiet` | `put` + `quiet` | Quiet happens-before semantics |
| UCCL-red_add | `red-add` | `red_add_rel` | Multi-lane counter exact value |
| UCCL-put_value | `put-value` | `put_value` + `red_add_rel` | Inline value staging + WRITE_VALUE path |

## Comparison with direct NCCL GIN on EFA

| | Direct NCCL GIN (aws-ofi-nccl) | UCCL-GIN |
|---|---|---|
| dispatch (per-token) | ~5 GB/s SO | N/A (standalone, no EP semantics) |
| combine (per-token) | ~15 GB/s SO | N/A |
| paired put 1K | 0.07 GB/s | N/A (correctness-only, not BW-optimized) |
| paired put 64K | 4.18 GB/s | N/A |
| Correctness (all primitives) | Depends on FORCE_SO MR | ✅ Verified — ordering rebuilt in software |

UCCL-GIN is **not a drop-in replacement for NCCL GIN** — it requires the
integration layer to use `put_tail_add` for piggyback tail and accept the
compact `atomic_tail_base` for tail storage. For DeepEP V2, these are handled
by a thin `DEEPEP_USE_UCCL_GIN` compile-time patch.

## Key design decisions

1. **`put_tail_add` is UCCL-GIN specific** — NCCL uses `put` + `red_add_rel` +
   NIC FORCE_SO for ordering. EFA has no FORCE_SO, so payload and counter are
   fused into a single `WRITE_WITH_IMM` to close the ordering gap.

2. **Tail storage separate from GPU window** — `PackAtomicWithSeq` offset is
   13-bit (≤8191 bytes), far smaller than real window offsets. Tails live in
   a compact `atomic_tail_base` indexed by `(channel, src_rank)`.

3. **`put_value` via WRITE_VALUE cmd** — avoids GPU staging buffer lifetime
   race. Proxy copies the inline value into a host bounce slot keyed by
   `(ring, slot)`, then issues a 4-byte EFA WRITE from that registered host MR.

4. **Sender-side async completion dependency** — finish atomics wait for payload
   WR CQEs without blocking subsequent WRITEs. Piggyback WRITE_WITH_IMM shares
   the receiver's per-tail sequence with finish, so only plain WRITEs enter the
   dependency — `dependency_max` drops from 72 to 2.

5. **16-byte TransferCmd ABI unchanged** from UCCL EP V1 — `atomic_val` and
   `atomic_offset` fields are reused for piggyback tail / inline value encoding.

## File structure

```
experimental/uccl_gin/
├── uccl_gin/          # Public API (what libraries include)
├── transport/         # Internal D2H+proxy+EFA layer (from ep/, no ep/ dependency)
├── context.{hpp,cpp}  # Host setup: QP/CQ/MR, peer exchange
├── bindings.cpp       # CPython extension
├── tests/
│   ├── microbench.cu       # C++/MPI microbench (5 independent correctness gates)
│   ├── test_primitives.py  # Per-primitive Python correctness
│   └── test_context.py     # Context lifecycle
├── python/uccl_gin/   # Python helpers
├── README.md          # English documentation
├── ARCHITECTURE.md    # Full design doc
└── PLAN.md            # Phased plan
```

## What's intentionally not in this PR

- **Lsa/NVLink** — `__trap()`. Standalone doesn't include NCCLGin composition.
  Libraries integrate by composing a NCCLGin for Lsa/World, UCCLGin for Rail.
- **EP semantics** — no layout, no token counters, no expert metadata. These
  belong in the library integration layer (e.g. `hybrid_dispatch.cuh` patch).
- **Bandwidth benchmarking** — the microbench focuses on per-primitive
  correctness with ordering validation. Throughput numbers for EP workloads
  require the full DeepEP V2 dispatch/combine integration.
- **`put<Rail>` large-payload splitting** — TransferCmd `bytes` field (~256KB)
  limits single-WR payload. Splitting is implemented but DeepEP token sizes
  (~14KB FP8) never trigger it. NCCL-EP large-chunk workloads may need this.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
