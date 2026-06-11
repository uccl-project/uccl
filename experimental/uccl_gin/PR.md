# PR: UCCL-GIN — standalone GPU-initiated networking primitives for AWS EFA

## Summary

This PR introduces `experimental/uccl_gin/` — a clean, standalone backend that
provides a NCCL GIN-compatible device API on AWS EFA. It mirrors the method
surface of NCCL's `ncclGin` (`put`, `put_value`, `red_add_rel`, `quiet`) so
that communication libraries (DeepEP V2, NCCL-EP, etc.) can run on EFA without
per-kernel source changes.

Cross-node (Rail) traffic goes through UCCL's D2H ring → CPU proxy → EFA
RDMA writes. Intentional scope: **no dependency on `ep/`**, no DeepEP
coupling. All transport code is copied into `transport/` and trimmed.

## Key design points

- **`uccl_gin::UCCLGin`** — template struct with the same method signatures as
  `deep_ep::elastic::handle::NCCLGin`. Rail branch uses UCCL transport; Lsa is
  stubbed (composed NCCLGin at integration time).

- **Piggyback tail** (`put_tail_add`) — payload and counter delta in one
  `WRITE_WITH_IMM`, closing the EFA ordering gap that NCCL solves with FORCE_SO.

- **Sender-side async completion dependency** — finish atomics wait for payload
  WR CQEs without blocking subsequent WRITEs.

- **All primitives independently tested** — C++ MPI microbench with per-primitive
  correctness gates (data integrity + ordering), verified on 2×p5en.48xlarge
  (16 GPUs).

## File structure

```
experimental/uccl_gin/
├── uccl_gin/          # Public API (what libraries include)
├── transport/         # Internal D2H+proxy+EFA layer (from ep/, no ep/ dependency)
├── context.{hpp,cpp}  # Host setup: QP/CQ/MR, peer exchange
├── bindings.cpp       # CPython extension
├── tests/
│   ├── microbench.cu       # C++/MPI microbench
│   ├── test_primitives.py  # Per-primitive Python correctness
│   └── test_context.py     # Context lifecycle
├── python/uccl_gin/   # Python helpers
├── ARCHITECTURE.md    # Full design doc
├── PLAN.md            # Phased plan
└── README.md
```

## Verified on server

2 × p5en.48xlarge (16×H200), AWS EFA, CUDA 13.0, aws-ofi-nccl master,
2 rails. All correctness gates pass (3 sizes: 1K/4K/64K):

```
UCCL-red_add counter: PASS
UCCL-put_value: PASS
UCCL-put/add (put+tail):   PASS
UCCL-tail/q (piggyback):   PASS
UCCL-put+q (put+quiet):    PASS
all correctness PASS
```

## What's intentionally not in this PR

- Lsa/NVLink — `__trap()`. Standalone doesn't include NCCLGin composition.
- EP semantics — no layout, no token counters, no expert metadata. These belong
  in the library integration layer.

## Why a new directory

`ep/` is the original UCCL EP V1 path and must keep running unchanged. The new
code is intentionally outside `ep/` with zero include dependency on it. All
transport code was copied then trimmed (dead `dispatch_recv_data_offset` removed,
`ep_configs.cuh` deleted, etc.).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
