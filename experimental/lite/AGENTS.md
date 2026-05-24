# AGENTS.md — `experimental/lite`

**UCCL-Lite** is a family of communication libraries targeting **consumer- /
low-end-GPU testbeds** — environments where NVLink and GPUDirect RDMA are
typically unavailable (e.g. GeForce RTX 4090/5090, NVIDIA L4, GB10 UM
devices). The top-level `experimental/lite` directory groups two
independently-built sub-projects:

```
experimental/lite/
├── lite-collective/   # Generic collectives (allreduce/allgather/…), MSCCLPP-derived
├── lite-ep/           # Expert-parallel dispatch/combine, DeepEPv2 + UCCL-EP based
└── AGENTS.md          # this file
```

Both sub-projects share the same target hardware envelope but are otherwise
**independent codebases with their own build systems, runtimes, and
benchmarks**. Treat them as siblings, not as layers of one stack.

---

## `lite-collective/` — MSCCLPP-derived generic collectives

- Upstream lineage: **MSCCLPP**, restructured into clearer layers.
- Layout: `core/` (runtime, IB, proxy, channels), `collective/` (algorithm
  selector + executor + allgather/allreduce variants), `nccl/` (NCCL API
  compatibility layer), `doc/`, `thirdparty/`, `scripts/`, top-level
  `Makefile`.
- Data path: **SM-free** — `cudaMemcpyAsync` + IB verbs on a CPU worker.
  All inter-node traffic stages through pinned host memory. See
  `lite-collective/doc/p2p-design.md`.
- Build:
  ```bash
  make -C experimental/lite/lite-collective              # libmint + libmint_collective
  make -C experimental/lite/lite-collective/nccl         # NCCL compat layer
  ```
- Smoke benchmark:
  ```bash
  experimental/lite/lite-collective/build/single_node_allreduce_runner
  ```
- See `lite-collective/AGENTS.md` for layered modification rules, upstream
  naming conventions (`MSCCLPP_*`, `mscclpp::`, `libmint*`), and validation
  matrix.

## `lite-ep/` — DeepEPv2 + UCCL-EP expert-parallel transport

- Upstream lineage: **DeepEPv2** (`ElasticBuffer`) with **UCCL-EP** wired in
  as the EP data transport (`EP_USE_UCCL_PROXY=1`). NCCL is still loaded for
  bootstrap, but NCCL GIN is not on the EP payload path.
- Layout: `csrc/` (kernels, JIT, `uccl/` transport, `elastic/`), `deep_ep/`
  (Python package: `buffers/`, `include/`, `utils/`), `tests/` (elastic
  benchmarks, GPU↔CPU micro-benches), `doc/`, top-level `Makefile` +
  `setup.py`.
- Target runtime modes: no-GDR / no-NVLink is the **primary** path on L4 +
  ConnectX testbeds; GDR is supported but secondary.
- Build:
  ```bash
  conda activate uccl
  make -C experimental/lite/lite-ep -j SM=89 PYTHON=python
  ```
- Smoke benchmark (single node, 2 procs):
  ```bash
  cd experimental/lite/lite-ep
  PYTHONPATH=$PWD EP_USE_UCCL_PROXY=1 EP_FORCE_NO_NVLINK=1 \
    timeout 120s python tests/elastic/test_ep.py \
      --num-processes 2 --num-tokens=128 --hidden=7168 \
      --num-topk=8 --num-experts=64 --test-first-only --skip-check \
      --do-handle-copy-modes=0 --expert-alignment-modes=128 \
      --fp8-dispatch-modes=0 --num-bench-tests=5
  ```
- See `lite-ep/AGENTS.md` for the full environment-variable matrix
  (GDR/no-GDR, UCCL proxy, JIT cache), correctness invariants (barrier-tag
  indexing, `sync_uccl_peers`, direct-combine ordering), and the
  `EP_UCCL_PROXY_TRANSPORT_VERSION` cadence.

---

## Working in `experimental/lite/`

- **Do not** introduce cross-imports between `lite-collective/` and
  `lite-ep/`. Shared concepts (host staging, proxy semantics) are
  intentionally re-implemented in each tree to keep upstream diffs small.
- When a change is scoped to one sub-project, run only that sub-project's
  build + smoke benchmark. The two trees do not share a top-level Makefile.
- Vendored code under `lite-collective/thirdparty/` and JIT/kernel sources
  under `lite-ep/csrc/` originate upstream — preserve naming and macros.
- Build outputs (`lite-collective/build/`, `lite-ep/build/`,
  `lite-ep/.jit-cache/`, `lite-ep/.bench/`) are generated; never commit or
  hand-edit them.

## External dependencies (host paths on this testbed)

- NCCL: `/home/yangz/nfs/zhongjie/nccl`
- nccl-tests: `/home/yangz/nfs/zhongjie/nccl-tests`
- DeepEP upstream: `uccl/thirdparty/DeepEP` (sibling to this directory)
