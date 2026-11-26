# UCCL Lite

This directory contains standalone CUDA collectives, fusion kernels, benchmarks,
application integrations, and design notes for PCIe GPU systems. MSCCL++
supplies the peer-path runtime; the optional CPU-router module uses its own
CUDA/NUMA packet pipeline. Keep changes scoped to this directory unless another
UCCL component actually needs updating.

- Keep application integrations and serving hooks under `integration/`, separate
  from the standalone kernels and CPU-router core. Keep Torch extensions,
  checkpoints, vendored dependency trees, and compiled artifacts out of source
  directories.
- Preserve MSCCL++, CUDA, NCCL, and CUTLASS API names. Project-owned names use
  `uccl_lite` or `UCCL Lite`.
- `src/collectives/` contains standalone collective algorithms and their harnesses.
  `include/uccl_lite/` contains reusable device/fusion headers.
  `benchmarks/` contains fusion harnesses, probes, and reference measurements.
- CPU-router is optional and independently linked. Keep its core free of NCCL,
  MSCCL++, and application hooks. The caller supplies the fallback callback.
- Never enable a routing backend based only on GPU count. Match dtype, exact
  size, placement, topology/software identity, and validated timing evidence.
  Keep fallback free of router buffer/worker initialization and preserve the
  explicit prepare boundary.
- Run `router-policy-check`, dispatcher checks, and appropriate eager/graph
  correctness tests for router changes. Record local calibration separately
  from the historical eight-GPU performance figures.
- Use the Makefile. Keep generated binaries and dependency files in `build/`.
- Match the repository's clang-format 14 configuration.
- Build affected CUDA targets. For collective changes, run correctness checks on
  the documented four-GPU topology; multi-node changes require two four-GPU hosts.
  State clearly when validation is limited to compilation.
- Benchmark reports must identify hardware, dtype, baseline, timing scope,
  repetitions, and whether a curve selects the best variant per size.
- Treat historical results as measurements of their recorded configurations,
  not validation of a subsequent source change. Preserve negative results and
  explicit experimental status.
