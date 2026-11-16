# Fresh two-node AllGather latency: 1 KiB–1 MiB

Measured on the two four-L4 hosts, 2026-11-16. [The spike recheck](retest/README.md) replaces the complete default/SM latency
series after protocol diagnostics. The README AllGather **latency
panel** now uses this fresh sweep; its **1 MiB–1 GiB throughput panel retains
the prior measurements**, identified separately in `figure_inputs.csv`.
The 1-MiB point in each panel therefore comes from its own measurement window.
Historical throughput is not validation of the refreshed source.

- Hardware: `mibura-sky-test-02` / `mibura-sky-test-01`, four NVIDIA L4/host,
  two NUMA groups/host and two IB rails. GPUs 0/1 and 2/3 form the local groups.
- FP32, eight GPUs, four MPI processes (two GPUs/process), `--bind-to none`.
  Bytes mean complete AG output per GPU; each GPU contributes one eighth.
- Fresh builds from the current working-tree UCCL source and current NCCL
  harness. `measured_source/` preserves exact source, including the shared
  host helper; no collective implementation was changed for this refresh.
- UCCL Lite keeps the plotted fixed shared-ready host-router configuration:
  256-KiB chunks, SM download, DMA upload, own-input GPU bypass, early consume,
  shared local exchange, readiness progress and MPI/UCX remote multicast.
  The controller applies its NUMA placement; NCCL uses its normal affinity.
- Baselines are fresh NCCL 2.28.7 runs. `prior` is true default; latency `sm`
  now fixes SIMPLE with CE disabled at every size to avoid the observed LL
  slowdown. `ce` retains the previous fixed preset. No per-size best variant
  is selected. Exact settings and protocol diagnostics are archived.
- Timing is slowest-process host submission through GPU completion, on both
  backends; setup, generation, full-output checks and MPI max reduction are
  outside timing. Every warmup and timed invocation is fully checked.
- Eleven powers-of-two sizes, 1,024–1,048,576 bytes; **three fresh jobs/series,
  20 warmups + 50 timed calls/size/job** for UCCL/CE; the refreshed default/SM
  curves use **100 warmups + 200 calls**. Reverse job order in round 2.
  Points are medians of three run means; min/max run means are retained.
  All 16,500 plotted timed calls pass full correctness checks. No best variant per size.

The plot uses the existing colors, lines, axes and panel styling. Raw samples,
commands, source/binary hashes, environment and all run-level ranges are kept.

## Reproduce

From the Lite root, with the same external CUDA/NCCL/MPI installations:

```bash
make -C benchmarks/data/ag_latency_2x4l4_20261116 -j4
# Stage build/ag_latency_2x4l4_20261116/ executables at the same path on node 1.
python3 benchmarks/data/ag_latency_2x4l4_20261116/run.py \
  --out build/ag_latency_2x4l4_repeat
# Recheck the full default/SM curves (100 warmups, 200 samples):
python3 benchmarks/data/ag_latency_2x4l4_20261116/retest/refresh.py \
  --out build/ag_latency_baselines_repeat
# Render archived latency + unchanged archived throughput:
python3 benchmarks/data/ag_latency_2x4l4_20261116/plot.py
```

The Makefile builds the measured source snapshot, keeping binaries in `build/`.
The runner refuses to overwrite an existing job; use a fresh output directory.
The underlying code is explicit eager experimental host routing; no automatic
backend policy is changed. The previous 1–8-MiB figure and its full dataset are
preserved in `nccl_tuned_2x4l4_20261116/`.
