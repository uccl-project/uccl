# Independently tuned NCCL references on two PRO6000 GPUs

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


This 2026-11-15 study adds tuned NCCL reference curves to the PRO6000 ×2
AllReduce, AllGather and ReduceScatter figures. The original UCCL and
default-NCCL curves remain unchanged. These new NCCL runs use the original
figure's NCCL-only harness, with optional NCCL-managed symmetric windows
added outside timing. This is a separate measurement window, not a fresh
paired UCCL benchmark.

The subsequent [ascending-only study](../nccl_ascending_pro6000_20261115/README.md)
adds a separate purple memcpy reference. This report describes the green
two-order reference; its largest-first failures remain recorded. The
`figures.json` hashes here describe the original three-curve figures committed
at `056b9efc`, before the additional ascending curve.

## Validated results

All 25,200 primary timed invocations passed full-output/input/guard checks:
two tuned runs and two fresh default-control runs over all 63 cases. Separate
selection traces validate all 63 scheduled keys. The reference executable
build passed; hashes confirm its timed/checking functions and NCCL enqueue
code are unchanged from the original harness.

At **1 GiB complete tensor per GPU**, bus bandwidth is:

| Collective | Fresh default NCCL | Tuned NCCL | Bandwidth increase |
| --- | ---: | ---: | ---: |
| AllReduce | 32.00 GB/s | **36.75 GB/s** | 14.8% |
| AllGather | 28.52 GB/s | **36.66 GB/s** | 28.5% |
| ReduceScatter | 31.08 GB/s | **36.77 GB/s** | 18.3% |

The 1-GiB settings are `NCCL_BUFFSIZE=1048576` for all three operations;
AllReduce also uses `NCCL_NTHREADS=128`. AR and AG use NCCL-managed registered
buffers; RS uses ordinary cudaMalloc buffers. Other sizes have their own
independently calibrated settings, recorded in the full schedule.

Across 1 MiB–1 GiB, geometric-mean speedups against fresh default controls
are **1.170× / 1.276× / 1.177×** for AR/AG/RS. Across the complete 1 KiB–1 GiB
grid they are **1.089× / 1.135× / 1.077×**. These are observed comparisons,
not confidence statements. Nominal wins occur at 15/21, 15/21 and 14/21
points respectively. Several small-message choices regress by about 5–6%
in independent validation; those results remain in the curves and CSV.
Even independently repeated default settings fluctuate at the microsecond scale.

The earlier approximately 49–50 GB/s memcpy AllGather results came from
ascending-size runs. Four memcpy configurations stalled in reverse full-grid
calibration, and all eight isolated reverse AR/AG probes timed out. Those
settings are excluded from the final curves. This evidence does not establish
the cause of the stalls or their behavior outside the recorded harness.

[cases.csv](cases.csv) contains every final latency, bus-bandwidth value,
selected configuration and fresh default control. The figures retain the
**original** default-NCCL curve; fresh controls are supplied here for comparing
the tuning itself. [transport.json](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/transport.json) records the actual
per-case selection: P2P channels for kernel collectives, and explicit
copy-engine collective selection at five AllGather sizes.

## Hardware, software and timing

- Two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs on one PCIe Gen5
  x16 switch (`PIX`), Xeon Platinum 8559C, one NUMA node, no NVLink.
- Driver 595.91.07, CUDA 13.0.88 compilation/runtime 13.0,
  NCCL 2.29.7+cuda13.2. No library, clock or system-wide environment changes.
- BF16, out-of-place AR/AG/RS; one process with two GPUs. Sizes are complete
  AR input, complete AG output, complete RS input bytes per GPU.
- Every power of two from 1 KiB to 1 GiB, for all three collectives.
- Slowest-rank eager CUDA events include submission gaps. Allocation,
  registration, communicator setup, input initialization and correctness
  checks are outside timing. This is not graph replay or application latency.
- Final validation uses 100 warmups and five repetitions of twenty timed
  calls per case. Forward/reverse runs use distinct seeds. Every invocation
  checks changing signed inputs, complete outputs, unchanged inputs and guards.
- Each point is the median of the two run medians; each run median is the
  median of its five repeat medians. Shading is the observed run range,
  not a confidence interval.

The CUDA source is the original `b0ac7b4a` NCCL-only harness, with the
archived `source.patch`. Its optional registered mode uses `ncclMemAlloc`,
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`, and matching cleanup.
All ranks use matching offsets, including the guard prefix. Both plain and
registered modes allocate capacity for 1 GiB even when a process tests only
a smaller contiguous size range, matching the original full-sweep capacity.
The source changes only allocation/registration/preparation; the original
collective calls, timing boundaries, generation, and verification remain.
It compiles with `NCCL_REFERENCE_ONLY` and links no UCCL implementation.

## Calibration and frozen selection

The earlier AllGather exploration tested P2P chunk size, collective buffer
size, channel count, thread count, peer-read mode, the P2P memcpy transport,
and the zero-CTA collective path. Its ascending-size results are preserved
in [allgather_exploration.tar.gz](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/allgather_exploration.tar.gz). They are not
substituted for the matched final measurements in these figures.

The matched study screens 23 configurations, including default. Each candidate
gets forward/reverse full-grid calibration with three warmups and 3×3 timed
calls per case. Extra default controls bracket each sweep; candidate order
is shuffled. `NCCL_*` and `UCCL_*` overrides are cleared before each process,
then only the recorded candidate settings are applied. The shared evaluation
lock is held, and GPU occupancy is checked before every process.

For each operation and exact size, a candidate must beat its bracketed
default control by at least 5% in **both** calibration orders. Among eligible
candidates, selection minimizes geometric-mean calibration latency. Otherwise
default remains selected. The resulting [schedule.json](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/schedule.json) is
frozen, hashed and timestamped in [selection.json](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/selection.json) before
final evaluation, which uses different seeds.

The tuned curve is therefore a **per-size calibrated benchmark schedule**.
It is not one global environment setting, an online NCCL dispatcher, or a
best-of-final-measurements envelope. Final regressions are retained. Adjacent
sizes with identical settings are grouped into one process; the reverse run
reverses both group and size order. Final fresh-default sweeps bracket the
tuned forward/reverse evaluation.

### Failed candidates remain excluded where unvalidated

The reverse all-operation calibration begins at large ReduceScatter sizes.
Four `NCCL_P2P_USE_CUDA_MEMCPY=1` settings stalled there after passing
ascending-size tests. Evaluator-owned stalled children were terminated after
a 90-second screening threshold; exact elapsed times and commands are in
the archived `*_abort.json` files. A successful ascending run is not treated
as sufficient validation.

To avoid confusing an RS failure with AR/AG behavior, failed configurations
also receive independent reverse-order AR and AG probes, with a 30-second
timeout. A configuration is eligible only for operation/size keys with
successful evidence in both directions. Failures and timeouts remain in
the raw archive; none contribute a winning timing.

`NCCL_P2P_USE_CUDA_MEMCPY=1` is the P2P transport's copy-engine option; it
retains the GPU collective kernel and CPU proxy progress. It differs from
`NCCL_CTA_POLICY=2`, which requests a zero-CTA collective and requires
registered symmetric windows. A requested knob is not proof of use: separate
selection traces record actual collective protocols and transport edges.

## Reproduction and review

- [build_probe.py](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/build_probe.py) asserts the original source hash and builds
  the optional-registration NCCL reference using the repository Makefile.
- [run.py](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/run.py) records commands, settings, timestamps, correctness status
  and hashes for calibration, supplementary probes, evaluation and traces.
- [select.py](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/select.py) produces the frozen schedule exclusively from
  calibration evidence; it does not read final measurements.
- [analyze.py](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/analyze.py) reconstructs all final medians from individual
  samples, validates counts, correctness and settings against the frozen
  schedule, and emits [cases.csv](cases.csv) and [summary.json](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/summary.json).
- The figure script is
  [../pro6000_b0ac7b4a_validation/plot.py](../pro6000_b0ac7b4a_validation/plot.py).
  Run it with `--preview` from `experimental/lite`; SVGs go to `docs/assets/`
  and PNG/PDF previews go to `build/figures/`.
- [check_traces.py](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/check_traces.py) validates actual selection at every
  scheduled key. Raw harness labels such as `nccl_default` are generic;
  the run manifests and frozen schedule identify each actual configuration.

Raw calibration, failure, supplementary, final and trace files are packaged
in [measurements.tar.gz](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_pro6000_20261115/measurements.tar.gz). For offline verification, copy
this directory to a temporary location, extract that archive there, and run
`python3 analyze.py` and `python3 check_traces.py`. Do not rerun selection on
the original package, since that would replace the frozen selection timestamp.
`*_runs.json` manifests remain directly reviewable outside the archive.
[checksums.json](checksums.json) records the packaged evidence hashes.

Remote generated sources, binaries and dependencies are kept under
`/home/xinze/dev/pro6000-integrated-b0ac7b4a-20261115-022114-retry1/source/experimental/lite/build/nccl_tuned_20261115/`.
No production algorithm was changed. This validates the NCCL reference on
two PRO6000 GPUs only; it does not validate four/eight-GPU UCCL algorithms,
graph mode or serving performance.
