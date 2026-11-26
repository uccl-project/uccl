# PRO6000 reproduction of the integrated branch

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/pro6000_b0ac7b4a_validation) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


Measured commit: `b0ac7b4a442fb49c61a3b0d051e7f7af7480d88b`, verified against
remote branch `codex/uccl-lite-retained-optimizations` before source export.
All 75 exported source/fixture files match this Git commit. Later documentation
commits do not change the measured implementation.

**The gain reproduces.** Two RTX PRO 6000 Blackwell Server Edition GPUs, PIX
topology, driver 595.91.07, CUDA 13.0.88, NCCL 2.29.7. NCCL uses its default
policy; logs verify P2P on every data channel, with no SHM/NET data edges.

| UCCL mode | AllReduce | AllGather | ReduceScatter | All 63 cases, forward / reverse |
|---|---:|---:|---:|---:|
| Forced P2P | 1.5277x | 1.6114x | 1.5815x | 1.5705x / 1.5758x |
| Calibrated joint | 1.5176x | 1.6058x | 1.5607x | 1.5633x / 1.5586x |

Speedup is NCCL/UCCL latency; above one wins. Operation columns are geometric
means over both runs' per-case ratios. Both modes win all 63 cases in both
runs; joint selects peer in all cases. On the 36-case 8 KiB–16 MiB subset,
forced peer scores 1.4923x / 1.5037x and joint scores 1.4944x / 1.4877x.
The earlier isolated joint score was 1.5742x; the integrated joint result is
within 1% of it. These are observed runs, not confidence intervals.

## Protocol and checks

BF16 out-of-place AR/AG/RS; powers of two from 1 KiB through 1 GiB. Size is
complete AR input, complete AG output, complete RS input. Each full sweep has
100 warmups and 5 repeats of 20 timed invocations per case. Latency is the
slowest-rank CUDA event interval including submission gaps; preparation,
initialization and correctness checks are outside timing. Every invocation
checks full output, unchanged input and guards. These are eager collective
results, not inference, CUDA graphs or concurrent-model-compute results.

Execution order: NCCL forward, peer forward, joint forward, joint reverse,
peer reverse, NCCL reverse. Reverse runs reverse both operation and size order.
There are 37,800 accepted primary timed invocations. Fresh independent
calibration uses a separate process and seed, 3 warmups and 3x5 calls per
backend/case; no final measurements are used to select routes.

Build, five host checks, two-GPU router eager/graph checks, joint selftest and
NCCL transport checks all passed. The shared node evaluation lock was held;
live process checks abort this evaluator's child if unrelated GPU work appears.
The initial minimal export omitted the historical policy-test CSV and failed
before any performance measurement. The corrected export includes that exact
tracked fixture; the failed attempt contributes no result. No source tuning,
clock changes, NCCL algorithm overrides or package changes were made.

## Files and offline verification

- `cases.csv`: reconstructed case medians and speedups for peer and joint.
- `summary.json`: operation/range aggregates, route counts and raw-file hashes.
- `measurements.tar.gz`: all six raw sample CSVs, calibration, commands,
  correctness logs, NCCL transport logs, hardware metadata and completion marker.
- `source-provenance.json`: source, executable and NCCL library hashes.
- `config.json`: exact commit, dependency paths and evaluation-lock path.
- `validate.py`: exact remote build/check/calibrate/evaluate runner.
- `analyze.py`: independently reconstructs all medians from raw samples and
  checks printed summaries, sample counts, routes and NCCL transport.
- `checksums.json`: hashes of the evidence package files.

To verify without a GPU, copy this directory to a temporary directory, run
`tar -xzf measurements.tar.gz`, then `python3 analyze.py`. It reconstructs
`cases.csv` and `summary.json` from the raw evidence. Keep the original package
unchanged when comparing hashes.

## Regenerate figures

To regenerate the three README figures from `cases.csv`, run from
`experimental/lite`:

```bash
python3 benchmarks/data/pro6000_b0ac7b4a_validation/plot.py --preview
```

SVGs go to `docs/assets/`; optional PNG/PDF previews go to `build/figures/`.
The plots preserve calibrated joint (peer at every size) and default NCCL P2P,
with the median of the forward/reverse run medians and shaded run ranges.
Latency covers 1 KiB–1 MiB; bus throughput covers 1 MiB–1 GiB, computed from
the central latency as `bytes / time` for AR and `0.5 × bytes / time` for AG/RS
(decimal GB/s). They also load the separately measured
[`../nccl_tuned_pro6000_20261115/cases.csv`](../nccl_tuned_pro6000_20261115/cases.csv)
as a third, tuned-NCCL reference. Its per-size settings were frozen using
independent calibration before final measurement. See the
[tuned-NCCL protocol](../nccl_tuned_pro6000_20261115/README.md) for preparation,
controls and transport evidence. The original two curves' data is unchanged;
regenerating the figures does not perform a new GPU measurement or select
the best final result per size.

The figures also load
[`../nccl_ascending_pro6000_20261115/cases.csv`](../nccl_ascending_pro6000_20261115/cases.csv)
as a fourth purple reference. Both repetitions of that study run sizes in
ascending order on one communicator per process. The green reference retains
its independent forward/reverse validation; these timing scopes are labeled
separately.

## Reproduce on aws-pro6000

The dependency paths in `config.json` refer to the existing installation on
that testbed. Run from the repository root in an exclusive GPU window. This
exports the pinned Git commit, not uncommitted files. The export includes the
policy check's required historical CSV.

```bash
run_dir=$(mktemp -d /home/xinze/dev/pro6000-reproduce.XXXXXX)
cp experimental/lite/benchmarks/data/pro6000_b0ac7b4a_validation/{config.json,validate.py,analyze.py} "$run_dir/"
git archive --format=tar.gz --output="$run_dir/source.tar.gz" \
  b0ac7b4a442fb49c61a3b0d051e7f7af7480d88b -- \
  experimental/lite/Makefile experimental/lite/AGENTS.md \
  experimental/lite/src experimental/lite/include experimental/lite/tests \
  experimental/lite/benchmarks/joint \
  experimental/lite/benchmarks/cpu_router/nccl_identity.hpp \
  experimental/lite/benchmarks/data/primitives_8l4.csv
python3 "$run_dir/validate.py"
python3 "$run_dir/analyze.py"
```

The runner checks the archive hash, reserves the existing shared lock, checks
GPU occupancy, builds from scratch and leaves every log in the new directory.
It clears inherited NCCL/UCCL overrides. On this CUDA installation the libraries
are in `lib`, so its Makefile command adds
`EXTRA_NVCC_FLAGS=-L/opt/pytorch/cuda/lib` and
`CXX="g++ -L/opt/pytorch/cuda/lib"`, with matching `LD_LIBRARY_PATH` for execution.
Use a new directory for every attempt. Different dependency versions or
hardware require recording a new configuration; results need not be identical.
