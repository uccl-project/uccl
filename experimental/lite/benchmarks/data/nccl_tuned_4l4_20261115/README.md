# Testbed 3: independently tuned NCCL CE and SM references

This November 15, 2026 study adds `NCCL CE (tuned)` and `NCCL SM (tuned)`
to the three four-L4 FP32 collective figures. The existing UCCL Lite and
default-NCCL measurements come from the
[November 13 study](../small_medium_4l4_20261113/README.md) and are preserved.
The new results are a separate measurement window; they do not revalidate
the historical UCCL implementation or establish a simultaneous paired speedup.

## Validated measurements

All 237 complete sweeps passed, including 49,770 checked timed collective
calls: 26,460 calibration, 21,420 independent evaluation and 1,890 fresh
default-confirmation calls. Sixteen SM points reverted to default settings
after validation; their rejected measurements remain in `initial_cases.csv`.

At 1 GiB complete tensor per GPU, bus throughput (GB/s) is:

| Collective | NCCL CE (tuned) | NCCL SM (tuned) | Fresh default control |
| --- | ---: | ---: | ---: |
| AllReduce | 20.07 | 17.75 | 17.75 |
| AllGather | 19.82 | 17.06 | 17.06 |
| ReduceScatter | 19.79 | 17.45 | 17.45 |

SM retains default settings at 1 GiB for all three collectives. The orange
figure curve remains the original November 13 default control; the fresh
controls above are supplied separately in the data. CE loses at small sizes;
those points are plotted in full.

## Hardware and timing

- `mibura-sky-test-02`: four NVIDIA L4 GPUs, PCIe Gen4, dual Xeon Silver
  4410Y, NUMA pairs `(0,1)` and `(2,3)`, no NVLink. CUDA 12.9 and NCCL 2.28.7.
  [Environment](environment.json) records GPU/driver details, topology,
  CUDA P2P read/write support, MPI version and source revision.
- FP32; 21 powers of two from 1 KiB through 1 GiB. Bytes mean complete
  AllReduce/ReduceScatter input or AllGather output per GPU.
- AllReduce: four MPI processes, in place, maximum-rank host
  enqueue-to-completion timing; five warmups and ten timed calls per run.
- AllGather/ReduceScatter: one process, out of place, maximum-GPU CUDA
  events including submission gaps; twenty warmups and ten timed calls.
- Input preparation, communicator initialization, allocation and complete
  output checks are outside timing. Every warmup and timed output is checked.
  CUDA-event and host-clock results are separate figures, never combined.
- Every process traverses the complete ascending size sweep on one communicator.
  Three independent evaluation runs produce a median of three run means.
  Shading shows the three-run range, also recorded in CSVs; it is not a confidence interval.
  Largest-first and CUDA-graph execution are outside this study.
- Bus throughput is `1.5 × bytes / time` for AR and `0.75 × bytes / time`
  for AG/RS, in decimal GB/s.

## CE and SM selection

The default NCCL topology policy chooses SM-based SHM paths on this machine
even though CUDA reports P2P support between all four GPUs. P2P candidates
set `NCCL_P2P_LEVEL=SYS` to admit every peer link; default-policy candidates
preserve NCCL's ordinary topology policy while tuning other parameters.

CE means NCCL's **P2P copy-engine transport**, enabled with
`NCCL_P2P_USE_CUDA_MEMCPY=1`. Ring and Simple are forced because NCCL 2.28.7
copies through that transport only for the Simple protocol. This still uses
SM kernels for reduction and protocol work; it is not a zero-SM collective.
Transport logs must show `P2P/CUMEM/CE` (or another explicit P2P `/CE` path).
Setting the memcpy variable alone, without admitting the peer links, was
observed to fall back to SHM and is not counted as a CE result.

SM candidates disable P2P memcpy. The candidate family includes the default
policy, buffer/channel tuning under that policy, ordinary P2P, P2P buffer sizes,
channel counts, a thread-count change,
Ring/Simple and peer reads. CE candidates vary buffers and channel counts.
All exact settings are in [candidates.json](candidates.json).

Two independently randomized calibration rounds select one configuration
per operation, exact size and family. [schedule.json](schedule.json) is frozen
before three fresh validation rounds. Validation runs each selected
configuration over the entire ascending sweep; only the preselected points
are evaluated for the final curves. No final-run best-of selection among
tuned candidates is performed.
These are calibrated benchmark schedules, not a portable dispatch policy.
Every candidate and all regressions are retained.
SM is the tuned default-NCCL baseline: a non-default setting is eligible only
if it beats the corresponding default control in both calibration rounds and
improves their median by at least 3%. Otherwise that point retains default
settings. Additional buffer/channel candidates under the default topology
policy were measured after the initial P2P candidate batch, before selection.
Calibration enables NCCL tuning-decision logs. Final evaluation disables
per-call tuning logs and retains only initialization/transport evidence,
so the plotted host timing does not include tuning-log I/O.

Validation exposed small-message regressions in some P2P configurations.
A non-default SM configuration is retained only if all three validation runs
beat their default controls and the median improvement is at least 3%.
Otherwise it returns to default settings; another tuned winner is not selected.
The resulting [final schedule](final_schedule.json) is then fixed, and default
settings are confirmed in three **new** complete sweeps per operation. Every
default point in the SM curve uses those confirmation runs. Retained tuned
points use their three independent validation runs. This fallback refinement
is based on validation evidence, so the entire resulting curve is not claimed
to be an untouched holdout experiment.

## Data and reproduction

- [cases.csv](cases.csv): final CE/SM medians and run ranges, selected
  configurations, and independently remeasured default controls.
- [selected_runs.csv](selected_runs.csv): all three evaluation run means
  or confirmation run means, their stage and each point's original calibration
  configuration/score. [initial_cases.csv](initial_cases.csv) retains the
  pre-fallback validation results, including the rejected regressions.
- [calibration.csv](calibration.csv) and [samples.csv](samples.csv): all
  successful candidate run means and every checked timed sample.
- [transport.json](transport.json): verified channel paths for every run.
- [validation.json](validation.json): sweep/sample totals and archive hash.
- [measurements.tar.gz](measurements.tar.gz): raw stdout, stderr, NCCL
  transport logs, run commands/settings, source snapshots, environment and
  binary/source/library hashes. Compiled binaries are not archived.

`build_probe.py` creates a NCCL-only copy of the existing AG/RS benchmark in
`build/`; [source.patch](source.patch) changes only its backend lists.
Input generation, full-output checks and timing functions are unchanged.
The AR control uses `benchmarks/collectives/nccl_4l4_mpi_baseline.cu` unchanged.
The copied AG/RS harness retains its legacy `# selected` summary strings;
these are ignored. Only explicitly labeled NCCL samples and actual NCCL
transport logs are used for this study.

From `experimental/lite` on the recorded host, with four idle GPUs:

```bash
python3 benchmarks/data/nccl_tuned_4l4_20261115/build_probe.py
make -f benchmarks/data/nccl_tuned_4l4_20261115/probe.mk -j2 nccl-four-tuning \
  CUDA_HOME=/usr/local/cuda-12.9 \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
python3 benchmarks/data/nccl_tuned_4l4_20261115/run.py calibrate
python3 benchmarks/data/nccl_tuned_4l4_20261115/run.py evaluate
python3 benchmarks/data/nccl_tuned_4l4_20261115/run.py confirm
python3 benchmarks/data/nccl_tuned_4l4_20261115/analyze.py
python3 scripts/plot_4l4_collectives.py
```

Use a fresh `build/nccl_tuned_4l4_20261115/` for a new study: the runner resumes
already recorded jobs. Restore archived source snapshots when reproducing this
revision. Plotting reads committed CSVs and performs no GPU work.
