# Eight-L4 fusion after standalone collective optimization

This is a fresh review of `benchmarks/fusion/router_fusion_8l4.cu`, not an
update of the November 11 measurements. The historical data and figures remain
unchanged. The tested implementation uses resident CUTLASS BF16 GEMM and CPU
packet progress. SGLang's multiprocess packet/row-ready integration is a separate
implementation and is not benchmarked here.

Results are in [results.md](results.md); [summary.csv](summary.csv) retains
both run medians and every measured configuration.

The fixed 512-KiB resident AR packet candidate improves all four tested prefill
shapes by 7.5%–19.4% over the old 128-KiB cap, in both reversed runs. This
remains an explicit calibrated option; the default and production routing policy
are unchanged. Other shapes still include fusion regressions.

## Implementation findings

The original harness prepares one CPU-progress pipeline for both sequential
cuBLAS+router and fused GEMM+router. Its default packet cap is 128 KiB for AR
and 512 KiB for AG/RS. It does not expose the newer standalone schedule as a
control: batched CPU job publication, parallel eager submission, small GPU
reduction/progress, per-source GPU AG progress, larger AG packets, mini-host AG,
and the 1–8 MiB host-ring RS schedule.

These are not interchangeable resident GEMM settings:

- `PacketPipeline::submit` rejects resident GEMM with GPU reduction, and rejects
  interleaved scatter with resident GEMM.
- `prepareGemm` rejects host-ring RS. Its standalone cooperative ring does not
  consume resident GEMM tile readiness.
- Parallel eager graph submission explicitly excludes fused GEMM. Resident
  launches still use the caller and obey the cooperative occupancy limit.
- CPU job batching already supports resident GEMM; it can be evaluated without
  changing the device synchronization protocol.

The revised harness exposes `--router-profile current` as an explicit
benchmark-only control with an independent standalone pipeline (or mini-host
AG). It follows the fixed `compact_collectives.cu` current schedule, including
the RS ring recovery. `--router-profile historical` preserves the old shared
configuration. `--batch-cpu-jobs 1` is an explicit candidate for the resident
CPU path; neither option changes a production dispatcher or serving default.

The checker now rejects NaN/infinity, and every full-operation call poisons
the final output before timing. This detects unwritten/stale final output;
it does not constitute adversarial testing of every intermediate readiness
dependency. Existing resident CTA/GEMM tile schedules are preserved.

Per-method log fields such as `packet_kib` describe the resident pipeline's
options; the separate standalone pipeline prints its own actual configuration
at initialization. The profile, exact command and those initialization lines
must be read together.

## Hardware and method

Host `wxz-l4`, eight NVIDIA L4s, ranks 0–3 on NUMA 0 and 4–7 on NUMA 1;
two-socket Intel Xeon CPU @ 2.20 GHz, 96 logical CPUs, driver 580.178.04,
CUDA 12.9.41, CUTLASS 4.5.0, NCCL 2.28.9. The submitting process is pinned to
CPU 47 for performance runs. Correctness tests retain the full permitted CPU
set so the dispatcher can inspect worker placement. GPU identities and topology
are in `hardware.csv`, `topology.txt`, `cpu.json`; library identity and build
command are in `software.json`.

BF16 inputs/outputs with FP32 GEMM accumulation. Two endpoint model-derived
shapes use `(N,K)=(4096,1536)` and `(5120,3200)`. AR/RS use
`M=1,64,2048,8192`; AG uses `M=8,64,2048,8192` so each of eight ranks owns an
integral row shard. This is 24 operation/shape pairs, not the complete previous
four-model sweep. Complete collective bytes/GPU are `2*M*N` for AR/RS input
and `2*M*K` for AG output; RS returns one eighth of its complete input.
The fused path also replaces cuBLAS with resident CUTLASS GEMM; its differences
cannot be attributed solely to communication/compute overlap.

Each shape has two process runs with opposite shape and configuration order.
Each process uses two warmup rounds and three measured rounds of ten calls
per method. The historical and current controls rotate cuBLAS+NCCL,
cuBLAS+router and fused+router within rounds. A third process configuration
measures only fused+router with CPU job batching enabled. That third
configuration is a diagnostic: isolated versus interleaved execution is a
timing-scope difference, so a small difference cannot establish a batching gain.

Each sample is the maximum-rank CUDA-event duration for the complete
AG+GEMM, GEMM+RS, or GEMM+AR operation, including host enqueue gaps. Preparation,
allocation, communicator setup, input generation, output poisoning and numerical
checking are excluded. There is no deliberate per-call idle. Every warmup and
timed full-operation invocation checks all output elements at 2% relative
tolerance. The deterministic positive input pattern is retained; this does not
test general BF16 rounding conformance, and the ring rounds at each hop.

`summary.csv` reports each run's sample median and the median of those two
medians. Every method/shape/configuration has 60 measured calls across the two
runs; no best packet size, tile, method or run is selected per plotted point.
The configuration comparison has 10,080 checked timed calls and 672 checked
warmups. The source and binary hashes, exact environments/commands, per-call
samples, exit statuses, and GPU ownership observations are retained. The runner
stops its own child if another GPU process appears; it does not reserve the host.

## Validation and reproduction

Build with the Makefile command in `software.json`. Then use
`python3 scripts/run_fusion_review_8l4.py checks`, followed by `sweep` and
`summary`. Use a fresh data destination in the runner for a new experiment;
existing evidence is never overwritten.

Policy, host/CUDA dispatcher, CPU BF16 and router-core checks are accompanied by
CPU-batched and GPU per-source eager/graph tests on the local four-rank 2+2
NUMA subset `(0,1,4,5)` and all eight GPUs. These are local subset tests, not
fresh tests on a separate four-GPU host. The original dispatcher CUDA attempt
failed because the test driver prematurely narrowed CPU affinity to one CPU;
the corrected full-affinity rerun passed. Both attempts remain in `raw/`.

The first current-control mini-host AG attempt, `(M,N,K)=(64,5120,3200)`,
failed output verification in warmup: the new harness passed a per-rank count
to an API that requires the complete output count. The fix is
`mini->gather(aCount * ranks, ports)`. The failed process contributes no timed
samples. Its log/manifest remain alongside the successful `_retry1` invocation.
`measured_source_v1/` and `source_hashes.json` preserve the first binary's
benchmark/runner; `source_hashes_v2.json` identifies the corrected build. The
only C++ difference is that mini-host call argument. Completed first-run
measurements that never entered the mini-host branch are retained; all
mini-host measurements and the reversed second run use the corrected binary.
Each invocation records its binary hash. No transport/device kernel changed
between these builds.

An unrelated GPU process appeared during the second-run current RS control at
`(64,5120,3200)`. The ownership guard terminated only this benchmark (exit -15)
and the affected invocation was excluded and retried after the GPUs were idle.
Its original manifest records the foreign PID. Resume manifests from version 3
onward record runner-only retry changes; the corrected benchmark binary is
unchanged.

## Explicit AR packet candidate and control boundaries

`python3 benchmarks/data/fusion_review_8l4_20261115/packet_candidates.py`
compares fixed 128-KiB and 512-KiB resident AR packet caps at the four tested
prefill shapes (both models, M=2048/8192). Shape and packet order reverse in
the second run. Each process interleaves all three complete-operation methods,
with two warmup rounds and 3 x 10 checked timed calls per method. The current
standalone control keeps its own fixed 512-KiB cap in both comparisons.
`packet_summary.csv` retains both runs; the fixed-cap comparison is separate
from the 24-shape table and does not choose the best cap per point. It adds
1,440 checked timed calls and 96 warmups.

The same script separately checks the RS ring at complete inputs of 1/8 MiB
and mini-host AG at complete outputs of 256 KiB/1 MiB using all three methods.
These four boundary invocations add 36 checked timed calls and 24 warmups;
their timings are excluded from performance tables. Four/eight-rank host-ring
checks also pass: 1,048,576 / 2,097,152 independent scalar checks, 270 asynchronous
calls and 3 x 140 graph calls each, including changing extents, guards and
rejected-call recovery. Logs/manifests remain in `raw/`.

A second brief unrelated GPU process interrupted the reversed packet run at
32B M=8192 / 512 KiB. That invocation was terminated and excluded, then retried
on idle GPUs. `packet_candidates.py` now resumes completed cases and preserves
any excluded partial samples in `raw/`. The pre-resume driver is archived in
`measured_source_v1/`; the measured C++ binary is unchanged.
