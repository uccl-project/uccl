# Four-L4 techniques on eight L4s — November 14, 2026

This experiment tests three transfers: a specialized mapped-host AllGather,
prepared persistent collectives, and the existing CPU-router tuning options.
The implementations are isolated in `benchmarks/cpu_router/compact_transport.cuh`
and `compact_collectives.cu`. No production dispatcher, default, C API, or
multiprocess integration is changed. The prototype uses no GPU peer accesses
and no CPU reduction workers; it uses the router module's NUMA-pinned memory.

## Hardware and measurement

`wxz-l4`: eight NVIDIA L4s, GPUs 0–3 on NUMA 0 and 4–7 on NUMA 1, Intel Xeon
2.20-GHz CPU, 96 logical CPUs, driver 580.178.04, CUDA 12.9, NCCL 2.28.9.
See `hardware.csv`, [GPU topology](../full_refresh_8l4_20261114/topology.txt), and `cpu.json`. NCCL uses its default local
P2P/SHM choices; P2P is not disabled. Submission is pinned to CPU 47 after
discovering the full permitted CPU set. The runner checks GPU ownership before,
during, and after each run and aborts its own child if another process appears.
This is an observed idle window, not an exclusive machine reservation.

All experiments are one-process BF16. AR is in place; AG/RS are out of place.
Bytes denote full AR/RS input or full AG output per GPU. Every warmup and timed
call uses a changed signed rank/position/generation-dependent input and checks
every output element and a 32-byte output guard. Inputs and all intermediate
sums in this benchmark are exactly representable in BF16; this isolates
transport correctness from reduction-order differences. It is not a general
BF16 numerical conformance test.

Preparation, allocation, input generation/completion, and checking are outside
timing. Every timed call starts with every producer stream complete. Mini-host
AG and router-tuning sweeps use maximum-rank per-call CUDA-event intervals,
including host submission gaps. Persistent sweeps use host command/enqueue to
global completion for **every** compared backend. Those timing scopes are
separate; their values must not be divided across tables.

The NCCL graph control is one fork/join graph across all eight GPUs. Its root
stream joins all ranks, and timing waits only for that root stream. Graph
creation and warmup are excluded. No persistent CTA is running in NCCL or
router windows. Resident timing ends after all eight system-release completion
flags are observed; it does not add an unnecessary CUDA synchronization.

Screening uses five warmups and three rotated repetitions of ten calls.
Confirmation and the extended tuning comparison use ten warmups and five
rotated repetitions of twenty calls, in two independent sweeps with reversed
variant order. A point is the median of repetition medians, then the median of
the two sweep medians. `summary.csv` retains the two-sweep min/max, which are
ranges rather than confidence intervals. No curve selects its best variant per
size or consults NCCL timing to choose a UCCL variant. Sparse-call diagnostics
insert 5 ms before each producer fill; the idle interval is outside timing.

## Variants

- `mini`: one cooperative 16-block × 256-thread grid per GPU. All blocks upload
  the local shard to its NUMA-local host bank, then source-specific block groups
  gather ready shards. Local data stays on GPU. All-reader credits prevent
  overwriting the bank across queued calls. The 8/16/32-block screens are
  retained; confirmation uses the same 16-block geometry at every size.
- `resident`: one ordinary persistent CTA × 256 threads per GPU. One packed
  64-bit doorbell carries epoch, operation, and extent. All communication uses
  NUMA-pinned host buffers; reductions accumulate in FP32 and round once to
  BF16. It handles mixed operations and extents, but requires an input-ready,
  host-blocking caller and fixed buffers. It reserves a CTA and polls until
  explicitly stopped. Load producer/check kernels before starting it; stop it
  before `cudaDeviceSynchronize`, or synchronize only caller streams. General
  model-compute interference and application graph
  integration are not measured.
- `router-base`: current dispatcher with regular CPU progress, 23 workers per
  NUMA, 64 transport-block cap, 512-KiB regular / 128-KiB small packet settings,
  2-MiB large-AG packet setting, the existing compact AR rule at 128/256 KiB,
  and the existing 128-KiB GPU RS choice. This is a current-source control,
  not the historical pre-optimization binary.
- `router-progress`: batch CPU jobs, four active RS workers at 256 KiB–1 MiB,
  GPU progress through 8-KiB AR / 1-MiB AG / 64-KiB RS. AG uses per-source
  readiness. AR uses four CPU workers and a 16-block cap at every size.
- `router-tuned`: `router-progress` plus four GPU submission workers per NUMA.
- `router-single`: `router-tuned` plus a one-CTA transport cutoff at 64 KiB.
  Above that cutoff it has the same configuration as `router-tuned`; timing
  differences there are run variation, not an additional mechanism.
- `router-full-ag`: `router-tuned` with GPU per-source progress at every AG
  size, using the existing explicit GPU-progress options. `router-full-ag-compact`
  additionally caps the grid at 16 blocks. Both are existing router mechanisms,
  added as controls in the separate attribution sweep, not new kernels.

## Results

The complete checked results are in `summary.csv`; the concise confirmed tables
are in [summary.csv](summary.csv). Screens, reversed confirmation, extended tuning, and
sparse-call probes remain separate experiments.

The attribution experiment narrows the useful new mini-host range to the
tested **256 KiB, 512 KiB, and 1 MiB** outputs: about **6–13%** faster than
the full-progress router. At 2 MiB it ties full GPU progress; at 4–16 MiB the
existing full-progress router is faster. Full GPU progress itself improves the
limited-progress control by **4–10%** at 2–16 MiB. Its 16-block grid cap loses
at larger sizes. There is no need to add mini-host solely for those large AG
gains.

At 1 KiB, resident AG/RS/AR take **21.69 / 14.90 / 17.46 us**, compared with
**58.17 / 67.80 / 67.60 us** for the tuned router in the same host-clock
experiment. Resident AG/RS remain faster through the tested 64-KiB limit.
AR is compelling through 16 KiB, near a tie at 32 KiB, and loses at 64 KiB
(196.39 versus 144.22 us). These are benefits of an explicit prepared execution
contract, not a transparent replacement for stream-enqueued collectives.

Existing tuning reduces dense ordinary 1-KiB AG/RS/AR latency by
**35.3% / 25.9% / 25.3%** relative to the current-source base configuration.
At 1 MiB the reductions are 12.9% / 7.7% / 6.6%. AR regresses 2.6% at 4 MiB.
The one-CTA option adds only small tiny-message gains and loses at 64-KiB AG.
After 5 ms of idle, parallel launch is slower than progress without launch
workers: at 1 KiB AG/RS/AR, **114.27 / 124.12 / 125.04 us** versus
**69.07 / 83.77 / 82.72 us**. Keep it workload-specific.

## Validation and retained diagnostics

The compact check exercises four asynchronously queued mini-host generations
with distinct inputs/results, delayed ranks, vector and unaligned scalar tails,
mixed resident operations, full output guards, and repeated start/stop. Each
run checks 114 invocations with an independent host-double oracle, on both
eight GPUs and the local four-GPU subset. This subset is not the earlier
four-L4 Xeon Silver machine. In-place AR is additionally checked on every
benchmark invocation. Existing router policy, host/CUDA dispatcher, CPU queue,
BF16 arithmetic, eager, and captured-graph checks are retained alongside it.
The existing router graph tests do not establish graph support for `mini` or
`resident`: mini-host explicitly rejects capture, and resident is a separate
host-blocking API.

Targeted Compute Sanitizer memcheck, with 32-byte allocation padding, reports
**zero errors** for both transport kernels on four and eight GPUs. The filters
instrument `miniGather` and `resident`; producer/check kernels are excluded.
Instrumenting the entire harness stalled after starting resident workers and
was terminated; that run is retained in `diagnostics/memcheck_4.*` and is not
counted as a pass. The filtered commands and scope are recorded in
`memcheck_transport_4.json` and `memcheck_transport_8.json`.

The first persistent checker stalled when the delay kernel was lazily loaded
on a GPU with a running persistent worker. Increasing CUDA connection count
did not fix it. Explicitly warming every producer/check kernel and runtime copy
path on every GPU before starting persistent workers fixed it. No stalled-run
timing is accepted. `diagnostics/` retains those logs and an aborted overlapping
smoke probe, excluded from all reported timing and successful checks.

Screening initially enabled the SHM-only 64-KiB GPU RS baseline option on this
GPU testbed. This affects the `router-base` RS screen at 64 KiB; it is not the
regular baseline at that point. The baseline option was removed before the
accepted full confirmation and tuning sweeps. The interrupted earlier
confirmation is retained under `diagnostics/` and excluded. Other candidates
explicitly use the size-limited GPU setting. `screen_to_confirmed.patch` records
the control correction and lifecycle guards, relative to `screen_hashes.json`.
`confirmed_hashes.json` identifies the main confirmation/tuning/sparse binary.
`confirmed_to_final.patch` adds only the full-progress AG controls and runner
entry; `final_hashes.json` identifies that attribution binary. The transport
kernels are identical across those two builds.

## Existing-option calibration

`scripts/calibrate_compact_transfer.py` uses the original adaptive-collectives
binary to generate fresh exact-size profiles for the dense tuned AG/RS/AR
configurations over 1 KiB–1 MiB, plus full-progress AG over 2–16 MiB. These use
their own paired NCCL controls, five warmups, and three repetitions of ten
calls. They are independent calibration measurements, not additional samples
for the prototype tables. Matching automatic replay and unmeasured-size probes
are retained. These profiles are scoped to the recorded hardware, software,
placement, options, fallback, and dense input-ready workload; they do not
authorize using launch workers for sparse workloads. Nothing installs them as
a global policy.

All **37** measured keys select the router in fresh automatic replay. All four
unmeasured-size probes retain NCCL with **no router initialization** (1,536 bytes
for AG/RS/AR; 3 MiB for large AG). See `auto_replay_summary.json`. The four
calibrations retain another **2,220** checked timed calls, separate from the
**90,920** calls in the prototype screening/confirmation/tuning/attribution and
sparse-call tables. Warmups and standalone correctness checks are additional.

## Reproduction

From `experimental/lite`, on an idle eight-L4 host:

```bash
make -j2 compact-collectives adaptive-collectives router-checks \
  BUILD=build/transfer_4to8_20261114 CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 NCCL_HOME=build/nccl-2.28.9
python3 scripts/run_compact_transfer.py checks
python3 scripts/run_compact_transfer.py screen
python3 scripts/run_compact_transfer.py regression
python3 scripts/run_compact_transfer.py confirm
python3 scripts/run_compact_transfer.py tuning
python3 scripts/run_compact_transfer.py sparse
python3 scripts/run_compact_transfer.py attribution
python3 scripts/calibrate_compact_transfer.py
python3 scripts/summarize_compact_transfer.py
```

Choose a fresh data destination before rerunning: the runner refuses to
overwrite existing records. Each run's JSON preserves command, environment,
return status, duration, and GPU-ownership observations. Final timing manifests
include binary SHA-256; `final_hashes.json` identifies sources and libraries.
Build logs are retained. No runtime calibration profile for either prototype
is generated, and neither is enabled automatically from GPU count or these
benchmark envelopes.
