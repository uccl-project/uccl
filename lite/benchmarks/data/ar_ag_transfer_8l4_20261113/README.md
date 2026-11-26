# Eight-L4 AllReduce / AllGather scheduling transfer

This experiment transfers ReduceScatter's CPU callback batching and eager GPU
submission lanes to AllReduce and AllGather. It also exposes independent,
size-limited GPU progress: flat GPU reduction for tiny AllReduce, and the
existing per-source readiness path for AllGather. All controls remain opt-in.

The fixed compact AllReduce configuration improves latency by **3.2–14.5%**
from 1 KiB through 512 KiB, with only **1.7%** at 1 MiB. AllGather improves by
**13.8–33.2%** from 1 KiB through 1 MiB. These standalone dense eager results do
not establish application or multiprocess drop-in speedups. The README embeds
the new 1-KiB–1-MiB latency figures; historical 1-KiB–1-GiB overview figures
remain available as separate links.

## Hardware and method

Eight NVIDIA L4s (SM89), no cross-NUMA peer access; ranks 0–3 on NUMA node 0,
ranks 4–7 on node 1. Intel Xeon CPU at 2.20 GHz, 96 logical CPUs; submission
CPU 47. Driver 580.178.04, CUDA 12.9, NCCL 2.28.9. GPU identities are in
[hardware.csv](hardware.csv); binary hashes, source identity, options and limits
are in [provenance.json](provenance.json).

BF16, single process, AllReduce in place and AllGather out of place. Bytes are
complete input bytes per rank for AR and complete output bytes per rank for AG.
Each sweep covers powers of two from 1 KiB to 4 MiB: ten warmups per backend,
five alternating NCCL/router repetitions of twenty calls, checking every output
element on every invocation. Each repeat reports the median of those twenty
slowest-rank CUDA-event intervals. Each sweep reports the median of five repeat
medians, and the table reports the median of two sweep medians. Sweep min/max
values in [summary.csv](summary.csv) are ranges, not confidence intervals.
The original event interval includes host submission gaps from the harness;
it is not isolated device-kernel execution time. Calls have no deliberate idle.

Confirmation reverses configuration and operation order. Each configuration
has its own paired NCCL control; ratios use that control's two-sweep median,
not another configuration's NCCL timing. NCCL P2P is left on within NUMA and
uses SHM across NUMA, matching the earlier GPU testbed. Compute processes are
checked before and after each sweep; occupied GPUs abort measurement.

There is one fixed optimized configuration per operation for the full sweep:
AR uses `compact_combined`, AG uses `combined`. The choice considers router
timings and regressions over the entire range; no size picks its best variant,
and selection does not use NCCL values. Every control remains in the summary.

## Latency (µs)

| Collective | Complete bytes/rank | Previous UCCL Lite | Optimized UCCL Lite | Paired NCCL | NCCL / optimized | Reduction vs previous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AllReduce | 1 KiB | 98.8 | 84.5 | 133.4 | 1.58x | 14.5% |
| AllReduce | 2 KiB | 97.3 | 85.0 | 131.6 | 1.55x | 12.6% |
| AllReduce | 4 KiB | 100.1 | 85.8 | 133.9 | 1.56x | 14.3% |
| AllReduce | 8 KiB | 105.0 | 96.3 | 138.2 | 1.44x | 8.3% |
| AllReduce | 16 KiB | 114.7 | 106.2 | 144.6 | 1.36x | 7.4% |
| AllReduce | 32 KiB | 131.6 | 126.5 | 155.6 | 1.23x | 3.9% |
| AllReduce | 64 KiB | 162.3 | 154.6 | 173.1 | 1.12x | 4.7% |
| AllReduce | 128 KiB | 198.7 | 190.5 | 215.3 | 1.13x | 4.1% |
| AllReduce | 256 KiB | 229.6 | 222.2 | 302.3 | 1.36x | 3.2% |
| AllReduce | 512 KiB | 311.3 | 298.2 | 431.4 | 1.45x | 4.2% |
| AllReduce | 1 MiB | 473.1 | 464.9 | 601.1 | 1.29x | 1.7% |
| AllReduce | 2 MiB | 812.5 | 816.1 | 995.3 | 1.22x | -0.4% |
| AllReduce | 4 MiB | 1480.4 | 1499.4 | 1789.4 | 1.19x | -1.3% |
| AllGather | 1 KiB | 106.5 | 74.2 | 119.6 | 1.61x | 30.3% |
| AllGather | 2 KiB | 104.7 | 73.2 | 116.2 | 1.59x | 30.1% |
| AllGather | 4 KiB | 106.2 | 74.0 | 117.2 | 1.58x | 30.4% |
| AllGather | 8 KiB | 113.4 | 75.8 | 119.3 | 1.57x | 33.2% |
| AllGather | 16 KiB | 110.6 | 78.6 | 122.1 | 1.55x | 28.9% |
| AllGather | 32 KiB | 116.0 | 84.0 | 127.5 | 1.52x | 27.6% |
| AllGather | 64 KiB | 122.9 | 91.6 | 136.7 | 1.49x | 25.4% |
| AllGather | 128 KiB | 137.7 | 101.9 | 159.5 | 1.57x | 26.0% |
| AllGather | 256 KiB | 175.4 | 128.8 | 204.5 | 1.59x | 26.6% |
| AllGather | 512 KiB | 213.2 | 181.8 | 273.4 | 1.50x | 14.8% |
| AllGather | 1 MiB | 308.2 | 265.7 | 369.2 | 1.39x | 13.8% |
| AllGather | 2 MiB | 489.7 | 483.6 | 564.0 | 1.17x | 1.3% |
| AllGather | 4 MiB | 853.0 | 841.5 | 947.2 | 1.13x | 1.4% |

At 2/4 MiB, compact AR records 816.1/1499.4 µs versus 812.5/1480.4 µs before,
and AG records 483.6/841.5 µs versus 489.7/853.0 µs before. These differences
do not establish an additional large-message gain. Sizes above 4 MiB were not
remeasured.

## Fixed configurations

All commands add `--main-cpu 47`. The current benchmark exposes the flags below;
its default packet rules remain enabled.

- `previous`: frozen pre-change binary, existing benchmark defaults. In
  particular, the prior benchmark already uses four CPU workers and sixteen
  transport blocks at AR 128/256 KiB; the optimization is not credited with
  introducing that rule.
- `batch`: current binary plus `--batch-cpu-jobs`, now covering every layout.
- `progress`: batch plus `--small-ar-gpu-progress-max-bytes 8192
  --small-ag-gpu-progress-max-bytes 1048576`. Above the respective limit,
  CPU progress remains enabled. AG uses per-source readiness below its limit.
- `combined`: progress plus `--gpu-launch-workers-per-node 4`.
- `compact_combined`: combined plus `--workers-per-node 4
  --transport-blocks 16` across every size. This is the AR table configuration.
- AR-only `cpu_parallel` and `cpu_parallel_compact`: batch plus submission
  lanes, respectively with regular or compact worker/grid settings, with CPU
  progress at every size. These run as two separate reversed-order sweeps
  after the main confirmation, and are not additional samples for other rows.

For example, from the Lite root, after building with the command in provenance:

```bash
build/ar_ag_transfer_20261113/uccl_lite_collectives \
  --gpus 8 --op allreduce --in-place --fallback nccl \
  --begin 1024 --end 1048576 --warmup 10 --iterations 20 --repeats 5 \
  --main-cpu 47 --batch-cpu-jobs --gpu-launch-workers-per-node 4 \
  --workers-per-node 4 --transport-blocks 16 \
  --small-ar-gpu-progress-max-bytes 8192 \
  --calibrate build/ar_ag_transfer_20261113/fresh_ar.profile
```

For AG, use `--op allgather`, omit `--in-place` and the compact worker/grid
flags, and set `--small-ag-gpu-progress-max-bytes 1048576`. Use fresh output
names and the clean environment recorded in each raw command manifest.
Regenerate local profiles after rebuilding. Archived profiles identify their
exact software and configuration and are measurement records.

## What transfers, and what does not

- **Submission lanes** reduce host serialization across devices. They reserve
  eight extra CPU threads and spin for up to 2 ms. New AR/AG sparse-call
  performance is not measured; the earlier RS experiment demonstrates the
  regression risk after workers sleep. Affinity and spare CPUs matter.
- **CPU callback batching** avoids creating unused per-lane completion handles
  and publishes the batch before notifying sleepers. It produces modest,
  size-dependent gains alone. AR 32 KiB is slower in the batch-only median;
  the raw run ranges are retained.
- **Small CPU crews** help AR at 512 KiB. Keeping the default CPU pool with
  the combined options instead records 322.0 µs there versus 311.3 µs before
  (3.5% slower), and 478.2 µs versus 473.1 µs at 1 MiB (1.1% slower).
  The fixed compact configuration avoids those regressions; it is not a
  per-size mix with the default-pool curve.
- **GPU progress** eliminates CPU packet callbacks and lets AG copy each source
  as it arrives. CPU batching plus GPU progress without extra launch lanes
  records 85.8/271.4 µs at AG 1 KiB/1 MiB, versus 106.5/308.2 µs before.
  The lanes then reduce these to 74.2/265.7 µs.
- **AR GPU arithmetic has a narrow useful range.** GPU progress without launch
  lanes adds little over CPU batching at tiny sizes. The separate CPU-only
  launch controls isolate it further but have visible sweep variation; compare
  their recorded ranges rather than treating phases as paired microsecond
  deltas. Preliminary out-of-place screening with GPU reduction at all sizes
  records 1482.2 µs at AR 1 MiB versus 495.1 µs for CPU reduction. Every GPU
  destination rereads all sources; the RS retained-shard savings do not carry
  over to a replicated AR result. The AR GPU-progress limit stays at 8 KiB in
  the fixed candidate. GPU arithmetic accumulates in FP32 and rounds once;
  CPU arithmetic rounds intermediate NUMA partials to BF16.
- **Per-source AG readiness** helps midrange AG beyond plain GPU progress.
  Preliminary existing-option screening records 134.7 versus 146.4 µs at
  256 KiB, and 272.4 versus 287.7 µs at 1 MiB, with submission lanes in both
  cases. Single-CTA and compact-grid controls are also retained, including
  slower results. These screening sweeps are separate from confirmation;
  some overlap CPU compilation. Initial AR screening is out of place and
  cannot be substituted for the in-place AR confirmation.

## Validation and files

[correctness.json](correctness.json) records 13 passing checks: policy (35),
host dispatcher (54), CPU executor/BF16 tests, 2,928 queued mixed eager cases
across four/eight GPUs, 2,000 captured mixed collectives and
53,138,272 independent randomized reduction scalar/guard checks. Tests include
in-place AR/AG, delayed producers, queue wrap, capture rejection/recovery and
progress-limit crossings. AR flat-GPU to AG per-source progress requires
`prepare` even when packet geometry is identical.

[auto_replay.json](auto_replay.json) checks both operations with the fixed
`combined` options: all 13 known sizes follow their measured eligibility;
an exact 1,536-byte size and a changed progress threshold retain NCCL with no
router initialization. The compact AR candidate has separate calibration
records and is never enabled using the default-pool profile.

- [summary.csv](summary.csv): 156 points, including all controls and regressions.
- `raw/`: exact stdout, stderr, local calibration profiles, command/environment
  manifests and losslessly gzip-compressed per-call samples.
- [source_hashes.json](source_hashes.json),
  [source_hashes_before.json](source_hashes_before.json) and
  [implementation.patch](implementation.patch): compiled source identities
  and the change from the recorded baseline commit.
- `validation/`: build and correctness logs; `reproduction/`: verbatim run
  scripts. Copy scripts into the ignored build directory before execution,
  reconstruct the baseline binary from its recorded source state, and choose
  fresh output names as described in provenance. Binaries are not committed.
- `summarize.py`: reproduces the summary from the raw confirmation CSVs.
- `plot.py`: reproduces the two-series [AllReduce](allreduce.svg) and
  [AllGather](allgather.svg) latency figures through 1 MiB, using the fixed
  table configurations and their paired NCCL controls. Fonts, colors and line
  styles match the other README figures.
- [checksums.json](checksums.json): SHA-256 for every archived file except itself.

The optional C++ module remains independent of NCCL. Defaults, exact calibration
requirements, and fallback behavior remain intact. The C wrapper and
multiprocess NCCL drop-in expose no new controls in this change.
