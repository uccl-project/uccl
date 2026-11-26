# Fresh 8×L4 GPU and SHM collectives: 1 KiB–1 GiB

This replaces every ordinary-call point in the six GPU/SHM overview figures
with new measurements. No historical tail, copied curve, or resident timing
is used. [summary.csv](summary.csv) lists all 126 operation/testbed/size points;
[summary.csv](summary.csv) is the figure input. [candidates.csv](candidates.csv)
also retains the base router, both sweep medians, and run ranges.

## Hardware and transport

`wxz-l4`, eight NVIDIA L4s; GPUs 0–3 on NUMA 0 and 4–7 on NUMA 1. Intel Xeon
2.20-GHz CPU, 96 logical CPUs, driver 580.178.04, CUDA 12.9, NCCL 2.28.9.
GPU identities, topology and CPU details are in `hardware.csv`, `topology.txt`
and `cpu.json`. The submitting thread is pinned to CPU 47 after discovering
the complete permitted CPU set for NUMA-local workers.

- **GPU testbed:** default NCCL P2P policy; CUDA peer capability is preserved.
- **SHM testbed:** explicitly disable CUDA peer access, remove peer links from
  the inspected benchmark topology, set `NCCL_P2P_DISABLE=1` and
  `NCCL_SHM_DISABLE=0`. NCCL IB is disabled and Socket is selected in both modes.
- UCCL uses GPU SM access to NUMA-local mapped host packets in both modes.
  The two testbeds are independently measured, not copies of the same curve.
- The initialization probes at both **1 KiB and 1 GiB** report **16 SHM/direct/direct channel
  edges for each operation in both modes**, including the default GPU mode.
  Thus these are default-policy versus explicitly disabled-P2P experiments;
  they must not be described as measured NCCL P2P-versus-SHM throughput.
  Raw INFO traces and parsed channel edges are retained under `raw/probe_*`.

## Method

BF16, one process owning eight GPUs. AR is in place; AG/RS are out of place.
Bytes denote full AR/RS input or full AG output **per GPU**. Every power of two
from 1 KiB to 1 GiB is measured: 21 sizes × 3 operations × 2 testbeds.

Each size/backend has **3 repetitions per sweep, each with 5 warmups followed
by 10 timed calls**. There are two complete sweeps. The second reverses testbed, operation,
size and initial backend order; backend order rotates within repetitions.
The same process and harness compare `current`, `router-base`, and NCCL.
No minimum over variants, runs, or NCCL settings selects a plotted point.

Timing is maximum-rank per-call CUDA-event duration, including host enqueue
gaps. Every input producer stream completes before timing begins. Allocation,
preparation, changed input generation, validation, and communicator setup are
excluded. Calls have no deliberate idle. Every warmup and timed invocation
checks every output element and a 32-byte output guard. Input values depend on
rank, position and generation, include both signs, and make all BF16 partial
sums exactly representable. This isolates transport correctness; it is not a
general BF16 rounding-conformance test.

The benchmark enqueues all eight verification kernels before waiting for any
rank, avoiding serialized large-tensor checking. Verification still completes
before the next fill/timer. The checker arithmetic and transport kernels are
unchanged. A point is the median of three per-repetition medians in each sweep,
then the median of the two sweep medians. Min/max in `candidates.csv` bound
those two sweep medians; they are not confidence intervals.

The full comparison contains **22,680 checked timed calls and 11,340 checked
warmup calls**. Transport probes, 1-GiB boundary checks and standalone regression
tests are separate and do not contribute timings to the figures. The runner
checks GPU ownership before, during and after every run, terminating only its
own benchmark if another GPU process appears. This is observation, not an
exclusive reservation.

## Fixed current configuration

The same schedule is used in both testbeds and fixed before the full sweeps:

| Operation | Complete bytes/GPU | Implementation |
| --- | --- | --- |
| AR | 1–8 KiB | Flat GPU reduction and GPU progress |
| AR | 16 KiB–4 MiB | CPU NUMA partial reduction/merge; four workers per NUMA, 16-block transport cap |
| AR | 8 MiB–1 GiB | CPU NUMA partial reduction/merge; regular 23-worker pool and 64-block cap |
| AG | 1–128 KiB | Packet AG with GPU progress; per-source consumption when consumer geometry permits |
| AG | 256 KiB–1 MiB | Mini-host cooperative grid, 16 × 256 threads, all-reader bank credits |
| AG | 2 MiB–1 GiB | Existing full per-source GPU progress, regular 64-block cap |
| RS | 1–64 KiB, 128 KiB | GPU reduction of each destination shard and GPU progress |
| RS | 256 KiB–1 MiB | CPU NUMA reduction/owner merge, at most four active workers per NUMA |
| RS | 2 MiB–1 GiB | CPU NUMA reduction/owner merge, regular scatter-worker cap of 16 per NUMA |

Router variants use batched CPU job publication and four eager GPU submission
lanes per NUMA (eight launch threads). The mini-host path has no CPU routing or
launch-worker pool. Router packets have a 512-KiB regular cap, 128-KiB cap for
AR through 4 MiB and RS at 256 KiB–4 MiB, and 2-MiB cap for AG from 4 MiB.
Actual packets may be smaller; for AG the cap accounts for rank expansion.

`router-base` uses the current source with regular CPU progress, worker/grid
settings, the existing 128/256-KiB compact AR rule and exact 128-KiB GPU RS
choice, with the same packet caps. It is retained as an ablation, not selected
when it happens to beat `current`. Larger AR uses the regular worker pool
instead of extending the small-message four-worker cap to 1 GiB.

This is a benchmark schedule, not a new production dispatcher policy. No
resident kernel, GEMM fusion, application integration, or NCCL fallback is
substituted for a plotted UCCL result. Parallel submission is evaluated for
dense calls here; the earlier transfer study retains its sparse-call losses.

## Results and regressions

At 1 GiB, complete collective latency is:

| Testbed | Operation | UCCL (ms) | Paired NCCL (ms) | NCCL / UCCL |
| --- | --- | ---: | ---: | ---: |
| GPU default | AR | 278.073 | 372.902 | 1.341× |
| GPU default | AG | 162.106 | 195.585 | 1.207× |
| GPU default | RS | 170.093 | 189.446 | 1.114× |
| Explicit SHM | AR | 278.148 | 361.643 | 1.300× |
| Explicit SHM | AG | 161.967 | 188.407 | 1.163× |
| Explicit SHM | RS | 171.829 | 189.161 | 1.101× |

AR and AG have lower UCCL medians at all 21 sizes in both testbeds. RS loses
to paired NCCL at **1, 2, 4 and 8 MiB** in both modes. Those points remain in
the six figures and are marked in the complete tables. The base-router
comparison, including small or negative tuning gains, remains in the CSVs.
The two testbeds' NCCL differences are different measurement windows under the
recorded policies, not evidence of a P2P-versus-SHM transport advantage.

## Validation and reproduction

Before the sweeps: four/eight-GPU compact checks (including explicit SHM), CPU
queue/BF16, policy, host/CUDA dispatcher, and eager/graph router checks. Every
operation also has a separate checked 1-GiB boundary run on both testbeds.
Commands, source/binary hashes, exit statuses, process observations, complete
per-call samples and negative results remain in the records.
The additional large-message transport probes call the same `bench` helper
with `begin=end=1073741824`, `warmup=1`, `iterations=1`, `repeats=1`,
`variants="nccl"`, and `probe=True` for each mode/operation; their exact commands
are in `raw/probe_1g_*.json`. They are excluded from plotted timing samples.

From `experimental/lite`:

```bash
make -j2 compact-collectives router-checks \
  BUILD=build/full_refresh_8l4_20261114 CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 NCCL_HOME=build/nccl-2.28.9
python3 scripts/run_full_8l4.py checks
python3 scripts/run_full_8l4.py probe
python3 scripts/run_full_8l4.py tail
python3 scripts/run_full_8l4.py full
python3 scripts/summarize_full_8l4.py
python3 benchmarks/data/overview_8l4_20261114/build.py
python3 benchmarks/data/overview_8l4_20261114/plot.py --preview
```

Use a fresh data destination for a new run; the runner refuses to overwrite
evidence. `raw/full_*.log` contains per-call samples and the actual implementation
for every size; its JSON records the exact command/environment and binary hash.
Build logs and source hashes are recorded separately. The previous compact and
historical measurements remain in their original datasets and Git history.
