# UCCL Lite AllReduce: large-message bottleneck and CE ring follow-up

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


**The measured target is met:** UCCL Lite wins **all 21 sizes in all five
independent validation rounds**, and its slowest observed run beats NCCL CE's
fastest observed run at every size. Policies were frozen before validation.

At 1 GiB, the medians are **36.180 versus 37.070 ms**
(**51.94 versus 50.69 GB/s**) for UCCL Lite and tuned NCCL CE.
At 64 MiB they are 2.300 versus 2.430 ms; at 1 KiB,
9.10 versus 186.90 µs. There are no remaining losing sizes in this
recorded configuration. This remains an experimental, hardware-specific policy.

This experiment follows the [earlier staged-CE measurements](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/pro6000_allreduce_ce_cpp_20261116/README.md).
The earlier results remain intact. This run adds a topology-aware CE ring,
corrects CPU affinity enforcement, and independently recalibrates both backends.
The NCCL reference calls `ncclAllReduce` directly from the matched C++ harness;
it is **not the official nccl-tests executable**.

[all five measurements](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116/summary.csv)

[Archived figure: Frozen-policy FP32/C++ comparison](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/assets/collectives-pro6000x8-allreduce-fp32-ce-ring.svg)

## Bottleneck found before changing the algorithm

The initial phase counters conflated copying with waiting. Nsight traces
separated actual GPU payload copies from the surrounding kernel time. In the
completion-safe staged implementation at 1 GiB, the eight-rank mean kernel
span is **44.75 ms**, of which **42.31 ms (94.5%)** is outgoing payload copying.
Those six copies do not overlap on a rank. PIX/NODE copies sustain about
**51 GB/s**, but the synchronized eight-rank SYS phases sustain only about
**25 GB/s per rank**. Local reduction alone cannot explain or close the gap.

The previously tried 4-MiB pipeline improves SYS throughput but uses 1,536
payload copies per rank/call plus their signal copies. Its actual payload
copies still serialize on each GPU, even with three MSCCL++ contexts, and the
PIX transfers slow down. Its slower result is retained, not selected.

The new Gray-code ring is `0 → 1 → 3 → 2 → 6 → 7 → 5 → 4 → 0`.
It uses one directed SYS edge per socket while the other GPUs transfer within
their sockets. Under the same enforced placement, its two SYS edges sustain
about **52 GB/s**. A diagnostic 2-lane/256-MiB-tile ring control with one copy stream has a mean
36.66-ms kernel span and 36.14 ms of payload copying per rank. These are
separate instrumented diagnostics, excluded from calibration and validation.

[Archived figure: Measured copy-time bottleneck](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/assets/collectives-pro6000x8-allreduce-ce-bottleneck.svg)

Both schedules transfer **1.75N bytes per rank**. The ring does not reduce
that volume. It actually increases aggregate SYS payload from 2N to 3.5N
across the eight ranks; distributing that traffic through the operation is
faster than the previous globally serialized topology phases. The benefit
comes from scheduling and overlap, not a smaller theoretical peer payload.

The final Nsight 2026.6.1 traces contain all 39 calls on all eight ranks, no
trace warnings, and exactly 1.75N peer bytes per rank per timed call (or per
call averaged over the complete batch for NCCL). The analysis uses the last
20 calls. For NCCL, a staging copy can outlive its local kernel, so its trace
analysis encloses the timed batch across all ranks. This diagnostic GPU span
is distinct from the host-timed benchmark below.

## Implementation

- `ring-ce` uses MSCCL++ PortChannels and copy engines. Independent lanes own
  disjoint tensor tiles. ReduceScatter partial sums occupy distinct immutable
  scratch positions. AllGather writes directly into the in-place output.
  A final peer acknowledgement proves outgoing copies drained before reuse.
- With `--streams 2`, ring lanes alternate between two MSCCL++ contexts,
  each owning a separate CUDA copy stream. This removes shared-stream ordering
  dependencies between lanes; the GPU payload copies themselves still serialize.
- AllGather publishes the last SM reduction once. A leader then advances
  CE-produced shards without synchronizing every worker CTA at each step.
  This removed measurable overhead at 32 MiB while retaining the final
  completion acknowledgement.
- `ring-host-ce` explicitly queues CUDA peer copies and GPU signal/reduction
  kernels on lane streams. The caller stream waits for every lane. All copies,
  kernels, enqueue costs, and event joins are included in timing. AllGather's
  signal-only kernel uses one CTA.
- LL, PUT, and the previous staged kernels remain candidates. Their algorithm
  bodies are unchanged. No production dispatch policy or application
  integration is added. UCCL Lite does not link or call NCCL.

The kernels and harness are in `include/uccl_lite/pro6000_allreduce.cuh` and
`src/collectives/pro6000_allreduce.cu`. The shared checker and placement code
are in `include/uccl_lite/allreduce_validation.cuh`.

## Placement correction

An affinity audit found that MSCCL++ expanded the initial CPU masks to the
GPU's NUMA node during initialization. In an unprofiled control, some main
and proxy threads had 192 allowed logical CPUs despite the requested
four-core mask. Thus the earlier report established matching *initial*
requests, not persistent matching worker masks.

Both backends now restore the original per-rank mask on every live thread
after warmup, outside timing, and verify every thread after the timed batch.
The allowed physical CPU ranges are 0–3, 4–7, 8–11, 12–15, 96–99, 100–103,
104–107, and 108–111. A mismatch fails the run. Raw affinity records retain
the pre-enforcement counts and enforced widths. Both backends are freshly
calibrated after this change; no earlier validation points enter this curve.

## Measurement and independent validation

Hardware: eight RTX PRO 6000 Blackwell Server Edition GPUs on `wxz-pro6000`,
dual AMD EPYC 9B45, four GPUs per NUMA node. XOR-1 peers are PIX, XOR-2 NODE,
and XOR-4 SYS. Driver 580.178.04, CUDA compiler 13.3.73/SM120, C++20 `-O3`,
NCCL 2.28.9, and the existing patched MSCCL++ installation from
`/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910/runtime_env.sh`.
The archive records GPU UUIDs, topology, CPU layout, library/header hashes,
source snapshots, binaries' hashes, and exact commands.

Eight C++ processes perform FP32 in-place SUM at all 21 powers of two from
1 KiB through 1 GiB. Each batch has 10 warmups followed by 20 continuously
queued calls. `steady_clock` encloses enqueue through final caller-stream
synchronization; reporting takes the slowest rank. Allocation, registration,
input initialization, checking, CPU-affinity enforcement, and rank barriers
are outside timing. All required staging, communication, and stream joins
are inside. Every process allocates full 1-GiB tensor capacity even when a
probe measures fewer sizes.

The checker uses signed, position/rank/seed-dependent exactly representable
FP32 values. Each size checks three changing seeds and three successive
calls per seed, then the complete output after the warmup plus timed batch.
It checks every element, nonfinite results, repeated buffer reuse, and
4-KiB guards around the active output and all seven UCCL staging regions.
Timed intermediate outputs are not copied out individually.

Calibration has **26 NCCL configurations and 34 UCCL Lite configurations**,
each run in three fresh-process rounds with deterministic shuffled order.
NCCL extends the previous 19-candidate search with channel/buffer choices
around its large-message optimum. Its accepted runs require actual
`P2P/CUMEM/CE` edges in all eight rank logs and per-size Simple-protocol
dispatch traces. CUDA
payload-copy traces independently corroborate CE use. The reference header's generic mode/geometry fields are unused; its captured
environment and dispatch logs identify NCCL's settings. Upstream NCCL 2.28.9
[skips user-buffer registration when the CE path is active](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/register/register.cc#L118).
The generic NCCL
kernel symbol can contain `LL`; per-operation dispatch logs and actual CE
copies identify the executed transport/protocol. NCCL dispatches the work-batch
function independently of the entry-kernel specialization ([NCCL 2.28.9 source](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/device/common.h#L363)).

The lowest median calibration time chooses each exact size independently
for each backend. Source/binary/platform hashes and policies are frozen
before validation. Five independent rounds sweep every selected fixed
configuration across the full grid (LL stops at its 4-MiB packet capacity).
Backends interleave and reverse order in alternate rounds. Only frozen
size/configuration pairs form the reported curve. There is no final-run
variant selection, replacement of failures, or average-speedup success test.

The final epoch has **180 successful calibration runs** and
**105 successful validation process runs**. All
**166,800 per-rank full-output checks**, plus every guard and
affinity check, pass. The audit verifies **1,024 NCCL rank logs**.
The frozen policy SHA256 is `fd23e24adb84a4301ecc769d2657ec8760dfecb08b13d6bcc80c81e0d5aec901`.

Centers are medians of five slowest-rank run means. Bands show observed
ranges, not confidence intervals. The report lists paired wins and the
stricter condition that UCCL Lite's worst run beats NCCL's best run.

## Correctness, failures, and limitations

The final ring kernels pass six suites on the documented four-L4 topology
at `l41`: single-lane, pipelined, wide-thread, two host-queued ring cases, and separate copy streams,
all through 1 GiB with complete outputs, guards, and reuse checks. Six
additional LL/PUT/chunk/host suites passed before the last ring-only edit.
Four-rank testing uses CUDA 12.9/SM89 and an isolated dependency under `build/`.
This is single-node eager validation; no graph or multi-node claim is made.

The archive retains slower configurations and failed setup/profile attempts:
an initial isolated build omitted a Makefile include, and the resulting
old-binary ring probes exited with unsupported-mode errors; one four-GPU
suite stopped before its next launch while GPU process cleanup was still
visible; old Nsight export attempts failed or produced incomplete child
traces. Normal child exit now allows profiler hooks to flush; the compatible
profiler's final traces pass complete-call/payload audits. An intermediate
affinity probe mistakenly reused the runtime-expanded main mask; the final
implementation preserves the original requested mask instead. None of
these attempts supplies a final performance point.

The first complete ring epoch (`previous-epoch/`) passed 150 calibration and
115 validation runs, but won only 3/5 pairs at 64 MiB. Its full 20/21 result
is retained. The final implementation adds separate copy streams, then
restarts both backends' calibration and all five validation rounds. No
measurements from that earlier epoch enter the final curve.

The 64-MiB instrumented two-stream run is slower than its shared-stream
control (about 2.53 versus 2.44 ms), unlike the uninstrumented probes. These
complete, correct traces are retained under `profiles/stream-comparison/`.
Profiling perturbs this case; its traced latency is not used to select or
validate performance. The source change removes ordering between lanes; we
do not infer an exact control-versus-compute breakdown from these traces.

The historical staged epoch's buffer-reuse failure and its repair remain in the
[earlier archive](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/pro6000_allreduce_ce_cpp_20261116/README.md). Historical
measurements validate only their recorded source and placement.

## Reproduction

On the recorded eight-GPU host, from `experimental/lite`:

```bash
bash scripts/run_pro6000_large_experiment.sh build/pro6000-large-run
```

This builds with Make, runs separate calibration, freezes both policies,
and executes five validation rounds. Use a new output directory. Runs own
`/home/wxzheng/uccl-eval.lock`, check GPU occupancy, refuse overwrites, and
terminate only their own process groups on timeout. No clock or system
configuration changes are required.

Diagnostic capture and analysis (profiler path supplied explicitly; the legacy
clock64 phase counters cover staged/chunk modes, so ring diagnostics use Nsight):

```bash
python3 -B scripts/profile_pro6000_allreduce.py \
  --output build/profile-ring --config ring-ce-2-256m --nsys /path/to/nsys
python3 -B scripts/analyze_pro6000_timeline.py \
  --input build/profile-ring/timeline-ours-ring-ce-2-256m.sqlite \
  --kernel arRingCE --output build/profile-ring/analysis.json
```

Offline audit and figures:

```bash
make pro6000-ar-evidence-check
make pro6000-ar-large-summary
make pro6000-ar-large-figures
```

`measurements.tar.gz` includes every calibration/validation log, per-rank NCCL
transport logs, frozen policy, source/hardware identities, correctness logs,
exploratory regressions, and exact GPU kernel/payload-copy rows from profiling.
`scripts/compact_pro6000_trace.py` preserves those rows without averaging and
records the full SQLite hash and extraction queries. Full and compact trace
analyses are checked for identical results. The large full NCCL profiler
report remains under `build/large-opt-20261116/build/profile-v2-nccl/` in the remote
`/home/wxzheng/uccl-lite-ar-fp32-20261116` experiment;
its hashes and exact reproduction command are packaged. Generated tools,
binaries, full traces and figure
previews stay under `build/`; published figures use `scripts/figure_style.py`.
