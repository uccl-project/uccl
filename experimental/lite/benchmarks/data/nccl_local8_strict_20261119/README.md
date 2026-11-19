# Strict eight-L4 NCCL drop-in qualification

**Qualified on the recorded `wxz-l4` configuration:** all **126/126** points
meet the zero-slowdown gate against freshly rerun original README UCCL Lite
binaries. The candidate runs unmodified MPI nccl-tests with no native NCCL
dependency or fallback. This qualification covers BF16 AllGather and SUM
AllReduce/ReduceScatter, eager execution, one GPU/process, every power of two
from 1 KiB through 1 GiB, and both legal layouts.

| Collective | Passing points | Worst candidate/original UCCL latency |
| --- | ---: | ---: |
| AllReduce | 42/42 | 0.97021 |
| AllGather | 42/42 | 0.98944 |
| ReduceScatter | 42/42 | 0.97280 |

These are per-point aggregate measurements, not a guarantee about every
individual call. **Native NCCL remains faster at 25/126 points**, with a worst
candidate/native ratio of 1.58285. The acceptance baseline follows the
multi-node port: the original UCCL implementations. No native-NCCL parity claim
is made. [Full original-UCCL comparison](comparison.csv),
[native comparison](native_comparison.json), and [gate result](verdict.json).

## Hardware, software and timing

- Eight NVIDIA L4s on `wxz-l4`, four per NUMA node; 48 reported Intel Xeon cores/96 logical CPUs, reported at 2.20 GHz. Driver 580.178.04; CUDA 12.9,
  `sm_89`; NCCL 2.28.9 headers/native control; Open MPI 4.1.2.
- Original AR/AG executable SHA-256:
  `3d1b584e218df5c3cec99a7a7b65423f7b36a031f20407a8924527c7e963ad44`.
  Original RS recovery executable:
  `f6e80074af2c70e625932743f6af343b4a161e50cc7fc63d5b9517a02a94a45d`.
- Original execution uses one process/eight GPUs, with AR in place and AG/RS
  out of place. Both candidate layouts are compared against those original
  paths. Candidate execution uses eight processes/one GPU each.
- Both sides use input-ready, maximum-rank per-call CUDA-event intervals,
  including host submission and all ABI staging/copies. Preparation, input
  generation and verification are excluded. Every warmup and timed call checks
  the complete output and its adjacent guard.
- Two independent jobs per layout/backend, reversing backend order in the
  second job. Each size/job has three repetitions of five warmups and ten
  timed calls. Each point is the median of repetition medians, then the median
  of job medians. **11,340 timed calls and 5,670 warmups** are retained across
  the 18 matched jobs. No trimming or best variant per size is used.
- Native NCCL is measured separately using unchanged nccl-tests' maximum-rank
  batch timing, five warmups and twenty iterations, in
  NCCL/UCCL/UCCL/NCCL order for each collective. Algorithms, protocols and
  channels remain at default; both backends use Socket networking with IB and
  the profiler plugin disabled. This batch timing is not substituted for the
  original-UCCL per-call gate.

[Hardware and identity](hardware.json), [provenance](provenance.json),
[source hashes](source_hashes.json), and [ELF dependencies](elf_dynamic.txt)
identify the measured configuration. The replacement library SHA-256 is
`a613c4f85541aaf39c5e08b4db2d12ee817b29942cbd0cb8dde423df0cb854c7`.

## Correctness and reuse

The implementation reuses the multi-node strict ABI source, HostControl
bootstrap/consensus, identity launcher, version script, ABI client and route
auditor. Its local backend uses the existing UCCL Lite multiprocess packet
transport. Construction is CPU-only; the first eligible call explicitly
prepares buffers, events and workers. Unsupported calls return errors.

The exact qualified library passed:

- Twelve complete native/candidate nccl-tests sweeps and three additional
  1-KiB–1-MiB sweeps through the `libnccl.so.2` alias with `LD_PRELOAD` removed.
  nccl-tests revision: `afd59abab774a6a4b492eb344bd0c845a0b7bb4d`, unchanged.
- Exact per-rank/per-size route counts, zero fallback, and native-library
  absence on all candidate ranks.
- 234 ABI calls/rank over two communicator lifetimes: changing streams,
  per-rank layouts and alignment, recycled allocations, delayed ranks,
  small/large transitions, full buffers and guards.
- 128 mixed queued calls/rank across alternating streams without host
  synchronization between calls, with distinct buffers and a delayed rank.
- Explicit rejection of unsupported dtype, product reduction, grouped
  collectives, nonempty capture and a missing profile. Rejected first calls
  do not prepare packet resources; zero-count empty graphs remain valid.
- Runtime `/proc` audits in every full candidate batch/per-call job: all
  non-worker threads avoid all worker/helper physical cores and SMT siblings.

The optional core's policy (35 cases), dispatcher (74 cases), CUDA dispatcher,
and four-/eight-GPU eager and graph checks also passed. Graph checks cover
1,000 mixed captured collectives, five replays and queue wrap. These core graph
checks do not imply nonempty graph support in the strict ABI adapter.

The shared network frontend and FP32 ABI client compiled. A new two-host
runtime check was not run: `l41` was reachable, but SSH authentication for
`wxzheng@10.10.55.1` was denied. The earlier two-host qualification remains tied
to its archived binary; this report qualifies the local eight-L4 replacement.

## Failed candidates and the worker-isolation fix

The [experiment index](experiments.json) and raw archive preserve slower
candidates. The initial port failed 29/126 screening points. A larger 2-MiB
packet trial failed 41/126. The first complete qualification failed 5/126,
including repeated 32-KiB AR stalls; a later complete campaign failed 3/126,
including a 1-MiB AR stall and small AG regressions.

Nsight reproduced approximately 3-ms collective kernels while launch latency
remained below about 30 microseconds. The CPU trace isolated a delayed helper:
one main worker spent 3,015 microseconds waiting/working, while its helper
started late and completed in 75 microseconds. The original isolation code
excluded only each process's own worker core. Other MPI/application threads
could still occupy another rank's worker or helper core.

The qualified configuration excludes every reserved worker/helper core and
SMT sibling from all non-worker threads in every rank. Workers and helpers
poll after their first job; GPU-flag waits also poll. Up to sixteen pinned CPU
threads can remain active until teardown. AG-only jobs leave reduction workers
asleep. This CPU cost and placement are part of the qualified configuration.

AG uses a single-CTA small path, parallel readiness polling, a per-source
single-packet schedule at 4–8 MiB, and a bulk pipeline thereafter. Legal
in-place AG avoids an unnecessary staging copy; overlapping RS stages only
its output. These choices form one fixed library configuration. Diagnostic
instrumentation and rejected libraries do not contribute timing samples to
the qualified curve.

Early negative-test controllers falsely treated MPI teardown as foreign GPU
work. Their interrupted records remain; the controller now tracks descendants
before CUDA contexts appear and waits briefly for their NVML teardown.

## Reproduce and audit

See [build and API scope](../../../integration/nccl_lite/LOCAL8.md). On the
recorded host, the supplied [profile](profile.tsv) is bound to the exact library,
launcher, software, environment, CPU topology and GPU UUIDs. Changing these
requires a new profile and qualification.

```bash
python3 -B benchmarks/data/nccl_local8_strict_20261119/reproduce.py

python3 -B tests/nccl_mscclpp/audit_local8.py \
  --tests build/local8_global_release_20261119 \
  --paired build/local8_global_reference_20261119 \
  --library build/nccl_local8/libuccl_lite_nccl_local8.so
```

The [independent audit output](audit.txt) records 12 full nccl-tests sweeps,
18 paired jobs and **0/126** zero-regression failures. It also checks ABI,
queue, rejection, loader and affinity evidence. Frozen executables and the
full Nsight reports remain under `build/`.

[measurements.tar.gz](measurements.tar.gz) contains raw logs, manifests, source
snapshots, checks, negative campaigns and diagnostic traces; it contains no
compiled libraries or executables. [raw_checksums.json](raw_checksums.json)
authenticates its members. Existing historical figures and measurements are
unchanged.
