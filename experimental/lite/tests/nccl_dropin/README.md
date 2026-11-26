# NCCL ABI checks on L4

Run unchanged nccl-tests against native NCCL and `libuccl_lite_nccl.so` in
NCCL → drop-in → drop-in → NCCL order. This harness deliberately removes any
inherited drop-in profile: it establishes ABI and fallback behavior, not
accelerated UCCL performance. The existing calibrated eight-L4 profile must
not be used as evidence for four L4s or two four-L4 hosts.

From `experimental/lite`:

```bash
make nccl-dropin-check NCCL_TESTS_BIN=$HOME/third_party/nccl-tests/build \
  DROPIN_TEST_OUT=build/dropin-4l4

# Separate graph-capture check, using a fresh output directory.
make nccl-dropin-check NCCL_TESTS_BIN=$HOME/third_party/nccl-tests/build \
  DROPIN_TEST_OUT=build/dropin-4l4-graph DROPIN_TEST_ARGS='--graph 5'
```

Defaults cover BF16 and FP32 AllReduce, AllGather, ReduceScatter, Broadcast,
and Reduce, 8 bytes through 64 MiB with doubling, both layouts, five warmup
iterations and twenty timed iterations per fresh process. Timings are the
nccl-tests host enqueue/completion batch means (default rank-average
reporting). Summary arrays list out-of-place then in-place;
each value is the median of two job means. No variant selection is used.
All raw repetitions, commands, binary hashes and observed topology are saved.
Small-message timings can be noisy; the ratio is evidence to inspect, not an
automatic performance certification or calibration profile.

For one process per GPU, use `--mpi`. For two four-GPU hosts, build nccl-tests
with `MPI=1`, stage the same tests, CUDA/NCCL libraries, interposer and core
libraries at identical absolute paths on both hosts, then run:

```bash
make nccl-dropin-check NCCL_TESTS_BIN=/absolute/path/to/mpi-tests \
  DROPIN_TEST_OUT=build/dropin-2x4l4 \
  DROPIN_TEST_ARGS='--hosts node0:4,node1:4 --mpirun /path/to/mpirun'
```

The MPI build must link `libmpi.so`. The harness also verifies that all four
or eight collective ranks actually appear, preventing independent one-rank
jobs from being accepted as a multi-rank test. Use `--suffix _perf_mpi` if the
MPI tests were renamed. Additional Open MPI transport settings may be supplied
with `--mpi-args`; NCCL/CUDA environment settings are applied equally to both
backends. A failed process, timeout, nonzero validation count, missing result
rows or missing correctness summary fails the run. Output directories must
be new so prior measurements cannot be silently overwritten.

The interposer now admits explicitly calibrated four- or eight-local-rank
profiles. Four-rank staged execution and grouped single-rank initialization
are supported, but the initial four-L4 calibration selected no shapes. This
harness remains profile-free and validates fallback. Two-host communicators
still use real NCCL and require a network backend integration, exact local
calibration, ordering/capture correctness and independent performance evidence.

For explicit local candidate correctness checks (including staged route
counters), run `check_candidate.py` with the same `--real-nccl`, `--tests`,
`--mpirun` and a fresh `--out`. This creates a calibration-only profile for
BF16 counts 32 and 131072, runs AR/AG/RS eager and graph tests, and requires
nonzero UCCL counters from every rank. It deliberately tests uncalibrated
candidates: its synthetic profile cannot be deployed and its short timings
must not be treated as performance evidence. Use `calibrate_dropin.py` to
produce an actual deployment profile; it checks both layouts and all four
alignments, and now also verifies that every timed candidate call was routed.

`local_four.py --real-nccl ... --out ...` stresses the four-rank scalar GPU
packet path with guarded buffers, 10 tail/boundary counts, three generations,
all three operations, two streams, and different in-place/alignment layouts
per rank. The RS count above packet capacity also checks the generic staged
transition. `--sanitizer /path/to/compute-sanitizer` runs per-rank memcheck;
CUDA API diagnostics are disabled because native NCCL deliberately probes
unavailable kernel images. Memory diagnostics and nonzero error exits remain
enabled. Every rank must report both route counters and a clean summary.

`validate_profile.py --real-nccl ... --profile ... --cases ... --out ...`
independently tests a nonempty deployment profile, including every selected
shape. Cases must include AG, RS and AR for the benchmark's transition checks.
It runs native → deployed → deployed → native, records all layout/alignment
repetitions, verifies selected-route counts, and exits nonzero if any case's
worst drop-in job median exceeds its best native control by more than 5%.
This includes fallback shapes: acceleration gains alone do not certify a
mixed workload. It never writes a new admission profile.

For admission using actual nccl-tests batch timing on four L4s, use
`integration/nccl_lite/calibrate_local_nccl.py --library ... --real-nccl ...
--tests ... --mpirun ... --out ...`. It runs two independent ABBA rounds at
each BF16 offset 0/1/2/4/8 elements, both layouts, with 20 warmups and 100
measured calls at maximum rank. Each admitted shape must beat every native
control by at least 5% in every alignment/layout. `local_profile.py` accepts
the same paths and `--profile ... --max-bytes 1G` for independent mixed-curve
deployment validation; it exits nonzero for any strict 5% regression.
Candidate and rejected results remain part of the record.

For two four-L4 hosts, build `make nccl-dropin-multinode` with an IB-enabled
`MSCCLPP_IB_BUILD`. This produces `build/nccl_lite/libuccl_lite_nccl_multinode.so`;
it retains the local BF16 paths and adds calibrated FP32 network AR/AG/RS.
Stage the network library, interposer, dependencies and Python launch helpers
at identical paths on both hosts. `network.py` probes both ordered hosts,
creates a calibration-only profile and runs unchanged MPI nccl-tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/nccl_dropin/network.py \
  --hosts node0:4,node1:4 --iface ibp55s0f0 \
  --mpirun /path/to/mpirun --real-nccl /path/to/libnccl.so.2 \
  --tests /path/to/mpi-nccl-tests --out build/network-check --abba
```

Defaults measure 5 warmup and 20 timed calls, both layouts, sizes 128 B–128 KiB.
`--warmup 20 --iterations 100 --unalign 1` is one misaligned test. Host batch
mean times include enqueue and completion; `-a 3` selects maximum rank.
`--graph 3` checks native graph replay. A graph captured before any eligible
call permanently keeps that communicator native; an already prepared route
transitions by importing its completion event as an external dependency.
nccl-tests eagerly warms the entire size range before graph capture. The
graph check therefore permits exactly those warmup route calls and requires
the first size's eager synchronization call when eligible. Empty collectives
do not disable routing; the first nonempty capture permanently selects NCCL.
No routed calls are permitted during capture or subsequent graph tests.

`make build/network_dropin_correctness MPICXX=/path/to/mpicxx` builds a client
that links only real NCCL/MPI/CUDA. Run it through `network.py --stress-binary
build/network_dropin_correctness` for full-buffer signed/index-dependent,
guarded tests with changing pointers, streams, sizes, in-place layouts, two
accelerated communicator lifetimes and a third native-only graph lifetime.
`--stale` or `--missing` tests one-host identity/file failure and requires
fallback on all ranks with no prepared network route.

`integration/nccl_lite/calibrate_network_dropin.py` takes the same hosts,
interface, MPI/test/NCCL paths and a fresh `--out`. It runs two independent ABBA
rounds for each of four pointer offsets (0/4/8/16 bytes), 20 warmups and 100
measured calls per job. A deployment row requires every layout/alignment's
worst UCCL job to beat its best native control by at least 5%; all negative
results are retained. `network.py --profile profile.tsv --abba` independently
validates that deployed profile without a calibration override. Candidate
profiles cannot be used by the deployment launcher.

The unified binary can also be chosen for local calibration/deployment via
`calibrate_dropin.py --library build/nccl_lite/libuccl_lite_nccl_multinode.so`
and `dropin_profile.py --library ...`. Profiles fingerprint the selected
binary; an older binary's measurements never admit the new build.

Calibration reserves the same explicit capacity in candidate and deployment
profiles. Removing rejected rows therefore does not silently change the bank
layout/registration configuration. Independent validation remains necessary
because real native-call history and application buffer placement can differ.
Both local and network launchers keep stale/missing ranks in the native vote;
profile rejection on one host must not strand valid peers in a collective.

`make -C integration/nccl_lite host-control-check` runs CPU-only 4-/8-process
transport checks without CUDA, NCCL, MSCCL++ or MPI. Use `--control-root` on
the local/network nccl-tests drivers and both calibration drivers to test
the independent CPU metadata path. Its common rank-0 IPv4 address is part of
the profile identity and must reach every rank through MPI's exported
environment. `network.py --operations all_reduce` can test an AR-only native
algorithm filter, for example `NCCL_ALGO=Tree` with `--stale`.

For controlled local subset screens, `local_profile.py --cases cases.json
--capacity 32768` accepts candidate `[operation,count]` pairs and preserves
the original candidate allocation capacity. These flags cannot modify a
production `--profile`. Candidate overrides and profiler identity bypasses
never constitute deployment admission.
