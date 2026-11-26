# Collective preparation and automatic local selection

Caller-baseline preparation now also qualifies a
[topology-derived source-ordered tree](peer-tree-runtime.md) on connected sparse
peer graphs, alongside the full-mesh direct and other registered candidates.

The interface now supports [dtype-aware BF16/FP32 selection](scalar-runtime.md).
FP16 can use a caller-native baseline without admitting a BF16/FP32 executor.
The descriptions below focus on the original BF16 peer/host comparison.

An optional [caller-owned native NCCL fallback](native-fallback.md) now supplies
the comparison baseline. That mode can retain native execution without peer/host
initialization, and it can compare host alternatives when peers are unavailable.
The original no-fallback mode described below still uses staged peer as baseline.

`PreparedLocalRuntime` compares the BF16 `peer-staged` and generic `host-packet`
providers on the participating topology. It is also registered as
`prepared-local` in the unified NCCL ABI. This is automatic selection during an
explicit collective preparation boundary; it is not transparent native-NCCL
fallback. The core also supports separately qualified
[owned graphs](prepared-graphs.md); external ABI capture remains unsupported.

The core lives in `include/uccl_lite/runtime/prepared_local.hpp` and
`src/runtime/prepared_local.cu`. The NCCL adapter lives in
`src/runtime/prepared_local_backend.cc`; collective algorithms remain under
`src/collectives/`. The standalone archive has no NCCL/MPI dependency. Its
optional host candidate links the CPU executor and host transport.

```bash
make prepared-local-runtime prepared-local-check nccl-runtime \
  MSCCLPP_IB_BUILD=/path/to/mscclpp/build-ib \
  MPICXX=/path/to/mpicxx
```

## Preparation contract

One process owns each GPU. The context contains a shared session ID, logical
rank, the ordered CUDA device list and maximum byte capacity. Construction is
CPU-only. All ranks call `prepare(LocalRequest)` with the same operation, NCCL
element count and array of per-rank layouts. Each layout records legal in-place
placement and the exact minimum input/output pointer alignment, capped at 256
bytes. These descriptors may differ between ranks. Invalid extents, conflicting
descriptors, differing GPU UUID order/software, changed topology/environment,
active work and capture reject before qualification.

The returned plan index selects one backend on every rank. Calls must supply
buffers matching that prepared layout; a mismatch is a contract error, never a
local decision to switch transport. Enqueue only validates and executes the
stored plan, with completion events chaining different backends and streams.
It performs no host rendezvous, resource probes, allocations or worker startup.
Repeated preparation reuses a live plan after identity and rank consensus.
Existing plans remain usable after other workloads are prepared.

Without a caller fallback, the peer baseline requires a complete CUDA peer-access graph and currently
supports 2–32 ranks as a storage bound. Host admission additionally requires
mapped-host support, known NUMA placement and AVX2. `UCCL_LITE_MP_*` experiment
knobs disable the automatic host candidate: those retained specializations need
separate configuration-specific calibration. Missing peer connectivity rejects
this selector rather than enabling a host path solely from GPU count.

## Resource model and measured admission

At the first idle prepare, rank zero probes GPU memory and directed peer paths
using private buffers, or loads a supplied hardware profile. Other ranks remain
idle. The common planner accounts for local staging read/write traffic, source
GPU reads, destination writes and contention on shared PCIe paths. Unknown
resources remain unknown. The reported `peerResourceEstimateUs` is advisory:
it omits arithmetic, launch, semaphore and access-granularity costs. It is not a
latency guarantee and never qualifies the host provider by itself. CPU/NIC
model integration remains separate unfinished migration work.

Each exact request then runs three ABBA rounds. Each block has two warmups and
three timed calls. A is the peer baseline; B is the host candidate. The host
provider is destroyed for both A blocks, including buffers and workers, and
created outside timing for B. Every warmup and timed call uses private buffers,
new input values, complete input/output/guard verification and rank-wide error
agreement. Timings are maximum-rank host completion in microseconds, including
enqueue and completion wait; initialization, CPU control collectives, input
generation and validation are outside the timed interval. All 18 timed samples
per provider are retained, without selecting a best repetition.

Host is selected only when correct, its median improves by more than 3%, and
its complete sample range is below the peer range. Otherwise peer remains the
baseline. A baseline correctness failure rejects preparation. Rank zero
broadcasts the decision. Unselected host resources are released; a host provider
is retained only if some prepared plan needs it. Future calls do not initialize
host resources merely because they use the peer baseline.

The current four-L4 results select peer for the small tested requests and host
for several larger requests. The baseline is the new multiprocess staged peer
algorithm, **not NCCL or the tuned single-process PeerCollectives families**.
This does not reverse earlier negative CPU-router results against different
baselines. No performance conclusion is claimed for arbitrary sizes, GPU models
or rank counts. The resource estimate noticeably underestimates scalar remote
load and synchronization costs; exact timings, rather than extrapolation of
that estimate, control admission.

## NCCL profile format

The generic profile tool selects `prepared-local` for eligible BF16 layouts
unless `--provider` explicitly requests an experiment implementation. It writes
`UCCL_LITE_RUNTIME_PROFILE_V2`. The provider/layout and host identity records have
the same structure as V1; each workload row additionally contains one encoded
layout per rank: `2 * alignment + in_place`. V1 and historical headers remain
available for explicit retained providers.

The shapes JSON accepts `[operation, count, rank_layouts]`, where each rank
layout is `[in_place_boolean, alignment_bytes]`. For example:

```json
[[1, 4096, [[false, 256], [true, 16], [false, 2], [true, 256]]]]
```

An omitted layout array means out-of-place, 256-byte alignment on every rank.
This matches ordinary CUDA allocations; offset pointers need their actual
alignment declared. Operation values are AG=1, RS=2, AR=3. The strict ABI retains
its complete-extent multiple-of-128 requirement. The core API also validates
scalar tails. One op/count has one collective layout contract in a profile;
duplicate shapes with different layouts reject rather than creating ambiguous
per-rank routing. Core callers can prepare separate handles for such layouts.

Probe and launch with the same CUDA-visible order and software/environment:

```bash
python3 -B integration/nccl_lite/runtime_profile.py probe \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --real-nccl /path/to/reference/libnccl.so.2 --output build/host.json
python3 -B integration/nccl_lite/runtime_profile.py candidate \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --records build/host.json --shapes build/shapes.json --dtype bfloat16 \
  --profile build/runtime.profile
```

`launch` revalidates the profile and `ncclCommInitRank` performs collective
preparation. V2 stores requested workloads, not trusted winners; each new
communicator qualifies its plans afresh. Persistent reuse across communicators
and native fallback remain outstanding. Unsupported calls reject without
initializing another backend.

## Validation scope

Local hardware is four NVIDIA L4s across two NUMA nodes on
`mibura-sky-test-02`, with two Xeon Silver 4410Y sockets. GPU order is shuffled
using `CUDA_VISIBLE_DEVICES=2,0,3,1`. Core checks exercise rank disagreement,
live-cache hits, 64/257/65536-element AR/AG/RS, mixed per-rank aliases/alignment,
old-plan reuse, stream transitions and a separate complete-buffer oracle. The
NCCL ABI client additionally queues producers and alternating transports, checks
bitwise AG, capture rejection/recovery and two communicator lifetimes.

CPU tests use synthetic rates to verify six-rank shared-link demand, memory
bandwidth constraints, missing-resource handling and exact sample admission;
these are not hardware measurements. GPU tests on up to four ranks do not
validate the 32-rank storage capacity or eight-rank experimental packet modes.
See the migration record for final run artifacts and sanitizer scope.
