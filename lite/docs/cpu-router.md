# Optional CPU-router module

The opt-in [reduction-placement experiment](reduction-placement.md) adds
destination-GPU arithmetic and interleaved scatter packets to the C++ module.
Existing CPU defaults and measurement-gated fallback behavior are retained.
The subsequent [GPU progress experiment and baseline comparison](baseline-comparison.md)
remove CPU packet-routing workers in an explicit GPU-reduction configuration.

CPU-router is an independent component for routing BF16 collective traffic
through NUMA-local mapped host memory. It is useful when direct peer paths are
unavailable or less efficient for the measured workload. It can be left entirely
uninitialized on systems where the existing backend wins.

## Components and build boundaries

| Component | Responsibility |
| --- | --- |
| `RegisteredMemory`, `MemorySpan` | GPU/NUMA allocation, registration, bounded views, retained lifetime |
| `Channel`, `DeviceChannel` | GPU/host transfers and completion tickets |
| `CpuExecutor` | NUMA-local worker lanes, bounded queues, FP32 accumulation with BF16 rounding |
| `RouterCore` | Reusable `forward`, `process`, and `multicast` operations |
| `PacketPipeline` / `CpuRouter` | AR/AG/RS packet programs, two-slot generation and reader-credit protocol |
| `BackendPolicy` | Pure host selection from capability and validated timing records |
| `CollectiveDispatcher` | Optional router around a caller-supplied existing-backend callback |
| `LocalCollectives` | Project-owned eager BF16 fallback for two four-GPU NUMA groups; no NCCL dependency |

Source lives under `src/cpu_router/`; public headers are under
`include/uccl_lite/cpu_router/`, plus
[backend_policy.hpp](../include/uccl_lite/backend_policy.hpp) and
[collective_dispatch.hpp](../include/uccl_lite/collective_dispatch.hpp).
The core is ported from the primitive implementation associated with source
revision `9d088c9`, including the medium-message ReduceScatter fix. Namespace,
exported C functions, and diagnostic environment variables use UCCL Lite names.
The local fallback is [separate CUDA code](../src/collectives/local_collectives.cu)
with a [public C++ interface](../include/uccl_lite/local_collectives.hpp).

```bash
make -j2 cpu-router CUTLASS_HOME=/path/to/cutlass
# build/cpu_router/libuccl_lite_cpu_router.a
# build/cpu_router/libuccl_lite_cpu_router.so

make local-collectives
# build/libuccl_lite_local_collectives.a

make -j2 adaptive-collectives NCCL_HOME=/path/to/nccl/build
make -j2 router-checks local-collectives-check
```

The module uses Linux/x86-64, AVX2, CUDA, pthreads, and CUTLASS 4.5.0. Its
packet pipeline retains resident-GEMM building blocks, hence the CUTLASS build
dependency. It does not link MSCCL++ or NCCL. The standalone adaptive
benchmark links NCCL only for an explicitly selected external control;
its default fallback is UCCL Lite local. The original `make`,
`single-node`, `fusion`, and their binaries do not link or initialize this module.

No framework patches, application integrations, or NCCL interposition shim are
included. The C wrapper is a direct low-level API, not an NCCL API replacement.

## Selection policy

`auto` is conservative and does not benchmark during a collective invocation.
The selection key includes collective, BF16 dtype, exact byte count, in-place
placement, topology identity, and the identity of the caller's fallback.
No interpolation or best-neighbor size policy is applied.

For AllReduce through 4 MiB and ReduceScatter from 256 KiB through 4 MiB,
the adaptive dispatcher prepares the router with 128 KiB packets by default.
On 8xL4 GPU, 128/256 KiB AllReduce also uses four CPU workers per NUMA node
and 16 transport blocks, based on paired local and NCCL measurements.
At exactly 128 KiB, ReduceScatter uses GPU reduction and GPU progress with
the regular packet limit. Other calls use the 512 KiB packet default and CPU
reduction. These choices happen in `prepare`, outside the timed invocation.
`--small-ar-packet-bytes 0`, `--small-rs-packet-bytes 0`, and
`--no-small-rs-gpu-progress` disable the respective example choices. Packet
geometry and reduction mode are part of the profile identity, so changing
them requires new calibration.

The optional `largeAllGatherPacketBytes` setting selects larger packets for
AllGather outputs of at least 4 MiB. On 8xL4 GPU, the benchmark sets it to
2 MiB and keeps the 512 KiB default below 4 MiB. The benchmark flag is
`--large-ag-packet-bytes 2097152`. This packet setting is also included in
the profile identity.

A [NUMA-local cache probe](../benchmarks/data/ddio_8l4_20261111/README.md)
measured the same CPU BF16 reduction after GPU writes to mapped host memory.
Evicting those writes increased reduction time at packet-sized working sets.
The probe establishes cache sensitivity, but this virtualized host does not
expose counters that would attribute the effect specifically to DDIO.

1. Unsupported dtype/topology/capacity retains the existing backend.
2. Only an exact local calibration record can enable the router. A newer
   matching record showing a regression disables it.
3. A candidate is selected only when it passed correctness, has at least three
   repetitions, median speedup is at least 1.03×, and its slowest run is faster
   than the fallback's fastest run. Overlapping ranges retain the fallback.
4. Unknown configurations and unmeasured sizes retain the existing backend.
   In the eight-GPU example that is UCCL Lite local; load the matching profile
   before relying on automatic AR/RS performance.

The archived eight-L4 profile describes its recorded hardware/software
configuration, but cannot capture a new host's PCIe contention, memory setup,
CPU model, or every NCCL setting. It remains a performance reference and is not
accepted as automatic routing evidence. Recalibrate when CPU, memory, or
workload conditions change.

The four-L4 host cannot match eight-GPU historical records. With no local
evidence, it stays on its existing backend and does not allocate router buffers
or start workers. Local evidence can opt in on any supported topology if it
actually demonstrates a stable gain; four GPUs are not categorically forbidden.

The [November 11 four-L4 calibration](../benchmarks/data/cpu_router_4l4_apply_20261111/README.md)
provides local BF16 evidence on `mibura-sky-test-02`. Its NCCL-backed
automatic replay was a diagnostic experiment. The [archived BF16 figures](../benchmarks/data/cpu_router_4l4_figures_20261111/README.md)
run the UCCL Lite CPU router at **all sizes** for AR, AG, and RS, including
losses; NCCL is only an external paired control. This machine has peer
access between all four GPUs, but the 8×L4 compact AR and GPU-progress RS
variants did not yield stable gains.
An [NCCL transport trace](../benchmarks/data/cpu_router_4l4_workers_20261111/README.md)
shows `SHM/direct/direct` on every Ring edge here, even though CUDA reports
peer capability. The transport distinction is host-packet upload-once fan-out
versus repeated SHM Ring hops, not router host traffic versus NCCL direct
GPU P2P.

The [worker-count follow-up](../benchmarks/data/cpu_router_4l4_workers_20261111/README.md)
shows why the eight-L4 default of 23 workers per NUMA is not a suitable
resource default for this four-L4 host: it occupies nearly all 24 logical
CPUs per node, while large AllGather is equally fast with eight workers
on distinct physical cores. The new **AllGather-only** eight-worker profile
had seven stable AG sizes in either placement under the older auto-policy
diagnostic. Fewer workers did not rescue AR/RS and made large AR slower,
so this host-specific profile does not change the module's global default.
The subsequent [AllGather transport-geometry sweep](../benchmarks/data/cpu_router_ag_4l4_geometry_20261111/README.md)
finds that the original 64-block grid and small per-source tiles limit the
four-L4 large-message path. A machine-specific 248-block upper bound
(180 actual blocks at 1 GiB), 4-MiB-cap profile
cuts out-of-place 1-GiB AG from about 42.8 to 37.6 ms in separate checked
paired windows, with NCCL near 48.1 ms in both. The historical auto-policy
replay would have chosen its NCCL control at 4/8 MiB, where the larger grid
regresses. Those losses remain visible in the archived BF16 router-only
figure. The [current FP32 AllGather figure](../benchmarks/data/fp32_packet_ag_per_source_4l4_20261111/README.md)
uses direct GPU paths through 8 MiB, then a per-source GPU-progress
host-packet variant from 16 MiB onward, with no NCCL fallback in the
UCCL curve.
For AllGather Copy, `agPerSourceProgress` lets each source GPU publish its
own system-release `uploaded` flag. Consumer blocks assigned to that source
wait only for that flag and start copying its tile immediately; one leader
then fans out readiness through GPU-local memory. A final consumer barrier
precedes reader-credit publication, so the two packet slots remain safe to
reuse. The [four-L4 FP32 ablation](../benchmarks/data/fp32_packet_ag_per_source_4l4_20261111/README.md)
records both the per-source path and the older all-sources-ready GPU path.
The standalone benchmark's `--router-only` mode forces the CPU router at
every size without constructing NCCL or another fallback. Its fresh
[four-L4 BF16 figures and replay logs](../benchmarks/data/cpu_router_4l4_figures_20261111/README.md)
use NCCL only as a separately timed external control.

`off` always retains the existing backend. `force` is a diagnostic override;
it still rejects unsupported dtypes, missing capabilities, or insufficient
capacity. It bypasses the performance decision, not correctness requirements.

## Reusable caller interface

One process owns all device ports. A caller provides its existing implementation
as a callback; the policy does not assume that callback is NCCL. For example:

```cpp
#include <uccl_lite/collective_dispatch.hpp>

uccl_lite::BackendPolicy policy;
policy.fallbackIdentity = "my-existing-backend:v1";
policy.measurements = uccl_lite::loadRouterProfile("local.profile");

uccl_lite::CollectiveDispatcher dispatcher(
    uccl_lite::inspectTopology(devices), policy,
    [&](const uccl_lite::CollectiveCall& call) { existingBackend.enqueue(call); });

// Outside graph capture, with valid persistent buffers and idle streams.
auto selection = dispatcher.prepare(call);
dispatcher.enqueue(call);
```

The callback must implement the supplied collective, dtype, placement, and
streams. `fallbackIdentity` must identify that actual implementation and change
when its algorithm/configuration changes. Do not label a custom callback with
an NCCL identity to reuse unrelated measurements.

For the eight-L4 topology, `LocalCollectives` provides that callback. It needs
two NUMA groups of four GPUs with peer access within each group. Its
`prepare()` allocates pinned host and GPU scratch outside timing; its
`enqueue()` is eager and synchronizes its submitted streams before returning.
The local path sends AllGather shards through NUMA-pinned host buffers. For
AllReduce and ReduceScatter through 2 MiB it uses eight pinned input buffers
and CPU reduction workers. Larger reductions sum within each four-GPU peer
group, exchange partials through host memory, then distribute results. This
path does not use NCCL or CPU-router workers. Other topologies need their own
caller-supplied fallback.

```cpp
#include <uccl_lite/local_collectives.hpp>

auto topology = uccl_lite::inspectTopology(devices);
uccl_lite::LocalCollectives local(topology, maxBytes);
uccl_lite::BackendPolicy policy;
policy.fallbackIdentity = uccl_lite::LocalCollectives::identity();
policy.measurements = uccl_lite::loadRouterProfile("caller-local.profile");
uccl_lite::cpu_router::CpuRouterOptions options;
options.maxTensorBytes = maxBytes;
uccl_lite::CollectiveDispatcher dispatcher(
    topology, policy,
    [&](const uccl_lite::CollectiveCall& c) { local.enqueue(c); }, options);

auto selection = dispatcher.prepare(call);  // Outside capture, streams idle.
if (!selection.cpuRouter) local.prepare();
dispatcher.enqueue(call);
```

The profile in this example must be calibrated for the caller's own callback
and topology identity. The standalone benchmark appends its own identity key,
so its profile is not interchangeable with another caller's profile.

`CollectiveCall::count` follows the usual collective API convention: full tensor
elements for AR, contribution elements for AG, output-shard elements for RS.
The policy normalizes these to complete per-rank bytes (AG output / RS input).
In-place aliasing and port device order are checked before dispatch. Unsupported
router types reach the existing callback without conversion.

`prepare` creates the router only for a selected router workload. `enqueue` never
implicitly initializes it. Inspect `decision().reason` and `routerInitialized()`
to audit selection. When introducing a new size that first selects the router,
call `prepare` at an initialization boundary before submitting that work.

The topology snapshot records GPU UUID/PCI/device order, NUMA placement, peer
capabilities, CPU model and permitted affinity, driver/toolkit, and router
geometry. Local profiles must match this identity. Recreate the dispatcher after
changing device visibility, affinity, or execution configuration.

## Paired calibration and the executable example

The included example uses BF16 and supplies `LocalCollectives` as its default
fallback on the eight-GPU topology. `--fallback nccl` retains the previous
external comparison mode. It allocates its own scratch tensors, verifies every
invocation outside timing, alternates backend ordering between repetitions,
and records the slowest-rank CUDA-event latency. Three run-level medians form
the decision's min/median/max range.

```bash
# Local paired measurement. The output must be a new file.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --begin 1048576 --end 1073741824 --warmup 5 --iterations 10 --repeats 3 \
  --calibrate build/local-out-of-place.profile

# Reuse the measurements. Nonwinning records select the fallback.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all --backend auto \
  --begin 1048576 --end 1073741824 --profile build/local-out-of-place.profile

# In-place is a separate workload key and needs its own measurements.
./build/uccl_lite_collectives --gpus 8 --fallback local --op all --in-place \
  --begin 1048576 --end 1073741824 --calibrate build/local-in-place.profile

# Force is useful for correctness and diagnostics, not a deployment default.
./build/uccl_lite_collectives --gpus 8 --fallback local --backend force --op all
```

`--devices 0,2,1,3,4,6,5,7` selects an explicit rank order. For the older
four-GPU NCCL comparison, pass `--gpus 4 --fallback nccl` explicitly.
`--backend off` measures the selected fallback. Calibration enables both
candidates intentionally; merely loading
a profile in automatic mode does not start CPU workers if all selected sizes lose.

Calibration retains both initialized candidates while comparing them. The CSV's
`dispatcher_router_initialized` field describes the dispatcher for that row;
the fallback row does not imply the separate router candidate has been destroyed.
In a subsequent automatic run, a disabled router is never created.

The example's eager submission/validation schedule differs from the archived
nccl-tests sweeps. Its calibration records apply to the measured fallback and
workload, not to a different process layout or CUDA Graph execution. No new
peak-bandwidth claim is inferred from these smoke/calibration runs.

## Lifetime and graph contract

Adaptive dispatch is currently eager-only on the CPU-router path. It rejects
capture rather than recording GPU work without the required CPU publication.
The low-level pipeline retains explicit `beginCapture/endCapture` and
`beginReplay/endReplay` APIs. Callers using those APIs must publish fresh CPU
jobs, launch all rank graphs, synchronize their streams, and finish replay
before reusing buffers. The supplied graph test exercises queue wrap and mixed
collectives; it is not an implicit graph adapter for the dispatcher.

Calls are serialized on stable per-device streams. Buffers and producer events
must outlive outstanding work. Synchronize before destroying a dispatcher or
changing buffer ownership. Initialization errors and numerical failures propagate;
the component does not silently retry partially executed collectives elsewhere.

The low-level module retains DMA and resident-GEMM experiments, but automatic
selection uses only the validated SM transport. The historical DMA medium-size
graph issue remains outside that path. The current module is single-node and
single-process; it is separate from the existing MPI two-node CPU-router program.

## Transport and calibration options

| Option | Default | Purpose |
| --- | ---: | --- |
| `transportVectorBytes` | 16 | Set 32 for native 256-bit accesses on SM100+ with CUDA 12.9+, with alignment/tail fallbacks. BF16 dtype is unchanged. |
| `singleNumaReduction` | false | CPU reduction directly into the pinned result packet on one NUMA node; retain staging across multiple nodes. |
| `singleCtaBytes` | 0 | Optional complete-tensor cutoff for one CTA to upload and consume packets serially. |
| `transportThreads` | 256 | Warp multiples from 32 to 1024, subject to actual kernel residency. |
| `transportBlocks` | 64 | Grid upper bound through 1024; CUDA occupancy caps the actual arithmetic/vector variant. |

The benchmark exposes these fields along with packet minimum, transport/CPU
grain, scatter workers, and per-source AG progress. Its `--main-cpu` option
pins only the benchmark's submission thread after topology and initial thread
setup. The library does not impose caller affinity.

These controls require local calibration; their defaults remain unchanged.

Dispatcher profiles now bind the containing ELF's in-memory build ID and the
call's minimum input/output alignment (capped at 256 bytes), in addition to the
existing exact workload and topology keys. This distinguishes native and
fallback vector paths and invalidates evidence after compiler, architecture,
flag, or code changes. Profile version 2 records alignment; version 1 remains
readable as unknown alignment for offline comparisons. Regenerate local profiles
after rebuilding the library/C++ callers. If no software build ID is available,
automatic mode retains the existing callback without router initialization.

## Eight-L4 ReduceScatter scheduling

The November 13 experiment adds four explicit C++ options. They do not enable
the router without matching local calibration. The standalone benchmark exposes
the corresponding flags; the C wrapper and multiprocess NCCL drop-in retain
their existing configuration interfaces.

| C++ option | Default | Benchmark flag and scope |
| --- | ---: | --- |
| `smallReduceScatterWorkersPerNode` | 0 | `--small-rs-workers-per-node 4` caps active CPU lanes for complete RS inputs from 256 KiB through 1 MiB. The executor pool and larger-message limit are unchanged. |
| `batchCpuJobs` | false | `--batch-cpu-jobs` publishes the RS callback batch before notifying sleeping lanes, without creating unused completion tickets. |
| `directScatterResults` | false | `--direct-scatter-results` acquires result-reader credits before merging into the pinned result packet, removing the temporary-to-result copy. |
| `interleaveScatter` with CPU reduction | false | `--cpu-interleave-scatter` uploads one stripe per destination and keeps CPU partial reduction and merge. CPU stripes support standalone SM transport; GPU interleaving keeps its existing interface. |

CPU destination stripes retain the public tensor layout, the local retained
contribution, and the original BF16 partial/merge/retained rounding order.
Padding in a final stripe is neither reduced nor copied to the output. Both
layouts track the actual readers of each partial and result bank before reuse.
Resident GEMM and flat producer-event stripes are outside the CPU interleaving
experiment. The dispatcher includes every new option in its profile identity;
CPU-interleaved and GPU-interleaved measurements are separate.

The recommended candidate for fresh calibration is
`--small-rs-workers-per-node 4 --batch-cpu-jobs`. Destination interleaving and
direct result writes remain experiments: neither consistently improves this
candidate over 1 KiB--1 MiB, and direct result writes regress at larger sizes.
All configurations and negative results are retained in the
[measurement record](../benchmarks/data/rs_opt_8l4_20261113/provenance.json),
[screening summary](../benchmarks/data/rs_opt_8l4_20261113/performance_summary.csv)
and [independent confirmation](../benchmarks/data/rs_opt_8l4_20261113/confirmation_summary.csv).

The test host has eight NVIDIA L4s, ranks 0--3 on NUMA node 0 and 4--7 on node 1,
driver 580.178.04, CUDA 12.9, and NCCL 2.28.9. These are BF16 out-of-place,
single-process eager measurements, with complete input bytes per rank. Each
configuration uses ten warmups and five alternating paired repetitions of twenty
calls, checking every output element on all ranks. Timing is the slowest-rank
CUDA-event interval in the original harness, including host submission gaps.
Each fixed configuration has its own NCCL control; no per-size best variant is
selected. Confirmation reverses configuration order and reports the median of
two sweep medians; run ranges are retained and are not confidence intervals.

| Complete input/rank | Current default | Worker cap + batch | Latency reduction | Paired NCCL / candidate |
| ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 211.5 us | 182.0 us | 13.9% | 1.16x |
| 512 KiB | 240.9 us | 216.8 us | 10.0% | 1.25x |
| 1 MiB | 333.1 us | 316.2 us | 5.1% | 1.12x |

The 2/4-MiB router latencies remain approximately unchanged. Each configuration
has a separately paired NCCL control, so the ratio column is not computed from
one shared NCCL curve. Earlier attempts during another process's GPU workload
are retained under `contended/` and excluded from these summaries. Final runs
checked for other GPU compute processes before and after each sweep.

From the Lite root, after building with the matching CUDA, CUTLASS and NCCL:

```bash
NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 \
LD_LIBRARY_PATH="$PWD/build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64" \
./build/uccl_lite_collectives --gpus 8 --fallback nccl --op reducescatter \
  --begin 1024 --end 1048576 --warmup 10 --iterations 20 --repeats 5 \
  --small-rs-workers-per-node 4 --batch-cpu-jobs --main-cpu 23 \
  --calibrate build/new-rs-batch.profile --samples build/new-rs-batch.csv
```

Choose fresh output paths. Replace `--calibrate` with `--profile` to replay the
matching profile in automatic mode. Software rebuilds invalidate archived
profiles, and a profile calibrated against NCCL cannot select against the local
fallback. Unknown configurations retain the callback without router allocation.
These measurements do not establish SGLang or multiprocess drop-in speedups.

Validation includes mixed eager operations on 4/8/3 GPUs, a one-NUMA subset,
delayed ranks, masks, misalignment and tails, independent randomized numerical
and guard checks, graph queue wrap and the existing C-API graph path. The
[check manifest](../benchmarks/data/rs_opt_8l4_20261113/correctness.json) records
commands and outcomes. Native CPU queue tests cover credit exhaustion, sleeping
workers and concurrent producers. ThreadSanitizer could not start on this host
because of an unexpected memory mapping; its failure logs are retained.

### Follow-up: small messages and host submission

Two further C++ options are available for fresh local calibration:

| C++ option | Default | Benchmark flag |
| --- | ---: | --- |
| `smallReduceScatterGpuProgressMaxBytes` | 0 | `--small-rs-gpu-progress-max-bytes 65536` extends GPU reduction/progress to complete RS inputs through 64 KiB. Larger inputs keep the existing placement choices, including the default 128-KiB GPU case. The limit accepts 128 B--64 KiB, in multiples of 16 B. |
| `gpuLaunchWorkersPerNode` | 0 | `--gpu-launch-workers-per-node 4` reserves up to four spare CPU launch lanes per NUMA group and submits its GPU graphs in parallel. Each lane spins for up to 2 ms after work before sleeping. |

The small-message option avoids CPU routing callbacks and the NUMA partial/merge
handshake. It uses the existing GPU FP32 reduction and BF16 result conversion;
its rounding order differs from hierarchical CPU BF16 partials. It does not
enable CPU destination interleaving. `--no-small-rs-gpu-progress` disables both
the new limit and the existing exact-size GPU choices.

Parallel submission applies to eager standalone SM transport. Capture/replay,
DMA and resident/concurrent producers retain their existing launch path. Calls
return after every launch API has returned, preserving ordering for subsequent
caller-stream events and consumers. This option requires explicit streams,
including `cudaStreamLegacy`; `nullptr` and `cudaStreamPerThread` are rejected.
Launch CPUs exclude router workers and the replay CPU. The dispatcher preserves
its inspected CPU allowance even if the submitting thread is pinned later;
`gpuLaunchAllowedCpus` can restrict it further. Both options, and any explicit
CPU allowance, have separate calibration identities. Defaults remain unchanged.

The dispatcher also delegates eager capture validation to the pipeline instead
of repeating the same per-device CUDA checks. The pipeline checks the correct
current device for each stream before publishing any commands or CPU jobs.

The [follow-up record](../benchmarks/data/rs_submit_8l4_20261113/provenance.json)
uses the same eight L4s, BF16 out-of-place data, CUDA 12.9 and NCCL 2.28.9.
The submitting CPU is 47 in every configuration, including the frozen previous
binary. Each configuration has ten warmups and five alternating paired repeats
of twenty checked calls. Two sweeps reverse configuration order. The table
reports the median of those two sweep medians, with no best-variant selection
per size. These are fresh measurements, separate from the CPU-23 table above.
The original slowest-rank event scope, including host submission gaps, is
unchanged. Calls have no deliberate idle interval.

| Complete input/rank | Previous cap4 + batch | Small-message GPU progress | GPU progress + parallel submission | Paired NCCL / combined |
| ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 99.3 us | 94.0 us | 84.2 us | 1.41x |
| 32 KiB | 115.7 us | 97.3 us | 88.6 us | 1.44x |
| 64 KiB | 137.5 us | 114.2 us | 104.7 us | 1.30x |
| 128 KiB | 151.0 us | 145.9 us | 138.2 us | 1.16x |
| 256 KiB | 175.6 us | 175.1 us | 165.9 us | 1.24x |
| 512 KiB | 211.7 us | 211.2 us | 207.4 us | 1.27x |
| 1 MiB | 307.7 us | 307.7 us | 306.2 us | 1.13x |

The [full confirmation](../benchmarks/data/rs_submit_8l4_20261113/final_summary.csv)
also retains serial-current and parallel-only controls, run ranges, and the
2/4-MiB points; the [plot](../benchmarks/data/rs_submit_8l4_20261113/summary.svg)
shows the combined configuration and its paired NCCL control through 1 MiB.
Each fixed configuration has its own paired NCCL control. No
additional 1-MiB improvement is established. Instrumented traces reduce the
1-KiB GPU-start skew from 46.5 us to 24.5 us for the combined configuration;
these trace timings are diagnostic, separate from the uninstrumented table.

Prefer the small-message GPU option when testing a general workload. Parallel
submission is a CPU-budget and call-spacing tradeoff: eight L4s use eight
additional submission threads. A diagnostic probe inserts 5 ms of host idle
time before each fill/start interval; at 1 KiB, GPU progress alone takes
121.9 us and adding parallel submission takes 146.9 us. At 64 KiB the respective
times are 140.3 and 164.4 us. These are single probe sweeps in a separate binary,
not points in the confirmation table. Short-spin launch workers, one launch
lane per node, and an SMT-conflicting submitting CPU also lost. All negatives,
probe source, and raw samples are retained. Calibrate with representative
producer work and call spacing before selecting the threaded option.

For example, use `--main-cpu 47` and add
`--small-rs-gpu-progress-max-bytes 65536` to the checked cap4/batch calibration
command above. For dense calls with spare
CPUs, separately calibrate a new profile with
`--gpu-launch-workers-per-node 4`. Choose new output paths for each variant;
archived profiles require their exact recorded binary and configuration.

The [validation manifest](../benchmarks/data/rs_submit_8l4_20261113/correctness.json)
records 17 passing checks: policy/dispatcher/executor/BF16 checks, 4,392 queued
mixed eager operations on 1/3/4/8 GPUs, 3,000 graph calls including the C wrapper,
and independent CPU/GPU scalar and guard oracles. The new
`uccl_lite_dispatcher_cuda_check` checks explicit and implicit capture on each
rank, rejection before command publication, eager recovery, thread-relative
stream rejection, and preparation across the 64-KiB placement boundary.
Matching automatic replay selects every recorded winning key; an unmeasured
1,536-byte call and a changed launch configuration retain NCCL without router
initialization. No multiprocess drop-in or application speedup is established.

## AllReduce and AllGather scheduling

The ReduceScatter follow-up's eager submission lanes also support AllReduce
and AllGather. `batchCpuJobs` now publishes callbacks for every packet layout
before waking CPU workers. Its calibration identity is versioned separately
from the earlier RS-only implementation.

Two additional C++ options limit GPU progress by complete tensor bytes:

| C++ option | Default | Standalone benchmark flag |
| --- | ---: | --- |
| `smallAllReduceGpuProgressMaxBytes` | 0 | `--small-ar-gpu-progress-max-bytes N` |
| `smallAllGatherGpuProgressMaxBytes` | 0 | `--small-ag-gpu-progress-max-bytes N` |

Nonzero limits are multiples of 16 bytes, from 16 bytes through 1 MiB. AllReduce
uses complete input bytes; AllGather uses complete output bytes per rank. Below
the respective limit, GPU progress replaces CPU packet callbacks. AllGather
also uses the existing per-source readiness path, so one source can be copied
while another source is uploading. Above the limit, the configured CPU path
remains available. Explicit global GPU mode retains its existing behavior.

AllReduce GPU arithmetic reads every source on every destination GPU and rounds
once in FP32, whereas the CPU path rounds intermediate NUMA partials to BF16.
It can lose substantially beyond tiny messages. Do not copy ReduceScatter's
64-KiB threshold into AllReduce without measurement.

Both options are off by default and require fresh calibration for the exact
operation, size, placement, topology and software. `prepare` is required when
crossing a progress limit or changing between flat and per-source progress,
even with identical packet geometry. Parallel submission retains its extra CPU
reservation, 2-ms spin and sensitivity to call spacing; it is opt-in. The C
wrapper and multiprocess NCCL drop-in expose no new controls in this change.

The [eight-L4 transfer report](../benchmarks/data/ar_ag_transfer_8l4_20261113/README.md)
records fixed-configuration paired measurements, slower candidates and
correctness checks separately from the historical overview figures.

## Checks

```bash
./build/uccl_lite_router_policy_check
./build/uccl_lite_dispatcher_check
./build/uccl_lite_dispatcher_cuda_check
./build/uccl_lite_cpu_bf16_check
./build/uccl_lite_router_core_check
./build/uccl_lite_primitives_check
./build/uccl_lite_primitive_example
./build/uccl_lite_router_check --gpus 4
./build/uccl_lite_router_graph_check --gpus 4
./build/uccl_lite_local_collectives_check
```

See [validation](validation.md) for the four-L4 checks and the current eight-L4
rerun. The [paired local/router calibration](../benchmarks/data/local_fallback_8l4_20261110_v3/README.md)
uses this port, a UCCL Lite fallback, and fresh automatic-selection replays.
The [NCCL comparison](../benchmarks/data/cpu_router_8l4_20261110/README.md)
is retained as a separate control; its profiles cannot select a backend against
the new fallback. Both differ from the older no-P2P primitive archive.
