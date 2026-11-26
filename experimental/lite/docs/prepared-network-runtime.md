# Caller-baseline selection across hosts

An optional [persistent plan snapshot](network-plan-cache.md) now reuses exact
resource/decision evidence across lifetimes, with fresh guarded revalidation.

`PreparedNetworkRuntime` compares eligible hierarchy and direct-exchange candidates with a
caller-owned baseline during explicit collective preparation. The core has no
NCCL or MPI link dependency; the existing native adapter supplies NCCL when
requested. `CollectiveRequest` names the shared operation/count/per-logical-rank
layout contract (the existing `LocalRequest` name remains an alias-compatible
spelling).

The [distributed portfolio](network-candidate-selection.md) includes general
FP32/BF16 direct exchange and the retained FP32 network hierarchy. The following
constraints describe the hierarchy specifically. Eligibility
requires its actual two-host/four-GPU layout, unique participant GPUs, two NUMA
pairs per host, complete peer/mapped-host support, cooperative-launch support,
a compatible compiled CUDA image, supported byte extent/alignment and a valid
network preflight. Its read-only phase resolves topology-derived execution ranks
and NIC candidates. When a resource profile is absent, eligible preparation can
[measure private GPU/NIC paths and concurrent cohorts](automatic-network-calibration.md)
before constructing the collective transport. An unsupported layout,
dtype, extent or preflight removes that hierarchy candidate. Direct exchange
is checked independently, and the caller remains when no candidate qualifies.
GPU count alone never enables an implementation.

```cpp
BackendContext context; // shared bootstrap ID, rank/device/ranks, capacity,
                        // stable participating local-device inventory
// Rank zero generates this CPU bootstrap ID and the application distributes it:
// context.id = PreparedNetworkRuntime::createUniqueId();
NetworkPreparationOptions options;
options.dtype = ScalarType::Float32;
options.probeResources = true; // default: measure an absent network profile
options.fallback = integration::makeNativeNcclFallback(
    context, nativeComm, nativeLibrary, configurationIdentity,
    ScalarType::Float32);
PreparedNetworkRuntime runtime(context, options);
auto plan = runtime.prepare(request); // collective, idle, groups closed
runtime.enqueue(plan, input, output, stream);
runtime.synchronize();
```

This interface supports 1–32 ranks under its one-process-per-rank contract.
The caller baseline must support the requested dtype and participants. Native-only
execution can cover layouts that the accelerator cannot represent, including
FP16. Heterogeneous host sizes and BF16 can now qualify direct exchange. Global `fallbackOnly` or `allowNetwork=false` avoids
network resource-profile reads and network preparation. Per-request
`fallbackOnly` uses the baseline while retaining resources needed by older plans.
Construction does not allocate GPU transport storage or start workers. CPU
bootstrap and all probing/qualification occur at prepare, never at enqueue.

## Agreement, qualification and lifetime

Preparation agrees on exact operation/count/layout, dtype, policy, capacity,
implementation and baseline identity. Live host/GPU/topology, per-rank transport
environment and applicable resource-profile contents are frozen for the runtime
lifetime. The supplied native communicator and environment are revalidated on
every prepare, including cache hits. Malformed or inconsistent configuration
rejects collectively. Candidate operational initialization failures are prepare
errors; they do not trigger independent per-rank fallback during execution.

Supported candidates run three ABBA rounds on private guarded buffers. Each block
has two warmups and three timed calls, producing 18 paired baseline and 18
candidate samples. Input values change every call. Shared qualification code
checks complete outputs, preserved inputs and all guards, including bitwise AG
edge values. The timed interval is maximum-rank host enqueue through backend
synchronization; network synchronization includes outbound-transport retirement.
Initialization, CPU control collectives, input generation and verification are
outside timing. This is not a GPU-event-only or throughput benchmark.

The network implementation must be correct, improve the baseline median by more
than 3%, and have a complete timing range below the baseline range. Otherwise
the caller remains selected. All samples and negative outcomes remain in
`NetworkSelection`. A failing baseline rejects preparation. This is whole-call
admission at an exact request; resource estimates alone cannot promote a backend.

Before a new comparison, the runtime destroys any retained UCCL network executor.
Candidate executors are also destroyed before the paired baseline blocks. After
qualification, the runtime recreates the executor if any old or new handle needs
it. Native-only initial preparation creates no network transport. Cached handles
retain their original selection; there is no per-enqueue routing decision.
Failed prepare disables all handles until a successful collective revalidation.

A common completion event orders different backends and streams. `synchronize`
also retires outstanding network sends. The native communicator remains owned
by the caller and can be used after runtime destruction. Calls must be outside
native group scopes and external capture; direct native calls outside this
runtime need caller-managed ordering. Owned network graphs, transparent ABI
fallback and persistent cross-process reuse are not implemented here.

## Build and verification

`make prepared-network-runtime prepared-network-check` builds static/shared core
libraries and a one-process-per-GPU correctness client. The client accepts
`NATIVE_LIBRARY automatic|fallback|unavailable|transitions|instrumented fp32|fp16`.
It uses interleaved MPI host ranks and reverse local GPU ownership, verifies an
independent CPU reference, mixed aliases/alignment, scalar tails, old handles,
cache reuse, configuration mismatch/recovery and native ownership over two
runtime lifetimes.

`transitions` adds an explicitly identified test-only 3-ms delay to the baseline
to force network/native choices and test both directions across streams. A queued
GPU delay keeps the dependency asynchronous despite that host delay. These
samples are not speedups against unmodified NCCL. `instrumented` uses a separate
correctness-only identity and does not require a timing winner.

Current validation hardware is two four-L4, dual-NUMA Xeon Silver 4410Y hosts
using native NCCL 2.28.7. Counts are 64/257/4096/65536, AR/AG/RS, two layout
patterns and two lifetimes. No best-variant curve is reported. The retained
candidate still has its {4,4}/FP32 contract; broader algorithms, missing-profile
automatic calibration, complete overlap costs and standard-ABI integration
remain separate work.

Final evidence is under `build/prepared-network-final/`. In `automatic/run.log`,
24 requests repeated over two lifetimes produce 48 decisions: four select the
network candidate and 44 retain NCCL. The four network decisions are AR at
65536 FP32 elements (256 KiB) in both tested layouts and both lifetimes. All
paired samples remain in the log; this does not assert a gain for other calls.

`fallback`, `six-ranks`, `half`, `unavailable` and `no-profile` cover explicit
native-only policy, two three-GPU hosts, FP16, incompatible MSCCL++ GDRCopy mode,
and a single-NIC configuration without the required cross-NUMA calibration.
Those cases initialize no UCCL network transport. `transitions` verifies both
network→native and native→network dependencies with the labeled delayed baseline.
Malformed per-rank network configuration rejects before baseline/transport
preparation; changing native configuration on a cached plan also rejects and
requires successful revalidation before any handle executes.

`build/prepared-network-memcheck/run.log` passes the distributed comparison and
independent application oracle with zero memory errors on all eight ranks.
Known MSCCL++ capability-probe API errors are suppressed; device memory checks
and error exit codes remain enabled. The instrumented identity is explicitly
correctness-only. `prepared-network-local-{scalar,graph}.log` validates the local
FP32/native and BF16 owned-graph paths after shared qualification extraction.
The core shared library's dependencies include CUDA/MSCCL++ and standard runtime
libraries, with no NCCL, MPI or CPU-router dependency.
