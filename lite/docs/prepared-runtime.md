# Unified collective preparation

`PreparedRuntime` is the common caller-baseline interface. Applications provide
their logical rank, current CUDA device, communicator size, capacity and a ready
baseline. They do not select a local/network implementation or provide host
counts and GPU groups.

```cpp
RuntimeContext context; // id, rank, device, ranks, capacity
// Rank zero generates PreparedRuntime::createUniqueId(); distribute that ID.
PreparationOptions options;
options.dtype = ScalarType::Float32;
options.fallback = integration::makeNativeNcclFallback(
    context, nativeComm, nativeLibrary, configurationIdentity, options.dtype);
PreparedRuntime runtime(context, options);
auto plan = runtime.prepare(request); // collective, idle, native groups closed
runtime.enqueue(plan, input, output, stream);
runtime.synchronize();
```

The bootstrap ID is distinct from the caller's native NCCL ID. Construction is
CPU-only. The first prepare discovers actual host identities and owned GPU UUIDs,
then resolves participating GPUs through each process's CUDA-visible view. The
participant map is revalidated on later preparation and is immutable for the
runtime lifetime. One process per participating GPU and 1–32 ranks are supported
by this interface contract.

## Discovery and selection

| Discovered scope | Preparation path |
| --- | --- |
| One host with consistent, complete participating CUDA views | Local peer/host/packet candidate qualification |
| Multiple hosts, including partial local CUDA views | Distributed hierarchy/direct candidate qualification |
| One host with incomplete or differing CUDA views | Caller-only execution |

This chooses a selector, not an accelerated backend. Each selector still checks
dtype, exact extent/layout, actual topology/software capabilities and measured
evidence. Unsupported network layouts, FP16, and unavailable paths can retain the
caller. Candidate selection and timing rules are those documented in
[local preparation](prepared-local-runtime.md) and
[distributed preparation](prepared-network-runtime.md).

Local selectors receive logical-rank device order. Distributed selectors receive
a stable per-host participating inventory and discover logical ownership through
metadata. A multi-host process seeing only its own GPU can qualify direct IB exchange
or retain the caller, without pretending it has a complete CUDA peer view. Missing peers are never inferred
from GPU count alone.

`placement()` reports the discovered scope and each logical participant's host,
GPU UUID and CUDA device. `selection()` returns a common backend enum plus each
candidate's eligibility, correctness, paired raw samples and available execution
rank/NIC/resource-model diagnostics. All qualification remains outside enqueue.

## Graphs and ordering

```cpp
auto graph = runtime.prepareGraph(request, input, output);
runtime.launchGraph(graph, stream);
runtime.synchronize();
runtime.releaseGraph(graph); // collective and idle
```

Single-host graph requests use independent local graph qualification. Distributed
and caller-only graph requests currently use the caller's graph implementation;
no eager network result is promoted as graph evidence. A caller that cannot
prepare graphs rejects collectively. Graph buffers must remain valid until
release or runtime destruction. Handles are never recycled.

The facade chains graph/eager calls across streams and delegates transport
retirement to the selected runtime. A common control envelope detects mismatched
prepare modes, requests, policies, handle state and release calls before entering
different selector protocols. Failed prepare disables execution until successful
collective revalidation. External capture and open native groups remain outside
the contract. The caller retains ownership of its native communicator.

## Build and validation

`make prepared-runtime prepared-runtime-check` builds static/shared libraries and
the same correctness client for local, distributed and caller-only cases. The
core has no NCCL/MPI link dependency; native NCCL is supplied by the separate
adapter. Optional host candidates use the CPU executor without linking a native
NCCL implementation into the core.

Combining the local and network algorithms exposed a C++ layout collision:
3-slot and 8-slot shared-host storage used the same inline class symbols.
`SharedHostPacketsStorage<SlotCount>` now makes those configurations distinct
types. The original failing log is `build/unified-prepared-local.log`; the small
64-KiB-capacity case is retained rather than enlarged to hide the failure.
`build/unified-storage-symbols.txt` records the separate instantiated symbols.

Validation uses the documented four-L4 local host and two real four-L4 hosts,
including interleaved MPI ranks, reverse GPU ownership, one visible GPU per rank,
two three-GPU hosts and FP16/native-only operation. Independent complete-buffer
oracles cover signed reductions, bitwise AG edge values, aliases/alignment,
retained handles, cache reuse, graph/eager producer-consumer chains, invalid
bindings, mode/release disagreement, recovery and native use after teardown.
Forced transition tests use an explicitly labeled 3-ms baseline delay; these
timings do not claim a gain against native NCCL.

This is the explicit core preparation API. The separate
[prepared NCCL ABI](prepared-nccl-abi.md) provides ordinary-call dispatch and
native group/capture handoff. Distributed preparation now supports
[automatic resource calibration](automatic-network-calibration.md),
[direct operation calibration](direct-model-calibration.md) and
[persistent plan evidence](network-plan-cache.md). Accelerated support remains
bounded by each executor's capabilities; unsupported requests retain the caller.

Final evidence includes `build/unified-prepared-local-fixed.log`,
`unified-prepared-shared.log`, and `unified-prepared-single.log`. The common
client's distributed, hidden-view, six-rank, FP16, forced-transition and fallback
cases are under `build/unified-prepared-cases/`. Both
`build/unified-prepared-local-memcheck.log` and
`build/unified-prepared-memcheck/run.log` report zero memory errors on every
rank, using correctness-only identities and preserving device memory checks.

The storage-layout change also passes the two-host public ABI sweep through
32 MiB in `build/unified-storage-network-regression/`, including V3 rank maps,
resource-evidence matching and collective rejection checks. The retained local
packet path passes complete extents through 64 MiB in
`build/unified-storage-peer-regression.log`. These are correctness checks;
historical timing does not qualify the rebuilt algorithms.

The distributed selector now [compares hierarchy, direct exchange and the caller](network-candidate-selection.md).
Multi-host partial visibility uses the distributed selector; single-host partial
visibility remains caller-only. Additional preparation can demote earlier plans
when their simultaneous resource pool no longer qualifies. Existing handles
keep working through the updated choice, and reports are refreshed.
