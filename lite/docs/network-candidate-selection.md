# Distributed candidate selection

The common prepared runtime now compares two distributed implementations with
the caller's baseline: the retained hierarchy and general direct exchange.
`allowNetwork` enables this portfolio; it does not select an algorithm from the
number of GPUs. The native-enabled prepared ABI and its automatic initialization
adapter use the same flow.

Hierarchy retains its actual FP32/two-host/four-GPU-per-host/layout contract.
Direct exchange supports FP32/BF16, scalar tails and heterogeneous host sizes
within its 2–32-rank capacity bound. It requires actual GPU RDMA/owner-visibility
support and active NIC paths. Same-host reciprocal peer access permits IPC;
otherwise direct exchange can use IB. Partial CUDA visibility on a multi-host
communicator now reaches this distributed selector. UUID matching reconstructs
each process's local inventory; remote CUDA ordinals are never treated as local
device IDs. Single-host partial visibility retains its existing caller policy.

## Evidence and choice

Each eligible candidate gets its own three ABBA rounds on private guarded
buffers: two warmups and three timed calls/block, retaining 18 native and
18 candidate observations. The statistic is maximum-rank host completion.
Input production, rendezvous, setup and verification are outside timing;
the candidate's own control/transport costs are inside timing. Native blocks
have neither candidate's registrations, buffers nor workers.

Every output/input/guard is checked. A candidate must pass the existing
more-than-3% median gain and nonoverlapping-range rule against its own native
baseline. Among candidates that qualify, the lowest current median wins;
ties keep hierarchy deterministically. Advertised bandwidth and unknown paths
cannot supply performance admission. Direct's provisional NIC prior is followed
by exact end-to-end measurement, including all its actual flows.

Direct preparation records per-rank transport/topology/implementation identities.
Every reconstruction must match them. The common C report includes both
candidates' samples and direct configuration identities. `RuntimeBackend` adds
`DirectExchange` as value 6, preserving previous values.

## Multiple retained executors

Different prepared requests may choose different implementations. When that
requires hierarchy and direct exchange simultaneously, the selector measures
each winning request again with the actual resource pool present. Baseline
blocks still run after retiring the entire pool. These checks again use
three ABBA rounds and the unchanged admission threshold.

Only previously unchecked requests need revalidation while the same pool is
retained. A request that fails the pool check is demoted to the caller. If the
pool shrinks to one implementation, its original isolated evidence applies.
Preparation can therefore update earlier plan choices; handles remain valid
and reports are refreshed. No transport is initialized or algorithm retuned
from enqueue.

The standard NCCL ABI separately qualifies the exported wrapper, including
rank agreement and completion tracking. If multiple executor keys survive,
it also measures their combined pool and repeats when that pool shrinks.
ABI reports expose `pool_native_us`, `pool_dispatch_us` and `pool_correct`.
Core evidence alone never admits the standard ABI path.

## Cache scope

The [versioned cache](network-plan-cache.md) stores candidate-specific evidence,
not a cached portfolio winner. V1 hierarchy records remain readable; V2 adds
FP32/BF16 direct records with exact rank layouts and execution identities.
Both candidates have independent checkpoints. A positive hit requires a fresh
six-sample ABBA comparison; a negative hit validates the caller without creating
that candidate. Identity or performance drift refreshes full qualification.

Cached winners compete using fresh checkpoint samples, not historic timing.
A cached hierarchy caller decision cannot suppress direct evaluation, or vice
versa. Per-candidate cache flags and checkpoint samples are reported separately;
the common decision is marked fresh if any eligible candidate or combined pool
required fresh qualification. `allowHierarchy` and `allowDirect` can isolate a
provider for diagnosis; both default to enabled.

## Validation and limits

`make network-candidates-check` runs CPU choice-policy checks and builds the
MPI/CUDA portfolio checker. It checks full buffers, mixed aliases, tails,
caller-only plans, live plan reuse, hierarchy checkpoints alongside fresh direct
trials, retained pool evidence and two lifetimes. Separate standard-ABI tests
cover actual mixed executor calls and native capture after retirement.

The delayed-native library is only a branch-coverage fixture; measurements
against it are not performance claims. Direct exchange remains synchronous
and has no graph executor. Distributed owned graphs and external-capture
handoff continue to use native. The strict native-free adapter's profile catalog
is unchanged by this native-backed portfolio. Wider hardware validation,
fuller resource/control/SM cost models remain
unfinished in the migration record.

Final evidence is in `build/network-portfolio-release/`: 28 runtime cases and
17 CPU checker executables pass. Core and standard ABI each show actual
hierarchy/direct coexistence with 18/18 pooled samples. A changing scheduling
fixture verifies demotion of an already returned direct handle and refreshed
reports. Tests include 3+1 ownership, multi-host one-GPU visibility, FP32/BF16
scalar tails, automatic initialization, FP16 native-only execution, local
four-L4 regression, legacy hierarchy cache failure/drift cases, and native
capture after retirement. Two-host/four-rank BF16 memcheck reports zero device
errors (only known capability-probe API reports are suppressed).

Hardware is the recorded dual-NUMA Xeon Silver 4410Y/L4 configuration; the
native reference is NCCL 2.28.7. Real native retains these tested small requests.
The 20-ms delayed-native and scheduled-delay fixtures establish branch coverage
only. No throughput/latency gain or best-variant curve is claimed. Source and
binary fingerprints are in `build/network-portfolio-artifacts.sha256`, with
matching staged artifacts on both hosts. More than two physical hosts/eight
physical GPUs is not validated by these runs.
