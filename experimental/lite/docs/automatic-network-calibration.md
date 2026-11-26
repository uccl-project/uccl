# Automatic network resource calibration during preparation

`PreparedNetworkRuntime` now fills an absent network resource profile during
eligible collective preparation. The unified `PreparedRuntime` and native
communicator C extension use the same path. Applications do not need to launch
an MPI calibration program before preparing the supported network candidate.
`NetworkPreparationOptions::probeResources` defaults to true; the unified
options and C `UCCL_LITE_PROBE_RESOURCES` flag control it as well.

The current accelerated algorithm still requires its actual two-host/four-GPU
FP32 contract, supported extent, two NUMA pairs per host, complete peer access,
mapped-host/cooperative support and a compatible compiled image. Calibration
does not relax these conditions. It runs only after discovery and capability
agreement, with idle callers outside capture. Native-only requests, disabled
network selection and unsupported layouts/dtypes do not start probes. If
probing is disabled and required physical evidence is missing, NCCL remains the
baseline without network buffers or workers.

## Resource measurements and algorithm admission

`src/runtime/network_probe.cc` uses the existing MSCCL++ bootstrap; the reusable
module has no MPI, NCCL or CPU-router dependency. It measures all discovered
active GPU/NIC candidate pairs in both directions for the algorithm's actual
cross-host GPU pairing. It then measures complete concurrent flow sets for
physical-prior rotations and uniform candidate-index choices, deduplicated and
rotated between repetitions. This is a bounded candidate set, not an exhaustive
search over every NIC assignment.

The probe uses private GPU buffers and raw `uint32` payloads. Independent host
checks compare every input and output word, including prefix/suffix guards, for
every warmup and timed call. GPU receive visibility is established before
checking. Registration failures remain explicit negative evidence; operational
or verification errors fail collective preparation. All temporary buffers,
registrations and connections are destroyed before whole-collective baseline
timing. Caller CPU affinity and NUMA memory policy are restored.

Isolated sizes are 16 bytes, 1 MiB and the context capacity clamped to 2–64 MiB;
concurrent groups use the two larger sizes. Each size retains three repetitions,
two warmups and five timed calls per repetition. Isolated timing covers host
RDMA posting through sender CQ flush. Concurrent timing additionally includes
the bootstrap start/end barriers; every sender and cohort duration is retained,
and the group statistic uses the slowest participant per call. Initialization,
receiver visibility, verification and registration are outside timing.

These are effective endpoint-path/cohort measurements. They include GPU memory,
PCIe/NUMA traversal, both NICs, network and host costs. They do not overwrite
advertised PCIe/NIC rates or invent a bandwidth for an unknown CPU link. A
provisional path with unknown physical bandwidth still requires matched
bidirectional observations and an exact complete concurrent flow set.

The resulting rail plan is only a resource prior. The exact user operation,
dtype, size and rank layouts still undergo independent guarded collective
qualification against the caller's NCCL baseline: three ABBA rounds, two warmups
and three timed calls/block, maximum-rank host enqueue-through-backend completion.
A candidate needs a median improvement greater than 3% and nonoverlapping
timing ranges. Measuring a path never by itself admits the collective algorithm.

## Evidence lifetime and compatibility

The immutable profile lives with the selector and is reused across requests.
Cached preparation revalidates live host/topology/transport identity; enqueue
does no discovery, probing or allocation. Negative observations are retained
even when no rail is admitted. By default no file is written. An optional
[plan-cache snapshot](network-plan-cache.md) persists this evidence together
with qualified decisions and reuses it across matching runtimes. C++ callers
can inspect `NetworkSelection::calibration` or the unified candidate's
`networkCalibration` and explicitly save it with `saveNetworkProfile`. C
diagnostics report its probe identity, scopes and path/cohort counts.

An explicitly supplied `UCCL_LITE_NETWORK_PROFILE` remains authoritative and is
not silently replaced. The v2 format now retains a per-profile cohort scope:
existing MPI records keep `concurrent-gpu-writes-max-cohort-with-mpi-barriers-v1`;
automatic records use
`concurrent-gpu-writes-max-cohort-with-mscclpp-bootstrap-barriers-v1`. The control
provenance records participating implementations and configuration. Old readers
that do not recognize the new scope reject it. Current readers preserve old v1
and v2 behavior, raw samples and explicit negatives. The CPU reporter prints
the stored scope, rather than labeling every profile as an MPI measurement.

Persistence across runtime lifetimes is opt-in. Automatic repair of an
explicitly supplied stale/incomplete profile, general multi-host accelerated
algorithms and full local/network/computation overlap models remain unfinished.

## Validation

`make automatic-network-check nccl-prepared-api-check planner-check` builds the
clients. The dedicated client checks the no-profile path with interleaved global
ranks and reversed GPU ownership on two four-L4 hosts, using single and dual
HCA views. It verifies all raw records, profile persistence, cache reuse, actual
collective buffers, affinity restoration and continued native communicator
ownership. Disabling probing on the single-HCA view retains native NCCL because
cross-NUMA bandwidth evidence is absent. Forced fallback starts no probes.

Raw profiles and runs are under `build/automatic-network-final/`. Single-HCA
calibration contains 8 directed paths and 1 complete cohort; dual-HCA calibration
contains 32 paths and 8 cohorts. Both compare FP32 AR at 4096 and 65536 elements.
The recorded runs retain NCCL at 4096 and select the hierarchy at 65536. This is
exact-request local qualification, not a general speedup or a best-variant curve.
Hardware is the documented dual-NUMA Xeon Silver 4410Y/L4 setup with 400-Gb/s
NIC ports; the native baseline is NCCL 2.28.7.

The C client checks all three collectives, layouts, rejection/recovery and
owned graph/eager ordering over two lifetimes on four and eight GPUs. The
supplied-profile ABI check in `build/automatic-network-profile-abi/` consumes
the new dual-HCA profile through 32 MiB, with mapping and rejection gates.
CPU tests cover both cohort scopes, unknown-scope rejection, old-format
compatibility, exact identity/flow matching and negative observations.

`build/automatic-network-identity/` additionally changes only rank zero's CPU
affinity after calibration: cached prepare rejects collectively, then recovers
with the same plan/profile after restoring affinity. Both single- and dual-HCA
runs pass. Final shared C-interface runs are in
`build/automatic-network-release/`; source and artifact hashes are recorded in
`build/automatic-network-artifacts.sha256`. These profiles are local calibration
records, separate from historical eight-GPU performance figures.
