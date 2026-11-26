# Network execution ranks, NUMA placement and NIC selection

The retained network executor still supports two hosts with four FP32 ranks on
each host. Its internal two-pair schedule is now mapped from actual per-host
NUMA placement. `pairedNumaRankOrder` is shared with the local packet planner.
Logical ranks remain the application's ranks; an execution rank is an internal
coordinate for the algorithm. CUDA device ordinals and physical NUMA node IDs
are passed separately to transport and shared host storage.

Preparation collectively verifies host grouping, ordered GPU UUIDs/device lists,
two NUMA pairs per host, peer/mapped-host capabilities and capacity. Each host
can have a different device permutation. A remapped communicator uses a fresh
collectively agreed bootstrap. Private buffers reorder AG output into logical
source order and RS input into execution shard order; caller aliases remain in
logical rank space. Completion events include permutation work and chain calls
across streams. Identity mappings retain direct buffers. AR retains FP32 sum
semantics; the algorithm does not promise identical rounding across reduction
trees. External capture remains unsupported.

Host membership is discovered from the gathered host identities, rather than
`global_rank / 4`. Application ranks may alternate between hosts and may own
CUDA devices in any unique assignment. Execution host order follows first
occurrence, keeping execution rank zero on the original root's host. Every
declared local GPU must have one owner. The [V3 profile](global-rank-placement.md)
binds these assignments through the NCCL frontend without relaxing V1/V2 checks.

Cross-host rail planning consumes the physical GPU pairs implied by the global
execution map. Each selected rail is interpreted in that process's HCA-name
mapping. Both GPU transport and the host-packet leader receive that rail. The
host stage no longer selects `IB0 + NUMA_group`, which previously bypassed the
planner and could address an unavailable second NIC.

## Incomplete physical bandwidth

Missing CPU/NUMA bandwidth stays unknown. By default, a path without complete
physical bounds remains unavailable. With an identity-matched resource profile,
the planner can include an active NIC as a provisional candidate marked
`requiresMeasurement`. Such paths require observed bidirectional GPU/NIC paths
and an exact complete concurrent-flow cohort. Isolated timings, another byte
extent, a subset of the flow set or stale software/hardware identity do not
suffice. A known physical prior remains available; provisional alternatives can
replace it only through the existing robust concurrent-measurement comparison.

This supports a measured single-NIC placement even when the cross-NUMA physical
link rate is unknown. It does not create that missing rate or claim end-to-end
collective performance. No profile and isolated-only evidence both reject when
no physical fallback exists. Corrupt or inconsistent profile configuration still
rejects collectively before transport allocation.

## PCIe synchronization correction

Whole-network memcheck exposed unsupported system-scoped remote atomic adds in
the retained local `MemoryChannel` signaling path on the tested L4 PCIe system.
Numerical checks alone had passed. Each direct channel now owns a device-local
outbound counter and publishes its sequence through a system release store;
MSCCL++ retains the acquire wait and local expected-inbound counter. Each channel
has one publishing lane per protocol phase, with existing CTA/grid barriers and
stream/graph dependencies ordering phases. Counter storage is prepared with the
transport and released after work/graphs retire. Upstream MSCCL++ is unchanged.
The standalone benchmark sharing this kernel header received the matching
counter ownership change.

## Validation and limits

Current hardware is two real four-L4, dual-NUMA Xeon Silver 4410Y hosts. The
final common-library logs are `build/network-mapping-signal-final-{default,
permuted,one-nic}/`. Each passes 126 complete-buffer FP32 ABI calls per rank,
through 32 MiB, over two communicator lifetimes, without native NCCL loaded.

- Default placement matches isolated and concurrent resource evidence with
  reversed HCA enumeration across hosts.
- Device orders `2,0,3,1` and `0,2,1,3` produce the asserted global execution
  order `1,3,0,2,4,6,5,7`. The existing calibration's ordered-view identity does
  not match this permutation, so it retains physical rails instead of reusing
  stale measurement evidence.
- Exposing only `mlx5_0` on each host matches the recorded `uniform-0-0` cohort;
  every GPU/host transport uses rail 0. This is a correctness test with reused
  resource evidence, not a new collective speedup measurement.
- Each run rejects per-rank profile absence, corruption and changed concurrent
  samples. `build/network-mapping-evidence-final/` separately verifies that
  absent or isolated-only evidence cannot enable the unknown physical path and that the
  complete profile permits automatic single-NIC execution.

`build/network-mapping-memcheck/run.log` preserves the failed remote-atomic
diagnostic. After correction, `build/network-mapping-memcheck-fixed/run.log`
passes two permuted four-GPU hosts with zero memory errors on all eight ranks.
This instrumented direct-registry check covers guarded AG/RS/AR at 4096 elements
and two lifetimes; it is not the 32-MiB ABI sweep. Known MSCCL++ capability-probe
API errors are suppressed, while device memory checking/error exit codes remain
enabled. Instrumented timing is never performance evidence.

The standalone permutation checker independently covers bitwise payloads,
guards, inverses, input preservation and invalid maps for 1/2/4/8/32 shards,
including scalar tails and 128-MiB complete extents. Its memcheck is in
`build/network-rank-permutation-memcheck.log`. CPU tests exercise unknown bounds,
matching/missing cohort evidence, stale identities and physically supported
priors. The local four-rank packet-mapping regression also passes.

The network validator now defaults to the generic runtime catalog/profile and
records CUDA-visible UUID order; `--legacy-profile` retains the historical
launcher. `--devices`, `--peer-devices` and `--expect-network-order` provide
explicit correctness scenarios. Arbitrary host/rank counts, a full joint
GPU/host/network stage model, automatic missing-profile calibration and native
whole-algorithm admission remain unfinished.
