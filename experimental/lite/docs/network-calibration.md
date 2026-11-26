# Effective GPU/NIC path measurements

Eligible prepared runtimes can now [measure an absent profile automatically](automatic-network-calibration.md)
through MSCCL++ bootstrap, without MPI. The standalone MPI probe below remains
available for explicit experiments with its original timing scope.

`make calibrate-network network-profile-report` builds a two-host GPU-memory
RDMA probe and a CPU-only profile reporter. The probe discovers host membership,
local CUDA devices and active HCA names; the number of ranks per host is an
experiment input, not a routing rule. It measures each matched GPU pair through
every discovered pair of unambiguous active HCA ports, in both directions.
The isolated phase activates one flow at a time. A second phase measures complete
concurrent flow cohorts; see [concurrent network measurements](network-concurrent-model.md)
for the separate timing scope, matching rules and current results.

These are **effective endpoint-path measurements**, not measurements of pure
NIC capacity. GPU memory, PCIe/NUMA paths, both HCAs, the network and host posting
all contribute. The profile does not overwrite advertised NIC or PCIe rates.
Exact concurrent cohorts are now measured, while local-traffic/computation
overlap and a complete distributed stage model remain necessary for general
network algorithm selection. The
registered network provider can now use them as a coupled rail-selection prior
after live identity agreement, as described below.

## Scope and identity

- Payload is raw `uint32` data. Byte accounting is one-way useful payload.
- Sizes are 16 bytes, 1 MiB and 64 MiB by default. Each size has three
  repetitions, two warmups per repetition and five timed calls per repetition.
  Every timed call is retained; no best repetition or per-size variant envelope
  is selected.
- Timing starts before MSCCL++ `Connection::write` and ends after the sender's
  completion-queue `flush`. This is CPU-posted write completion, not GPU-reader
  latency or the latency of a GPU-driven PortChannel collective.
- The receiver explicitly flushes GPUDirect RDMA writes to the owner scope when
  its CUDA capability requires it. This flush, input generation, MPI control
  and complete GPU input/output/guard verification are outside timing. Both
  source preservation and untouched receive-buffer regions are checked on
  every warmup and timed call.
- There is no collective or NCCL baseline in this resource probe.

`NetworkProfile` stores both host names and complete discovery identities,
source/destination GPU UUIDs, physical NIC names and ports, direction, transport
implementation identity, probe ELF build ID and raw sample series. Transport
identity includes the MSCCL++/verbs/provider build IDs, verbs mode, kernel and
GDRCopy implementation/driver identity, plus HCA GUID and firmware identity. The v2
transport identity additionally covers GID selection, active GID values/types,
port addressing and transport logging/GDRCopy configuration. The MSCCL++
`MSCCLPP_FORCE_DISABLE_GDR` flag controls GDRCopy; it does not mean GPU-memory
RDMA registration is disabled. Physical NIC names rather than MSCCL++
enumeration indices identify paths.

The versioned format fixes the timing scope. Format v2 adds concurrent cohorts;
v1 remains readable for historical reporting. New path measurements restore
caller CPU/NUMA policy after setup and have a distinct scope; old placement-scope
records are not promoted to current operational estimates. Failed registration paths remain
explicit rejected entries. Atomic replacement prevents publishing partial
profiles. The loader rejects malformed, truncated, duplicate or unobserved
records. `estimateNetworkPath` additionally requires exact current identities
for both hosts and the exact directed endpoint key. It uses medians at measured
sizes and interpolates only between increasing observed medians; it does not
extrapolate. A missing or stale path yields no estimate. A consumer must obtain
the remote host identity through live communicator agreement; copying that
identity from the file is not validation.

The API is in `include/uccl_lite/network_calibration.hpp`; persistence and
bounded estimates live in `src/planner/network_profile.cc`. It remains separate
from runtime backend admission; a resource file does not admit a collective.

## Runtime integration

Set `UCCL_LITE_NETWORK_PROFILE` on every participating rank to a copy of the
same profile. The strict launchers include its file hash in application-profile
identity. Network preparation agrees on profile presence/content, then exchanges
live host/transport identities, GPU UUIDs and rail candidates. Preparation also
agrees on topology-derived execution ranks and physical NUMA nodes. Identity
rank mappings reuse the bootstrap; other mappings create a collectively agreed
execution bootstrap while preserving application rank order. Resource profiles and
rail choices are resolved during prepare; enqueue does not reread profiles or
revise the rail plan.

Rank zero jointly selects both ends of each declared peer pair. The model uses
the largest common **observed** byte size across all eligible directional paths,
the worse of their two isolated medians, and accumulated transmit/receive loads
on shared directed PCIe/NIC capacity bounds. Constrained pairs are assigned
first, with stable physical-NIC ordering. The resulting transport indices are
interpreted in each rank's own HCA enumeration and agreed before initialization.

This is an advisory model for equal-sized bidirectional network flows. When an
exact concurrent baseline cohort is available, the planner can compare complete
measured alternatives, requiring a robust gain over that baseline. Logs separate
isolated, physical-bound and cohort costs. Neither form includes GPU-driven
signaling, local collective traffic or computation overlap. The retained network algorithm still declares its
actual two-host/four-rank-per-host capability. Cross-host peers are paired in
execution-rank space and translated back to physical GPU endpoints for this
model, rather than assuming equal logical device ordinals are partners.

Absent, stale or incomplete measurement coverage retains the physical-capacity
plan for the whole communicator when such a plan exists. A corrupt file, different record sets or
inconsistent profile presence rejects collectively before transport allocation;
there is no partial per-rank adoption. Explicit valid rail overrides remain
available for experiments, including paths whose capacity is not modeled, but
invalid/unavailable port indices reject collectively. Cross-NUMA measurements
do not fabricate an unknown shared CPU-interconnect capacity.

When a matching profile is supplied, the candidate catalog can retain an
active GPU/NIC endpoint whose physical path bandwidth is incomplete. Such a
candidate has `requiresMeasurement=true`, unknown path capacity, and only known
NIC capacity bounds. It cannot replace a supported physical prior using isolated
timings alone. Matching bidirectional path evidence **and an exact complete
concurrent-flow cohort** are required before executing it. If neither a physical
fallback nor this complete evidence exists, every rank rejects preparation.
This permits a measured single-NIC configuration without assigning a fabricated
rate to its unknown cross-NUMA link. It remains rail preparation within an
explicitly admitted algorithm, not whole-collective performance qualification.

The host-packet transport now uses its execution leader's selected physical
rail, including explicit overrides. It no longer substitutes a protocol group
number for a NIC slot. The GPU and host stages still need a full joint traffic/
overlap model; sharing a selected NIC is not a measurement of that whole pipeline.
See [network rank mapping](network-rank-mapping.md) for the execution contract
and current validation.

In the installed MSCCL++ implementation, CUDA `host-no-atomic` uses GDRCopy to
forward completion signals. Explicitly combining that mode with
`MSCCLPP_FORCE_DISABLE_GDR=1` is therefore rejected collectively before transport
allocation. This is a mode-specific requirement, not a general RDMA/GDRCopy
equivalence. Actual registration and transport setup still validate capabilities.

## Reproduce

```bash
make calibrate-network network-profile-report \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx

python3 -B benchmarks/probes/run_network.py \
  --hosts 10.10.55.2,10.10.55.1 --out build/network-calibration-new \
  --mpirun /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  --iface ibp55s0f0 --hcas mlx5_1,mlx5_0 --peer-hcas mlx5_0,mlx5_1 \
  --ranks-per-host 4 --wait-seconds 600

build/uccl_lite_network_profile_report \
  build/network-calibration-new/network.profile > build/network-summary.csv
```

The output directory must be new. The launcher copies the built probe to the
second host, waits for two observations of idle GPUs, and records process/device
observations during the run. It refuses to publish a usable result if a foreign
GPU process appears or monitoring fails. It never stops foreign processes.
This is an observed GPU-idle gate, not a cluster reservation or a guarantee of
exclusive CPU/fabric access. NVML can report an exiting probe rank without an
executable name; only previously observed probe PIDs receive that cleanup
exception. Unknown PIDs and known PIDs with a different executable still reject.

The reporter summarizes stored records only. It deliberately does not claim
that the current machine or remote endpoint matches them.

## Current evidence

`build/network-calibration-runtime-final/` contains the earlier isolated two-host run:
`manifest.json`, `resources.jsonl`, `probe.log`, `network.profile` and
`summary.csv`. The earlier `network-calibration-clean/` result retains its older
identity. These isolated-only artifacts predate the explicit restored-placement
scope; the current producer/consumer evidence is in `network-concurrent-calibration/`.
The intermediate
`network-calibration-identity-v2/` run predates inclusion of the GDRCopy
implementation/driver IDs and also remains a recorded earlier configuration. Hardware is
two hosts with four NVIDIA L4s each, dual Xeon Silver
4410Y NUMA domains, and two advertised 400-Gb/s ConnectX HCAs per host. The HCA
enumeration differs between hosts. All 32 directed endpoint paths passed full
verification, with 96 size observations and 1,440 retained timed samples.

In the earlier clean run, at 64 MiB the per-path median one-way payload rate
ranged from approximately 17.82 to 18.04 GB/s; the 16-byte median host-post/CQ-flush
duration ranged from 3.45 to 3.91 microseconds. These are measurements of that
recorded probe and
configuration, not pure NIC bandwidth or a collective speedup claim. All paths
are retained; no preferred path was selected for the report.

The earlier `network-calibration-slots/` run passed its data oracle but was
rejected by the original monitor's handling of an exiting known probe PID. Its
log and flagged profile are preserved. The monitor regression covers that
cleanup race and verifies that foreign work still rejects.

CPU profile tests cover stale local/remote identities, changed GPU/NIC pairs,
directionality, interpolation bounds, nonmonotonic data, missing repetitions,
negative evidence, atomic replacement and corruption. Concurrent network
traffic and GPU-driven protocol overhead remain unmeasured. CPU coupled-plan
tests cover shared-capacity decisions, both directions, physical NIC ordering,
HCA reordering and incomplete evidence. `build/measured-network-final-{matched,
stale,missing,none}/` passes 126 ABI calls per rank and two communicator
lifetimes on two real four-L4 hosts. The matched run also checks that all eight
ranks reject one rank's corrupt or missing profile configuration. Direct
registry checks cover valid and invalid explicit rail overrides. The
`host-no-atomic`/disabled-GDRCopy combination is independently checked as an
expected collective rejection in `build/measured-network-gdrcopy-rejected.log`;
the initial unsupported-mode experiment is retained in
`build/measured-network-no-gdrcopy.log`.
