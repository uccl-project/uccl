# Topology-driven collective runtime migration

Status: the core organization and topology-driven selection migration is
implemented. See the [current capability and scope audit](topology-runtime-status.md)
for entry points and remaining support limits. This chronological record retains
earlier stage status and does not claim arbitrary-topology or NCCL feature parity.
Historical benchmark records and their machine-specific names remain unchanged.

## End-state requirements

- One core topology model, independent of CPU-router and the NCCL ABI adapter.
- Discover participating GPU identities, rank/device mappings, PCIe bridges and
  shared links, NUMA placement, network ports, transport capabilities, and
  topology/software identity. Unknown information must remain explicit.
- Model effective PCIe, GPU-memory, host-memory, NUMA-interconnect and NIC
  bandwidth, latency, shared-resource contention, and pipeline stages. Keep
  advertised capabilities distinct from actual measurements.
- Filter algorithms by collective, dtype, exact size, placement, alignment,
  capture mode and supported topology before comparing costs. GPU count alone
  must never admit a backend.
- Produce executable plans: rank groups, paths, algorithm, transport, protocol,
  channels, chunks and worker placement. Replace fixed rank arithmetic with
  topology-derived mappings. Preserve required numerical ordering semantics.
- Cache plans and prepare their resources outside capture and the timed path.
  Keep the caller's fallback free of CPU-router allocations and workers.
- Keep NCCL ABI adaptation in integration/nccl_lite, with core algorithms,
  transport, topology, planning and runtime organized by responsibility. Remove
  machine-specific build variants from the primary interface; retain honest
  support restrictions on specialized kernels.
- Validate CPU policy/dispatcher behavior, four-GPU eager and graph correctness,
  and two-host correctness for network changes. Match timing evidence to the
  new implementation; historical performance never qualifies a changed source.

## Current source organization

| Responsibility | Location | Selection input |
| --- | --- | --- |
| Hardware discovery and resource calibration | `src/topology/`, `include/uccl_lite/topology.hpp` | GPU/PCIe/NUMA/NIC identities, capabilities and measured rates |
| Algorithm feasibility and cost models | `src/planner/`, `include/uccl_lite/planner.hpp` | Exact workload, shared-resource demands and dependency graph |
| Preparation, qualification and plan lifetime | `src/runtime/`, `include/uccl_lite/runtime/` | Live topology, candidate correctness, paired timings and native baseline |
| Device execution and transports | `src/collectives/`, `src/multinode/`, optional `src/cpu_router/` | Prepared rank/path/chunk/worker plans |
| Application and NCCL adaptation | `integration/` | Communicator ownership and application buffer/workload contracts |

Machine counts belong in support constraints and measurement records, not in
public algorithm identities or a GPU-count dispatch switch. Existing specialized
executors retain their actual limits; the common frontend falls back when their
requirements or measured advantage are absent. Historical machine-specific
build targets and benchmark folders are retained for reproducibility.

## Implemented so far

- Extracted Topology and inspectTopology into include/uccl_lite/topology.hpp and
  src/topology. A standalone topology archive links without CPU-router,
  MSCCL++, or NCCL. The existing router archive includes these objects for its
  existing callers.
- Added passive PCIe hierarchy and active/inactive InfiniBand port discovery.
  Directed shared uplinks are represented once. Rank order follows the caller;
  unknown NUMA and memory bandwidth stay unknown. PCIe maximum capability and
  NIC advertised line rate are labeled advertised, never measured. Link and
  network state now contribute to the topology identity.
- Added synthetic sysfs tests for shuffled GPU order, non-contiguous NUMA IDs,
  shared uplink width, NIC rate/state, missing PCI paths and unknown rates.
- Removed the implicit Auto-to-Force router transition from JointCollectives.
  Missing peer support without admitted router evidence now reports that no
  backend is admitted. Explicit Router mode remains available for validation.
- Rebuild static archives from their complete current object lists. An old
  router archive contained obsolete topology.o/call.o/dispatch.o; incremental
  ar replacement retained these members and selected an obsolete Topology ABI.

- Extracted Collective, ScalarType and Workload into collective_types.hpp so
  planner and runtime interfaces do not depend on router policy.
- Added the CPU-only planner library with separate algorithm, transport and
  protocol descriptors. Candidate admission checks collective, dtype, byte
  extent, placement, alignment, capture support, peer/mapped-host capabilities,
  rank permutations and group partitions. Instantiated resource models are bound
  to the exact workload and topology identity, preventing stale byte volumes or
  rank mappings from being reused for another call.
- Added resource-stage cost estimation. Concurrent demands on shared directed
  resources accumulate; independent resources overlap; stages and repeated
  pipeline groups add. GPU/host memory and NIC rates participate in the same
  model as PCIe links. Unknown bandwidth/latency or stage overhead cannot yield
  a winning estimate. Advertised rates require explicit modeling opt-in.
- Added atomic local resource calibration updates tied to topology/software
  identity, with per-resource provenance and value-derived cache identity.
  Partial updates retain the provenance of unchanged resources. Versioned
  persistence and real GPU probes are now available as described below.
- Planner capability/resource tests pass (32 checks), including contention,
  full-duplex overlap, memory-driven choice reversal, inactive NICs, capture,
  dtype and placement rejection, and stale/partial calibration behavior.
  Peer executor integration described below now consumes generated models;
  strict and network executors still need migration.

- Added executable peer candidate generation for direct loads, a coordinator,
  and bulk copy/reduction/distribution. Models count GPU-memory accesses and
  physical PCIe path traffic, including shared links and serial bulk phases.
  Rank order is explicit, and supported BF16 extents are checked before tuning.
- PeerCollectives now selects and caches plans during prepare for the exact
  collective/dtype/size/placement/alignment on its immutable topology snapshot.
  enqueue only reads prepared plans; unprepared workloads reject. Scratch grows
  only during prepare. Fixed 64-KiB/256-KiB/4-MiB/64-MiB selection thresholds
  were removed. Automatic preparation now fills missing resource data and
  qualifies exact workloads as described below; when automatic calibration is
  explicitly disabled, incomplete data retains the general direct baseline.
  No historical crossover is imported.
- Added explicit diagnostic family selection for testing/calibration and a
  read-only selected-plan interface. Coordinator launches are grid-capped for
  arbitrary supported extents; existing grid-stride loops preserve coverage.
- Updated both application bridges to inspect real topology and prepare both
  AG and RS keys. Both libraries compile. Their local C ABI checks pass on two
  and four GPUs with counts 1, 257 and 4096, alternating AG/RS and rejecting
  unprepared calls. This does not validate their multi-host path.
- Direct, coordinator and bulk each pass 225 guarded producer/consumer and
  stream-transition cases on both two and four L4s. Model-driven execution also
  passes 225 cases on each configuration. The test model is synthetic and is
  used only to exercise decisions and dispatch, not to claim performance.
  Its four-rank AR chooses coordinator for small calls and bulk for large calls;
  its symmetric two-rank AR retains coordinator because staging does not reduce
  the PCIe bottleneck there. A test that incorrectly expected the four-rank
  crossover on two ranks was corrected to match the independently derived cost.

- Added separate end-to-end peer-copy and remote-read path resources. Passive
  discovery creates only paths allowed by the corresponding directional CUDA
  capability. Path rates do not overwrite individual PCIe bridge capacities.
- Added a real GPU-resource probe API and make calibrate-hardware. It measures
  per-GPU vector-copy memory bandwidth and directed peer-copy/remote-read paths,
  verifies every output element, and restores the caller's device and the peer
  access state it owns. Private buffers are prepared outside timing/capture.
- Added hardware-profile-v1 persistence with atomic replacement, original
  repetition samples, workload/scope metadata, and the probe artifact's ELF
  build identity. Malformed, truncated, unobserved or stale records reject.
  Hardware discovery identity is now separate from backend-specific policy
  identity, so a matched hardware probe can be used by different executors
  without weakening router policy calibration checks.
- Peer executors can load a matching profile through PeerTuning.hardwareProfile
  or UCCL_LITE_HARDWARE_PROFILE, before creating peer resources. Mismatched
  hardware rejects. Automatic first-use probing and workload qualification are
  now wired into preparation; shared CPU/NUMA and network resources still need
  further discovery and calibration.
- A real four-L4 sweep produced 4 GPU-memory and 24 directed protocol-path fits,
  retaining 56 raw sample series. It uses uint4, 16-byte and 64-MiB sizes, two
  warmups, five timed iterations, and three repetitions with GPU-event timing.
  Memory rates count read+write traffic, path rates useful one-way traffic.
  Rates/intercepts are two-size affine fits of these probes, not intrinsic
  latency or measured bridge capacities. No baseline comparison or per-size
  winner envelope was used. Data stays under build/ as local calibration.

- Extended directional GPU probes to remote writes, independently of remote
  reads and CE copies. Coordinator output stores no longer use a read-path
  throughput proxy. The current four-GPU probe covers 36 directed protocol paths
  plus four GPU-memory resources (the earlier 24-path record above describes
  the previous probe configuration).
- Peer models now use measured end-to-end protocol paths, with known shared
  PCIe links as advertised bandwidth bounds. Such bounds do not invent link
  latency and cannot form a performance prediction without measured operational
  demands. Missing CPU-link capacity is not fabricated.
- Default peer prepare now automatically probes incomplete hardware data once
  per executor, orders eligible candidates by a data-movement prior, and times
  all supported families on private guarded BF16 buffers at the exact workload
  size/alignment. Trials rotate order, use two warmups and three timed calls in
  each of three repetitions, with host enqueue-to-completion timing. Every
  repetition checks all outputs, preserved inputs and prefix/suffix guards.
- Automatic selection retains the direct baseline unless a candidate improves
  its median by more than 3% and has a complete timing range below the baseline
  range; the fastest qualifying candidate is selected. Timing samples and the
  hardware calibration identity remain in the cached plan. An exact-workload
  nonnegative residual includes unmodeled costs, not just launch overhead; it
  is never extrapolated to other sizes or topologies. Qualification currently
  lives for the executor lifetime rather than being reused across processes.
- Enqueue performs no probing, allocation or candidate timing. Prepare rejects
  capture before probing, including cached workloads; application preparation
  validates capture before synchronizing devices. Explicit diagnostic family
  selection and synthetic-model tests bypass automatic calibration.
- Two- and four-GPU default automatic runs each pass all 225 guarded transition
  cases and retain all candidate timing repetitions. Selection differs between
  topologies/workloads without hard-coded crossover sizes. A deliberately empty
  candidate is rejected despite fast timing; missing-baseline trials reject
  before invocation, and failed trials restore the caller's device.

- Unified the strict NCCL frontend behind a core CollectiveBackend interface
  and registry. One new nccl-runtime library links peer-packet, host-packet and
  network-hierarchy implementations; legacy targets use the same frontend
  object with restricted registries. The frontend no longer compiles in a
  selected dtype, host count or ranks-per-host constant.
- Renamed the local FP32 implementation to PeerPacketCollectives in
  src/collectives/peer_packet.cu, eliminating its NetworkCollectives symbol
  collision with the network implementation. The old source path forwards for
  compatibility. Moved the BF16 packet adapter out of the NCCL integration
  header into src/runtime/host_packet_backend.cc.
- Moved CPU control/bootstrap into an independent core header and expanded its
  bounded participant range. CPU tests cover 1/3/4/5/8/9 ranks and failure cases.
  NCCL session IDs are now independent of provider-private bootstrap IDs;
  private IDs are generated and broadcast after capability consensus.
- Transport preparation now completes in communicator initialization. The
  frontend checks each rank's actual peer/mapped-host capabilities as well as
  profile/software identity before constructing the selected provider. Profile
  state belongs to the communicator rather than a process-global cached dtype.
- Added CPU-only provider introspection and a generic runtime_profile.py tool.
  It uses the compiled catalog and actual host records, without L4 model-name
  checks or a local4/local8 mode argument. Unsupported/ambiguous dtype/layout
  matches reject. This is still an explicit candidate-profile workflow, not
  completed automatic multiprocess planner admission.
- Unified and restricted libraries compile without a native NCCL dependency.
  Catalog checks validate all three providers and backend-independent IDs.
  The four-L4 strict ABI oracle passes through the new catalog-driven workflow:
  198 calls/rank, two lifetimes, FP32 guards/aliases/unaligned pointers through
  32 MiB. Network and eight-GPU providers still need topology-specific execution
  after these lifecycle changes; compilation is not runtime validation.

- Added a CPU-only network rail planner using actual GPU--NIC PCI paths, NIC
  bandwidth values/bounds, and projected load on shared resources. It assigns
  constrained GPUs first and resolves HCA names through MSCCL++'s actual
  transport-slot mapping, including MSCCLPP_HCA_DEVICES ordering. No GPU ordinal
  division determines a rail. The adapter currently excludes ambiguous
  multi-active-port HCAs and unknown GPU--NIC bandwidth paths.
- Passed the participating local device set from the NCCL communicator to the
  network provider. Removed the fixed rail equality from network initialization;
  initialization instead validates that each rail names a real IB device.
  MSCCL++ negotiates each endpoint's local transport index independently.
  Canonical local rank/GPU layout and the two-host/four-rank algorithm remain
  separate restrictions; this change does not remove those yet.
- Added a read-only runtime topology fingerprint API and included it in both
  current profile launchers. PCIe paths, NIC activity and bandwidth bounds now
  participate in profile identity rather than being absent from that gate.
- Verified the second host is available: mibura-sky-test-02 / 10.10.55.2 and
  mibura-sky-test-01 / 10.10.55.1, each with four L4s. Strict FP32 ABI checks
  pass with reversed HCA enumeration on both hosts and with reversed ordering
  only on host 0. Each run checks 126 calls/rank, two communicator lifetimes,
  guarded buffers, legal aliases, input preservation and capture rejection.
  Logged assignments confirm GPU-local NIC selection even when corresponding
  ranks have different transport indices on the two hosts.
- A two-host nccl-tests sweep passed AR/AG/RS correctness across 1 KiB--1 GiB,
  both layouts, with six candidate jobs. The short screening comparison uses
  native/candidate/candidate/native ordering, one warmup and two timed calls,
  FP32, eight MPI ranks, maximum-rank host enqueue/completion batch means, and
  one compiled algorithm policy (no per-size best-of measurement envelope).
  It is not performance admission: 3 of 126 paired rows were slower than
  native NCCL, with worst ratio 1.21634 at 8-MiB out-of-place AllReduce. Raw
  ratios, commands, identities and negative results remain in
  build/network-auto-rail-sweep-1g. That sweep preceded the identity-only API
  addition; the latest identity-bound asymmetric ABI rerun also passes.

- Extended shared discovery to online NUMA domains, including domains without
  selected GPUs, and their relative OS distance matrix. Non-contiguous node
  IDs are handled explicitly. Interconnect links are recorded with unknown
  bandwidth/latency; relative distances are never converted to either.
- Added CPU-only host-copy resource registration and probing. Paths are scoped
  by source NUMA, destination NUMA and exact destination-worker CPU IDs. Private
  mappings are bound and verified; complete output and input checks occur
  outside timing. Cancellation/join logic handles worker-side placement errors
  without changing caller affinity.
- Host profiles reuse versioned hardware-profile persistence and carry raw
  repetitions, probe/memcpy build identities and libc configuration. Local and
  remote resources uniformly count one-way payload bytes; generic node memory
  and physical interconnect rates remain unknown. Changing worker allocation
  produces a distinct resource identity. Register matching paths before loading
  a saved profile. No CPU-router is admitted on copy measurements alone.
- Host path/placement/persistence/failure tests pass, as do updated topology
  tests for relative distances and non-contiguous NUMA IDs. GPU calibration and
  the four-GPU peer executor regression pass after the topology schema update.

- Replaced the single-process peer executor's two/four-rank whitelist with
  explicit kernel argument capacity (currently 1..4). All-pairs capability,
  unique device IDs, dtype, alignment, placement and exact workload checks
  remain required. Joint dispatch now defers rank support to backend capability
  checks instead of its own 2/4/8 whitelist.
- Added one-rank identity execution using bit-preserving device copies and a
  completion chain across caller streams. Coordinator/bulk plans are not
  admitted for one rank. Three-rank bulk AR still requires exactly divisible
  shards; direct/coordinator paths handle odd extents.
- Updated validation buffers and helpers to map logical ranks through actual
  device order. Numerical reference calculations remain unchanged. Tests cover
  three GPUs in order 3,0,2, one GPU at device 2, and four GPUs in order 2,0,3,1.
- Direct, coordinator and bulk each pass 225 three-rank guarded stream/family
  transition cases; synthetic-model and automatic selection also pass. The
  one-rank and permuted four-rank automatic runs pass their 225 cases. Each
  final run also executes the independent broad finite BF16/double oracle and
  AG special-bit-pattern checks. One- and three-rank joint selftests pass
  router/peer mechanics, odd tails, async input reuse, guards and capture
  rejection/recovery. This does not generalize the retained multiprocess NCCL
  providers' separate rank/layout contracts.

- Moved common device-port/call declarations out of router implementation
  headers, and CollectiveCall::workload into src/runtime/collective_call.cc.
  The existing PacketPort ABI spelling is retained by the common definition;
  peer and qualification code no longer include router dispatch headers.
- Added standalone peer-runtime static/shared packages, with CUDA compilation
  flags tracked for their objects. The shared link uses -z defs and actual
  dependency/symbol checks confirm no CPU-router implementation, NCCL or
  MSCCL++ dependency. The two-GPU application bridge now links this package.
- Standalone static and dynamic consumers both pass four-GPU AR/AG/RS with
  automatic qualification, shuffled devices and scalar tails. Router policy,
  host dispatcher, CUDA dispatcher and 1000 captured mixed-collective replay
  regressions pass after the shared call-data extraction.
- Inspected the multiprocess host transport: its generic path already accepts
  2..32 ranks, while optional optimized packet modes explicitly require eight.
  The current HostPacketCollectives adapter and profile descriptor still bind
  it to eight; migrating that adapter/profile contract is the next rank task,
  and has not been claimed complete here.

- Parameterized HostPacketCollectives by the actual communicator rank count,
  replacing its fixed 8-rank creation and 16-byte count factors. The underlying
  generic host transport supports a 2..32 rank capacity; optimized eight-rank
  packet modes remain separate conditional paths.
- Added explicit layout-range capabilities to backend registration and a
  UCCL_LITE_RUNTIME_PROFILE_V1 format carrying provider plus actual host layout.
  The frontend instantiates a communicator-local layout for extent checks,
  identity consensus and backend creation. Legacy fixed-layout headers still
  work. The generic launcher uses registered dtype/layout capabilities and
  rejects multi-host layouts for the single-host host transport.
- Verified the unified NCCL entry with 2, 3 and 4 MPI ranks in BF16, and three
  ranks under CUDA_VISIBLE_DEVICES=2,0,3. Each run passes 96 calls per rank,
  two lifetimes, aligned/unaligned buffers, all legal aliases, full input/output
  and guard checks, and capture rejection followed by eager recovery. The
  legacy four-rank FP32 profile also passes its 198-call/two-lifetime ABI check.
  These are explicit experimental candidate checks, not automatic performance
  admission or validation of untested larger groups.

- Added an optional RouterPlanDispatcher with NUMA-local worker candidate
  generation. Unknown placement retains fallback. Candidate metadata is
  CPU-only; only an exactly admitted winner is instantiated during prepare.
  Enqueue uses cached choices and rejects unprepared admitted configurations.
- Worker plans use exact existing dispatcher identities and paired collective
  evidence. The smallest measured admitted median wins; scoped host-copy data
  supplies only an advisory local-partial memory prior. It never predicts
  AllGather flag publication or overrides validated collective measurements.
- GPU checks exercise switching between admitted worker configurations,
  capture rejection before state changes, fallback teardown and re-entry.
  CPU checks verify missing/stale evidence, unknown placement and unmeasured
  sizes never initialize a router. Existing eager/capture and 1000-call graph
  replay regressions pass.
- Added a real worker-plan calibration executable with baseline blocks free of
  router workers. On the local four-L4 1-MiB BF16 AR workload, 1/2/4-worker
  configurations did not qualify against the already prepared peer backend;
  the selector correctly retained fallback. Samples and all negative outcomes
  are persisted, with exact workload/configuration/build identities.

- Added persistent exact-workload peer plan records, matched by topology,
  executor artifact, dtype, collective, extent, layout and alignment. The
  loader reconstructs only currently supported plan families and re-evaluates
  the same qualification rule from retained raw repetitions; it does not trust
  an externally supplied winner or serialized GPU objects.
- Cache updates use a file lock and atomic replacement. CPU tests cover stale
  code/topology/alignment, missing family evidence, corruption, regressions
  restoring the direct baseline and concurrent writers preserving entries.
  Real four-GPU standalone execution passes cold calibration and a second
  process requiring cache hits for all three collectives.
- Reviewed the native-fallback interposer before broader integration. Its
  per-rank opportunistic group/capture fallback cannot simply be transplanted
  into a multi-provider automatic runtime: dispatch changes require a
  collective-consistent preparation/disable contract. The prepared BF16
  integration below now supplies collective selection; safe native fallback
  remains outstanding. No unsafe local-only switching was enabled.

- Added `peer-staged`, a BF16 multiprocess peer provider using actual rank
  count and MSCCL++ CUDA IPC registration. It supports the same AR/AG/RS ABI
  contract as host-packet, with a declared 2–32 rank capacity and complete-peer
  capability requirement. It preserves logical rank order under CUDA-visible
  device permutations and stages inputs before output writes, including aliases.
  Separate readiness/reader-retirement kernels and cross-stream event chaining
  protect staging reuse. The registry now contains comparable peer/host BF16
  implementations; explicit experiment creation rejects ambiguous matches unless
  `--provider` identifies a supported candidate. This is not automatic selection.
- The first peer-staged implementation passed numerical tests but failed
  Compute Sanitizer because this MSCCL++ version's signal uses remote system
  atomics unsupported on the local PCIe links. Replaced that publication with a
  monotonic release-store sequence on each pair's single-writer semaphore slot,
  retaining MSCCL++ registration/storage and acquire-wait. Final 2/3/4-rank
  tests pass 168 calls/rank across two lifetimes; shuffled three/four GPU order,
  aliases, full guards/input preservation, queued producers, alternating streams
  and every BF16 AllGather bit pattern are checked. Corrected four-rank memcheck
  reports zero memory errors on all ranks. Graph execution remains unsupported;
  rejection/recovery is tested. No performance qualification is claimed.

- Added `PreparedLocalRuntime` and the `prepared-local` NCCL provider. BF16
  single-host profiles now default to a collective selector rather than a
  manually named peer/host implementation. V2 profiles declare every rank's
  layout/alignment for each exact operation/count. Preparation agrees on
  descriptors, ordered GPU UUIDs, software, per-rank topology/transport
  environment and idle state. Enqueue executes a common plan handle, avoiding
  unsafe local-only switching when ranks have different pointers or aliases.
- The first prepare probes GPU resources once (or loads a supplied profile).
  The shared planner models staged peer memory traffic and shared PCIe demand.
  The estimate is explicitly advisory and omits synchronization, launch and
  arithmetic/access-granularity costs. Synthetic six-rank tests verify memory
  bandwidth constraints, shared-link contention and unknown-resource rejection.
- Exact per-rank layouts are calibrated using private guarded buffers and three
  ABBA rounds, two warmups and three checked timed calls per block. Timing is
  max-rank host completion; all 18 samples per provider are retained. The host
  provider is absent during peer baseline blocks. Only a correct host candidate
  with >3% median gain and nonoverlapping ranges is admitted. Root broadcasts
  the decision; unused host resources are destroyed. A live cache reuses the
  same request after metadata agreement; earlier plans survive later prepares.
- Real four-L4 runs exercise both choices: smaller tested calls keep staged
  peer, while several larger tested calls select generic host. This baseline is
  not NCCL or the tuned single-process peer executor. The independent public
  NCCL ABI check exercises 120 calls/rank, two communicator lifetimes, mixed
  per-rank layout/alignment, queued producers and alternating transports. Core
  tests additionally cover scalar tails, conflicting rank metadata and live
  cache reuse. Capture remains an explicit rejection. See
  [prepared local selection](prepared-local-runtime.md) for scope and commands.

- Added a standalone MSCCL++ GPU-RDMA path probe and versioned network profile.
  Paths are keyed by both hosts, ordered GPU UUID endpoints, physical NIC/port
  names, direction and transport identity. Raw per-call observations and
  rejected registrations persist; endpoint-path rates never replace physical
  NIC/PCIe rates. Exact-identity estimates interpolate only within observed
  intervals and retain unknowns outside them.
- A two-real-host/four-L4-per-host run passed all 32 directed GPU/NIC path
  combinations, including differing HCA enumeration. Every warmup and timed
  call checks complete source/output/guards after explicit receiver RDMA
  visibility. GPU process monitoring passed after fixing an NVML cleanup-name
  race; no foreign process was stopped. The recorded scope is isolated
  CPU-posted write plus sender CQ flush, not GPU-reader completion or full
  collective latency. See [network calibration](network-calibration.md).

- Connected network path profiles to collective network preparation. Live
  per-rank host/transport identity, GPU UUID and directed physical constraints
  are exchanged through the provider bootstrap, which is then reused by the
  executor. The root jointly selects both ends of each declared peer pair from
  complete directional evidence and shared PCIe/NIC bounds; enqueue does not
  perform discovery or make local routing decisions.
- Selection uses the largest common observed byte point as an isolated-path
  prior, explicitly separate from concurrent performance qualification. Missing
  or stale coverage retains the physical-capacity plan for all ranks. Corrupt
  or inconsistent profile configuration rejects collectively. Valid explicit
  rails remain available without requiring a complete capacity model; invalid
  rail indices are rejected before transport initialization.
- Strengthened transport identity with GID index/table/addressing, GDRCopy
  configuration, library/driver IDs and logging configuration. The installed
  CUDA host-no-atomic stack requires GDRCopy for completion forwarding; explicitly
  disabling it in that mode now rejects collectively before allocating transport
  buffers. This is not a generic assertion that RDMA requires GDRCopy.

- Added concurrent network cohorts to profile format v2. Each record names the
  complete directed flow set and exact per-flow bytes, retaining every sender
  duration and every rank's timed cohort window. MPI start/end barriers bound
  the complete transfer epoch so staggered posting cannot masquerade as higher
  simultaneous throughput. Control-library/configuration identity is retained.
- The probe restores caller CPU affinity and NUMA policy after MSCCL++ setup;
  live identity now includes the caller's memory policy. New isolated data has
  an explicit restored-placement scope. Legacy files remain readable/reportable
  but old placement-scope records do not become current operational estimates.
- The network planner consumes concurrent data only for exact full flow sets
  and observed extents. It requires a measured baseline before comparing
  alternatives, applies the >3%/nonoverlapping-range rule, and recomputes prior
  diagnostics for the chosen group. Missing groups retain the advisory prior.
  The root decision and profile digest now include complete concurrent evidence.
- A two-host/four-L4-per-host run measured eight cohort placements, each with
  eight directed flows at 1 MiB and 64 MiB per flow, with complete checks on all
  warmups and timed calls. At 64 MiB, topology placement measured 6.834 ms per
  maximum cohort versus approximately 13.1 ms for uniform NIC choices. These
  timings include MPI barriers and are not GPU-driven collective latency.
  See [concurrent network measurements](network-concurrent-model.md).

- Added a caller-owned fallback contract to the prepared local runtime and a
  separate native NCCL adapter under integration/nccl_lite. The adapter borrows
  a ready blocking communicator, resolves the explicit native implementation,
  validates rank/count/device and configuration, and never destroys the caller's
  communicator. Core/router modules remain independent of NCCL.
- Peer and host alternatives now have separate paired fallback baselines,
  private complete-buffer qualification and the existing robust-gain rule.
  Baseline blocks have no UCCL peer/host instance; unselected resources are
  released. Missing peer capability no longer rejects the whole prepared
  runtime when a caller fallback is supplied. Collective fallback-only handles
  create no peer/host instances and skip probing/qualification.
- Added common ordering across prepared native/peer/host calls, live-cache
  fallback revalidation, and invalidation after failed preparation. A failed
  prepare must be followed by successful collective revalidation before old
  handles execute. Per-rank implementation or fallback-policy disagreement
  rejects before backend initialization.
- Real four-L4 comparisons retained native NCCL 2.28.7 for all nine tested
  AR/AG/RS/count combinations; the staged peer and generic host candidates did
  not qualify. Negative samples are preserved. A separately labeled delayed
  baseline exercises peer-to-native cross-stream dependencies without claiming
  a gain against unmodified NCCL. See [native fallback](native-fallback.md).

- Extended the generic multiprocess peer executor to FP32 while retaining BF16.
  Byte widths, scalar alignment, alias offsets and preparation identity are
  dtype-aware. FP32 AG preserves payload bits, including NaNs/infinities/signed
  zero; reductions accumulate in FP32 and are checked against an independent
  CPU reference. FP16 currently uses a matching caller-native fallback only.
- Single-host FP32 profiles now select `prepared-local-fp32` for supported
  layouts, including two and three ranks. The retained peer-packet algorithm is
  a separately gated candidate: exact dtype/extent/alignment, complete peer
  access, mapped host memory and the actual two-NUMA-pair placement are required
  before qualification. Interleaved GPU order and scalar tails use generic
  peer/native paths instead of forcing the specialization.
- Paired caller-baseline selection now includes the retained FP32 candidate.
  The recorded four-L4 native comparison selected it for four of 18 exact
  requests and retained native NCCL for 14, preserving all samples. This is not
  a blanket performance claim. See [dtype-aware runtime](scalar-runtime.md).

## Remaining work

The staged peer executor now has an [owned graph execution API](peer-graphs.md)
with collective program agreement, fixed bindings and graph/eager stream
ordering. The [prepared graph selector](prepared-graphs.md) now adds independent
peer/native graph qualification and retained binding reconstruction. General
external capture and transparent standard-ABI integration remain in item 2.

1. Extend the worker-plan model beyond its advisory local memory prior with
   reduction-specific, shared-resource and transport measurements, and complete
   full multi-host stage models. Exact isolated paths and concurrent flow
   cohorts now participate in live-identity-checked rail preparation; arbitrary
   flow combinations, local collective traffic, GPU-driven protocol costs and
   compute overlap still require modeling and validation.
   GPU memory, GPU protocol paths and host-copy probes now exist; physical PCIe
   and NUMA-link capacities remain bounds/unknown rather than invented rates.
   GDR support is still validated by transport registration, not inferred from
   GPU count or topology distance.
2. Extend the executable capability registry beyond PeerCollectives to packet,
   CPU-router, and network backends. Generalize the remaining rank restrictions
   and extend the prepared BF16 single-host selector to the remaining dtypes,
   algorithms and network layouts beyond the current BF16/FP32 paths. The unified NCCL-facing runtime still needs
   general automatic admission and transparent native fallback/group/capture
   semantics. The new native-backed local BF16 prepared interface establishes
   the callback and ordering contract, but is not a standard-ABI interposer;
   explicit preparation/requested-workload profiles remain necessary today.
   Extend persistent qualified-plan reuse beyond the single-process peer
   executor to the multiprocess registry, and improve model fidelity for reduction/overlap costs beyond the existing
   exact-workload empirical residual.
3. Generalize rank mappings, separate algorithm/transport/protocol parameters,
   and migrate fixed machine configurations behind the capability registry.
4. Introduce plan caching and the shared preparation/runtime interface; move
   transport execution out of the NCCL integration headers.
5. Consolidate build entry points and update live integrations, documentation
   and validation harnesses. Preserve historical data and reproduction sources.
6. Run the full applicable validation and matched calibration, then audit every
   end-state requirement above before declaring the goal complete.

## Validation record

Commands and logs are kept under build/ and are not historical performance data.

- make topology topology-check router-policy-check: compiled.
- build/uccl_lite_topology_check: passed synthetic topology checks.
- build/uccl_lite_topology 2 0 3 1: discovered this host's four L4s, PCI paths,
  two NUMA domains, and NIC ports with shuffled rank order.
- build/uccl_lite_router_policy_check: 35 checks passed.
- build/uccl_lite_dispatcher_check: 74 checks passed.
- build/uccl_lite_router_graph_check --devices 2,0,3,1 --workers-per-node 2:
  passed 1000 captured mixed collectives, five replays, producer/consumer
  dependencies and ring wrap.
- Initial dispatcher CUDA runs crashed due to the stale archive members above;
  rebuilding the archive removed the obsolete members and resolved the crash.
- build/uccl_lite_joint_policy_check: passed for 2/4/8-rank synthetic topologies;
  Auto does not force routing, explicit validation mode remains CPU-only.
- build/uccl_lite_dispatcher_cuda_check after archive reconstruction: passed
  external/implicit capture rejection, eager recovery and AR/AG boundaries.
- Graph replay check above passed again after archive reconstruction.
- build/uccl_lite_joint_peer_check 4: passed 225 family/stream transitions,
  aligned/scalar tails, delayed producers, guards, input preservation, device
  restoration and pending destruction.
- make cpu-router: static and shared libraries rebuilt. The static archive now
  contains exactly the current object list; obsolete topology.o/call.o/dispatch.o
  are absent.
- clang-format 14 check and git diff --check passed for the new source.

Local validation hardware: four NVIDIA L4 GPUs in two NUMA domains. These are
correctness checks, not throughput measurements or multi-host validation.

Planner-stage validation: make planner planner-check topology topology-check
cpu-router router-policy-check joint-policy-check build/uccl_lite_dispatcher_check
build/uccl_lite_dispatcher_cuda_check compiled. CPU checks passed after the
Topology ABI extension. See build/planner-integration-build.log and
build/planner-dispatcher-cuda.log for the local run.

The planner-stage GPU dispatcher rerun passed external/implicit capture
rejection, eager recovery and AR/AG boundary checks. The final planner suite
passes 30 checks; clang-format 14 and git diff --check pass.

Peer-executor-stage evidence: build/peer-plan-*-2gpu.log and
build/peer-plan-*-4gpu.log contain forced-family correctness runs;
build/peer-plan-model-gpu.log and build/peer-plan-model-test-2gpu-final.log
contain synthetic-model dispatch checks. Application checks are in
build/peer-plan-bridge2-check.log and build/peer-plan-bridge4-check.log.
MPI is installed at /usr/mpi/gcc/openmpi-4.1.7rc1 (not on the default PATH);
future root Makefile MPI builds should set MPICXX to that bin/mpicxx path.

Final peer-stage checks: the four-GPU joint selftest passed its independent
broad finite BF16/double oracle, odd tails, asynchronous reuse, guards, stale
profiles and capture rejection/recovery (build/peer-plan-joint-selftest.log).
Router policy (35 checks) and dispatcher (74 checks) also passed. No throughput
claim is made for this changed implementation; automatic real-host probing and
performance qualification remain required work.

Hardware-profile-stage checks: planner, peer-planner, topology, router-policy,
dispatcher and profile persistence checks passed. Loading the four-GPU profile
through UCCL_LITE_HARDWARE_PROFILE passed the peer executor's 225 guarded
transition cases (build/hardware-profile-peer-check.log). Actual sweep data is
in build/local-gpu-resources.profile and build/local-gpu-resources.log. Do not
reuse those numbers as validation of subsequent source changes.

The final GPU calibration check passed full-buffer verification, device
restoration and peer-access ownership, and confirmed that applying path results
does not mark physical links measured (build/gpu-calibration-check.log).
The four-GPU synthetic-model executor regression also passed after the topology
schema extension (build/calibration-model-regression.log).

Automatic-preparation-stage evidence: build/automatic-peer-4gpu-final.log and
build/automatic-peer-2gpu.log contain the chosen families and every timing
sample for 15 exact workloads on four and two L4s. Their headers specify dtype,
base algorithm, timing scope, warmups, iterations, repetitions and per-size
selection. These are local qualification runs, not NCCL comparisons or
validation of historical performance curves. build/automatic-joint-selftest.log
passes the independent broad finite BF16/double oracle, odd tails, guards,
input reuse, stale-policy rejection and capture rejection/recovery. The
peer-qualification check rejects a no-op and verifies failure cleanup.

Both rebuilt application bridges pass the alternating AG/RS, scalar-tail and
unprepared-call checks with default automatic preparation enabled
(build/automatic-bridge2-check.log, build/automatic-bridge4-check.log). Planner
32-check suite, peer planner, hardware profile, joint policy, router-policy
35-check suite and dispatcher 74-check suite passed. Formatting and diff checks
passed. The overall migration is still incomplete: primary NCCL runtime
unification, remaining rank/topology restrictions, CPU/NIC resources and
multi-host execution/validation remain outstanding.

Unified-runtime-stage evidence: build/runtime-canonical-id-abi/abi.log is the
final four-GPU strict ABI run after independent session/private IDs. Earlier
four-GPU runs used both core2 and unbound candidate launch placement; none are
performance admission. build/runtime-bootstrap-check.log covers CPU control
1/3/4/5/8/9 ranks, three lifetimes, late root, wrong sessions, peer disconnect
and occupied ports. runtime_catalog.py and backend_spec_check.cc pass, including
heterogeneous descriptor layout mapping and separate restricted registries.
At that stage the network provider still retained its canonical rank/rail
schedule. The later network-planning stage below replaces the rail schedule;
its canonical rank and fixed host-group restrictions remain outstanding.

The generic runtime probe was also checked with CUDA_VISIBLE_DEVICES=2,0:
its participant UUID list follows CUDA ordinals 2,0 from the unrestricted view,
not the four-device NVML inventory. The strict ABI run was repeated after this
probe change and passed. This is discovery validation, not proof that every
retained provider supports arbitrary physical NUMA/rail permutations.

Network-rail-stage evidence:
- build/network-auto-rail-reversed/abi.log: both hosts reversed HCA ordering,
  all 16 rank/lifetime assignments verified and strict ABI passed.
- build/network-auto-rail-asymmetric/abi.log: host 0 reversed, host 1 normal;
  corresponding ranks use different rail indices and strict ABI passed.
- build/network-auto-rail-identity-abi: repeat after including shared topology
  fingerprints in profile identity; both host records contain NIC/path data.
- build/network-auto-rail-sweep-1g: 21 sizes x 3 operations x 2 layouts, two
  candidate jobs per operation plus matched native controls; all correctness
  and route assertions passed. Performance negatives are retained as above.
- Network planner unit checks pass for shuffled GPUs, reordered/noncontiguous
  transport indices, shared NIC load, changed NIC bandwidth, down NICs,
  ambiguous ports and unknown paths. No kernel numerical checker was relaxed.

The final-source repeat in build/network-auto-rail-final-1g passed all 12 jobs
(6 candidate, 6 native), each covering 21 sizes and both layouts. This confirms
1 KiB--1 GiB correctness after topology fingerprints and final formatting.
Its short screening comparison retains 3 slower rows out of 126; worst ratio
1.144010 at 8388608 bytes, all_reduce, layout 0 (0=out-of-place).
The same one-warmup/two-iteration ABBA scope applies; this is not a sustained
performance qualification. No negative row was discarded.

Host/NUMA-stage evidence: build/host-calibration-final-check.log covers enforced
placement, complete copy/input checks, worker resource identities, persistence,
invalid-node cancellation and caller affinity. build/host-stage-gpu-calibration.log
and build/host-stage-peer-check.log cover GPU probe ownership and 225 guarded
peer transitions after the shared schema change. The initial host probe v1 used
different byte accounting locally; v2 resources explicitly use the
host-copy-payload prefix and one-way payload accounting, preventing silent reuse
of old differently normalized values.

Final host/NUMA-stage evidence: the isolated run is
build/host-copy-payload-isolated.profile with
build/host-copy-payload-isolated.log. It was collected after all GPU/network
jobs completed; the earlier final-named run overlapped a correctness job and
is not used as an isolated calibration result. The profile records exact CPU
model/affinity in discovery identity, 32/64 MiB per worker, four destination
workers, two warmups, five iterations and three repetitions, with no baseline.
Current measured path values are payload throughput, not total memory traffic.
The two-host asymmetric-rail ABI regression also passed after the NUMA schema
change: build/host-stage-network-abi/abi.log, 126 calls/rank and two lifetimes.
Host copy resources are now measurable/persistent, but CPU-router execution
plan generation and worker choice have not yet been migrated to consume them.

Rank-generalization-stage artifacts: build/peer-three-{direct,coordinator,bulk,
model-test}.log; build/peer-three-auto.log; build/peer-single-final.log;
build/peer-four-permuted.log; build/peer-single-selftest.log and
build/peer-three-selftest.log. Device permutations refer to logical rank order;
reference output ownership and source reduction order continue to use logical
ranks. These are correctness and local-qualification tests, not historical
performance revalidation. Peer GPU argument capacity is still four; larger
providers and the multiprocess runtime need further migration.

Standalone-runtime-stage evidence: build/peer-runtime-auto-shared.log and
build/peer-runtime-auto-static.log pass without router/NCCL/MSCCL++ linkage;
readelf/nm checks verify that package boundary. build/peer-independent-bridge-check.log
passes the application bridge, while build/core-call-dispatcher-cuda.log and
build/core-call-router-graph.log pass eager/capture handling and graph replay.
See docs/peer-runtime.md for the current standalone API and build entry point.

Variable-host-rank-stage evidence: build/host-rank-validation/ranks{2,3,4}.log
and build/host-rank-validation/permuted.log use the public NCCL/MPI/CUDA ABI only.
Counts are 64/256/4096/65536 elements with BF16; complete extents vary by
collective and world size. Runtime catalogs/layout-range checks pass, and
build/host-rank-legacy-fp32/abi.log preserves the old FP32 profile path. Single-host
eight-GPU optimized modes and larger generic worlds remain untested locally.

CPU worker-plan evidence: build/router-plan-release-cuda.log and the
router_plan_check executable cover synthetic admission/switching/capture and
initialization-free fallback. build/router-plan-dispatcher-regression.log and
build/router-plan-graph-regression.log cover existing GPU checks. Real paired
samples/profiles are in build/router-workers-current.{log,profile,profile.csv};
previous local/final-named runs are retained with their own build identities.
See docs/router-worker-plans.md for timing scope and API constraints. This new
optional selector does not yet replace every application's dispatcher or the
multiprocess host transport's configuration mechanism.

Peer-plan-persistence-stage evidence: peer_profile_check covers exact matching,
shared admission-rule reuse, concurrent updates and corrupt/incomplete records.
The cold/warm GPU run directory is recorded in build/peer-cache-test-path.txt;
both processes pass standalone four-GPU AR/AG/RS with shuffled devices and
scalar tails, and the second process requires fromCache=true. Raw timing scope
is fixed to two warmups, three calls per repetition, three repetitions, host
completion and full guard/input/output checks. No device addresses are reused.

Multiprocess-peer-stage evidence: build/peer-staged-validation/final{2,3,4}.log
contains the final release-store implementation's 168-call/rank ABI checks.
The host candidate also passes the expanded oracle in host-regression.log.
memcheck.log retains the rejected remote-atomic implementation; the corrected
memcheck-store.log has zero memory errors on each rank and a passing full-buffer
oracle. Only API-error reporting was disabled for MSCCL++'s expected VMM/fabric
capability probes, not device memory instrumentation or API memory-access
checking. runtime_catalog.py checks explicit candidate matching and ambiguity
rejection. The declared 32-rank storage capacity is not validation above four
GPUs. These runs do not establish comparative throughput or admit a provider
for automatic routing.

Collective-preparation-stage final evidence:

- build/prepared-local-{four-final,three-final,two-peer-only}.log covers the
  current core, live-cache agreement, mismatched-rank rejection, per-rank mixed
  layouts, scalar tails, old-plan reuse and independent full-buffer validation.
  The two-rank run explicitly disables the host candidate with an unsupported
  experiment configuration and proves that no host transport is initialized.
- build/prepared-local-memcheck-final.log passes the same four-rank core oracle
  with zero memory errors on every rank. Resource probing is disabled for this
  instrumentation run; API-error reporting alone is disabled for MSCCL++'s
  expected VMM/fabric probes. Instrumented timings are not performance evidence.
- build/prepared-nccl-final/{host.json,shapes.json,runtime.profile,abi.log}
  records current library/host identity and exact rank layouts. The V2 automatic
  NCCL entry passes 120 calls/rank across two communicator lifetimes, including
  queued producers, alternating selected transports and capture recovery. Both
  peer and host decisions appear in the raw qualification records.
- build/prepared-legacy-fp32/abi.log preserves V1/historical-header behavior:
  198 FP32 full-buffer calls/rank and two lifetimes on four L4s.
- build/prepared-network-abi/abi.log passes 126 calls/rank and two lifetimes on
  10.10.55.2 and 10.10.55.1 (four L4s each), with opposite HCA enumeration on
  the hosts. This validates the shared ABI change; it does not add network
  auto-qualification or establish new network performance.
- build/prepared-local-cpu-checks.log passes planner/resource/profile/catalog,
  registry, 35 router policy and 74 dispatcher checks. The expanded GPU
  dispatcher and 1000-call/five-replay router graph regressions pass in
  build/prepared-router-{dispatcher,graph}.log. CUDA targets and formatting
  checks pass.

Uninstrumented core timings use four NVIDIA L4s on the documented dual-NUMA
Xeon Silver 4410Y host, BF16 and the staged multiprocess peer baseline. The
scope is max-rank host completion for each exact request: three ABBA rounds,
two warmups and three checked timed calls per block. All samples, including
unqualified candidates, are retained. The selector chooses per exact request;
there is no plotted best-variant envelope or claim against native NCCL. NIC
resource calibration, broader providers, native fallback and persistent
multiprocess qualification remain incomplete.

Network-path-calibration-stage evidence: build/network-calibration-clean/
contains the current probe's successful two-host run, host/transport/probe
identities, GPU process observations, all raw samples, complete-buffer
verification log and summary.csv. There are 32 directed paths, 96 observed
size/path points and 1,440 timed calls (16 B / 1 MiB / 64 MiB, three repetitions,
two warmups and five timed calls per repetition). Hardware is two four-L4
hosts with dual Xeon Silver 4410Y NUMA domains and two advertised 400-Gb/s HCAs
per host. The baseline is absent; the timing scope is sender CPU-posted write
and CQ flush with receiver visibility/validation outside timing. At 64 MiB the
per-path median useful one-way rate is 17.82–18.04 GB/s; this is not pure NIC
capacity, aggregate multi-flow throughput or a best-variant curve.

network_profile_check passes exact endpoint/direction/software matching,
negative evidence, bounded interpolation, atomic replacement and corruption
checks. tests/topology/network_probe_driver.py covers the observed NVML
known-PID cleanup race without accepting a foreign executable/PID. The earlier
flagged run remains at build/network-calibration-slots/; its numerical checks
passed, but it is not the selected calibration artifact. The new network
profile is not yet a runtime admission mechanism: live remote-identity
agreement, shared-flow calibration and network plan integration remain pending.

Measured-network-plan-stage final evidence:

- build/network-calibration-runtime-final/ holds the refreshed 32-path/1,440-call
  resource profile with current GID and GDRCopy implementation/driver identity.
  The earlier clean and identity-v2 artifacts remain measurements of their
  recorded identity definitions, not current matched-profile evidence.
- build/measured-network-final-{matched,stale,missing,none}/ each passes 126
  full-buffer ABI calls/rank and two lifetimes on two real four-L4 hosts, with
  opposite HCA enumeration. The matched case requires matched=1 on every rank;
  stale and missing cases require matched=0 on every rank while retaining the
  physical-capacity plan. The matched run also passes all-rank rejection of
  one rank's missing/corrupt resource-profile configuration.
- build/measured-network-explicit.log and
  build/measured-network-invalid-override.log check the registry's explicit
  rail path and collective invalid-index rejection. The installed CUDA
  host-no-atomic transport requires GDRCopy for signal forwarding; the initial
  disabled-GDRCopy attempt is retained as a negative result, and
  build/measured-network-gdrcopy-rejected.log proves the now-explicit collective
  preflight rejection before transport allocation.
- build/measured-network-local-regression/ passes the unchanged four-GPU FP32
  ABI oracle (198 calls/rank, two lifetimes). CPU physical/measured rail,
  profile, planner, peer and catalog checks pass, as do the affected CUDA builds
  and clang-format 14 checks.

This stage makes no collective throughput claim. The coupled model combines
isolated bidirectional path medians at a common observed payload size with
shared directed physical bounds. Concurrent NIC contention, local collective
traffic and compute overlap remain outside that model. Whole-backend automatic
admission, broader layouts/algorithms and safe native fallback are still open.

Concurrent-network-stage evidence: build/network-concurrent-calibration/
contains the current profile-format-v2 run, source binary hash, GPU process
observations, 32 isolated paths and eight complete concurrent placements.
Each cohort has eight directed flows at 1 MiB and 64 MiB per flow, three
repetitions, two warmups and five timed calls per repetition. All 240 timed
cohort calls retain eight sender durations and eight cohort durations (3,840
values), in addition to the 1,440 isolated timed calls. groups.csv and paths.csv
report every observed configuration without a best-variant envelope. Cohort
timing includes MPI start/end barriers; receiver visibility and full validation
are outside timing. Caller affinity/NUMA policy is restored after setup.

build/network-groups-final-{matched,isolated,subset,legacy}/ each passes 126
full-buffer ABI calls/rank and two lifetimes on two real four-L4 hosts, with
opposite HCA enumeration. The matched case requires concurrent evidence on all
ranks; isolated-only and seven-flow subset fixtures require its absence. The
legacy placement-scope fixture retains the physical plan. The matched run also
proves that changing only one rank's concurrent raw samples rejects all eight
ranks, alongside corrupt/missing configuration checks.

network_group_check and measured_network_plan_check cover exact whole-flow
matching, max-per-call aggregation, sender/cohort preservation, scope/extent
rejection, legacy parsing, partial negative records, a synthetic concurrent
winner overriding the isolated prior, and rejection of overlapping samples or
an unmeasured baseline. Existing planner/profile/catalog regressions pass.
The matched runtime retains the measured topology placement. General network
algorithms, unmodeled local traffic/compute overlap and safe native fallback
remain unfinished; these cohort measurements are not a full-collective gain.

Native-fallback-stage evidence: build/native-fallback-current-{auto,only,host,
single,mixed}.log covers four-rank automatic comparison with resource probes,
initial fallback-only execution, three-rank host-only candidacy, one-rank native
execution, and a deterministic peer-to-native cross-stream chain. The mixed
case adds an explicitly identified test-only delay to its baseline; its timings
are not claims against native NCCL. All modes validate aliases/alignment, scalar
tails, old handles, metadata disagreement, changed native configuration on a
cached plan, revalidation after failure and native communicator use after the
adapter/runtime have been destroyed.

build/native-fallback-memcheck.log passes the instrumented four-rank automatic
comparison/oracle with zero memory errors on all ranks. Its identity explicitly
marks correctness-only instrumentation. The earlier attempt to require a
specific delayed-baseline winner under memcheck is retained at
build/native-fallback-current-memcheck.log; it failed a timing expectation,
not a memory-access check. Instrumented choices/timings are not promoted as
uninstrumented performance evidence.

build/native-fallback-legacy-core.log passes the existing peer-baseline core
oracle. The 35 router policy and 74 dispatcher checks pass in
build/native-fallback-{router-policy,dispatcher}.log. Native adapter linkage
has only CUDA/standard runtime dependencies, not CPU-router/MSCCL++/NCCL;
native NCCL is loaded explicitly through the caller-specified implementation.
The paired benchmark uses BF16, four shuffled L4 devices, three ABBA rounds,
two warmups and three checked timed calls per block, max-rank host completion
including adapter completion tracking. All paired samples are retained; there
is no best-variant curve or claimed native speedup in this run. Transparent
NCCL ABI fallback, grouped/captured prepared calls and broader/native-backed
multi-host selection remain unfinished.

Scalar-generalization-stage evidence:

- build/scalar-abi-final/{two,three,four-permuted,four-packet}/ passes the public
  FP32 NCCL ABI oracle on current code: two lifetimes, full guards/input
  preservation, mixed aliases/alignment, bitwise AG edge values and numerical
  reductions. In-place 64-element shards explicitly exercise four-byte scalar
  arithmetic at 256-byte alignment. Capture rejects before execution and later
  eager calls recover.
- build/scalar-fp32-{three,two,permuted}.log covers generic core execution and
  scalar tails. build/scalar-packet-final.log retains exact qualification
  samples and checks the retained specialization on its admitted NUMA layout.
  build/scalar-native-final.log retains paired native/candidate samples: four
  of 18 requests chose peer-packet, 14 retained NCCL 2.28.7. Hardware is the
  documented four-L4 dual-NUMA Xeon Silver 4410Y host, FP32, max-rank host
  completion, three ABBA rounds, two warmups and three timed checked calls per
  block. This is per-request admission, not a best-variant plotted curve.
- build/scalar-half-final.log validates native-only FP16 without admitting
  BF16/FP32 alternatives. Dtype and invalid FP32 alignment disagreement reject
  before backend initialization.
- build/scalar-memcheck.log exercises generic and retained FP32 paths and reports
  zero memory errors on all four ranks. Its timings are correctness-only data.
- build/scalar-bf16-regression.log and build/scalar-native-regression.log preserve
  the existing BF16 baseline and cross-backend native ordering behavior. CPU
  capability/admission, planner/catalog, 35 router policy and 74 dispatcher
  checks pass. The affected CUDA targets compile with clang-format 14 checks.

The core is typed per runtime instance. This does not yet provide arbitrary
mixed-dtype standard-ABI dispatch, qualified FP16 acceleration, generalized
network algorithms, or transparent grouped/captured native fallback.

Owned-peer-graph-stage evidence:

- `make nccl-runtime peer-graph-check prepared-local-check prepared-scalar-check`
  compiles the affected CUDA executors, unified adapter and regression clients.
  Clang-format 14 accepts the changed C++/CUDA files.
- `build/peer-graph-{bf16,fp32,three,two}.log` passes the guarded graph oracle on
  four shuffled L4s for both dtypes, three shuffled L4s for BF16 and two shuffled
  L4s for FP32. Every case runs 1,000 iterations across each of two communicator
  lifetimes. Cases include in-place chains, graph/eager ordering, repeated
  cross-stream launches of one executable, rank skew and rejection/recovery.
- `build/peer-graph-memcheck.log` passes four-rank FP32 with 100 iterations per
  lifetime and zero memory errors on all ranks. These checks do not time or
  qualify graph candidates. Source and checker artifact hashes are recorded in
  `build/peer-graph-artifacts.sha256`.
- `build/peer-graph-eager-bf16.log` and
  `build/peer-graph-eager-fp32-corrected.log` check existing prepared eager paths
  after relinking against the changed staged executor. The provider catalog
  and dependency-isolation regression also passes.

At this stage graph ownership was confined to `PeerStagedCollectives`. The
subsequent prepared graph selector adds independent graph qualification; the
strict adapter still rejects capture. No eager timing evidence is promoted to
graph admission. See
[owned peer graphs](peer-graphs.md) for the lifetime contract and limits.

Prepared-graph-selection-stage evidence:

- The core now exposes collective `prepareGraph`/`releaseGraph`, stream-ordered
  `launchGraph`, and graph-mode selection inspection. Graph mode participates
  in request consensus and cache lookup; eager evidence cannot qualify a graph.
  Peer/native graph comparison uses the same guarded ABBA admission rule.
  Host and retained packet candidates remain excluded until they implement
  graph execution. The caller adapter checks loaded NCCL graph capability.
- Idle preparation releases retained graph executables before isolated timing
  and reconstructs their fixed bindings afterward. Recovery via a cache hit
  restores required executors too. Mixed graph/eager calls share completion
  ordering; baseline-only initial preparation creates no UCCL candidate buffers
  or workers. The native communicator remains caller-owned.
- `build/prepared-graph-native-final.log` passes four-rank FP32 checks with native
  NCCL 2.28.7 and retains NCCL for all 18 exact requests in each of two lifetimes.
  These are AR/AG/RS at 64/257/4096 elements with two per-rank layout patterns,
  three ABBA rounds, two warmups and three timed calls/block, maximum-rank host
  launch-to-completion timing. All samples and negative outcomes remain in the
  log; there is no best-variant envelope or claimed staged-peer graph speedup.
- `build/prepared-graph-transitions.log` validates four-rank BF16 peer→native and
  native→peer producer/consumer graph chains across streams. Its explicit
  test-only 3-ms baseline delay forces mixed backends and is not NCCL performance
  evidence. `prepared-graph-three.log` and `prepared-graph-fallback.log` cover
  three-rank FP32 peers and two-rank BF16 native-only operation, respectively.
- Every graph test checks full allocations/guards, input preservation, aliases,
  scalar tails, bitwise AG edge values, old graph rebuilding, mode disagreement,
  invalid bindings before resource initialization, rejection/recovery and two
  communicator lifetimes. The native modes also check ordinary native operation
  after runtime destruction. Four-rank FP32 memcheck reports zero memory errors
  on all ranks in `build/prepared-graph-memcheck.log`; its timing is correctness
  instrumentation only.
- Affected CUDA/native targets compile. Existing eager and native stream
  transition oracles pass in `prepared-graph-eager-regression.log` and
  `prepared-graph-native-regression.log`; provider catalog, peer/local planner,
  35 router policy and 74 dispatcher checks pass. Native adapter linkage still
  contains only CUDA/standard runtime dependencies. Source/checker hashes and
  hardware are retained in `build/prepared-graph-artifacts.sha256` and
  `build/prepared-graph-hardware.csv`.

See [prepared graphs](prepared-graphs.md). The strict NCCL ABI still rejects
external capture, and graph-program fusion/qualification, multi-host graphs,
transparent native ABI fallback and the other remaining model/generalization
requirements are not complete.

Algorithm-build-ownership-stage evidence:

- The common runtime and prepared archive now use
  `build/collectives/{peer_packet,host_packet_control,host_packet_device}.o`.
  `src/collectives/module.mk` owns these rules and build-flag stamps. Retained
  peer parameters are defined alongside the algorithm in
  `src/collectives/peer_packet_flags.mk`; the historical integration policy
  references them instead of owning the algorithm's flags. The common peer
  object uses the configured IB-build headers matching its linked MSCCL++.
- Legacy local/host reproduction targets retain separate objects and knobs.
  Changing `UCCL_LITE_NCCL_LOCAL_POLICY` or `LOCAL8_PACKET_BYTES` leaves common
  runtime commands unchanged. `build/algorithm-build-isolation.log` records
  the dry-run comparison. `HOST_PACKET_BYTES` governs the common host candidate
  and is stamped for both control and device compilation. No renamed target
  bypasses capability checks or exact-request performance qualification.
- Removed duplicate scalar/native/prepared build rules. Common and legacy
  targets compile without duplicate-recipe warnings, and the rebuilt prepared
  archive contains each canonical member exactly once with no obsolete
  `local_strict.o`, `multiprocess.o` or `packet.o` members. Build logs are
  `build/algorithm-build-{final,legacy}.log`; the provider catalog check passes.
- Current four-L4 FP32 retained-candidate correctness passes in
  `build/algorithm-build-packet.log`. Four shuffled L4s pass BF16 prepared
  selection/guards in `algorithm-build-host.log` and owned graph/eager/cache
  checks over two lifetimes in `algorithm-build-graph.log`. These validate the
  rebuilt local artifacts; historical performance records are unchanged.
  The legacy eight-GPU target was compiled, not executed on eight GPUs.
- `make help`, the README and build documentation now lead with the common
  runtime, planner and calibration entry points. Historical benchmark names
  remain tied to their recorded configurations. Network reproduction-policy
  dependencies, remaining rank specializations and the other incomplete
  requirements above still need migration.

Network-build-ownership-stage evidence:

- The common runtime and direct network-registry checker now link
  `build/multinode/network_hierarchy.o`, built by `src/multinode/module.mk`.
  The retained implementation's parameters live in
  `src/multinode/network_hierarchy_flags.mk`; historical integration policies
  reference those definitions while compiling their own reproduction object.
  Common runtime commands are unchanged by legacy `base`/`medium24` network
  policies, local policies or local8 packet-size overrides. See
  `build/network-build-isolation.log`.
- `build/network-build-abi/manifest.json` and its logs validate the rebuilt
  common library on two real four-L4 hosts, with reversed HCA enumeration.
  Each rank passes 126 complete-buffer FP32 ABI calls across two communicator
  lifetimes, through 32 MiB. Isolated-path and concurrent-cohort evidence match
  on all ranks. Missing/mismatched profiles, corrupt profiles, and a one-rank
  change to concurrent samples reject collectively on all eight ranks.
  Candidate processes load no native NCCL. This is correctness validation,
  with no baseline timing or claimed performance improvement.
- `build/network-build-direct/{explicit,invalid}.log` validates direct registry
  use on the same two hosts. Explicit local transport slots resolve through
  each host's HCA enumeration, pass guarded AG/RS/AR and two lifetimes, while an
  invalid rail on one rank rejects all eight before transport initialization.
- The main runtime, direct checker and historical network library compile;
  catalog, network-path planner, concurrent-profile and measured-rail tests
  pass. `make network-runtime-check` exposes the direct checker build. Artifact
  hashes are recorded in `build/network-build-artifacts.sha256`.

The retained network algorithm still declares its {4,4}/FP32 contract. Moving
its build ownership does not generalize its rank arithmetic or establish
automatic whole-algorithm admission. Those changes and the remaining end-state
requirements above remain open.

Topology-derived-peer-rank-stage evidence:

- `peerPacketRankOrder` now constructs the retained packet algorithm's two
  execution pairs from physical NUMA placement, independent of logical device
  order. Unknown nodes, 3+1/single-node layouts, missing peer/mapped-host support,
  wrong dtype, unsupported extent/alignment and other rank counts still reject.
  CPU checks include interleaved and noncontiguous NUMA IDs as well as negative
  layouts. `LocalSelection::peerPacketRankOrder` exposes the resulting map.
- The CUDA adapter agrees on ordered device UUIDs, ordinals, NUMA nodes,
  permutation and capacity before initializing transport. It remaps AG output
  and RS input through private prepared storage, preserves logical aliases and
  includes final permutation work in cross-stream completion. Identity mappings
  retain direct buffers. AR uses the existing FP32 tree and numerical tolerance.
  Exact-request qualification includes all new permutation costs.
- Shared host storage now accepts an explicit CUDA device and physical NUMA
  nodes; node masks span multiple words when needed. The CPU-copy helper also
  distinguishes protocol group from physical node. Existing network callers
  retain their default contract, which was revalidated below.
- Four-L4 checks pass with interleaved `2,0,3,1`, reversed `3,2,1,0` and identity
  `0,1,2,3` visible-device orders. `build/packet-mapping-full.log` covers complete
  AG/RS extents through 64 MiB with full guards, mixed aliases/alignment, bitwise
  AG edge values and signed FP32 reductions over two lifetimes. Queued AG→RS→AR
  chains, mapping disagreement and capture rejection/recovery also pass.
  Smaller reversed/identity cases and core selection are in
  `packet-mapping-{reversed,identity,prepared}.log`.
- `build/packet-mapping-memcheck.log` reports zero device memory errors on all
  four ranks for the interleaved quick case. `packet-mapping-native.log` retains
  all paired samples: packet qualifies for five of 18 FP32 operation/count/layout
  requests and native NCCL 2.28.7 remains selected for 13. Hardware is the
  documented four-L4 dual-NUMA Xeon Silver 4410Y host, counts 64/257/65536,
  three ABBA rounds, two warmups and three timed calls/block, maximum-rank host
  completion including permutation work. No best-variant envelope is used;
  instrumented timings are not performance evidence.
- `build/packet-mapping-network/manifest.json` passes 126 guarded ABI calls/rank
  and two lifetimes on two real four-L4 hosts, with reversed HCA enumeration,
  matched isolated/concurrent evidence and all-rank mismatch/corruption
  rejection. `packet-mapping-graph-regression.log` passes the existing BF16 graph
  interface. Core/legacy CUDA targets compile; catalog, local planner,
  35 router policy and 74 dispatcher checks pass. Source/artifact hashes are in
  `build/packet-mapping-artifacts.sha256`.

See [peer packet mapping](peer-packet-mapping.md). This removes the caller-rank
ordering restriction for the two-pair candidate, not its four-participant or
two-NUMA-pair execution contract. General network rank mappings and the other
unfinished end-state requirements remain open.

Network-rank-and-evidence-stage evidence:

- The shared local-stage helper is now named `pairedNumaRankOrder`. Network
  preparation agrees on per-host GPU UUID/device order, NUMA pairs, capacity and
  a global execution-to-logical permutation. Actual CUDA devices and physical
  NUMA IDs are separate from protocol rank/group coordinates. Remapped AG/RS
  staging preserves application source/shard order and aliases, with completion
  events covering the permutation. Cross-host rail selection uses the physical
  endpoints implied by the execution map. The retained layout remains {4,4}.
- The host-packet leader now uses its selected rail, instead of deriving an IB
  slot from its NUMA group. The common network validator defaults to the generic
  runtime catalog/profile and CUDA-visible UUID order; explicit device-order
  assertions and a legacy-profile option preserve both current and historical
  workflows. Hardware inventory/participant UUIDs are recorded separately.
- Initial single-NIC automatic preparation failed because unknown cross-NUMA
  bounds caused rejection before reading resource measurements. That diagnostic
  remains in `build/network-rank-mapping-one-nic/`. Active-port candidates with
  incomplete physical bounds now remain explicitly unknown and require matching
  directed path plus exact complete concurrent-flow evidence. Known physical
  priors are preserved; provisional alternatives cannot win on isolated samples
  alone. Missing/isolated-only evidence with no physical fallback still rejects.
  CPU tests cover these gates, stale identity, reverse-path absence, wrong byte
  extent and measured promotion over a known physical prior.
- Whole-network memcheck then exposed unsupported remote system-scoped atomic
  adds in the retained local peer signaling path, despite passing numerical
  checks. `build/network-mapping-memcheck/run.log` preserves the failure. Each
  direct channel now owns a device-local outgoing counter and publishes through
  a system release store, retaining MSCCL++ acquire waits. Existing single-lane
  publication and phase/stream/graph ordering provide the single-writer contract.
  Counter lifetime follows the transport. The standalone benchmark sharing this
  header was updated and compiled; upstream MSCCL++ was not changed.
- After that correction, `build/network-mapping-memcheck-fixed/run.log` passes
  guarded AG/RS/AR at 4096 elements and two lifetimes on two permuted four-L4
  hosts, with zero memory errors on all eight ranks. The separate bitwise
  permutation kernel passes memcheck for 1/2/4/8/32 shards, scalar tails, input
  preservation and full guards through 128-MiB extents. Its log is
  `build/network-rank-permutation-memcheck.log`. Instrumentation is correctness
  evidence only; it does not qualify performance.
- The final common library passes `build/network-mapping-signal-final-{default,
  permuted,one-nic}/`: 126 full-buffer FP32 ABI calls/rank through 32 MiB,
  two communicator lifetimes, two real four-L4 hosts, no native NCCL loaded.
  Default placement matches path/cohort measurements under reversed HCA
  enumeration. Different device orders assert the map `1,3,0,2,4,6,5,7` and retain
  physical rails when the old ordered-view calibration identity does not match.
  A single exposed `mlx5_0` uses the recorded complete `uniform-0-0` cohort and
  rail 0 for both GPU and host transports. All cases reject profile mismatch,
  corruption and one-rank concurrent-sample changes.
- `build/network-mapping-evidence-final/` validates the current direct-registry
  binary: absent and isolated-only profiles reject the single-NIC unknown path
  on all ranks before transport setup; complete evidence permits automatic
  execution. Local peer-mapping regression, catalog, local/network planners,
  concurrent-profile checks, 35 router policy and 74 dispatcher checks pass.
  Core, legacy and affected standalone CUDA targets compile with clang-format
  14 checks. Source and tested-artifact hashes are recorded in
  `build/network-mapping-artifacts.sha256`.

These are correctness and resource-evidence checks on the documented dual-NUMA
Xeon Silver 4410Y/L4 systems, not a new whole-collective benchmark or best-variant
curve. See [network rank mapping](network-rank-mapping.md). Arbitrary host/rank
counts, automatic calibration of missing network profiles, complete local/host/
network overlap costs, native whole-algorithm admission and transparent ABI
fallback/group/capture remain unfinished.

Global-rank-placement-stage evidence:

- Network host membership now follows gathered host identities, not global
  rank division. Device ownership is resolved by unique actual CUDA devices/
  UUIDs within each host, not rank modulo local participant count. Execution
  host order keeps the original root's host first; NUMA pairing then produces
  the internal permutation. The retained two-host/four-GPU algorithm contract
  remains unchanged. Eight processes on one four-GPU host and duplicate GPU
  assignments reject before transport initialization.
- The strict frontend accepts V3 profiles with one explicit host/device binding
  per logical rank. Runtime tools validate map extent, bounds, types, uniqueness
  and advertised frontend support. C++ parsing and initialization independently
  validate the map and live ownership. V1/V2 and legacy profiles preserve their
  canonical binding semantics; the change does not weaken existing identity
  checks or imply performance admission.
- Distributed providers receive a stable local inventory while resolving
  logical ownership collectively. Changing logical rank assignment therefore
  keeps the physical resource-calibration view stable. Local providers retain
  logical-rank device order. The original root's reachable IPv4 address remains
  an explicit bootstrap requirement.
- `build/network-global-ranks/` passes direct-registry canonical, interleaved
  host-rank and reverse-device cases with matching path/cohort evidence, full
  guards, mixed aliases, three data generations and two lifetimes through
  32-MiB complete AG/RS extents. It also checks duplicate devices. The separate
  one-host rejection is in `network-global-ranks-one-host-rejection.log`.
- `build/rank-inventory-final-{interleaved,reverse,v1,legacy}/` validates the
  final common library through public NCCL APIs: 126 full-buffer calls/rank,
  two lifetimes, two real four-L4 hosts and no native NCCL loaded. V3 asserts
  execution maps `0,2,4,6,1,3,5,7` and `6,4,2,0,7,5,3,1` while retaining matching
  resource evidence. Wrong actual GPU binding and a duplicate rank map inserted
  after launcher verification reject all eight ranks through the frontend.
  Resource-profile mismatch/corruption/group-sample rejection remains intact.
- `build/rank-inventory-local-final/{v3,v2}/run.log` passes the single-host
  prepared FP32 ABI oracle for V3 reverse ownership and V2 compatibility, with
  mixed aliases/alignment, scalar reductions, bitwise AG and two lifetimes.
  `build/rank-inventory-memcheck-final/run.log` passes the current direct checker
  on eight interleaved/reverse ranks with zero memory errors, at 4096 elements
  with mixed aliases and two lifetimes. Instrumented timing is not performance
  evidence. Catalog/map validation, local/network planners, 35 router policy
  and 74 dispatcher checks pass; all affected common/legacy targets compile.

See [global rank placement](global-rank-placement.md) and
`build/rank-placement-artifacts.sha256`. These correctness checks use the
documented dual-NUMA Xeon Silver 4410Y/L4 hosts and do not claim a new speedup.
The frontend still requires explicit candidate/workload profiles. Distributed
whole-algorithm admission against a caller-native baseline, arbitrary algorithm
group sizes, full overlap modeling and transparent ABI fallback/group/capture
remain outstanding.

Distributed-caller-baseline-admission-stage evidence:

- `PreparedNetworkRuntime` adds collective exact-request network/native
  comparison with a caller-owned baseline, stable host/GPU/software/profile
  identity, live plan caching, stream ordering and retained-plan restoration.
  `CollectiveRequest` provides the shared request spelling. Private guarded
  qualification storage and full-buffer verification are shared with the local
  selector in `src/runtime/qualification.cuh`, with internal linkage for each
  runtime's probe kernels and artifact marker.
- Network preflight checks real host/NUMA/peer/mapped-host layout, supported
  compiled CUDA image/cooperative capability, exact extent and network path
  availability before transport construction. Unsupported layouts, scalar tails,
  FP16 and unavailable paths retain the caller. Malformed/inconsistent
  configuration rejects collectively. Operational candidate initialization
  failures remain prepare errors, never per-rank enqueue fallback.
- Eligible requests use three ABBA rounds, two warmups and three timed calls
  per block, retaining 18 baseline and 18 candidate samples. Timing is
  maximum-rank host enqueue through backend synchronization, including network
  send retirement; preparation, rendezvous, input generation and verification
  are outside timing. The >3% median/nonoverlapping-range rule governs promotion.
  Network resources are absent during baseline blocks and restored afterward
  only for selected old/new handles. All candidates must pass complete guards,
  inputs and outputs; an incorrect baseline aborts preparation.
- `build/prepared-network-final/automatic/run.log` passes two real four-L4
  hosts with interleaved logical ranks and reverse GPU ownership. AR/AG/RS at
  64/257/4096/65536 elements and two layouts are prepared over two lifetimes:
  four of 48 decisions select network and 44 retain NCCL 2.28.7. Network wins
  occur at 65536-element FP32 AR in both layouts/lifetimes. All samples and
  negative outcomes are retained, without a best-variant curve or blanket gain.
- Sibling `fallback`, `six-ranks`, `half`, `unavailable` and `no-profile` logs
  pass native-only policy, two three-GPU hosts, FP16, unavailable GDRCopy mode,
  and missing cross-NUMA resource evidence. They create no UCCL network
  transport. `transitions` uses an explicitly labeled test-only 3-ms baseline
  delay to force both backend directions; queued GPU delay and full guards
  verify cross-stream producer/consumer ordering. These delayed times are not
  performance evidence against NCCL.
- Every mode checks metadata/configuration rejection, aliases/alignment, scalar
  tails, old handles, cache reuse, revalidation after failure and native
  communicator use after runtime destruction. `build/prepared-network-memcheck/`
  passes eight ranks with zero memory errors under a correctness-only identity.
  `prepared-network-local-scalar.log` and `prepared-network-local-graph.log`
  preserve the four-rank FP32/native and BF16 owned-graph oracles after shared
  qualification extraction. Catalog/planner checks and affected CUDA builds pass.
- `make prepared-network-runtime prepared-network-check` exposes the core and
  validation client. Shared-library dependencies contain CUDA/MSCCL++ and standard
  libraries, not NCCL, MPI or CPU-router. Source/checker/library hashes are in
  `build/prepared-network-artifacts.sha256`. Hardware is the documented dual-NUMA
  Xeon Silver 4410Y/L4 setup; instrumentation and correctness-only paths do not
  establish a performance result.

See [distributed preparation](prepared-network-runtime.md). This is an explicit
core caller-baseline interface, not transparent standard-ABI native fallback.
Automatic missing-profile calibration, broader accelerated algorithms/layouts,
persistent distributed plan reuse, network graphs, full overlap modeling and
ABI group/capture semantics remain unfinished.

Unified-preparation-stage evidence:

- `PreparedRuntime` accepts a caller's rank/device/size/capacity and baseline,
  discovers actual host/GPU ownership and CUDA visibility, and selects local,
  distributed or caller-only preparation. Applications no longer choose the
  local/network class or provide a rank-derived GPU grouping. A common request
  and evidence interface preserves each candidate's exact eligibility,
  correctness, paired samples and available execution-map/resource diagnostics.
  Algorithm admission remains in the qualified selectors, not host/GPU count.
- Complete consistent single-host views use local candidate qualification;
  complete multi-host views use distributed qualification. Incomplete or
  differing views retain the caller. One visible GPU per rank is validated on
  one and two hosts without constructing accelerator resources. Stable per-host
  inventories and logical local ordering follow the existing selector contracts.
- Local graph qualification remains independent of eager evidence. Distributed
  and caller-only graphs use the caller graph backend, with shared graph/eager
  completion ordering. A common control envelope rejects mismatched operation
  modes, requests, policies and graph release before entering child protocols.
  Fixed bindings, release, rejection/recovery and caller communicator ownership
  remain explicit; native group scopes and external capture are unsupported.
- Unified static linking exposed a genuine ODR layout conflict: local 8-slot and
  network 3-slot `SharedHostPackets` implementations shared inline class symbols.
  `SharedHostPacketsStorage<SlotCount>` now gives them distinct types. The
  original failure is preserved in `build/unified-prepared-local.log`; the
  small-capacity workload was kept. Symbol evidence is in
  `build/unified-storage-symbols.txt`.
- `build/unified-prepared-local-fixed.log`, `unified-prepared-shared.log` and
  `unified-prepared-single.log` pass static/shared local and single-rank cases.
  `build/unified-prepared-cases/` passes two four-GPU hosts, two three-GPU hosts,
  per-process hidden GPU views, FP16/native-only, fallback policy and forced
  local/distributed transitions. Independent CPU references check complete
  buffers/guards, signed reductions, bitwise AG, aliases/alignment, old plans,
  cache reuse, graph/eager dependencies, disagreement/recovery and native use
  after destruction over two lifetimes. Forced transitions use a labeled
  test-only 3-ms baseline delay, not NCCL performance evidence.
- `build/unified-prepared-local-memcheck.log` and
  `build/unified-prepared-memcheck/run.log` pass four and eight ranks with zero
  memory errors under correctness-only identities. The changed shared storage
  also passes the two-host 126-call/rank ABI sweep through 32 MiB with V3 maps
  and rejection gates in `build/unified-storage-network-regression/`; local
  retained packet correctness passes through 64 MiB in
  `build/unified-storage-peer-regression.log`.
- `make prepared-runtime prepared-runtime-check` builds static/shared packages
  and both link-mode clients. Common/legacy targets compile, catalog/planner
  checks and 35 router policy/74 dispatcher checks pass. The common library
  links CUDA/MSCCL++ and standard libraries, not native NCCL or MPI. Artifact
  hashes are recorded in `build/unified-prepared-artifacts.sha256`.

See [unified preparation](prepared-runtime.md). This common core API still needs
standard-ABI integration; it does not complete transparent native fallback,
arbitrary accelerated layouts, missing-profile automatic calibration, full
overlap models, persistent distributed plan reuse or ABI group/capture handling.

Native-communicator C-extension-stage evidence:

- `integration/nccl_lite/prepared_api.h/.cc` exposes versioned `ucclLite*`
  attachment, exact-request preparation, eager execution, owned graphs,
  synchronization, diagnostic and detach entry points. Native rank/device/count
  are queried from the borrowed ready communicator; users provide neither a
  host count nor a provider name. The extension delegates discovery/admission
  to `PreparedRuntime`. It exports no standard NCCL replacements and links no
  native NCCL or MPI; the baseline is loaded from the explicit native path.
- C-compatible options, request/layout arrays and opaque session/plan tokens
  replace C++ types at this boundary. Detached session IDs never reuse; graph
  and eager tokens have distinct tags. Errors return NCCL status plus a
  thread-local diagnostic, without exceptions crossing C. Malformed collective
  requests participate in core consensus. A failed prepare requires recovery
  before execution; invalid local session/output pointers remain caller errors.
- JSON diagnostics retain dtype, exact request/layouts, placement, source
  identities, candidate eligibility/correctness, execution maps/rails and all
  paired timing samples. Length queries and short-buffer writes are checked.
  Released graphs retain their historical selection report but cannot execute.
  The first test incorrectly expected report deletion after release; that
  failure is preserved in `build/nccl-prepared-api-local.log`. The interface and
  test now agree with the core's retained-diagnostic behavior. Report-size
  queries also avoid reading an unused input capacity when no buffer is given.
  Final review found an invalid-mode graph release could return on one rank
  before peers entered consensus; invalid release tokens now reach the core
  rejection envelope, and every FP32 case checks single-rank misuse/recovery.
  Earlier builds and observations remain in `build/nccl-prepared-api-validation/`.
- `build/nccl-prepared-api-release/` contains final shared-library checks for
  four local ranks, eight ranks on two four-L4 hosts, two three-rank hosts,
  one rank, eight ranks with one visible GPU/process, FP16 and forced fallback.
  Each case prepares 36 requests over two lifetimes: AR/AG/RS at 64/257/4096
  elements, out-of-place and mixed per-rank aliases, minimally aligned pointers.
  Independent full-buffer scalar references check outputs, inputs and guards.
  FP32 cases additionally check fixed-binding graphs, cross-stream graph/eager
  chains with GPU skew, mode/capture/released-handle rejection, and recovery.
  FP16 checks are eager only. Every case verifies native communicator use after
  detach. Pure C header compilation and export/dependency checks pass.
- Final local FP32 qualification records 60 correct eligible candidate trials
  and selects peer-packet for one of 36 decisions, retaining NCCL for 35. On
  two four-L4 hosts, 24 requests qualify and time the network candidate but all
  36 decisions retain NCCL. The remaining cases retain NCCL throughout; hidden
  views, FP16, forced fallback and two three-rank hosts initialize no
  accelerated resources. `summary.json` was independently checked against all
  paired samples and the promotion rule. These are local exact-request
  qualification observations, not a best-variant curve or general speedup.
- Hardware is the documented dual-NUMA Xeon Silver 4410Y/L4 setup; baseline is
  NCCL 2.28.7. FP32 uses three ABBA rounds, two warmups and three timed
  calls/block, maximum-rank host enqueue-through-backend completion; probing,
  setup and independent verification are outside timing. Selected graph
  evidence is independent of eager evidence. Source/checker/library hashes are
  in `build/nccl-prepared-api-artifacts.sha256`; staged local and remote binary
  hashes match the final build. No new kernel implementation was introduced.

See [native communicator C extension](nccl-prepared-api.md). This completes the
explicit C attachment/preparation boundary. It does not complete standard NCCL
ABI interposition, native group completion tracking, external-capture handoff,
broader accelerated topologies, automatic missing-profile calibration, full
overlap models or persistent distributed plan reuse.

Automatic-network-resource-stage evidence:

- Added `src/runtime/network_probe.cc` as reusable runtime resource measurement
  using MSCCL++ bootstrap, without MPI/NCCL/CPU-router dependencies. It measures
  all active directed GPU/NIC candidate pairs for the algorithm's actual peer
  mapping, then complete concurrent cohorts from bounded prior rotations and
  uniform candidate-index configurations. Independent host references check
  every input/output word and guard for every call. Private allocations,
  registrations and connections end before collective qualification; caller
  CPU affinity and NUMA memory policy are restored.
- Eligible network prepare automatically measures an absent profile when
  `probeResources` is enabled (the default), after actual topology/capability
  agreement. The unified runtime and C options propagate the same policy.
  Explicit input profiles remain authoritative; disabled/unsupported/native-only
  preparation creates no probe resources. Registration negatives are retained;
  failed operational probing is a preparation error. An unsuccessful automatic
  model cannot silently substitute a physical-only plan. Negative evidence is
  retained even when rail selection cannot admit a candidate.
- The immutable profile is reused across requests and exposed through
  `NetworkSelection::calibration` and unified candidate diagnostics. Cached
  prepare revalidates live host/topology/transport identity. No calibration,
  allocation or independent decision occurs at enqueue; no file is written
  automatically. Applications may explicitly persist the raw profile. This
  completes absent-profile probing for the retained network contract, not
  automatic repair of an explicit stale/incomplete profile or persistent plan
  reuse across runtimes.
- v2 profiles now carry their actual cohort scope. Existing MPI records and v1
  compatibility remain intact; bootstrap records use
  `concurrent-gpu-writes-max-cohort-with-mscclpp-bootstrap-barriers-v1` and retain
  per-host control provenance. The reporter emits the stored scope. CPU tests
  cover roundtrip, unknown-scope rejection, exact full-flow/extent/identity
  matching, negative samples and the existing shared-resource planners.
- `build/automatic-network-final/` passes two four-L4 hosts with interleaved
  logical ranks and reversed GPU ownership. No input profile is supplied.
  Single-HCA probing records 8 directed paths and 1 complete cohort; dual-HCA
  probing records 32 paths and 8 cohorts. Both compare FP32 AR at 4096 and 65536
  elements, retaining NCCL for 4096 and selecting the hierarchy for 65536 in
  these runs. Disabled probing on the single-HCA view retains NCCL because its
  cross-NUMA physical evidence is incomplete. Forced fallback starts no probes.
  These are local exact-request outcomes, not a general speedup claim.
- Resource sizes are 16 bytes, 1 MiB and capacity clamped to 2–64 MiB (2 MiB in
  these checks); cohorts use the larger two sizes. Every size uses 2 warmups and
  5 timed calls in each of 3 repetitions. Isolated timing is host posting/CQ
  flush; concurrent timing includes bootstrap start/end barriers and retains
  every sender/cohort duration. Verification and setup are outside timing.
  GPU/PCIe/NUMA/NIC contributions are not misrepresented as pure NIC capacity;
  unknown component bandwidths remain unknown.
- `build/automatic-network-identity/` passes single/dual-HCA live-identity
  rejection and recovery: changing one rank's CPU affinity invalidates cached
  preparation collectively; restoring it reuses the same plan and raw profile.
  The independent scalar oracle verifies caller buffers and native communicator
  use after teardown. `build/automatic-network-release/` contains final shared
  C-interface four/eight-GPU eager/owned-graph, cache, alias, rejection/recovery
  and forced-fallback checks over two lifetimes. Automatic probing runs once
  per eligible runtime, not once per request.
- `build/automatic-network-profile-abi/` consumes the newly persisted dual-HCA
  profile through the existing strict ABI, with 32-MiB full-buffer coverage,
  V3 rank maps and rejection gates. Its library hash matches the current strict
  build. The standalone MPI probe remains unchanged for its recorded scope.
  Native and MPI dependencies remain outside the reusable runtime. Hardware is
  the documented dual-NUMA Xeon Silver 4410Y/L4 setup with 400-Gb/s NIC ports;
  the whole-collective baseline is NCCL 2.28.7 and the existing 3-round ABBA
  qualification rule still controls algorithm admission.

See [automatic network calibration](automatic-network-calibration.md).
Source/checker/library hashes are in `build/automatic-network-artifacts.sha256`.
Broader accelerated topology/algorithm contracts, remaining host/NUMA resource
models, full overlap modeling, persistent distributed plan reuse and standard
NCCL ABI group/external-capture forwarding remain unfinished.

Topology-derived-peer-tree-stage evidence:

- Added explicit required CUDA access edges to `ExecutionPlan` and generic
  capability validation. `peerTreeParents` generates a root-zero spanning tree
  using measured directed read costs and GPU-memory bandwidth. Unknown rates
  yield an unestimated capability/locality prior, never a zero-cost estimate.
  The tree movement model includes GPU-memory demands, measured paths and
  shared physical PCIe bounds; synchronization/launch overhead is measured by
  qualification rather than claimed by the movement-only estimate.
- `PeerStagedCollectives` now has direct and source-ordered tree modes sharing
  IPC lifetime, completion and owned-graph handling. Tree mode imports only
  neighbors and verifies the exact route, host, dtype, capacity and local edge
  capabilities collectively before allocation. Each logical source is forwarded
  through the tree; AR/RS accumulate in ascending source order in FP32, with one
  final BF16 rounding, while AG remains bitwise. Relay reuse retires adjacent
  readers. RS's full-input forwarding and its cost are explicit, not disguised
  as an optimized partial-reduction tree.
- Caller-baseline local preparation compares four alternatives: direct, host,
  retained peer packet and tree. The generated tree is fixed at context capacity
  for the runtime lifetime, preserving old plans/graphs. Graph qualification is
  independent of eager evidence; native blocks contain no retained alternative
  executors. The same >3% median/nonoverlapping-range rule governs selection.
  `allowPeer` gates both direct and tree; independent C++ family controls serve
  policy/diagnostic use. Parent maps and movement estimates are available in
  common/C diagnostics. Native-free strict prepared selection retains its direct
  baseline requirement.
- CPU fixtures verify connected sparse 2/3/4/8/32-rank graphs, one-way/disconnected
  rejection, exact required-edge revalidation, changed trees under GPU-memory
  or peer-bandwidth bottlenecks, unknown-rate handling and the fourth candidate's
  qualification. All existing planner, 35 router-policy, 74 dispatcher and
  shared-runtime catalog checks pass. Affected CUDA/core/ABI targets compile.
- An initial BF16/star memcheck run reported a full-buffer oracle mismatch but
  no sanitizer access error (`build/peer-tree-memcheck/bf16-star.log`). Added
  diagnostics and eight repeats did not reproduce it. Review identified missing
  per-writer system publication in relay forwarding; every writer now executes
  a system fence and forwarding accesses are volatile. The original failure's
  precise causality was not established by a reproducible input. Both the
  negative result and pre-change repetitions are retained.
- `build/peer-tree-final/` passes the final publication implementation: four-L4
  path/star FP32/BF16 through 32 MiB, three-rank FP32 and two-rank BF16, with
  cancellation-sensitive exact source-order oracles, full buffers/inputs/guards,
  in-place layouts, scalar tails, eager/graph alternation and two lifetimes.
  Seven four-rank memcheck jobs pass the oracle with zero memory errors, including
  four independent BF16/star repeats. Known MSCCL++ capability-probe API reports
  are suppressed; device memory checking/error exit remains enabled.
- Unified three/four-rank `tree` fixtures disable other alternatives and add a
  clearly labeled test-only 3-ms native delay to force tree/native graph/eager
  transitions. They validate old handles, cross-stream GPU skew, rejection and
  recovery, retained bindings and caller ownership. They are correctness
  fixtures, not performance results. Existing direct FP32/BF16 graphs and real
  native BF16 graph selection also pass.
- The real four-L4 C-interface FP32 run contains 36 exact requests over two
  lifetimes: all tree trials are correct, none wins, 32 decisions retain NCCL
  2.28.7 and 4 select peer packet. All paired samples and parent maps remain in
  the log; `summary.json` independently checks the promotion rule. Timing is
  maximum-rank host enqueue-to-stream completion with 3 ABBA rounds, 2 warmups
  and 3 timed calls/block; setup and verification are outside timing. No general
  gain or best-variant curve is claimed. Hardware is the documented dual-NUMA
  Xeon Silver 4410Y/L4 setup.
- `build/peer-tree-network/` passes two four-L4 hosts through the shared C API,
  automatic network resource calibration, owned native graphs and forced
  fallback after the common interface changes. Source and binary hashes are in
  `build/peer-tree-artifacts.sha256`. GPU tree tests use selected path/star edges
  on fully peer-accessible hardware; physically sparse GPU hardware and more
  than four GPUs are not claimed as GPU-validated configurations.

See [source-ordered peer trees](peer-tree-runtime.md). General distributed
algorithm/layout contracts, host/NUMA and overlap models, persistent distributed
plan reuse and standard NCCL group/external-capture forwarding remain unfinished.

Persistent-distributed-preparation-stage evidence:

- Added CPU-only `network_cache.hpp` / `src/planner/network_cache.cc` with a
  versioned eager snapshot containing immutable resource evidence and exact
  qualified requests. Context matching binds hardware/placement, code, native
  baseline/configuration, capacity, transport and UCCL/CUDA execution or
  instrumentation settings. Records additionally bind dtype, operation, extent,
  every rank's layout, execution permutation and rail assignment. No GPU count
  or message-range interpolation admits a cached plan.
- Optional `NetworkPreparationOptions::planCache`, unified `networkPlanCache`
  and `UCCL_LITE_NETWORK_PLAN_CACHE` enable persistence. Rank zero owns file I/O
  and broadcasts validated snapshots; remote nodes need no shared filesystem.
  The default remains no file I/O. Enqueue performs no reads, writes, discovery,
  probing or independent fallback decisions. Global/request native-only paths
  do not consume persisted network records; existing runtime identity remains
  immutable.
- Cached hierarchy wins require a fresh guarded ABBA checkpoint (2 warmups and
  3 timed calls/block) in addition to the original 18/18 samples. Correctness
  failure evicts the record and cannot promote that candidate in the same
  prepare. Performance drift triggers a full new 3-round comparison. Cached
  caller decisions run one guarded baseline block without initializing a
  candidate. Original and checkpoint samples are separate, with `fromCache`,
  cache status and common/C diagnostic fields. Native and live identities are
  revalidated before new evidence is published.
- File updates use an exclusive lock, bounded parsing, synced private temporary
  files and atomic replacement. Concurrent writers merge only identical
  contexts; a different context replaces old records. Up to 4096 newest full
  qualifications are retained within a 64-MiB snapshot limit. Malformed files
  reject without replacement. Resource string/file codecs share validation and
  preserve both legacy formats and MPI/bootstrap cohort scopes.
- CPU checks pass exact request/layout binding, forged decision/sample
  rejection, truncation/version errors, embedded profile roundtrip, concurrent
  writers, context replacement, eviction and bounded retention. Existing
  resource-profile and cohort checks and the runtime catalog also pass.
- `build/network-cache-final/` passes separate process launches on two real
  four-L4 hosts, interleaved logical ranks and reversed GPU ownership: cold,
  positive hit, execution-map miss, embedded-resource mismatch, incorrect
  baseline, performance drift, native cold/hit, configuration miss, malformed
  cache and forced fallback. The other host has no cache `state/` directory.
  Independent full-buffer/guard references, in-memory reuse and native
  communicator ownership are verified.
- The controlled positive fixture uses a labeled dynamic 3-ms scheduling delay
  on the native baseline; these are correctness/control-flow tests, not NCCL
  gains. Cold comparison makes 30 baseline calls; positive reuse makes 10 and
  retains 6/6 checkpoint samples. Native reuse makes 5 with 0/3 checkpoint
  samples and no candidate initialization. Removing the scheduling delay causes
  a checkpoint plus full refresh (40 baseline calls) and returns to native.
- The real shared C-interface run checks 36 requests over two lifetimes. Twelve
  eligible second-lifetime requests reuse persisted native decisions; automatic
  resource calibration runs once instead of twice. Eight-rank forced fallback
  and four-rank local regressions also pass, including owned graphs, rejection
  recovery and caller ownership. Hardware is the documented dual-NUMA Xeon
  Silver 4410Y/L4 setup; baseline is NCCL 2.28.7. Timing remains maximum-rank
  host enqueue-through-backend completion, with setup/verification excluded.
  No historical machine-specific result or best-variant curve qualifies this
  implementation.

See [persistent distributed preparation](network-plan-cache.md). Source and
artifact identities are recorded in `build/network-cache-artifacts.sha256`.
The current cache covers the retained network contract and eager plans; it does
not persist application pointers or transfer eager evidence to graphs. Broader
distributed algorithms/layouts, host/NUMA and full overlap models, and standard
NCCL group/external-capture forwarding remain unfinished.

Dependency-and-overlap-model-stage evidence:

- Added `PlanTask` graphs as an alternative to serial `PlanStage` groups.
  The CPU model follows explicit dependencies, overlaps independent ready work
  and recomputes shared-bandwidth allocations at completion/readiness events.
  Its declared arbitration rule is max-min fairness in fractions of isolated
  progress; startup latency/known overhead occupies no bandwidth. This is a
  resource-model assumption, not a hardware scheduling guarantee.
- Required rates and operational latency remain mandatory. Unknown overhead,
  missing rates, cycles, invalid/duplicate dependencies or task IDs, mixed
  serial/graph models, overflow and excessive graph work produce no estimate.
  Physical-only tasks cannot predict execution. Results include task timelines,
  serialized work, dependency-only critical path and aggregate resource load.
  These model quantities are exposed in common/C diagnostics without claiming
  measured application latency.
- The peer-tree generator now describes per-rank input copies, accumulator
  initialization, forwarding levels, consumption and retirement. Dependencies
  match the executable's stream order and neighbor handshakes; unrelated ranks
  are no longer forced through a modeled global barrier. Tree preparation
  compares the measured-edge prior with deduplicated BFS trees using this graph
  score at its fixed-capacity AR basis, then freezes the chosen tree for the
  lifetime. Missing estimates retain the original capability-aware prior.
  Full correctness and paired measurement still govern actual admission.
- Analytical CPU fixtures produce 500/700/900-us spans for independent,
  shared-memory and fully shared three-chunk/three-stage pipelines. These are
  synthetic fixtures, not machine results. Tests cover uneven sharing, delayed
  arrivals, coupled resources, control/dependency nodes and invalid models.
  Three hundred deterministic random DAGs verify dependency order, isolated
  throughput, total load/critical-path/serial consistency and task-index-order
  invariance. Sparse 2/3/4/8/32-rank tree fixtures independently verify exact
  neighbor dependencies. A fast-edge chain fixture selects a different,
  lower-cost tree after overlap scoring, proving the model affects construction.
- `build/overlap-model-release/` passes the final four-L4 tree/native graph/eager transitions,
  real automatic selection and forced fallback over two lifetimes. All 36 real
  tree trials are correct and have finite graph diagnostics. None wins on this
  configuration: 35 requests retain NCCL 2.28.7 and 1 selects peer packet. The
  transition fixture uses its labeled 3-ms baseline delay only for correctness.
  Timing remains maximum-rank host enqueue-through-completion, three ABBA
  rounds, two warmups and three timed calls/block; setup and verification are
  excluded. Hardware is the documented dual-NUMA Xeon Silver 4410Y/L4 setup.
  No speedup or best-variant curve is claimed for the model update.
- `build/overlap-model-release/` also passes two four-L4 hosts through the common C
  interface, automatic network calibration, owned native graphs and fallback.
  Affected CUDA/core/strict packages compile, runtime catalog checks pass, and
  the 35 policy/74 dispatcher checks remain green. No CUDA kernel body changed.
  Source/binary hashes are in `build/overlap-model-artifacts.sha256`.

See [dependency and shared-bandwidth modeling](overlap-model.md). Explicit
control/polling costs, SM occupancy, unmeasured shared cuts and a complete task
graph for the distributed hierarchy remain outside this movement-only model.
Host/NUMA measurement coverage, broader distributed algorithms/layouts and
standard NCCL group/external-capture forwarding also remain unfinished.

Earlier overlap-model runs remain under `build/overlap-model-final/` and
`build/overlap-model-network/`. Final review additionally clears partial
comparison quantities on unavailable estimates; the final build and all 144
C-interface report rows are verified against the staged local/remote binaries.

Prepared-standard-NCCL-ABI-stage evidence:

- Added `integration/nccl_lite/prepared_abi.h/.cu` and a separate native-enabled
  shared library. After `ucclLiteNcclPrepare` on a ready native communicator,
  ordinary host AR/AG/RS calls may dispatch to topology-qualified executors.
  Actual native communicator handles and unprepared native behavior are
  preserved. Host collective/group/lifecycle wrappers and profiler aliases are
  exported; strict native-free and explicit-handle interfaces remain separate.
- ABI admission compares the exported wrapper, including CPU rank agreement,
  layout matching, executor and completion recording, against direct original
  native calls. Three ABBA rounds retain 18/18 samples and check every private
  output/input/guard. Alternative resources are absent in native blocks. The
  unchanged >3% median/nonoverlapping-range rule prevents a core-only gain from
  being treated as an ABI gain. Native identity and live topology/software/
  configuration are revalidated before restoring admitted resources.
- Every prepared global call agrees on sequence, operation/count/dtype, actual
  rank layouts, device and execution mode before any accelerated dispatch.
  Mismatch/unadmitted work hands execution permanently to native. P2P and
  zero-count requests do not rendezvous with nonparticipants; a local native
  request propagates at the next global call. No lazy transport initialization
  or calibration occurs at enqueue.
- Grouped native work may be deferred, so no event is recorded after each
  deferred native call. Streams instead wait on the frozen final accelerated
  completion. External capture imports that event as an external dependency;
  the lease survives UCCL resource retirement and remains until native
  communicator destruction completes. Internal native calls bypass dispatch
  and group tracking. Explicit async-error queries inspect the UCCL event;
  native internal ready-checks do not do so during capture.
- Initial grouped tests incorrectly depended on an AR result inside the same
  native group. An independent native-only reference also observed the old AG
  input. Formal group checks now use independent inputs depending on the prior
  accelerated operation. Mixed group boundaries stalled both implementations;
  the native-only case reached its 30-second timeout. Prepared first handoff
  now rejects inconsistent grouped/captured status collectively before native
  submission and permits retry after closing groups. Original logs remain.
- A 3-ms scheduling fixture correctly failed admission when a dispatch sample
  reached 4 ms. The final fixture uses a separate native-library artifact with
  a fixed 20-ms host delay solely for transition coverage; no threshold was
  weakened. Capture regression then exposed internal native async-query
  reentry; the explicit/internal distinction fixes it. Both failures and
  diagnostic observations are retained, not counted as performance evidence.
- `build/prepared-abi-verified/` passes four-L4 FP32 eager AR/AG/RS/profiler
  aliases, groups, mixed-status rejection/recovery, capture/replay with changed
  input after UCCL retirement, subset P2P, one-rank layout changes, zero counts,
  non-SUM fallback, group simulation and real native/fallback runs over two
  lifetimes. BF16 checks cover all three operations, 257-element scalar tails,
  minimum alignment, full buffers/guards and native capture. Capture and P2P
  each pass four-rank memcheck with zero errors; only known API-probe reports
  are suppressed. Pure C header, export and dependency checks pass.
- `build/prepared-abi-network/` passes two real four-L4 hosts, interleaved logical
  ranks and reversed GPU ownership, for eager, group, capture, P2P, layout and
  native/fallback behavior. Real 64-element FP32 preparation retains NCCL on
  one and two hosts; forced acceleration uses the labeled delay fixture only.
  Timing is maximum-rank host completion with 2 warmups and 3 timed calls/block;
  setup/rendezvous/verification are excluded. Hardware is the documented dual-
  NUMA Xeon Silver 4410Y/L4 setup, native NCCL 2.28.7. No best-variant curve or
  general speedup is claimed.
- Unchanged `nccl-tests` passes `LD_PRELOAD` with one process/four GPUs, 8-byte
  through 1-MiB AR, validation enabled and zero out-of-bounds values. Initial
  loader-path failure and independent single-GPU launches are retained and are
  not mislabeled as four-GPU validation. Final source and binary identities are
  in `build/prepared-abi-artifacts.sha256`; staged local/remote artifacts match.

See [prepared standard NCCL calls](prepared-nccl-abi.md). This is explicit,
idle preparation for ready blocking communicators, not automatic workload
discovery at communicator initialization. It does not resume acceleration after
native ownership, integrate device-side/RMA operations or provide cancellable
UCCL transport recovery. Revoke/shrink require prior resource release; abort
retains active UCCL state for process-lifetime cleanup. Broader distributed
algorithms/layouts, remaining host/NUMA/control/SM models and automatic front-end
preparation remain unfinished.

Automatic-native-initialization stage:

- Added bounded exact-workload manifests and opt-in standard
  `ncclCommInitRank`/`ncclCommInitRankConfig` preparation. The manifest declares
  dtype/count/layout/capacity and allowed capability classes, not machine names,
  algorithms, rank groups or link rates. Uniform layouts materialize against
  the actual native communicator; topology/resource selection stays in the core.
- Native metadata agreement discovers process/GPU ownership, distributes an
  independent preparation ID and checks manifest/interposer identities before
  entering the existing qualification flow. Metadata buffers/streams are freed
  before timing. Native config contents contribute to the baseline identity.
- Grouped initialization batches all local metadata collectives after the outer
  native group completes and remains native. Mixed grouped/ungrouped, multiple
  local ranks and active prepared resources also retain native. Explicit
  nonblocking initialization and `ncclCommInitAll` pass through. Uniform opt-in
  deployment and effective blocking configuration are required; hidden cached
  native configuration overrides are outside the supported contract.
- Missing/malformed/mismatched supported manifests reject collectively; no
  compatible request retains native. Parser tests additionally reject negative
  unsigned values rather than allowing modular conversion into valid fields.
  Diagnostic state is removed at communicator destruction/abort.
- CPU checks in `build/auto-init-final/` pass 16 checker executables, including topology,
  memory/bandwidth-sensitive plan construction, dependency modeling, profile/
  cache checks, 35 router-policy checks and 74 dispatcher checks. The initial
  CUDA build passes local 2/3/4-L4 automatic AR/AG/RS over two lifetimes, plus
  four-L4 group/capture handoff, real-native selection, forced fallback and
  explicit-prepare regression. Eleven initialization cases cover configuration
  errors, nested/mixed groups, two ranks/process, nonblocking, opt-out, no file,
  incompatible extent and fallback. Full buffers/guards are checked.
- Real native NCCL 2.28.7 retains all three tested 64-element FP32 requests on
  four L4s. Accelerated-branch tests use the separate 20-ms delayed-native test
  library solely for coverage. Qualification remains maximum-rank host
  completion, three ABBA rounds, two warmups and three timed calls/block,
  18/18 samples; setup/verification are excluded. There is no performance gain
  or best-variant curve claim. Hardware remains the documented dual-NUMA Xeon
  Silver 4410Y/L4 setup. Earlier pre-parser-hardening records remain in
  `build/auto-init-verified/`.

See [automatic native initialization](automatic-nccl-init.md). This supersedes
only the preceding stage's missing initialization adapter: future workloads
must still be declared before initialization. Arbitrary workload discovery,
broader distributed executors and the remaining host/NUMA/control/SM cost
models remain unfinished. The current network executor's two-host/four-GPU
support limit is not generalized by automatically discovering its topology.

The first two-host malformed-manifest run exposed an initialization completion
race: debugger stacks showed rank 6 past native abort in the test's MPI barrier
while other ranks were still inside the adapter's native AllGather stream
synchronization. This run was stopped after diagnosis and is a failure, not a
pass. Logs/stacks remain under `build/auto-init-final/`; its staged binary is
preserved in `build/auto-init-before-completion/`. No application/test barrier
was added before abort. The adapter now creates a separate CPU completion ID
before manifest parsing and joins all local participants concurrently after
GPU metadata completion. It waits for every rank before returning any normal
or configuration-error decision. Three successive actual two-host regression
runs, each with two lifetimes, pass the unchanged immediate-abort test in
`build/auto-init-completion-verified/malformed-regression-*.log`.

The first completion implementation then stalled during concurrent MSCCL++
bootstrap teardown in the two-ranks/process case (the captured stack is in
`UnixSocketServer::stop`). That unsuccessful attempt remains in
`build/auto-init-completion-verified/` with its binary preserved in
`build/auto-init-before-host-control/`. Completion now uses the existing
independent `bootstrap::HostControl` channel, so each local rank owns its socket
lifetime and does not start/stop MSCCL++'s process-level Unix service. The root
advertises an IPv4 address discovered from the bootstrap interface or default
route; opaque native/MSCCL++ IDs are never parsed as addresses.

Final automatic-initialization validation is in `build/auto-init-release/`:
26 runtime cases and all 16 CPU checker executables pass. The final build
rechecks local 2/3/4-GPU AR/AG/RS, four-GPU group/capture/native behavior, all
initialization modes and default-interface discovery. Two real four-L4 hosts
pass AR/AG/RS, capture/replay, real-native decisions, malformed-manifest errors
and mixed grouped/ungrouped initialization with interleaved logical ranks and
reversed GPU ownership. Both immediate-abort and two-ranks/process regressions
also pass three additional repetitions, each with two lifetimes. No test-side
synchronization was inserted to hide the original completion race.

Unchanged native `nccl-tests`, one process/four GPUs, passes automatic-mode
`LD_PRELOAD` for 8-byte through 1-MiB AR with validation enabled and zero
out-of-bounds values. This exercises the native `ncclCommInitAll` path. All
44 prepared/final JSON reports match the final ABI build identity; staged
program/library hashes match across both hosts. Source and artifact fingerprints
are in `build/auto-init-artifacts.sha256`. The final real-native 64-element FP32
requests retain NCCL on both one and two hosts. This stage changes startup and
qualification orchestration, not CUDA algorithm bodies or admission thresholds.
The earlier stated hardware, timing scope and no-speedup-claim limits apply.

General direct-exchange execution stage:

- Added `DirectExchangeCollectives` under `src/collectives/` and its common
  `CollectiveBackend` factory under `src/runtime/`. The algorithm has no fixed
  host partition or four/eight-rank arithmetic. Prepare discovers host boot
  identities, UUID ownership, actual visible CUDA ordinals, reciprocal peer
  capability and active NIC paths. Local peer edges use IPC; other edges require
  GPU RDMA and owner-visibility support. Physical bandwidth bounds select a NIC
  prior when available; provisional unknown paths remain explicitly unmeasured.
- FP32/BF16 AR/AG/RS use bounded per-source slots and scalar tails. Source order
  is logical rank order, with FP32 accumulation and one BF16 rounding. Host
  transport flush/collective agreement precedes consumption, and a second
  agreement protects reuse. Calls are synchronous and not graph-capable. There
  is no NCCL/MPI/router dependency in the executor, and no buffer/transport setup
  before prepare. Discarding an unprepared candidate does not touch CUDA.
- Added independent per-host/host-count/total-rank capacity bounds to
  `BackendSpec`, with heterogeneous/synthetic boundary tests. The standalone
  archive contains its two typed candidate registrations and descriptor methods.
  The primary runtime catalog remains seven providers: this new executor is
  **not** automatically admitted yet. A generic candidate competition/evidence
  path must precede activation; existing hierarchy restrictions were not relaxed.
- The first stream-setup harness observed a previous-generation BF16 source in
  a 3+1 in-place AllGather. Pageable default-stream H2D had no dependency on the
  separate nonblocking consumer stream. Initialization now runs on the consumer
  stream; the exact full-buffer/source-order oracle is unchanged. The original
  failure and binary remain under `build/direct-exchange-final/` and
  `build/direct-exchange-network/check.before-stream-fix`. Five full 3+1 repeats
  after this correction pass; causality beyond the established missing stream
  dependency is not claimed from that single original observation.
- Final standalone/package validation in `build/direct-exchange-validated/`
  passes 11 runs: local four-L4 FP32/BF16, actual two-host 1+1/2+2/3+3/4+4 and
  unequal 3+1 layouts, plus eight ranks with one visible GPU per process. Two
  four-rank memchecks report zero device errors; known MSCCL++ capability-probe
  API reports alone are suppressed. Tests include all three operations,
  cancellation-sensitive source order, bitwise gather, full inputs/outputs/
  guards, aliases, scalar/chunk boundaries, repeated generations, two lifetimes,
  rejected call/capture/configuration cases, deferred destruction and the
  common backend factory. CPU/NUMA placement is preserved.
- Affected core/strict/native ABI packages compile. The seven-provider catalog,
  35 router-policy checks and 74 dispatcher checks pass. Current primary-core
  artifacts also pass real-native single/two-host selection and local native
  capture handoff in `build/direct-exchange-package-verified/`; their binaries
  are checked against the final workspace. Source/artifact hashes are recorded
  in `build/direct-exchange-artifacts.sha256`.

This is correctness evidence on the recorded dual-NUMA Xeon Silver 4410Y/L4
machines, not evidence of a latency/bandwidth win. The tested application
capacity is 4 MiB (1 MiB for memcheck), explicit executor chunks are 64 KiB,
and the factory's default is 256 KiB. No benchmark curve or automatic
performance threshold was introduced. More than two physical hosts and more
than eight physical GPUs remain untested. See [direct exchange](direct-exchange.md).
The next required step is multi-candidate distributed qualification and
selection, including generalized evidence/cache identity; this stage does not
complete the requested automatic topology-driven runtime.

Distributed portfolio qualification stage:

- General direct exchange now participates in `PreparedNetworkRuntime` beside
  hierarchy and the caller. FP32/BF16 tails and heterogeneous host sizes can
  qualify independently of hierarchy's retained 2x4 contract. Read-only GPU
  RDMA/owner-visibility, compiled-image and active-NIC checks precede candidate
  construction. Missing physical rates remain unknown; exact checked timings,
  not advertised capacity or GPU count, decide admission.
- Both candidates have separate guarded 3-ABBA evidence (18/18 samples, two
  warmups and three timed calls/block). The existing >3% median/nonoverlapping
  range rule is unchanged. The lowest current qualified median wins. A cached
  hierarchy competes using fresh checkpoint samples; its history cannot hide
  a newly qualified direct candidate. Persistent records remain hierarchy-only;
  direct is freshly measured in a new runtime and cached in memory by request.
- Retaining two executor kinds triggers full qualification with both resource
  sets present. Native blocks have neither. A failed pool check demotes the
  request and updates old common-runtime handles/reports; an unchanged pool
  only requires validation of newly added winning requests. Standard NCCL ABI
  dispatch independently rechecks its actual wrapper pool and repeats when
  the set of admitted executor keys shrinks. Reports retain isolated, pooled
  and cache-checkpoint samples separately.
- Multi-host partial CUDA visibility now reaches distributed selection. UUID
  lookup reconstructs each process's inventory for both the core and ABI;
  peer ranks' CUDA ordinals are never blindly reused locally. Exact direct
  configuration identities are checked at reconstruction and standard-ABI
  promotion. Owned distributed graphs and native external capture retain their
  native path; direct is still synchronous and not graph-capable.
- Added a deliberate scheduling-change fixture: after isolated direct/hierarchy
  qualification, the native test baseline becomes faster during pool checks.
  The old direct handle is correctly demoted to Caller, its common report is
  refreshed, and input/output/guard correctness holds over two lifetimes. This
  is a controlled test fixture, not a measured speedup.
- The shared scalar test harness now produces input on the consumer stream and
  keeps original host input storage separate from expected results until DMA
  completes. Oracle values, tolerance and full-buffer checks were not weakened.
  Dedicated legacy hierarchy/cache suites explicitly disable direct to preserve
  their narrow cache-call-count assertions; the new default-portfolio tests
  separately exercise cached hierarchy plus fresh direct and pool evidence.
- `build/network-portfolio-release/` passes 28 runtime cases and 17 CPU checker
  executables. Actual two-host tests cover BF16, FP32 tails, 3+1 ownership,
  eight one-GPU-visible processes, native-only/FP16 paths, combined core and ABI
  pools, automatic communicator preparation, old-plan demotion, and native
  graph handoff. The prior hierarchy cache passes cold, warm, changed mapping,
  resource mismatch, bad baseline, drift, stale identity, native-cold/warm,
  malformed snapshot and forced fallback. Local four-L4 eager/BF16 regressions
  pass. Two-host/four-rank BF16 memcheck reports zero device errors; known
  MSCCL++ capability-probe API reports alone are suppressed.
- Real NCCL 2.28.7 retains the tested 64/257-element requests on the recorded
  dual-NUMA Xeon Silver 4410Y/L4 systems. The fixed 20-ms delayed library and
  scheduled-delay fixture cover positive/negative admission branches only.
  Timing is maximum-rank host completion; setup, outer rendezvous and full
  verification are excluded, while each implementation's own control and
  transport work are included. No best-variant curve or performance gain is
  claimed. Final ABI report IDs and staged/current binary hashes match across
  both hosts; see `build/network-portfolio-artifacts.sha256`.

See [distributed candidate selection](network-candidate-selection.md). This
supersedes the preceding stage's pending native-backed candidate integration.
The strict native-free profile catalog, persistent direct evidence, larger
physical deployments, single-host partial-view alternatives and fuller
host/NUMA/control/SM resource models remain outside this stage. The overall
migration goal remains active.

Strict distributed provider stage:

- The primary strict catalog now has nine providers, adding automatic FP32 and
  BF16 distributed preparation. Its reference is general direct exchange, not
  native NCCL. Each declared reference request gets five fully checked private
  calls (two warmups/three timed observations), including scalar tails and
  hierarchy-ineligible shapes. Eligible hierarchy requests use the shared
  three-ABBA comparison against that reference, with the existing >3% median
  and nonoverlapping-range gate. Reference resources remain part of both paired
  configurations; hierarchy resources alone are added/removed.
- Host-count, per-host and total-rank bounds are exported through the catalog
  and interpreted by the Python profile builder. Byte granularity is now a
  `BackendSpec` property: the new distributed provider permits scalar tails,
  while retained providers keep their 128-byte limits. Automatic provider
  selection is unique for multi-host FP32/BF16; named legacy experiments and
  single-host defaults remain unchanged. V3 maps support interleaved logical
  ranks and reversed devices on unequal host layouts.
- Runtime probing/launching no longer requires a native NCCL reference file.
  Reference-free identities explicitly record that status. Modern records
  exclude the historical helper's fixed eight-GPU packet presets and retain
  actual runtime/transport environment instead. Supplying a native reference
  still binds its artifact hash without loading it.
- `build/strict-distributed-release/` passes eleven strict runs: two-host 1+1,
  2+2, 3+1, 3+3 and 4+4 deployments, FP32/BF16, minimal one-element profiles,
  mixed in-place layouts, scalar tails, two lifetimes, a bad UUID rejected
  collectively before backend setup, and the existing four-GPU strict FP32
  client. The two-host/four-rank BF16 memcheck reports zero device errors;
  only known capability-probe API reports are suppressed. All successful
  initialization/destruction markers confirm no native NCCL image/fallback.
- Four additional current native-backed regressions pass in
  `build/strict-distributed-native-regression/`: local eager, distributed BF16,
  two-executor ABI portfolio, and real-native selection. Catalog/legacy-provider
  matching, descriptor checks, 35 router-policy and 74 dispatcher checks pass.
  Shared core/ABI artifacts were rebuilt after the descriptor change. Staged
  libraries/checkers/profile helpers match across both hosts and the current
  workspace; fingerprints are in `build/strict-distributed-artifacts.sha256`.
- On the recorded dual-NUMA Xeon Silver 4410Y/L4 setup, appropriate eight-rank
  FP32 requests actually select hierarchy with 18/18 paired samples; BF16 and
  tail requests use the validated direct reference. Timing is maximum-rank host
  completion with setup/verification outside the measured interval. This is
  evidence relative to UCCL direct exchange, not a speedup over NCCL. No
  best-variant curve is claimed. Preliminary reference health-check timings may
  include resources retained for earlier plans; only the separately labeled
  paired observations determine hierarchy admission.

See [strict distributed selection](strict-distributed-selection.md). This
supersedes the prior stage's missing strict distributed provider. Per-rank
isolated CUDA visibility is still outside the host-record V2/V3 strict format;
that case remains supported by the separate native-backed automatic interface.
Generic persistent direct evidence, fuller resource/control models and wider
physical deployment validation remain pending. The overall goal stays active.

Strict per-rank UUID profile stage:

- Added opt-in V4 profiles for the distributed prepared providers. Each logical
  rank has its own host, owned UUID, driver/runtime/ABI versions and identity
  binding. The builder derives host populations from rank records instead of
  equating visible-device count with participating-rank count. Probe/launch
  accepts explicit logical rank and owned local device, with ownership/view
  included in the identity hash.
- C++ initialization resolves UUIDs only within the current process's view.
  Global votes validate UUID ownership, not another process's CUDA ordinal.
  Missing local peer visibility stays missing; the direct reference can use IB.
  Hierarchy is considered only when all ranks have a complete, consistent
  topology view within each host. The V4 capability is explicit, so legacy
  providers cannot accidentally consume a profile that bypasses their mapping
  assumptions. V1-V3 keep their prior interpretation.
- Initial real probing caught a Python device-enumeration variable shadowing
  the requested owned device; it was corrected before any V4 GPU execution.
  The failure remains in `build/strict-rank-view-verified/probe-error.log`, and
  a CPU regression now checks that ownership survives enumeration. Builder
  checks additionally cover rank order, duplicate/bad UUIDs, identity/view
  changes, malformed device bindings and unsupported profile versions.
- `build/strict-rank-view-verified/` passes ten V4 cases: isolated GPU visibility,
  different per-rank full-view orders, mixed views on a 3+1 layout, uniform
  full views, and rejection of bad UUID, rank identity, host population and an
  unsupported legacy provider. Actual two-host runs verify all AR/AG/RS buffers,
  inputs, guards, mixed aliases and scalar tails over two lifetimes, without
  loading native NCCL. Uniform eight-rank FP32 still selects hierarchy through
  measured qualification; incomplete/inconsistent views use the validated direct
  reference. Two-host/four-rank FP32 memcheck reports zero device errors; only
  known capability-probe API reports are suppressed.
- Four V3/local legacy checks pass in `build/strict-rank-view-legacy/`, and four
  current native-backed checks pass in `build/strict-rank-view-native-regression/`.
  CPU catalog, ownership, descriptor, 35 router-policy and 74 dispatcher checks
  pass. The current source/helper and binary artifacts match both staged hosts;
  fingerprints are in `build/strict-rank-view-artifacts.sha256`.

See [strict rank profiles](strict-rank-profiles.md). Hardware remains the two
Xeon Silver 4410Y/four-L4 hosts; larger physical deployments are not validated.
Reference checks use two warmups/three timed observations; hierarchy admission
uses three ABBA rounds with 18/18 paired samples, maximum-rank host completion,
and full-buffer verification outside timing. These are measurements against
UCCL direct exchange, not a speedup over NCCL, and no best-variant curve is
claimed. This closes the strict per-rank visibility gap from the previous stage;
persistent direct evidence and fuller resource/control modeling remain pending.
The overall goal stays active.

Direct persistent-evidence stage:

- Versioned the network plan snapshot by candidate kind. Hierarchy-only files
  retain the V1 encoding; V2 supports mixed hierarchy/direct records, FP32/BF16,
  2–32-rank exact layouts and per-rank direct configuration identities. Candidate
  kind participates in update/lookup/eviction keys. Old hierarchy constraints,
  raw 18/18 samples, the 64-MiB/4096-record bounds, locking and atomic replacement
  remain enforced. Direct writes without resource updates preserve an existing
  same-context hierarchy calibration.
- Direct winners now reuse historical evidence only after a fresh 6/6 ABBA
  checkpoint and matching reconstructed execution identities. Identity/performance
  drift refreshes full qualification; incorrect candidates are evicted, while
  bad baseline/cache data errors before promotion. Cached caller decisions use
  one checked baseline block and create no direct executor. Both candidates
  compete using current checkpoint samples, never historical durations; final
  resource-pool qualification remains separate.
- Common C reports expose per-candidate cache flags/status and historical versus
  checkpoint samples. Whole-decision cache status is true only when all eligible
  candidates were reused and no fresh pool qualification was required. The
  normal 3% median/nonoverlapping-range gate was not changed.
- CPU schema tests cover V1/V2 roundtrips, mixed-kind keys, typed scalar tails,
  variable rank extents, malformed identities/decisions, independent eviction
  and bounded/concurrent updates. A targeted test exposed loss of hierarchy
  calibration when a direct-only writer supplied no profile; its failure is
  retained in `build/direct-cache-verified/merge-regression-before.log`, and the
  corrected merge passes the same regression.
- `build/direct-cache-release/` passes 36 runtime cases. BF16 1+1 and FP32 3+1
  cases verify cold/warm, identity/layout changes, bad baseline, performance
  drift, stale context, cached caller, malformed files and fallback, including
  exact native-call counters (30 cold, 10 positive warm, 5 cached caller, 40
  refresh). Eight-rank portfolio tests reuse both candidates and validate the
  combined pool; an intentionally tiny historical direct time cannot displace
  the faster current hierarchy checkpoint. C API reports show all 18 second-
  session requests cached with independent 6/6 checkpoint arrays.
- The standard ABI pool and retained hierarchy V1 cold/warm/drift/caller cases
  pass. Two-host/four-rank BF16 cold and warm memchecks report zero device errors;
  only known MSCCL++ capability-probe API reports are suppressed. Strict V4
  isolated-view BF16 and uniform-view FP32 regressions pass. Current/staged
  artifacts match both hosts; see `build/direct-cache-artifacts.sha256`.

Hardware remains the recorded two Xeon Silver 4410Y/four-L4 hosts. Timings are
maximum-rank host completion: cold 3 ABBA rounds, two warmups and three timed
calls/block; positive warm one ABBA round. Setup and external verification are
excluded, while candidate-internal work is included. Delay fixtures and modified
historical samples are test-only coverage, not measured gains over NCCL. No
best-variant curve is claimed. Cached caller decisions intentionally do not
search for newly emerging gains without invalidation. This closes the persistent
direct-candidate evidence gap; broader resource/control modeling and larger
physical deployments remain pending, so the overall goal stays active.

Direct traffic/dependency model stage:

- Extracted `directExchangeSteps` into the CPU planner and made the actual
  executor consume it. AR/AG staging and RS per-destination staging therefore
  share their sender ordering with the cost graph. The prepared global IPC/IB
  matrix drives both transfers and diagnostic traffic classification. No
  schedule allocation or model construction was added to enqueue.
- Added exact logical GPU read/write, peer/IB payload, control payload, copy,
  reduction, stream-synchronization and flush counters. Shared status/call
  structs determine control payload sizes. The backend and common C report
  expose counters for actual prepared routes; cached caller-only decisions do
  not fabricate a current direct report. Bytes are logical traffic, not measured
  DRAM/NIC transactions or a performance-admission claim.
- Added a bounded direct `PlanCandidate` generator using the existing topology,
  resource-demand and dependency estimator. Sender chains overlap across ranks;
  memory/path demands and shared physical bounds contend. Explicit conservative
  publication/retirement joins separate chunks. All extra costs default to
  unknown. Missing GPU memory/path/control data yields no usable estimate, and
  oversized graph materialization rejects. The runtime does not yet calibrate
  all inputs to this advisory direct graph automatically.
- An independent CPU oracle verifies all traffic and every generated dependency
  for FP32/BF16, all three operations, tails and 2/3/4/8/32 logical ranks.
  Synthetic calibrated fixtures demonstrate a memory-bandwidth-driven choice
  reversal between existing peer-direct and staged-direct models, shared NIC
  load and chunk control overhead. Invalid/asymmetric routes, duplicated memory
  accounting, missing data and model-budget overflow reject.
- `build/direct-model-verified/` passes eleven executor checks, including four-L4
  local operation, actual two-host 1+1/2+2/3+3/4+4/3+1 layouts, isolated GPU
  visibility, the common backend and two four-rank memchecks with zero device
  errors. Only known capability-probe API reports are suppressed. All existing
  full-buffer/source-order/alias/tail/lifetime checks remain intact, and route
  counters are compared with the actual prepared transports.
- Four native-backed/API checks pass in `build/direct-model-runtime/`, including
  the two-executor ABI pool and BF16 capture handoff. All 36 C API traffic reports
  match independently derived operation/route formulas. Two strict V4 regressions
  pass in `build/direct-model-strict/`. Sixteen CPU checker executables pass.
  Final source/binary identities and both hosts' staged artifacts match; see
  `build/direct-model-artifacts.sha256`.

See [the direct model](direct-exchange-model.md). Hardware validation remains the
recorded two Xeon Silver 4410Y/four-L4 hosts. These tests establish correctness
and accounting, not speedup over NCCL; the 20-ms native-delay fixture is only
branch coverage. No best-variant curve is claimed. The model's global-drain
approximation, control calibration, GPU cache/SM effects and wider physical
validation remain explicit limits. Exact guarded timing still governs actual
admission, and the overall migration goal stays active.

Automatic direct operational-calibration and final scope-audit stage:

- The distributed prepared runtime now measures the first direct executor's
  actual staging/self/gather/reduction operations, all directed IPC/IB flows,
  entry agreements, publication/retirement and required owner visibility.
  Private buffers are checked completely after every probe. Calibration runs
  before candidate timing and is reused within the runtime; configuration
  identities must match a reconstructed executor. Enqueue does no probing.
- Real small-copy observations invalidated a forced positive linear bandwidth
  fit. Its failure is retained in `build/direct-calibration-verified/`.
  Operation-duration tables now accept nonmonotonic positive data, interpolate
  only inside measured payload coverage, and reject missing/mixed-scope data.
  Known physical bounds constrain the graph; unknown shared cuts are explicit.
  The GPU service envelope is scoped to this implementation and working set,
  not advertised as physical DRAM bandwidth.
- Common C reports expose model identity, observation count, unavailable reason
  and unmodeled resources, alongside existing task/resource estimates. Models
  are held in memory and do not replace exact guarded ABBA admission. Disabled
  probing and cached caller decisions do not create a direct model; positive
  cache reuse can perform a fresh calibration outside its timed checkpoint.
- The final build passes 12 runtime cases in
  `build/direct-calibration-release/`: local four-GPU FP32; actual two-host
  1+1 BF16, 3+1 FP32 and 4+4 FP32 probes; isolated-view 2+2 BF16 memcheck;
  common C reporting; disabled probing; the eight-rank candidate portfolio;
  BF16 ABI handoff; ordinary eight-rank native ABI; and direct cold/warm cache
  checks. The memcheck reports zero device errors with only handled capability
  probe API reports suppressed. All 36 C model reports have finite estimates,
  68 observations, current configuration bindings and retained 18/18 admission
  samples. Calibration identity reuse is checked across each runtime's requests.
- Strict V4 isolated BF16 and uniform eight-rank FP32 regressions pass in
  `build/direct-calibration-strict/`, with full buffers, aliases, scalar tails,
  two lifetimes and no native NCCL image. All 13 planner checker executables
  pass. Source/binary identities and matching artifacts on both hosts are in
  `build/direct-calibration-artifacts.sha256`. An omitted launcher argument
  and an incorrectly staged single-host checker were corrected; their failed
  logs remain separately named. Neither required an executor/oracle change.

The [calibration report](direct-model-calibration.md) records hardware, dtype,
timing boundaries, repetitions and signed prediction errors. The comparison is
against separately timed direct execution, not a speedup over NCCL. Earlier
larger errors remain recorded; there is no accuracy-bound or best-variant-curve
claim. Delayed-native fixtures exercise transitions only.

The final [scope audit](topology-runtime-status.md) checks the original request:
shared discovery and resource models, executable topology-derived mappings,
capability filtering before measurement, preparation/caching outside enqueue,
optional independent routing, and responsibility-based source/API organization.
These primary paths are implemented. Remaining limits concern supported
algorithms, model fidelity and physical deployments; specialized implementations
retain explicit constraints and fallback instead of presenting GPU-count modes
as general topology support.
