# Current topology-driven runtime

The primary runtime now discovers the participating hardware and prepares a
qualified collective plan. Applications no longer select a machine mode such
as `local4`. This completes the core organization and selection migration;
it does not establish feature or performance parity with NCCL on every system.

## Responsibility and execution

| Responsibility | Implementation | Current behavior |
| --- | --- | --- |
| Discovery | `src/topology/` | GPU UUID/PCI identity, CUDA access edges, PCIe hierarchy and shared links, NUMA, network ports and software identity |
| Resource evidence | `src/topology/`, runtime probes | GPU/host-memory and transfer measurements; measured, advertised and unknown quantities remain distinct |
| Planning | `src/planner/` | Exact workload capabilities, rank groups, sparse trees, directed paths, resource contention and dependency costs |
| Preparation and selection | `src/runtime/` | Automatic participant discovery, candidate construction, independent correctness, paired timing, plan reuse and resource lifetime |
| Execution | `src/collectives/`, `src/multinode/` | Prepared peer, staged, packet, tree, hierarchy and direct algorithms |
| Optional CPU routing | `src/cpu_router/` | NUMA-local worker configurations, exact evidence gates and caller-supplied fallback |
| Application/ABI adaptation | `integration/` | Explicit C API, NCCL ABI dispatch, opt-in initialization and native ownership |

The common entry point is [PreparedRuntime](prepared-runtime.md). It accepts a
communicator and exact requests, discovers placement and selects the applicable
local or distributed preparation flow. The application does not supply host
counts or NUMA groups. Algorithm admission additionally checks dtype, extent,
layout, alignment, capture support and transport capabilities.

Resource models determine supported paths and planning priors, including
memory-sensitive tree construction and shared GPU–NIC resource selection.
Current guarded collective measurements decide whether an eligible candidate
beats the baseline. A bandwidth estimate alone never bypasses correctness or
admits a candidate. Plans and calibration are bound to hardware, software and
execution configuration identities.

Preparation owns probes, buffers, transports, workers and tuning. Enqueue
uses an existing plan. Caller-only fallback initializes no router resources.
Native-backed operation preserves the caller's NCCL communicator; the separate
strict runtime uses a validated reference and loads no native NCCL image.

## Public entry points

- `make prepared-runtime`: common core API.
- `make nccl-prepared-api`: preparation through a native communicator C API.
- `make nccl-prepared-abi`: ordinary NCCL calls using prepared plans, with
  native group/capture handoff and optional automatic manifest preparation.
- `make nccl-runtime`: strict experimental NCCL frontend and runtime catalog.
- `make topology calibrate-hardware planner`: discovery, calibration and
  CPU-only planning libraries/tools.

Historical benchmark directories and restricted compatibility targets retain
their recorded names. They do not define the primary runtime's algorithm
identities or create GPU-count-only dispatch rules.

## Explicit limits

- The general direct executor supports FP32/BF16 and 2–32 logical ranks.
  Physical validation covers up to two hosts and eight L4 GPUs, including
  unequal host populations and isolated CUDA visibility. Capacity bounds are
  not validation of larger deployments.
- The retained network hierarchy requires two actual four-GPU hosts and its
  full peer/NUMA/IB capability checks. Its name describes the algorithm;
  the restriction remains visible rather than being presented as generality.
- Distributed eager calls can use direct or hierarchy. Distributed graphs
  currently retain the caller's graph implementation. A single host with
  incomplete/differing CUDA views also retains the caller.
- The ABI preparation contract requires exact workload/layout information;
  its automatic initialization adapter consumes an opt-in manifest. Arbitrary
  first-call tuning during capture is not supported.
- The [direct operation model](direct-model-calibration.md) is advisory.
  Its host-completed working-set service data is not a physical DRAM peak;
  unknown shared cuts, cache behavior and control overlap remain explicit.
  Models need not predict every timing accurately for measured qualification
  to reject an unsuitable candidate.

The [migration record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/topology-runtime-refactor.md) preserves the sequence of
changes, negative results and validation artifacts. Earlier statements there
about pending work describe their stage, not the present capability matrix.
