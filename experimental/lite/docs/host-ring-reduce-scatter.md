# Opt-in host-memory ReduceScatter ring

`PacketPipelineOptions::hostRingReduceScatter` selects a standalone BF16
ReduceScatter implementation through mapped, NUMA-local host buffers. It is
experimental and disabled by default. It does not require GPU P2P access, call
NCCL, or create CPU reduction workers. The CPU publishes commands and can submit
cached CUDA graphs through the existing NUMA launch lanes.

Each GPU runs one transport kernel. Every CTA owns the same strided set of
16-byte vectors for the lifetime of the prepared pipeline. A rank uploads the
preceding destination shard, then performs `ranks - 1` ring exchanges, adding
its local contribution on each exchange. The final exchange writes its own
output shard. The two pinned banks hold intermediate BF16 partials. Each hop
rounds to BF16; results can differ from the CPU NUMA-partial schedule, flat FP32
GPU accumulation, and NCCL's chosen reduction order.

Per-CTA system acquire/release signals express both readiness and reader
credits. A writer waits for the next rank to consume a bank before reusing it.
The final progress state releases both banks for the next command. CTA barriers
order all participating writers before the leader publishes readiness. Loads
bypass GPU L1 so later generations cannot reuse stale mapped-host data. A
separate command ticket controls CUDA graph replay; the graph does not capture
a fixed host generation.

The option requires:

- SM transport, GPU reduction and GPU progress; no concurrent or fused producer.
- Complete-source, complete-destination ReduceScatter with at least two ranks.
- Separate, nonoverlapping, 16-byte-aligned input and output buffers.
- Each destination shard has a multiple of eight BF16 elements and fits in
  `packetBytes`. `maxTensorBytes` still bounds the complete input.
- A common resident grid, capped at 128 CTAs and by the least permissive GPU's
  occupancy. Prepared device order and streams remain fixed.

Other operations and unsupported layouts are rejected by a direct pipeline.
The adaptive dispatcher preserves the caller's fallback for unsupported or
uncalibrated Auto workloads, without allocating a router. Force mode rejects
unsupported workloads. Ring has a distinct software/configuration identity;
existing flat or CPU-reduction measurements cannot authorize it. An application
must calibrate exact message size, dtype, placement, topology, software and timing
scope before enabling it. GPU count alone does not select the algorithm.

The test target `uccl_lite_host_ring_check` covers an independent scalar oracle
including all BF16 encodings (NaNs compared by class), changing extents,
more queued calls than the command-ring depth, repeated multi-call CUDA graphs,
output guards and recovery after rejected calls. `dispatcher_check` covers
allocation-free fallback and calibration separation. Current experiment status
and raw runs are recorded in
[`rs_recovery_8l4_20261114`](../benchmarks/data/rs_recovery_8l4_20261114/README.md).
The benchmark's fixed `current` schedule uses this option only at 1–8 MiB;
`previous` reproduces the pre-recovery selection in the same executable.
Library defaults remain unchanged.
