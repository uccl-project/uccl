# Lite-collective AllReduce design

This document records the current native AllReduce design for the L4 no-NVLink
/ no-GDR testbed. Performance tables live in
[`perf-allreduce.md`](perf-allreduce.md), plot artifacts live in
[`plots/allreduce/summary.md`](../plots/allreduce/summary.md), and the generator
is [`plots/allreduce/plot_allreduce.py`](../plots/allreduce/plot_allreduce.py).

## Current dispatch shape

`ncclAllReduce` first attempts topology-specific native paths, then falls back
to the real NCCL library only when native execution is unsupported. The native
entry point is `runSendRecvAllReduce` in
[`native_collectives.cu`](../nccl/native_collectives.cu).

The current native routing is:

| Topology | Primary native path | Default scope |
| --- | --- | --- |
| `1nx4g` | Existing registered MSCCLPP algorithms selected by `algorithm_selector.cc` | all sizes, but 64KiB-1MiB remains weak |
| `2nx1g` | Dedicated two-channel ring/SIMPLE copy-engine path | `float/sum`, `>=64MiB`; smaller sizes keep previous routing |
| `2nx2g` | Mapped-host small AllReduce, then generic/native fallback | `float/sum`, `<=64KiB` small path |
| `2nx4g` | Mapped/two-leader small paths; chunked and pipelined RS+AG large path | `float/sum`, small path plus large divisible counts |

## 2nx4g: ReduceScatter plus AllGather

The main multi-GPU two-node algorithm is still:

```text
full input on each rank
  -> native ReduceScatter produces this rank's reduced shard
  -> native AllGather distributes all reduced shards
  -> full reduced output on each rank
```

This is the right shape for a no-GDR PCIe system because it reduces data before
the broadcast-style phase and reuses the tuned ReduceScatter and AllGather
building blocks.

For 2nx4g, large shards are processed by
`runChunkedGpuLocalReduceScatter2Node`, which delegates to
`runPipelinedNumaPairReduceScatter2Node` when the layout is exactly four local
ranks. The pipelined path double-buffers NUMA-pair host partials and uses
per-slot RDMA ready signals to avoid source-buffer reuse races. After the local
shard is reduced, `runSendRecvAllReduce` calls Lite AllGather on that shard.

Small 2nx4g messages use `runSmallMappedAllReduce2Node` up to 64KiB and
`runSmallTwoLeaderAllReduce2Node` for the 128KiB/256KiB window. Those paths
reduce host-staged small tensors with CPU leaders and mapped host memory. They
beat NCCL for most small points, but the 64KiB and 128KiB rows remain slightly
behind the best NCCL run.

## 2nx1g: two-channel ring/SIMPLE copy-engine path

NCCL debug output for 2nx1g no-GDR selects `AllReduce -> Algo RING proto
SIMPLE` with two channels and GDR disabled. The committed Lite path mirrors that
data order for large messages:

```text
for each channel chunk:
  stage peer half D2H
  RDMA-write peer half to the remote receive slot
  H2D the remote own half and GPU-add it into output
  stage reduced own half D2H
  RDMA-write final own half to the remote receive slot
  H2D the remote final peer half into output
```

Implementation details:

- `runTwoRankRingSimpleAllReduce2Node` is scoped to `nRanks == 2`,
  `nRanksPerNode == 1`, `float/sum`.
- It uses the existing leader connection and pair connection as two independent
  ring channels.
- It uses 4MiB channel chunks by default; `MSCCLPP_NCCL_2RANK_RING_CHUNK_BYTES`
  exists for benchmarking.
- It is enabled by default only for `>=64MiB`; smaller messages keep the
  previous path because the CPU-driven handshake overhead is still too high.
- `MSCCLPP_NCCL_2RANK_RING_ALLREDUCE=1` forces the path for benchmarking.
- It uses an independent `twoRankRingEpoch`, not `pairEpoch`, so it cannot
  interfere with pair-RDMA ack protocols used by ReduceScatter paths.

This path improves large 2nx1g AllReduce: the latest 128MiB validation was
correct and reached about 15.66GB/s versus a fresh NCCL no-GDR baseline around
14.7GB/s.

## 2nx2g: small mapped-host path

2nx2g uses the generalized mapped-host small AllReduce path for `float/sum`
messages up to 64KiB. The path is deliberately not generalized to arbitrary
local-rank counts; the supported local counts are currently two and four.

Larger 2nx2g messages remain on the generic native/fallback routing. They are
correct, but they do not yet beat NCCL across the 1MiB-1GiB band.

## 1nx4g: single-node selector path

Single-node AllReduce still uses the registered MSCCLPP algorithm selector. It
is competitive for tiny messages and large messages, but it loses badly in the
64KiB-1MiB window. Attempts to force generic native RS+AG, `allreduce8`, and
fullmesh selector variants were slower or unsafe for in-place buffers.

The likely future direction is a dedicated single-node copy-engine
ReduceScatter followed by the existing fast single-node AllGather, with explicit
input staging for in-place safety.

## Performance artifacts

The generated AllReduce plots mirror the AllGather plot layout:

- [`plots/allreduce/plot_data.csv`](../plots/allreduce/plot_data.csv) contains
  out-of-place and in-place nccl-tests metrics plus source-log paths.
- [`plots/allreduce/summary.md`](../plots/allreduce/summary.md) contains
  benchmark settings, summary tables, per-topology latency/bandwidth tables, and
  the source-log inventory.
- PNG plots are split by topology and metric:
  - `*_latency_128B_1MiB.png`
  - `*_busbw_1MiB_1GiB.png`

Current out-of-place summary from the generated clean rerun
(`.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/`; each sweep
waited for zero GPU utilization and no compute processes before starting):

| Setup | Latency geomean Lite speedup | BusBW geomean Lite/NCCL | 1GiB Lite GB/s | 1GiB NCCL GB/s | 1GiB ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1nx4g` | 0.44x | 0.81x | 17.54 | 17.54 | 1.00x |
| `2nx1g` | 0.09x | 0.63x | 16.23 | 14.55 | 1.12x |
| `2nx2g` | 1.15x | 0.91x | 14.71 | 15.34 | 0.96x |
| `2nx4g` | 1.58x | 1.02x | 16.28 | 14.82 | 1.10x |

Interpretation:

- 2nx4g is the most complete AllReduce win: small messages are mostly faster,
  and large messages beat NCCL after 128MiB. The 64MiB large-row source log has
  a known outlier and should not be over-interpreted.
- 2nx1g now has a large-message path that is competitive with, and sometimes
  faster than, NCCL from 64MiB upward. The sub-64MiB path remains blocked by
  proxy/handshake latency.
- 2nx2g small-message latency improved, but larger messages still trail NCCL.
- 1nx4g needs a dedicated midrange algorithm; the current selector path is not
  enough for 64KiB-1MiB.

## Rejected alternatives

The following measured alternatives were not kept:

- AVX512 CPU AllReduce for 2nx4g: correct after fixes, but CPU reduce and host
  memory contention limited 1GiB to about 5.9GB/s versus NCCL around 15.4GB/s.
- 2nx4g small-message staged, hybrid, replicated, partitioned, payload-signal,
  `updateAndSync`, flush-reordering, and CPU worker-pool variants: did not close
  the 64KiB/128KiB gap.
- 1nx4g generic native RS+AG and selector swaps: slower and/or in-place unsafe.
- 2nx1g direct-QP full-exchange, async stream, four-slot, and zero-copy-reduce
  variants: correct but below NCCL; best direct-QP async result was about
  11.6GB/s at 32MiB versus NCCL around 14.7GB/s.
- A closer 2nx1g GPU-FIFO prototype: correct but slower than the copy-engine
  ring path on this codebase. Raw mapped-host GPU writes are around 25GB/s, so
  the remaining midrange bottleneck is the synchronization/proxy structure, not
  mapped-write bandwidth.

## Open limitations

- The 2nx1g ring path is CPU-driven and only enabled by default for large
  messages. A true NCCL-like persistent proxy/FIFO would be needed to compete
  below 64MiB.
- `count % worldSize == 0` remains the best-supported multi-node RS+AG shape.
  Irregular counts use fallback paths.
- In-place and out-of-place AllReduce can diverge materially. The generated
  plots use out-of-place for consistency with AllGather, but `plot_data.csv`
  preserves in-place rows for analysis.
