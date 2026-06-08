# Lite-collective AllGather design

## Goal

Lite AllGather targets two-node consumer-GPU systems where inter-node data must
stage through pinned host memory.  The implementation keeps NCCL API
compatibility, but avoids the generic grouped P2P AllGather path for the tuned
two-node cases.  Instead it uses host-memory algorithms that match the topology:

- `2nx1g`: one rank per node, optimized for low latency and then pipelined
  host-staged chunks.
- `2nx2g`: two local ranks per node, one host slab per node.
- `2nx4g`: two NUMA/NIC-local rank groups per node, one host slab per group.

The benchmark size reported by `nccl-tests` is the total AllGather output size
(`bytesPerRank * nRanks`).  Most code thresholds are expressed either in total
output bytes (`fullBytes`) or per-rank bytes (`bytesPerRank`); keep that
distinction explicit when tuning.

## Dispatch model

`runLiteAllGather` handles multi-node layouts before NCCL fallback:
[`lite_allgather.cu:L2315-L2375`](../nccl/lite_allgather.cu#L2315-L2375).
The current dispatch is:

1. If the layout is two-node and below the topology-specific small cutoff, try
   `runSmallOrdered`, then `runSmallFallback`.
2. Otherwise, build the local NIC-group layout.
3. If there is more than one NIC group and this is not `2nx2g`, run
   `runNumaSplit`.
4. Otherwise, run `runSingleSlab`.

Small cutoffs are topology classes, not one-off message-size hacks:
[`lite_allgather.cu:L1264-L1268`](../nccl/lite_allgather.cu#L1264-L1268).

| Topology | Small path cutoff, total output bytes | Medium/large path |
| --- | ---: | --- |
| `2nx1g` | `<2MiB` | `runSingleSlab`, with a 512KiB chunk pipeline for per-rank `1MiB-16MiB` |
| `2nx2g` | `<128KiB` | `runSingleSlab` using 512KiB chunks |
| `2nx4g` | `<128KiB` | `runNumaSplit` across NUMA/NIC-local groups |

Single-node AllGather is still selected by the collective algorithm layer; this
document covers the multi-node NCCL-adapter path.

## Small ordered slot path

Two-node small messages use `runSmallOrdered`:
[`lite_allgather.cu:L1878-L2104`](../nccl/lite_allgather.cu#L1878-L2104).
Each epoch owns a ring-buffered host slot laid out in final AllGather output
order:

```text
[rank0 bytes][rank1 bytes]...[rankN bytes][ready flag]
```

The common flow is:

1. Each local rank places its input at the global-rank offset in the local host
   slot.
2. The node leader waits for local `d2hReady` flags.
3. The leader posts direct-QP RDMA writes for the local node block plus the
   ready flag into the remote node's matching slot.
4. Each rank waits for the remote ready flag.
5. Each rank finishes with either one H2D of the complete output slot or a
   direct self-copy plus H2D of the remote part.
6. Slot wrap synchronizes the stream, polls signaled QP completions, and uses a
   bootstrap barrier before reuse.

The small path has topology-specific subpaths:

- `2nx1g` below 64KiB total output can use mapped-host register-copy kernels to
  pack the local segment and poll/copy the remote segment without an additional
  CPU-side H2D launch.
- `2nx2g` up to 4KiB total output can use GPU pack and receive kernels around
  the same ordered host slot; larger small rows use the host slot with D2H/H2D.
- `2nx4g` uses the ordered host slot to avoid per-peer P2P sends and CPU repack.

The direct-QP helpers for ordered slots and pipelined chunks are in
[`lite_allgather.cu:L640-L704`](../nccl/lite_allgather.cu#L640-L704).

## Single-slab path

`runSingleSlab` is the default multi-node host-slab algorithm for one NIC group:
[`lite_allgather.cu:L1637-L1710`](../nccl/lite_allgather.cu#L1637-L1710).
It uses separate D2H and H2D streams when available:

1. Copy the local rank chunk into the group send slab.
2. Wait for local ranks in the group.
3. The group leader writes the contiguous group block to each peer node's receive
   slab, then writes a remote-ready control flag.
4. Ranks wait for peer ready flags.
5. Ranks copy local and remote slabs into final `recvbuff` order.
6. Ring-slot reuse is protected by either per-slot H2D events or an explicit
   remote ack when only one slot fits.

Large transfers are chunked by `pipelineChunkBytes` and slab capacity:
[`lite_allgather.cu:L1258-L1277`](../nccl/lite_allgather.cu#L1258-L1277).
The `2nx2g` single-slab path uses 512KiB chunks; the default path uses 2MiB
chunks.

For `2nx1g`, per-rank `1MiB-16MiB` uses `runOneRankChunkPipeline`:
[`lite_allgather.cu:L1533-L1635`](../nccl/lite_allgather.cu#L1533-L1635).
This pre-copies the self part when needed, posts each 512KiB D2H chunk, writes
the chunk plus a per-chunk ready value to the remote node, and H2Ds remote chunks
as soon as their ready values arrive.  In `nccl-tests` total-output terms this
pipeline covers `2MiB-32MiB` on `2nx1g`.

## NUMA-split 2nx4g path

`runNumaSplit` is used when the local GPU/NIC layout exposes more than one
contiguous NIC group and `nRanksPerNode != 2`:
[`lite_allgather.cu:L2106-L2313`](../nccl/lite_allgather.cu#L2106-L2313).
On L40/L41 `2nx4g`, this creates two groups:

```text
local ranks 0,1 -> NUMA0/NIC0 group
local ranks 2,3 -> NUMA1/NIC1 group
```

Each group exchanges only its contiguous rank block with the matching remote
group.  After remote-ready flags arrive, each rank assembles all local and
remote group blocks into final AllGather order.  This turns the inter-node
payload into large contiguous host-memory writes and uses both NICs instead of a
single node leader.

## Why this can beat NCCL no-GDR

NCCL no-GDR must also stage through host memory on this hardware.  Lite
AllGather tries to win by reducing avoidable orchestration:

- ordered slots avoid generic per-peer send/recv scheduling for small two-node
  messages;
- direct-QP writes remove part of the `Connection::write` overhead on hot paths;
- one-rank chunking overlaps D2H readiness, host RDMA, and remote H2D;
- medium/large `2nx4g` traffic is split into NUMA/NIC-local slabs.

The design still keeps all inter-node payload movement host-staged.  The
smallest `2nx1g`/`2nx2g` paths may use tiny CUDA kernels for register-copy or
mapped-host receive work, but GPUDirect RDMA is disabled by default
(`kEnableOneRankGpuDirect = false`).

## Latest benchmark status

Latest full rerun:
`experimental/lite/lite-collective/.tmp/collective-benchmarks/ag-rerun-it50-w20-128B-1G-20260608-100509/`.

Command shape:

```bash
scripts/run-nccl-tests.sh --backend {mscclpp,nccl} \
  --test all_gather --topology inter \
  --hosts 10.10.55.1,10.10.55.2 \
  --gpus <topology-gpus> \
  --min-bytes 128 --max-bytes 1G \
  --iters 50 --warmup-iters 20 -- -c 1
```

Environment:

```text
NCCL_SOCKET_IFNAME=ibp55s0f0
NCCL_IB_HCA=mlx5_0,mlx5_1
NCCL_NET_GDR_LEVEL=0
MSCCLPP_SOCKET_IFNAME=ibp55s0f0
MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1
```

Out-of-place summary from the latest rerun:

| Topology | Correctness | Latency wins, 128B-1MiB | Busbw wins, 1MiB-1GiB | Remaining misses |
| --- | --- | ---: | ---: | --- |
| `2nx1g` | `#wrong=0` | 12/14 | 11/11 | 128B, 64KiB latency |
| `2nx2g` | `#wrong=0` | 13/14 | 11/11 | 8KiB latency |
| `2nx4g` | `#wrong=0` | 13/14 | 11/11 | 16KiB latency |

Representative rows:

| Topology | Size | Metric | Lite | NCCL | Result |
| --- | ---: | --- | ---: | ---: | --- |
| `2nx1g` | 1MiB | latency | 72.74 us | 96.96 us | Lite faster |
| `2nx1g` | 1GiB | busbw | 13.17 GB/s | 12.45 GB/s | Lite faster |
| `2nx2g` | 1MiB | latency | 84.14 us | 89.03 us | Lite faster |
| `2nx2g` | 1GiB | busbw | 20.76 GB/s | 15.42 GB/s | Lite faster |
| `2nx4g` | 1MiB | latency | 78.82 us | 118.33 us | Lite faster |
| `2nx4g` | 1GiB | busbw | 18.73 GB/s | 14.89 GB/s | Lite faster |

The report generated from raw logs is in
`.tmp/collective-benchmarks/ag-rerun-it50-w20-128B-1G-20260608-100509/allgather-results.md`.

## Limitations and next work

- The performance gate is not fully closed: each topology still has one or two
  latency misses below 1MiB.
- The L40/L41 testbed only validates two nodes.  The host-slab protocol supports
  multi-node layouts, but `>2` nodes need separate benchmarking.
- `2nx2g` intentionally stays on the single-slab path for medium/large messages;
  NUMA splitting is skipped because GPU0/1 are on the same NUMA locality on the
  target setup.
- Avoid adding branches that only optimize one exact message size.  Future work
  should improve a size class or topology class with a reusable algorithmic
  reason.
