# Lite-collective AllGather design

## At a glance

Lite AllGather is the native NCCL-shim AllGather implementation for PCIe-only,
no-GPUDirect-RDMA testbeds.  Inter-node payloads are always staged through
pinned host memory; the performance work is about making that staging explicit,
coalesced, and topology-aware instead of using the generic grouped P2P
send/recv path.

| Topic | Summary |
| --- | --- |
| Entry points | `ncclAllGather` first tries the single-node CudaIpc ring in [`nccl.cu`](../nccl/nccl.cu#L2499), then `runLiteAllGather` in [`lite_allgather.cu`](../nccl/lite_allgather.cu#L2328) handles multi-node host-slab layouts |
| Target layouts | `1nx4g` large-message single-node CudaIpc ring; `2nx1g`, `2nx2g`, `2nx4g`, plus a general multi-node host-slab fallback |
| `1nx4g` large messages | CudaIpc ring for total output `>=8MiB`; smaller messages fall back to real NCCL |
| Small messages | Two-node ordered host slots with RDMA-written ready flags |
| `2nx1g` large messages | Dedicated 512KiB chunk pipeline through 1GiB per-rank input |
| `2nx2g` large messages | Single host slab, 512KiB chunks |
| `2nx4g` large messages | NUMA/NIC split host slabs when symmetric NIC groups exist |
| Remote data-ready signal | RDMA atomic update to the peer `rdmaReady` epoch counter |
| GDR status | Disabled by default (`kEnableOneRankGpuDirect = false`) |

## Size terminology

Keep these two sizes separate when reading code or benchmark logs:

| Name | Meaning | Example on `2nx1g` |
| --- | --- | --- |
| `bytesPerRank` | Input bytes contributed by one rank; most code thresholds use this. | `512MiB` when the nccl-tests row says `1GiB` |
| `fullBytes` / nccl-tests `size` | Total AllGather output, `bytesPerRank * nRanks`. | `1GiB` for two ranks each contributing `512MiB` |

This distinction matters because the `2nx1g` fast path is capped by
`bytesPerRank`, not by the nccl-tests output-size column.

## Dispatch model

The NCCL shim has a single-node fast path before the multi-node
`runLiteAllGather` path:

1. `ncclAllGather` handles trivial `worldSize == 1` and forced NCCL fallback.
2. If all ranks are on one node (`nRank == nRanksPerNode`), try
   `runIntraNodeCudaIpcAllGather`.
3. The single-node path only accepts large messages (`fullBytes >= 8MiB`),
   CUDA IPC event synchronization, non-capturing streams, and a valid
   CudaIpc-mappable `recvbuff` allocation.  Unsupported cases return
   `ncclInvalidUsage`, so the wrapper falls back to real NCCL when the dlopen
   fallback is configured.
4. Multi-node layouts continue through `runLiteAllGather`.

`runLiteAllGather` handles multi-node layouts before the final NCCL fallback.
The multi-node order is intentionally short:

1. Reject unsupported or invalid multi-node layouts.
2. For two-node messages below the topology-specific small cutoff, try
   `runSmallOrdered`, then `runSmallFallback`.
3. Build the local NIC-group layout.
4. If the layout has multiple NIC groups and is not `2nx2g`, run
   `runNumaSplit`.
5. Otherwise run `runSingleSlab`; `2nx1g` large-message pipeline selection
   happens inside this path.

| Topology | Small cutoff, total output bytes | Medium/large path |
| --- | ---: | --- |
| `1nx4g` | `<8MiB` uses real NCCL fallback | `runIntraNodeCudaIpcAllGather` CudaIpc ring |
| `2nx1g` | `<2MiB` | `runOneRankChunkPipeline` for per-rank `1MiB-1GiB`; then generic single slab |
| `2nx2g` | `<128KiB` | `runSingleSlab` with 512KiB chunks |
| `2nx4g` | `<128KiB` | `runNumaSplit` across NUMA/NIC-local groups when available |
| `>2` nodes | topology-dependent | generic host-slab path; benchmark separately |

Code pointers:

| Area | Code |
| --- | --- |
| Single-node CudaIpc ring threshold | [`nccl.cu:L62`](../nccl/nccl.cu#L62) |
| Single-node CudaIpc ring implementation | [`nccl.cu:L1312-L1500`](../nccl/nccl.cu#L1312-L1500) |
| `ncclAllGather` dispatch and fallback order | [`nccl.cu:L2499-L2601`](../nccl/nccl.cu#L2499-L2601) |
| Constants and thresholds | [`lite_allgather.cu:L45-L69`](../nccl/lite_allgather.cu#L45-L69) |
| Remote data-ready atomic signal | [`lite_allgather.cu:L619-L631`](../nccl/lite_allgather.cu#L619-L631) |
| Context setup and 2nx1g slab sizing | [`lite_allgather.cu:L786-L852`](../nccl/lite_allgather.cu#L786-L852) |
| Small cutoffs and slot counts | [`lite_allgather.cu:L1289-L1302`](../nccl/lite_allgather.cu#L1289-L1302) |
| `2nx1g` chunk pipeline | [`lite_allgather.cu:L1570-L1685`](../nccl/lite_allgather.cu#L1570-L1685) |
| Single-slab path | [`lite_allgather.cu:L1687-L1766`](../nccl/lite_allgather.cu#L1687-L1766) |
| Small ordered path | [`lite_allgather.cu:L1928-L2156`](../nccl/lite_allgather.cu#L1928-L2156) |
| NUMA split path | [`lite_allgather.cu:L2158-L2361`](../nccl/lite_allgather.cu#L2158-L2361) |
| Top-level dispatch | [`lite_allgather.cu:L2364-L2434`](../nccl/lite_allgather.cu#L2364-L2434) |

## Protocols

### 1. Single-node CudaIpc ring (`1nx4g` large messages)

The retained `1nx4g` path is a single-node ring over CudaIpc mapped receive
buffers.  It is tuned for large messages and is intentionally not used below
`8MiB` total output, where real NCCL is already competitive and has lower setup
risk.

Preconditions:

- all ranks are local (`nRank == nRanksPerNode`);
- total AllGather output is at least `8MiB`;
- CUDA IPC event synchronization is enabled;
- the caller stream is not under CUDA graph capture;
- every rank's `recvbuff` is in a single CUDA allocation that can be mapped by
  peer ranks.

Logical layout:

```text
recvbuff on every rank:
[rank0 bytes][rank1 bytes][rank2 bytes][rank3 bytes]

ring direction:
rank0 -> rank1 -> rank2 -> rank3 -> rank0
```

For rank `r`, `next = (r + 1) % nRanks` and
`prev = (r + nRanks - 1) % nRanks`.

Flow:

1. Each rank publishes its `recvbuff` address and, when the allocation changes,
   exchanges a CudaIpc memory handle for the whole receive allocation.  The cache
   is keyed by CUDA allocation base, size, and `CU_POINTER_ATTRIBUTE_BUFFER_ID`
   so address reuse after `cudaFree`/`cudaMalloc` cannot reuse a stale mapping.
2. Each receiver records a CudaIpc ready event on the caller stream.  The writer
   waits on the next rank's ready event before writing into that rank's mapped
   receive buffer, so remote writes cannot race earlier work on the receiver's
   stream.
3. If `sendbuff` is not already the local slot in `recvbuff`, the rank copies its
   own input into `recvbuff[rank]`.
4. The ring runs for `nRanks - 1` steps.  At step `s`, rank `r` sends block
   `(r - s + nRanks) % nRanks` to `next` with one `cudaMemcpyAsync` on the
   per-peer IPC stream:

   ```text
   step 0: send own block
   step 1: forward the block just received from prev
   step 2: forward the next received block
   ...
   ```

   The destination address is the peer's mapped `recvbuff[sendBlock]` slot, so
   no receiver-side unpacking is needed.
5. After each outbound copy, the writer records a dedicated AllGather done event
   and signals a shared-memory generation counter.  The receiver waits on the
   predecessor's done event before it forwards that block or returns to the
   caller.
6. Ready/done generation acknowledgements prevent re-recording shared IPC events
   before the peer has enqueued its wait.  This is required because the same
   per-peer interprocess events are reused across AllGather calls.
7. Before returning, the caller stream waits for the outbound IPC stream, keeping
   normal NCCL stream-ordering semantics.

The path is not chunk-pipelined inside a single rank block: each ring step moves
one `bytesPerRank` block.  Pipelining happens at ring-step granularity: all ranks
copy to their `next` peer concurrently, then each rank forwards the block it just
received from `prev`.

### 2. Small ordered host slots

Small two-node messages use `runSmallOrdered`.  Each epoch gets a ring-buffered
host slot laid out in final AllGather order:

```text
[rank0 bytes][rank1 bytes]...[rankN bytes][ready flag]
```

Flow:

1. Each rank copies its input into the global-rank offset of the local host slot.
2. The node leader waits for local `d2hReady` flags.
3. The leader RDMA-writes the local node block and ready flag into the remote
   node's matching slot.
4. Each rank waits for the remote ready flag.
5. Each rank copies the final ordered slot to `recvbuff`, or uses a self-copy
   fast path plus H2D for the remote part.
6. Slot wrap synchronizes before reuse.

Topology-specific small-message shortcuts:

- `2nx1g` below 64KiB total output can use mapped-host register-copy kernels.
- `2nx2g` up to 4KiB total output can use GPU pack/receive kernels around the
  same ordered slot.
- `2nx4g` uses ordered host slots to avoid per-peer P2P scheduling and CPU
  repacking.

### 3. `2nx1g` one-rank chunk pipeline

For `2nx1g`, per-rank `1MiB-1GiB` uses `runOneRankChunkPipeline`.  In nccl-tests
output-size terms this covers up to `2GiB`; the retained validation below covers
the requested `32MiB-1GiB` output range.

The path is intentionally different from generic single-slab exchange:

1. Allocate a full-message staging slot for the one local rank.  The current
   cap is `kOneRankMaxBytesPerRank = 1GiB`.
2. Split the per-rank input into 512KiB chunks.
3. Record one D2H event per chunk after copying it into the send slab.
4. Send chunks in a one-chunk window: wait for D2H readiness, RDMA-write the
   data, then RDMA-write a per-chunk ready value.
5. The receiver waits for each ready value and immediately H2Ds that remote
   chunk into final output order.

Two correctness details are easy to miss:

- `kPipeValueStride` is derived from the maximum chunk count so ready values do
  not overlap across epochs.
- Each chunk's ready value is stored in its own stable slot in the registered
  send slab.  Do not reuse one host control word for all flag writes: the NIC may
  DMA-read the source after later chunks have overwritten it.

This path fixes the previous `2nx1g` large-message cliff where messages above
the old 16MiB per-rank fast-path limit fell into the slower generic single-slab
steady state.

### 4. Generic single-slab path

`runSingleSlab` is the default host-slab algorithm when all local ranks share one
NIC group, or when no better topology-specific path applies.

Flow:

1. Copy the local rank chunk into a group send slab.
2. Wait for all local ranks in the group.
3. The group leader writes the contiguous local group block to each peer node's
   receive slab, then publishes the remote-ready epoch with an RDMA atomic
   update.
4. Ranks wait for peer ready flags.
5. Ranks copy local and remote slab blocks into final `recvbuff` order.
6. Slot reuse is protected by per-slot H2D events or by explicit remote acks when
   only one slot fits.

Chunk sizing:

- `2nx2g` uses 512KiB chunks.
- The generic default uses 2MiB chunks.
- Messages beyond the `2nx1g` one-rank fast path intentionally keep the old
  16MiB ring capacity for the generic fallback, rather than silently expanding
  its reuse window.

The `rdmaReady` control word is intentionally updated with an RDMA atomic, not a
plain RDMA write.  The atomic is posted after the data writes on the same
connection and then flushed; this gives the receiver a remote-visible barrier
before it copies from the host receive slab.  Because not every AllGather epoch
uses `rdmaReady` (for example, small ordered slots have their own slot flag), the
sender tracks the last published epoch per peer and atomically adds the delta so
the remote `rdmaReady[node]` counter reaches the exact epoch being waited on.

### 5. NUMA/NIC split path

`runNumaSplit` is used when the local GPU/NIC layout exposes multiple symmetric
contiguous groups and `nRanksPerNode != 2`.  On the L40/L41 `2nx4g` setup, that
usually means:

```text
local ranks 0,1 -> NUMA0/NIC0 group
local ranks 2,3 -> NUMA1/NIC1 group
```

Each group exchanges only its contiguous rank block with the matching remote
group.  After the peer group's atomic `rdmaReady` update arrives, each rank
assembles all local and remote group blocks into final AllGather order.  This
uses both NICs and avoids a single node leader pushing all inter-node traffic.

## Why this beats NCCL no-GDR

NCCL no-GDR also stages inter-node traffic through host memory on this hardware.
Lite wins when it makes that staging cheaper:

- `1nx4g` large messages bypass host staging entirely and use a direct CudaIpc
  ring; each rank writes directly into final peer receive slots, with only
  ready/done event synchronization around the peer copies;
- small messages avoid generic grouped-send orchestration;
- hot paths use direct-QP RDMA writes for data and an RDMA atomic barrier for
  remote data-ready signaling;
- `2nx1g` overlaps D2H readiness, RDMA writes, and remote H2D at 512KiB
  granularity;
- `2nx4g` splits traffic across NUMA/NIC-local slabs.

The design still keeps all inter-node payload movement host-staged.  Tiny CUDA
kernels are used only for small local packing/copying helpers; GPUDirect RDMA is
not part of the default payload path.

## Latest benchmark status

Benchmark convention:

- `iters=50`, `warmup=20`
- compare latency for `128B-1MiB`
- compare bus bandwidth for `1MiB-1GiB`
- report one direction when out-of-place and in-place are similar

Environment used for the latest full sweep:

```text
NCCL_SOCKET_IFNAME=ibp55s0f0
NCCL_IB_HCA=mlx5_0,mlx5_1
NCCL_NET_GDR_LEVEL=0
NCCL_BUFFSIZE=4194304
MSCCLPP_SOCKET_IFNAME=ibp55s0f0
MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1
```

Command shape:

```bash
scripts/run-nccl-tests.sh --test all_gather \
  --backend {mscclpp,nccl} --topology inter \
  --hosts 10.10.55.1,10.10.55.2 \
  --gpus {0|0,1|0,1,2,3} \
  --min-bytes 128 --max-bytes 1G \
  --step-factor 2 --iters 50 --warmup-iters 20
```

Latest multi-node full comparison: [`perf-allgather.md`](perf-allgather.md).
The summary below lists out-of-place bus bandwidth at the 1GiB row:

| Topology | Lite GB/s | NCCL GB/s | Lite/NCCL |
| --- | ---: | ---: | ---: |
| `2nx1g` | 17.55 | 12.43 | 1.41x |
| `2nx2g` | 20.55 | 15.26 | 1.35x |
| `2nx4g` | 18.83 | 14.88 | 1.27x |

All rows reported `#wrong=0`.  Raw logs:
`.tmp/collective-benchmarks/ag-perf-allgather-20260608-164008/`.

Latest single-node `1nx4g` sweep, using the same iteration convention and
`--topology intra --gpus 0,1,2,3`, also reported `#wrong=0`.  The native path is
active from `8MiB` upward; smaller rows use the real NCCL fallback.  Out-of-place
bus bandwidth highlights:

| Size | Lite GB/s | NCCL GB/s | Lite/NCCL |
| ---: | ---: | ---: | ---: |
| `8MiB` | 17.78 | 15.71 | 1.13x |
| `32MiB` | 19.72 | 16.29 | 1.21x |
| `1GiB` | 19.95 | 16.46 | 1.21x |

Raw logs:
`.tmp/collective-benchmarks/ag-1nx4g-postfix-20260609-060049/`.

## Operational notes

- The `2nx1g` 1GiB fast path increases pinned/shared host memory for that
  topology.  With two nodes and one local rank, each process maps a 2GiB send
  slab and a 2GiB receive slab.
- Do not reintroduce the generic grouped P2P AllGather branch for multi-node
  AllGather.  The retained design is host-memory AllGather.
- Keep `MSCCLPP_NCCL_LIB_PATH` configured for `mscclpp` benchmark runs.  The
  `1nx4g` path deliberately falls back to real NCCL below `8MiB`, and the
  fallback is also the safety net for unsupported single-node cases.
- Avoid ad-hoc optimizations for one exact message size.  New branches should be
  justified by a topology class or a reusable protocol improvement.
- `>2` node correctness follows the host-slab protocol structure, but the
  current performance evidence is from two-node L40/L41 runs; benchmark larger
  node counts before claiming performance.
