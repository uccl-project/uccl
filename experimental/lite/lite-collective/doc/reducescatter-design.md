# Lite-collective ReduceScatter design

## At a glance

ReduceScatter is split by topology: the `1nx4g` implementation lives in
[`nccl/ReduceScatter/single-node.cu`](../nccl/ReduceScatter/single-node.cu),
while inter-node paths live in
[`nccl/ReduceScatter/multi-node.cu`](../nccl/ReduceScatter/multi-node.cu).
The normal `1nx4g`, `2nx1g`, `2nx2g`, and `2nx4g`, `ncclFloat32`, `ncclSum`
paths are native-only across the benchmark size range; they do not fall back to
real NCCL for small or large rows.  Forced NCCL fallback via the environment
remains a debugging escape hatch.

The current 2nx4g implementation is correct across 128B-1GiB, but it is not yet
a full performance win over a fair tuned NCCL baseline where both libraries may
use local PCIe P2P and neither may use GDR (`NCCL_P2P_DISABLE=0`,
`NCCL_P2P_LEVEL=SYS`, `NCCL_NET_GDR_LEVEL=0`).  It uses a native host-slot path
for tiny rows and a GPU NUMA-pair path with chunk pipelining for larger rows.
The latest 1MiB-16MiB retune disables local IPC event synchronization on the
small single-chunk rows and extends direct partner-row CudaIpc copies through the
mid-size rows.  It makes 2nx4g win at 1MiB, 2MiB, 8MiB, and 16MiB in the current
no-GDR benchmark; 4MiB remains a small tuned-NCCL win.
The 1nx4g large-message path uses a native CudaIpc ring.  Rows whose output
shard exceeds the tuned one-shot chunk size are split into 16MiB chunks,
so the 128MiB-1GiB total-size rows stay on the native path instead of falling
through to real NCCL.

The 2nx1g path is simpler and now wins from 64KiB through 1MiB in the latency
table and almost all of the 1MiB-1GiB bus-bandwidth table.  The remaining 2nx1g
gaps are 128B-32KiB latency and the 2MiB bus-bandwidth row.

## Dispatch

For supported single-node and inter-node communicators, `ncclReduceScatter` first
tries `runLiteInterReduceScatter`.  For multi-node ReduceScatter, a native rejection is
returned to the caller instead of silently falling back to NCCL; this keeps
benchmark results honest while the native algorithm is being optimized.
The implementation accepts only:

- `nRanks == 2` and `nRanksPerNode == 1`;
- `nRanks == 4` and `nRanksPerNode == 4`;
- `nRanks == 4` and `nRanksPerNode == 2`;
- `nRanks == 8` and `nRanksPerNode == 4`;
- `datatype == ncclFloat32`;
- `op == ncclSum`.

Unsupported cases return `ncclInvalidUsage`/`ncclInvalidArgument`.

## Native 1nx4g flow

For four ranks on one node, large rows use the same reduce-scatter ring shape as
NCCL: each rank sends one shard to its next local peer, receives one shard from
its previous local peer, adds the matching local shard, and forwards the partial
for three ring steps.  The ring uses CudaIpc scratch, not host staging or real
NCCL.

The communicator scratch is fixed-size, so rows above the tuned one-shot chunk
size are processed as 16MiB output-shard chunks.  The chunk size can be tuned
with `MSCCLPP_NCCL_RS_LOCAL_RING_CHUNK_BYTES`; the parser accepts raw bytes or
`K`/`M`/`G` suffixes.

## Native 2nx1g flow

For two ranks total, each rank owns one output shard and has only one remote
peer:

```text
local shard       = sendbuff[this rank's shard]
remote-owned send = sendbuff[peer rank's shard]
```

Through 1MiB per rank, the native path uses a mapped-host fast path when host
mapping succeeds: a GPU kernel writes only the peer-owned shard into a mapped
host slot, RDMA writes that slot to the peer, and a GPU kernel reads the incoming
mapped-host shard and adds it to the local shard in `sendbuff`.  This is safe
for in-place ReduceScatter because the incoming contribution is never copied over
the local shard before the add kernel reads it.  Larger rows use the general
D2H/RDMA/H2D GPU-add pipeline.

## Native 2nx4g flow

For each local rank, the native path computes two local-node partials on GPU:

```text
localPartial  = sum(local node ranks' shard for this global rank)
remotePartial = sum(local node ranks' shard for matching remote global rank)
```

The local GPU phase uses the same NUMA-pair shape as the previous prototype:

1. Pack only the rows needed by the NUMA-pair partner.
2. CudaIpc-copy partner rows into the partner GPU's scratch.
3. Reduce within the NUMA pair, reading this rank's own rows directly from
   `sendbuff` instead of first staging them through scratch.
4. CudaIpc-copy one cross-pair partial.
5. Produce `localPartial` and `remotePartial`.
6. D2H `remotePartial` into a pinned host slot.
7. RDMA-write that slot to the matching rank on the remote node.
8. H2D the incoming remote partial directly into `recvbuff` and add the local
   partial in place.

The implementation uses a per-node shared control block for local CudaIpc
readiness and per-rank RDMA ready/ack epochs for host-slot reuse.

## No-CudaIPC mode

`MSCCLPP_NCCL_RS_NO_CUDAIPC=1` forces ReduceScatter through host SHM/RDMA for
layouts with more than one local rank.  This mode does not call the local
CudaIpc scratch setup and does not open peer GPU pointers or IPC events.

For single-node `1nx4g`, tiny shards use the CPU host path because it has the
lowest fixed latency: each rank D2Hs its rows into a per-rank pinned host slab,
waits for local ranks through the shared CPU control block, CPU-reduces the
target shard, and H2Ds the final shard.  The cutoff defaults to 64KiB per rank
and can be changed with `MSCCLPP_NCCL_RS_NO_CUDAIPC_HOST_SMALL_BYTES`.

For mid-size `1nx4g` shards, the default host-read bulk path D2Hs only the
three shards owned by the other local ranks into source-local mapped SHM, waits
for local availability, then a GPU reduction kernel reads the remote rows from
mapped host memory and adds the local shard directly from `sendbuff`.  This
avoids the old CPU reduction bottleneck and the three H2D copies from the first
bulk-DMA design.  Its chunk size defaults to 2MiB and can be changed with
`MSCCLPP_NCCL_RS_NO_CUDAIPC_BULK_CHUNK_BYTES`.  Setting
`MSCCLPP_NCCL_RS_NO_CUDAIPC_HOST_READ=0` falls back to the source-local
bulk-DMA scratch path.

For larger `1nx4g` shards, the default path switches to a direct mapped-SHM
ring.  Each rank sends the reduce-scatter ring shards through its mapped SHM
mailbox with GPU copy/add kernels and GPU-visible control flags.  Stream memory
operations are used for the ring flag wait/store path by default
(`MSCCLPP_NCCL_RS_NO_CUDAIPC_DIRECT_RING_STREAM_MEMOPS=0` restores the kernel
flag path).  This avoids the all-to-all host-read barrier and raises
large-message bus bandwidth on the L4 testbed from about 12.3GB/s to about
13.6GB/s, but it still trails NCCL's forced-SHM direct protocol.  NCCL's
forced-SHM ReduceScatter uses two `RING/SIMPLE` channels; forcing NCCL down to
one channel drops it to about 13.4GB/s, matching this Lite single-channel ring.
The switch point defaults to 1MiB per receive rank
(`MSCCLPP_NCCL_RS_NO_CUDAIPC_DIRECT_RING_MIN_BYTES`) and the ring chunk size
defaults to 16MiB (`MSCCLPP_NCCL_RS_NO_CUDAIPC_DIRECT_RING_CHUNK_BYTES`).
`MSCCLPP_NCCL_RS_NO_CUDAIPC_DIRECT_RING=0` disables this large-message ring and
keeps the host-read path.

Multi-node no-CudaIPC layouts continue to use the host/RDMA CPU-reduce path.
Its chunk size defaults to 256KiB and can be changed with
`MSCCLPP_NCCL_RS_NO_CUDAIPC_CHUNK_BYTES`.  CPU reductions use an AVX-512F helper
when available, with scalar fallback; `MSCCLPP_NCCL_RS_DISABLE_AVX512=1` forces
the scalar path for debugging.

Use `NCCL_P2P_DISABLE=1 NCCL_NET_DISABLE=1` for the matching single-node NCCL
baseline so NCCL also cannot use direct local GPU P2P or network transport.  On
the current L40/L41 testbed this native no-CudaIPC path is a
functional/reference path rather than the fastest path; the normal CudaIpc path
remains the performance path.

Rejected no-CudaIPC alternatives include an all-to-all direct mapped-SHM GPU
kernel, row-parallel D2H streams, copy/add unrolling, chunk-level multistream
launches, DMA-ring staging, receiver-local mailboxes, same- and
opposite-direction two-channel direct rings, fused wait+add kernels, a
cooperative single-kernel ring, and a full host CPU-reduce path for large rows.
All were correctness-clean but slower than the retained hybrid: the all-to-all
direct and row-parallel variants stayed near the same ~12.2GB/s mapped-host
bandwidth ceiling, naive channel splits added launch/control overhead without
reproducing NCCL's persistent FIFO/proxy structure, fused wait+add over-polls
mapped control memory from many blocks, the cooperative ring loses too much
parallel memory bandwidth, and the CPU-reduce variant was substantially slower.

## Size policy

The 2nx1g path uses the mapped-host GPU path through 1MiB per rank.  This beats
the older tiny CPU-final path across the retained small and mid rows, but the
2MiB total-size row still trails NCCL.  Lowering the cutoff to send that row
through the general path improves the isolated row but exposes a size-transition
hang in mixed-size runs, so the stable cutoff remains 1MiB per rank.  Larger
rows use a five-slot host/scratch pipeline with deferred H2D ack; 2MiB chunks
are the best stable setting across 4MiB-1GiB on L40/L41.

The `2nx2g` path uses the shared host-slot path below 128KiB total message size
before entering the two-node/two-GPU hierarchy.  This keeps the low fixed-latency
host path for tiny rows while sending 128/256KiB through the hierarchy, which is
faster there.  The hierarchy still uses the round-robin-HCA policy for small rows
when multiple IB transports are available, but the available-transport count is
cached so every timed collective does not pay an IB-device enumeration cost.
Without that cache, 2nx2g small-message latency was dominated by the policy probe
rather than the ReduceScatter data path.

The current native-only implementation has two internal regimes on `2nx4g`:

| Regime | Policy |
| --- | --- |
| `<512KiB` total size | Shared host-slot path: each rank D2Hs its input into a per-node shm slab, computes two local partials on CPU, RDMA-writes the remote-owned partial, and H2Ds the final result. Each rank's shm slice is placed on the GPU's NUMA node, final CPU output stays in the recv slab rather than overwriting the input slab, the local/final CPU reductions use a runtime AVX-512F fast path with scalar fallback, and the H2D ack is deferred to the next safe stream-completion point. |
| `>=512KiB` total size | GPU-local NUMA-pair path: compute local/remote partials in GPU scratch, D2H or mapped-host-write the remote partial, exchange it with the matching remote rank, H2D or mapped-host-read the incoming remote partial, and add the local partial in place. Single-chunk rows avoid the async side-stream pipeline; multi-chunk rows use four compact scratch slots, deferred H2D ack, tiered chunk sizes, and a deeper local lead for long rows.  On 2nx4g, 1MiB and larger rows use direct partner-row CudaIpc copies instead of pack-then-copy. |

The previous implementation produced correct wins only around 512KiB by falling
back elsewhere.  The current code removes that fallback, so performance gaps are
visible instead of hidden.

## RDMA control-write optimization

The main GPU path bypasses `Connection::write` for the data+ready pair on each
chunk.  It stages the data RDMA write unsignaled, stages the 8-byte ready flag
as an inline write, posts both WRs with one doorbell, and polls only
periodically.  The same inline signal helper is used for the ack path.  This
removes one signaled CQE and one doorbell from the steady chunk path and is most
visible around 1MiB-16MiB.

## Mapped-host finalization and split-final reduce

For 2nx4g rows up to 1MiB per rank, the GPU-local path can write the
remote-node partial directly into the mapped, RDMA-registered host send slot
instead of first writing GPU scratch and then launching a D2H copy.  Single-chunk
final add can also read the incoming RDMA slot through the mapped host pointer
instead of launching an H2D copy.  The policy is controlled by
`MSCCLPP_NCCL_RS_MAPPED_SEND_FINAL_REDUCE=0/1` and
`MSCCLPP_NCCL_RS_HOST_READ_FINAL_ADD=0/1`.

For async rows up to 1MiB per rank, split-final reduce launches a remote-only
final-reduce kernel first, starts host/RDMA progress from that remote partial,
then launches a local-only final-reduce kernel.  Scratch-slot reuse is delayed
until the final add finishes because the host-read variant keeps the incoming
host slot live until the add kernel has consumed it.  This is controlled by
`MSCCLPP_NCCL_RS_SPLIT_FINAL_REDUCE=0/1`.

## Local IPC event synchronization

The 2nx4g GPU-local path records CUDA interprocess events after the partner and
cross-pair CudaIpc copies, then makes the consumer stream wait on the peer's
event.  A CPU epoch handshake is published immediately after each event record
is enqueued and before the peer waits on the event; this avoids reusing an event
record before the peer has observed the right epoch.
`MSCCLPP_NCCL_RS_IPC_EVENT_SYNC=0/1` can still force the mode off or on.  Event
sync avoids host-side stream drains, but the 2nx4g 1MiB-4MiB single-chunk rows
are faster with CPU epoch synchronization on the L40/L41 no-GDR testbed.

## Large-message final-add overlap

For 2nx4g rows with `bytesPerRank >= 32MiB` (256MiB total size), the final H2D
of the remote-node partial and the GPU add now stay on the H2D side stream.
The main user stream waits only on the last in-flight slot events before the
collective returns.  This keeps local pack/reduce work on the main stream from
being blocked by each chunk's final add while still protecting scratch-slot
reuse with `slotDoneEvents`.  `MSCCLPP_NCCL_RS_ASYNC_FINAL_ADD=0/1` can force
the mode off or on.  On L40/L41 this is neutral below the threshold and improves
the 256MiB-1GiB steady-state rows slightly.

## Rejected experiment: direct peer-input kernel loads

One attempted redesign registered each local rank's `sendbuff` with CudaIpc and
used a single kernel to read all four local ranks' inputs directly, avoiding
the staged NUMA-pair copies.  It was correct but much slower on L4 PCIe:
roughly `0.2-0.8GB/s` bus bandwidth at 4MiB-16MiB total sizes.  Do not revive
that path without fresh hardware evidence.

## Rejected experiment: full host-reduce for mid-size rows

The CPU host-reduce idea was tested for 2nx4g:

1. D2H each GPU's input to pinned host memory local to the GPU NUMA node.
2. CPU-reduce the local node's four partials.
3. RDMA-write only the local reduced partial.
4. CPU-final-reduce with the remote partial.
5. H2D only the final shard.

This is the current tiny-row strategy, but extending it through 64KiB-256KiB
did not beat NCCL.  A two-stage NUMA-pair CPU reduce variant was also tested;
after fixing a cross-partial/RDMA slot aliasing bug it was correct but much
slower.  For 64KiB and above, the GPU NUMA-pair path remains the better native
choice on the L40/L41 PCIe testbed.

The same host strategy was also spot-tested at 2MiB/4MiB total sizes after the
AVX-512 path was added.  It completed correctly but was much slower
(approximately 265us/506us), so it is not used to cover the 4MiB/8MiB gap.

## CPU SIMD note

The L40/L41 hosts are Intel Xeon Silver 4410Y CPUs with AVX-512F.  The tiny
host path uses GCC vector extensions in an `avx512f`-targeted helper so the
native binary does not need global AVX-512 compiler flags and still keeps a
scalar fallback for hosts without AVX-512F.  This improves the CPU-reduce-heavy
64KiB-256KiB total-size rows, but it does not address the large-message GPU /
host-staging pipeline bottleneck.
