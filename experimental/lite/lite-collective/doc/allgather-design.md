# Lite-collective AllGather design

## Primary design

The multi-node AllGather design is a two-node host-slab exchange.  Because the
testbed has no GPUDirect RDMA, every inter-node transfer must already pass
through CPU memory.  On 2nx4g the implementation splits that host memory by
NUMA domain and NIC:

```text
GPU0/GPU1 sendbuff -> NUMA0 host slab -> NIC0 RDMA -> remote NUMA0 slab
GPU2/GPU3 sendbuff -> NUMA1 host slab -> NIC1 RDMA -> remote NUMA1 slab
each rank H2Ds both NUMA groups into recvbuff
```

For 2nx4g, messages at or above 128KiB keep this dual-NIC path.  Smaller
messages use a pipelined single-leader host-slab protocol with ring-buffered
slots and direct QP writes.  That removes the per-iteration remote ack and most
`Connection::write` overhead, bringing small-message latency close to NCCL.

## Implementation details

The public `ncclAllGather` wrapper first gives the single-node algorithm
collection a chance to run.  If that is not selected or returns unsupported, the
wrapper calls `runSendRecvAllGather` before NCCL dlopen fallback:
[`nccl.cu:L2159-L2235`](../nccl/nccl.cu#L2159-L2235) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2159:1)).

For two-node layouts, `runSendRecvAllGather` dispatches to the host-slab fast
path; otherwise it falls back to a chunked grouped send/recv implementation:
[`native_collectives.cu:L2640-L2712`](../nccl/native_collectives.cu#L2640-L2712) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:2640:1)).

The host context is owned by each node leader or NUMA-group leader.  It creates
shared-memory slabs, maps the leader-owned names into all local ranks, NUMA
places and pins the mappings, registers the slabs with mscclpp, and connects to
the matching remote leader:
[`native_collectives.cu:L513-L760`](../nccl/native_collectives.cu#L513-L760) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:513:1)).

The steady-state data path is:

1. Each local rank copies its input chunk into its NUMA group's `sendSlab`.
2. Each group leader waits for the two local `d2hReady` flags in its group.
3. Each group leader writes its group slab to the matching remote group slab.
4. Every local rank waits for both groups' remote-ready flags.
5. Every local rank copies both local groups and both remote groups into the
   correct positions in `recvbuff`.
6. Local ranks publish `h2dDone`; each group leader sends a remote ack before
   the next chunk reuses the slab.

That fast path is implemented in
[`native_collectives.cu:L2076-L2198`](../nccl/native_collectives.cu#L2076-L2198) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:2076:1)).
Large messages stay on the host-slab path by chunking at the slab capacity:
[`native_collectives.cu:L1835-L1876`](../nccl/native_collectives.cu#L1835-L1876) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1835:1)).

The small-message path is implemented in
[`native_collectives.cu:L1984-L2074`](../nccl/native_collectives.cu#L1984-L2074) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1984:1)).
It uses one node-leader RDMA exchange, appends a ready flag to each ring slot,
polls that flag from the remote host slab, and copies the local and remote node
blocks into `recvbuff` on the CUDA stream.  The direct-QP helper is in
[`native_collectives.cu:L522-L568`](../nccl/native_collectives.cu#L522-L568) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:522:1)).

## Single-node path

Single-node AllGather uses the existing mscclpp fullmesh algorithms.  On the
L40/L41 PCIe target, the selector prefers `default_allgather_fullmesh` for
aligned sizes because it is faster than `fullmesh2` for the 1MiB target:
[`algorithm_selector.cc:L168-L199`](../collective/algorithm_selector.cc#L168-L199) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:168:1)).

## Why it beats NCCL no-GDR

The native path wins because it removes avoidable per-peer network/proxy work
and uses both NICs on 2nx4g.  On this hardware, host memory is unavoidable;
packing local ranks into NUMA-local slabs converts the inter-node phase into
two contiguous RDMA writes plus small control updates.  The H2D phase is then
local to each rank and can use ordinary CUDA copies from pinned host memory.

Current 2nx4g/1MiB performance:

| Backend | Out-of-place | In-place | Correct |
| --- | ---: | ---: | --- |
| NCCL no-GDR | `119.07 us` | `117.32 us` | Yes |
| Lite native | `84.57 us` | `83.64 us` | Yes |

## Limitations

The NUMA-split path is specialized for two nodes with four local ranks and at
least two available IB transports.  Other two-node layouts use the single
host-slab path; other layouts use the generic chunked send/recv path.  The
design also depends on reliable local shared memory, NUMA placement, and mscclpp
IB registration during communicator setup.

Small messages below 128KiB are now close to NCCL and beat it on several sizes,
but not every row.  The retained small path improves latency from roughly
`34-39 us` to about `18-25 us` for 128B-32KiB and about `20 us` at 64KiB.
