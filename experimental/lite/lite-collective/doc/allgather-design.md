# Lite-collective AllGather design

## Primary design

The multi-node AllGather design is a two-node host-slab exchange.  Because the
testbed has no GPUDirect RDMA, every inter-node transfer must already pass
through CPU memory.  For medium/large messages, the implementation splits local
ranks into contiguous NIC groups based on `nRanksPerNode` and available HCAs:

```text
group0 sendbuffs -> group0 host slab -> selected NIC -> remote group0 slab
group1 sendbuffs -> group1 host slab -> selected NIC -> remote group1 slab
...
each rank H2Ds all local and remote groups into recvbuff
```

On the L40/L41 2nx4g case this becomes two groups: GPU0/1 and GPU2/3, using
the two local NICs.  On 2nx2g with GPU0/1, both GPUs are on the same NUMA node,
so the medium/large path uses one group.  Smaller messages use a pipelined
single-leader host-slab protocol with ring-buffered slots and direct QP writes.
Each slot is laid out in global-rank order, so the remote node writes its half
directly into the missing half of the local output slot.  That removes the
per-iteration remote ack, most `Connection::write` overhead, and the CPU repack
step.

## Implementation details

The public `ncclAllGather` wrapper first gives the single-node algorithm
collection a chance to run.  If that is not selected or returns unsupported, the
wrapper calls `runLiteAllGather` before NCCL dlopen fallback:
[`nccl.cu:L2159-L2235`](../nccl/nccl.cu#L2159-L2235) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2159:1)).

For two-node layouts, `runLiteAllGather` dispatches to the host-slab fast
path; otherwise it falls back to a chunked grouped send/recv implementation:
[`lite_allgather.cu:L1125-L1174`](../nccl/lite_allgather.cu#L1125-L1174) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:1125:1)).

The host context is owned by each node leader or NUMA-group leader.  It creates
shared-memory slabs, maps the leader-owned names into all local ranks, NUMA
places and pins the mappings, registers the slabs with mscclpp, and connects to
the matching remote leader:
[`lite_allgather.cu:L250-L751`](../nccl/lite_allgather.cu#L250-L751) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:250:1)).

The steady-state data path for `>=128KiB` is:

1. Each local rank copies its input chunk into its NIC group's `sendSlab`.
2. Each group leader waits for all local `d2hReady` flags in its group.
3. Each group leader writes its group slab to the matching remote group slab.
4. Every local rank waits for all groups' remote-ready flags.
5. Every local rank copies all local groups and all remote groups into the
   correct positions in `recvbuff`.
6. Local ranks publish `h2dDone`; each group leader sends a remote ack before
   the next chunk reuses the slab.

That fast path is implemented in
[`lite_allgather.cu:L989-L1112`](../nccl/lite_allgather.cu#L989-L1112) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:989:1)).
Large messages stay on the host-slab path by chunking at the slab capacity:
[`lite_allgather.cu:L750-L896`](../nccl/lite_allgather.cu#L750-L896) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:750:1)).

### Small-message ordered direct-QP ring-slot path

The `<128KiB` path is implemented in
[`lite_allgather.cu:L899-L987`](../nccl/lite_allgather.cu#L899-L987) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:899:1)).
It is algorithmically different from the medium/large NUMA-split path:

1. The per-epoch host slot is laid out in final AllGather output order:
   `[rank0][rank1]...[rankN][ready_flag]`.
2. Each rank D2Hs its input directly into the slot offset for its global rank.
3. The node leader waits for local ranks, then posts direct-QP RDMA writes for
   the local node block and a ready flag into the remote node's matching
   `sendSlab` slot.
4. Remote ranks poll the flag in their local `sendSlab`, then each rank performs
   one H2D copy of the whole output slot to `recvbuff`.
5. Slot reuse is guarded by a ring-slot wrap barrier: stream sync, QP completion
   polling, and a bootstrap barrier.

The direct-QP helpers are in
[`lite_allgather.cu:L201-L247`](../nccl/lite_allgather.cu#L201-L247) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:201:1)).
The route split between `<128KiB` and `>=128KiB` is in
[`lite_allgather.cu:L1125-L1174`](../nccl/lite_allgather.cu#L1125-L1174) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:1125:1)).

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

Current stable 2nx4g performance:

| Backend | Out-of-place | In-place | Correct |
| --- | ---: | ---: | --- |
| NCCL no-GDR, 1MiB | `118.20 us` | `117.53 us` | Yes |
| Lite native, 1MiB | `85.74 us` | `85.37 us` | Yes |
| NCCL no-GDR, geomean 128B-1GiB | baseline | baseline | Yes |
| Lite native, geomean 128B-1GiB | `1.167x` speedup | `1.181x` speedup | Yes |

## Limitations

The grouped slab path is specialized for two-node layouts with at least two
local ranks, at least two IB transports, and GPUs spanning multiple NUMA nodes.
It supports arbitrary `nRanksPerNode <= 8`; groups are contiguous local-rank
ranges, split when the actual GPU NUMA node changes and capped by the available
HCA count.  Other layouts use the generic chunked send/recv path.  The design
also depends on reliable local shared memory, NUMA placement, and mscclpp IB
registration during communicator setup.  The generalized path was smoke-tested
on 2nx2g and 2nx4g.

Small messages below 128KiB now beat NCCL in the retained sweep.  With
`--iters 50 --warmup-iters 10`, the native path wins `9/10` out-of-place rows
and `8/10` in-place rows below 128KiB, with geomean speedups `1.111x` and
`1.084x`.
