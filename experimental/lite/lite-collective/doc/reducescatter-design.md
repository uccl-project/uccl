# Lite-collective ReduceScatter design

## Primary design

ReduceScatter has two native designs:

1. Single-node ReduceScatter uses mscclpp memory-channel writes into peer
   scratch buffers, then each rank reduces the shard it owns.
2. Two-node 2nx4g float/sum ReduceScatter uses a GPU-local NUMA-pair fan-in
   before doing pairwise host-staged RDMA between matching local ranks across
   nodes.

The second path is the current optimization focus.  It is correct and much
faster than the initial native implementation, but it does not yet beat NCCL
no-GDR on 2nx4g.

## Single-node path

The single-node kernel writes the target shard data for every peer into that
peer's scratch memory via mscclpp memory-channel operations, signals/waits on
the memory channels, then reduces local plus remote scratch rows into the
output:
[`reducescatter_rs.cu:L68-L132`](../collective/reducescatter_rs.cu#L68-L132) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/reducescatter_rs.cu:68:1)).

The algorithm is registered as `default_reducescatter_rs`:
[`algorithm_collection_builder.cc:L135-L139`](../collective/algorithm_collection_builder.cc#L135-L139) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_collection_builder.cc:135:1)).

The selector uses this algorithm for single-node SUM and MIN:
[`algorithm_selector.cc:L203-L212`](../collective/algorithm_selector.cc#L203-L212) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:203:1)).

## Two-node host and IPC context

The two-node context owns a shared host work slab and control block.  It also
registers mscclpp IB memory for inter-node partial exchange and a CudaIpc
scratch registration for local GPU-to-GPU scratch exchange:
[`native_collectives.cu:L383-L636`](../nccl/native_collectives.cu#L383-L636) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:383:1)).

The control block has separate epochs for host-staged block exchange, pairwise
RDMA exchange, local CudaIpc copy readiness, cross-pair readiness, and final
local completion:
[`native_collectives.cu:L55-L80`](../nccl/native_collectives.cu#L55-L80) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:55:1)).

## 2nx4g NUMA-pair fast path

The 2nx4g path is specialized for `ncclFloat32 + ncclSum`.  It chooses the
NUMA-pair local path when there are four ranks per node and the scratch buffer
can hold the packed rows:
[`native_collectives.cu:L1437-L1459`](../nccl/native_collectives.cu#L1437-L1459) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1437:1)).

The local GPU phase does this:

```text
local ranks 0/1 reduce within one NUMA pair
local ranks 2/3 reduce within one NUMA pair
one cross-pair partial exchange completes the local-node reduction
the code emits two partials:
  - this rank's final local-node partial
  - the remote-node shard partial that must be sent to the matching remote rank
```

The CUDA kernels for packing, pair reduction, cross-pair final reduction, and
the dedicated final add are in
[`native_collectives.cu:L806-L1132`](../nccl/native_collectives.cu#L806-L1132) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:806:1)).

The runtime flow is:

1. Pack local pair data and partner data.
2. Copy partner rows into the peer GPU's registered scratch over CudaIpc.
3. Reduce pair partials and prepare a cross-pair partial.
4. Copy the cross-pair partial into the target GPU's scratch.
5. Produce the local partial and the remote partial.
6. D2H the remote partial into a pinned host slot.
7. RDMA-write that slot to the matching local rank on the remote node.
8. H2D the incoming remote partial and run the dedicated float add into
   `recvbuff`.

Steps 1-5 are implemented in
[`native_collectives.cu:L1485-L1538`](../nccl/native_collectives.cu#L1485-L1538) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1485:1)).
The pairwise D2H/RDMA/H2D/final-add phase is implemented in
[`native_collectives.cu:L1658-L1727`](../nccl/native_collectives.cu#L1658-L1727) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1658:1)).

## Generic two-node fallback

If the GPU-local path is unsupported, the implementation can fall back to a
host-staged path.  It D2Hs each rank's full input into a host slab, reduces the
local-node contributions in CPU memory, RDMA-writes the remote partial block,
adds the incoming remote partial on CPU, and H2Ds the final shard:
[`native_collectives.cu:L1291-L1416`](../nccl/native_collectives.cu#L1291-L1416) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1291:1)).

For non-two-node or unsupported cases, `runSendRecvReduceScatter` falls back to
a chunked grouped send/recv plus GPU row reduction:
[`native_collectives.cu:L1740-L1827`](../nccl/native_collectives.cu#L1740-L1827) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1740:1)).

## Current performance and gap

Current 2nx4g/1MiB performance:

| Backend | Out-of-place | In-place | Correct |
| --- | ---: | ---: | --- |
| NCCL no-GDR | `107.65 us` | `107.59 us` | Yes |
| Lite native | `123.29 us` | `123.35 us` | Yes |

The native path is close but still slower.  The likely remaining costs are:

- Multiple local CudaIpc scratch copies and synchronization points.
- D2H of the remote partial before pairwise RDMA.
- H2D of the incoming partial before final add.
- The final stream synchronization needed before the shard can be consumed.

Rejected experiments and their measured costs are listed in the runbook:
[`l40-l41-p2p-runbook.md:L71-L104`](l40-l41-p2p-runbook.md#L71-L104) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/doc/l40-l41-p2p-runbook.md:71:1)).
