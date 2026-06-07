# Lite-collective AllToAll design

## Primary design

AllToAll is implemented primarily by recognizing the grouped
`ncclSend`/`ncclRecv` pattern used by `nccl-tests`.  The default path keeps local
and remote peers in the same grouped P2P execution pipeline:

```text
self pair     -> direct D2D copy
same-node peer -> CudaIpc send/recv path
remote peer    -> D2H pinned staging, mscclpp IB RDMA write, H2D on receiver
```

This unified grouped path is currently better than the specialized optimized
slab paths on 2nx4g because it preserves local/remote overlap.

## Grouped P2P interception

`ncclGroupEnd` first removes self send/recv pairs and executes them locally:
[`nccl.cu:L2313-L2341`](../nccl/nccl.cu#L2313-L2341) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2313:1)).
The self-pair implementation handles cross-stream ordering and then issues a
device-to-device copy:
[`alltoall.cu:L742-L773`](../nccl/alltoall.cu#L742-L773) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:742:1)).

After self-pair handling, the group code initializes peer contexts, separates
custom P2P operations from dlopen fallback operations, exchanges CudaIpc receive
buffer addresses, and executes custom sends/recvs:
[`nccl.cu:L2343-L2562`](../nccl/nccl.cu#L2343-L2562) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2343:1)).

The API surface used by the grouped optimizer is declared in
[`alltoall.hpp:L9-L35`](../nccl/alltoall.hpp#L9-L35) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.hpp:9:1)).

## Explicit `ncclAllToAll` API

The explicit `ncclAllToAll` wrapper validates arguments, handles the self slice
with a D2D copy, then expands every non-self peer into grouped send/recv calls:
[`alltoall.cu:L863-L940`](../nccl/alltoall.cu#L863-L940) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:863:1)).

`ncclAllToAllv` currently only handles the degenerate world-size-one case and
otherwise reports unavailable:
[`alltoall.cu:L948-L975`](../nccl/alltoall.cu#L948-L975) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:948:1)).

## Opt-in remote slab paths

There are two optimized remote-only paths gated by `MSCCLPP_NCCL_ALLTOALL_OPT`:

- `MSCCLPP_NCCL_ALLTOALL_OPT=node` uses one node-wide host slab.
- Any other nonzero value uses per-NUMA-group host slabs.

The env parsing is in
[`alltoall.cu:L275-L283`](../nccl/alltoall.cu#L275-L283) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:275:1)).

Both modes create shared-memory slabs, pin them, register them with mscclpp,
and connect node leaders over IB:
[`alltoall.cu:L332-L425`](../nccl/alltoall.cu#L332-L425) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:332:1)) and
[`alltoall.cu:L427-L539`](../nccl/alltoall.cu#L427-L539) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:427:1)).

The node-wide remote path packs all remote-bound chunks from a local rank into
the shared send slab, has the leader RDMA-write the aggregate remote slab, then
each rank H2Ds the chunks addressed to itself:
[`alltoall.cu:L668-L738`](../nccl/alltoall.cu#L668-L738) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:668:1)).

The per-NUMA path splits the same idea across two local groups and runs a thread
per group:
[`alltoall.cu:L542-L665`](../nccl/alltoall.cu#L542-L665) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:542:1)).

The grouped matcher only takes these optimized paths for the 2-node 8-rank
layout with contiguous remote send buffers; after executing remote peers it
removes those operations and leaves local peers for the normal grouped path:
[`alltoall.cu:L776-L855`](../nccl/alltoall.cu#L776-L855) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/alltoall.cu:776:1)).

## Current performance and lesson

Current 1MiB status:

| Topology | NCCL out | Lite native out | Status |
| --- | ---: | ---: | --- |
| 1nx2g | `59.52 us` | `41.04 us` | Native wins |
| 2nx1g | `74.32 us` | `73.93 us` | Native narrowly wins |
| 2nx4g | `96.85 us` | `107.44 us` | Native is correct but slower |

The main lesson from the optimized paths is that remote-only slab aggregation is
not automatically better.  The optimized paths peel remote peers and
synchronize before local peers run, which loses remote/local overlap.  The
default grouped P2P path keeps all peers together and is therefore the retained
default for 2nx4g.

