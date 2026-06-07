# Lite-collective native collectives design overview

This document summarizes the native `lite-collective` design for AllReduce,
AllGather, ReduceScatter, and AllToAll on the L40/L41 PCIe-only testbed.  The
important constraint is that inter-node traffic has no GPUDirect RDMA, so every
inter-node path must cross host memory.  The implementation treats host staging
as a first-class data structure instead of a hidden fallback buffer.

## Current support and performance snapshot

| Collective | Native coverage | Current 2nx4g status | Primary design | Code |
| --- | --- | --- | --- | --- |
| AllGather | 1nx2g, 2nx1g, 2nx2g, 2nx4g | Beats NCCL no-GDR at 1MiB: 2nx1g `82.91/79.96 us` vs `97.72/96.96 us`; 2nx4g `85.74/85.37 us` vs `118.20/117.53 us` | `<128KiB`: ordered ring-slot host slab with direct-QP data+flag writes and one full-output H2D; `>=128KiB`: dynamic NIC-group host slabs; 2nx1g skips generic fallback and copies only the remote chunk for in-place or >=128KiB | [`lite_allgather.cu:L942-L1232`](../nccl/lite_allgather.cu#L942-L1232) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/lite_allgather.cu:942:1)) |
| ReduceScatter | 1nx2g, 2nx1g, 2nx4g | Correct and improved at 1MiB, but still behind NCCL no-GDR: `123.29/123.35 us` vs `107.65/107.59 us` | Single-node memory-channel RS; 2nx4g float/sum path does NUMA-pair local GPU fan-in, pairwise host-staged RDMA, and final GPU add | [`native_collectives.cu:L1418-L1728`](../nccl/native_collectives.cu#L1418-L1728) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1418:1)) |
| AllReduce | 1nx2g, 2nx1g, 2nx4g | Beats NCCL no-GDR at 1MiB: `232.54/234.32 us` vs `1281.14/783.76 us` | Compose native ReduceScatter plus native AllGather when the count is divisible by world size; keep a hierarchical fallback for irregular counts | [`native_collectives.cu:L1829-L1908`](../nccl/native_collectives.cu#L1829-L1908) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1829:1)) |
| AllToAll | 1nx2g, 2nx1g, 2nx4g | Beats NCCL at 1nx2g/2nx1g 1MiB; 2nx4g remains slower: `107.44/101.25 us` vs `96.85/95.29 us` | Intercept grouped send/recv, do self pairs as D2D copies, use unified custom P2P for local CudaIpc and remote host-staged IB; optimized node/per-NUMA remote slabs are opt-in | [`nccl.cu:L2313-L2562`](../nccl/nccl.cu#L2313-L2562) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2313:1)) |

The support and benchmark source of truth is the L40/L41 runbook:
[`l40-l41-p2p-runbook.md:L20-L69`](l40-l41-p2p-runbook.md#L20-L69) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/doc/l40-l41-p2p-runbook.md:20:1)).

## Shared design principles

1. Use mscclpp directly for the native communication substrate.  Host slabs are
   pinned with `cudaHostRegister`, registered with mscclpp, and exchanged over
   mscclpp `Connection::write` rather than routed through NCCL fallback.
2. Aggregate inter-node work at node granularity where possible.  On a no-GDR
   system, small per-rank network transactions pay extra CPU/proxy overhead; a
   node leader writing one contiguous slab is cheaper for AllGather and parts of
   AllToAll.
3. Keep intra-node movement on GPU paths.  Single-node collectives use existing
   mscclpp memory-channel algorithms when available, and multi-node
   ReduceScatter uses CudaIpc scratch exchange before crossing the network.
4. Keep NCCL fallback as a compatibility escape hatch, but make native paths run
   before dlopen fallback in the NCCL shim wrappers.

## How native collectives enter the NCCL shim

The public NCCL APIs first try registered single-node mscclpp algorithms, then
the custom native send/recv based implementations, and only then dlopen NCCL
fallback.  AllReduce, ReduceScatter, and AllGather are wired in
[`nccl.cu:L1990-L2235`](../nccl/nccl.cu#L1990-L2235) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:1990:1)).

The single-node selector registers and chooses existing mscclpp algorithms for
AllReduce, AllGather, and ReduceScatter.  Relevant selector/registration code:
[`algorithm_selector.cc:L104-L212`](../collective/algorithm_selector.cc#L104-L212) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:104:1)) and
[`algorithm_collection_builder.cc:L120-L145`](../collective/algorithm_collection_builder.cc#L120-L145) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_collection_builder.cc:120:1)).

Multi-node selection is intentionally not in the generic algorithm selector yet:
[`algorithm_selector.cc:L214-L229`](../collective/algorithm_selector.cc#L214-L229) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:214:1)).
Instead, the NCCL shim calls native two-node helpers directly before falling
back to NCCL.

## Why the current winners beat NCCL no-GDR

AllGather wins because the required host staging is made explicit, coalesced,
and NUMA-aware.  On 2nx4g, each node ships two packed host slabs through the two
local NICs instead of one node-leader slab through one NIC, while NCCL no-GDR
still pays per-channel/proxy overhead around staged transfers.  The native path
also uses RDMA-written host control flags, so the remote side does not need
another bootstrap round trip on the steady-state fast path.

AllReduce wins because the fast path uses ReduceScatter followed by AllGather.
For 2nx4g, each rank exchanges a shard instead of repeatedly moving the full
vector through host staging.  Even though the native ReduceScatter phase is not
yet faster than NCCL by itself, the composed path avoids NCCL's expensive
full-vector no-GDR AllReduce behavior on this testbed.

## Current gaps

ReduceScatter is close but still not faster than NCCL on 2nx4g.  The remaining
cost is mostly local GPU scratch exchange, stream synchronization, host-staged
pairwise exchange, and the final H2D plus add.  The latest retained design is
documented in [ReduceScatter design](reducescatter-design.md).

AllToAll is correct and useful, but the 2nx4g native path is still slower than
NCCL.  The unified grouped P2P path is the best retained default because it
overlaps local and remote peers; the opt-in slab paths serialize remote work
before local work and have not closed the gap.  Details are in
[AllToAll design](alltoall-design.md).

## Reading path

1. [AllGather design](allgather-design.md)
2. [ReduceScatter design](reducescatter-design.md)
3. [AllReduce design](allreduce-design.md)
4. [AllToAll design](alltoall-design.md)
