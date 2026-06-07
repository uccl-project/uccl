# Lite-collective AllReduce design

## Primary design

The multi-node AllReduce fast path composes native ReduceScatter and native
AllGather:

```text
full input on each rank
  -> native ReduceScatter produces this rank's reduced shard
  -> native AllGather distributes all reduced shards
  -> full reduced output on each rank
```

This is the right shape for a no-GPUDirect-RDMA PCIe system because it reduces
data before the broadcast-style phase.  It also reuses the two tuned native
building blocks instead of introducing a separate full-vector host-staged
AllReduce path.

## Implementation details

The public wrapper tries the registered mscclpp single-node algorithm first,
then calls `runSendRecvAllReduce` for multi-node native execution, then falls
back to dlopen NCCL only if native execution is unsupported:
[`nccl.cu:L1990-L2075`](../nccl/nccl.cu#L1990-L2075) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:1990:1)).

For two-node layouts with counts divisible by `worldSize`, the fast path writes
the rank's reduced shard into the correct location in `recvbuff`, then calls
AllGather on that shard:
[`native_collectives.cu:L1829-L1866`](../nccl/native_collectives.cu#L1829-L1866) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1829:1)).

For irregular counts or unsupported fast-path cases, the implementation keeps a
hierarchical fallback.  That fallback reduces across local ranks, exchanges a
local partial with the same local rank on the remote node, then reduces the two
partials:
[`native_collectives.cu:L1191-L1288`](../nccl/native_collectives.cu#L1191-L1288) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1191:1)).

If the call is out-of-place and the generic non-two-node path is used, the code
copies input to output first and then runs the in-place RS+AG composition:
[`native_collectives.cu:L1868-L1896`](../nccl/native_collectives.cu#L1868-L1896) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1868:1)).

## Single-node path

Single-node AllReduce uses the existing registered mscclpp algorithms.  On
NVIDIA PCIe GPUs, the selector sends medium messages to RSAG because it avoids
packet overhead and is faster on the 1MiB L40/L41 target:
[`algorithm_selector.cc:L104-L165`](../collective/algorithm_selector.cc#L104-L165) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:104:1)).

The algorithms are registered in the collection builder, including RSAG,
pipeline RSAG, and zero-copy variants:
[`algorithm_collection_builder.cc:L120-L134`](../collective/algorithm_collection_builder.cc#L120-L134) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_collection_builder.cc:120:1)).

## Why it beats NCCL no-GDR

At 2nx4g/1MiB, NCCL no-GDR AllReduce is very expensive because full-vector
traffic goes through NCCL's no-GDR network/proxy path.  The native fast path
does not move the full vector as one monolithic inter-node AllReduce.  It first
reduces into per-rank shards, then distributes those shards using the fast
host-slab AllGather path.

Current 2nx4g/1MiB performance:

| Backend | Out-of-place | In-place | Correct |
| --- | ---: | ---: | --- |
| NCCL no-GDR | `1281.14 us` | `783.76 us` | Yes |
| Lite native | `232.54 us` | `234.32 us` | Yes |

The speedup comes from algorithmic traffic shaping, not from bypassing host
memory.  The design accepts the host-staging constraint and reduces the amount
and granularity of traffic that must use it.

## Limitations

The best multi-node path depends on native ReduceScatter.  ReduceScatter alone
does not yet beat NCCL on 2nx4g, so further AllReduce gains should come from
the same ReduceScatter bottleneck work.  The fastest path also requires
`count % worldSize == 0`; otherwise the implementation uses the hierarchical
fallback.
