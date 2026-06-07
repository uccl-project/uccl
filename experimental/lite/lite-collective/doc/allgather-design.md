# Lite-collective AllGather design

## Primary design

The multi-node AllGather design is a two-node host-slab exchange.  Because the
testbed has no GPUDirect RDMA, every inter-node transfer must already pass
through CPU memory.  The implementation makes that host memory a per-node
aggregate slab:

```text
GPU rank i sendbuff
  -> D2H into local shared host slab slot i
  -> node leader RDMA-writes the whole local slab to the remote node slab
  -> each local rank H2Ds both local and remote slabs into recvbuff
```

For 2nx4g, this changes the inter-node shape from many rank-to-rank transfers
into one contiguous block per node.  That is the main reason the native path
beats NCCL no-GDR at 1MiB.

## Implementation details

The public `ncclAllGather` wrapper first gives the single-node algorithm
collection a chance to run.  If that is not selected or returns unsupported, the
wrapper calls `runSendRecvAllGather` before NCCL dlopen fallback:
[`nccl.cu:L2159-L2235`](../nccl/nccl.cu#L2159-L2235) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/nccl.cu:2159:1)).

For two-node layouts, `runSendRecvAllGather` dispatches to the host-slab fast
path; otherwise it falls back to a chunked grouped send/recv implementation:
[`native_collectives.cu:L2113-L2185`](../nccl/native_collectives.cu#L2113-L2185) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:2113:1)).

The host context is owned by each node leader.  It creates shared-memory slabs,
maps the leader-owned names into all local ranks, pins the mappings, registers
the slabs with mscclpp, and connects to the remote node leader:
[`native_collectives.cu:L472-L662`](../nccl/native_collectives.cu#L472-L662) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:472:1)).

The steady-state data path is:

1. Each local rank copies its input into its slot in `ctx.sendSlab`.
2. The leader waits for all local `d2hReady` flags.
3. The leader writes the whole local block to `ctx.remoteRecvMemory`.
4. The leader writes a remote control flag so the other node can proceed.
5. Each rank copies the local block and remote block from host slabs to its
   output buffer.
6. Local ranks publish `h2dDone`; the leader sends a remote ack before the next
   epoch reuses the slab.

That fast path is implemented in
[`native_collectives.cu:L1571-L1663`](../nccl/native_collectives.cu#L1571-L1663) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/nccl/native_collectives.cu:1571:1)).

## Single-node path

Single-node AllGather uses the existing mscclpp fullmesh algorithms.  On the
L40/L41 PCIe target, the selector prefers `default_allgather_fullmesh` for
aligned sizes because it is faster than `fullmesh2` for the 1MiB target:
[`algorithm_selector.cc:L168-L199`](../collective/algorithm_selector.cc#L168-L199) ([VS Code](vscode://file/home/yangz/nfs/zhongjie/uccl.worktrees/copilot-collectives-support-table-implementation/experimental/lite/lite-collective/collective/algorithm_selector.cc:168:1)).

## Why it beats NCCL no-GDR

The native path wins because it removes avoidable per-peer network/proxy work.
On this hardware, host memory is unavoidable; packing local ranks into one
host slab per node converts the inter-node phase into one contiguous RDMA write
plus one small control update.  The H2D phase is then local to each rank and can
use ordinary CUDA copies from pinned host memory.

Current 2nx4g/1MiB performance:

| Backend | Out-of-place | In-place | Correct |
| --- | ---: | ---: | --- |
| NCCL no-GDR | `119.27 us` | `116.62 us` | Yes |
| Lite native | `88.33 us` | `88.28 us` | Yes |

## Limitations

The host-slab path is specialized for two-node layouts and bounded by the
configured maximum bytes per rank.  Other layouts use the generic chunked
send/recv path.  The design also depends on reliable local shared memory and
msccclpp IB registration during communicator setup.

