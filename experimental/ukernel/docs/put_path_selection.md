# Put path selection (IPC vs device vs RDMA)

Design notes on how the ukernel CCL routes same-host cross-GPU puts, why
the original latency-metric load balancer was disabled, and what a
correct adaptive selector would look like.

## Current state

Same-host cross-GPU puts are pinned to **IPC** (`pick_put_path` returns
`PutPath::Ipc` unconditionally for same-host peers). Remote peers always
use RDMA. `UK_CCL_PUT_PATH=device|ipc|rdma` forces any path for A/B
benchmarking.

Measured on the A40 pair (2 ranks, 256MB AllGather):

| path | throughput | note |
|---|---|---|
| IPC (sliding window) | 70 GB/s aggregate | the fast path |
| device (kernel copy, 8 blocks) | ~36 GB/s | single-block was ~18 GB/s |
| RDMA loopback | ~2 GB/s | data loops through host/NIC |

Pinning same-host puts to IPC took 256MB AllReduce from 12.5ms to
5.5ms (beating native NCCL's 6.35ms) and AllGather from 15.9ms to
3.8ms.

**B300 (2026-08-06) changes the device-path verdict for AllToAll**: with
the copy op reduced to plain vectorized LD/ST (the same mechanism NCCL
uses intra-node; TMA bulk copies hang on peer-mapped addresses), the
device path at `UK_CCL_DEV_BLOCKS=64` reaches ~400 GB/s algbw for 256MB
alltoall at 8 ranks — ~15% faster than the IPC/CE path (715 -> ~600us)
and it flattens the rank-scaling curve (4r ~= 8r). It still loses
marginally at 2 ranks (324 vs 310us). So the selector should be
rank-count and message-size aware, not a single same-host default.

## Why the latency-metric balancer failed

The original `pick_put_path` picked the path minimizing
`inflight × latency_ns` (a Little's-law style queueing estimate). It was
wrong for two structural reasons:

1. **The sliding window inflates the IPC latency metric.** IPC
   completions are published in bursts after a window sync, so the
   measured per-put completion latency looks high — even though IPC has
   the highest throughput. The metric and the physical truth disagreed.
2. **The device path has low latency but low capacity.** A single-block
   persistent-kernel copy task completes fast (small tasks, low latency)
   but only sustains ~18 GB/s. Latency is a poor proxy for capacity,
   and the device path's capacity additionally depends on
   `blocks_per_worker` (8 blocks ≈ 40 GB/s, 64 blocks ≈ 52 GB/s), which
   the metric never knew.

## When each path can genuinely win

| path | when it can win | verdict |
|---|---|---|
| IPC | almost always, same-host — it *is* the GPU copy-engine/DMA path | correct default at 2 ranks and for small messages |
| device | (a) copy engines contended or PCIe-capped; (b) vectorized LD/ST at high block counts — **beats IPC by ~15% for 4+ rank large-message AllToAll on B300**; (c) the real win is a whole-collective-in-kernel mode — no CPU round trips, small-message latency approaches native | current per-put device path wins the large-message multi-rank AllToAll case; revisit the selector to pick it there |
| RDMA (same-host) | essentially never — data loops through host/NIC | keep excluded from same-host selection |

Cross-node traffic is always RDMA (the ring seams between nodes); the
selector only ever decides same-host IPC-vs-device.

## Correct adaptive design: capacity probes, not latency

A proper selector measures what it needs instead of guessing:

1. **At path setup, probe each candidate**: one large transfer for
   capacity (GB/s), one small transfer for latency; cache the results.
2. **Select by message size**: large messages → highest-capacity path;
   small messages → lowest-latency path.
3. **Same-host excludes RDMA** unconditionally.
4. **Re-probe periodically / on path failure** (e.g., IPC completions
   stall → fall back to device).
5. The capacity probe naturally accounts for configuration-dependent
   device throughput (it measures the current `blocks_per_worker`).

Estimated size: 100-200 lines, behind an env switch so the current
"pinned IPC" behavior stays the default until a machine demonstrates a
need.

## Recommendation

- Keep "same-host → IPC" as the default for 2 ranks and small messages.
- Use the device path for 4+ rank large-message AllToAll on B300
  (`UK_CCL_PUT_PATH=device UK_CCL_DEV_BLOCKS=64`) — measure and pick per
  rank count / message size.
- Do not reintroduce latency-metric-based selection — it was measuring
  the wrong quantity.
- Cross-node (the multi-node goal) is RDMA-only regardless; the selector
  only ever decides same-host IPC-vs-device.
