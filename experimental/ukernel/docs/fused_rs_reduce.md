# AllReduce copy+reduce fusion (fused RS / AG)

Status: **implemented on `uk-300`** — `UK_CCL_FUSE_REDUCE_COPY=1`
(fused reduce+copy in the reduce-scatter phase) and
`UK_CCL_FUSE_AG_COPY=1` (fused device copy in the all-gather phase).
Measured gain: +7.5% at 4 ranks, +18.8% at 8 ranks (256M AllReduce,
OOP). The earlier remote-read reduce variant is a dead end and is kept
behind `UK_CCL_FUSE_RS_REDUCE` (default 0) only as a reference.

## Motivation

The AllReduce ring is a serialized per-hop critical path: RS (7 hops) +
AG (7 hops) at 8 ranks, with a host signal chain on every hop
(task done -> dev drain -> enqueue signal -> host ring write -> peer
poll -> enqueue next task). CE contention
([ce_contention.md](ce_contention.md)) and per-tile host-signal latency
dominate the gap to native NCCL (which keeps sync in-kernel with LL
flags). The fusion attacks both: reduce and copy happen in one device
task, and completion is signaled with a device-written flag that the
host polls directly.

## Design space: two fused-RS shapes

### Remote-read reduce — dead end (UK_CCL_FUSE_RS_REDUCE)

The receiver's reduce kernel reads the peer's send-source buffer directly
over NVLink (NCCL LL-style), and the sender's signal means "data ready"
instead of "put landed". Measured 1.4-1.5x SLOWER at 2/4/8 ranks:
the remote read is latency-bound — throughput scales almost linearly
with block count (43 GB/s at BLK=8 -> 179 GB/s at BLK=64) and collapses
at BLK=128, which defeats the few-SM goal. It also proved that CE
contention is NOT the dominant AllReduce cost: a ring step has only 8
concurrent copies (one per rank), not the 56-way alltoall peak.

### Fused reduce+copy — the winning shape (UK_CCL_FUSE_REDUCE_COPY)

Each RS RecvReduce task reduces its shard and forwards it to the next
rank's accumulation buffer in the same task — device LD/ST write to the
peer (the alltoall-proven direction). This removes the reduce->put host
transition and the separate put op from the ring's per-hop critical
path.

## Completion signaling: device-flag slots

B300 reports `gpuDevAttrHostNativeAtomicSupported=0`, so kernels cannot
claim the shared IPC signal ring. Instead each fused task owns a
single-writer slot in a host-mapped flag area:

- The device task writes the salted tag with a plain store +
  `__threadfence_system` (no atomics); the matching WaitSignal polls the
  slot from the host. This removes the dev-drain -> enqueue -> host
  ring-write transitions.
- Slots are collision-free (`pair*K + tile`, K = plan tile bound);
  plans that exceed the fixed flag area fall back to host-written
  signals.
- G>1 uses counted waits: poll `flag_count` consecutive slots and
  complete when all match `base_tag + i`.
- Two correctness fixes are folded in: the unconditional salted tag in
  `make_cmd` (a fused-signal conditional zeroed tags for ordinary ops
  and deadlocked AllReduce), and a dedicated `signal_tag` field so the
  flag tag cannot clobber `TaskArgs.redTypeRaw` (slot 0 tripped the
  reduction assert).

Toggles: `UK_CCL_FUSE_REDUCE_COPY` (default 0, forces G=1), and
`UK_CCL_DEVICE_FLAGS` (default on when fused).

## Fused AG copy (UK_CCL_FUSE_AG_COPY=1)

The AG forward becomes a device copy task (read my output, write next's
output) with an inline device-completion flag — no CE, no host signal
per hop. It reuses the RS flag machinery (per-tile slots, counted waits,
capacity fallback). The executor routes these puts to the device backend
only; `UK_CCL_DEV_FIFOS=4` over-subscribes the SMs (4x64 > 148 SMs) and
collapses, so keep the default 2 workers.

## Results (B300, 256M AllReduce, LT=16 TM=8M IB=16 BLK=64, n=20, OOP)

| config | 4r | 8r |
|---|---:|---:|
| fuse=0 | 1177us | 2007us |
| fused RS | 1122us | 1560us |
| fused RS+AG | 1129us | **1487us** |
| native | ~669us | ~719us |

All wrong=0; 8-rank stress (n=100) wrong=0 (validates flag write
ordering under sustained load). RS fusion: +7.5% at 4r, +18.8% at 8r.
AG fusion is neutral at 4 ranks (worker serialization offsets the
host-chain savings) and ~5% at 8 ranks (1560 -> 1487us). The remaining
~770us gap at 8 ranks is the ring's serialized critical path (14 hops x
per-hop latency) and the residual AG/put pipeline — see
[optimization_framework.md](optimization_framework.md).

LT sweep: LT=16 BLK=64 remains the fused-path optimum; deeper tiles
regress (per-tile host cost caps depth); BLK=128 over-subscribes and
collapses.

## Build speed

`persistent_kernel_ops.cu` with `TMA_REDUCE=1 REDUCE_SMEM_KB=224` takes
15-25min (TMA bulk/warp-spec template instantiations). C++-only
iterations relink in ~1min. Validation builds: `make VALIDATE=1 -j8
nccl` disables the TMA paths (the fused work runs on the vector LD/ST
path); keep `TMA_REDUCE=1` only for final perf builds.
