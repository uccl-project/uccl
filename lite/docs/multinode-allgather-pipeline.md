# AllGather: copy engines and chunked forwarding

This explicit experimental path extends the small-message direct multicast in
`multinode_allreduce.cu`. It supports the same two-node, four-GPU-per-node FP32
AllGather layout. `N` always denotes complete **output bytes per GPU**; each GPU
provides `S=N/8` bytes. The existing `--ag-auto` policy is unchanged.

## Schedule

`--ag-pipeline sm` forwards chunks with SM peer writes;
`--ag-pipeline ce` uses MSCCL++ CUDA-IPC PortChannels. In the installed runtime,
`CudaIpcConnection::write` calls `CudaIpcStream::memcpyD2D`, which submits
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...)` on nonblocking CUDA streams.
The IB proxy and local-copy proxy are independent. The optional CPU-router module
is not part of these GPU-memory paths.

At each invocation:

1. Post up to `--ag-depth` network chunks from immutable input to the corresponding
   remote GPU's receive staging slot.
2. Copy the current own chunk to the GPU's own output slot, and forward it to the
   three local peers. SM mode uses direct writes; CE mode queues DMA copies.
3. Wait for this chunk's incoming IB notification. Refill the network lookahead
   immediately, before forwarding the received chunk.
4. Copy the received chunk to this GPU's output and forward it to the other local
   GPUs. Advance to the next chunk without waiting for outgoing local DMA.
5. Send one final local notification per peer, wait for incoming local completion,
   and drain outgoing local DMA and IB operations.

Each output shard has exactly one producer on every destination GPU. The
output is never a forwarding source. Local consumers only use it after collective
completion, so intermediate per-copy local notifications are unnecessary. The
first pilot used those notifications and retained their cost in the raw logs.
The network still signals each chunk, because its recipient forwards chunks as
they arrive. Small pieces increase proxy submissions and NIC notifications;
more overlap does not imply lower end-to-end latency.

`--ag-piece-bytes` counts **bytes of one rank's shard**, not complete output bytes.
A fixed 4 MiB piece means there is only one network chunk through 32 MiB complete
output. Any gain in that range is from the copy/synchronization schedule, not
inter-chunk pipelining. At 64 MiB output that configuration uses two network
chunks; the 1 MiB configuration uses eight.

## NUMA-pair relay experiment

`--ag-pipeline ce-pair` assumes the documented groups `{0,1}` and `{2,3}`. GPU r
sends its own and received remote chunks to its same-NUMA mate `r^1`, and to
staging on one cross-NUMA peer `r^2`. After receiving its cross peer's two chunks,
it copies them to its own output and forwards them within its NUMA pair. The
four diagonal output shards are supplied by that relay.

The total local copy payload remains `24S` per node. Cross-NUMA payload drops
from `16S` with flat fan-out to `8S` with pair relay (the existing ring uses `12S`).
However, relay introduces a cross-peer completion dependency for each chunk and
extra staging reads. It is an opt-in measured alternative, not a presumed win.

## Lifetime and supported launches

Two receive slots alternate by eager launch number. A rank cannot reach call
`i+2` without receiving the partner's call `i+1`, so it cannot overwrite call i's
receive data before the partner finishes reading it. All local outgoing DMA is
flushed before return. Pair relay additionally uses two alternating pairs of
local scratch slots, with the same progression bound on the cross-NUMA channel.
Output offsets use the current shard size; staging offsets use the maximum
configured shard size, so different message sizes never make slots overlap.
The existing allocation has enough room: IB scratch reserves at least `2*Smax`,
and local scratch reserves `8*Smax` while the relay uses `4*Smax`.

The harness accepts complete output extents divisible by 512 bytes, at least
1024 bytes. Piece size must be at least 512 bytes and divisible by 16; a shorter
last chunk is supported. Thus `uint4` copies never discard a partial vector.
At most 64 chunks per rank are allowed, with lookahead 1–64. Grid occupancy is
checked against the selected kernel because multiple CTAs use grid barriers.
Start with `-B 1 -T 256`; more CTAs can increase NUMA contention and synchronization.
The general executable's default block count is 24, so set it explicitly.

The slot number is advanced at each host-side eager launch. Replaying a captured
CUDA graph would need a device-side generation protocol; this experiment does
not supply or validate that API. It does not automatically select a backend from
GPU count, and does not introduce a CPU-router policy entry.

## Reproduce

Build and stage identical binaries and libraries on both hosts as described in
[the measurement record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/ag_pipeline_2x4l4_20261115/README.md).
For an explicit 4 MiB-piece CE experiment:

```bash
HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR="$PWD/build/ag_pipeline_20261115" \
  bash scripts/run_multinode.sh multinode_allreduce \
  --op allgather --ag-pipeline ce --ag-piece-bytes 4194304 --ag-depth 4 \
  -B 1 -T 256 -b 1024 -e 67108864 -w 20 -n 20 --check-each 1 -c 1
```

`scripts/run_multinode_ag_pipeline.sh` runs three backend-alternating rounds with
explicit serial, ring and flat UCCL controls, four new fixed configurations, and
NCCL defaults plus Ring+Simple. `scripts/summarize_multinode_ag_pipeline.py`
rejects missing modes/rounds or nonzero output errors before writing the summary.
