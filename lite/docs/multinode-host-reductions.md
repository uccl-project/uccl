# Experimental shared-host reductions

`make multinode_host_reduce` builds an explicitly selected, eager-only FP32
AllReduce/ReduceScatter experiment. It transfers the shared-packet,
readiness-driven progress, and reader-credit techniques from the November 15
AllGather experiment. **It is slower than the existing GPU implementations in
the measured 1–64 MiB sweep.** No automatic policy or README performance curve
selects it. See the [measurement record](../benchmarks/data/host_reduce_2x4l4_20261115/README.md).

## Schedule and ownership

Two MPI processes per host each own one NUMA group and its two GPUs. Input
bytes per GPU are N; AllReduce produces N bytes and ReduceScatter N/8. Each
input is divided into eight destination shards, each further divided into
chunks. Chunks visit the eight destinations in turn, distributing reduction
work between the NUMA groups. Chunk size counts bytes of the original input,
not an AllGather input shard or a two-GPU host packet.

1. Both GPUs upload a chunk into their group's pinned shared packet. Each
   upload records a CUDA completion event. The groups exchange readiness
   notifications only after both uploads complete.
2. The destination's local NUMA group owns the CPU reduction of the four local
   GPU contributions. It reads the other group's packet directly through the
   shared mapping. There is no intermediate local MPI payload copy.
3. For ReduceScatter, only the node that does not own the destination sends its
   partial to the owner. This sends N/2 bytes per node over the whole call.
   AllReduce exchanges node partials in both directions, N bytes per node.
4. The owner adds the received partial and publishes completion to its local
   peer. ReduceScatter downloads only to the destination GPU; AllReduce lets
   all four GPUs read the shared result. SM kernels perform the downloads.
5. Slot reuse waits for upload notifications, outgoing network transfers,
   GPU consumption, and the local peer's consumption ACK. In AllReduce the
   outgoing node partial is immutable until its send completes, before the
   remote partial is added in place.

Three slots progress independently. Separate tags per slot and ordered reuse
keep generations distinct; no global barrier occurs inside the collective.
MPI shared windows require the unified memory model. Each allocation is first
touched on its owning NUMA node and remains registered through all readers.
The shared helpers are also used by the existing host AllGather harness.

`--pair-reduce 1` explicitly enables a GPU pair reduction experiment. Each GPU
reduces half the chunk using its NUMA mate's input, then uploads that half.
The CPU reduces two group partials instead of four GPU contributions. This
requires CUDA peer access, checked at initialization; it halves D2H traffic but
adds peer reads, kernel launches and a device staging buffer. It improved the
AllReduce host experiment and regressed ReduceScatter in the recorded runs.
It is disabled by default.

## Why AllGather's gain did not transfer

AllGather uploads only N/8 bytes from each GPU. Reductions must account for N
bytes from every GPU. This experiment uploads 4N bytes per node (2N with pair
reduction) before network exchange. The existing GPU hierarchy reduces and
partitions data locally before its inter-node exchange. Removing a local host
copy does not remove this extra D2H traffic or CPU reduction work. These are
traffic and scheduling observations, not profiler attribution of latency.
A subsequent experiment should complete local GPU ReduceScatter before host
staging, then apply shared packets and readiness progress to reduced shards.

## Run

Build with the installed MPI compiler, then stage the executable at the same
absolute path on both hosts:

```bash
make multinode_host_reduce MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  bash scripts/run_multinode_host_reduce.sh --op reducescatter \
  -b 1048576 -e 67108864 -k 1048576 -w 5 -n 10 -r 3
```

Use `--op allreduce` for AllReduce and `--pair-reduce 1` for the optional pair
stage. The runner compares NCCL Ring+Simple in the same four-process harness;
`--nccl 0` disables that comparison. Every warmup and timed call checks the
entire output against rank/index/generation-dependent exact FP32 integer data.
`--delay-consumer-rank 0..7` injects a device-side delay before downloads.
Sizes must be divisible by 128, at least 1024 bytes and at most 1 GiB; chunks
must be divisible by 128 and at most 16 MiB. Short final chunks are supported.
Only eager execution is implemented. This standalone harness does not modify
the optional CPU-router core or its dispatcher.
