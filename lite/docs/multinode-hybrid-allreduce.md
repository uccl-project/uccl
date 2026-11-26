# GPU ReduceScatter followed by host-packet AllGather

`--host-gather 1 --op allreduce` is an explicit eager AllReduce path for the
documented two four-L4 hosts. Reduction arithmetic executes entirely on GPUs.
It composes the existing tuned GPU ReduceScatter with the shared-host packet
AllGather techniques. The [measured combined operation](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/hybrid_ar_2x4l4_20261115/README.md)
improves median latency across the tested 1-MiB–1-GiB range. Default GPU policies
and the optional CPU-router dispatcher are unchanged.

## GPU reduction and handoff

The GPU phase uses the README ReduceScatter selection: serial local reduction
through 1 MiB, packed ring above 1 MiB through 16 MiB, flat reduction between
16 and 32 MiB, and direct ring from 32 MiB. Ring kernels
use 96 CTAs, 512 threads, and the existing 8-MiB target-piece policy. The result
is one fully reduced N/8-byte shard on each of eight GPUs, where N is the
AllReduce input/output size per GPU.

AllReduce remains in place. For packed ReduceScatter, the pack kernel reads the
current input directly from the output/data region into separate scratch.
Every rank completes packing before a timed MPI barrier allows peer receives
to overwrite that data region. This avoids an extra full-input copy and ensures
consecutive AllReduce calls consume the preceding call's result. The barrier
and GPU-stream completion before host gathering are included in timing.

## Eight-rank host gathering

Each MPI process owns one GPU. The two processes in each NUMA group upload
adjacent pieces into one pinned host packet. The even local rank owns that
packet and sends it to both remote NUMA-group leaders through MPI/UCX. Local
GPU consumers share mapped views of both local source packets, avoiding an
intermediate local CPU/MPI payload copy. Each GPU copies its own shard from
GPU memory and reads the other seven shards from host packets.

There are three slots. Each NUMA group allocates an own paired-source packet
and two remote paired-source packets per slot: 18C bytes/group, where C is
`--host-piece-bytes` (256 KiB by default). Pages are explicitly bound and first
touched on the owner's NUMA node, then registered for GPU access. The local
MPI shared window must use the unified memory model.

The progress loop checks upload events and MPI requests instead of waiting for
a future upload. In this version, leaders wait for all four local upload-ready
notifications before sending their paired packet. Local output consumption can
proceed while remote packets are in flight. Remote consumption starts only
after both remote packet receives complete. Packet tails use the current piece
length for the two adjacent sources; slot bases use the fixed capacity.

A process acknowledges a slot only after GPU consumption completes. A leader
also waits for both outgoing network requests. Every local process waits for
all three peer acknowledgments before reuse. Thus an odd-rank uploader cannot
overwrite a shared packet while its leader's HCA still reads it, and receive
buffers cannot be reused while the paired GPU consumer is still reading them.
Standard MPI receive posting protects cross-node slot reuse; this path does
not issue uncredited raw RDMA writes into reusable receive slots.

At completion all GPU reads and MPI sends using the prepared slots are done.
The explicit preparation/destruction boundary owns registration, streams,
events and the shared window. The ordinary GPU path constructs none of these
host-gather resources.

## Traffic tradeoff

This composition moves more inter-node payload than hierarchical AllReduce:
N/2 bytes/node for GPU ReduceScatter plus N bytes/node for host AllGather's
two remote destinations, compared with N bytes/node in the original hierarchy.
It replaces the local GPU AllGather with faster host-packet fan-out on this
PCIe topology. The measured end-to-end gain includes that tradeoff.

No CPU reduction, NCCL implementation dependency, or serving hook is added to
this path. The host engine depends on CUDA and MPI; MSCCL++ supplies the GPU
ReduceScatter and its peer/network progress.

## Small-message path

The [small-message regression fix](../benchmarks/data/ar_small_fix_2x4l4_20261115/README.md)
retains the single-CTA GPU AllReduce through 256 KiB when the explicit hybrid
option is selected. From 512 KiB upward the combined GPU-RS/host-AG operation
is unchanged. Small-only sweeps skip construction of the host-gather engine.
This is a fixed cutoff measured on the documented two-node FP32 L4 topology;
it does not enable an automatic application router. The previous all-hybrid
measurements remain valid for their recorded configuration.

## Run and validate

```bash
make multinode_allreduce multinode_cpu_router \
  BUILD=build/hybrid_ar_20261115 \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl
HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR="$PWD/build/hybrid_ar_20261115" \
  bash scripts/run_multinode_hybrid.sh --op allreduce \
  -b 1048576 -e 1073741824 -w 20 -n 10 --check-each 1 -c 1
```

Stage identical binaries/libraries at the same paths on both hosts first.
The hybrid runner selects MPI/UCX host transport; the ordinary GPU runner uses
MPI only for bootstrap. `--op allgather` isolates the eight-rank gather engine.
Host pieces must be 128-byte multiples from 128 bytes through 16 MiB; supported
full extents are 512-byte aligned, 1 KiB through 1 GiB. The hybrid GPU reduction
uses its fixed tuned launch settings. Graph capture/replay and CUDA-event
benchmark timing are not supported for this host-driven operation.

`--host-repeat-check 3` adds a correctness-only test of three consecutive
in-place AllReduce calls without refilling input. It checks the full output
against the expected repeated sum before ordinary timed calls. It covers the
serial, packed and direct-ring handoffs. The original single-call checker is
unchanged.

MPI/shared-memory setup was observed to leave a stale CUDA invalid-resource
handle probe status before the first kernel. The engine clears only that probe
status before checking its own launches. All owned CUDA operations are checked
at return, and other CUDA error statuses propagate. Diagnostic logs and
subsequent checked runs are retained in the measurement record.
