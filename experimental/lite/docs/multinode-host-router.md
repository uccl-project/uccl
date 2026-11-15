# NUMA host-router AllGather

This is an explicit experimental data path for two hosts with four L4 GPUs each,
FP32, and two NUMA groups per host. Each group process owns two GPUs. It extends
`multinode_host_allgather.cu`; the independently linked CPU-router core and its
production dispatcher are unchanged.

## Data path

For full output N bytes/GPU, each GPU contributes N/8. Two D2H transfers write
adjacent pieces directly into their NUMA group's pinned host packet; there is no
separate CPU concatenation copy. Each group sends its packet to both remote NUMA
groups through its local HCA. Each destination GPU reconstructs standard rank
order from the four groups' packets.

Three slots bound upload, network transfer, and GPU consumption. The original
step schedule is retained. `--progress ready` instead tests completion and
advances whichever work is ready, without blocking consumption of old packets
on the upload of a future packet. Network sends and host relays retain logical
piece order, because multiple pieces share MPI tags.

The useful new combination is:

```text
--local-input self --local-exchange shared --early-consume 1
--progress ready --gpu-workers 0 --upload dma --download sm -k 262144
```

Its changes are:

- The GPU copies its own shard from its original GPU input, avoiding a redundant
  host-to-device copy. It still uploads that shard for other consumers.
- Source packets live in MPI shared windows. The other local NUMA group maps
  those same pages, so its GPUs read the source packet without an intermediate
  CPU/MPI payload copy. Source pages are NUMA-bound and first-touched by their
  owner before peer CUDA registration. A unified window model is required.
- Local and remote halves of the output can be consumed separately. The local
  half starts after the local packet is ready, while remote transfers continue.
- Ready-driven progress overlaps the stages rather than imposing a fixed
  `upload(j), send(j-1), consume(j-2)` sequence.

Shared exchange trades more cross-NUMA GPU reads for eliminating a CPU copy and
its transfer/coordination cost. It does not assume those reads are free. Network
payload remains N bytes per node per direction (each group's data is sent to both
remote groups). Each GPU reads 7N/8 from host for `local-input self`, plus a local
GPU copy of N/8. This is a different traffic distribution from GPU P2P AllGather.

## Lifetime protocol

A source announces a shared packet only after both D2H producers complete.
The peer acknowledges only after **both** GPU consumers finish that slot.
Sender retirement requires its own GPU consumers, remote sends, the peer's
acknowledgment, and the readiness/acknowledgment messages all to complete.
Ready and acknowledgment tags distinguish slots, so out-of-order GPU completion
cannot acknowledge the wrong slot. A slot's metadata and payload cannot change
while any worker, NIC, or GPU reader still uses them.

Optional GPU workers keep one CUDA context and a dedicated NUMA CPU per GPU.
The main thread remains the sole MPI caller. Release/acquire epoch counters
publish requests and completion. With split consumption, workers process both
phase sequence numbers even when the second request arrives before the worker
observes the first. Exceptions propagate to the main thread. Workers stop before
CUDA events and mapped memory are destroyed.

## Retained alternatives

- `--network paired`: one inter-node copy, followed by a host relay to the other
  destination NUMA group. It halves network bytes but adds a dependent local
  host transfer; exploratory measurements did not favor it.
- `--local-input pair`: read both same-NUMA GPU inputs directly. Peer access is
  explicitly checked/prepared. This is a hybrid path, not pure host consumption.
- `--upload sm`: a GPU producer writes host packets and can simultaneously fill
  its own output shard. System visibility is established before upload completion.
- `--download dma`: use CUDA memcpy operations instead of mapped-host SM reads.
- `--gpu-workers 1`, `--lean-context 1`, and `--consume-threads 128|256|512` remain
  explicit tuning controls; none is selected automatically.

## Optional direct host RDMA

`make multinode_host_rdma_allgather` adds `--network rdma`. The MPI-only target
keeps its existing dependencies. This optional executable uses MSCCL++ CPU IB
endpoints to register host buffers and write directly to remote host packets.
It uses no GPU RDMA registration and no GPU atomics.

The receiver grants a slot credit before upload begins. The sender writes data
only after the matching credit and then publishes completion with native
**host-memory** RDMA atomics. This uses the runtime's ordered atomic notification
path, including for memory regions that permit relaxed ordering. Source queues
are drained before any packet or token is reused. Both remote destinations have
independent credit and completion counters. MPI remains responsible for bootstrap
and the local-group control path.

## Measurement scope

The comparison uses full-output bytes/GPU and bus bandwidth `7N/(8t)`, with
slowest-group host enqueue-to-completion timing. The two GPU streams in each of
four MPI processes are both completed before return. The GPU-only baseline uses
eight MPI processes, so this is an end-to-end implementation comparison. NCCL
controls inside the host harness use the same four-process/two-GPU ownership.

Before comparisons, NCCL receives checked, untimed priming calls. A diagnostic
showed the first host run after unprimed NCCL setup was slower than subsequent
runs in the same process. Priming removes that first-size state effect; it does
not move collective work out of the reported timed calls. Per-backend warmups
and complete-output checks are retained.

This implementation is eager-only. It is not wired into automatic dispatch or
serving hooks. Any integration must preserve explicit preparation and match
process/GPU placement, NUMA/HCA mapping, dtype, exact size, software identity,
and validated timings before selecting a route. Core graph regression tests do
not establish graph support for this MPI-driven host data path.

See [results and reproduction commands](../benchmarks/data/ag_host_router_2x4l4_20261115/README.md).
