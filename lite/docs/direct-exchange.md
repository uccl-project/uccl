# General direct-exchange executor

`make direct-exchange direct-exchange-check` builds an experimental FP32/BF16
executor and its correctness client. The public interface is
`include/uccl_lite/collectives/direct_exchange.hpp`; its implementation lives in
`src/collectives/direct_exchange.cu`. The static archive also exposes
`runtime::directExchangeBackend(dtype)` through the common `CollectiveBackend`
interface. Its standalone catalog contains those two typed experimental
candidates; the strict native-free catalog remains separate.

This removes a fixed host/GPU partition from the execution algorithm. It now
participates in [distributed candidate selection](network-candidate-selection.md)
through the common prepared runtime and native-enabled ABI. Exact-workload
qualification remains mandatory. The strict native-free profile catalog is
separate and has not been broadened by this integration.

## Participants and routes

Construction accepts a bootstrap ID, logical rank/count, current CUDA device,
application capacity and dtype. It allocates no GPU buffers or transports.
Collective `prepare()` discovers boot identities, GPU UUIDs and the visible
CUDA inventory. It checks unique process/GPU ownership and matching capacity,
dtype and chunk configuration. Capacity bounds are 2–32 ranks and at most
1 GiB of application extent per rank; they are not hardware validation claims.

Each pair uses CUDA IPC when both ranks discover same-host peer access.
Otherwise it requires an active IB device plus supported GPU registration and
RDMA write visibility. UUID matching is independent of the CUDA ordinal seen
by another process. Missing peer visibility can therefore use IB, including
between GPUs on the same host, rather than inventing a peer mapping.

NIC assignment first uses the existing GPU–NIC/shared-resource bandwidth
planner on the participating local inventory. When physical path bounds are
incomplete, explicit experimental construction may use an active provisional
rail. Unknown rates remain unknown, and NIC advertised rates remain advertised.
That provisional transport choice is not algorithm admission or evidence of a
performance gain. CPU affinity and NUMA memory policy are restored after setup.

The common backend descriptor now supports host-count, total-rank and
per-host-rank bounds independently of fixed `ranksPerHost` lists. These bounds
admit a *layout for further checking*, never a transport or performance result.
Physical ownership/rank order still comes from discovery.

## Data and completion contract

There is one bounded send chunk and one receive slot per logical source.
Each source is the only writer to its slot. AllGather exchanges raw bits;
AllReduce and ReduceScatter accumulate FP32 values in increasing logical
source-rank order, rounding BF16 only once at the output. This preserves a
specified source order even when logical ranks are interleaved across hosts.

All three operations support positive scalar counts, scalar alignment, tails
and legal NCCL-style in-place layouts. Partial overlaps reject collectively.
ReduceScatter sends each destination's corresponding input shard. Application
capacity covers the complete AllGather output, ReduceScatter input or AllReduce
tensor. The private workspace is `(ranks + 1) * chunkBytes`, bounded separately.

Enqueue is host-synchronous and serialized. Input production must be ordered
on the supplied CUDA stream. Stream completion precedes host-issued transport
writes; sender flush and CPU agreement precede receiver visibility/consumption.
A second agreement follows receiver stream completion and protects slot reuse.
Only the caller stream is drained by enqueue; `synchronize()` does not wait on
unrelated application streams. Captured calls reject before submitting work;
owned/external graphs are not implemented for this executor.

Transport setup and registration occur only at prepare. This executor has no
NCCL, MPI or CPU-router dependency. The test program uses MPI only for launch
and ID distribution. Operational CUDA/transport failures are errors; this is
not a fault-recovery implementation and does not switch transports at enqueue.

## Correctness evidence

The checker verifies every input/output/guard word, raw AllGather payloads,
source-order cancellation cases, scalar tails, chunk boundaries, legal aliases,
repeated generations, alternating caller streams and two communicator
lifetimes. It also checks rejected capture/count/alias calls, descriptor
mismatch, placement restoration and the common backend adapter.

An initial run using pageable `cudaMemcpy` input initialization on the default
stream observed one previous-generation BF16 source in a 3+1 in-place
AllGather. That harness had no producer-to-consumer dependency on its separate
nonblocking stream: pageable H2D calls can return before DMA completion.
Input production now uses `cudaMemcpyAsync` on the consumer stream, with the
same full-buffer/source-order oracle. No tolerance or transport barrier was
relaxed. The original observation is retained in `build/direct-exchange-final/`;
this missing dependency is established, while a single prior observation does
not independently prove its complete causal history.
([CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html))

This stage reports correctness, not a throughput or latency improvement. The
CPU rendezvous and serial host dispatch are deliberate limitations of this
first general executor. Persistent direct evidence and fuller cost modeling remain in the
topology-runtime migration scope.

Final evidence is in `build/direct-exchange-validated/`: 11 runs cover four-L4
local FP32/BF16; two physical hosts with 1+1, 2+2, 3+3, 4+4 and unequal 3+1
GPU layouts; isolated one-GPU visibility per process; and four-rank memcheck
for both dtypes. All pass, including the common factory/catalog and deferred
destruction checks. Memcheck reports zero device errors; only known MSCCL++
capability-probe API reports are suppressed. Tests use 4-MiB application capacity
(1 MiB under memcheck), 64-KiB explicit chunks and the factory's 256-KiB default,
three input generations and two lifetimes. Hardware is the dual-NUMA Xeon Silver
4410Y/L4 configuration. More than two physical hosts/eight GPUs is not validated.
The baseline is an independent CPU reference, not a performance competitor;
there is no timed benchmark or best-variant curve in this correctness report.
Current primary-runtime native/capture regressions are separately retained in
`build/direct-exchange-package-verified/`. Source and binary fingerprints are
in `build/direct-exchange-artifacts.sha256`.

The executor and [direct cost model](direct-exchange-model.md) now share the
sender schedule. Prepared route traffic is available through the backend and
C report interfaces; model predictions use [compatible primitive calibration](direct-model-calibration.md)
when resource probing is enabled.
