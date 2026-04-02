# Efficient host staging design for P2P communication on consumer GPUs without GPUDirect RDMA

## Overview

UCCL-Lite implements `ncclSend`/`ncclRecv` for consumer-grade GPUs that lack
GPUDirect RDMA but have direct PCIe peer-to-peer (P2P) capability.
The entire data path is **SM-free** — no GPU kernels are launched for data
movement.  All work is driven by the CPU via `cudaMemcpyAsync` and direct
InfiniBand verb posts, yielding lower latency and higher throughput than NCCL
on the same hardware.

### Target hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA L4 (Ada Lovelace, PCIe Gen4 ×16, compute 8.9) |
| NIC | 2× ConnectX-7 (mlx5), 400 Gb/s IB per port |
| Topology | No GPUDirect RDMA; PCIe P2P supported intra-node |

---

## Architecture

```
ncclSend(sendbuff, stream)          ncclRecv(recvbuff, stream)
        │                                    │
  ┌─────▼──────────────┐            ┌────────▼─────────────┐
  │ cudaEventRecord    │            │ cudaLaunchHostFunc   │
  │   (syncEvt, stream)│            │   (poll semaphore)   │
  │ cudaStreamWaitEvent│            │   on user stream     │
  │   (dedicated, sync)│            └──────────────────────┘
  └─────┬──────────────┘
        │
  ┌─────▼──────────────┐
  │ cudaMemcpyAsync    │  Intra: D2D via PCIe P2P
  │                    │  Inter: D2H to pinned staging
  └─────┬──────────────┘
        │
  ┌─────▼──────────────┐
  │ cudaEventRecord    │  completion event (per-WorkItem pool)
  │ cudaStreamQuery    │  flush CUDA command buffer
  └─────┬──────────────┘
        │ SPSC ring push
  ┌─────▼──────────────┐
  │  Worker Thread      │  Intra: signal shm semaphore
  │  (CPU, spinning)    │  Inter: ibv_post_send (RDMA write)
  └─────────────────────┘
```

### Key design principles

1. **Zero kernel launches** — `cudaMemcpyAsync` drives all data movement,
   avoiding SM occupancy and kernel launch overhead (~2-5 μs).
2. **CPU as control plane** — A dedicated worker thread per peer polls CUDA
   events and posts IB work requests directly, bypassing the GPU→FIFO→proxy
   chain that NCCL uses.
3. **Lock-free SPSC queue** — The main thread enqueues work items; the worker
   thread dequeues and executes.  No mutexes, no syscalls on the fast path.
4. **Per-WorkItem event pool** — A ring-buffered pool of CUDA events (2×
   queue capacity) prevents the re-recording deadlock that arises when
   nccl-tests pipelines hundreds of iterations without stream synchronization.
5. **Shared-memory address exchange** — Peer receive-buffer addresses are
   published via cache-line-aligned atomics in shared memory, eliminating
   per-iteration TCP round-trips.

---

## Intra-Node Path (CudaIpc)

For peers on the same node, UCCL-Lite uses CUDA IPC to map the remote GPU's
receive buffer into the sender's address space, then performs a direct
device-to-device `cudaMemcpyAsync` over PCIe P2P.

```
Sender (GPU 0)                        Receiver (GPU 1)
─────────────                         ──────────────
cudaEventRecord(syncEvt, userStream)
cudaStreamWaitEvent(ipcStream, syncEvt)
cudaMemcpyAsync(remoteDst, sendbuff,  ← CudaIpc mapped ptr
                D2D, ipcStream)
cudaEventRecord(completionEvt,
                ipcStream)
    │                                  cudaLaunchHostFunc(userStream):
    ▼ SPSC push                           poll shmSemaphore->wait()
Worker: poll completionEvt
        shmSemaphore->signal()  ───────▶  unblocks HostFunc
```

**Allocation-based IPC caching**: The full `cudaMalloc` allocation is
registered once via `cuMemGetAddressRange` + `cudaIpcGetMemHandle`.
Per-iteration buffer addresses are communicated through shared memory
(~10 ns), avoiding repeated IPC handle exchange over TCP (~35 μs).

---

## Inter-Node Path (IB RDMA, batched pipeline)

For peers on different nodes, data travels: GPU → pinned host staging
(D2H) → RDMA write to remote pinned host staging → remote GPU (H2D on
the receive side is handled by the receiver's own `cudaMemcpyAsync`).

### Small messages (≤ 256 KB)

Single D2H copy + single RDMA write.  The worker thread posts one
`ibv_post_send` as soon as the D2H completion event fires.

### Large messages (> 256 KB)

The transfer is split into adaptive batches:

```
┌──────────────────────────────────────────────┐
│ D2H stream:  [batch0 D2H] [batch1 D2H] ...  │
│                 │              │              │
│              event[0]      event[1]           │
│                 │              │              │
│ Worker:  wait──▶post WRs  wait──▶post WRs    │
│              (N chunks)      (N chunks)       │
└──────────────────────────────────────────────┘
```

- **Batch size** = `min(16, totalChunks/4)` × 256 KB (adapts to transfer size)
- **RDMA chunk size** = 256 KB per `ibv_post_send` WR
- **Progress counter**: After the last batch, a 64-bit counter is RDMA-written
  to the remote side, serving as a completion signal.
- **CQ drain**: Completions are polled every 64 signaled WRs to prevent SQ
  overflow.

### Receive side

The receiver first waits for data to arrive, then copies it to the GPU:

- **IPC (intra-node)**: The sender writes directly to `recvbuff` via CudaIpc.
  The receiver enqueues a `cudaLaunchHostFunc` on the user stream that
  spin-waits on the shm semaphore until the sender signals completion.
  No H2D copy is needed — data is already in GPU memory.

- **IB small message (inter-node)**: The receiver spin-waits on a host-memory
  semaphore (`h2hSemaphore->wait()`) until the sender's RDMA write completes,
  then enqueues a `cudaMemcpyAsync` (H2D) on the user stream.

- **IB large message (inter-node)**: The receiver polls a progress counter in
  the staging buffer for each batch.  As each batch arrives via RDMA, the
  receiver immediately enqueues an H2D `cudaMemcpyAsync` for that batch on a
  dedicated `h2dStream`, pipelining H2D copies with incoming RDMA writes.
  A final `cudaStreamWaitEvent` chains the H2D completions back to the user
  stream.

---

## Stream Dependency Correctness

The NCCL API contract requires that `ncclSend` reads `sendbuff` with respect
to the ordering of the caller's `stream` parameter.  Since UCCL-Lite uses
dedicated streams (ipcStream / d2hStream) for data movement, explicit
cross-stream synchronization is required:

```cpp
cudaEvent_t syncEvt = peerCtx.nextSyncEvent();
cudaEventRecord(syncEvt, userStream);        // capture user work
cudaStreamWaitEvent(dedicatedStream, syncEvt); // wait before reading sendbuff
```

A per-peer ring-buffered sync event pool (256 events) prevents re-recording
races when the application pipelines many iterations.

---

## Performance Comparison vs NCCL

Benchmarked with `nccl-tests sendrecv_perf`, 20 iterations, 10 warmup.
NCCL baseline is **tuned** with `NCCL_P2P_NET_CHUNKSIZE=2M` (best config found
via parameter sweep; default 128 KB leaves ~20% throughput on the table for
large messages).

### Intra-Node (2× L4, same machine, PCIe P2P)

| Size | UCCL-Lite (μs) | NCCL (μs) | UCCL-Lite BW | NCCL BW | Speedup |
|------|---------------:|----------:|-------------:|--------:|--------:|
| 8 B | 9.2 | 7.2 | — | — | 0.8× |
| 1 KB | 8.1 | 7.5 | 0.13 GB/s | 0.14 GB/s | 0.9× |
| 64 KB | 12.4 | 18.0 | 5.3 GB/s | 3.6 GB/s | **1.4×** |
| 1 MB | 60.3 | 79.5 | 17.4 GB/s | 13.2 GB/s | **1.3×** |
| 16 MB | 822 | 908 | 20.4 GB/s | 18.5 GB/s | **1.1×** |
| 256 MB | 13,014 | 14,287 | **20.6 GB/s** | **18.8 GB/s** | **1.10×** |

### Inter-Node (2× L4, two nodes, ConnectX-7 IB)

NCCL tuned with `NCCL_P2P_NET_CHUNKSIZE=2097152` (2 MB).

| Size | UCCL-Lite (μs) | NCCL tuned (μs) | UCCL-Lite BW | NCCL BW | Speedup |
|------|---------------:|----------------:|-------------:|--------:|--------:|
| 8 B | 13.2 | 20.4 | — | — | **1.55×** |
| 1 KB | 13.4 | 20.5 | 0.08 GB/s | 0.05 GB/s | **1.53×** |
| 32 KB | 15.9 | 24.3 | 2.1 GB/s | 1.3 GB/s | **1.53×** |
| 64 KB | 20.9 | 27.8 | 3.1 GB/s | 2.4 GB/s | **1.33×** |
| 256 KB | 43.1 | 50.2 | 6.1 GB/s | 5.2 GB/s | **1.17×** |
| 512 KB | 54.7 | 75.2 | 9.6 GB/s | 7.0 GB/s | **1.37×** |
| 1 MB | 81.1 | 103.5 | 12.9 GB/s | 10.1 GB/s | **1.28×** |
| 4 MB | 265.7 | 340.5 | 15.8 GB/s | 12.3 GB/s | **1.28×** |
| 16 MB | 1,019 | 1,148 | 16.5 GB/s | 14.6 GB/s | **1.13×** |
| 64 MB | 3,283 | 4,115 | 20.4 GB/s | 16.3 GB/s | **1.25×** |
| 256 MB | 12,327 | 15,860 | **21.8 GB/s** | **16.9 GB/s** | **1.29×** |

### NCCL Tuning Notes

A parameter sweep over NCCL environment variables found that
`NCCL_P2P_NET_CHUNKSIZE` is the dominant knob for host-staged IB throughput:

| NCCL_P2P_NET_CHUNKSIZE | 256 MB BW | vs default |
|------------------------|-----------|-----------|
| 128 KB (default) | 13.8 GB/s | baseline |
| 256 KB | 15.2 GB/s | +10% |
| 512 KB | 15.9 GB/s | +15% |
| 1 MB | 16.7 GB/s | +21% |
| **2 MB** | **17.0 GB/s** | **+23%** |
| 4 MB | 17.0 GB/s | +23% (1MB regresses) |

Other variables tested with negligible impact: `NCCL_BUFFSIZE` (4–64 MB),
`NCCL_NTHREADS` (256–512), `NCCL_PROTO=Simple`, `NCCL_IB_PCI_RELAXED_ORDERING`.
`NCCL_IB_QPS_PER_CONNECTION=2` caused hangs on our hardware.

### Summary

| Metric | Intra-Node | Inter-Node |
|--------|-----------|-----------|
| Small-message latency | 9 μs vs 7 μs (NCCL wins) | **13 μs vs 20 μs** (1.55× better) |
| Peak throughput | **20.6 vs 18.8 GB/s** (+10%) | **21.8 vs 16.9 GB/s** (+29%) |

---

## Why UCCL-Lite Is Faster

### Inter-node small-message latency (13 μs vs 20 μs)

NCCL's data path for a single `ncclSend`:

```
ncclSend ──▶ enqueue kernel params
         ──▶ kernel launch (~2 μs)
         ──▶ GPU writes FIFO command to BAR1 (~1 μs)
         ──▶ proxy thread polls FIFO (~1-2 μs)
         ──▶ Connection::write() → ibv_post_send
         ──▶ NIC DMA + network
```

UCCL-Lite's data path:

```
ncclSend ──▶ cudaMemcpyAsync D2H (~3 μs for small msg)
         ──▶ worker polls cudaEvent (~1 μs)
         ──▶ direct ibv_post_send (pre-built WR, ~0.5 μs)
         ──▶ NIC DMA + network
```

The key savings:
- **No kernel launch**: Eliminates ~2 μs of kernel launch + scheduling overhead.
- **No GPU→CPU signaling via BAR1/FIFO**: NCCL's proxy must poll a BAR1
  memory region written by the GPU kernel, adding ~1-2 μs of PCIe round-trip
  latency. UCCL-Lite uses `cudaEventQuery` which polls a host-visible flag
  set by the CUDA runtime's copy engine.
- **Direct QP access**: The worker holds a raw `ibv_qp` with pre-registered
  memory regions and pre-built scatter/gather lists. NCCL's proxy goes
  through a `Connection::write()` abstraction layer.

### Inter-node large-message throughput (21.8 vs 16.9 GB/s, tuned NCCL)

Both NCCL and UCCL-Lite pipeline host-staged RDMA transfers.  The throughput
gap comes from **per-slice overhead** — UCCL-Lite's pipeline stages are
individually much cheaper.

#### NCCL's pipeline architecture

NCCL uses an 8-slot ring buffer (`NCCL_STEPS = 8`) as its pipeline mechanism.
With the tuned `NCCL_P2P_NET_CHUNKSIZE=2M`, NCCL auto-expands the staging
buffer to ~16 MB (8 slots × 2 MB each):

```
GPU kernel ──SM copy──▶ staging[slot] ──FIFO write──▶ proxy polls ──▶ isend()
     ▲                                  (BAR1)        (BAR1 read)     (ncclNet)
     │                                                                    │
     └─── proxy writes head counter ◀── proxy polls test() ◀── CQ ◀──────┘
```

For a 256 MB transfer: `256 MB / 2 MB = 128 slices`, cycling through 8 slots.
Each slot is reused 16 times.  The pipeline depth is 8: up to 8 slices can be
in flight simultaneously (GPU copying slot N while NIC transmits slot N-3, etc.).

#### UCCL-Lite's pipeline architecture

UCCL-Lite uses a 256-slot SPSC ring buffer.  Data is split into batches of up
to `16 × 256 KB = 4 MB`, with each batch producing one CUDA event and multiple
RDMA work requests:

```
cudaMemcpyAsync ──DMA engine──▶ staging ──eventRecord──▶ worker polls
                                                         eventQuery
     ▲                                                       │
     │                                                  ibv_post_send
     │                                                  (direct, pre-built WR)
     └──── CQ drain every 64 WRs ◀── ibv_poll_cq ◀──────────┘
```

For a 256 MB transfer: `256 MB / (16 × 256 KB) = 64 batches`, each batch
producing 16 RDMA WRs of 256 KB.

```
Time ──▶
D2H:   [batch0  4MB] [batch1  4MB] [batch2  4MB] ...
RDMA:        [batch0  16×WR] [batch1  16×WR] ...
                          ▲ overlap ▲
```

#### Per-slice overhead comparison

The fundamental reason for the throughput gap is that each pipeline stage
carries fixed overhead, and UCCL-Lite's overhead per stage is 2-3× lower:

| Pipeline stage | NCCL | UCCL-Lite | Why |
|----------------|------|-----------|-----|
| **D2H copy** | GPU SM kernel copies to staging | `cudaMemcpyAsync` (DMA engine) | DMA engine is dedicated HW, zero SM usage |
| **GPU→CPU signal** | Kernel writes `connFifo[slot]` to BAR1 | `cudaEventRecord` + `cudaEventQuery` | BAR1 read costs ~1-2 μs PCIe round-trip; event query reads host memory (~0.3 μs) |
| **RDMA post** | `ncclNet->isend()` → net plugin → `ibv_post_send` | Direct `ibv_post_send` with pre-built WR | No abstraction layers, pre-staged scatter/gather lists |
| **Completion check** | `ncclNet->test()` per slot, every iteration | Signal every 64th WR, batch `ibv_poll_cq` | 64× fewer CQ polls for same data volume |
| **Proxy scheduling** | Shared proxy thread round-robins all connections | Dedicated worker thread per peer | Zero contention, no scheduling jitter |

Estimated per-stage overhead:

| | NCCL (tuned, 2 MB chunks) | UCCL-Lite |
|---|------|-----------|
| Per-slice overhead | ~8-12 μs | ~3-5 μs |
| Typical slice size | 2 MB | 4 MB (batched) |
| Slices for 256 MB | 128 | 64 |
| **Total overhead** | **~1.0-1.5 ms** | **~0.2-0.3 ms** |

For 256 MB, UCCL-Lite's larger batch size (4 MB vs 2 MB) halves the number of
pipeline stages, and each stage is 2-3× cheaper.  Combined, this yields a ~5×
reduction in total pipeline overhead.  Note that with NCCL's default 128 KB
chunk size, the gap is even larger (512 slices, ~4-6 ms overhead).

#### Why the DMA engine vs SM copy matters

NCCL's GPU kernel uses SM threads (with `LL128` or `Simple` protocols) to
copy data from the user buffer to the host-pinned staging buffer.  This has
three costs:

1. **SM occupancy**: The copy kernel occupies SMs for the entire transfer
   duration, competing with compute workloads.
2. **Kernel execution overhead**: Each kernel launch has ~2 μs fixed cost
   (parameter setup, grid scheduling, warp dispatch).
3. **Bandwidth**: SM-driven copies through PCIe achieve slightly lower
   sustained throughput than the dedicated DMA copy engine, because the
   SMs must issue load/store instructions rather than using HW DMA.

UCCL-Lite's `cudaMemcpyAsync` routes through the GPU's PCIe copy engine —
a dedicated hardware unit that runs independently of the SM array.  Zero SM
usage, zero kernel launch, and marginally higher sustained bandwidth.

#### Why BAR1 signaling is expensive

NCCL's GPU kernel signals the proxy thread by writing a `connFifo` entry
to a BAR1-mapped memory region.  The proxy thread must **read** this region
to detect new work.  Each BAR1 read is an uncached PCIe transaction that
requires a full round-trip (~1-2 μs on PCIe Gen4).  With 128 slices for a
256 MB transfer (tuned), BAR1 polling alone adds ~0.1-0.3 ms (with the
default 512 slices, it would be ~0.5-1 ms).

UCCL-Lite avoids BAR1 entirely.  `cudaEventRecord` writes a host-visible
flag when the DMA engine completes, and `cudaEventQuery` reads this flag
from ordinary host memory — a ~0.3 μs operation with no PCIe traffic.

#### Pipeline depth: 8 vs 256

NCCL's 8-slot ring buffer means at most 8 slices can be in flight.  If any
stage stalls (e.g., slow CQ completion, proxy thread preempted), the pipeline
drains quickly.  UCCL-Lite's 256-slot SPSC queue provides 32× more pipeline
depth, absorbing transient stalls without pipeline bubbles.

In practice, the effective pipeline depth is bounded by the NIC's send queue
depth and available staging memory, not the ring buffer size.  But the larger
buffer provides headroom for burst scheduling without back-pressure.

### Intra-node throughput (20.6 vs 18.8 GB/s)

NCCL uses GPU kernels (SM-based copy) for intra-node P2P, which competes
with compute workloads for SM resources. UCCL-Lite uses `cudaMemcpyAsync`
device-to-device, which routes through the PCIe P2P copy engine — zero SM
usage and slightly higher sustained bandwidth for bulk transfers.

### Intra-node small-message latency (9 μs vs 7 μs — NCCL wins)

For very small intra-node transfers, NCCL's GPU kernel approach has lower
latency because:
- The GPU kernel is already on the execution stream — no cross-stream
  synchronization needed.
- UCCL-Lite pays ~1 μs for `cudaEventRecord` + `cudaStreamWaitEvent` to
  synchronize between the user stream and the dedicated IPC stream.
- UCCL-Lite's `cudaLaunchHostFunc` on the receive side has higher overhead
  than NCCL's GPU-side flag polling for small messages.

This is an acceptable trade-off: intra-node small-message latency is less
critical than inter-node latency in multi-node training, and the SM-free
design preserves GPU compute capacity.

---

## Implementation Details

### Shared-Memory Semaphore (ShmSemaphore)

Each send-recv peer pair shares a POSIX shared-memory block with cache-line-
aligned atomic fields:

```cpp
struct ShmBlock {
  alignas(64) atomic<uint64_t> counter;          // signal/wait semaphore
  alignas(64) atomic<uint64_t> recvbuffAddr;     // per-iteration recv ptr
  alignas(64) atomic<uint64_t> addrGeneration;   // bumped each publish
  alignas(64) atomic<uint64_t> allocBase;        // cudaMalloc allocation base
  alignas(64) atomic<uint64_t> allocGeneration;  // bumped on new allocation
};
```

Memory ordering: the receiver writes all fields with `relaxed` stores, then
does a `release` store on `addrGeneration`. The sender does an `acquire` load
on `addrGeneration`, then `relaxed` loads on everything else.

### SPSC Ring Buffer

A fixed-size (256-slot) lock-free single-producer single-consumer ring buffer
connects the main thread to the worker thread:

```cpp
WorkItem ring[256];
alignas(64) atomic<uint32_t> head;  // producer
alignas(64) atomic<uint32_t> tail;  // consumer
```

Back-pressure: `push()` spins when `head - tail >= 256`. This naturally
throttles the producer if the worker falls behind.

### Per-WorkItem Event Pool

To avoid the CUDA event re-recording deadlock (where `nccl-tests` pipelines
iterations faster than the worker can process them), each `SendRecvWorkerState`
maintains a ring-buffered pool of `512` CUDA events (2× queue capacity).
Each work item gets a unique event index via `allocEvent()`, ensuring no event
is polled by the worker while being re-recorded by the producer.

---

## Limitations and Future Work

- **Single QP per peer**: Each peer pair uses one IB QP. Multiple QPs or
  multi-path could improve large-message throughput.
- **Intra-node small-message latency**: The cross-stream event sync adds
  ~2 μs overhead vs NCCL's in-stream kernel approach. A future optimization
  could use a lightweight in-stream signaling mechanism.
- **Static pipeline parameters**: The RDMA chunk size (256 KB) and D2H batch
  multiplier (16) are compile-time constants. An auto-tuning system that
  calibrates these at init time based on measured D2H/RDMA bandwidths and
  per-batch overhead would be optimal across different hardware platforms.
- **RDMA write-with-immediate**: Combining the data write and completion
  signal into a single RDMA operation could reduce small-message latency
  by ~1-2 μs.
