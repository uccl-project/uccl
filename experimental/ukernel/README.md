# UKernel

Minimal build and test entry points for `experimental/ukernel`.

## Prerequisites

- CUDA / ROCm toolchain for NVIDIA / AMD builds
- RDMA (libibverbs, libnl) dependencies used by `transport` and `ccl`
- system-installed GDRCopy (`gdrapi.h` + `libgdrapi`) for `device` and `ccl` (NVIDIA only)
- for the Python bindings: `torch` and `nanobind` (`pip install nanobind`)

## Install GDRCopy

`experimental/ukernel` no longer builds `thirdparty/gdrcopy`.
Install GDRCopy from NVIDIA upstream first.

### System-wide install (requires sudo)

```bash
sudo apt-get update
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) libelf-dev
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make CUDA=/usr/local/cuda
sudo make CUDA=/usr/local/cuda prefix=/usr/local install
```

### Custom prefix install (no sudo required)

If you don't have root access or want to keep GDRCopy in a local directory:

```bash
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make CUDA=/usr/local/cuda
mkdir -p /home/$USER/gdrcopy
make CUDA=/usr/local/cuda prefix=/home/$USER/gdrcopy install
```

Then pass the custom paths when building ukernel:

```bash
cd experimental/ukernel
make -f Makefile \
    GDRCOPY_INCLUDEDIR=/home/$USER/gdrcopy/include \
    GDRCOPY_LIBDIR=/home/$USER/gdrcopy/lib
```

The same `GDRCOPY_INCLUDEDIR` and `GDRCOPY_LIBDIR` variables work for
per-module builds as well:

```bash
make -C src/device GDRCOPY_INCLUDEDIR=/path/to/include GDRCOPY_LIBDIR=/path/to/lib
make -C src/ccl   GDRCOPY_INCLUDEDIR=/path/to/include GDRCOPY_LIBDIR=/path/to/lib
```

## Build

NVIDIA:

```bash
cd experimental/ukernel
make clean -f Makefile
make -j$(nproc) -f Makefile
```

AMD / ROCm:

```bash
cd experimental/ukernel
make clean -f Makefile.rocm
make -j$(nproc) -f Makefile.rocm
```

Common overrides:

```bash
make -f Makefile CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80 \
    GDRCOPY_INCLUDEDIR=/usr/include GDRCOPY_LIBDIR=/usr/local/lib
```

All path variables support environment variable override. When working on a
machine with non-standard paths, export them once in your shell profile
instead of passing them on every command:

```bash
# Example: custom CUDA and GDRCopy installed to user home
export CUDA_HOME=/usr/local/cuda-13
export GDRCOPY_INCLUDEDIR=$HOME/gdrcopy_install/include
export GDRCOPY_LIBDIR=$HOME/gdrcopy_install/lib
```

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_HOME` / `CUDA_PATH` | `/usr/local/cuda` | CUDA toolkit root (auto-detected from `/usr/local/cuda-*` if missing) |
| `GDRCOPY_INCLUDEDIR` | `/usr/include` | Path to `gdrapi.h` |
| `GDRCOPY_LIBDIR` | (empty, system default) | Path to `libgdrapi.so` |
| `CONDA_LIB_HOME` | `/usr/lib` | Library search path for system libs |
| `SM` | `80 86 89 100` | GPU compute capability |

## NCCL Compatibility

`include/nccl.h` + `src/ccl/nccl.cc` implement a source-level drop-in for the
NCCL C API.  Build `libnccl.so` and install into a standard `include/` +
`lib/` layout:

```bash
cd experimental/ukernel
make nccl
# → build/nccl/include/nccl.h  +  build/nccl/lib/libnccl.so

# Custom install prefix:
make nccl NCCL_PREFIX=/opt/ukernel_nccl
```

Run standard [nccl-tests](https://github.com/NVIDIA/nccl-tests) via
`LD_LIBRARY_PATH`:

```bash
# Build nccl-tests (MPI required for multi-rank)
cd thirdparty/nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda \
    MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi \
    NCCL_HOME=$(pwd)/../../experimental/ukernel/build/nccl \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90" \
    -j$(nproc)

# Multi-rank, same node (2 GPUs, one process per rank — exercises
# ncclGetUniqueId + ncclCommInitRank). Result validation is ON by
# default; keep -c 1 (do NOT pass -c 0, or the run measures nothing —
# and a 1-rank run without mpirun measures nothing either, since
# single-rank collectives are no-ops).
cd ../../experimental/ukernel
UK_CCL_DEBUG=2 CUDA_VISIBLE_DEVICES=6,7 mpirun -np 2 -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    -x CUDA_VISIBLE_DEVICES \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -c 1

# Single process, multiple GPUs (exercises ncclCommInitAll, which
# initializes one communicator per device on its own thread):
LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -g 2

# Multi-node: rank 0 packs its NIC IP into the unique ID and peers
# connect to it. On multi-homed hosts pick the interface via
# NCCL_SOCKET_IFNAME (prefix match, e.g. "eth" matches eth0/eth1);
# a warning is printed if it matches no interface.
mpirun -np 2 -H node0,node1 \
    -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    -x NCCL_SOCKET_IFNAME=eth0 \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -g 1
```

Supported APIs: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitAll`,
`ncclAllReduce` (ring + opt-in binary tree), `ncclAllGather`,
`ncclReduceScatter`, `ncclAllToAll` (in-place only), `ncclBarrier`,
`ncclCommDestroy`, `ncclCommAbort`, `ncclCommFinalize`,
`ncclCommGetAsyncError`, `ncclGetErrorString`, `ncclGetVersion`.

In-place semantics match NCCL: AllReduce supports both placements;
AllGather / ReduceScatter detect NCCL's in-place form (sendbuff pointing
inside recvbuff, and vice versa) and run the in-place algorithm variant;
AllToAll requires in-place (`sendbuff == recvbuff`). Unsupported:
`ncclBroadcast`, `ncclReduce`, `ncclSend`, `ncclRecv` return
`ncclInvalidUsage` — of the stock nccl-tests binaries, only
`all_reduce_perf` (both placements), `all_gather_perf` and
`reduce_scatter_perf` pass; `broadcast_perf` / `reduce_perf` /
`alltoall_perf` / `sendrecv_perf` fail by design (those APIs are not
implemented).

Binary-tree AllReduce is opt-in via `UK_CCL_TREE_THRESHOLD_BYTES`
(default 0 = never). With `nranks == 2` the tree degenerates to the
ring's shape, so the crossover can only be calibrated on a larger-rank
environment.

## Test

Each layer has its own suite; the per-layer READMEs carry the full test
lists and two-process run commands:

- [src/transport/README.md](src/transport/README.md) — transport adapter
  integration tests and suite runner
- [src/device/README.md](src/device/README.md) — device kernel tests
- [src/ccl/README.md](src/ccl/README.md) — CCL unit tests, per-backend
  e2e tests, same-node and cross-node runs, debug switches

Top-level shortcuts:

```bash
cd experimental/ukernel
make transport_test     # transport adapter tests
make device_test SM=80  # device kernel tests
make ccl_test SM=80     # CCL unit + integration build
make transport_suite    # transport integration suite
```

Manual two-process transport check (IPC requires both GPUs visible to
both processes; the test defaults to server `--gpu=0`, client `--gpu=1`,
override with `--gpu`/`--peer-gpu`):

```bash
cd experimental/ukernel/src/transport
make test-integration
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_integration communicator --role=server --case=exchange --transport ipc --exchanger-port 16979
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_integration communicator --role=client --case=exchange --transport ipc --exchanger-ip 127.0.0.1 --exchanger-port 16979
```

Benchmarks live in [benchmarks/](benchmarks/) — see its README.

## Python bindings

`experimental/ukernel/py` contains two `torch`-based extensions built
with nanobind over the same `transport + ccl + device` stack:

- `ukernel_ccl` — collectives behind a persistent `ProcessGroup`:
  `allreduce` (ring), `alltoall` (equal-split, in place), `barrier`,
  plus an async API (`allreduce_submit` / `alltoall_submit` → handle,
  with `poll` / `wait` / `status` / `error_message` / `release`) for
  compute/communication overlap. Peer setup and memory registration
  (`prepare()`) are cached per collective shape + pointer set, so
  steady-state calls are submit + wait. Optional
  `signal_group_tiles` aggregates one signal per G tiles (2–4 is usually
  best for small messages).
- `ukernel_p2p` — point-to-point `Communicator`: peer connect/accept,
  `reg_ipc` / `reg_rdma` buffer registration, `send` / `signal` /
  `wait_data` blocking helpers, and async
  `send_put_async` / `send_signal_async` / `wait_signal_async` + `poll`.

Build both extensions in place:

```bash
cd experimental/ukernel/py
python setup.py build_ext --inplace
```

Run tests (2+ GPUs; explicit ports avoid collisions). See
[py/README.md](py/README.md) for the full test and benchmark reference.

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=16998 python tests/test_collective.py &
CUDA_VISIBLE_DEVICES=6,7 RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=16998 python tests/test_collective.py &

# or via torchrun
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_collective.py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_p2p.py
```

Minimal usage under `torchrun`:

```python
import os
import torch
from ukernel_ccl import ProcessGroup

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", rank))

torch.cuda.set_device(local_rank)
pg = ProcessGroup(
    rank=rank,
    world_size=world,
    gpu_id=local_rank,
    exchanger_ip=os.environ.get("MASTER_ADDR", "127.0.0.1"),
    exchanger_port=int(os.environ.get("MASTER_PORT", "16998")),
)

x = torch.randn(1024 * world, device="cuda", dtype=torch.float32)
pg.allreduce(x, tile_bytes=65536)

y = torch.randn(1024 * world, device="cuda", dtype=torch.float32)
pg.alltoall(y, tile_bytes=65536)  # inplace, equal split

send = torch.randn(13, device="cuda", dtype=torch.float32)
recv = torch.empty(13, device="cuda", dtype=torch.float32)
pg.alltoallv(recv, send,
             output_split_sizes=[4, 5, 4],
             input_split_sizes=[4, 5, 4],
             tile_bytes=65536)
```

Current constraints:

- payload tensors must be CUDA, on the process group's GPU, and
  contiguous (collectives run in place; non-contiguous input is
  rejected loudly)
- element counts must be divisible by `world_size` for
  `allreduce` / equal-split `alltoall`
- `alltoallv` takes separate output/input tensors with explicit split
  sizes (elements); `input_split_sizes[rank]` must equal
  `output_split_sizes[rank]`
- scratch buffer is managed internally by the executor

## Modules

- [src/transport](src/transport/) — RDMA/IPC/TCP transport adapters, Communicator, OOB exchange
- [src/device](src/device/) — GPU SM persistent kernels, FIFO task manager, GDRCopy ops
- [src/ccl](src/ccl/) — collective planner/lower, SprayExecutor, backends
- [py](py/) — `ukernel_ccl` / `ukernel_p2p` Python bindings
- [benchmarks](benchmarks/) — transport and device benchmarks
- [include/util](include/util/) — shared lock-free jring, pause intrinsic

## Performance measurements

Two-GPU AllReduce, A40 pair (CUDA 6,7; P2P over PCIe gen4). All numbers
are true 2-rank runs — the shim rows are `mpirun -np 2` over
nccl-tests, the spray row is the native SprayExecutor benchmark
(`test_perf_spray_allreduce`). A single-process `-g 1` nccl-tests run
is a 1-rank no-op (launch overhead only) and must NOT be used as a
performance number.

| Size | shim nccl-tests | native NCCL 2.29.7 |
|---|---|---|
| 256KB | 95us | 25.6us |
| 1MB | 181us | 60.0us |
| 4MB | 447us | 140.4us |
| 16MB | 1686us | 454.4us |
| 64MB | 2012us | 1691.8us |
| 256MB | 5462us | 6352.8us |

256MB AllReduce now beats native NCCL (5.5ms / 49 GB/s vs 6.35ms /
42 GB/s); AllGather is at parity (3.8ms); ReduceScatter lags
(8.3ms vs 3.8ms native) and is the next target. The jump came from
routing same-host puts over IPC (the latency-based path balancer was
misrouting onto the device/RDMA paths), 256-thread device blocks, and
larger tiles for large messages.

> **IMPORTANT — run mpirun with CPU binding disabled.** OpenMPI's
> default hwloc CPU binding pins shim processes to a subset of cores,
> which starves the GPU-completion wait path (drain-thread polling /
> WaitValue detection / persistent-kernel wakeup). On this machine it
> inflated small-message AllReduce latency ~50x (256KB: 9.4ms → 167us).
> Native NCCL is unaffected (hardware signals, no CPU polling). Always
> launch the shim under:
>
> ```bash
> mpirun -np 2 --mca plm_rsh_agent sh --mca hwloc_base_binding_policy none \
>     -x LD_LIBRARY_PATH ... -x CUDA_VISIBLE_DEVICES ...
> ```
> (`OMPI_MCA_hwloc_base_binding_policy=none` env var is equivalent;
> `--mca plm_rsh_agent sh` makes mpirun fork locally without ssh.)

Two known gaps, both under investigation:

1. **Shim small-message latency ~8x vs native** (was ~50x before the
   MPI binding fix): at 256KB the shim path is 167.5us vs native NCCL
   21.8us (7.7x). The remaining gap narrows with size — 1MB: 419us vs
   60.7us (6.9x), 4MB: 1254us vs 141.6us (8.9x) — and vanishes in the
   bandwidth-saturated regime (64MB+: ~20-25ms both).

   Measured decomposition (probe programs, A40 pair, true 2-rank):
   each shim `ncclAllReduce` call = CPU submit ~20us + GPU stream-wait
   ~149us (native NCCL: ~15us + ~20us). The GPU-side 129us gap is
   entirely inside the collective's own execution — a CUDA event
   enqueued right after the call completes at the same time as a full
   stream sync, so there is no extra done-flag / WaitValue-release
   delay. Fixed-latency floor is data-size-independent: AllToAll of 64B
   costs ~50us, 64KB ~55us; AllReduce 64KB ~115us (the extra ~60us is
   the reduce path's extra synchronization rounds). Signal matching
   itself is fast (0.1-0.4us/wait in `test_signal_backend_e2e`), and
   the floor is identical for forced `UK_CCL_PUT_PATH=device|ipc`, so
   it is not a per-path issue. Remaining suspects: per-collective GPU
   pipeline latency (kernel-launch sequence + cross-rank signal
   round-trips) versus native NCCL's ~20us end-to-end.

2. **Same-node P2P bandwidth ~5 GB/s busbw**: far below the PCIe
   gen4 P2P ceiling (~30 GB/s). At 256MB both the spray path and the
   shim path saturate at ~5.1-5.2 GB/s busbw, vs native NCCL 43 GB/s
   (256MB in 6.2ms vs ~103ms). The IPC/device P2P path is not
   saturated. Related observation: DeviceBackend's persistent kernel
   stalls under load — `test_perf_p2p_copy` floods stderr with
   `[dev-stall] fifoN pending=... head/tail advancing slowly` forensic
   logs, i.e. tasks sit in the FIFO while the kernel drains one at a
   time. That is a separate throughput problem on the device path.

   **Root-caused and fixed (partially)**: the shim's fixed 64KB tile
   size capped large-message throughput — 256MB meant 4096 tiles, and
   per-tile fixed overhead (signal matching, scheduling, put
   post-processing) dominated. `nccl.cc` now sizes tiles adaptively via
   the shared `adaptive_tile_bytes` helper in `coll_config.h` (also
   used by `test_perf_spray_allreduce`): messages at or below the sweet
   spot (default 64KB) move as ONE tile, and larger messages are tiled
   to at most 256 tiles (`max(sweet_spot, ceil(bytes/256))`, rounded up
   to 32B so tile boundaries stay aligned for the device copy's
   vectorized path; `UK_CCL_TILE_MIN_BYTES` overrides the sweet spot).
   Measured A/B (A40/L40S, nccl-tests): 64KB floor wins 256KB-4MB by
   6-29% (more tiles pipeline the ring better than per-tile cost
   hurts); 16MB prefers a 256KB-1MB floor (-16%); >=64MB is
   transport/device-bound regardless of the floor.
   Result: 256MB AllReduce 103ms → 54ms
   (matches the spray path 53.8ms; 64MB +45%). Small messages
   unchanged (verified with a no-interference probe: 256KB 164us, 1MB
   364us). AllGather 256MB: 22.6ms; ReduceScatter 256MB: 71.2ms. The
   remaining ~5-10 GB/s busbw (vs 43 GB/s native) is now the transport
   path itself (IPC puts dominate same-node), not the shim.

   **Reduce kernel vectorized (NCCL-style)**: the device reduce path
   (`read_reduce_store` in `device/ops/ops.h`) was a scalar, non-
   coalesced element loop; it now uses 16B-aligned wide loads/stores
   (`TypedVec<T,N>`, N = 16/sizeof(T)) with the reduce op folded in at
   compile time, plus a coalesced scalar fallback for unaligned tile
   offsets. Kernel-level: 5.6x (1 block) to 14x (64 blocks) on 256MB
   fp32. End-to-end (nccl-tests, A40/L40S pair): AllReduce 256MB
   54.1ms → 27.6ms, ReduceScatter 256MB 71.4ms → 44.7ms, AllReduce
   64MB −42%; AllGather unchanged (put-bound). The blocks=1 collective
   wall time is still dominated by the serialized per-tile IPC puts,
   not the reduce compute.

   **IPC send sliding window**: the send worker synchronized after
   EVERY put (copy + ~10us fixed overhead serialized), capping 1MB-tile
   puts at ~37 GB/s vs ~52 GB/s for large puts.
   `IpcAdapter::send_worker` now keeps up to `UK_CCL_IPC_BATCH` (default
   16) puts in flight: each put gets its own event and a round-robin
   stream (consecutive tiles copy in parallel), launches run ahead of
   completions, and puts are completed FIFO with data-before-signal
   semantics (per-put event sync only at the front, when the window is
   full or the ring empties — small messages drain with no added
   latency). RDMA needs no equivalent: `ibv_post_send` is already
   async with a CQ poll thread and a `kMaxInflightWrs` window. Measured
   on a loaded box (other tenants active): 256MB AllGather (put-only)
   −4.5% vs window=1; needs a re-check on an idle machine for the full
   effect.

   **Reduce kernel ILP + multi-block**: `read_reduce_store` keeps 4
   independent 16B vectors in flight per thread — single-GPU 256MB fp32
   reduce: 8 blocks ~153 GB/s, 16 blocks ~218 GB/s (vs ~72 GB/s at 8
   blocks before), so fewer blocks saturate DRAM. An experiment
   replacing the multi-block kernel's per-task grid barrier with a
   lock-free slice pipeline (per-task counters + single-writer tail
   publish) was numerically correct but regressed badly at 16-64 blocks
   (every block polling the FIFO head with system-scope atomics costs
   O(blocks) per task, exceeding the slice work), and a cudaFree hang
   on the sync buffer was traced to system-scope atomics on the GDR
   tail. Reverted to the original barrier-based `multiPersistentKernel`;
   with the ILP reduce it reaches 256MB AllReduce 11.9ms (45 GB/s) in
   the spray benchmark at 64 blocks on the A40/L40S pair. Multi-block
   (blocks_per_worker > 1) is now fully usable in the shim too, after
   three fixes: the MultiBlockSync buffer is stream-ordered
   (cudaMallocAsync/cudaFreeAsync — plain cudaFree hung the context on
   this driver), the sync buffer is zeroed before every kernel launch
   (a relaunched kernel otherwise read the previous kernel's exit phase
   and returned before consuming tasks), and the per-task phase waits
   use `<` instead of `!=` so a block preempted past a phase catches up
   instead of deadlocking. nccl-tests at UK_CCL_DEV_BLOCKS=64: full
   256K-256M sweep, 0 wrong, 256MB 14.8ms (18.1 GB/s), no stalls.
