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

> **Troubleshooting — conda toolchain breaks the build**: if linking
> fails with `undefined reference to '__nptl_change_stack_perm@GLIBC_PRIVATE'`
> (or other `GLIBC_PRIVATE` symbols), the conda linker is being used
> against the system glibc. Deactivate conda and make sure the system
> toolchain is first on `PATH` before running `make`:
>
> ```bash
> conda deactivate
> export PATH=/usr/local/cuda/bin:/usr/bin:/bin:$PATH
> which gcc g++ ld        # all must point to /usr/bin, not miniconda
> ```
>
> If the `tests` still fail (e.g. `gdrcopy_pplat` is linked by nvcc,
> which can pick up conda's g++), either skip them — `make lib` then
> `sudo make install` — or force the system compiler:
> `/usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++ ...`.


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

The NCCL shim (`libnccl.so`) builds on both platforms: `make nccl`
(NVIDIA) or `make -f Makefile.rocm nccl` (ROCm); the shim API and
`nccl.cc` use the `gpu_rt.h` wrappers (`gpuStream_t` etc.), which map to
hip types on ROCm, so the exported .so is ABI-compatible with
ROCm-built nccl-tests and rccl-tests. On ROCm the install also ships as
`librccl.so.1` (RCCL's SONAME), so rccl-tests runs against the shim with
an `LD_LIBRARY_PATH` swap just like nccl-tests.

Common overrides:

```bash
make -f Makefile CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80 \
    GDRCOPY_INCLUDEDIR=/usr/include GDRCOPY_LIBDIR=/usr/local/lib
```

> **Build for your GPU only — pass `SM=<arch>`, not the default 4-arch
> list.** The default compiles compute_80/86/89/100; with the reduce-ILP
> dispatch in `persistent_kernel_ops.cu` that makes ptxas very slow, and
> B300 (sm_103) is not even in the default list. Always build the target
> capability:
>
> ```bash
> make SM=103 ENABLE_TMA=0 -j8 nccl          # B300 / GB300
> make SM=86  ENABLE_TMA=0 -j8 nccl          # A40
> make SM=89  ENABLE_TMA=0 -j8 device_bench  # L40S etc.
> ```
>
> `ENABLE_TMA=0` keeps the TMA code paths off (they auto-enable for
> SM ≥ 90); omit it only if you intentionally want TMA. On machines with
> many cores, cap the parallelism (`-j8`..`-j16`) — `-j$(nproc)` on a
> 100+ core box thrashes and can look like a hang.
>
> nvcc comes from `CUDA_HOME/bin/nvcc` (falling back to PATH only when
> the toolkit is absent) — so an active conda base no longer hijacks the
> build with a CUDA 12.x nvcc that cannot target sm_103.
>
> If you have a conda base active, deactivate it before building (or at
> least `export PATH=/usr/local/cuda/bin:/usr/bin:/bin:$PATH`): conda's
> `ld` also breaks the link step (`cannot find -lgdrapi` — it does not
> search `/usr/local/lib`), and on ROCm machines the same applies to
> conda's compiler. Set `GDRCOPY_LIBDIR=/usr/local/lib` if gdrcopy was
> installed outside the system default search path.
>
> Reduce-kernel ILP is a build-time knob: `make ... REDUCE_ILP=8` (default
> 4, values 4/8/16; see `docs/reduce_ilp_tuning.md`). Build-time keeps the
> fully-unrolled kernel cheap to compile (~20 s instead of ~20 min per
> device file).

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
| `SM` | `80 86 89 100` | GPU compute capability; set to ONE arch (e.g. `SM=103`) to build only that arch instead of the slow default list |

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
# Pull only the nccl-tests submodule (the other thirdparty/
# submodules stay untouched):
git submodule update --init --depth 1 thirdparty/nccl-tests

# Compile against a NORMAL NCCL install (headers + lib are only needed
# at build time; which libnccl.so runs is decided at runtime by
# LD_LIBRARY_PATH — so perftests are never built against the shim).
cd thirdparty/nccl-tests

# Find the native NCCL prefix if you don't know it offhand:
ldconfig -p | grep libnccl                    # runtime .so location
dpkg -L libnccl-dev 2>/dev/null | grep -E 'nccl\.h|libnccl\.so'   # Ubuntu/Debian
# If nccl.h is already on the default CUDA include path, NCCL_HOME can be
# omitted entirely — nccl-tests then uses the default search paths.

# Locate MPI (OpenMPI; nccl-tests links -lmpi). MPI_HOME must contain
# include/ and lib/ (or lib64/) directly:
readlink -f "$(which mpirun)"                 # real binary -> install prefix
find /usr /opt -name mpi.h 2>/dev/null | head -3
# Ubuntu/Debian OpenMPI: MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
# conda OpenMPI:          MPI_HOME=$CONDA_PREFIX

# Generate NVCC_GENCODE from the GPUs actually present instead of
# hardcoding archs (each unique compute capability gets a gencode entry):
export NVCC_GENCODE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
  | sort -u | awk -F. '{printf " -gencode=arch=compute_%d%d,code=sm_%d%d", $1, $2, $1, $2}')"
echo "NVCC_GENCODE=$NVCC_GENCODE"

# Locate MPI first — see the find snippet above. On Ubuntu/Debian OpenMPI:
make MPI=1 CUDA_HOME=/usr/local/cuda \
    MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1 \
    NCCL_HOME=/usr/lib/x86_64-linux-gnu \
    NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103" \
    -j$(nproc)

# Multi-rank, same node (2 GPUs, one process per rank — exercises
# ncclGetUniqueId + ncclCommInitRank). Result validation is ON by
# default; keep -c 1 (do NOT pass -c 0, or the run measures nothing —
# and a 1-rank run without mpirun measures nothing either, since
# single-rank collectives are no-ops).
cd ../../experimental/ukernel
CUDA_VISIBLE_DEVICES=6,7 mpirun --mca hwloc_base_binding_policy none -np 2 \
    -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib -x CUDA_VISIBLE_DEVICES \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -c 1

# Debug (optional): propagate UK_CCL_DEBUG into the ranks — the shell env
# alone does NOT reach them; 1=executor, 2=+transport, 3=all. Look for
# "[pick rN] op[x] peer=y -> path=z" (z: 0=device 1=IPC 2=RDMA).
CUDA_VISIBLE_DEVICES=6,7 mpirun --mca hwloc_base_binding_policy none -np 2 \
    -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib -x CUDA_VISIBLE_DEVICES \
    -x UK_CCL_DEBUG \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 4M -f 2 -g 1 -c 1 -n 3

# Single process, multiple GPUs (exercises ncclCommInitAll, which
# initializes one communicator per device on its own thread):
LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -g 2

# Multi-node: rank 0 packs its NIC IP into the unique ID and peers
# connect to it. On multi-homed hosts pick the interface via
# NCCL_SOCKET_IFNAME (prefix match, e.g. "eth" matches eth0/eth1);
# a warning is printed if it matches no interface.
mpirun --mca hwloc_base_binding_policy none -np 2 -H node0,node1 \
    -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    -x NCCL_SOCKET_IFNAME=eth0 \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -g 1
```

ROCm / RCCL: use the vendored **rccl-tests** (ROCm's nccl-tests fork,
same nccl\* API). Build it against the normal RCCL install — the binary
is swapped to the shim at runtime via `LD_LIBRARY_PATH` (the ROCm shim
install includes `librccl.so.1`):

```bash
# Pull only the rccl-tests submodule:
git submodule update --init --depth 1 thirdparty/rccl-tests

# Find the RCCL install (usually /opt/rocm):
find /opt/rocm /usr -name 'librccl.so*' 2>/dev/null | head

cd thirdparty/rccl-tests
# Auto-generate GPU_TARGETS from the GPUs actually present (no
# hardcoded gfx list):
export GPU_TARGETS="$(rocm_agent_enumerator | grep '^gfx' | sort -u | tr '\n' ',' | sed 's/,$//')"
echo "GPU_TARGETS=$GPU_TARGETS"

# rccl-tests looks for <MPI_HOME>/openmpi/{include,lib}, so on
# Ubuntu/Debian use MPI_HOME=/usr/lib/x86_64-linux-gnu — NOT the
# .../openmpi prefix that nccl-tests takes.
make MPI=1 ROCM_PATH=/opt/rocm \
    MPI_HOME=/usr/lib/x86_64-linux-gnu \
    GPU_TARGETS="$GPU_TARGETS" \
    -j$(nproc)

# Run against the shim (multinode/other args identical to nccl-tests):
cd ../experimental/ukernel
mpirun --mca hwloc_base_binding_policy none -np 2 \
    -x LD_LIBRARY_PATH=$(pwd)/build/nccl/lib \
    -x HIP_VISIBLE_DEVICES \
    ../../thirdparty/rccl-tests/build/all_reduce_perf -b 1M -e 256M -g 1
```

NCCL compatibility — supported APIs, in-place semantics, unsupported
ops, and the drop-in ABI — lives in
[`docs/nccl_compatibility.md`](docs/nccl_compatibility.md).

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
- [docs/put_path_selection.md](docs/put_path_selection.md) — design notes on IPC/device/RDMA path selection
- [docs/alltoall_comparison.md](docs/alltoall_comparison.md) — AllToAll comparison plan vs user-space MoE implementations (DeepEP)
- [docs/perf_test_procedure.md](docs/perf_test_procedure.md) — how to build and run the shim/native/spray perf tests

## Performance measurements

Two-GPU AllReduce, A40 pair (CUDA 6,7; P2P over PCIe gen4). All numbers
are true 2-rank runs — the shim rows are `mpirun -np 2` over
nccl-tests, the spray row is the native SprayExecutor benchmark
(`test_perf_spray_allreduce`). A single-process `-g 1` nccl-tests run
is a 1-rank no-op (launch overhead only) and must NOT be used as a
performance number.

| Size | shim nccl-tests | native NCCL 2.29.7 |
|---|---|---|
| 256KB | 82us | 25.6us |
| 1MB | 101us | 60.0us |
| 4MB | 159us | 140.4us |
| 16MB | 419us | 454.4us |
| 64MB | 1558us | 1691.8us |
| 256MB | 5506us | 6352.8us |

256MB AllReduce (5.5ms / 49 GB/s), AllGather (2.8ms / 94 GB/s) and
ReduceScatter (3.1ms / 87 GB/s) all beat native NCCL (6.35 / 3.78 /
3.9ms). The wins came from routing same-host puts over IPC (the
latency-based path balancer was misrouting onto the device/RDMA
paths), 256-thread × 8-block device kernels (the RS output copy was a
1-block task at ~18 GB/s), and larger tiles for large messages.

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
>
> Two more easy-to-miss env rules:
> - **Propagate env with `-x`.** `UK_CCL_DEBUG=2 ... mpirun` in the shell
>   does NOT reach the rank processes; add `-x UK_CCL_DEBUG` (and
>   `-x UK_CCL_PUT_PATH` / `-x UK_BAR1_WINDOW_MB` if you set those).
> - **A/B put-path knobs**: `UK_CCL_PUT_PATH=device|ipc|rdma` forces the
>   same-host data path (unset = normal IPC for same-host, RDMA remote).
>   Use it to isolate whether a slowdown is transport selection or the
>   executor/device kernels.

## Current gaps

256MB AllReduce (5.5ms), AllGather (2.8ms) and ReduceScatter (3.1ms)
all beat native NCCL (6.35 / 3.78 / 3.9ms) on the A40 pair; 16MB
AllReduce is at parity (455us). Two gaps remain, both acceptable for
the target workloads:

1. **Small-message latency (~3x native below 4MB)**: 256KB AllReduce
   77-88us vs native 25us. The gap is the per-phase CPU round trips in
   the copy-offload architecture (submit + IPC put launch/sync + stream
   gates); it narrows with size and reaches parity around 16MB. The
   native SprayExecutor path measures 63us at 256KB, so the shim's
   prepare/gate overhead accounts for roughly 15us of it. Closing it
   fully would require running the whole small-message collective
   in-kernel (no CPU round trips), which is deferred.

2. **AllToAll has no NCCL-native comparison**: NCCL exposes no
   dedicated AllToAll primitive — engines build it from
   `ncclSend`/`ncclRecv` groups, which the shim does not implement. The
   comparison target is user-space MoE implementations (DeepEP); see
   [docs/alltoall_comparison.md](docs/alltoall_comparison.md). Our
- [docs/perf_test_procedure.md](docs/perf_test_procedure.md) — how to build and run the shim/native/spray perf tests
   native AllToAll (spray) reaches 2.9ms / 93 GB/s at 256MB.

How the large-message wins were reached (`6ae8d24d..HEAD`):
ILP-vectorized reduce kernel, pipelined IPC send window, same-host
puts pinned to IPC (the latency-based path balancer was misrouting onto
the device/RDMA paths — design notes in
[docs/put_path_selection.md](docs/put_path_selection.md)), 1MB tile
sweet spot, 256-thread × 8-block device kernels, and a fast-path
prepare(). Multi-block teardown/relaunch/phase bugs were fixed along
the way (stream-ordered MultiBlockSync free, d_sync reset before every
launch, `<` phase waits).

### B300 snapshot (2026-08-05, 2/4/8 ranks, tuned config)

Full tables and commands live in
[docs/b300_native_nccl_measurements.md](docs/b300_native_nccl_measurements.md)
and [docs/alltoall_comparison.md](docs/alltoall_comparison.md). Summary
(shim vs native NCCL 2.29.7, all wrong=0):

| collective | 2r | 4r | 8r |
|---|---:|---:|---:|
| AllReduce 256M shim | 578us / 464 GB/s | 1026us / 262 GB/s | 1943us / 138 GB/s |
| AllReduce 256M native | 521us / 515 GB/s | 673us / 399 GB/s | 719us / 373 GB/s |
| AllToAll 256M shim | 378us / 710 GB/s | 628us / 427 GB/s | 884us / 303 GB/s |
| AllToAll 256M native | 416us / 646 GB/s | 425us / 632 GB/s | 433us / 621 GB/s |

AllReduce is 1.1x/1.5x/2.7x native at 2/4/8 ranks (per-tile host
signal chain); AllToAll beats native at 2 ranks and trails at 4/8
(every send is staged through scratch after the in-place race fix).
Next optimization targets: signal aggregation + batched waits for
AllReduce, copy-engine staging + copy/put pipelining for AllToAll.
