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
| `SM` | `80 86 89` | GPU compute capability |

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
