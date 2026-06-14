# UKernel

Minimal build and test entry points for `experimental/ukernel`.

## Prerequisites

- CUDA/Rocm toolchain for NVIDIA/AMD builds
- RDMA / verbs dependencies used by `transport` and `ccl`
- system-installed GDRCopy (`gdrapi.h` + `libgdrapi`) for `device` and `ccl` (only Nvidia)
- `torchrun` available for CCL multiprocess integration tests

## Quick Install GDRCopy (System)

`experimental/ukernel` no longer builds `thirdparty/gdrcopy`.  
Please install GDRCopy from NVIDIA upstream on your machine first.

Ubuntu quick path (source build):

```bash
sudo apt-get update
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) libelf-dev
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make CUDA=/usr/local/cuda
sudo make CUDA=/usr/local/cuda prefix=/usr/local install
```

Optional: if `libgdrapi.so` is not in the default linker path, pass `GDRCOPY_LIBDIR` when building:

```bash
cd experimental/ukernel
make GDRCOPY_LIBDIR=/usr/local/lib
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
make -f Makefile CUDA_PATH=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80
```

## Test

Run all transport tests:

```bash
cd experimental/ukernel
make transport_test
```

Build transport benchmark:

```bash
cd experimental/ukernel
make transport_bench
```

Manual two-process transport check:

```bash
cd experimental/ukernel/src/transport
make test-integration
CUDA_VISIBLE_DEVICES=5,6 ./test_transport_integration communicator --role=server --case=exchange --transport ipc --exchanger-port 16979
CUDA_VISIBLE_DEVICES=5,6 ./test_transport_integration communicator --role=client --case=exchange --transport ipc --exchanger-ip 127.0.0.1 --exchanger-port 16979
```

For IPC checks, expose both peer GPUs to both processes. The transport
integration test defaults to server `--gpu=0` and client `--gpu=1` within the
visible device list; use `--gpu`/`--peer-gpu` to override.

Run all device tests:

```bash
cd experimental/ukernel
make device_test SM=80
```

Run all CCL tests:

```bash
cd experimental/ukernel
make ccl_test SM=80
```

## Python Binding

`experimental/ukernel/py` contains a `torch`-based Python extension that wraps
the `ccl` executor behind a persistent `ProcessGroup` object. The binding takes
CUDA `torch.Tensor` inputs directly and runs `allreduce` / `alltoall` through
the existing `transport + ccl + device` stack.

Build the extension in place:

```bash
cd experimental/ukernel/py
python setup.py build_ext --inplace
```

Run Python tests (requires 2+ GPUs):

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 test_collective.py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 test_p2p.py

# rdma battle
UK_P2P_TRANSPORT=rdma UCCL_P2P_MODE=rdma NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py

on Nvidia local:
        Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.038 |            0.05 |       0.249 |          0.01 |       0.162 |          0.01
      4096 B |         0.041 |            0.20 |       0.288 |          0.03 |       0.174 |          0.05
     16384 B |         0.057 |            0.57 |       0.252 |          0.13 |       0.170 |          0.19
     65536 B |         0.108 |            1.21 |       0.305 |          0.43 |       0.155 |          0.85
    262144 B |         0.342 |            1.53 |       0.543 |          0.97 |       0.380 |          1.38
   1048576 B |         1.212 |            1.73 |       1.433 |          1.46 |       1.240 |          1.69
   4194304 B |         5.063 |            1.66 |       5.318 |          1.58 |       4.459 |          1.88
  16777216 B |        20.246 |            1.66 |      20.235 |          1.66 |      17.275 |          1.94
  67108864 B |        80.218 |            1.67 |      80.537 |          1.67 |      68.708 |          1.95
 268435456 B |       330.008 |            1.63 |     320.356 |          1.68 |     273.579 |          1.96

# ipc battle
UK_P2P_TRANSPORT=ipc UCCL_P2P_MODE=ipc CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py

on Nvidia local:
        Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.030 |            0.07 |       0.037 |          0.06 |       0.210 |          0.01
      4096 B |         0.027 |            0.30 |       0.037 |          0.22 |       0.207 |          0.04
     16384 B |         0.029 |            1.13 |       0.039 |          0.84 |       0.218 |          0.15
     65536 B |         0.031 |            4.28 |       0.041 |          3.16 |       0.200 |          0.66
    262144 B |         0.039 |           13.57 |       0.048 |         10.86 |       0.214 |          2.46
   1048576 B |         0.072 |           28.97 |       0.084 |         25.10 |       0.209 |         10.04
   4194304 B |         0.210 |           39.97 |       0.212 |         39.48 |       0.254 |         33.06
  16777216 B |         0.693 |           48.39 |       0.696 |         48.18 |       0.804 |         41.75
  67108864 B |         2.636 |           50.92 |       2.645 |         50.74 |       2.773 |         48.39
 268435456 B |        10.297 |           52.14 |      10.319 |         52.03 |      10.826 |         49.59

on AMD0:
         Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.030 |            0.07 |       0.039 |          0.05 |       0.142 |          0.01
      4096 B |         0.030 |            0.27 |       0.039 |          0.21 |       0.125 |          0.07
     16384 B |         0.031 |            1.06 |       0.039 |          0.83 |       0.127 |          0.26
     65536 B |         0.044 |            2.96 |       0.031 |          4.20 |       0.124 |          1.05
    262144 B |         0.052 |           10.16 |       0.041 |         12.80 |       0.135 |          3.90
   1048576 B |         0.083 |           25.39 |       0.073 |         28.70 |       0.146 |         14.39
   4194304 B |         0.206 |           40.71 |       0.253 |         33.20 |       0.268 |         31.36
  16777216 B |         0.697 |           48.13 |       0.764 |         43.90 |       0.758 |         44.27
  67108864 B |         2.656 |           50.54 |       2.797 |         47.99 |       2.715 |         49.44
 268435456 B |        10.495 |           51.15 |      10.920 |         49.16 |      10.548 |         50.90

# tcp
UK_P2P_TRANSPORT=tcp UCCL_P2P_MODE=ipc CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
         Size |  ukernel (ms) |  ukernel (GB/s)
------------------------------------------------------
      1024 B |         3.318 |            0.00
      4096 B |         3.470 |            0.00
     16384 B |         3.339 |            0.01
     65536 B |         3.410 |            0.04
    262144 B |         3.439 |            0.15
   1048576 B |         3.602 |            0.58
   4194304 B |         3.679 |            2.28
  16777216 B |         4.104 |            8.18
  67108864 B |         6.058 |           22.15
 268435456 B |        13.548 |           39.63
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
    exchanger_port=int(os.environ.get("MASTER_PORT", "29500")),
    transport="auto",
)

x = torch.randn(1024 * world + 1, device="cuda", dtype=torch.float32)
pg.allreduce(x, tile_bytes=65536, num_streams=2)

y = torch.randn(1024 * world, device="cuda", dtype=torch.float32)
pg.alltoall(y, tile_bytes=65536, num_streams=2)

send = torch.randn(13, device="cuda", dtype=torch.float32)
recv = torch.empty(13, device="cuda", dtype=torch.float32)
dist.all_to_all_single(
    recv,
    send,
    output_split_sizes=[4, 5, 4],
    input_split_sizes=[4, 5, 4],
    group=pg,
    tile_bytes=65536,
    num_streams=2,
)
```

Current Python binding constraints:

- collective payload tensors must be CUDA and contiguous
- `allreduce` supports non-divisible element counts
- equal-split and variable-split `all_to_all_single` are both supported
- variable-split `all_to_all_single` requires explicit
  `input_split_sizes/output_split_sizes` whose sums match local tensor numel

## Modules

- [`src/transport/README.md`](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/transport/README.md)
- [`src/device/README.md`](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/device/README.md)
- [`src/ccl/README.md`](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/ccl/README.md)
- [`benchmarks/README.md`](/Users/jacelau/code/opencode/uccl/experimental/ukernel/benchmarks/README.md)
