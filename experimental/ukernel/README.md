# UKernel

Minimal build and test entry points for `experimental/ukernel`.

## Prerequisites

- CUDA toolchain for NVIDIA builds
- RDMA / verbs dependencies used by `transport` and `ccl`
- system-installed GDRCopy (`gdrapi.h` + `libgdrapi`) for `device` and `ccl`
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
CUDA_VISIBLE_DEVICES=0,6,7 torchrun --nproc_per_node=3 test_collective.py  # for 3-rank tests
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 test_p2p.py

# rdma battle
UK_P2P_TRANSPORT=uccl UCCL_P2P_MODE=rdma NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
        Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.063 |            0.03 |       0.316 |          0.01 |       0.096 |          0.02
      4096 B |         0.093 |            0.09 |       0.306 |          0.03 |       0.099 |          0.08
     16384 B |         0.199 |            0.16 |       0.366 |          0.09 |       0.103 |          0.32
     65536 B |         0.249 |            0.53 |       0.431 |          0.30 |       0.159 |          0.82
    262144 B |         0.437 |            1.20 |       0.687 |          0.76 |       0.385 |          1.36
   1048576 B |         1.238 |            1.69 |       1.846 |          1.14 |       1.244 |          1.69
   4194304 B |         5.232 |            1.60 |       5.699 |          1.47 |       4.480 |          1.87
  16777216 B |        19.853 |            1.69 |      20.715 |          1.62 |      17.434 |          1.92
  67108864 B |        79.071 |            1.70 |      78.673 |          1.71 |      69.287 |          1.94
 268435456 B |       319.966 |            1.68 |     319.911 |          1.68 |     275.867 |          1.95

        Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.074 |            0.03 |       0.711 |          0.00 |       0.097 |          0.02
      4096 B |         0.086 |            0.10 |       0.316 |          0.03 |       0.098 |          0.08
     16384 B |         0.196 |            0.17 |       0.240 |          0.14 |       0.099 |          0.33
     65536 B |         0.234 |            0.56 |       0.300 |          0.44 |       0.150 |          0.88
    262144 B |         0.416 |            1.26 |       0.631 |          0.83 |       0.372 |          1.41
   1048576 B |         1.272 |            1.65 |       1.388 |          1.51 |       1.227 |          1.71
   4194304 B |         5.083 |            1.65 |       5.726 |          1.46 |       4.475 |          1.87
  16777216 B |        20.159 |            1.66 |      24.035 |          1.40 |      17.449 |          1.92
  67108864 B |        79.323 |            1.69 |      78.945 |          1.70 |      69.256 |          1.94
 268435456 B |       319.898 |            1.68 |     315.428 |          1.70 |     275.951 |          1.95

        Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         0.086 |            0.02 |       0.140 |          0.01 |       0.137 |          0.01
      4096 B |         0.090 |            0.09 |       0.121 |          0.07 |       0.132 |          0.06
     16384 B |         0.180 |            0.18 |       0.149 |          0.22 |       0.133 |          0.25
     65536 B |         0.246 |            0.53 |       0.205 |          0.64 |       0.151 |          0.87
    262144 B |         0.410 |            1.28 |       0.410 |          1.28 |       0.374 |          1.40
   1048576 B |         1.268 |            1.65 |       1.243 |          1.69 |       1.225 |          1.71
   4194304 B |         5.050 |            1.66 |       5.033 |          1.67 |       4.437 |          1.89
  16777216 B |        19.814 |            1.69 |      19.923 |          1.68 |      17.447 |          1.92
  67108864 B |        78.855 |            1.70 |      78.982 |          1.70 |      69.237 |          1.94
 268435456 B |       319.691 |            1.68 |     315.106 |          1.70 |     276.135 |          1.94

# ipc battle
UK_P2P_TRANSPORT=ipc UCCL_P2P_MODE=ipc CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
         Size |  ukernel (ms) |  ukernel (GB/s) |   UCCL (ms) |   UCCL (GB/s) |   NCCL (ms) |   NCCL (GB/s)
--------------------------------------------------------------------------------------------------------------------------------
      1024 B |         3.318 |            0.00 |       0.042 |          0.05 |       0.091 |          0.02
      4096 B |         3.470 |            0.00 |       0.042 |          0.20 |       0.090 |          0.09
     16384 B |         3.339 |            0.01 |       0.064 |          0.51 |       0.087 |          0.37
     65536 B |         3.410 |            0.04 |       0.078 |          1.68 |       0.098 |          1.34
    262144 B |         3.439 |            0.15 |       0.185 |          2.84 |       0.088 |          5.96
   1048576 B |         3.602 |            0.58 |       0.566 |          3.70 |       0.102 |         20.66
   4194304 B |         3.679 |            2.28 |       1.374 |          6.10 |       0.250 |         33.50
  16777216 B |         4.104 |            8.18 |       4.683 |          7.17 |       0.799 |         41.99
  67108864 B |         6.058 |           22.15 |      17.865 |          7.51 |       2.758 |         48.66
 268435456 B |        13.548 |           39.63 |      70.594 |          7.61 |      10.814 |         49.64

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
pg.allreduce(x, tile_bytes=65536, num_flows=2)

y = torch.randn(1024 * world, device="cuda", dtype=torch.float32)
pg.alltoall(y, tile_bytes=65536, num_flows=2)

send = torch.randn(13, device="cuda", dtype=torch.float32)
recv = torch.empty(13, device="cuda", dtype=torch.float32)
dist.all_to_all_single(
    recv,
    send,
    output_split_sizes=[4, 5, 4],
    input_split_sizes=[4, 5, 4],
    group=pg,
    tile_bytes=65536,
    num_flows=2,
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
