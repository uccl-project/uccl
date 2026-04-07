# UKernel

Minimal build and test entry points for `experimental/ukernel`.

## Prerequisites

- CUDA toolchain for NVIDIA builds
- RDMA / verbs dependencies used by `transport` and `ccl`
- `thirdparty/gdrcopy` initialized and built for `device` and `ccl`
- `torchrun` available for CCL multiprocess integration tests

If submodules are missing:

```bash
git submodule update --init --recursive
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
./test_transport_integration communicator --role=server --case=exchange --exchanger-port 16979
./test_transport_integration communicator --role=client --case=exchange --exchanger-ip 127.0.0.1 --exchanger-port 16979
```

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
UK_P2P_TRANSPORT=uccl NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
        Size |  ukernel (ms) |  ukernel (GB/s) |   NCCL (ms) |   NCCL (GB/s)
      1024 B |         0.225 |            0.01 |       0.121 |          0.02
      4096 B |         0.265 |            0.03 |       0.123 |          0.07
     16384 B |         0.345 |            0.10 |       0.200 |          0.16
     65536 B |         0.407 |            0.32 |       0.201 |          0.65
    262144 B |         0.576 |            0.91 |       0.383 |          1.37
   1048576 B |         1.380 |            1.52 |       1.231 |          1.70
   4194304 B |         5.127 |            1.64 |       4.428 |          1.89
  16777216 B |        19.979 |            1.68 |      17.213 |          1.95
  67108864 B |        79.735 |            1.68 |      68.675 |          1.95
 268435456 B |       319.598 |            1.68 |     273.423 |          1.96

# ipc battle
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
        Size |  ukernel (ms) |  ukernel (GB/s) |   NCCL (ms) |   NCCL (GB/s)
      1024 B |         3.295 |            0.00 |       0.072 |          0.03
      4096 B |         3.398 |            0.00 |       0.069 |          0.12
     16384 B |         3.317 |            0.01 |       0.087 |          0.38
     65536 B |         3.448 |            0.04 |       0.090 |          1.45
    262144 B |         3.425 |            0.15 |       0.087 |          6.06
   1048576 B |         3.615 |            0.58 |       0.114 |         18.40
   4194304 B |         3.514 |            2.39 |       0.250 |         33.50
  16777216 B |         3.794 |            8.84 |       0.794 |         42.26
  67108864 B |         6.243 |           21.50 |       2.758 |         48.66
 268435456 B |        13.531 |           39.68 |      10.814 |         49.65
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
