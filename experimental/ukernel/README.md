# UKernel

`experimental/ukernel` is a layered runtime with three core modules:

- `transport`: peer bootstrap, path selection, async send/recv, progress
- `device`: device-side FIFO/task/worker execution
- `ccl`: collective planning + executor on top of transport/device

## Prerequisites

- CUDA toolchain (NVIDIA build path)
- RDMA / verbs dependencies (`libibverbs`, `librdmacm`)
- system-installed GDRCopy (`gdrapi.h` + `libgdrapi.so`)
- `torchrun` for multiprocess CCL/Python integration tests

If `libgdrapi.so` is outside default linker paths, pass `GDRCOPY_LIBDIR`.

## Build

NVIDIA:

```bash
cd experimental/ukernel
make clean -f Makefile
make -j$(nproc) -f Makefile
```

ROCm:

```bash
cd experimental/ukernel
make clean -f Makefile.rocm
make -j$(nproc) -f Makefile.rocm
```

Common override example:

```bash
make -f Makefile CUDA_PATH=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80
```

## Common Targets

Tests:

```bash
make transport_test
make device_test SM=80
make ccl_test SM=80
```

Benchmarks:

```bash
make bench_transport
make bench_gdrcopy
make bench_all
```

Compatibility aliases:

```bash
make transport_bench   # alias of bench_transport
make bench             # alias of bench_all
```

## Python Binding

Build extension:

```bash
cd experimental/ukernel/py
python setup.py build_ext --inplace
```

Run common tests:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 test_collective.py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 test_p2p.py
```

Run p2p benchmark flavors:

```bash
# Collective/rdma path
# UCCL_DEBUG_VLOG_LEVEL=1 : check fast path
UK_P2P_TRANSPORT=uccl UCCL_P2P_MODE=rdma NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py

# p2p/rdma path
UK_P2P_TRANSPORT=rdma UCCL_P2P_MODE=rdma NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py

# IPC path
UK_P2P_TRANSPORT=ipc UCCL_P2P_MODE=ipc NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py

# TCP path
UK_P2P_TRANSPORT=tcp UCCL_P2P_MODE=ipc NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 bench_p2p.py
```

Naming note:

- `UK_P2P_TRANSPORT=uccl` is a compatibility-facing logical selector.
- In ukernel transport, this selector currently maps to the RDMA adapter backend.
- `UCCL_P2P_MODE` controls standalone UCCL behavior and is independent of ukernel adapter naming.

## Module Docs

- [transport README](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/transport/README.md)
- [device README](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/device/README.md)
- [ccl README](/Users/jacelau/code/opencode/uccl/experimental/ukernel/src/ccl/README.md)
- [benchmarks README](/Users/jacelau/code/opencode/uccl/experimental/ukernel/benchmarks/README.md)
