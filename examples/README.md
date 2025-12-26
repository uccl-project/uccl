# UCCL PyTorch Examples

This folder contains examples demonstrating how to use **UCCL** (Ultra and Unified CCL) with PyTorch for distributed training on both **NVIDIA** and **AMD** GPU clusters.

---

## Overview

| Platform | Training Script | Launch Script | Description |
|----------|----------------|---------------|-------------|
| **NVIDIA CUDA** | `ddp_cuda_test.py` | `ddp_cuda_run.sh` | CIFAR-10 ResNet-18 training with NCCL/UCCL |
| **AMD ROCm** | `ddp_amd_test.py` | `ddp_amd_run.sh` | CIFAR-10 ResNet-18 training with RCCL/UCCL |
| **Both** | `multi_pg_test.py` | `multi_pg_run.sh` | Multi-process-group stress test |

---

## Prerequisites

### For NVIDIA GPUs (CUDA)

1. **PyTorch with CUDA support**
2. **NCCL** (bundled with CUDA)
3. **torchvision** (for training examples)
4. Optional: **UCCL** build providing `libnccl-net-uccl.so`

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For AMD GPUs (ROCm)

1. **PyTorch with ROCm support**
2. **RCCL** (bundled with ROCm)
3. **torchvision** (for training examples)
4. Optional: **UCCL** build providing `librccl-net-uccl.so`

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Building UCCL

Follow the instructions in [`collective/rdma/README.md`](../collective/rdma/README.md) to build the UCCL plugin.

```bash
# For NVIDIA GPUs
cd $UCCL_HOME/collective/rdma && make -j

# For AMD GPUs
cd $UCCL_HOME/collective/rdma && make -f Makefile.rocm -j
```

### Environment Setup for UCCL

```bash
export UCCL_HOME=<path/to/uccl>            # Root of your UCCL checkout
export CONDA_LIB_HOME=$CONDA_PREFIX/lib    # Contains libglog.so (if using conda)
```

If you see `UCCL_* set by environment to xx` printed, UCCL is successfully loaded.

<p align="left"> <img src="./uccl_output.png" alt="" width="700"> </p>

---

## 1. NVIDIA CUDA Examples (`ddp_cuda_*`)

### Single-Node Training (8 GPUs)

```bash
# Using standard NCCL
./ddp_cuda_run.sh nccl single 128 10

# Using UCCL
UCCL_HOME=/path/to/uccl ./ddp_cuda_run.sh uccl single 128 10
```

### Multi-Node Training (2 nodes x 8 GPUs)

```bash
# Master node (rank 0)
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 \
UCCL_HOME=/path/to/uccl ./ddp_cuda_run.sh uccl multi 128 10

# Worker node (rank 1)
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=1 WORLD_SIZE=2 \
UCCL_HOME=/path/to/uccl ./ddp_cuda_run.sh uccl multi 128 10
```

### Arguments

```bash
./ddp_cuda_run.sh [BACKEND] [MODE] [BATCH_SIZE] [EPOCHS]

# BACKEND : nccl | uccl
# MODE    : single | multi
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `UCCL_HOME` | Path to UCCL checkout | `/home/user/uccl` |
| `NUM_GPUS_PER_NODE` | GPUs per node (default: 8) | `4` |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0,1,2,3` |
| `NCCL_IB_HCA` | IB HCA devices | `mlx5_0:1,mlx5_1:1` |
| `NCCL_SOCKET_IFNAME` | Network interface | `eth0` |

---

## 2. AMD ROCm Examples (`ddp_amd_*`)

### Single-Node Training (4 GPUs)

```bash
# Using standard RCCL
./ddp_amd_run.sh nccl single 128 10

# Using UCCL
./ddp_amd_run.sh uccl single 128 10
```

### Multi-Node Training (2 nodes x 4 GPUs)

```bash
# Master node (rank 0)
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 \
./ddp_amd_run.sh uccl multi 128 10

# Worker node (rank 1)
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=1 WORLD_SIZE=2 \
./ddp_amd_run.sh uccl multi 128 10
```

### Arguments

```bash
./ddp_amd_run.sh [BACKEND] [MODE] [BATCH_SIZE] [EPOCHS]

# BACKEND : nccl | uccl
# MODE    : single | multi
```

---

## 3. Multi-Process-Group Stress Test (`multi_pg_*`)

This test creates four overlapping process groups and drives concurrent collectives on separate CUDA/HIP streams:

```
WORLD  - implicit default group
BIG    - explicit group containing all ranks
EVEN   - ranks {0,2,4,...}
ODD    - ranks {1,3,5,...}
```

During each iteration:
1. Resets per-rank tensors
2. Launches **four collectives** asynchronously:
   - all-reduce (WORLD)
   - broadcast (EVEN)
   - all-reduce (ODD)
   - all-gather (BIG)
3. Waits for completion and synchronizes

### Single-Node

```bash
./multi_pg_run.sh nccl single 4 100 4096   # BACKEND MODE GPUS ITER SIZE
```

### Multi-Node

```bash
# Master node
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 \
./multi_pg_run.sh uccl multi 4 100 4096

# Worker node
MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=1 WORLD_SIZE=2 \
./multi_pg_run.sh uccl multi 4 100 4096
```

### Arguments

```bash
./multi_pg_run.sh [BACKEND] [MODE] [NUM_GPUS] [ITERS] [TENSOR_SIZE]
```

---

## Troubleshooting

### General

- **Enable verbose logging**: `export NCCL_DEBUG=INFO GLOG_v=1`
- **Check GPU visibility**: `nvidia-smi` (CUDA) or `rocm-smi` (ROCm)

### NVIDIA-Specific

- **IB GID index issues**: Try `NCCL_IB_GID_INDEX=3`
- **NCCL version mismatch**: Use `LD_PRELOAD=<path to libnccl.so.2>`
- **Select specific NICs**: `NCCL_IB_HCA=mlx5_2:1`

### AMD-Specific

- **GPU visibility**: Ensure `HIP_VISIBLE_DEVICES` matches GPUs with RNIC visibility
- **Import order**: Always `import torch` before `import uccl.p2p` for AMD GPUs

### Multi-Node

- **Gloo connection failures**: Set `GLOO_SOCKET_IFNAME=<interface>`
- **Firewall issues**: Ensure ports are open between nodes

---

## Performance Tips

### UCCL Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UCCL_NUM_ENGINES` | Number of UCCL engines | 4 |
| `UCCL_PORT_ENTROPY` | Paths/QPs per engine | 8 |
| `UCCL_CHUNK_SIZE_KB` | Max chunk size per WQE | 128 |
| `UCCL_IB_HCA` | IB devices to use | auto |
| `UCCL_IB_GID_INDEX` | GID index for RoCE | -1 |

### NCCL Tuning

```bash
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
```

---

## References

- [UCCL Project](https://github.com/uccl-project/uccl)
- [UCCL Collective RDMA README](../collective/rdma/README.md)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
