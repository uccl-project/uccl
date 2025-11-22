# vLLM Multi-Node Expert Parallel Deployment Guide

This guide provides example scripts and instructions for deploying vLLM with Expert Parallelism (EP) across multiple nodes.

## 🎯 Overview

**Expert Parallelism (EP)** allows experts in Mixture-of-Experts (MoE) models to be deployed on separate GPUs, increasing locality, efficiency, and throughput. EP is typically coupled with Data Parallelism (DP).

## 📦 Prerequisites

Before deploying vLLM with EP, ensure you have:

### Hardware Requirements

- **Multi-GPU nodes** (typically 8 GPUs per node)
- **High-speed interconnect** (InfiniBand, AWS EFA, or high-bandwidth Ethernet)
- **GPU memory** sufficient for model size + KV cache

### Software Requirements

- **Python 3.8+**
- **PyTorch** with CUDA support
- **vLLM** with EP support
- **Network access** between nodes

## 🚀 Installation

### 1. Install vLLM with EP Support

Follow the official guide:
```bash
# Install vLLM (latest version with EP support)
pip install vllm
```

For detailed EP setup, refer to:
📖 [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html)

### 2. Install DeepGEMM Library

DeepGEMM provides optimized kernels for MoE operations:

```bash
# Clone and install DeepGEMM
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
pip install -e .
```

📖 [DeepGEMM Installation Guide](https://github.com/deepseek-ai/DeepGEMM#installation)

### 3. Install EP Kernels

```bash
# Install DeepEP and pplx-kernels
# Follow vLLM's guide for EP kernels setup
```

### 4. (Optional) AWS EFA Setup

For AWS instances with EFA:

```bash
# Install AWS OFI-NCCL plugin
# This is pre-installed on AWS Deep Learning AMIs
sudo apt-get install aws-ofi-nccl
```

### 5. (Optional) Disaggregated Serving

For prefill/decode split deployments:

```bash
# Install gdrcopy, ucx, and nixl
pip install nixl

# For optimal performance, install gdrcopy
# See: https://github.com/NVIDIA/gdrcopy
```

## ⚙️ Configuration

### Backend Selection

vLLM provides three EP communication backends:

| Backend | Use Case | Features | Best For |
|---------|----------|----------|----------|
| `pplx` | Single node | Chunked prefill support | Development, intra-node |
| `deepep_high_throughput` | Multi-node prefill | Grouped GEMM | High throughput, prefill-dominated |
| `deepep_low_latency` | Multi-node decode | CUDA graph support | Low latency, decode-dominated |
| `allgather_reducescatter` | Multi-node | NCCL-based | InfiniBand/EFA networks |

### Network Interface Detection

Find your network interface:

```bash
# List all network interfaces
ip addr show

# Common interface names:
# - eth0, eno1, enp0s3 (Ethernet)
# - ib0, ib1 (InfiniBand)
# - enp74s0, ens5 (Custom/AWS EFA)
```

### Environment Setup

Edit the provided scripts (`launch_vllm_node1.sh` and `launch_vllm_node2.sh`) to configure:

1. **PYTHONPATH** - Paths to vLLM, DeepGEMM, and EP kernels
2. **LD_LIBRARY_PATH** - Path to PyTorch libraries
3. **Network interfaces** - Set `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`
4. **Backend** - Choose appropriate `VLLM_ALL2ALL_BACKEND`

## 🚢 Deployment

### Single Node Deployment

For single-node deployment (e.g., 8 GPUs on one node):

```bash
# Using pplx backend (recommended for single node)
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 \
    vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

### Multi-Node Deployment (2+ Nodes)

#### Step 1: Start Node 1 (Primary)

On the **first node** (primary node that handles API requests):

```bash
# Get Node 1's IP address
NODE1_IP=$(hostname -I | awk '{print $1}')

# Launch Node 1
bash launch_vllm_node1.sh $NODE1_IP 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

**Arguments:**
- `NODE1_IP` - IP address of Node 1
- `13345` - RPC port for coordination
- `deepseek-ai/DeepSeek-V3-0324` - Model to serve
- `16` - Total DP size (across all nodes)
- `8` - Local DP size (GPUs on this node)
- `8` - Number of API servers

#### Step 2: Start Node 2+ (Secondary)

On **each additional node** (secondary nodes in headless mode):

```bash
# Use Node 1's IP (not this node's IP!)
NODE1_IP="10.1.59.30"  # Replace with actual Node 1 IP

# Launch Node 2 (headless)
bash launch_vllm_node2.sh $NODE1_IP 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

**Arguments:**
- `NODE1_IP` - IP address of **Node 1** (primary)
- `13345` - Same RPC port as Node 1
- `deepseek-ai/DeepSeek-V3-0324` - Same model as Node 1
- `16` - Same total DP size as Node 1
- `8` - Local DP size on this node
- `8` - Starting rank (= sum of previous nodes' local DP)

### Example: 2-Node Deployment

**Configuration:**
- 2 nodes × 8 GPUs = 16 GPUs total
- DP size = 16 (8 per node)
- Model: DeepSeek-V3-0324

**Node 1 (10.1.59.30):**
```bash
bash launch_vllm_node1.sh 10.1.59.30 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

**Node 2 (10.1.60.57):**
```bash
bash launch_vllm_node2.sh 10.1.59.30 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

### Startup Sequence

1. **Start Node 1 first** - Wait for API servers to start
2. **Wait 30-60 seconds** - Allow model loading and initialization
3. **Start Node 2** - It will connect to Node 1 via RPC
4. **Verify connection** - Check logs for "Connected all rings"

