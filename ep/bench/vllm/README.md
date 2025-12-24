# vLLM + UCCL-EP Multi-Node Expert Parallel Deployment Guide

This guide provides example scripts and instructions for deploying vLLM with Expert Parallelism (EP) across multiple nodes on AWS p5en.

## 🚀 Installation

### 0. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install numpy torch
```

Note: we do not recommend installing to EFS shared file systems, as it is too slow. 

### 1. Install vLLM with EP Support

Follow the official guide:
```bash
# Install vLLM: latest version with timeout fix (https://github.com/vllm-project/vllm/pull/27444)
git clone https://github.com/vllm-project/vllm.git
cd vllm
# This may take 5-10 minutes.
uv pip install -e .
```

For detailed EP setup, refer to [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html)

### 2. Install DeepGEMM Library

DeepGEMM provides optimized kernels for MoE operations:

```bash
# Clone and install DeepGEMM
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
cat install.sh
# cuobjdump used by https://github.com/deepseek-ai/DeepGEMM/blob/9b680f428484625f4f35dc3617f134187c6bcd4a/csrc/jit/kernel_runtime.hpp#L44
sudo apt install nvidia-cuda-toolkit -y
# /bin/cuobjdump instead of $CUDA_HOME/bin/cuobjdump in cuda12.8
sudo ln -s /bin/cuobjdump /usr/local/cuda-12.8/bin/cuobjdump
./install.sh
uv pip install dist/*.whl --force-reinstall
```

Refer to [DeepGEMM Installation Guide](https://github.com/deepseek-ai/DeepGEMM#installation), if hitting any issues.

### 3. Install EP Kernels

Refer to [../deep_ep_wrapper/README.md](../deep_ep_wrapper/README.md) to install UCCL-EP's drop-in replacement for DeepEP.

Refer to vLLM's guide for the original DeepEP and pplx-kernels setup.

### 4. (Optional) AWS EFA Setup

For AWS instances with EFA, install AWS OFI-NCCL plugin, which is pre-installed on AWS Deep Learning AMIs


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

Edit the provided scripts (`launch_vllm_head.sh` and `launch_vllm_worker.sh`) to configure:

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
bash launch_vllm_head.sh $NODE1_IP 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

#### Step 2: Start Node 2+ (Secondary)

On **each additional node** (secondary nodes in headless mode):

```bash
# Use Node 1's IP (not this node's IP!)
NODE1_IP="10.1.59.30"

# Launch Node 2 (headless)
bash launch_vllm_worker.sh $NODE1_IP 13345 deepseek-ai/DeepSeek-V3-0324 16 8 8
```

**Arguments:**
- `NODE1_IP` - IP address of **Node 1** (primary)
- `13345` - Same RPC port as Node 1
- `deepseek-ai/DeepSeek-V3-0324` - Same model as Node 1
- `16` - Same total DP size as Node 1
- `8` - Local DP size on this node
- `8` - Starting rank (= sum of previous nodes' local DP)
