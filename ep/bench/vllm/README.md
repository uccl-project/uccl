# vLLM + UCCL-EP Multi-Node Expert Parallel Deployment Guide

This guide provides example scripts and instructions for deploying vLLM with Expert Parallelism (EP) across multiple nodes on AWS p5en.

## 🚀 Installation

### 0. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install numpy torch setuptools
```

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
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
cat install.sh
# cuobjdump used by https://github.com/deepseek-ai/DeepGEMM/blob/9b680f428484625f4f35dc3617f134187c6bcd4a/csrc/jit/kernel_runtime.hpp#L44
# If you could not find cuobjdump in your servers, install it by: 
sudo apt install nvidia-cuda-toolkit -y
# If your server's cuobjdump is under /bin instead of $CUDA_HOME/bin, set soft link to make DeepGEMM happy: 
sudo ln -s /bin/cuobjdump /usr/local/cuda/bin/cuobjdump
./install.sh
uv pip install dist/*.whl --force-reinstall
```

Refer to [DeepGEMM Installation Guide](https://github.com/deepseek-ai/DeepGEMM#installation), if hitting any issues.

### 3. Install EP Kernels

Refer to [../../deep_ep_wrapper/README.md](../../deep_ep_wrapper/README.md) to install UCCL-EP's drop-in replacement for DeepEP.

Refer to vLLM's guide for the original DeepEP and pplx-kernels setup.

### 4. (Optional) AWS EFA Setup

For AWS instances with EFA, install AWS OFI-NCCL plugin, which is pre-installed on AWS Deep Learning AMIs

## ⚙️ Configuration

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

### Backend Selection

vLLM provides three EP communication backends:

| Backend | Use Case | Features | Best For |
|---------|----------|----------|----------|
| `pplx` | Single node | Chunked prefill support | Development, intra-node |
| `deepep_high_throughput` | Multi-node prefill | Grouped GEMM | High throughput, prefill-dominated |
| `deepep_low_latency` | Multi-node decode | CUDA graph support | Low latency, decode-dominated |
| `allgather_reducescatter` | Multi-node | NCCL-based | InfiniBand/EFA networks |

### Environment Setup

Edit the provided scripts (`launch_vllm_head.sh` and `launch_vllm_worker.sh`) to configure:

1. **Network interfaces** - Set `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`
1. **Backend** - Choose appropriate `VLLM_ALL2ALL_BACKEND`
1. **Model storage** - eg, `export HF_HOME=/emlfsx_southeast3/eml-folder/xzhiying`


## 🚢 Deployment

### Single Node Deployment

For single-node deployment (e.g., 8 GPUs on one node):

```bash
# Using pplx backend (recommended for single node)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
--all2all-backend pplx \
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--enable-expert-parallel
```

### Multi-Node Deployment (2+ Nodes)

#### Step 1: Start Node 0 (Primary)

On the **first node** (primary node that handles API requests):

```bash
bash launch_vllm_head.sh 10.4.164.146 13345 deepseek-ai/DeepSeek-V3-0324 16 8 1 8
```

#### Step 2: Start Node 1+ (Secondary)

On **each additional node** (secondary nodes in headless mode):

```bash
# Launch Node 1 (headless)
bash launch_vllm_worker.sh 10.4.164.146 13345 deepseek-ai/DeepSeek-V3-0324 16 8 1 8
```

**Arguments:**
- `10.4.164.146` - IP address of **Node 0**, should be the IP of the `NCCL_SOCKET_IFNAME`
- `13345` - RPC port
- `deepseek-ai/DeepSeek-V3-0324` - Same model as Node 1
- `16` - Total DP size
- `8` - Local DP size on this node
- `1` - Local TP size on this node
- `8` - For node 0, number of API servers; for others, starting rank (= sum of previous nodes' local DP)
