# vLLM + UCCL-EP Multi-Node Expert Parallel Deployment Guide

This guide provides example scripts and instructions for deploying vLLM with Expert Parallelism (EP) across multiple nodes on AWS p5en.

## Installation

### Rrerequisite

Run `nvcc --version` to see which cuda toolkit you are using. This will be the one all the following libraries compile with. 
Note that `nvidia-smi` shows the driver-supported max CUDA version instead of the cuda toolkit. 
Below assumes `cu128`. 

### 0. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install numpy setuptools pybind11
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 1. Install vLLM with EP Support

Follow the [vLLM official guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-wheels):
```bash
uv pip install vllm --torch-backend=cu128
```

If you use `cu130` or above, we suggest building vllm from source, as we find its wheel still relies on `cu12x` libcudart.so.12 (as of 02/21/2026). 
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# This may take 20-30 minutes.
uv pip install -e .
```

For EP details, refer to [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html)

### 2. Install DeepGEMM Library

DeepGEMM provides optimized kernels for MoE operations:

```bash
# Clone and install DeepGEMM
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && cd DeepGEMM

# cuobjdump used by https://github.com/deepseek-ai/DeepGEMM/blob/9b680f428484625f4f35dc3617f134187c6bcd4a/csrc/jit/kernel_runtime.hpp#L44
# If you could not find cuobjdump in your servers, install it by: 
sudo apt install nvidia-cuda-toolkit -y
# If your server's cuobjdump is under /bin instead of $CUDA_HOME/bin, set soft link to make DeepGEMM happy: 
sudo ln -s /bin/cuobjdump /usr/local/cuda/bin/cuobjdump

# Ignore the final install error, as it was targetting non-uv env
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

Find your network interface and IP:

```bash
# List all network interfaces
ip addr show

# Common interface names:
# - eth0, eno1, enp0s3 (Ethernet)
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

Edit the provided script `launch_vllm.sh` to configure:

1. **Network interfaces** - Set `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`
1. **Backend** - Choose appropriate `VLLM_ALL2ALL_BACKEND`
1. **Model storage** - Set `HF_HOME` to some folder with large storage
1. **DeepGEMM JIT cache** - Set `DG_JIT_CACHE_DIR` to some non-shared folder on each node


## Deployment

Use the unified script **`launch_vllm.sh`** with role `head` or `worker`:

```bash
# Usage: launch_vllm.sh <head|worker> <NODE1_IP> [RPC_PORT] [MODEL] [BACKEND] [TOTAL_DP_SIZE] [LOCAL_DP_SIZE] [LOCAL_TP_SIZE] [API_SERVERS_or_START_RANK]
```

### Step 1: Start Node 0 (Primary)

On the **first node** (primary node that handles API requests):

```bash
bash launch_vllm.sh head 172.31.41.55 13345 deepseek-ai/DeepSeek-V3-0324 deepep_high_throughput 2 1 8 1
```

### Step 2: Start Node 1+ (Secondary)

On **each additional node** (secondary nodes in headless mode):

```bash
bash launch_vllm.sh worker 172.31.41.55 13345 deepseek-ai/DeepSeek-V3-0324 deepep_high_throughput 2 1 8 1
```

**Arguments (positional):**
- `head` | `worker` - Role: primary (API) or secondary (headless).
- `NODE1_IP` - IP of Node 0 (use `hostname -I` on Node 0).
- `RPC_PORT` - e.g. 13345.
- `MODEL` - e.g. deepseek-ai/DeepSeek-V3-0324.
- `BACKEND` - e.g. allgather_reducescatter, deepep_high_throughput, deepep_low_latency.
- `TOTAL_DP_SIZE` - Total data-parallel size across all nodes (e.g. 2 for 2×8-GPU nodes).
- `LOCAL_DP_SIZE` - Data-parallel size on this node (e.g. 1).
- `LOCAL_TP_SIZE` - Tensor-parallel size on this node (e.g. 8).
- **Head:** 9th = API_SERVERS (e.g. 1). **Worker:** 9th = START_RANK (node 1: 1; node 2: 2; etc.).

## vLLM Serving Benchmark Results

```
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 1000 \
  --request-rate 10 \
  --max-concurrency 256 \
  --seed 42 \
  --ignore-eos \
  --save-result \
  --result-dir ./results \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99
```

**Model:** `deepseek-ai/DeepSeek-V3-0324`  
**Request rate:** 10 RPS  
**Prompts:** 1000  
**Input / Output tokens:** 1024 / 256  
**Max concurrency:** 256  

| Mode | Req Throughput (req/s) | Output Tok Throughput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|------|------------------------|-------------------------------|----------------|---------------|----------------|---------------|
| Allgather + ReduceScatter | 8.93 | 2285.11 | 303.50 | 655.98 | 81.11 | 95.55 |
| UCCL-EP - Low Latency | 9.22 | 2359.13 | 278.52 | 775.17 | 59.35 | 78.61 |
