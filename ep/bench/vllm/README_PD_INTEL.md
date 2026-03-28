# vLLM PD Disaggregation on Intel RDMA NICs

This guide describes how to run vLLM with Prefill-Decode (PD) disaggregation
using Expert Parallelism (using UCCL EP) and KV Cache transfer (using NIXL with UCCL P2P) on Intel irdma NICs.

**Architecture**: Separate prefill and decode EP groups communicate KV cache
via RDMA using NixlConnector with UCCL-P2P backend. EP all-to-all uses
UCCL-EP RDMA for inter-node expert communication.

## Prerequisites

- NVIDIA GPU with CUDA 12.x
- Intel irdma NIC (e.g., `irdma-mkp0`)
- Python 3.12, conda environment (e.g., `uccl-vllm`)
- vLLM 0.17.1+

## 1. Build and Install Dependencies

### 1.1 Install Python packages

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install meson zmq
```

### 1.2 Build UCCL-EP

```bash
cd ~/uccl/uccl/ep
make -j
pip install .
```

### 1.3 Build and Install UCCL-P2P

UCCL-P2P provides the RDMA library used by NIXL's UCCL backend for KV cache
transfer between prefill and decode nodes.

```bash
cd ~/uccl/uccl/p2p
make PREFIX=$HOME/install/uccl -j
make PREFIX=$HOME/install/uccl install
```

This installs:
- `$HOME/install/uccl/lib/libuccl_p2p.so`
- `$HOME/install/uccl/include/uccl_engine.h`
- `$HOME/install/uccl/include/common.h`

### 1.4 Build and Install NIXL

NIXL provides the NixlConnector used by vLLM for RDMA-based KV cache transfer.
It discovers UCCL-P2P via `LIBRARY_PATH` and `CPATH`.

```bash
# Make UCCL-P2P discoverable
export LIBRARY_PATH=$HOME/install/uccl/lib:$LIBRARY_PATH
export CPATH=$HOME/install/uccl/include:$CPATH

# Clone and build NIXL
cd $HOME
git clone https://github.com/ai-dynamo/nixl.git
cd ~/nixl
meson setup build --prefix=$HOME/install/nixl
cd build
ninja
yes | ninja install
cd ..
pip install .
```

> **Note**: If GDS (GPU Direct Storage) is not installed, add
> `-Ddisable_gds_backend=true` to the `meson setup` command. To check:
> `ldconfig -p | grep cufile`

### 1.5 Fix NIXL module name for conda

When building from source, NIXL installs as `nixl_cu12` but vLLM imports `nixl`.
Create a symlink to bridge the gap:

```bash
CONDA_NS=uccl-vllm  # your conda env name
ln -s $HOME/miniforge3/envs/$CONDA_NS/lib/python3.12/site-packages/nixl_cu12 \
      $HOME/miniforge3/envs/$CONDA_NS/lib/python3.12/site-packages/nixl
```

### 1.6 Verify installation

```bash
python -c "from nixl._api import nixl_agent; print('NIXL: ok')"
python -c "import uccl; print('UCCL: ok')"
```

## 2. Runtime Environment

Ensure UCCL-P2P and NIXL libraries are on `LD_LIBRARY_PATH` before launching:

```bash
export LD_LIBRARY_PATH=$HOME/install/uccl/lib:$HOME/install/nixl/lib:$LD_LIBRARY_PATH
```

## 3. Launch PD Disaggregation

All commands are run from the UCCL repo root directory.

### Script usage

```
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh <role> <PREFILL_HEAD_IP> <DECODE_HEAD_IP> [START_RANK]
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-30B-A3B-FP8` | HuggingFace model name |
| `BACKEND` | `deepep_low_latency` | EP all-to-all backend (`deepep_low_latency` or `allgather_reducescatter`) |
| `PREFILL_DP_SIZE` | `2` | Number of prefill nodes |
| `DECODE_DP_SIZE` | `2` | Number of decode nodes |
| `NIXL_BACKEND` | `UCCL` | KV transfer backend (`UCCL`, `UCX`, `Mooncake`) |

### Ports

| Port | Purpose |
|------|---------|
| 8100 | Prefill API server |
| 8000 | Decode API server |
| 9000 | Disagg proxy (user-facing) |
| 13345 | EP RPC coordination |

### 3.1 Example: 2 Prefill + 2 Decode (4 nodes)

Launch each command on its respective node:

```bash
# Node 0 (Prefill Head)
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh prefill-head <PREFILL_HEAD_IP> <DECODE_HEAD_IP>

# Node 1 (Prefill Worker)
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh prefill-worker <PREFILL_HEAD_IP> <DECODE_HEAD_IP>

# Node 2 (Decode Head)
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh decode-head <PREFILL_HEAD_IP> <DECODE_HEAD_IP>

# Node 3 (Decode Worker)
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh decode-worker <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
```

### 3.2 Launch the Disagg Proxy

Run on any node (can be one of the prefill/decode nodes):

```bash
bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh proxy <PREFILL_HEAD_IP> <DECODE_HEAD_IP>
```

## 4. Verify Successful Launch

### Prefill Head

```
(ApiServer_0 pid=224788) INFO 03-28 02:01:52 [api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8100
(ApiServer_0 pid=224788) INFO:     Application startup complete.
(ApiServer_1 pid=224789) INFO:     Application startup complete.
```

### Prefill Worker

```
(EngineCore_DP1 pid=291560) INFO 03-28 02:01:50 [nixl_connector.py:538] Initializing NIXL Scheduler 9fe8b541-...
(EngineCore_DP1 pid=291560) INFO 03-28 02:01:50 [vllm.py:747] Asynchronous scheduling is enabled
```

### Decode Head

```
(ApiServer_0 pid=164959) INFO 03-28 02:02:25 [api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8000
(ApiServer_0 pid=164959) INFO:     Application startup complete.
(ApiServer_1 pid=164960) INFO:     Application startup complete.
```

### Decode Worker

```
(EngineCore_DP1 pid=1288850) INFO 03-28 02:02:23 [nixl_connector.py:538] Initializing NIXL Scheduler 6fb96535-...
(EngineCore_DP1 pid=1288850) INFO 03-28 02:02:23 [vllm.py:747] Asynchronous scheduling is enabled.
```

### Proxy

```
Listening on http://0.0.0.0:9000
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

## 5. Test with a Single Request

Send a request to the proxy:

```bash
curl -s http://<PROXY_IP>:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-FP8",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 1000
  }' | jq -r '.choices[0].message.content'
```

## 6. Run Benchmark

```bash
vllm bench serve \
  --backend openai-chat \
  --host <PROXY_IP> --port 9000 \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 50 --request-rate 2 --max-concurrency 16 \
  --seed 42 --ignore-eos --save-result --result-dir ./results_2p2d \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
```

## 7. Other Configurations

For other PD configurations (2P+1D, 1P+2D, 1P+1D) and baseline mode (EP-only, no disaggregation),
refer to the script header comments in `ep/bench/vllm/launch_vllm_pd_intel_nic.sh`.

## 8. EP Backend Selection

The `BACKEND` env var controls the EP all-to-all communication pattern:

| Backend | Description |
|---------|-------------|
| `deepep_low_latency` | DeepEP: fine-grained per-expert RDMA transfers via UCCL-EP |
| `allgather_reducescatter` | NCCL AllGather + ReduceScatter: fewer, larger collective ops |

```bash
# Use AllGather+ReduceScatter backend
BACKEND=allgather_reducescatter bash ep/bench/vllm/launch_vllm_pd_intel_nic.sh prefill-head ...
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nixl'`**: Create the `nixl` -> `nixl_cu12` symlink (section 1.5).
- **RDMA failures / `ibv_reg_dmabuf_mr` errors**: Ensure `NCCL_IB_HCA` and `UCCL_IB_HCA` match your irdma device (`ibv_devinfo`).
- **Timeout waiting for workers**: Check that all nodes use the same `PREFILL_HEAD_IP` / `DECODE_HEAD_IP` and that port 13345 is reachable.
- **KV transfer hangs**: Verify `LD_LIBRARY_PATH` includes `$HOME/install/uccl/lib` on all nodes.
