# Agent Runbook: vLLM Built-in EP and Lite-EP on l40/l41

Last verified: 2026-06-04

This runbook is for future agents that need to recreate the working vLLM
environment, rebuild Lite-EP, sync l41, and run both the vLLM built-in EP
baseline and the Lite-EP DeepEPv2 path.

The review-oriented companion document is
[vllm-lite-ep-review.md](vllm-lite-ep-review.md).

## Known-good Target

| Item | Value |
| --- | --- |
| l40 host | `mibura-sky-test-01`, IB IP `10.10.55.1` |
| l41 host | `mibura-sky-test-02`, IB IP `10.10.55.2` |
| GPUs | 4 x NVIDIA L4 per node |
| Model | `/ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat` on both nodes |
| vLLM | 0.22.0 |
| Torch import version | 2.11.0+cu129 |
| Torch CUDA | 12.9 |
| Transformers | 4.57.6 |
| Triton | 3.6.0 |
| Parallelism | TP=1, DP=8, EP=8 |
| Built-in backend | `allgather_reducescatter` |
| Lite backend | `deepep_high_throughput` |
| Lite transport | `LITE_EP_TRANSPORT=uccl-no-gdr`, `LITE_EP_NVLINK=0` |

On l41, `importlib.metadata.version('torch')` may report stale metadata
(`2.10.0`) because the SSD env was copied. Trust `import torch;
torch.__version__`, which reports `2.11.0+cu129` in the validated setup.

## Paths

On l40:

```bash
REPO=/home/yangz/nfs/zhongjie/uccl
EP_DIR=$REPO/experimental/lite/lite-ep
PY_ENV=/home/yangz/nfs/miniconda3/envs/uccl
NCCL_ROOT=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
MODEL=/ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat
```

On l41:

```bash
EP_DIR=/ssd1/dsv2lite/src/lite-ep
PY_ENV=/ssd1/dsv2lite/envs/uccl312
NCCL_ROOT=/ssd1/dsv2lite/deps/nccl
MODEL=/ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat
```

Use `ssh 10.10.55.2`, not DNS names, for the high-speed path.

## Clean Stale Runs

Always clean old vLLM processes before relaunching. Use bracketed `pkill`
patterns so the cleanup shell does not kill itself.

```bash
pkill -f '[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true
pkill -f '[r]un_vllm_' || true
ssh -o ConnectTimeout=10 10.10.55.2 "pkill -f '[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true; pkill -f '[r]un_vllm_' || true"
```

Check that nothing remains:

```bash
pgrep -af '[r]un_vllm_|[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true
ssh -o ConnectTimeout=10 10.10.55.2 "pgrep -af '[r]un_vllm_|[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true"
```

## Verify or Recreate the Python Environment

Preferred path: reuse the known-good l40 conda env and keep l41 on SSD.
Avoid running long vLLM jobs from the NFS env on l41.

Verify l40:

```bash
source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
conda activate uccl
python - <<'PY'
import importlib.metadata as md
import torch
for pkg in ['vllm', 'torch', 'transformers', 'triton']:
    try:
        print(f'{pkg}={md.version(pkg)}')
    except Exception as exc:
        print(f'{pkg}=missing:{type(exc).__name__}')
print(f'torch_import={torch.__version__}')
print(f'torch_cuda={torch.version.cuda}')
PY
```

Expected key values on l40:

```text
vllm=0.22.0
torch=2.11.0+cu129
transformers=4.57.6
triton=3.6.0
torch_import=2.11.0+cu129
torch_cuda=12.9
```

If vLLM is missing or too old, install vLLM 0.22 in the `uccl` env. Do
not use vLLM 0.19 with this torch runtime.

```bash
python -m pip install --upgrade 'vllm==0.22.0'
```

If l41's SSD env is missing or broken, sync the known-good env from l40
to l41 SSD. The copied env is what the validated run used.

```bash
rsync -aH --delete --whole-file --info=progress2 \
  -e 'ssh -T -c aes128-gcm@openssh.com -o Compression=no -o ConnectTimeout=10' \
  /home/yangz/nfs/miniconda3/envs/uccl/ \
  10.10.55.2:/ssd1/dsv2lite/envs/uccl312/
```

Verify l41 by importing torch, not by trusting torch dist-info metadata:

```bash
ssh -o ConnectTimeout=10 10.10.55.2 "/ssd1/dsv2lite/envs/uccl312/bin/python - <<'PY'
import importlib.metadata as md
import torch
print('vllm=' + md.version('vllm'))
print('torch_metadata=' + md.version('torch'))
print('torch_import=' + torch.__version__)
print('torch_cuda=' + str(torch.version.cuda))
PY"
```

## Verify Model Weights

The model must already be present on both nodes. l41 may not be able to
resolve Hugging Face reliably, so do not depend on online download during
the smoke run.

```bash
ls -lh /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat/model-*.safetensors
ssh -o ConnectTimeout=10 10.10.55.2 "ls -lh /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat/model-*.safetensors"
```

The validated model has four safetensors shards totaling about 30 GiB.
Its config is `deepseek_v2` with 64 routed experts, which divides cleanly
across 8 EP ranks.

## Build Lite-EP on l40

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep
source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
conda activate uccl
make SM=89 PYTHON=python
```

The build should include `-DUSE_RECEIVER_BARRIER` unless explicitly
disabled with `EP_UCCL_RECEIVER_BARRIER=0`.

Run the vLLM shim unit test. Suppress the NCCL duplicate-runtime guard for
this test because the test imports torch before `deep_ep`.

```bash
EP_SUPPRESS_NCCL_CHECK=1 python -m unittest tests.utils.test_vllm_buffer_compat -v
```

Expected:

```text
Ran 3 tests
OK
```

## Sync Lite-EP and NCCL to l41

Sync the Lite-EP tree after every rebuild:

```bash
rsync -aH --delete --whole-file --info=progress2 \
  -e 'ssh -T -c aes128-gcm@openssh.com -o Compression=no -o ConnectTimeout=10' \
  /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep/ \
  10.10.55.2:/ssd1/dsv2lite/src/lite-ep/
```

If l41's NCCL copy is missing, sync the NCCL root used at build time:

```bash
ssh -o ConnectTimeout=10 10.10.55.2 'mkdir -p /ssd1/dsv2lite/deps'
rsync -aH --delete --whole-file --info=progress2 \
  -e 'ssh -T -c aes128-gcm@openssh.com -o Compression=no -o ConnectTimeout=10' \
  /home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl/ \
  10.10.55.2:/ssd1/dsv2lite/deps/nccl/
```

Check extension timestamps and sizes match:

```bash
stat -c '%y %s %n' /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep/deep_ep/_C.cpython-312-x86_64-linux-gnu.so
ssh -o ConnectTimeout=10 10.10.55.2 "stat -c '%y %s %n' /ssd1/dsv2lite/src/lite-ep/deep_ep/_C.cpython-312-x86_64-linux-gnu.so"
```

## Create Launch Scripts

The validated scripts were created under `/tmp`. They are intentionally
separate for built-in baseline and Lite-EP so the built-in run cannot
accidentally inherit Lite-EP environment variables.

Create `/tmp/run_vllm_builtin_ep_2node.sh` on l40, then copy it to l41:

```bash
cat >/tmp/run_vllm_builtin_ep_2node.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROLE=${1:?role head|worker}
MASTER_ADDR=${2:-10.10.55.1}
RPC_PORT=${3:-13447}
MODEL=${4:-/ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat}
START_OR_API=${5:-1}

if [[ "$(hostname)" == "mibura-sky-test-02" ]]; then
  PYTHON_BIN=/ssd1/dsv2lite/envs/uccl312/bin/python
  export HF_HOME=/ssd1/dsv2lite/hf
  export VLLM_CACHE_ROOT=/ssd1/dsv2lite/vllm-cache
  export TMPDIR=/ssd1/dsv2lite/tmp
  export TRITON_CACHE_DIR=/ssd1/dsv2lite/triton-cache
  LOG_DIR=/ssd1/dsv2lite/logs
else
  source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
  conda activate uccl
  PYTHON_BIN=$CONDA_PREFIX/bin/python
  export HF_HOME=/ssd1/yangz/hf-cache
  export VLLM_CACHE_ROOT=/ssd1/yangz/vllm-cache
  export TMPDIR=/ssd1/yangz/tmp
  export TRITON_CACHE_DIR=/ssd1/yangz/triton-cache
  LOG_DIR=/ssd1/yangz/vllm-logs
fi
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_WORLD_SIZE=4
unset PYTHONPATH
unset LITE_EP_TRANSPORT
unset LITE_EP_NVLINK
unset EP_USE_UCCL_PROXY
unset UCCL_FORCE_NO_GDR
unset EP_FORCE_HOST_WINDOW
unset EP_FORCE_NO_NVLINK
unset EP_DISABLE_GIN
unset EP_UCCL_PROXY_ACTIVE
unset EP_NCCL_ROOT_DIR
unset LD_PRELOAD
export NCCL_SOCKET_IFNAME=ibp55s0f0
export GLOO_SOCKET_IFNAME=ibp55s0f0
export TP_SOCKET_IFNAME=ibp55s0f0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export VLLM_USE_DEEP_GEMM=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export VLLM_LOGGING_LEVEL=INFO
export OMP_NUM_THREADS=1

ENV_PREFIX=$(dirname "$(dirname "$PYTHON_BIN")")
PY_VER=$($PYTHON_BIN - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)
CU13_LIB=$ENV_PREFIX/lib/$PY_VER/site-packages/nvidia/cu13/lib
TORCH_LIB=$($PYTHON_BIN - <<'PY'
import os, torch
print(os.path.join(torch.__path__[0], 'lib'))
PY
)
export LD_LIBRARY_PATH="$CU13_LIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

COMMON_ARGS=(
  "$MODEL"
  --trust-remote-code
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --tensor-parallel-size 1
  --data-parallel-size 8
  --data-parallel-size-local 4
  --data-parallel-address "$MASTER_ADDR"
  --data-parallel-rpc-port "$RPC_PORT"
  --max-model-len 512
  --gpu-memory-utilization 0.80
  --served-model-name deepseek-v2-lite-chat-builtin-ep
  --enforce-eager
)

if [[ "$ROLE" == "head" ]]; then
  exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${COMMON_ARGS[@]}" --api-server-count "$START_OR_API"
elif [[ "$ROLE" == "worker" ]]; then
  exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${COMMON_ARGS[@]}" --data-parallel-start-rank "$START_OR_API" --headless
else
  echo "role must be head or worker" >&2
  exit 2
fi
EOF
chmod +x /tmp/run_vllm_builtin_ep_2node.sh
scp /tmp/run_vllm_builtin_ep_2node.sh 10.10.55.2:/tmp/run_vllm_builtin_ep_2node.sh
```

Create `/tmp/run_vllm_lite_ep_2node.sh` on l40, then copy it to l41:

```bash
cat >/tmp/run_vllm_lite_ep_2node.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROLE=${1:?role head|worker}
MASTER_ADDR=${2:-10.10.55.1}
RPC_PORT=${3:-13445}
MODEL=${4:-deepseek-ai/DeepSeek-V2-Lite-Chat}
START_OR_API=${5:-1}

EP_DIR=/home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep
NCCL_ROOT=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
STUB=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
PYTHON_BIN=/home/yangz/nfs/miniconda3/envs/uccl/bin/python

if [[ "$(hostname)" == "mibura-sky-test-02" ]]; then
  PYTHON_BIN=/ssd1/dsv2lite/envs/uccl312/bin/python
  EP_DIR=/ssd1/dsv2lite/src/lite-ep
  NCCL_ROOT=/ssd1/dsv2lite/deps/nccl
  STUB=
  export HF_HOME=/ssd1/dsv2lite/hf
  export VLLM_CACHE_ROOT=/ssd1/dsv2lite/vllm-cache
  export TMPDIR=/ssd1/dsv2lite/tmp
  export TRITON_CACHE_DIR=/ssd1/dsv2lite/triton-cache
  LOG_DIR=/ssd1/dsv2lite/logs
else
  source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
  conda activate uccl
  PYTHON_BIN=$CONDA_PREFIX/bin/python
  export HF_HOME=/ssd1/yangz/hf-cache
  export VLLM_CACHE_ROOT=/ssd1/yangz/vllm-cache
  export TMPDIR=/ssd1/yangz/tmp
  export TRITON_CACHE_DIR=/ssd1/yangz/triton-cache
  LOG_DIR=/ssd1/yangz/vllm-logs
fi
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_WORLD_SIZE=4
export PYTHONPATH="$EP_DIR:${PYTHONPATH:-}"
export EP_NCCL_ROOT_DIR=$NCCL_ROOT
export EP_SUPPRESS_NCCL_CHECK=1
export LITE_EP_TRANSPORT=uccl-no-gdr
export LITE_EP_NVLINK=0
export NCCL_GIN_TYPE=2
export DISABLE_SM90_FEATURES=1
export EP_JIT_CACHE_DIR=$EP_DIR/.jit-cache
export NCCL_SOCKET_IFNAME=ibp55s0f0
export GLOO_SOCKET_IFNAME=ibp55s0f0
export TP_SOCKET_IFNAME=ibp55s0f0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export UCCL_IB_HCA=mlx5_0,mlx5_1
export EP_UCCL_HOST_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.10\.' | head -1)
export VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=INFO

ENV_PREFIX=$(dirname "$(dirname "$PYTHON_BIN")")
PY_VER=$($PYTHON_BIN - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)
CU13_LIB=$ENV_PREFIX/lib/$PY_VER/site-packages/nvidia/cu13/lib
TORCH_LIB=$($PYTHON_BIN - <<'PY'
import os, torch
print(os.path.join(torch.__path__[0], 'lib'))
PY
)
export LD_LIBRARY_PATH="$CU13_LIB:$TORCH_LIB:$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$STUB" ]]; then
  export LD_PRELOAD="$NCCL_ROOT/lib/libnccl.so.2:$STUB${LD_PRELOAD:+:$LD_PRELOAD}"
else
  export LD_PRELOAD="$NCCL_ROOT/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

COMMON_ARGS=(
  "$MODEL"
  --trust-remote-code
  --enable-expert-parallel
  --all2all-backend deepep_high_throughput
  --tensor-parallel-size 1
  --data-parallel-size 8
  --data-parallel-size-local 4
  --data-parallel-address "$MASTER_ADDR"
  --data-parallel-rpc-port "$RPC_PORT"
  --max-model-len 512
  --gpu-memory-utilization 0.80
  --served-model-name deepseek-v2-lite-chat-ep
)

if [[ "$ROLE" == "head" ]]; then
  exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${COMMON_ARGS[@]}" --api-server-count "$START_OR_API"
elif [[ "$ROLE" == "worker" ]]; then
  exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "${COMMON_ARGS[@]}" --data-parallel-start-rank "$START_OR_API" --headless
else
  echo "role must be head or worker" >&2
  exit 2
fi
EOF
chmod +x /tmp/run_vllm_lite_ep_2node.sh
scp /tmp/run_vllm_lite_ep_2node.sh 10.10.55.2:/tmp/run_vllm_lite_ep_2node.sh
```

## Run the Built-in vLLM EP Baseline

Run the worker on l41 first, then the head on l40:

```bash
ssh -o ConnectTimeout=10 10.10.55.2 'mkdir -p /ssd1/dsv2lite/logs && cd /ssd1/dsv2lite && /tmp/run_vllm_builtin_ep_2node.sh worker 10.10.55.1 13447 /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat 4 2>&1 | tee /ssd1/dsv2lite/logs/vllm_builtin_worker.log'
```

In another terminal:

```bash
mkdir -p /ssd1/yangz/vllm-logs
cd /ssd1/yangz
/tmp/run_vllm_builtin_ep_2node.sh head 10.10.55.1 13447 /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat 1 2>&1 | tee /ssd1/yangz/vllm-logs/vllm_builtin_head.log
```

Wait until the API is ready:

```bash
curl -sS --max-time 10 http://127.0.0.1:8000/v1/models | python -m json.tool
```

Send the built-in baseline prompt:

```bash
curl -sS --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2-lite-chat-builtin-ep",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "In one sentence, say hello and confirm this is a two-node eight-GPU vLLM expert-parallel smoke test."}
    ],
    "max_tokens": 64,
    "temperature": 0
  }' | python -m json.tool
```

Expected answer contains:

```text
Hello! Yes, this is a two-node eight-GPU vLLM expert-parallel smoke test.
```

Stop both services before switching to Lite-EP.

## Run vLLM Through Lite-EP

Clean stale processes first, then start the l41 worker:

```bash
ssh -o ConnectTimeout=10 10.10.55.2 'mkdir -p /ssd1/dsv2lite/logs && cd /ssd1/dsv2lite && /tmp/run_vllm_lite_ep_2node.sh worker 10.10.55.1 13445 /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat 4 2>&1 | tee /ssd1/dsv2lite/logs/vllm_ep_worker.log'
```

Start the l40 head:

```bash
mkdir -p /ssd1/yangz/vllm-logs
cd /ssd1/yangz
/tmp/run_vllm_lite_ep_2node.sh head 10.10.55.1 13445 /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat 1 2>&1 | tee /ssd1/yangz/vllm-logs/vllm_ep_head.log
```

If debugging UCCL startup, prefix both launches with `EP_UCCL_DEBUG=1`,
but do not leave debug enabled for routine runs because it writes very
large logs.

Check readiness:

```bash
curl -sS --max-time 10 http://127.0.0.1:8000/v1/models | python -m json.tool
```

Expected served model id:

```text
deepseek-v2-lite-chat-ep
```

Send the Lite-EP smoke prompt:

```bash
curl -sS --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2-lite-chat-ep",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "In one sentence, say hello and confirm this is a two-node eight-GPU vLLM expert-parallel smoke test using the lite EP path."}
    ],
    "max_tokens": 64,
    "temperature": 0
  }' | python -m json.tool
```

Expected answer contains:

```text
Hello! Yes, this is a two-node eight-GPU vLLM expert-parallel smoke test using the lite EP path.
```

## Run Official vLLM Serving Benchmarks

Use vLLM's official serving benchmark after the corresponding server is
already running and `/v1/models` returns successfully. The benchmark client
needs the local tokenizer path; otherwise it treats the served model name
(`deepseek-v2-lite-chat-builtin-ep` or `deepseek-v2-lite-chat-ep`) as a
Hugging Face repo and fails with `RepositoryNotFoundError`.

The batch-size-1 comparison uses single concurrency and fixed random prompt /
output lengths:

- `--dataset-name random`
- `--random-input-len 67`
- `--random-output-len 64`
- `--num-warmups 8`
- `--num-prompts 8`
- `--max-concurrency 1`
- `--temperature 0`
- `--ignore-eos`

Lite-EP vLLM performance note: the compatibility shim keeps its
`ElasticBuffer` capacity in grow-only mode.  This avoids a per-MoE-layer
`all_reduce(MAX)` used only for buffer sizing and prevents prefill→decode
capacity shrink/recreate.  Set
`LITE_EP_VLLM_ALWAYS_ALLREDUCE_MAX_TOKENS=1` only when debugging uneven token
counts across EP ranks.

For the built-in EP server:

```bash
source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
conda activate uccl

vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-v2-lite-chat-builtin-ep \
  --tokenizer /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 67 \
  --random-output-len 64 \
  --num-warmups 8 \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos \
  --temperature 0 \
  --seed 20260604 \
  --save-result \
  --result-dir /tmp \
  --result-filename vllm_official_builtin_temp0_warm8.json \
  2>&1 | tee /tmp/vllm_official_builtin_temp0_warm8.log
```

For the Lite-EP server:

```bash
source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
conda activate uccl

vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-v2-lite-chat-ep \
  --tokenizer /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 67 \
  --random-output-len 64 \
  --num-warmups 8 \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos \
  --temperature 0 \
  --seed 20260604 \
  --save-result \
  --result-dir /tmp \
  --result-filename vllm_official_lite_temp0_warm8.json \
  2>&1 | tee /tmp/vllm_official_lite_temp0_warm8.log
```

Known-good official results on 2026-06-04:

| Path | Output tok/s | Request/s | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | Median TPOT | P99 TPOT | Mean ITL | P99 ITL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| built-in EP | 19.24 | 0.301 | 107.7 ms | 106.5 ms | 115.4 ms | 51.09 ms | 50.93 ms | 52.08 ms | 50.29 ms | 53.02 ms |
| Lite-EP before grow-only buffer reuse | 7.85 | 0.123 | 2023.3 ms | 1203.3 ms | 7339.96 ms | 97.35 ms | 97.96 ms | 100.40 ms | 95.83 ms | 1086.02 ms |
| Lite-EP after grow-only buffer reuse | 13.79 | 0.22 | 136.42 ms | 136.71 ms | 144.73 ms | 71.49 ms | 70.30 ms | 75.94 ms | 70.38 ms | 98.84 ms |

After the benchmark, run the cleanup commands below. Lite-EP shutdown can
print NCCL/TCPStore warnings if the head exits before every worker finishes its
process-group teardown; those warnings are post-run cleanup noise as long as
the benchmark itself reports `Failed requests: 0` and GPU processes are gone.

## Failure Signatures and Fixes

`Duplicate NCCL runtime found` during local unit tests:

- Cause: the test imports torch before `deep_ep`, so torch may load a
  different NCCL from the linked Lite-EP extension.
- Fix for tests: set `EP_SUPPRESS_NCCL_CHECK=1`.
- Runtime service also sets `EP_SUPPRESS_NCCL_CHECK=1` and preloads the
  intended NCCL with `LD_PRELOAD=$NCCL_ROOT/lib/libnccl.so.2`.

`Failed to register atomic_buffer_ptr MR: Bad address` or
`Timed out waiting for UCCL proxy readiness`:

- Rebuild with receiver barrier enabled.
- Ensure the Lite-EP tree on l41 is synced after rebuild.
- Confirm the code has page-aligned shared atomic per-rank slices and all
  proxy threads receive the shared atomic pointer. See
  [vllm-lite-ep-review.md](vllm-lite-ep-review.md) for source files.
- Relaunch with `EP_UCCL_DEBUG=1` and check that every rank/thread enters
  `run_dual`.

`ncclGin* is undefined` during a one-rank ad hoc smoke:

- This is a separate NCCL Gin JIT path and was not the validated vLLM
  Lite-EP path. Use the full 8-rank vLLM service to validate the shim.

vLLM worker stalls or NFS wait on l41:

- Use `/ssd1/dsv2lite/envs/uccl312` and `/ssd1/dsv2lite/src/lite-ep` on
  l41. Do not launch l41 vLLM workers from the NFS conda env.

## Post-run Cleanup

After a successful or failed run, stop the services and verify the API is
down:

```bash
pkill -f '[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true
pkill -f '[r]un_vllm_' || true
ssh -o ConnectTimeout=10 10.10.55.2 "pkill -f '[r]un_vllm_' || true; pkill -f '[v]llm serve /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat' || true"
curl -sS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null && echo 'api-still-up' || echo 'api-down'
```

Expected final line:

```text
api-down
```
