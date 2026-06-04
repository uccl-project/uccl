# vLLM Lite-EP Integration Review Notes

Last verified: 2026-06-04

This note records what was done to run vLLM expert parallelism on the
l40/l41 testbed first through vLLM's built-in EP path, then through the
Lite-EP DeepEPv2 shim. It is meant for code review, so it focuses on the
behavioral decisions, touched files, validation, and remaining caveats.

The command-oriented reproduction guide is
[vllm-lite-ep-agent-runbook.md](vllm-lite-ep-agent-runbook.md).

## Result

The final Lite-EP run completed a real vLLM chat-completions request on
two nodes and eight L4 GPUs:

| Item | Value |
| --- | --- |
| Nodes | l40 `10.10.55.1`, l41 `10.10.55.2` |
| GPUs | 4 per node, 8 total |
| Model | `/ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat` |
| vLLM | 0.22.0 |
| PyTorch import version | 2.11.0+cu129 |
| Parallelism | TP=1, DP=8, EP=8 |
| Built-in baseline backend | `allgather_reducescatter` |
| Lite backend | `deepep_high_throughput` |
| Lite transport | `LITE_EP_TRANSPORT=uccl-no-gdr`, `LITE_EP_NVLINK=0` |

Successful Lite-EP smoke response:

```text
Hello! Yes, this is a two-node eight-GPU vLLM expert-parallel smoke test using the lite EP path.
```

The head log showed:

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

The services were stopped after validation; no matching `vllm serve` or
`run_vllm_lite_ep_2node` processes were left on l40 or l41.

## Why vLLM Needed a Shim

vLLM 0.22.0 still imports the DeepEP v1 public surface, especially
`deep_ep.Buffer` and `deep_ep.Config`, for the `deepep_high_throughput`
all-to-all manager. Lite-EP exposes DeepEPv2's `ElasticBuffer` instead.
Directly pointing vLLM at Lite-EP therefore did not work until Lite-EP
provided a vLLM-compatible `Buffer` facade.

The chosen adapter is intentionally narrow:

- Implement the high-throughput `Buffer` API that vLLM uses.
- Back it with `ElasticBuffer.dispatch()` and `ElasticBuffer.combine()`.
- Reject DeepEP v1 low-latency APIs with `NotImplementedError` because
  their layout and lifecycle semantics differ from Lite-EP DeepEPv2.
- Resolve vLLM's EP CPU group to the matching device process group when
  vLLM passes the CPU group into the DeepEP manager.

Primary files:

- [../deep_ep/buffers/vllm_compat.py](../deep_ep/buffers/vllm_compat.py)
- [../deep_ep/__init__.py](../deep_ep/__init__.py)
- [../tests/utils/test_vllm_buffer_compat.py](../tests/utils/test_vllm_buffer_compat.py)

## Code Changes by Area

### Python vLLM Compatibility Layer

[../deep_ep/buffers/vllm_compat.py](../deep_ep/buffers/vllm_compat.py)
adds `Config` and `Buffer` so vLLM can keep using its existing DeepEP v1
integration path. The shim records `num_experts` during
`get_dispatch_layout()`, lazily constructs an `ElasticBuffer` on first
dispatch, and reuses it while the token/hidden/top-k shape remains
unchanged.

[../deep_ep/__init__.py](../deep_ep/__init__.py) exports `Buffer` and
`Config` from the package root so `import deep_ep; deep_ep.Buffer` works
as vLLM expects.

[../tests/utils/test_vllm_buffer_compat.py](../tests/utils/test_vllm_buffer_compat.py)
uses a fake `ElasticBuffer` to test the vLLM-facing contract without
starting CUDA or distributed workers.

### Build and Include Fixes

The extension needed a few include/build fixes after adding the vLLM
surface and receiver-barrier mode:

- [../setup.py](../setup.py) now defines `USE_RECEIVER_BARRIER` by
  default unless `EP_UCCL_RECEIVER_BARRIER=0` is set.
- [../csrc/python_api.cpp](../csrc/python_api.cpp) includes pybind/torch
  before `elastic/buffer.hpp` to avoid pybind type visibility issues.
- [../csrc/kernels/backend/api.cuh](../csrc/kernels/backend/api.cuh)
  includes `pybind11/pytypes.h`.
- [../csrc/kernels/elastic/api.hpp](../csrc/kernels/elastic/api.hpp)
  includes common layout/math headers used by the elastic declarations.

### UCCL Receiver-Barrier and Atomic Buffer Fixes

The first Lite-EP vLLM prompt failed in UCCL proxy startup with:

```text
Failed to register atomic_buffer_ptr MR: Bad address
Timed out waiting for UCCL proxy readiness
```

The successful fix has three parts:

1. Avoid native RDMA atomics under receiver-barrier mode.

   [../csrc/uccl/src/rdma.cpp](../csrc/uccl/src/rdma.cpp) and
   [../csrc/uccl/src/proxy.cpp](../csrc/uccl/src/proxy.cpp) no longer
   request `IBV_ACCESS_REMOTE_ATOMIC` for the main window, atomic window,
   or QP access flags when `USE_RECEIVER_BARRIER` is defined. The atomic
   MR also avoids `IBV_ACCESS_REMOTE_READ` in receiver-barrier mode; the
   remote side only needs remote writes for the Lite-EP signal path.

2. Make shared atomic slices page-aligned.

   [../csrc/elastic/buffer.hpp](../csrc/elastic/buffer.hpp) rounds the
   per-rank shared atomic slice up to 4096 bytes while still zeroing only
   `kAtomicBufferSize` bytes. This prevents local ranks from registering
   non-page-aligned POSIX shared-memory slices.

3. Give every proxy thread the same atomic buffer pointer.

   [../csrc/uccl/src/uccl_proxy.cpp](../csrc/uccl/src/uccl_proxy.cpp) and
   [../csrc/uccl/include/uccl_proxy.hpp](../csrc/uccl/include/uccl_proxy.hpp)
   now initialize `atomic_buffer_ptr_` for all proxy threads when a shared
   atomic window exists. Previously only thread 0 received the pointer,
   which was incompatible with `kNumProxyThs=4`: remote writes can arrive
   on any per-thread QP, and each thread must advertise valid atomic MR
   metadata.

After these changes, all ranks on both nodes entered `run_dual`, atomic MR
registration succeeded, and the vLLM prompt completed.

## Validation Performed

Build:

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-ep
source /home/yangz/nfs/miniconda3/etc/profile.d/conda.sh
conda activate uccl
make SM=89 PYTHON=python
```

Unit test:

```bash
EP_SUPPRESS_NCCL_CHECK=1 python -m unittest tests.utils.test_vllm_buffer_compat -v
```

Result:

```text
Ran 3 tests in 0.275s
OK
```

Built-in vLLM baseline:

- Backend: `allgather_reducescatter`
- Model name: `deepseek-v2-lite-chat-builtin-ep`
- Result: `/v1/chat/completions` returned HTTP 200 with the expected
  two-node eight-GPU confirmation.

Lite-EP vLLM validation:

- Backend: `deepep_high_throughput`
- Model name: `deepseek-v2-lite-chat-ep`
- Result: `/v1/models` returned the served model, then
  `/v1/chat/completions` returned HTTP 200 with the expected Lite-EP
  confirmation.

The Lite-EP logs also showed no remaining instances of:

```text
Failed to register atomic_buffer_ptr MR
Timed out waiting for UCCL proxy readiness
```

Official vLLM serving benchmark:

```bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model <served-model-name> \
  --tokenizer /ssd1/dsv2lite/models/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 67 --random-output-len 64 \
  --num-warmups 8 --num-prompts 8 \
  --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0 \
  --seed 20260604 --save-result
```

The tokenizer path is required; otherwise `vllm bench serve` treats the served
model name as a Hugging Face repo and fails before sending requests. The
official batch-size-1 results were:

| Path | Served model | Output tok/s | Mean TTFT | Mean TPOT | Mean ITL |
| --- | --- | ---: | ---: | ---: | ---: |
| built-in EP | `deepseek-v2-lite-chat-builtin-ep` | 19.24 | 107.7 ms | 51.09 ms | 50.29 ms |
| Lite-EP | `deepseek-v2-lite-chat-ep` | 7.85 | 2023.3 ms | 97.35 ms | 95.83 ms |

## Important Negative Results

- vLLM 0.19.0 was not usable with the current torch runtime. Importing
  `vllm._C` failed with a C++ ABI mismatch, so the environment was moved
  to vLLM 0.22.0.
- Qwen1.5-MoE-A2.7B-Chat has 60 experts, which is not divisible by the 8
  EP ranks. DeepSeek-V2-Lite-Chat has 64 routed experts and is a better
  8-rank smoke model.
- A single-rank ad hoc Lite-EP runtime smoke can hit an NCCL Gin JIT path
  and fail with missing `ncclGin*` symbols. That path is not the validated
  vLLM UCCL proxy path; the decisive validation is the real 8-rank vLLM
  service.
- l41 should not run vLLM from the NFS Python environment for long jobs;
  it previously stalled in NFS/RPC waits. The working setup uses the SSD
  copy at `/ssd1/dsv2lite/envs/uccl312`.

## Review Checklist

- Confirm the vLLM shim only exposes high-throughput APIs and fails loudly
  for low-latency APIs.
- Confirm the shim resolves vLLM's CPU EP group to the device group before
  calling `ElasticBuffer` collectives.
- Confirm `USE_RECEIVER_BARRIER` is the default build mode and all verbs
  access flags agree with receiver-barrier semantics.
- Confirm every UCCL proxy thread has a valid shared atomic pointer and MR.
- Confirm the shared atomic per-rank stride remains page-aligned if
  `kAtomicBufferSize` changes.
- Rerun the unit test and the 8-GPU Lite-EP vLLM smoke after any changes
  in the linked files above.
