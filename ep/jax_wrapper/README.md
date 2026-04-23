# uccl_ep_jax -- JAX bindings for UCCL-EP

This package exposes the [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
and [mori](https://github.com/ROCm/mori)-style MoE dispatch / combine
APIs on top of the UCCL-EP communication runtime (`uccl.ep`).

It supports the two JAX execution models:

1. **Single-process, multi-thread** -- one Python process owns every
   local GPU and each thread pins itself to one device. The rendezvous
   uses an in-process KV store. This is the regime covered by
   `jax.local_device_count() > 1` and `jax.process_count() == 1`.
2. **Multi-process, single-GPU-per-process** (`jax.process_count() > 1`) --
   the classic `jax.distributed.initialize` setup, identical to the
   model used by `transformer_engine/jax` and `mori.jax`.

The active mode is auto-detected via `jax.process_count()` /
`jax.local_device_count()`.

## Install

```bash
# Prerequisite: build + install uccl.ep first (see ../README.md).
pip install -e .
```

## Quickstart

### Multi-process (e.g., launched with `mpirun` or `torchrun` wrapping)

```python
import jax
import uccl_ep_jax as ucx

jax.distributed.initialize()  # standard JAX multi-controller bootstrap

num_experts = 32
num_tokens = 128
hidden = 4096
num_topk = 8

num_rdma_bytes = ucx.get_low_latency_rdma_size_hint(
    num_tokens, hidden, jax.device_count(), num_experts
)
buffer = ucx.initialize(
    num_rdma_bytes=num_rdma_bytes,
    low_latency_mode=True,
    num_experts=num_experts,
)

@jax.jit
def step(x, topk_idx, topk_weights):
    recv_x, recv_count, handle = ucx.low_latency_dispatch(
        x, topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        num_ranks=jax.device_count(),
        use_fp8=True,
    )
    # When ``use_fp8=True`` the receive side returns a (values, scales)
    # tuple; pick the values for the combine:
    recv_values = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    return ucx.low_latency_combine(recv_values, topk_idx, topk_weights, handle)

combined_x = step(x, topk_idx, topk_weights)

ucx.shutdown()
```

### Single-process, multi-thread

```python
import threading
import jax
import uccl_ep_jax as ucx

def worker(local_rank: int, world_size: int):
    num_rdma_bytes = ucx.get_low_latency_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )
    buf = ucx.initialize(
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_experts=num_experts,
        local_rank=local_rank,
        global_rank=local_rank,
        global_world_size=world_size,
        local_world_size=world_size,
    )
    # ...run dispatch/combine on this thread...
    ucx.shutdown(buf)

world_size = jax.local_device_count()
threads = [
    threading.Thread(target=worker, args=(i, world_size)) for i in range(world_size)
]
for t in threads: t.start()
for t in threads: t.join()
```

## API

All public dispatch/combine entry points are real XLA custom-call
primitives. They are lowered to ``stablehlo.custom_call`` and can be
used inside ``jax.jit``, ``shard_map``, and participate in
``jax.vjp`` / ``jax.grad`` via ``jax.custom_vjp`` — matching the
Primus-Turbo pattern.

| Function | Notes |
| --- | --- |
| `moe_dispatch(...)` | High-throughput MoE dispatch; intranode and internode paths selected automatically from `num_rdma_ranks`; `custom_vjp`. |
| `moe_combine(...)` | High-throughput MoE combine; path selected from the handle arity (6-tuple = intranode, 10-tuple = internode); `custom_vjp`. |
| `low_latency_dispatch(...)` | Low-latency RDMA/IBGDA dispatch as XLA custom call. |
| `low_latency_combine(...)` | Low-latency RDMA/IBGDA combine as XLA custom call. |
| `register_ffi_targets()` | Idempotent; called automatically on first use. |

### Shared infrastructure

| Function | Notes |
| --- | --- |
| `initialize(...)` | Boot the C++ runtime, proxies, and IPC exchange for the calling thread/process. |
| `shutdown(buf=None)` | Destroy the runtime bound to the current thread (or the one passed in). |
| `get_buffer()` | Return the `Buffer` registered on the current thread. |
| `get_dispatch_config(num_ranks)` / `get_combine_config(num_ranks)` | Recommended tuning configs. |

The execution-mode helpers (`detect_execution_mode`,
`is_single_process_mode`, `is_multi_process_mode`) mirror the pattern
used by TransformerEngine JAX.

## Execution modes (primitive + FFI)

The FFI primitive layer supports all three JAX execution layouts
transparently — the same code paths are exercised, only the
bootstrap / ``initialize(...)`` call shape differs:

* **Single-process, single-GPU** (`jax.process_count() == 1`,
  `jax.local_device_count() == 1`). One Python thread owns the only
  local GPU; `uccl_ep_jax.initialize(...)` is called once on the main
  thread and `local_rank` defaults to `0`. In practice only
  `low_latency_dispatch` / `low_latency_combine` are meaningful here
  (high-throughput `moe_dispatch` requires `num_ranks >= 2` at the
  UCCL-EP C++ layer).
* **Single-process, multi-thread multi-GPU** (`jax.process_count() == 1`,
  `jax.local_device_count() > 1`). Each worker thread owns one GPU
  and calls `uccl_ep_jax.initialize(local_rank=i, ...)` from inside
  its thread. The C++ runtime for that thread is registered in the
  process-wide FFI map under the **CUDA ordinal** the thread pinned
  to. At execute time XLA calls the handler with the CUDA stream for
  that device, the handler looks up the correct `Buffer` via
  `cudaGetDevice()`, and because `Buffer::comm_stream` was created on
  that device everything just works. The per-thread topology
  (`num_rdma_ranks`, `num_max_nvl_peers`, `source_meta_bytes`) is
  auto-detected from a thread-local registry.
* **Multi-process** (`jax.process_count() > 1`, typically one Python
  process per GPU). `uccl_ep_jax.initialize(...)` is called once per
  process; there is a single registered `Buffer` in the FFI map.

A single subtle C++ fix is relevant here: `Buffer::internode_prepare`
used to unconditionally release the GIL, which is correct when called
from a Python thread but crashes when called from an XLA worker thread
(which doesn't hold the GIL). The current C++ code checks
`PyGILState_Check()` at runtime and only releases when the GIL is
actually held — so both the eager (Python-thread) and primitive
(XLA-thread) call paths are safe.

## Autodiff

The primitive ``moe_dispatch`` and ``moe_combine`` are decorated with
``jax.custom_vjp`` and mirror Primus-Turbo's forward/backward structure:

* **Backward of ``moe_dispatch``** replays ``moe_combine`` on the
  incoming gradient, re-using the cached layout handle.
* **Backward of ``moe_combine``** replays a **cached-mode** dispatch
  (``moe_cached_dispatch`` / ``moe_internode_cached_dispatch``) on the
  incoming gradient — the same path the PyTorch wrapper takes when a
  ``handle`` is reused.

Both paths are available in intranode and internode configurations and
are selected automatically based on the handle arity
(6-tuple = intranode, 10-tuple = internode).

## Limitations / follow-ups

* ``moe_dispatch`` / ``moe_combine`` use the static-upper-bound path
  on every forward call
  (``num_worst_tokens = num_tokens * num_ranks``). Exposing a separate
  entry point that reuses an existing handle on the forward pass (so
  subsequent iterations skip layout computation) is a nice optimization
  for training loops that pin the token count; today it is already
  exposed implicitly through the ``custom_vjp`` backward pass.
* The internode throughput path assumes ``num_rdma_ranks > 1`` means
  "proper multi-node"; for the edge case of a single RDMA rank the
  intranode path is picked automatically.
