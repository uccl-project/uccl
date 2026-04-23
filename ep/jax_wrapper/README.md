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

recv_x, recv_count, handle, event, hook = ucx.low_latency_dispatch(
    x, topk_idx, num_tokens, num_experts, use_fp8=True,
)
combined_x, event, hook = ucx.low_latency_combine(
    recv_x[0], topk_idx, topk_weights, handle,
)

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

Two API flavours are shipped. The primitive API is the recommended one:

### Primitive / XLA-custom-call API (jit-friendly)

These are exported from the top-level module and are the default names
(``uccl_ep_jax.moe_dispatch`` etc). They are lowered to
``stablehlo.custom_call`` and therefore can be used inside ``jax.jit``,
``shard_map``, and participate in ``jax.vjp`` / ``jax.grad`` via
``jax.custom_vjp`` — matching the Primus-Turbo pattern.

| Function | Notes |
| --- | --- |
| `moe_dispatch(...)` | High-throughput intranode MoE dispatch; `custom_vjp`. |
| `moe_combine(...)` | High-throughput intranode MoE combine; `custom_vjp`. |
| `low_latency_dispatch(...)` | Low-latency RDMA/IBGDA dispatch as XLA custom call. |
| `low_latency_combine(...)` | Low-latency RDMA/IBGDA combine as XLA custom call. |
| `register_ffi_targets()` | Idempotent; called automatically on first use. |

### Eager API (for bootstrapping / debugging)

Same entry points with the ``_eager`` suffix
(``uccl_ep_jax.moe_dispatch_eager`` etc). They skip ``jit`` and read raw
device pointers via ``unsafe_buffer_pointer``. Useful while wiring up
bootstrap, or when you need the cached-handle / internode throughput
paths that the primitive layer does not yet cover.

| Function | Notes |
| --- | --- |
| `moe_dispatch_eager(...)` | Intranode + internode; selected by `num_rdma_ranks`. |
| `moe_combine_eager(...)` | Intranode + internode. |
| `low_latency_dispatch_eager(...)` / `low_latency_combine_eager(...)` | Return `(event, hook)` for fine-grained overlap. |

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

## Limitations / follow-ups

* The primitive `moe_dispatch` / `moe_combine` currently cover the
  **intranode** path (``num_rdma_ranks == 1``). For internode
  high-throughput today you can use the ``_eager`` variants. Wiring the
  internode path through the primitive layer requires surfacing
  ``num_recv_tokens`` as a runtime dynamic shape; that is a planned
  follow-up.
* ``moe_dispatch`` / ``moe_combine`` currently use the non-cached path
  on every call. Caching the layout handle across iterations (as the
  PyTorch wrapper does when ``handle`` is reused) will be added next.
* The backward rule of the primitive ``moe_combine`` is currently a
  zero-gradient placeholder pending the cached-dispatch primitive; the
  backward rule of the primitive ``moe_dispatch`` already does the
  right thing (calls the primitive ``moe_combine`` on the upstream
  gradient).
