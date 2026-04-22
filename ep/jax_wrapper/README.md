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

| Function | Notes |
| --- | --- |
| `initialize(...)` | Boot the C++ runtime, proxies, and IPC exchange for the calling thread/process. |
| `shutdown(buf=None)` | Destroy the runtime bound to the current thread (or the one passed in). |
| `get_buffer()` | Return the `Buffer` registered on the current thread. |
| `low_latency_dispatch(...)` | Port of `Buffer.low_latency_dispatch` (RDMA/IBGDA path). |
| `low_latency_combine(...)` | Port of `Buffer.low_latency_combine`. |
| `moe_dispatch(...)` | Primus-Turbo-compatible intranode high-throughput dispatch. |
| `moe_combine(...)` | Primus-Turbo-compatible intranode high-throughput combine. |
| `get_dispatch_config(num_ranks)` | Recommended tuning config for dispatch. |
| `get_combine_config(num_ranks)` | Recommended tuning config for combine. |

The execution-mode helpers (`detect_execution_mode`,
`is_single_process_mode`, `is_multi_process_mode`) mirror the pattern
used by TransformerEngine JAX.

## Limitations / follow-ups

* `moe_dispatch` / `moe_combine` currently target the intranode path
  only; the internode (RDMA) path is still accessed through
  `low_latency_dispatch` / `low_latency_combine`.
* The ops execute eagerly (they block until inputs are ready) similar
  to `mori.jax.ops`. Exposing them as XLA custom-call primitives with
  proper batching / VJP rules (as Primus-Turbo does) is a planned
  follow-up.
