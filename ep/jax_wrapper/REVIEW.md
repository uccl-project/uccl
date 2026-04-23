# Code review guide -- `uccl_ep_jax` (JAX bindings for UCCL-EP)

This document is a review aid for the PR adding JAX support on top of
`uccl.ep`. It summarizes what was added, why, and where to look. All
file paths are relative to the repository root.

## TL;DR

| Commit | Purpose | Lines |
| --- | --- | --- |
| `18d2362b` | Initial eager API + bootstrap + mode detection | ~1.6k |
| `41c9d5c4` | Eager `moe_dispatch` / `moe_combine` internode path | ~400 |
| `6eb63dce` | C++ FFI bridge + primitive + `custom_vjp` (intranode) | ~700 |
| `f770ad7d` | Primitive internode path (`num_rdma_ranks > 1`) | ~690 |
| `18f3713a` | Cached-mode dispatch -- real `moe_combine` backward | ~470 |
| `bb1769da` | Primitive safety for single-process multi-thread + multi-process | ~280 |
| `4e4227c9` | Primitive also supports single-process single-GPU | ~140 |

Total: 23 files changed, +5188 lines. All 17 pure-Python tests pass
on a CPU-only box with a stubbed `uccl.ep`.

## 1. Scope and execution modes

The new package exposes MoE dispatch/combine APIs on top of the
existing UCCL-EP C++ runtime for JAX. Three JAX execution layouts are
supported, all driven by the same `uccl_ep_jax.initialize(...)`:

| Layout | `jax.process_count()` | `jax.local_device_count()` | Bootstrap |
| --- | --- | --- | --- |
| Single-process, single-GPU | 1 | 1 | one `initialize()` call, `local_rank` defaults to 0 |
| Single-process, multi-thread multi-GPU | 1 | >1 | one `initialize(local_rank=i, ...)` per worker thread |
| Multi-process (one GPU per process) | >1 | >=1 | one `initialize()` per process, uses `jax.distributed` |

The primitive FFI handler is the same code path in all three modes;
only the Python bootstrap differs. See
`ep/jax_wrapper/README.md` -> "Execution modes (primitive + FFI)".

## 2. File map

```
ep/
├── README.md                          ## new section pointing to the package
├── src/uccl_ep.cc                     ## + ~730 lines of FFI bridge
└── jax_wrapper/
    ├── README.md                      ## user-facing docs
    ├── REVIEW.md                      ## this file
    ├── setup.py                       ## pip install -e .
    ├── examples/
    │   ├── test_low_latency_multithread.py
    │   └── test_low_latency_multiprocess.py
    ├── tests/                         ## pure Python, CPU-only, stubs uccl.ep
    │   ├── test_mode_detection.py
    │   ├── test_primitive_tracing.py
    │   └── test_threaded_primitive.py
    └── uccl_ep_jax/
        ├── __init__.py                ## re-exports
        ├── mode.py                    ## execution-mode detection + per-thread rank
        ├── kv_store.py                ## in-process KV store for single-proc rendezvous
        ├── bootstrap.py               ## initialize() / shutdown() / Buffer
        ├── config.py                  ## Config + recommended tuning tables
        ├── ops.py                     ## eager API (*_eager), kept for debugging
        ├── primitive/
        │   ├── __init__.py            ## re-exports
        │   ├── registry.py            ## register_ffi_targets()
        │   ├── _common.py             ## pack_ints32, dtype codes, ...
        │   └── _calls.py              ## 8 jax.ffi.ffi_call wrappers
        └── lax/
            ├── __init__.py
            ├── low_latency.py         ## jit-friendly low_latency_dispatch/combine
            └── moe.py                 ## jit-friendly moe_dispatch/combine + custom_vjp
```

## 3. Public API

### Primitive API (recommended, jit-friendly)

```python
import uccl_ep_jax as ux

# Bootstrap once per thread/process.
buf = ux.initialize(num_rdma_bytes=..., low_latency_mode=True,
                    num_experts=..., local_rank=...)

# High-throughput intranode + internode, auto-selected.
recv_x, recv_topk_idx, recv_topk_weights, handle = ux.moe_dispatch(
    x, topk_idx, topk_weights,
    num_experts=..., num_ranks=..., config=ux.get_dispatch_config(num_ranks))
combined = ux.moe_combine(recv_x, handle, topk_weights=recv_topk_weights,
                          num_ranks=num_ranks)

# Low-latency RDMA/IBGDA path (works on 1 GPU too).
recv_x, recv_count, handle = ux.low_latency_dispatch(
    x, topk_idx, num_max_dispatch_tokens_per_rank=..., num_experts=...,
    num_ranks=..., use_fp8=False)
combined, _, _ = ux.low_latency_combine(recv_x, topk_idx, topk_weights, handle)

ux.shutdown(buf)
```

All of the above can appear inside `jax.jit`, `shard_map`, and
`jax.vjp` / `jax.grad`.

### Eager API (debug / bootstrap validation)

Same names with `_eager` suffix
(`moe_dispatch_eager`, `low_latency_dispatch_eager`, etc). Reads raw
device pointers via `unsafe_buffer_pointer`; does **not** work
inside `jax.jit`.

## 4. C++ FFI bridge (`ep/src/uccl_ep.cc`)

A new `uccl_jax_ffi::` namespace at the bottom of the file exposes
**eight** XLA legacy custom-call handlers (ABI:
`void(cudaStream_t, void**, const char* opaque, size_t opaque_len, XlaCustomCallStatus*)`):

| Target name | Purpose |
| --- | --- |
| `uccl_ll_dispatch` | low-latency dispatch |
| `uccl_ll_combine` | low-latency combine |
| `uccl_moe_dispatch` | intranode high-throughput dispatch |
| `uccl_moe_combine` | intranode high-throughput combine |
| `uccl_moe_internode_dispatch` | internode (`num_rdma_ranks > 1`) dispatch |
| `uccl_moe_internode_combine` | internode combine |
| `uccl_moe_cached_dispatch` | intranode cached-replay (combine backward) |
| `uccl_moe_internode_cached_dispatch` | internode cached-replay (combine backward) |

Each handler:

1. Casts the opaque byte blob to a per-op `struct` with int32 fields
   only (so Python can `struct.pack("i"*N, ...)` a byte-identical
   layout).
2. Resolves the active `Buffer*` via `cudaGetDevice()` and a
   process-global `unordered_map<int, Buffer*> g_jax_ffi_buffers` that
   is populated by the Python bootstrap.
3. Calls the existing `Buffer::intranode_*` / `Buffer::internode_*` /
   `Buffer::low_latency_*` methods with raw pointers taken straight
   from `buffers[]`.

New Python bindings:

- `ep.register_jax_ffi_buffer(device_index, buffer)`
- `ep.unregister_jax_ffi_buffer(device_index)`
- `ep.get_jax_ffi_targets() -> {name: PyCapsule}`

### One correctness fix to `Buffer::internode_prepare`

```diff
-    nb::gil_scoped_release release;
+    std::optional<nb::gil_scoped_release> release;
+    if (PyGILState_Check()) release.emplace();
```

Required because the FFI handler runs on an XLA worker thread that
does not hold the GIL. Without the guard, the internode primitive
would have aborted on the first call.

## 5. Python primitive layer (`uccl_ep_jax/primitive/`)

### `_calls.py` -- eight `jax.ffi.ffi_call` wrappers

One function per C++ target. Each:

- Declares output shapes (`jax.ShapeDtypeStruct`). For dispatch these
  are `num_worst_tokens = num_tokens * num_ranks` static upper bounds
  (same convention as Primus-Turbo).
- Packs static parameters into `legacy_backend_config` using
  `_common.pack_ints32`.
- Calls `jax.ffi.ffi_call(target_name, out, custom_call_api_version=1,
  legacy_backend_config=bytes, ...)`.

### `registry.py`

`register_ffi_targets()` -- idempotent, thread-safe. Resolves the
active JAX platform (`cuda` or `rocm`) from `jax.local_devices()[0]`
and registers each capsule returned by
`uccl.ep.get_jax_ffi_targets()` with `api_version=0`.

## 6. Python lax layer (`uccl_ep_jax/lax/`)

### `low_latency.py`

Thin wrappers around `low_latency_dispatch_call` /
`low_latency_combine_call`. Return `(recv_x, recv_count, handle)` /
`combined`. No `custom_vjp` here because the low-latency path is
usually used in inference / serving contexts.

### `moe.py` -- the most important file to review

Routing:

- `moe_dispatch` picks the primitive based on `num_rdma_ranks`
  (auto-detected from the bound `Buffer` or caller-provided).
- `moe_combine` picks intranode vs internode from `len(handle)`:
  6-tuple = intranode, 10-tuple = internode.
  The handle is a plain tuple of `jax.Array`s (no Python strings),
  which is required by `custom_vjp`.

`jax.custom_vjp` rules (mirror of Primus-Turbo):

- **`_moe_dispatch_bwd`**: call `moe_combine` on the upstream gradient.
  Uses the cached handle, so this is a single kernel call.
- **`_moe_combine_fwd`**: capture `(handle, num_recv_tokens, config)`
  in residuals. `num_recv_tokens = x.shape[0]` at the forward call.
- **`_moe_combine_bwd`**: call the cached-dispatch primitive
  (`moe_cached_dispatch_call` or `moe_internode_cached_dispatch_call`
  based on `len(handle)`). Returns a real `grad_x`, zero-filled
  `grad_handle`, `None` for `topk_weights` / `bias`.

End-to-end: a `jax.grad(fwd)` of `moe_dispatch` + `moe_combine` in
internode mode lowers to **four** XLA custom calls:

```
forward:  uccl_moe_internode_dispatch  +  uccl_moe_internode_combine
backward: uccl_moe_internode_combine   +  uccl_moe_internode_cached_dispatch
```

## 7. Bootstrap (`uccl_ep_jax/bootstrap.py`)

### `initialize(...)`

1. Resolve execution mode (`detect_execution_mode`).
2. Resolve `(global_rank, global_world_size, local_rank,
   local_world_size)`:
   - Multi-process: from `jax.process_index()` / `jax.process_count()`.
   - Single-process: `local_rank` caller-provided. For single-GPU
     (`jax.local_device_count() == 1`), `local_rank` defaults to 0.
   - Multi-thread multi-GPU requires `local_rank` from the caller.
3. `ep.set_device(device_index)` + `ep.get_device()` to capture the
   **actual CUDA ordinal** the runtime ended up on.
4. Allocate the RDMA scratch buffer (`ep.get_rdma_buffer`).
5. Spawn proxies (`ep.Proxy`).
6. OOB rendezvous (peer meta -> IPC handles -> `runtime.sync(...)`).
7. Register the runtime in the FFI map under the captured CUDA ordinal
   (`ep.register_jax_ffi_buffer(cuda_device_index, runtime)`).

### `Buffer` (Python class)

Wraps the C++ `uccl.ep.Buffer` object. Adds:

- `num_rdma_ranks`, `num_max_nvl_peers`, `source_meta_bytes` --
  auto-detection sources for the primitive layer.
- `cuda_device_index` -- persisted so `destroy()` unregisters under
  the same key regardless of which thread tears it down.

### Per-thread registry

`_thread_buffer_tls` (a `threading.local`) + `_buffers_by_global_rank`
(keyed by `global_rank`, protected by a lock). `get_buffer()` returns
the `Buffer` for the calling thread. The primitive layer uses this
for auto-detecting `num_rdma_ranks` etc at trace time.

## 8. Rendezvous (`uccl_ep_jax/kv_store.py`)

Two backing stores unified by `KVClient`:

- **`InProcessKeyValueStore`**: a `{key: bytes}` map behind a
  `threading.Condition`. Used in single-process mode.
- **JAX `DistributedRuntimeClient`**: obtained via
  `jax._src.distributed.global_state.client`. Used in multi-process
  mode.

Both expose the same methods (`key_value_set_bytes`,
`blocking_key_value_get_bytes`), so the bootstrap logic is identical
in the two modes.

`KVClient` adds `all_gather(namespace, rank, world_size, value)` and
`barrier(namespace, rank, world_size)` helpers.

## 9. Tests (`ep/jax_wrapper/tests/`)

All tests are pure Python + `jax` + `pytest`; no GPU, no
`uccl.ep` build required (module is stubbed).

### `test_mode_detection.py` (4)
- In-process KV round trip.
- Thread-safe all-gather.
- Execution-mode detection returns a valid enum.
- Config tables populated for common rank counts.

### `test_primitive_tracing.py` (8)
- Low-latency dispatch / combine lower to the expected `custom_call`
  target.
- Intranode `moe_dispatch` lowers to `uccl_moe_dispatch` only.
- Internode `moe_dispatch` lowers to `uccl_moe_internode_dispatch`.
- Internode `moe_combine` lowers to `uccl_moe_internode_combine`.
- Upper-bound output shape check
  (`eval_shape.shape == (num_tokens * num_ranks, hidden)`).
- Intranode `moe_combine` backward emits `uccl_moe_cached_dispatch`.
- Internode `moe_combine` backward emits
  `uccl_moe_internode_cached_dispatch`.

### `test_threaded_primitive.py` (5)
- Two worker threads with different fake `Buffer`s trace in parallel
  and each lowering only references its own topology's custom call.
- Primitive topology override works without a bound `Buffer` (CI
  friendliness).
- Single-process single-GPU lowering works without explicit
  `local_rank`.
- Single-process multi-thread end-to-end tracing (intranode thread
  never sees the internode target and vice versa).
- `register_ffi_targets()` is concurrency-safe.

Current state: **17 passed**.

## 10. Known limitations

- **`moe_dispatch` / `moe_combine` need `num_ranks >= 2`**. This is a
  constraint of the underlying UCCL-EP C++ layer, not JAX. Single-GPU
  callers should use `low_latency_dispatch` / `low_latency_combine`.
- **GPU integration tests** live in `ep/jax_wrapper/examples/` and
  require a CUDA/ROCm `uccl.ep` build. They have not been exercised
  in this branch.
- **The eager API is kept for debugging** (`moe_dispatch_eager`,
  etc). It is *not* jit-friendly (uses `unsafe_buffer_pointer()`); in
  production you want the primitive API.

## 11. Review checklist

Focus areas, in rough priority order:

1. `ep/src/uccl_ep.cc` -- the eight FFI handlers. In particular:
   - Opaque struct layouts line up with the Python `pack_ints32`
     calls? (search `pack_ints32(` in
     `ep/jax_wrapper/uccl_ep_jax/primitive/_calls.py` and diff field
     order against the `MoE*Opaque` structs).
   - Buffer-index order in each handler (`buffers[0]`, `buffers[1]`,
     ...) matches the order of operands + outputs declared in
     `_calls.py`.
   - `PyGILState_Check()` guard in `internode_prepare` is correct.
2. `ep/jax_wrapper/uccl_ep_jax/lax/moe.py` -- the `custom_vjp` logic
   (handle arity, cached-replay in `_moe_combine_bwd`, residuals).
3. `ep/jax_wrapper/uccl_ep_jax/bootstrap.py` -- CUDA-ordinal capture
   and registration / deregistration paths.
4. `ep/jax_wrapper/uccl_ep_jax/primitive/_calls.py` -- output shape
   declarations (especially the internode dispatch's 17 outputs and
   their shape expressions).
5. Everything else (mode detection, KV store, eager wrappers) is
   straightforward plumbing.
