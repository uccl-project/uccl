"""JAX bindings for UCCL-EP.

This package mirrors the Primus-Turbo ``primus_turbo.jax.lax.moe`` API
(``moe_dispatch`` / ``moe_combine``) on top of the UCCL-EP C++ runtime
(``uccl.ep``) and adds low-latency RDMA/IBGDA dispatch and combine
entry points.

All public ops are real XLA custom-call primitives registered through
``jax.ffi.register_ffi_target``. They:

* lower to ``stablehlo.custom_call``, so they work inside ``jax.jit``,
  ``shard_map``, etc.;
* participate in autodiff via ``jax.custom_vjp`` (dispatch's backward
  is combine, combine's backward is a cached-mode dispatch replay --
  matching the Primus-Turbo pattern);
* auto-select the intranode or internode kernel based on
  ``num_rdma_ranks``.

Three JAX execution layouts are supported, all driven by a single
:func:`initialize` entry point:

1. **Single-process, single-GPU** (``jax.process_count() == 1``,
   ``jax.local_device_count() == 1``) -- ``local_rank`` defaults to 0.
   In practice only :func:`low_latency_dispatch` /
   :func:`low_latency_combine` are meaningful on a single rank.
2. **Single-process, multi-thread multi-GPU**
   (``jax.process_count() == 1``, ``jax.local_device_count() > 1``) --
   one worker thread per GPU. Each thread calls
   ``initialize(local_rank=i, ...)``.
3. **Multi-process** (``jax.process_count() > 1``) -- the classic JAX
   distributed setup, same pattern as NVIDIA/TransformerEngine and
   ROCm/mori.

The active layout is auto-detected; see :mod:`uccl_ep_jax.mode`.
"""

from .config import Config, set_default_num_sms, get_dispatch_config, get_combine_config
from .mode import (
    JaxExecutionMode,
    detect_execution_mode,
    is_multi_process_mode,
    is_single_process_mode,
    get_global_rank,
    get_global_world_size,
    get_local_rank,
    get_local_world_size,
)
from .bootstrap import (
    initialize,
    shutdown,
    get_buffer,
    Buffer,
    get_low_latency_rdma_size_hint,
    get_nvl_buffer_size_hint,
    get_rdma_buffer_size_hint,
)

# Public XLA custom-call primitives. These are the only user-facing
# dispatch/combine entry points; they mirror the Primus-Turbo API and
# are backed by handlers in ``ep/src/uccl_ep.cc`` (``uccl_jax_ffi::``
# namespace).
from .lax import (
    moe_dispatch,
    moe_combine,
    low_latency_dispatch,
    low_latency_combine,
)
from .primitive import register_ffi_targets

__all__ = [
    "Config",
    "set_default_num_sms",
    "get_dispatch_config",
    "get_combine_config",
    "JaxExecutionMode",
    "detect_execution_mode",
    "is_multi_process_mode",
    "is_single_process_mode",
    "get_global_rank",
    "get_global_world_size",
    "get_local_rank",
    "get_local_world_size",
    "initialize",
    "shutdown",
    "get_buffer",
    "Buffer",
    "moe_dispatch",
    "moe_combine",
    "low_latency_dispatch",
    "low_latency_combine",
    "get_low_latency_rdma_size_hint",
    "get_nvl_buffer_size_hint",
    "get_rdma_buffer_size_hint",
    "register_ffi_targets",
]
