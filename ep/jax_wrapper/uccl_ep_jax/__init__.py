"""JAX bindings for UCCL-EP.

This package mirrors the Primus-Turbo `primus_turbo.jax.lax.moe` API
(`moe_dispatch` / `moe_combine`) on top of the UCCL-EP C++ runtime
(`uccl.ep`).

Two JAX execution modes are supported:

1. **Single-controller / single-process mode** -- one Python process owns
   multiple GPUs (`jax.process_count() == 1`,
   `jax.local_device_count() > 1`). User code typically spawns one Python
   thread per GPU for true parallelism. Each thread must create its own
   `Buffer` bound to its local device. An in-process rendezvous handles
   the out-of-band exchange.

2. **Multi-controller / multi-process mode** -- every GPU is owned by a
   dedicated Python process (`jax.process_count() > 1`). The JAX
   distributed client (`jax.distributed.initialize`) is used for the
   out-of-band key-value exchange, matching the pattern used by
   NVIDIA/TransformerEngine and ROCm/mori.

The high-level APIs exposed here (`moe_dispatch`, `moe_combine`,
`low_latency_dispatch`, `low_latency_combine`) operate eagerly (outside
of `jax.jit`), similar to how ROCm/mori's JAX ops are invoked today. This
keeps the integration simple while we do not yet own a dedicated XLA
custom call / FFI primitive for EP communication.
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
)
from .ops import (
    moe_dispatch as moe_dispatch_eager,
    moe_combine as moe_combine_eager,
    low_latency_dispatch as low_latency_dispatch_eager,
    low_latency_combine as low_latency_combine_eager,
    get_low_latency_rdma_size_hint,
)

# Primitive + FFI (jit-friendly) public API. These mirror the Primus-Turbo
# ``primus_turbo.jax.lax.moe`` entry points, including ``custom_vjp``
# support, and are the recommended path for real JAX training loops.
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
    "moe_dispatch_eager",
    "moe_combine_eager",
    "low_latency_dispatch_eager",
    "low_latency_combine_eager",
    "get_low_latency_rdma_size_hint",
    "register_ffi_targets",
]
