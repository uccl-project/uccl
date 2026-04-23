"""Low-level XLA custom-call wrappers backing :mod:`uccl_ep_jax.lax`.

These helpers build on :func:`jax.ffi.ffi_call` (legacy ABI
``custom_call_api_version=1``) to turn the UCCL-EP dispatch/combine
kernels into proper XLA ops that can be used inside ``jax.jit`` /
``shard_map`` / ``jax.vjp``.

All attributes (shapes, config, bool flags) are packed into a single
``legacy_backend_config`` byte blob whose binary layout matches the
``struct``s declared in ``ep/src/uccl_ep.cc``'s ``uccl_jax_ffi``
namespace.
"""

from .registry import register_ffi_targets, TARGET_NAMES
from ._calls import (
    low_latency_dispatch_call,
    low_latency_combine_call,
    moe_dispatch_call,
    moe_combine_call,
    moe_internode_dispatch_call,
    moe_internode_combine_call,
    moe_cached_dispatch_call,
    moe_internode_cached_dispatch_call,
)

__all__ = [
    "register_ffi_targets",
    "TARGET_NAMES",
    "low_latency_dispatch_call",
    "low_latency_combine_call",
    "moe_dispatch_call",
    "moe_combine_call",
    "moe_internode_dispatch_call",
    "moe_internode_combine_call",
    "moe_cached_dispatch_call",
    "moe_internode_cached_dispatch_call",
]
