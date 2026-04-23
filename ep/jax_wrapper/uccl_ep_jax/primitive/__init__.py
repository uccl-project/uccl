"""Low-level XLA custom-call wrappers backing :mod:`uccl_ep_jax.lax`.

These helpers build on :func:`jax.ffi.ffi_call` with the XLA Typed FFI
(``custom_call_api_version=4`` at the MLIR level, which is the default
for ``ffi_call``, and ``api_version=1`` at the
:func:`jax.ffi.register_ffi_target` level) to turn the UCCL-EP
dispatch/combine kernels into proper XLA ops that can be used inside
``jax.jit`` / ``shard_map`` / ``jax.vjp``.

All scalar attributes (shapes, config, bool flags) are passed as
strongly-typed ``int32`` keyword attributes to ``ffi_call``; the C++
side (see ``ep/src/uccl_ep_jax.cc``) decodes them one-by-one through
``Attr<int32_t>("name")`` on a ``ffi::Ffi::Bind()`` chain.
"""

from .registry import register_ffi_targets, TARGET_NAMES
from ._calls import (
    moe_low_latency_dispatch_call,
    moe_low_latency_combine_call,
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
    "moe_low_latency_dispatch_call",
    "moe_low_latency_combine_call",
    "moe_dispatch_call",
    "moe_combine_call",
    "moe_internode_dispatch_call",
    "moe_internode_combine_call",
    "moe_cached_dispatch_call",
    "moe_internode_cached_dispatch_call",
]
