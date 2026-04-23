"""Public jit-friendly MoE dispatch / combine API.

These functions are built on top of :mod:`uccl_ep_jax.primitive` and
are the recommended entry points for JAX users. They:

* work inside ``jax.jit`` (lowered to an XLA ``custom_call``),
* participate in autodiff (``jax.vjp``, ``jax.grad``) via
  :func:`jax.custom_vjp` — matching the Primus-Turbo pattern where the
  backward pass of dispatch is combine and vice-versa,
* support the high-throughput intranode path (``moe_dispatch`` /
  ``moe_combine``) and the low-latency internode path
  (``low_latency_dispatch`` / ``low_latency_combine``).

Internode high-throughput (``num_rdma_ranks > 1``) is still served by
the eager wrappers in :mod:`uccl_ep_jax.ops` today; moving it to a
primitive requires propagating a runtime-only ``num_recv_tokens``, which
is out of scope for the initial primitive landing.
"""

from .low_latency import low_latency_dispatch, low_latency_combine
from .moe import moe_dispatch, moe_combine

__all__ = [
    "low_latency_dispatch",
    "low_latency_combine",
    "moe_dispatch",
    "moe_combine",
]
