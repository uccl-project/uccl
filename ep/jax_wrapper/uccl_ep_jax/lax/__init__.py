"""Public jit-friendly MoE dispatch / combine API.

These functions are built on top of :mod:`uccl_ep_jax.primitive` and
are the recommended entry points for JAX users. They:

* work inside ``jax.jit`` (lowered to an XLA ``custom_call``),
* participate in autodiff (``jax.vjp``, ``jax.grad``) via
  :func:`jax.custom_vjp` -- matching the Primus-Turbo pattern where the
  backward pass of dispatch is combine and vice-versa,
* support the high-throughput path (``moe_dispatch`` /
  ``moe_combine``, which auto-selects intranode and internode based
  on ``num_rdma_ranks``) and the low-latency RDMA/IBGDA path
  (``moe_low_latency_dispatch`` / ``moe_low_latency_combine``).
"""

from .moe_low_latency import moe_low_latency_dispatch, moe_low_latency_combine
from .moe import moe_dispatch, moe_combine

__all__ = [
    "moe_low_latency_dispatch",
    "moe_low_latency_combine",
    "moe_dispatch",
    "moe_combine",
]
