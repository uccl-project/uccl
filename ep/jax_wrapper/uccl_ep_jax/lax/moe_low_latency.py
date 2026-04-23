"""Low-latency MoE dispatch / combine exposed as jit-friendly XLA ops.

These are the low-latency (RDMA / IBGDA) counterparts of
:func:`uccl_ep_jax.moe_dispatch` / :func:`uccl_ep_jax.moe_combine`.
They target DeepEP's low-latency kernels which are typically used for
MoE inference / serving.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ..primitive import (
    moe_low_latency_combine_call,
    moe_low_latency_dispatch_call,
    register_ffi_targets,
)


def _ensure_registered():
    register_ffi_targets()


# ---------------------------------------------------------------------------
# moe_low_latency_dispatch
# ---------------------------------------------------------------------------


def moe_low_latency_dispatch(
    x: jax.Array,
    topk_idx: jax.Array,
    *,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    num_ranks: int,
    use_fp8: bool = False,
    round_scale: bool = False,
    use_ue8m0: bool = False,
):
    """Jit-friendly low-latency MoE dispatch.

    Returns ``(recv_x, recv_count, handle)`` where ``handle`` is the
    tuple consumed by :func:`moe_low_latency_combine`.

    The underlying custom call runs synchronously on the compute
    stream XLA provides, so there is no ``event`` / ``hook`` to manage.

    Note: users must have called ``uccl_ep_jax.initialize(...)`` on
    the calling thread/process; this function just drives the XLA op.
    """
    _ensure_registered()
    outputs = _dispatch_impl(
        x, topk_idx,
        num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts=num_experts, num_ranks=num_ranks,
        use_fp8=use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
    )
    (recv_x, recv_count, recv_src_info, recv_layout_range, recv_x_scales) = outputs
    # Normalize the "no fp8 scales" case for the user: return None
    # rather than a zero-size placeholder.
    if not use_fp8:
        recv_x_scales = None

    handle = (
        recv_src_info, recv_layout_range,
        int(num_max_dispatch_tokens_per_rank), int(x.shape[1]), int(num_experts),
    )
    recv = (recv_x, recv_x_scales) if use_fp8 else recv_x
    return recv, recv_count, handle


def _dispatch_impl(x, topk_idx, **kw):
    return moe_low_latency_dispatch_call(x, topk_idx, **kw)


# ---------------------------------------------------------------------------
# moe_low_latency_combine
# ---------------------------------------------------------------------------


def moe_low_latency_combine(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    handle: Tuple,
    *,
    use_logfmt: bool = False,
    zero_copy: bool = False,
):
    """Jit-friendly low-latency MoE combine."""
    _ensure_registered()
    (src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts) = handle
    return _combine_impl(
        x, topk_idx, topk_weights, src_info, layout_range,
        num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts=num_experts,
        use_logfmt=use_logfmt, zero_copy=zero_copy,
    )


def _combine_impl(x, topk_idx, topk_weights, src_info, layout_range, **kw):
    return moe_low_latency_combine_call(
        x, topk_idx, topk_weights, src_info, layout_range, **kw
    )
