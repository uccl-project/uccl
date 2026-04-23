"""High-throughput MoE dispatch / combine with autodiff.

The structure mirrors Primus-Turbo's ``primus_turbo.jax.lax.moe``:

* :func:`moe_dispatch` is a ``custom_vjp`` whose backward rule invokes
  ``moe_combine`` on the received gradients.
* :func:`moe_combine` is a ``custom_vjp`` whose backward rule invokes
  ``moe_dispatch`` (cached-mode replay, which is a planned follow-up;
  today it emits a zero-gradient placeholder).

Both intranode (``num_rdma_ranks == 1``) and internode
(``num_rdma_ranks > 1``) paths are implemented as XLA ``custom_call``s
via :mod:`uccl_ep_jax.primitive`.  The user can either:

* pass ``num_rdma_ranks=...`` explicitly (useful for offline tracing
  without a bound ``Buffer``), or
* omit it, in which case the primitive auto-detects by looking up the
  per-thread ``Buffer`` installed by :func:`uccl_ep_jax.initialize`.

The number of received tokens is upper-bounded statically by
``num_tokens * num_ranks`` (Primus-Turbo's ``num_worst_tokens``
convention), keeping all output shapes trace-time constants.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..config import Config, get_combine_config, get_dispatch_config
from ..primitive import (
    moe_combine_call,
    moe_dispatch_call,
    moe_internode_combine_call,
    moe_internode_dispatch_call,
    register_ffi_targets,
)


def _ensure_registered():
    register_ffi_targets()


def _resolve_topology(
    num_rdma_ranks: Optional[int],
    num_max_nvl_peers: Optional[int],
    source_meta_bytes: Optional[int],
) -> Tuple[int, int, int]:
    """Resolve the topology parameters needed by the primitives.

    If the caller did not pass them explicitly we look up the per-thread
    ``Buffer`` installed by :func:`uccl_ep_jax.initialize`.
    """
    if (
        num_rdma_ranks is not None
        and num_max_nvl_peers is not None
        and source_meta_bytes is not None
    ):
        return int(num_rdma_ranks), int(num_max_nvl_peers), int(source_meta_bytes)
    from ..bootstrap import get_buffer

    buf = get_buffer()
    if num_rdma_ranks is None:
        num_rdma_ranks = buf.num_rdma_ranks
    if num_max_nvl_peers is None:
        num_max_nvl_peers = buf.num_max_nvl_peers
    if source_meta_bytes is None:
        source_meta_bytes = buf.source_meta_bytes
    return int(num_rdma_ranks), int(num_max_nvl_peers), int(source_meta_bytes)


# ---------------------------------------------------------------------------
# Handle layout
# ---------------------------------------------------------------------------
#
# Instead of stuffing a string tag into the handle tuple (not allowed
# inside ``custom_vjp``), we use two different fixed-arity layouts:
#
#   intranode  (6 arrays):
#       (rank_prefix_matrix, channel_prefix_matrix,
#        recv_channel_prefix_matrix, recv_src_idx,
#        is_token_in_rank, send_head)
#
#   internode (10 arrays):
#       (is_token_in_rank,
#        sender_rdma_channel_prefix_matrix,   # unused by combine
#        sender_gbl_channel_prefix_matrix,    # unused by combine
#        recv_rdma_channel_prefix_matrix,
#        recv_rdma_rank_prefix_sum,
#        recv_gbl_channel_prefix_matrix,
#        recv_gbl_rank_prefix_sum,            # unused by combine
#        recv_src_meta, send_rdma_head, send_nvl_head)
#
# ``len(handle)`` disambiguates without resorting to Python strings.


# ---------------------------------------------------------------------------
# moe_dispatch
# ---------------------------------------------------------------------------


def moe_dispatch(
    x: Union[jax.Array, Tuple[jax.Array, jax.Array]],
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_experts: int,
    num_ranks: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
    *,
    num_rdma_ranks: Optional[int] = None,
    num_max_nvl_peers: Optional[int] = None,
    source_meta_bytes: Optional[int] = None,
):
    """High-throughput MoE dispatch as a JAX primitive.

    Automatically selects between the intranode and internode XLA custom
    calls based on ``num_rdma_ranks`` (auto-detected from the bound
    ``Buffer`` when omitted).

    Returns ``(recv_x, recv_topk_idx, recv_topk_weights, handle)``.
    ``handle`` is a 6-tuple of arrays for the intranode path and a
    10-tuple for the internode path. ``len(handle)`` is what
    :func:`moe_combine` uses to pick the matching path.
    """
    _ensure_registered()
    config = config or get_dispatch_config(num_ranks)
    nrr, nnp, smb = _resolve_topology(
        num_rdma_ranks, num_max_nvl_peers, source_meta_bytes
    )
    return _moe_dispatch(
        x, topk_idx, topk_weights,
        num_experts, num_ranks, expert_alignment, config, nrr, nnp, smb,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def _moe_dispatch(x, topk_idx, topk_weights, num_experts, num_ranks,
                  expert_alignment, config,
                  num_rdma_ranks, num_max_nvl_peers, source_meta_bytes):
    return _moe_dispatch_impl(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts, num_ranks=num_ranks,
        expert_alignment=expert_alignment, config=config,
        num_rdma_ranks=num_rdma_ranks,
        num_max_nvl_peers=num_max_nvl_peers,
        source_meta_bytes=source_meta_bytes,
    )


def _moe_dispatch_impl(*, x, topk_idx, topk_weights, num_experts, num_ranks,
                       expert_alignment, config,
                       num_rdma_ranks, num_max_nvl_peers, source_meta_bytes):
    if isinstance(x, tuple):
        x_arr, x_scales = x
    else:
        x_arr, x_scales = x, None

    num_tokens = int(x_arr.shape[0])
    num_worst_tokens = num_tokens * num_ranks

    if num_rdma_ranks > 1:
        outs = moe_internode_dispatch_call(
            x_arr, topk_idx, topk_weights,
            num_experts=num_experts,
            num_ranks=num_ranks,
            num_rdma_ranks=num_rdma_ranks,
            num_max_nvl_peers=num_max_nvl_peers,
            source_meta_bytes=source_meta_bytes,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            config=config,
            x_scales=x_scales,
        )
        (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
         is_token_in_rank, num_tokens_per_rank, num_tokens_per_rdma_rank,
         num_tokens_per_expert,
         rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
         gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
         recv_src_meta, recv_rdma_channel_prefix_matrix,
         recv_gbl_channel_prefix_matrix,
         send_rdma_head, send_nvl_head) = outs

        handle = (
            is_token_in_rank,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
        )
    else:
        outs = moe_dispatch_call(
            x_arr, topk_idx, topk_weights,
            num_experts=num_experts, num_ranks=num_ranks,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens, config=config,
            x_scales=x_scales,
        )
        (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
         is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert,
         rank_prefix_matrix, channel_prefix_matrix,
         recv_channel_prefix_matrix, recv_src_idx, send_head) = outs
        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        )

    recv_x_out = (recv_x, recv_x_scales) if x_scales is not None else recv_x
    return recv_x_out, recv_topk_idx, recv_topk_weights, handle


def _moe_dispatch_fwd(x, topk_idx, topk_weights, num_experts, num_ranks,
                      expert_alignment, config,
                      num_rdma_ranks, num_max_nvl_peers, source_meta_bytes):
    out = _moe_dispatch_impl(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts, num_ranks=num_ranks,
        expert_alignment=expert_alignment, config=config,
        num_rdma_ranks=num_rdma_ranks,
        num_max_nvl_peers=num_max_nvl_peers,
        source_meta_bytes=source_meta_bytes,
    )
    residuals = (out[3], config)  # handle + config
    return out, residuals


def _moe_dispatch_bwd(num_experts, num_ranks, expert_alignment, config_outer,
                      num_rdma_ranks_outer, num_max_nvl_peers_outer,
                      source_meta_bytes_outer, residuals, grad_output):
    handle, config = residuals
    grad_recv_x_out, _g_recv_topk_idx, g_recv_topk_weights, _g_handle = grad_output

    if isinstance(grad_recv_x_out, tuple):
        grad_recv_x, _ = grad_recv_x_out
    else:
        grad_recv_x = grad_recv_x_out

    grad_x, grad_topk_weights = _moe_combine_impl(
        x=grad_recv_x, handle=handle, topk_weights=g_recv_topk_weights,
        bias=None, num_ranks=num_ranks,
        config=get_combine_config(num_ranks),
    )

    grad_x_out = (grad_x, None) if isinstance(grad_recv_x_out, tuple) else grad_x
    return (grad_x_out, None, grad_topk_weights)


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


# ---------------------------------------------------------------------------
# moe_combine
# ---------------------------------------------------------------------------


def moe_combine(
    x: jax.Array,
    handle: Tuple,
    topk_weights: Optional[jax.Array] = None,
    bias: Union[jax.Array, Tuple[jax.Array, jax.Array], None] = None,
    num_ranks: int = 1,
    config: Optional[Config] = None,
):
    """High-throughput MoE combine as a JAX primitive.

    The intranode/internode path is selected automatically from the
    length of ``handle``: 6-tuple → intranode, 10-tuple → internode.
    """
    _ensure_registered()
    config = config or get_combine_config(num_ranks)
    return _moe_combine(x, handle, topk_weights, bias, num_ranks, config)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _moe_combine(x, handle, topk_weights, bias, num_ranks, config):
    combined, _ = _moe_combine_impl(
        x=x, handle=handle, topk_weights=topk_weights, bias=bias,
        num_ranks=num_ranks, config=config,
    )
    return combined


def _moe_combine_impl(*, x, handle, topk_weights, bias, num_ranks, config):
    bias_0, bias_1 = _unpack_bias(bias)
    num_topk = 0 if topk_weights is None else int(topk_weights.shape[1])

    if len(handle) == 10:
        # Internode layout.  Indices 3/4/5 are the recv_*_prefix_matrix /
        # recv_*_rank_prefix_sum tensors -- the ones that
        # `Buffer.internode_combine` in the PyTorch wrapper feeds into
        # the runtime as `rdma_channel_prefix_matrix_ptr`,
        # `rdma_rank_prefix_sum_ptr`, `gbl_channel_prefix_matrix_ptr`.
        (
            is_token_in_rank,
            _sender_rdma_channel_prefix_matrix,
            _sender_gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            _recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle
        combined, combined_topk = moe_internode_combine_call(
            x=x, topk_weights=topk_weights, bias_0=bias_0, bias_1=bias_1,
            is_combined_token_in_rank=is_token_in_rank,
            rdma_channel_prefix_matrix=recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum=recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix=recv_gbl_channel_prefix_matrix,
            src_meta=recv_src_meta, send_rdma_head=send_rdma_head,
            send_nvl_head=send_nvl_head,
            num_topk=num_topk, config=config,
        )
    else:
        assert len(handle) == 6, (
            f"Unexpected moe_dispatch handle length: {len(handle)}"
        )
        (rank_prefix_matrix, channel_prefix_matrix,
         _recv_channel_prefix_matrix, src_idx, _is_tir, send_head) = handle
        num_recv_tokens = int(send_head.shape[0])
        combined, combined_topk = moe_combine_call(
            x=x, topk_weights=topk_weights, bias_0=bias_0, bias_1=bias_1,
            src_idx=src_idx, rank_prefix_matrix=rank_prefix_matrix,
            channel_prefix_matrix=channel_prefix_matrix, send_head=send_head,
            num_recv_tokens=num_recv_tokens, num_topk=num_topk, config=config,
        )

    if topk_weights is None:
        combined_topk = None
    return combined, combined_topk


def _moe_combine_fwd(x, handle, topk_weights, bias, num_ranks, config):
    combined, _combined_topk = _moe_combine_impl(
        x=x, handle=handle, topk_weights=topk_weights, bias=bias,
        num_ranks=num_ranks, config=config,
    )
    residuals = (handle, config)
    return combined, residuals


def _moe_combine_bwd(num_ranks_outer, config_outer, residuals, grad_combined):
    handle, config = residuals
    # The backward of combine is a cached-mode dispatch replay.
    # Cached-mode primitives are a planned follow-up; for now we return
    # a zero gradient to keep the interface consistent.
    grad_x = jnp.zeros(grad_combined.shape, grad_combined.dtype)
    grad_handle = jax.tree_util.tree_map(jnp.zeros_like, handle)
    grad_topk = None
    grad_bias = None
    return (grad_x, grad_handle, grad_topk, grad_bias)


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unpack_bias(bias):
    if bias is None:
        return None, None
    if isinstance(bias, tuple):
        assert len(bias) == 2
        return bias
    return bias, None
