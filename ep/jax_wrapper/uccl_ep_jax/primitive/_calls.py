"""Thin wrappers around :func:`jax.ffi.ffi_call` for each UCCL-EP op.

Each function returns a tuple of ``jax.Array``s, making it usable from
inside ``jax.jit``.  Shapes / dtypes must be fully determined from the
inputs and a small number of static parameters (e.g., ``num_experts``,
``num_worst_tokens``) - we follow Primus-Turbo's lead and use
``num_worst_tokens = num_tokens * num_ranks`` as a static upper bound on
the number of received tokens for the high-throughput path.

ABI note (typed FFI):
    All calls below use the XLA Typed FFI. Static integer parameters are
    passed as keyword arguments to ``call(...)`` and surface in the
    MLIR ``stablehlo.custom_call`` as typed ``i32`` attributes, which
    the C++ ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` bindings in
    ``ep/src/uccl_ep_jax.cc`` decode via ``Attr<int32_t>("...")``. The
    legacy ``custom_call_api_version`` / ``legacy_backend_config`` path
    is not used anymore: XLA:GPU removed execution support for the
    legacy ABI (STATUS_RETURNING / ORIGINAL) in late-2025, and the only
    surviving path is ``API_VERSION_TYPED_FFI`` (MLIR enum ``4``), which
    is the default selected by ``jax.ffi.ffi_call`` when
    ``custom_call_api_version`` is omitted.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ._common import itemsize, np_dtype_to_code


def _i32(v) -> np.int32:
    """Coerce a Python int/bool to a numpy ``int32`` so ``ffi_call`` emits
    a typed ``i32`` attribute (the C++ side binds ``Attr<int32_t>``)."""
    return np.int32(int(v))


_fp8_e4m3 = getattr(jnp, "float8_e4m3fnuz", None) or jnp.float8_e4m3fn


# ---------------------------------------------------------------------------
# moe_low_latency_dispatch
# ---------------------------------------------------------------------------


def moe_low_latency_dispatch_call(
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
    """Return ``(recv_x, recv_count, recv_src_info, recv_layout_range,
    recv_x_scales)``.

    ``recv_x_scales`` is always materialized; when ``use_fp8`` is
    False it is a zero-size placeholder (easier to keep a fixed
    output arity for jit).
    """
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])
    num_topk = int(topk_idx.shape[1])
    num_local_experts = num_experts // num_ranks
    num_recv_tokens = num_ranks * num_max_dispatch_tokens_per_rank
    recv_dtype = _fp8_e4m3 if use_fp8 else jnp.bfloat16

    if use_fp8:
        if use_ue8m0:
            scales_shape = (num_local_experts, hidden // 512, num_recv_tokens)
            scales_dtype = jnp.int32
        else:
            scales_shape = (num_local_experts, hidden // 128, num_recv_tokens)
            scales_dtype = jnp.float32
    else:
        # Zero-size placeholder so the output count is static.
        scales_shape = (0,)
        scales_dtype = jnp.float32

    result_shapes = (
        jax.ShapeDtypeStruct((num_local_experts, num_recv_tokens, hidden), recv_dtype),
        jax.ShapeDtypeStruct((num_local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((num_local_experts, num_recv_tokens), jnp.int32),
        jax.ShapeDtypeStruct((num_local_experts, num_ranks), jnp.int64),
        jax.ShapeDtypeStruct(scales_shape, scales_dtype),
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_low_latency_dispatch",
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x,
        topk_idx,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        num_topk=_i32(num_topk),
        num_max_dispatch_tokens_per_rank=_i32(num_max_dispatch_tokens_per_rank),
        num_experts=_i32(num_experts),
        use_fp8=_i32(bool(use_fp8)),
        round_scale=_i32(bool(round_scale)),
        use_ue8m0=_i32(bool(use_ue8m0)),
    )


# ---------------------------------------------------------------------------
# moe_low_latency_combine
# ---------------------------------------------------------------------------


def moe_low_latency_combine_call(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    src_info: jax.Array,
    layout_range: jax.Array,
    *,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    use_logfmt: bool = False,
    zero_copy: bool = False,
):
    num_tokens = int(topk_idx.shape[0])
    num_topk = int(topk_idx.shape[1])
    hidden = int(x.shape[2])
    out_shape = jax.ShapeDtypeStruct((num_tokens, hidden), x.dtype)

    call = jax.ffi.ffi_call(
        "uccl_moe_low_latency_combine",
        out_shape,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x,
        topk_idx,
        topk_weights,
        src_info,
        layout_range,
        x_dim0=_i32(x.shape[0]),
        x_dim1=_i32(x.shape[1]),
        x_dim2=_i32(x.shape[2]),
        num_tokens=_i32(num_tokens),
        num_topk=_i32(num_topk),
        num_max_dispatch_tokens_per_rank=_i32(num_max_dispatch_tokens_per_rank),
        num_experts=_i32(num_experts),
        use_logfmt=_i32(bool(use_logfmt)),
        zero_copy=_i32(bool(zero_copy)),
        src_info_dim0=_i32(src_info.shape[0]),
        src_info_dim1=_i32(src_info.shape[1]),
        layout_range_dim0=_i32(layout_range.shape[0]),
        layout_range_dim1=_i32(layout_range.shape[1]),
    )


# ---------------------------------------------------------------------------
# moe_dispatch (intranode, jit-friendly)
# ---------------------------------------------------------------------------


def _empty_like_scales(x: jax.Array, x_scales: Optional[jax.Array]):
    """Pick a (possibly empty) companion array for the optional ``x_scales``
    operand, and return ``(op, has_scales, num_scales, strides)``."""
    if x_scales is None:
        dummy = jnp.zeros((0,), dtype=jnp.float32)
        return dummy, 0, 0, 0, 0
    assert x_scales.ndim in (1, 2), "x_scales must be 1D or 2D"
    num_scales = 1 if x_scales.ndim == 1 else int(x_scales.shape[1])
    # JAX arrays are dense row-major, so we can reconstruct strides from shape.
    if x_scales.ndim == 1:
        token_stride, hidden_stride = 1, 0
    else:
        token_stride = int(x_scales.shape[1])
        hidden_stride = 1
    return x_scales, 1, num_scales, token_stride, hidden_stride


def moe_dispatch_call(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    num_ranks: int,
    expert_alignment: int = 1,
    num_worst_tokens: Optional[int] = None,
    config,
    x_scales: Optional[jax.Array] = None,
):
    """High-throughput intranode dispatch.

    Returns ``(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
    is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert,
    rank_prefix_matrix, channel_prefix_matrix,
    recv_channel_prefix_matrix, recv_src_idx, send_head)``.

    ``num_worst_tokens`` defaults to ``num_tokens * num_ranks`` and is used
    as a static upper bound on the number of received tokens so the
    output shape is known at trace time.
    """
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])
    num_topk = int(topk_idx.shape[1])
    if num_worst_tokens is None:
        num_worst_tokens = num_tokens * num_ranks
    num_worst_tokens = int(num_worst_tokens)

    (
        x_scales_op,
        has_x_scales,
        num_scales,
        scale_token_stride,
        scale_hidden_stride,
    ) = _empty_like_scales(x, x_scales)

    num_channels = int(config.num_sms) // 2

    scales_out_shape = (
        (num_worst_tokens, num_scales)
        if has_x_scales and x_scales is not None and x_scales.ndim == 2
        else ((num_worst_tokens,) if has_x_scales else (0,))
    )
    scales_out_dtype = x_scales.dtype if x_scales is not None else jnp.float32

    result_shapes = (
        jax.ShapeDtypeStruct((num_worst_tokens, hidden), x.dtype),          # recv_x
        jax.ShapeDtypeStruct(scales_out_shape, scales_out_dtype),          # recv_x_scales
        jax.ShapeDtypeStruct((num_worst_tokens, num_topk), topk_idx.dtype),# recv_topk_idx
        jax.ShapeDtypeStruct((num_worst_tokens, int(topk_weights.shape[1])), topk_weights.dtype),# recv_topk_weights
        jax.ShapeDtypeStruct((num_tokens, num_ranks), jnp.bool_),          # is_token_in_rank
        jax.ShapeDtypeStruct((num_ranks,), jnp.int32),                     # num_tokens_per_rank
        jax.ShapeDtypeStruct((num_experts,), jnp.int32),                   # num_tokens_per_expert
        jax.ShapeDtypeStruct((num_ranks, num_ranks), jnp.int32),           # rank_prefix_matrix
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),        # channel_prefix_matrix
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),        # recv_channel_prefix_matrix
        jax.ShapeDtypeStruct((num_worst_tokens,), jnp.int32),              # recv_src_idx
        jax.ShapeDtypeStruct((num_tokens, num_ranks), jnp.int32),          # send_head
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_dispatch",
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, x_scales_op, topk_idx, topk_weights,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_element_size=_i32(itemsize(x.dtype)),
        num_scales=_i32(num_scales),
        scale_token_stride=_i32(scale_token_stride),
        scale_hidden_stride=_i32(scale_hidden_stride),
        num_topk=_i32(num_topk),
        num_experts=_i32(num_experts),
        expert_alignment=_i32(expert_alignment),
        num_worst_tokens=_i32(num_worst_tokens),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
        has_x_scales=_i32(has_x_scales),
    )


# ---------------------------------------------------------------------------
# moe_combine (intranode, jit-friendly)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# moe_cached_dispatch (intranode replay using cached handle)
# ---------------------------------------------------------------------------


def moe_cached_dispatch_call(
    x: jax.Array,
    is_token_in_rank: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    recv_src_idx: jax.Array,
    send_head: jax.Array,
    *,
    num_recv_tokens: int,
    config,
    x_scales: Optional[jax.Array] = None,
):
    """Replay an intranode dispatch using the layout handle produced by
    :func:`moe_dispatch_call`.

    Output shape is ``(num_recv_tokens, hidden)`` — same as the first
    element of the original dispatch output, but the caller is
    responsible for passing the matching ``num_recv_tokens`` (the
    ``num_worst_tokens`` of the original dispatch).
    """
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])

    (x_scales_op, has_x_scales, num_scales, scale_token_stride,
     scale_hidden_stride) = _empty_like_scales(x, x_scales)

    scales_out_shape = (
        (int(num_recv_tokens), num_scales)
        if has_x_scales and x_scales is not None and x_scales.ndim == 2
        else ((int(num_recv_tokens),) if has_x_scales else (0,))
    )
    scales_out_dtype = x_scales.dtype if x_scales is not None else jnp.float32

    result_shapes = (
        jax.ShapeDtypeStruct((int(num_recv_tokens), hidden), x.dtype),  # recv_x
        jax.ShapeDtypeStruct(scales_out_shape, scales_out_dtype),       # recv_x_scales
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_cached_dispatch",
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, x_scales_op, is_token_in_rank, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix,
        recv_src_idx, send_head,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_element_size=_i32(itemsize(x.dtype)),
        num_scales=_i32(num_scales),
        scale_token_stride=_i32(scale_token_stride),
        scale_hidden_stride=_i32(scale_hidden_stride),
        num_recv_tokens=_i32(num_recv_tokens),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
        has_x_scales=_i32(has_x_scales),
    )


# ---------------------------------------------------------------------------
# moe_internode_cached_dispatch (replay using cached internode handle)
# ---------------------------------------------------------------------------


def moe_internode_cached_dispatch_call(
    x: jax.Array,
    is_token_in_rank: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    *,
    num_recv_tokens: int,
    num_rdma_recv_tokens: int,
    num_experts: int,
    config,
    x_scales: Optional[jax.Array] = None,
):
    """Replay an internode dispatch using the layout handle produced by
    :func:`moe_internode_dispatch_call`."""
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])

    (x_scales_op, has_x_scales, num_scales, scale_token_stride,
     scale_hidden_stride) = _empty_like_scales(x, x_scales)

    scales_out_shape = (
        (int(num_recv_tokens), num_scales)
        if has_x_scales and x_scales is not None and x_scales.ndim == 2
        else ((int(num_recv_tokens),) if has_x_scales else (0,))
    )
    scales_out_dtype = x_scales.dtype if x_scales is not None else jnp.float32

    result_shapes = (
        jax.ShapeDtypeStruct((int(num_recv_tokens), hidden), x.dtype),  # recv_x
        jax.ShapeDtypeStruct(scales_out_shape, scales_out_dtype),       # recv_x_scales
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_internode_cached_dispatch",
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, x_scales_op, is_token_in_rank, rdma_channel_prefix_matrix,
        gbl_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
        recv_gbl_rank_prefix_sum,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_element_size=_i32(itemsize(x.dtype)),
        num_scales=_i32(num_scales),
        scale_token_stride=_i32(scale_token_stride),
        scale_hidden_stride=_i32(scale_hidden_stride),
        num_recv_tokens=_i32(num_recv_tokens),
        num_rdma_recv_tokens=_i32(num_rdma_recv_tokens),
        num_experts=_i32(num_experts),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
        has_x_scales=_i32(has_x_scales),
    )


# ---------------------------------------------------------------------------
# moe_internode_dispatch (jit-friendly, multi-node)
# ---------------------------------------------------------------------------


def moe_internode_dispatch_call(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    num_ranks: int,
    num_rdma_ranks: int,
    num_max_nvl_peers: int,
    source_meta_bytes: int,
    expert_alignment: int = 1,
    num_worst_tokens: Optional[int] = None,
    config,
    x_scales: Optional[jax.Array] = None,
):
    """Internode high-throughput dispatch as an XLA custom call.

    Returns the full internode handle in a deterministic tuple order,
    matching the C++ FFI handler:

        (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
         is_token_in_rank,
         num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
         rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
         gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
         recv_src_meta, recv_rdma_channel_prefix_matrix,
         recv_gbl_channel_prefix_matrix,
         send_rdma_head, send_nvl_head)
    """
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])
    num_topk = int(topk_idx.shape[1])
    if num_worst_tokens is None:
        num_worst_tokens = num_tokens * num_ranks
    num_worst_tokens = int(num_worst_tokens)

    (x_scales_op, has_x_scales, num_scales, scale_token_stride,
     scale_hidden_stride) = _empty_like_scales(x, x_scales)

    num_channels = int(config.num_sms) // 2

    scales_out_shape = (
        (num_worst_tokens, num_scales)
        if has_x_scales and x_scales is not None and x_scales.ndim == 2
        else ((num_worst_tokens,) if has_x_scales else (0,))
    )
    scales_out_dtype = x_scales.dtype if x_scales is not None else jnp.float32

    result_shapes = (
        jax.ShapeDtypeStruct((num_worst_tokens, hidden), x.dtype),           # recv_x
        jax.ShapeDtypeStruct(scales_out_shape, scales_out_dtype),            # recv_x_scales
        jax.ShapeDtypeStruct((num_worst_tokens, num_topk), topk_idx.dtype),  # recv_topk_idx
        jax.ShapeDtypeStruct((num_worst_tokens, int(topk_weights.shape[1])), topk_weights.dtype),  # recv_topk_weights
        jax.ShapeDtypeStruct((num_tokens, num_ranks), jnp.bool_),            # is_token_in_rank
        jax.ShapeDtypeStruct((num_ranks,), jnp.int32),                       # num_tokens_per_rank
        jax.ShapeDtypeStruct((num_rdma_ranks,), jnp.int32),                  # num_tokens_per_rdma_rank
        jax.ShapeDtypeStruct((num_experts,), jnp.int32),                     # num_tokens_per_expert
        jax.ShapeDtypeStruct((num_rdma_ranks, num_channels), jnp.int32),     # rdma_channel_prefix_matrix
        jax.ShapeDtypeStruct((num_rdma_ranks,), jnp.int32),                  # recv_rdma_rank_prefix_sum
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),          # gbl_channel_prefix_matrix
        jax.ShapeDtypeStruct((num_ranks,), jnp.int32),                       # recv_gbl_rank_prefix_sum
        jax.ShapeDtypeStruct((num_worst_tokens, int(source_meta_bytes)), jnp.uint8),  # recv_src_meta
        jax.ShapeDtypeStruct((num_rdma_ranks, num_channels), jnp.int32),     # recv_rdma_channel_prefix_matrix
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),          # recv_gbl_channel_prefix_matrix
        jax.ShapeDtypeStruct((num_tokens, num_rdma_ranks), jnp.int32),       # send_rdma_head
        jax.ShapeDtypeStruct((num_worst_tokens, int(num_max_nvl_peers)), jnp.int32),   # send_nvl_head
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_internode_dispatch",
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, x_scales_op, topk_idx, topk_weights,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_element_size=_i32(itemsize(x.dtype)),
        num_scales=_i32(num_scales),
        scale_token_stride=_i32(scale_token_stride),
        scale_hidden_stride=_i32(scale_hidden_stride),
        num_topk=_i32(num_topk),
        num_experts=_i32(num_experts),
        expert_alignment=_i32(expert_alignment),
        num_worst_tokens=_i32(num_worst_tokens),
        source_meta_bytes=_i32(source_meta_bytes),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
        has_x_scales=_i32(has_x_scales),
    )


# ---------------------------------------------------------------------------
# moe_internode_combine (jit-friendly, multi-node)
# ---------------------------------------------------------------------------


def moe_internode_combine_call(
    x: jax.Array,
    topk_weights: Optional[jax.Array],
    bias_0: Optional[jax.Array],
    bias_1: Optional[jax.Array],
    is_combined_token_in_rank: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    src_meta: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    *,
    num_topk: int,
    config,
):
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])
    num_combined_tokens = int(is_combined_token_in_rank.shape[0])

    has_topk = topk_weights is not None
    if topk_weights is None:
        topk_weights = jnp.zeros((0,), dtype=jnp.float32)
    has_b0 = bias_0 is not None
    if bias_0 is None:
        bias_0 = jnp.zeros((0,), dtype=x.dtype)
    has_b1 = bias_1 is not None
    if bias_1 is None:
        bias_1 = jnp.zeros((0,), dtype=x.dtype)

    combined_shape = jax.ShapeDtypeStruct(
        (num_combined_tokens, hidden), x.dtype
    )
    combined_topk_shape = jax.ShapeDtypeStruct(
        (num_combined_tokens, int(num_topk)) if has_topk else (0,),
        topk_weights.dtype if has_topk else jnp.float32,
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_internode_combine",
        (combined_shape, combined_topk_shape),
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, topk_weights, bias_0, bias_1,
        is_combined_token_in_rank, rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
        src_meta, send_rdma_head, send_nvl_head,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_dtype_code=_i32(np_dtype_to_code(x.dtype)),
        x_element_size=_i32(itemsize(x.dtype)),
        num_topk=_i32(num_topk),
        num_combined_tokens=_i32(num_combined_tokens),
        has_topk_weights=_i32(has_topk),
        has_bias_0=_i32(has_b0),
        has_bias_1=_i32(has_b1),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
    )


def moe_combine_call(
    x: jax.Array,
    topk_weights: Optional[jax.Array],
    bias_0: Optional[jax.Array],
    bias_1: Optional[jax.Array],
    src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    *,
    num_recv_tokens: int,
    num_topk: int,
    config,
):
    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])

    has_topk = topk_weights is not None
    if topk_weights is None:
        topk_weights = jnp.zeros((0,), dtype=jnp.float32)
    has_b0 = bias_0 is not None
    if bias_0 is None:
        bias_0 = jnp.zeros((0,), dtype=x.dtype)
    has_b1 = bias_1 is not None
    if bias_1 is None:
        bias_1 = jnp.zeros((0,), dtype=x.dtype)

    combined_shape = jax.ShapeDtypeStruct((int(num_recv_tokens), hidden), x.dtype)
    combined_topk_shape = jax.ShapeDtypeStruct(
        (int(num_recv_tokens), int(num_topk)) if has_topk else (0,),
        topk_weights.dtype if has_topk else jnp.float32,
    )

    call = jax.ffi.ffi_call(
        "uccl_moe_combine",
        (combined_shape, combined_topk_shape),
        has_side_effect=True,
        vmap_method="sequential",
    )
    return call(
        x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
        channel_prefix_matrix, send_head,
        num_tokens=_i32(num_tokens),
        hidden=_i32(hidden),
        x_dtype_code=_i32(np_dtype_to_code(x.dtype)),
        x_element_size=_i32(itemsize(x.dtype)),
        num_topk=_i32(num_topk),
        num_recv_tokens=_i32(num_recv_tokens),
        has_topk_weights=_i32(has_topk),
        has_bias_0=_i32(has_b0),
        has_bias_1=_i32(has_b1),
        num_sms=_i32(config.num_sms),
        num_max_nvl_chunked_send_tokens=_i32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=_i32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=_i32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=_i32(config.num_max_rdma_chunked_recv_tokens),
    )
