"""High-level dispatch / combine operations exposed to JAX users.

These wrappers mirror the Primus-Turbo public API
(:func:`moe_dispatch` / :func:`moe_combine`) and add low-latency
variants (:func:`low_latency_dispatch` / :func:`low_latency_combine`)
that are unique to UCCL-EP.

Both the *intranode* (NVLink/xGMI only) and *internode* (RDMA +
NVLink/xGMI) paths are supported; the active path is selected based on
``Buffer.num_rdma_ranks`` just like the PyTorch wrapper.

The operations execute eagerly on the JAX default stream (i.e., they
block until the required inputs are materialized and produce ready
``jax.Array`` values) which matches the approach used by
``mori.jax.ops``.  When the user runs inside a ``jax.jit`` region they
should hide these calls behind a ``jax.pure_callback`` /
``jax.experimental.io_callback`` boundary - a future iteration can wrap
them as proper XLA FFI custom calls.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .bootstrap import Buffer, get_buffer
from .config import Config, get_combine_config, get_dispatch_config


# ---------------------------------------------------------------------------
# Generic helpers: obtain raw device pointers / element sizes from JAX arrays
# ---------------------------------------------------------------------------


def _ensure_jax_array(x):
    import jax
    import jax.numpy as jnp

    if isinstance(x, jax.Array):
        return x
    return jnp.asarray(x)


def _device_ptr(arr) -> int:
    """Return the raw CUDA/HIP pointer backing a single-device JAX array."""
    if arr is None:
        return 0
    arr = _ensure_jax_array(arr)
    # Ensure the computation is materialized before we read the pointer.
    arr.block_until_ready()
    return int(arr.unsafe_buffer_pointer())


def _element_size(arr) -> int:
    arr = _ensure_jax_array(arr)
    return int(arr.dtype.itemsize)


def _dtype_code(arr) -> int:
    """Map a JAX dtype onto the integer codes used by ``uccl.ep``."""
    import jax.numpy as jnp

    if arr is None:
        return 0
    dt = arr.dtype
    table = {
        jnp.uint8: 0,
        jnp.int8: 1,
        jnp.int16: 2,
        jnp.int32: 3,
        jnp.int64: 4,
        jnp.float16: 5,
        jnp.bfloat16: 6,
        jnp.float32: 7,
        jnp.float64: 8,
        jnp.bool_: 9,
    }
    if hasattr(jnp, "float8_e4m3fn") and dt == jnp.float8_e4m3fn:
        return 10
    if hasattr(jnp, "float8_e4m3fnuz") and dt == jnp.float8_e4m3fnuz:
        return 10
    for t, code in table.items():
        if np.issubdtype(dt, t):
            return code
    raise ValueError(f"Unsupported JAX dtype for uccl combine: {dt}")


def _compute_stream_ptr(buf: Buffer, arr=None) -> int:
    """Return a CUDA stream pointer to chain the EP kernels against.

    JAX does not surface per-array streams publicly; we fall back to the
    default stream (0). The comm stream owned by the C++ runtime handles
    the actual ordering.
    """
    return 0


def _empty(shape, dtype, like=None):
    """Allocate a JAX-owned device array of the requested shape / dtype."""
    import jax
    import jax.numpy as jnp

    if like is not None:
        dev = like.device
    else:
        dev = jax.local_devices()[0]
    return jax.device_put(jnp.empty(shape, dtype=dtype), device=dev)


def _row_major_strides(arr) -> Tuple[int, int]:
    """Return ``(token_stride, hidden_stride)`` in *element* units.

    JAX arrays don't expose a public stride attribute, so we reconstruct
    the row-major strides from the shape. This matches how JAX currently
    materializes arrays on device (dense, C-contiguous).
    """
    if arr is None:
        return 0, 0
    if arr.ndim == 1:
        return 1, 0
    if arr.ndim == 2:
        return int(arr.shape[1]), 1
    raise ValueError(f"Unsupported x_scales rank: {arr.ndim}")


def _slice_rows(arr, num_rows: int):
    """Materialize ``arr[:num_rows]`` as a JAX array.

    The returned array is a proper JAX owned copy rather than a view, so
    the caller is free to return it without worrying about the parent
    staying alive.
    """
    import jax.numpy as jnp

    if arr is None:
        return None
    if int(num_rows) >= int(arr.shape[0]):
        return arr
    return jnp.asarray(arr[:num_rows])


# ---------------------------------------------------------------------------
# Low-latency dispatch / combine
# ---------------------------------------------------------------------------


def low_latency_dispatch(
    x,
    topk_idx,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    *,
    use_fp8: bool = True,
    round_scale: bool = False,
    use_ue8m0: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    cumulative_local_expert_recv_stats=None,
    dispatch_wait_recv_cost_stats=None,
    buffer: Optional[Buffer] = None,
):
    """JAX port of ``Buffer.low_latency_dispatch``.

    Returns ``(recv_x, recv_count, handle, event, hook)`` matching the
    semantics of the PyTorch wrapper.
    """
    import jax.numpy as jnp

    buf = buffer or get_buffer()
    assert buf.low_latency_mode, "buffer was not created with low_latency_mode=True"

    x = _ensure_jax_array(x)
    topk_idx = _ensure_jax_array(topk_idx)
    assert x.ndim == 2, "x must be [num_tokens, hidden]"
    assert topk_idx.ndim == 2, "topk_idx must be [num_tokens, num_topk]"

    num_tokens, hidden = int(x.shape[0]), int(x.shape[1])
    num_topk = int(topk_idx.shape[1])
    num_ranks = buf.world_size
    num_local_experts = num_experts // num_ranks
    num_recv_tokens = num_ranks * num_max_dispatch_tokens_per_rank

    for proxy in buf.proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            num_tokens=num_tokens,
            hidden=hidden,
            num_experts=num_experts,
        )

    if use_fp8:
        fp8_dtype = getattr(jnp, "float8_e4m3fnuz", None) or jnp.float8_e4m3fn
        recv_dtype = fp8_dtype
    else:
        recv_dtype = jnp.bfloat16

    packed_recv_x = _empty((num_local_experts, num_recv_tokens, hidden), recv_dtype, like=x)
    packed_recv_count = _empty((num_local_experts,), jnp.int32, like=x)
    packed_recv_src_info = _empty((num_local_experts, num_recv_tokens), jnp.int32, like=x)
    packed_recv_layout_range = _empty((num_local_experts, num_ranks), jnp.int64, like=x)

    packed_recv_x_scales = None
    packed_recv_x_scales_ptr = 0
    if use_fp8:
        if use_ue8m0:
            storage = _empty(
                (num_local_experts, hidden // 512, num_recv_tokens),
                jnp.int32,
                like=x,
            )
        else:
            storage = _empty(
                (num_local_experts, hidden // 128, num_recv_tokens),
                jnp.float32,
                like=x,
            )
        packed_recv_x_scales = jnp.transpose(storage, (0, 2, 1))
        packed_recv_x_scales_ptr = _device_ptr(storage)

    event, hook = buf.runtime.low_latency_dispatch(
        _device_ptr(x),
        num_tokens,
        hidden,
        _device_ptr(topk_idx),
        num_tokens,
        num_topk,
        _device_ptr(packed_recv_x),
        packed_recv_x_scales_ptr,
        _device_ptr(packed_recv_count),
        _device_ptr(packed_recv_src_info),
        _device_ptr(packed_recv_layout_range),
        _device_ptr(cumulative_local_expert_recv_stats)
        if cumulative_local_expert_recv_stats is not None
        else 0,
        _device_ptr(dispatch_wait_recv_cost_stats)
        if dispatch_wait_recv_cost_stats is not None
        else 0,
        _compute_stream_ptr(buf, x),
        int(num_max_dispatch_tokens_per_rank),
        int(num_experts),
        bool(use_fp8),
        bool(round_scale),
        bool(use_ue8m0),
        bool(async_finish),
        bool(return_recv_hook),
    )

    handle = (
        packed_recv_src_info,
        packed_recv_layout_range,
        int(num_max_dispatch_tokens_per_rank),
        hidden,
        num_experts,
    )
    recv_x_out = (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x
    return recv_x_out, packed_recv_count, handle, event, hook


def low_latency_combine(
    x,
    topk_idx,
    topk_weights,
    handle,
    *,
    use_logfmt: bool = False,
    zero_copy: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    out=None,
    combine_wait_recv_cost_stats=None,
    buffer: Optional[Buffer] = None,
):
    """JAX port of ``Buffer.low_latency_combine``."""
    buf = buffer or get_buffer()
    assert buf.low_latency_mode, "buffer was not created with low_latency_mode=True"

    (src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts) = handle

    x = _ensure_jax_array(x)
    topk_idx = _ensure_jax_array(topk_idx)
    topk_weights = _ensure_jax_array(topk_weights)

    if out is None:
        combined_x = _empty((int(topk_idx.shape[0]), int(hidden)), x.dtype, like=x)
    else:
        combined_x = out

    event, hook = buf.runtime.low_latency_combine(
        _device_ptr(x),
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.shape[2]),
        _device_ptr(topk_idx),
        int(topk_idx.shape[0]),
        int(topk_idx.shape[1]),
        _device_ptr(topk_weights),
        _device_ptr(src_info),
        int(src_info.shape[0]),
        int(src_info.shape[1]),
        _device_ptr(layout_range),
        int(layout_range.shape[0]),
        int(layout_range.shape[1]),
        _device_ptr(combine_wait_recv_cost_stats)
        if combine_wait_recv_cost_stats is not None
        else 0,
        _compute_stream_ptr(buf, x),
        int(num_max_dispatch_tokens_per_rank),
        int(num_experts),
        bool(use_logfmt),
        bool(zero_copy),
        bool(async_finish),
        bool(return_recv_hook),
        _device_ptr(combined_x),
    )
    return combined_x, event, hook


def get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_ranks: int,
    num_experts: int,
) -> int:
    """Mirror of ``uccl.ep.get_low_latency_rdma_size_hint``."""
    from .bootstrap import _require_uccl_ep

    return int(
        _require_uccl_ep().get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )
    )


# ---------------------------------------------------------------------------
# High-throughput dispatch / combine - Primus-Turbo API
# (intranode when num_rdma_ranks == 1, internode when > 1)
# ---------------------------------------------------------------------------


def _get_dispatch_layout(buf: Buffer, topk_idx, num_experts: int):
    """Compute ``num_tokens_per_rank``, ``num_tokens_per_rdma_rank``,
    ``num_tokens_per_expert`` and ``is_token_in_rank`` on device."""
    import jax.numpy as jnp

    num_tokens, num_topk = int(topk_idx.shape[0]), int(topk_idx.shape[1])
    num_ranks = buf.world_size

    num_tokens_per_rank = _empty((num_ranks,), jnp.int32, like=topk_idx)
    num_tokens_per_rdma_rank = None
    if buf.num_rdma_ranks > 1:
        num_tokens_per_rdma_rank = _empty(
            (buf.num_rdma_ranks,), jnp.int32, like=topk_idx
        )
    num_tokens_per_expert = _empty((num_experts,), jnp.int32, like=topk_idx)
    is_token_in_rank = _empty((num_tokens, num_ranks), jnp.bool_, like=topk_idx)

    buf.runtime.get_dispatch_layout(
        _device_ptr(topk_idx),
        num_tokens,
        num_topk,
        num_experts,
        _device_ptr(num_tokens_per_rank),
        _device_ptr(num_tokens_per_rdma_rank) if num_tokens_per_rdma_rank is not None else 0,
        _device_ptr(num_tokens_per_expert),
        _device_ptr(is_token_in_rank),
        None,
        False,
        False,
        _compute_stream_ptr(buf, topk_idx),
    )
    return (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    )


def _intranode_dispatch(
    buf: Buffer,
    x_arr,
    x_scales,
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int,
    uccl_config,
):
    import jax.numpy as jnp

    num_tokens, hidden = int(x_arr.shape[0]), int(x_arr.shape[1])
    num_ranks = buf.world_size

    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
    ) = _get_dispatch_layout(buf, topk_idx, num_experts)

    num_channels = int(getattr(uccl_config, "num_sms", 20)) // 2
    rank_prefix_matrix = _empty((num_ranks, num_ranks), jnp.int32, like=x_arr)
    channel_prefix_matrix = _empty((num_ranks, num_channels), jnp.int32, like=x_arr)

    num_recv_tokens, num_recv_tokens_per_expert_list, _ = buf.runtime.intranode_prepare(
        _device_ptr(num_tokens_per_rank),
        _device_ptr(is_token_in_rank),
        _device_ptr(num_tokens_per_expert),
        num_tokens,
        num_experts,
        _device_ptr(rank_prefix_matrix),
        _device_ptr(channel_prefix_matrix),
        int(expert_alignment),
        0,
        uccl_config,
        None,
        False,
        False,
        _compute_stream_ptr(buf, x_arr),
    )

    num_scales = 0 if x_scales is None else (1 if x_scales.ndim == 1 else int(x_scales.shape[1]))
    scale_token_stride, scale_hidden_stride = _row_major_strides(x_scales)

    recv_x = _empty((num_recv_tokens, hidden), x_arr.dtype, like=x_arr)
    recv_topk_idx = _empty(
        (num_recv_tokens, int(topk_idx.shape[1])), topk_idx.dtype, like=x_arr
    )
    recv_topk_weights = _empty(
        (num_recv_tokens, int(topk_weights.shape[1])), topk_weights.dtype, like=x_arr
    )
    recv_src_idx = _empty((num_recv_tokens,), jnp.int32, like=x_arr)
    recv_channel_prefix_matrix = _empty(
        (num_ranks, num_channels), jnp.int32, like=x_arr
    )
    send_head = _empty((num_tokens, num_ranks), jnp.int32, like=x_arr)
    recv_x_scales = None
    if x_scales is not None:
        recv_x_scales = _empty(
            (num_recv_tokens,) if x_scales.ndim == 1 else (num_recv_tokens, num_scales),
            x_scales.dtype,
            like=x_arr,
        )

    buf.runtime.intranode_dispatch(
        _device_ptr(x_arr),
        num_tokens,
        hidden,
        _element_size(x_arr),
        _device_ptr(x_scales) if x_scales is not None else 0,
        int(num_scales),
        int(scale_token_stride),
        int(scale_hidden_stride),
        _device_ptr(topk_idx),
        int(topk_idx.shape[1]),
        _device_ptr(topk_weights),
        _device_ptr(is_token_in_rank),
        _device_ptr(rank_prefix_matrix),
        _device_ptr(channel_prefix_matrix),
        int(num_experts),
        0,
        False,
        uccl_config,
        int(num_recv_tokens),
        _device_ptr(recv_x),
        _device_ptr(recv_x_scales) if recv_x_scales is not None else 0,
        _device_ptr(recv_topk_idx),
        _device_ptr(recv_topk_weights),
        _device_ptr(recv_channel_prefix_matrix),
        _device_ptr(recv_src_idx),
        _device_ptr(send_head),
        None,
        False,
        False,
        _compute_stream_ptr(buf, x_arr),
    )

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


def _internode_dispatch(
    buf: Buffer,
    x_arr,
    x_scales,
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int,
    uccl_config,
):
    """Internode (multi-node) high-throughput dispatch.

    Mirrors ``Buffer.internode_dispatch`` in ``deep_ep_wrapper/buffer.py``.
    """
    import jax.numpy as jnp

    num_tokens, hidden = int(x_arr.shape[0]), int(x_arr.shape[1])
    num_topk = int(topk_idx.shape[1])
    num_ranks = buf.world_size
    num_rdma_ranks = buf.num_rdma_ranks
    num_channels = int(getattr(uccl_config, "num_sms", 20)) // 2

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    ) = _get_dispatch_layout(buf, topk_idx, num_experts)
    assert num_tokens_per_rdma_rank is not None

    rdma_channel_prefix_matrix = _empty(
        (num_rdma_ranks, num_channels), jnp.int32, like=x_arr
    )
    recv_rdma_rank_prefix_sum = _empty((num_rdma_ranks,), jnp.int32, like=x_arr)
    gbl_channel_prefix_matrix = _empty(
        (num_ranks, num_channels), jnp.int32, like=x_arr
    )
    recv_gbl_rank_prefix_sum = _empty((num_ranks,), jnp.int32, like=x_arr)

    num_scales = (
        0 if x_scales is None else (1 if x_scales.ndim == 1 else int(x_scales.shape[1]))
    )
    scale_token_stride, scale_hidden_stride = _row_major_strides(x_scales)

    (
        num_recv_tokens,
        num_rdma_recv_tokens,
        num_recv_tokens_per_expert_list,
        _,
    ) = buf.runtime.internode_prepare(
        _device_ptr(num_tokens_per_rank),
        _device_ptr(num_tokens_per_rdma_rank),
        _device_ptr(num_tokens_per_expert),
        _device_ptr(is_token_in_rank),
        num_tokens,
        hidden,
        _element_size(x_arr),
        int(num_scales),
        num_topk,
        num_experts,
        int(expert_alignment),
        0,  # num_worst_tokens
        uccl_config,
        _device_ptr(rdma_channel_prefix_matrix),
        _device_ptr(recv_rdma_rank_prefix_sum),
        _device_ptr(gbl_channel_prefix_matrix),
        _device_ptr(recv_gbl_rank_prefix_sum),
        None,
        False,
        False,
        _compute_stream_ptr(buf, x_arr),
    )

    # Over-allocate so every rank has at least 1 row (matches the PyTorch
    # wrapper: zero-row tensors produce null pointers which trip C++
    # assertions).
    alloc_recv_tokens = max(num_recv_tokens, 1)
    alloc_rdma_recv_tokens = max(num_rdma_recv_tokens, 1)

    recv_x = _empty((alloc_recv_tokens, hidden), x_arr.dtype, like=x_arr)
    recv_x_scales = None
    if x_scales is not None:
        recv_x_scales = _empty(
            (alloc_recv_tokens,) if x_scales.ndim == 1 else (alloc_recv_tokens, num_scales),
            x_scales.dtype,
            like=x_arr,
        )
    recv_topk_idx = _empty(
        (alloc_recv_tokens, num_topk), topk_idx.dtype, like=x_arr
    )
    recv_topk_weights = _empty(
        (alloc_recv_tokens, int(topk_weights.shape[1])), topk_weights.dtype, like=x_arr
    )
    source_meta_bytes = int(buf.runtime.get_source_meta_bytes())
    recv_src_meta = _empty(
        (alloc_recv_tokens, source_meta_bytes), jnp.uint8, like=x_arr
    )
    recv_rdma_channel_prefix_matrix = _empty(
        (num_rdma_ranks, num_channels), jnp.int32, like=x_arr
    )
    recv_gbl_channel_prefix_matrix = _empty(
        (num_ranks, num_channels), jnp.int32, like=x_arr
    )
    send_rdma_head = _empty((num_tokens, num_rdma_ranks), jnp.int32, like=x_arr)
    send_nvl_head = _empty(
        (alloc_rdma_recv_tokens, int(buf.runtime.get_num_max_nvl_peers())),
        jnp.int32,
        like=x_arr,
    )

    buf.runtime.internode_dispatch(
        _device_ptr(x_arr),
        num_tokens,
        hidden,
        _element_size(x_arr),
        _device_ptr(x_scales) if x_scales is not None else 0,
        int(num_scales),
        int(scale_token_stride),
        int(scale_hidden_stride),
        _device_ptr(topk_idx),
        num_topk,
        _device_ptr(topk_weights),
        _device_ptr(is_token_in_rank),
        _device_ptr(rdma_channel_prefix_matrix),
        _device_ptr(recv_rdma_rank_prefix_sum),
        _device_ptr(gbl_channel_prefix_matrix),
        _device_ptr(recv_gbl_rank_prefix_sum),
        num_experts,
        0,  # num_worst_tokens
        False,  # cached_mode
        int(num_rdma_recv_tokens),
        uccl_config,
        _device_ptr(recv_x),
        _device_ptr(recv_x_scales) if recv_x_scales is not None else 0,
        _device_ptr(recv_topk_idx),
        _device_ptr(recv_topk_weights),
        _device_ptr(recv_src_meta),
        _device_ptr(recv_rdma_channel_prefix_matrix),
        _device_ptr(recv_gbl_channel_prefix_matrix),
        _device_ptr(send_rdma_head),
        _device_ptr(send_nvl_head),
        None,
        False,
        False,
        _compute_stream_ptr(buf, x_arr),
    )

    # Slice back to the real number of received tokens.
    recv_x = _slice_rows(recv_x, num_recv_tokens)
    if recv_x_scales is not None:
        recv_x_scales = _slice_rows(recv_x_scales, num_recv_tokens)
    recv_topk_idx = _slice_rows(recv_topk_idx, num_recv_tokens)
    recv_topk_weights = _slice_rows(recv_topk_weights, num_recv_tokens)
    recv_src_meta = _slice_rows(recv_src_meta, num_recv_tokens)
    send_nvl_head = _slice_rows(send_nvl_head, max(num_rdma_recv_tokens, 1))

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
    recv_x_out = (recv_x, recv_x_scales) if x_scales is not None else recv_x
    return recv_x_out, recv_topk_idx, recv_topk_weights, handle


def moe_dispatch(
    x: Union[object, Tuple[object, object]],
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
    *,
    buffer: Optional[Buffer] = None,
):
    """Primus-Turbo-compatible MoE dispatch over UCCL-EP.

    Automatically selects the intranode path (``num_rdma_ranks == 1``)
    or the internode path (``num_rdma_ranks > 1``), matching the
    behaviour of :meth:`Buffer.dispatch` in ``deep_ep_wrapper``.

    Returns ``(recv_x, recv_topk_idx, recv_topk_weights, handle)``.
    """
    buf = buffer or get_buffer()
    config = config or get_dispatch_config(buf.world_size)
    uccl_config = config.to_uccl_config()

    if isinstance(x, tuple):
        x_arr, x_scales = x
    else:
        x_arr, x_scales = x, None
    x_arr = _ensure_jax_array(x_arr)
    if x_scales is not None:
        x_scales = _ensure_jax_array(x_scales)
    topk_idx = _ensure_jax_array(topk_idx)
    topk_weights = _ensure_jax_array(topk_weights)

    if buf.num_rdma_ranks > 1:
        return _internode_dispatch(
            buf, x_arr, x_scales, topk_idx, topk_weights,
            num_experts, expert_alignment, uccl_config,
        )
    return _intranode_dispatch(
        buf, x_arr, x_scales, topk_idx, topk_weights,
        num_experts, expert_alignment, uccl_config,
    )


def _intranode_combine(
    buf: Buffer, x, handle, topk_weights, bias, uccl_config
):
    (
        rank_prefix_matrix,
        _,
        channel_prefix_matrix,
        src_idx,
        is_recv_token_in_rank,
        send_head,
    ) = handle
    bias_0, bias_1 = _unpack_bias(bias)

    num_recv_tokens = int(send_head.shape[0])
    num_topk = 0 if topk_weights is None else int(topk_weights.shape[1])

    combined_x = _empty((num_recv_tokens, int(x.shape[1])), x.dtype, like=x)
    combined_topk_weights = None
    if topk_weights is not None:
        combined_topk_weights = _empty(
            (num_recv_tokens, num_topk), topk_weights.dtype, like=x
        )

    buf.runtime.intranode_combine(
        _device_ptr(x),
        int(x.shape[0]),
        int(x.shape[1]),
        _dtype_code(x),
        _element_size(x),
        _device_ptr(topk_weights) if topk_weights is not None else 0,
        num_topk,
        _device_ptr(bias_0) if bias_0 is not None else 0,
        _device_ptr(bias_1) if bias_1 is not None else 0,
        _device_ptr(src_idx),
        num_recv_tokens,
        _device_ptr(rank_prefix_matrix),
        _device_ptr(channel_prefix_matrix),
        _device_ptr(send_head),
        uccl_config,
        _device_ptr(combined_x),
        _device_ptr(combined_topk_weights)
        if combined_topk_weights is not None
        else 0,
        None,
        False,
        False,
        _compute_stream_ptr(buf, x),
    )
    return combined_x, combined_topk_weights


def _internode_combine(
    buf: Buffer, x, handle, topk_weights, bias, uccl_config
):
    """Internode (multi-node) high-throughput combine.

    Mirrors ``Buffer.internode_combine`` in ``deep_ep_wrapper/buffer.py``.
    The handle layout matches the one produced by ``_internode_dispatch``.
    """
    import jax.numpy as jnp

    (
        is_combined_token_in_rank,
        _,
        _,
        rdma_channel_prefix_matrix,
        rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        _,  # gbl_rank_prefix_sum (unused on the combine side)
        src_meta,
        send_rdma_head,
        send_nvl_head,
    ) = handle
    bias_0, bias_1 = _unpack_bias(bias)

    num_combined_tokens = int(is_combined_token_in_rank.shape[0])
    num_topk = 0 if topk_weights is None else int(topk_weights.shape[1])

    # Guard against zero-row inputs, matching the PyTorch wrapper: we
    # allocate at least one row so ``unsafe_buffer_pointer`` is valid.
    alloc_combined_tokens = max(num_combined_tokens, 1)
    if int(x.shape[0]) == 0:
        x = _empty((1, int(x.shape[1])), x.dtype, like=x)
    if int(src_meta.shape[0]) == 0:
        src_meta = _empty((1, int(src_meta.shape[1])), src_meta.dtype, like=x)

    combined_x = _empty(
        (alloc_combined_tokens, int(x.shape[1])), x.dtype, like=x
    )
    combined_topk_weights = None
    if topk_weights is not None:
        combined_topk_weights = _empty(
            (alloc_combined_tokens, num_topk), topk_weights.dtype, like=x
        )

    buf.runtime.internode_combine(
        _device_ptr(x),
        int(x.shape[0]),
        int(x.shape[1]),
        _dtype_code(x),
        _element_size(x),
        _device_ptr(topk_weights) if topk_weights is not None else 0,
        num_topk,
        _device_ptr(bias_0) if bias_0 is not None else 0,
        _device_ptr(bias_1) if bias_1 is not None else 0,
        _device_ptr(src_meta),
        num_combined_tokens,
        _device_ptr(is_combined_token_in_rank),
        _device_ptr(rdma_channel_prefix_matrix),
        _device_ptr(rdma_rank_prefix_sum),
        _device_ptr(gbl_channel_prefix_matrix),
        _device_ptr(send_rdma_head),
        _device_ptr(send_nvl_head),
        uccl_config,
        _device_ptr(combined_x),
        _device_ptr(combined_topk_weights)
        if combined_topk_weights is not None
        else 0,
        None,
        False,
        False,
        _compute_stream_ptr(buf, x),
    )

    combined_x = _slice_rows(combined_x, num_combined_tokens)
    if combined_topk_weights is not None:
        combined_topk_weights = _slice_rows(combined_topk_weights, num_combined_tokens)
    return combined_x, combined_topk_weights


def moe_combine(
    x,
    handle,
    topk_weights=None,
    bias=None,
    config: Optional[Config] = None,
    *,
    buffer: Optional[Buffer] = None,
):
    """Primus-Turbo-compatible MoE combine over UCCL-EP.

    Automatically selects intranode or internode based on
    ``num_rdma_ranks``. The ``handle`` must come from :func:`moe_dispatch`
    of the matching path.
    """
    buf = buffer or get_buffer()
    config = config or get_combine_config(buf.world_size)
    uccl_config = config.to_uccl_config()

    x = _ensure_jax_array(x)

    if buf.num_rdma_ranks > 1:
        return _internode_combine(buf, x, handle, topk_weights, bias, uccl_config)
    return _intranode_combine(buf, x, handle, topk_weights, bias, uccl_config)


def _unpack_bias(bias):
    if bias is None:
        return None, None
    if isinstance(bias, tuple):
        assert len(bias) == 2, "bias tuple must have length 2"
        return bias
    return bias, None
