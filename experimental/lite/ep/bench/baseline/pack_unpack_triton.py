"""
Triton-based MoE pack/unpack kernels.
Vendor-agnostic alternative to CUDA/HIP kernels for portability.

These kernels match the functionality in pack_unpack_kernels.cu but use
Triton for cross-platform GPU support (NVIDIA and AMD).

Implementation uses true Triton GPU kernels for pack/unpack operations,
with a unified buffer approach to work around Triton's lack of
pointer-to-pointer support.
"""

import os
import torch
import triton
import triton.language as tl
from typing import List, Tuple

# =============================================================================
# Triton Kernels
# =============================================================================


@triton.jit
def count_expert_tokens_kernel_triton(
    recv_topk_idx_ptr,
    expert_counts_ptr,
    total_items,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count tokens assigned to each expert.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < total_items:
            expert_id = tl.load(recv_topk_idx_ptr + idx)
            if expert_id >= 0:
                tl.atomic_add(expert_counts_ptr + expert_id, 1)


@triton.jit
def pack_moe_kernel_f32_triton(
    # Input tensors
    x_ptr,  # (num_tokens, hidden_dim) - float32
    topk_idx_ptr,  # (num_tokens * experts_per_token,) - int64
    topk_weights_ptr,  # (num_tokens * experts_per_token,) - float32
    # Output buffer as uint32
    unified_buffer_ptr,
    # Per-rank info
    rank_base_offsets_ptr,  # (world_size,) - base offset for each rank
    rank_counters_ptr,  # (world_size,) - atomic counters
    # Dimensions
    num_tokens,
    hidden_dim,
    experts_per_token,
    num_local_experts,
    # Sizes (in uint32 units)
    u32_per_token: tl.constexpr,
    u32_per_item: tl.constexpr,
):
    """Pack MoE data (float32 tokens) into unified buffer."""
    pid = tl.program_id(0)
    pair_idx = pid
    total_pairs = num_tokens * experts_per_token

    if pair_idx >= total_pairs:
        return

    token_id = pair_idx // experts_per_token

    # Load expert ID
    expert_id = tl.load(topk_idx_ptr + pair_idx)

    # Skip padding (-1)
    if expert_id < 0:
        return

    # Compute target rank and local expert ID
    target_rank = expert_id // num_local_experts
    local_expert_id = expert_id % num_local_experts

    # Atomically allocate position in target rank's buffer
    item_pos = tl.atomic_add(rank_counters_ptr + target_rank, 1)

    # Compute offset in unified buffer (in uint32 units)
    rank_base = tl.load(rank_base_offsets_ptr + target_rank)
    u32_offset = rank_base + item_pos * u32_per_item

    # Pack token data (float32 -> uint32, same size)
    x_base = token_id * hidden_dim
    for h in range(hidden_dim):
        val = tl.load(x_ptr + x_base + h)
        # Reinterpret float32 bits as uint32
        val_bits = val.to(tl.int32, bitcast=True).to(tl.uint32)
        tl.store(unified_buffer_ptr + u32_offset + h, val_bits)

    u32_offset += u32_per_token

    # Pack local expert ID (int64 = 2 uint32)
    local_expert_i64 = local_expert_id.to(tl.int64)
    u32_low = (local_expert_i64 & 0xFFFFFFFF).to(tl.uint32)
    u32_high = ((local_expert_i64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    tl.store(unified_buffer_ptr + u32_offset, u32_low)
    tl.store(unified_buffer_ptr + u32_offset + 1, u32_high)
    u32_offset += 2

    # Pack weight (float32 -> uint32)
    weight_val = tl.load(topk_weights_ptr + pair_idx)
    weight_bits = weight_val.to(tl.int32, bitcast=True).to(tl.uint32)
    tl.store(unified_buffer_ptr + u32_offset, weight_bits)


@triton.jit
def pack_moe_kernel_f16_triton(
    # Input tensors (viewed as uint16 for bit-preserving copy)
    x_ptr,  # (num_tokens, hidden_dim) - viewed as uint16
    topk_idx_ptr,  # int64
    topk_weights_ptr,  # float32
    # Output buffer as uint16
    unified_buffer_ptr,
    # Per-rank info
    rank_base_offsets_ptr,
    rank_counters_ptr,
    # Dimensions
    num_tokens,
    hidden_dim,
    experts_per_token,
    num_local_experts,
    # Sizes (in uint16 units)
    u16_per_token: tl.constexpr,
    u16_per_item: tl.constexpr,
):
    """Pack MoE data (float16/bfloat16 as uint16) into unified buffer."""
    pid = tl.program_id(0)
    pair_idx = pid
    total_pairs = num_tokens * experts_per_token

    if pair_idx >= total_pairs:
        return

    token_id = pair_idx // experts_per_token

    expert_id = tl.load(topk_idx_ptr + pair_idx)

    if expert_id < 0:
        return

    target_rank = expert_id // num_local_experts
    local_expert_id = expert_id % num_local_experts

    item_pos = tl.atomic_add(rank_counters_ptr + target_rank, 1)

    rank_base = tl.load(rank_base_offsets_ptr + target_rank)
    u16_offset = rank_base + item_pos * u16_per_item

    # Pack token data (direct uint16 copy)
    x_base = token_id * hidden_dim
    for h in range(hidden_dim):
        val = tl.load(x_ptr + x_base + h)
        tl.store(unified_buffer_ptr + u16_offset + h, val)

    u16_offset += u16_per_token

    # Pack local expert ID (int64 = 4 uint16)
    local_expert_i64 = local_expert_id.to(tl.int64)
    for i in range(4):
        u16_part = ((local_expert_i64 >> (i * 16)) & 0xFFFF).to(tl.uint16)
        tl.store(unified_buffer_ptr + u16_offset + i, u16_part)
    u16_offset += 4

    # Pack weight (float32 = 2 uint16)
    weight_val = tl.load(topk_weights_ptr + pair_idx)
    weight_bits = weight_val.to(tl.int32, bitcast=True)
    u16_low = (weight_bits & 0xFFFF).to(tl.uint16)
    u16_high = ((weight_bits >> 16) & 0xFFFF).to(tl.uint16)
    tl.store(unified_buffer_ptr + u16_offset, u16_low)
    tl.store(unified_buffer_ptr + u16_offset + 1, u16_high)


@triton.jit
def unpack_moe_kernel_f32_triton(
    buffer_ptr,  # uint32 buffer
    num_items,
    output_base,
    recv_x_ptr,  # float32 output
    recv_idx_ptr,  # int64 output
    recv_weights_ptr,  # float32 output
    hidden_dim,
    u32_per_token: tl.constexpr,
    u32_per_item: tl.constexpr,
):
    """Unpack items from buffer (float32 tokens)."""
    pid = tl.program_id(0)
    item_idx = pid

    if item_idx >= num_items:
        return

    u32_offset = item_idx * u32_per_item
    out_idx = output_base + item_idx

    # Unpack token data
    x_out_base = out_idx * hidden_dim
    for h in range(hidden_dim):
        val_bits = tl.load(buffer_ptr + u32_offset + h)
        val = val_bits.to(tl.int32).to(tl.float32, bitcast=True)
        tl.store(recv_x_ptr + x_out_base + h, val)

    u32_offset += u32_per_token

    # Unpack expert ID (2 uint32 -> int64)
    u32_low = tl.load(buffer_ptr + u32_offset)
    u32_high = tl.load(buffer_ptr + u32_offset + 1)
    idx_val = u32_low.to(tl.int64) | (u32_high.to(tl.int64) << 32)
    tl.store(recv_idx_ptr + out_idx, idx_val)
    u32_offset += 2

    # Unpack weight
    weight_bits = tl.load(buffer_ptr + u32_offset)
    weight_val = weight_bits.to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(recv_weights_ptr + out_idx, weight_val)


@triton.jit
def unpack_moe_kernel_f16_triton(
    buffer_ptr,  # uint16 buffer
    num_items,
    output_base,
    recv_x_ptr,  # float16/bfloat16 output (treated as uint16 for bit-preserving copy)
    recv_idx_ptr,  # int64 output
    recv_weights_ptr,  # float32 output
    hidden_dim,
    u16_per_token: tl.constexpr,
    u16_per_item: tl.constexpr,
):
    """Unpack items from buffer (float16/bfloat16 tokens)."""
    pid = tl.program_id(0)
    item_idx = pid

    if item_idx >= num_items:
        return

    u16_offset = item_idx * u16_per_item
    out_idx = output_base + item_idx

    # Unpack token data (bit-preserving copy via uint16)
    x_out_base = out_idx * hidden_dim
    for h in range(hidden_dim):
        val_bits = tl.load(buffer_ptr + u16_offset + h)
        # Store directly - the output pointer type handles the reinterpretation
        tl.store(recv_x_ptr + x_out_base + h, val_bits)

    u16_offset += u16_per_token

    # Unpack expert ID (4 uint16 -> int64)
    u16_0 = tl.load(buffer_ptr + u16_offset)
    u16_1 = tl.load(buffer_ptr + u16_offset + 1)
    u16_2 = tl.load(buffer_ptr + u16_offset + 2)
    u16_3 = tl.load(buffer_ptr + u16_offset + 3)
    idx_val = (
        u16_0.to(tl.int64)
        | (u16_1.to(tl.int64) << 16)
        | (u16_2.to(tl.int64) << 32)
        | (u16_3.to(tl.int64) << 48)
    )
    tl.store(recv_idx_ptr + out_idx, idx_val)
    u16_offset += 4

    # Unpack weight (2 uint16 -> float32)
    u16_low = tl.load(buffer_ptr + u16_offset)
    u16_high = tl.load(buffer_ptr + u16_offset + 1)
    weight_bits = u16_low.to(tl.int32) | (u16_high.to(tl.int32) << 16)
    weight_val = weight_bits.to(tl.float32, bitcast=True)
    tl.store(recv_weights_ptr + out_idx, weight_val)


# =============================================================================
# Python Wrapper Functions
# =============================================================================


def count_expert_tokens_triton(
    recv_topk_idx: torch.Tensor,
    num_local_experts: int,
) -> torch.Tensor:
    """Count tokens assigned to each expert using Triton kernel."""
    total_items = recv_topk_idx.numel()
    device = recv_topk_idx.device

    expert_counts = torch.zeros(num_local_experts, dtype=torch.int32, device=device)

    if total_items == 0:
        return expert_counts

    BLOCK_SIZE = 256
    num_programs = (total_items + BLOCK_SIZE - 1) // BLOCK_SIZE

    count_expert_tokens_kernel_triton[(num_programs,)](
        recv_topk_idx,
        expert_counts,
        total_items,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_counts


def pack_moe_data_to_buffers_triton(
    x: torch.Tensor,  # (num_tokens, hidden_dim)
    topk_idx: torch.Tensor,  # (num_tokens, experts_per_token)
    topk_weights: torch.Tensor,  # (num_tokens, experts_per_token)
    num_experts: int,
    world_size: int,
    device: torch.device,
    buffers: List[torch.Tensor],
) -> torch.Tensor:
    """Pack MoE data into buffers using Triton kernels."""
    num_tokens, hidden_dim = x.shape
    experts_per_token = topk_idx.shape[1]
    num_local_experts = num_experts // world_size

    x_elem_size = x.element_size()
    bytes_per_token = hidden_dim * x_elem_size
    bytes_per_idx = topk_idx.element_size()  # 8 for int64
    bytes_per_weight = topk_weights.element_size()  # 4 for float32
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    max_bytes_per_rank = buffers[0].numel()

    # Flatten inputs
    topk_idx_flat = topk_idx.view(-1).contiguous()
    topk_weights_flat = topk_weights.view(-1).contiguous()

    total_pairs = num_tokens * experts_per_token
    rank_counters = torch.zeros(world_size, dtype=torch.int32, device=device)

    if x_elem_size == 4:  # float32
        # Use uint32 buffer
        max_u32_per_rank = max_bytes_per_rank // 4
        u32_per_token = hidden_dim
        u32_per_item = u32_per_token + 2 + 1  # token + idx(2) + weight(1)

        unified_buffer = torch.zeros(
            world_size * max_u32_per_rank, dtype=torch.uint32, device=device
        )
        rank_base_offsets = torch.arange(
            0,
            world_size * max_u32_per_rank,
            max_u32_per_rank,
            dtype=torch.int64,
            device=device,
        )

        pack_moe_kernel_f32_triton[(total_pairs,)](
            x,
            topk_idx_flat,
            topk_weights_flat,
            unified_buffer,
            rank_base_offsets,
            rank_counters,
            num_tokens,
            hidden_dim,
            experts_per_token,
            num_local_experts,
            u32_per_token=u32_per_token,
            u32_per_item=u32_per_item,
        )

        # Copy back to individual buffers
        unified_bytes = unified_buffer.view(torch.uint8)
        for rank in range(world_size):
            start = rank * max_bytes_per_rank
            buffers[rank].copy_(unified_bytes[start : start + max_bytes_per_rank])

    else:  # float16/bfloat16 (2 bytes)
        # Use uint16 buffer
        max_u16_per_rank = max_bytes_per_rank // 2
        u16_per_token = hidden_dim
        u16_per_item = u16_per_token + 4 + 2  # token + idx(4 u16) + weight(2 u16)

        unified_buffer = torch.zeros(
            world_size * max_u16_per_rank, dtype=torch.uint16, device=device
        )
        rank_base_offsets = torch.arange(
            0,
            world_size * max_u16_per_rank,
            max_u16_per_rank,
            dtype=torch.int64,
            device=device,
        )

        # View input as uint16 for bit-preserving copy
        x_u16 = x.view(torch.uint16)

        pack_moe_kernel_f16_triton[(total_pairs,)](
            x_u16,
            topk_idx_flat,
            topk_weights_flat,
            unified_buffer,
            rank_base_offsets,
            rank_counters,
            num_tokens,
            hidden_dim,
            experts_per_token,
            num_local_experts,
            u16_per_token=u16_per_token,
            u16_per_item=u16_per_item,
        )

        # Copy back to individual buffers
        unified_bytes = unified_buffer.view(torch.uint8)
        for rank in range(world_size):
            start = rank * max_bytes_per_rank
            buffers[rank].copy_(unified_bytes[start : start + max_bytes_per_rank])

    per_rank_bytes = rank_counters.to(torch.int32) * bytes_per_item
    return per_rank_bytes


def unpack_moe_data_from_buffers_triton(
    buffers: List[torch.Tensor],
    per_rank_recv_bytes: torch.Tensor,
    num_local_experts: int,
    hidden_dim: int,
    world_size: int,
    device: torch.device,
    x_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Unpack MoE data from buffers using Triton kernels."""
    x_elem_size = torch.tensor([], dtype=x_dtype).element_size()
    bytes_per_token = hidden_dim * x_elem_size
    bytes_per_idx = 8  # int64
    bytes_per_weight = 4  # float32
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    per_rank_items = per_rank_recv_bytes // bytes_per_item
    item_offsets = torch.zeros(world_size, dtype=torch.int32, device=device)
    if world_size > 1:
        item_offsets[1:] = torch.cumsum(per_rank_items[:-1], dim=0)

    total_items = per_rank_items.sum().item()
    if total_items == 0:
        total_items = 1

    recv_x = torch.empty((total_items, hidden_dim), dtype=x_dtype, device=device)
    recv_topk_idx = torch.empty(total_items, dtype=idx_dtype, device=device)
    recv_topk_weights = torch.empty(total_items, dtype=weight_dtype, device=device)

    per_rank_items_cpu = per_rank_items.cpu()
    item_offsets_cpu = item_offsets.cpu()

    for rank in range(world_size):
        num_items = per_rank_items_cpu[rank].item()
        if num_items == 0:
            continue

        output_base = item_offsets_cpu[rank].item()

        if x_elem_size == 4:  # float32
            u32_per_token = hidden_dim
            u32_per_item = u32_per_token + 2 + 1
            buffer_u32 = buffers[rank].view(torch.uint32)

            unpack_moe_kernel_f32_triton[(num_items,)](
                buffer_u32,
                num_items,
                output_base,
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                hidden_dim,
                u32_per_token=u32_per_token,
                u32_per_item=u32_per_item,
            )
        else:  # float16/bfloat16
            u16_per_token = hidden_dim
            u16_per_item = u16_per_token + 4 + 2
            buffer_u16 = buffers[rank].view(torch.uint16)
            # View output as uint16 for bit-preserving copy
            recv_x_u16 = recv_x.view(torch.uint16)

            unpack_moe_kernel_f16_triton[(num_items,)](
                buffer_u16,
                num_items,
                output_base,
                recv_x_u16,
                recv_topk_idx,
                recv_topk_weights,
                hidden_dim,
                u16_per_token=u16_per_token,
                u16_per_item=u16_per_item,
            )

    expert_counts = count_expert_tokens_triton(recv_topk_idx, num_local_experts)
    recv_num_tokens_per_expert = expert_counts.tolist()

    return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert


# =============================================================================
# Backend Selection
# =============================================================================

BACKEND = os.getenv("UCCL_PACK_UNPACK_BACKEND", "cuda")


def get_backend() -> str:
    return BACKEND


def set_backend(backend: str):
    global BACKEND
    if backend not in ("cuda", "triton", "cpu"):
        raise ValueError(f"Unknown backend: {backend}")
    BACKEND = backend


def pack_moe_data_to_buffers(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    world_size: int,
    device: torch.device,
    buffers: List[torch.Tensor],
    backend: str = None,
) -> torch.Tensor:
    backend = backend or BACKEND
    if backend == "triton":
        return pack_moe_data_to_buffers_triton(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )
    elif backend == "cuda":
        from pack_unpack_cuda import pack_moe_data_to_buffers_cuda

        return pack_moe_data_to_buffers_cuda(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )
    else:
        from pack_unpack_cuda import pack_moe_data_to_buffers_cpu

        return pack_moe_data_to_buffers_cpu(
            x, topk_idx, topk_weights, num_experts, world_size, device, buffers
        )


def unpack_moe_data_from_buffers(
    buffers: List[torch.Tensor],
    per_rank_recv_bytes: torch.Tensor,
    num_local_experts: int,
    hidden_dim: int,
    world_size: int,
    device: torch.device,
    x_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    backend: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    backend = backend or BACKEND
    if backend == "triton":
        return unpack_moe_data_from_buffers_triton(
            buffers,
            per_rank_recv_bytes,
            num_local_experts,
            hidden_dim,
            world_size,
            device,
            x_dtype,
            idx_dtype,
            weight_dtype,
        )
    elif backend == "cuda":
        from pack_unpack_cuda import unpack_moe_data_from_buffers_cuda

        return unpack_moe_data_from_buffers_cuda(
            buffers,
            per_rank_recv_bytes,
            num_local_experts,
            hidden_dim,
            world_size,
            device,
            x_dtype,
            idx_dtype,
            weight_dtype,
        )
    else:
        from pack_unpack_cuda import unpack_moe_data_from_buffers_cpu

        return unpack_moe_data_from_buffers_cpu(
            buffers,
            per_rank_recv_bytes,
            num_local_experts,
            hidden_dim,
            world_size,
            device,
            x_dtype,
            idx_dtype,
            weight_dtype,
        )
