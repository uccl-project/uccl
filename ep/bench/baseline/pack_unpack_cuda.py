"""
CUDA-accelerated pack/unpack functions for MoE data
This replaces the slow Python loop-based implementation
"""

import torch
from typing import List, Tuple
import sys
import os
from pathlib import Path

# Try to import the CUDA extension
try:
    # Add the tests directory to the path so we can find the .so file
    tests_dir = Path(__file__).parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    import moe_pack_unpack

    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"Warning: moe_pack_unpack CUDA extension not available: {e}")
    print(f"  Expected location: {Path(__file__).parent}")
    print(
        f"  Run 'cd tests && python setup_pack_unpack.py build_ext --inplace' to build it."
    )


def pack_moe_data_to_buffers_cuda(
    x: torch.Tensor,  # (num_tokens, hidden_dim)
    topk_idx: torch.Tensor,  # (num_tokens, experts_per_token)
    topk_weights: torch.Tensor,  # (num_tokens, experts_per_token)
    num_experts: int,
    world_size: int,
    device: torch.device,
    buffers: List[
        torch.Tensor
    ],  # world_size buffers (can be regular tensor or nvshmem tensor)
) -> torch.Tensor:
    """
    Pack MoE data into buffers - CUDA accelerated version

    Each item format: [token_data | local_expert_id | weight]

    Args:
        x: Token embeddings
        topk_idx: Expert assignments for each token
        topk_weights: Routing weights
        num_experts: Total number of experts
        world_size: Number of ranks
        device: Target device
        buffers: Pre-allocated buffers for each rank

    Returns:
        per_rank_bytes: (world_size,) bytes to send to each rank
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Please build it first.")

    # Call CUDA kernel
    per_rank_bytes = moe_pack_unpack.pack_moe_data(
        x, topk_idx, topk_weights, buffers, num_experts, world_size
    )

    return per_rank_bytes


def unpack_moe_data_from_buffers_cuda(
    buffers: List[torch.Tensor],  # world_size buffers
    per_rank_recv_bytes: torch.Tensor,  # (world_size,)
    num_local_experts: int,
    hidden_dim: int,
    world_size: int,
    device: torch.device,
    x_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Unpack MoE data from buffers - CUDA accelerated version

    Args:
        buffers: Received buffers from each rank
        per_rank_recv_bytes: Bytes received from each rank
        num_local_experts: Number of local experts
        hidden_dim: Hidden dimension size
        world_size: Number of ranks
        device: Target device
        x_dtype: Data type for token embeddings
        idx_dtype: Data type for expert indices
        weight_dtype: Data type for routing weights

    Returns:
        recv_x: (total_recv_tokens, hidden_dim) received token embeddings
        recv_topk_idx: (total_recv_tokens,) local expert IDs
        recv_topk_weights: (total_recv_tokens,) routing weights
        recv_num_tokens_per_expert: List[int] tokens per local expert
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Please build it first.")

    # Convert dtype to ScalarType enum
    dtype_map = {
        torch.float32: torch.float32,
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.float8_e4m3fn: torch.float8_e4m3fn,
        torch.float8_e5m2: torch.float8_e5m2,
        torch.int32: torch.int32,
        torch.int64: torch.int64,
    }

    x_scalar_type = dtype_map.get(x_dtype, torch.float32)
    idx_scalar_type = dtype_map.get(idx_dtype, torch.int64)
    weight_scalar_type = dtype_map.get(weight_dtype, torch.float32)

    # Call CUDA kernel
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert,
    ) = moe_pack_unpack.unpack_moe_data(
        buffers,
        per_rank_recv_bytes,
        num_local_experts,
        hidden_dim,
        world_size,
        x_scalar_type,
        idx_scalar_type,
        weight_scalar_type,
    )

    return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert


# CPU fallback version (original implementation) for comparison/testing
def pack_moe_data_to_buffers_cpu(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    world_size: int,
    device: torch.device,
    buffers: List[torch.Tensor],
) -> torch.Tensor:
    """Original CPU-based implementation (slow but works)"""
    num_tokens, hidden_dim = x.shape
    num_topk = topk_idx.shape[1]
    num_local_experts = num_experts // world_size

    bytes_per_token = hidden_dim * x.element_size()
    bytes_per_idx = topk_idx.element_size()
    bytes_per_weight = topk_weights.element_size()
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    # Count items needed per rank
    per_rank_item_counts = torch.zeros(world_size, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        for j in range(num_topk):
            expert_id = topk_idx[i, j].item()
            if expert_id == -1:
                continue
            target_rank = expert_id // num_local_experts
            per_rank_item_counts[target_rank] += 1

    per_rank_bytes = per_rank_item_counts * bytes_per_item

    # Collect items for each rank
    rank_items = [[] for _ in range(world_size)]
    for i in range(num_tokens):
        for j in range(num_topk):
            expert_id = topk_idx[i, j].item()
            if expert_id == -1:
                continue
            target_rank = expert_id // num_local_experts
            local_expert_id = expert_id % num_local_experts
            rank_items[target_rank].append((i, j, local_expert_id))

    # Pack into buffers
    for target_rank in range(world_size):
        offset = 0
        for token_id, topk_pos, local_expert_id in rank_items[target_rank]:
            buf = buffers[target_rank]

            # Pack token data
            token_bytes = x[token_id].view(torch.uint8)
            buf[offset : offset + bytes_per_token].copy_(token_bytes)
            offset += bytes_per_token

            # Pack local expert ID - use numpy to ensure correct byte representation
            idx_tensor = torch.tensor(
                [local_expert_id], dtype=topk_idx.dtype, device="cpu"
            )
            idx_bytes_np = idx_tensor.numpy().tobytes()
            idx_bytes_gpu = torch.frombuffer(idx_bytes_np, dtype=torch.uint8).to(device)
            buf[offset : offset + bytes_per_idx].copy_(idx_bytes_gpu)
            offset += bytes_per_idx

            # Pack weight
            weight_tensor = topk_weights[token_id, topk_pos : topk_pos + 1]
            weight_bytes = weight_tensor.view(torch.uint8)
            buf[offset : offset + bytes_per_weight].copy_(weight_bytes)
            offset += bytes_per_weight

    return per_rank_bytes


def unpack_moe_data_from_buffers_cpu(
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
    """Original CPU-based implementation (slow but works)"""
    bytes_per_token = hidden_dim * torch.tensor([], dtype=x_dtype).element_size()
    bytes_per_idx = torch.tensor([], dtype=idx_dtype).element_size()
    bytes_per_weight = torch.tensor([], dtype=weight_dtype).element_size()
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    total_recv_items = int(per_rank_recv_bytes.sum().item()) // bytes_per_item

    recv_x = torch.empty((total_recv_items, hidden_dim), dtype=x_dtype, device=device)
    recv_topk_idx = torch.empty((total_recv_items,), dtype=idx_dtype, device=device)
    recv_topk_weights = torch.empty(
        (total_recv_items,), dtype=weight_dtype, device=device
    )

    recv_item_id = 0
    for sender_rank in range(world_size):
        recv_bytes = int(per_rank_recv_bytes[sender_rank].item())
        if recv_bytes == 0:
            continue

        buf = buffers[sender_rank]
        offset = 0

        while offset < recv_bytes:
            # Unpack token data - copy to avoid alignment issues
            token_bytes = buf[offset : offset + bytes_per_token].clone()
            recv_x[recv_item_id] = token_bytes.view(x_dtype).view(hidden_dim)
            offset += bytes_per_token

            # Unpack expert ID - use numpy for unaligned data
            idx_bytes = buf[offset : offset + bytes_per_idx].cpu().numpy()
            if idx_dtype == torch.int64:
                idx_val = int.from_bytes(idx_bytes.tobytes(), byteorder="little")
            elif idx_dtype == torch.int32:
                idx_val = int.from_bytes(
                    idx_bytes.tobytes(), byteorder="little", signed=True
                )
            else:
                idx_val = torch.frombuffer(idx_bytes.tobytes(), dtype=idx_dtype)[
                    0
                ].item()
            recv_topk_idx[recv_item_id] = idx_val
            offset += bytes_per_idx

            # Unpack weight - use numpy for unaligned data
            weight_bytes = buf[offset : offset + bytes_per_weight].cpu().numpy()
            weight_val = torch.frombuffer(weight_bytes.tobytes(), dtype=weight_dtype)[
                0
            ].item()
            recv_topk_weights[recv_item_id] = weight_val
            offset += bytes_per_weight

            recv_item_id += 1

    # Count tokens per expert
    recv_num_tokens_per_expert = []
    for expert_id in range(num_local_experts):
        count = (recv_topk_idx == expert_id).sum().item()
        recv_num_tokens_per_expert.append(count)

    return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert
