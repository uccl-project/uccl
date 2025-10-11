# ruff: noqa: T201
"""
Unified MoE Benchmark: Compare three communication methods using standardized MoE-style I/O.

Input: x (tokens), topk_idx (routing), topk_weights
Output: recv_x, recv_topk_idx, recv_topk_weights, recv_expert_counts

Communication methods:
1. PyTorch Sparse All-to-All (all_to_all_single)
2. NVSHMEM PUT-based
3. Dense All-to-All (baseline)

Feature: Separate timing measurement for communication and unpack operations.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import nvshmem.core as nvshmem
import torch
from cuda.core.experimental import Device
from nvshmem.core import Teams

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from all_to_all_utils import (
    MoEConfig,
    RankTestData,
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    PyTorchStreamWrapper,
    nvshmem_init,
)

# Try to import CUDA kernel version
try:
    from pack_unpack_cuda import (
        pack_moe_data_to_buffers_cuda,
        unpack_moe_data_from_buffers_cuda,
        CUDA_AVAILABLE,
    )
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"Warning: CUDA pack/unpack kernels not available: {e}")

logger = logging.getLogger(__name__)


# ============================================================================
# MoE Data Packing/Unpacking Functions
# ============================================================================

def pack_moe_data_to_buffers(
    x: torch.Tensor,  # (num_tokens, hidden_dim)
    topk_idx: torch.Tensor,  # (num_tokens, experts_per_token)
    topk_weights: torch.Tensor,  # (num_tokens, experts_per_token)
    num_experts: int,
    world_size: int,
    device: torch.device,
    buffers: list[torch.Tensor],  # world_size buffers (regular or nvshmem tensors)
) -> torch.Tensor:
    """
    Pack MoE data into buffers for all-to-all communication.

    Each item format: [token_data | local_expert_id | weight]

    Args:
        x: Token embeddings
        topk_idx: Expert assignments for each token
        topk_weights: Routing weights
        num_experts: Total number of experts across all ranks
        world_size: Number of processes
        device: Target device
        buffers: Pre-allocated buffers for each destination rank

    Returns:
        per_rank_bytes: (world_size,) bytes to send to each rank
    """
    num_tokens, hidden_dim = x.shape
    num_topk = topk_idx.shape[1]
    num_local_experts = num_experts // world_size

    bytes_per_token = hidden_dim * x.element_size()
    bytes_per_idx = topk_idx.element_size()
    bytes_per_weight = topk_weights.element_size()
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    # Count items to send to each rank
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

    # Pack data into buffers
    for target_rank in range(world_size):
        offset = 0
        for token_id, topk_pos, local_expert_id in rank_items[target_rank]:
            buf = buffers[target_rank]

            # Pack token data
            token_bytes = x[token_id].view(torch.uint8)
            buf[offset:offset + bytes_per_token].copy_(token_bytes)
            offset += bytes_per_token

            # Pack local expert ID
            idx_tensor = torch.tensor([local_expert_id], dtype=topk_idx.dtype, device=device)
            idx_bytes = idx_tensor.view(torch.uint8)
            buf[offset:offset + bytes_per_idx].copy_(idx_bytes)
            offset += bytes_per_idx

            # Pack weight
            weight_tensor = topk_weights[token_id, topk_pos:topk_pos+1]
            weight_bytes = weight_tensor.view(torch.uint8)
            buf[offset:offset + bytes_per_weight].copy_(weight_bytes)
            offset += bytes_per_weight

    return per_rank_bytes


def unpack_moe_data_from_buffers(
    buffers: list[torch.Tensor],  # world_size buffers
    per_rank_recv_bytes: torch.Tensor,  # (world_size,)
    num_local_experts: int,
    hidden_dim: int,
    world_size: int,
    device: torch.device,
    x_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Unpack MoE data from received buffers.

    Args:
        buffers: Received buffers from each rank
        per_rank_recv_bytes: Number of bytes received from each rank
        num_local_experts: Number of experts on this rank
        hidden_dim: Hidden dimension size
        world_size: Number of processes
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
    bytes_per_token = hidden_dim * torch.tensor([], dtype=x_dtype).element_size()
    bytes_per_idx = torch.tensor([], dtype=idx_dtype).element_size()
    bytes_per_weight = torch.tensor([], dtype=weight_dtype).element_size()
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    total_recv_items = int(per_rank_recv_bytes.sum().item()) // bytes_per_item

    recv_x = torch.empty((total_recv_items, hidden_dim), dtype=x_dtype, device=device)
    recv_topk_idx = torch.empty((total_recv_items,), dtype=idx_dtype, device=device)
    recv_topk_weights = torch.empty((total_recv_items,), dtype=weight_dtype, device=device)

    recv_item_id = 0
    for sender_rank in range(world_size):
        recv_bytes = int(per_rank_recv_bytes[sender_rank].item())
        if recv_bytes == 0:
            continue

        buf = buffers[sender_rank]
        offset = 0

        while offset < recv_bytes:
            # Unpack token data
            token_bytes = buf[offset:offset + bytes_per_token]
            recv_x[recv_item_id] = token_bytes.view(x_dtype).view(hidden_dim)
            offset += bytes_per_token

            # Unpack expert ID
            idx_bytes = buf[offset:offset + bytes_per_idx]
            recv_topk_idx[recv_item_id] = idx_bytes.view(idx_dtype)[0]
            offset += bytes_per_idx

            # Unpack weight
            weight_bytes = buf[offset:offset + bytes_per_weight]
            recv_topk_weights[recv_item_id] = weight_bytes.view(weight_dtype)[0]
            offset += bytes_per_weight

            recv_item_id += 1

    # Count tokens per expert (similar to ep.dispatch output)
    recv_num_tokens_per_expert = []
    for expert_id in range(num_local_experts):
        count = (recv_topk_idx == expert_id).sum().item()
        recv_num_tokens_per_expert.append(count)

    return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert


# ============================================================================
# Main Benchmark Function
# ============================================================================

@torch.inference_mode()
def bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
) -> tuple[tuple[int, int], torch.Tensor] | tuple[None, None]:
    device = pgi.device
    num_dp = pgi.world_size // dp_size
    dp_rank = pgi.rank // dp_size

    # Check if experts can be evenly distributed
    if moe.num_experts % pgi.world_size != 0:
        if pgi.rank == 0:
            print(f"⚠️  Skipping: {moe.num_experts} experts cannot be evenly divided by {pgi.world_size} ranks")
        return None, None

    rng = torch.Generator()
    rng.manual_seed(dp_rank + 1)
    rank_data = RankTestData(moe, rng, use_max_tokens=True)

    num_local_experts = moe.num_experts // pgi.world_size

    # ========== MoE Input Data (unified input format) ==========
    num_tokens = rank_data.num_tokens
    x = torch.randn((num_tokens, moe.hidden_dim), dtype=moe.in_dtype, device=device)
    topk_idx = rank_data.indices.to(device)  # (num_tokens, experts_per_token) - ensure on GPU
    topk_weights = torch.randn((num_tokens, moe.experts_per_token), dtype=torch.float32, device=device)

    if pgi.rank == 0:
        print(f"\n[MoE Input Data]")
        print(f"  Tokens: {num_tokens}, Hidden: {moe.hidden_dim}, Experts/token: {moe.experts_per_token}")
        print(f"  Example: Token 0 → Experts {topk_idx[0].tolist()}")

    # ========== Prepare buffers for all three communication methods ==========

    # 1. PyTorch Sparse: Calculate bytes needed per rank
    bytes_per_token = moe.hidden_dim * moe.in_dtype.itemsize
    bytes_per_idx = topk_idx.element_size()
    bytes_per_weight = topk_weights.element_size()
    bytes_per_item = bytes_per_token + bytes_per_idx + bytes_per_weight

    num_local_experts_calc = moe.num_experts // pgi.world_size
    per_rank_item_counts = torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        for j in range(moe.experts_per_token):
            expert_id = topk_idx[i, j].item()
            if expert_id == -1:
                continue
            target_rank = expert_id // num_local_experts_calc
            per_rank_item_counts[target_rank] += 1

    per_rank_send_bytes = per_rank_item_counts * bytes_per_item

    # Create correctly-sized torch buffers
    torch_send_bufs = []
    for i in range(pgi.world_size):
        size = int(per_rank_send_bytes[i].item())
        torch_send_bufs.append(torch.empty(max(size, 1), dtype=torch.uint8, device=device))

    # Measure CPU pack time
    torch_stream = torch.cuda.current_stream()
    cpu_pack_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
    for start_ev, end_ev in cpu_pack_events:
        start_ev.record(torch_stream)
        pack_moe_data_to_buffers(
            x, topk_idx, topk_weights, moe.num_experts, pgi.world_size, device, torch_send_bufs
        )
        end_ev.record(torch_stream)
    torch_stream.synchronize()
    cpu_pack_time_us = sum(s.elapsed_time(e) * 1e3 for s, e in cpu_pack_events) / len(cpu_pack_events)

    # Merge into single sendbuf for all_to_all_single
    torch_sparse_sendbuf = torch.cat([torch_send_bufs[i][:int(per_rank_send_bytes[i].item())]
                                       for i in range(pgi.world_size)])
    sparse_in_splits = per_rank_send_bytes.tolist()

    per_rank_recv_bytes = torch.zeros_like(per_rank_send_bytes)
    torch.distributed.all_to_all_single(per_rank_recv_bytes, per_rank_send_bytes)
    sparse_out_splits = per_rank_recv_bytes.tolist()

    total_sparse_bytes = int(per_rank_send_bytes.sum().item())
    torch_sparse_recvbuf = torch.empty(max(int(per_rank_recv_bytes.sum().item()), 1),
                                        dtype=torch.uint8, device=device)

    # 2. NVSHMEM: Symmetric memory buffers
    per_rank_send_bytes_all = [torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
                                for _ in range(pgi.world_size)]
    per_rank_recv_bytes_all = [torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
                                for _ in range(pgi.world_size)]
    torch.distributed.all_gather(per_rank_send_bytes_all, per_rank_send_bytes)
    torch.distributed.all_gather(per_rank_recv_bytes_all, per_rank_recv_bytes)

    nvshmem_send_bufs = []
    for target_rank in range(pgi.world_size):
        max_send_size = max(
            int(per_rank_send_bytes_all[sender][target_rank].item())
            for sender in range(pgi.world_size)
        )
        send_buf = nvshmem.tensor((max(max_send_size, 1),), dtype=torch.uint8)
        nvshmem_send_bufs.append(send_buf)

    nvshmem_recv_bufs = []
    for sender_rank in range(pgi.world_size):
        max_recv_size = max(
            int(per_rank_recv_bytes_all[receiver][sender_rank].item())
            for receiver in range(pgi.world_size)
        )
        recv_buf = nvshmem.tensor((max(max_recv_size, 1),), dtype=torch.uint8)
        nvshmem_recv_bufs.append(recv_buf)

    # Pack data into NVSHMEM buffers
    pack_moe_data_to_buffers(
        x, topk_idx, topk_weights, moe.num_experts, pgi.world_size, device, nvshmem_send_bufs
    )

    # 3. Dense baseline
    effective_tokens_per_expert = max(1, (total_sparse_bytes // pgi.world_size // num_local_experts) // bytes_per_item)
    a2a_shape = (pgi.world_size, num_local_experts, effective_tokens_per_expert * bytes_per_item)
    dense_a2a_tensor = torch.randint(0, 256, a2a_shape, dtype=torch.uint8, device=device)
    dense_a2a_out_tensor = torch.empty_like(dense_a2a_tensor)
    dense_a2a_bytes = dense_a2a_tensor.numel()

    if pgi.rank == 0:
        print(f"  Sparse bytes: {total_sparse_bytes:,} ({total_sparse_bytes/1e6:.2f} MB)")
        print(f"  Dense bytes:  {dense_a2a_bytes:,} ({dense_a2a_bytes/1e6:.2f} MB)")
        print(f"  Reduction: {dense_a2a_bytes/total_sparse_bytes:.2f}x\n")

    cuda_dev = Device(device.index)
    cuda_stream = cuda_dev.create_stream()

    # 4. CUDA kernel + torch.distributed (pre-pack to avoid repeated packing in loop)
    if CUDA_AVAILABLE:
        # Method 3: Pre-create buffers and pack
        cuda_torch_send_bufs = []
        for i in range(pgi.world_size):
            size = int(per_rank_send_bytes[i].item())
            cuda_torch_send_bufs.append(torch.empty(max(size, 1), dtype=torch.uint8, device=device))

        # Measure CUDA pack time
        cuda_pack_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
        for start_ev, end_ev in cuda_pack_events:
            start_ev.record(torch_stream)
            pack_moe_data_to_buffers_cuda(
                x, topk_idx, topk_weights, moe.num_experts, pgi.world_size, device, cuda_torch_send_bufs
            )
            end_ev.record(torch_stream)
        torch_stream.synchronize()
        cuda_pack_time_us = sum(s.elapsed_time(e) * 1e3 for s, e in cuda_pack_events) / len(cuda_pack_events)

        cuda_torch_sendbuf = torch.cat([cuda_torch_send_bufs[i][:int(per_rank_send_bytes[i].item())]
                                          for i in range(pgi.world_size)])
        cuda_torch_recvbuf = torch.empty(max(int(per_rank_recv_bytes.sum().item()), 1),
                                           dtype=torch.uint8, device=device)

        # Method 4: NVSHMEM buffers
        cuda_nvshmem_send_bufs = []
        for target_rank in range(pgi.world_size):
            max_send_size = max(
                int(per_rank_send_bytes_all[sender][target_rank].item())
                for sender in range(pgi.world_size)
            )
            send_buf = nvshmem.tensor((max(max_send_size, 1),), dtype=torch.uint8)
            cuda_nvshmem_send_bufs.append(send_buf)

        cuda_nvshmem_recv_bufs = []
        for sender_rank in range(pgi.world_size):
            max_recv_size = max(
                int(per_rank_recv_bytes_all[receiver][sender_rank].item())
                for receiver in range(pgi.world_size)
            )
            recv_buf = nvshmem.tensor((max(max_recv_size, 1),), dtype=torch.uint8)
            cuda_nvshmem_recv_bufs.append(recv_buf)

        # Pre-pack data into NVSHMEM buffers (only once)
        pack_moe_data_to_buffers_cuda(
            x, topk_idx, topk_weights, moe.num_experts, pgi.world_size, device, cuda_nvshmem_send_bufs
        )

    # ========== Benchmark ==========
    def run() -> tuple[float, ...]:
        num_samples = 10
        # Event timeline:
        # e0: start
        # e1: Method 1 communication complete
        # e2: Method 1 (CPU + torch.dist) complete
        # e3: Method 2 communication complete
        # e4: Method 2 (CPU + NVSHMEM) complete
        # e5: Dense complete
        # e6: Method 3 communication complete (if CUDA)
        # e7: Method 3 (CUDA + torch.dist) complete (if CUDA)
        # e8: Method 4 communication complete (if CUDA)
        # e9: Method 4 (CUDA + NVSHMEM) complete (if CUDA)
        num_events = 10 if CUDA_AVAILABLE else 6
        events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
            for _ in range(num_samples)
        ]

        torch_stream = torch.cuda.current_stream()
        torch_stream_wrapped = PyTorchStreamWrapper(torch_stream)
        team = Teams.TEAM_WORLD

        for event_list in events:
            if CUDA_AVAILABLE:
                e0, e1, e2, e3, e4, e5, e6, e7, e8, e9 = event_list
            else:
                e0, e1, e2, e3, e4, e5 = event_list
            nvshmem.collective.barrier(team, torch_stream_wrapped)

            e0.record(torch_stream)

            # ========== Method 1: CPU pack/unpack + torch.distributed ==========
            torch.distributed.all_to_all_single(
                output=torch_sparse_recvbuf,
                input=torch_sparse_sendbuf,
                output_split_sizes=sparse_out_splits,
                input_split_sizes=sparse_in_splits,
            )
            e1.record(torch_stream)  # Communication complete

            torch_recv_bufs = []
            offset = 0
            for i in range(pgi.world_size):
                size = sparse_out_splits[i]
                torch_recv_bufs.append(torch_sparse_recvbuf[offset:offset+size])
                offset += size

            _, _, _, _ = unpack_moe_data_from_buffers(
                buffers=torch_recv_bufs,
                per_rank_recv_bytes=per_rank_recv_bytes,
                num_local_experts=num_local_experts,
                hidden_dim=moe.hidden_dim,
                world_size=pgi.world_size,
                device=device,
                x_dtype=moe.in_dtype,
                idx_dtype=topk_idx.dtype,
                weight_dtype=topk_weights.dtype,
            )
            e2.record(torch_stream)  # Method 1 complete

            # ========== Method 2: CPU pack/unpack + NVSHMEM ==========
            for target_rank in range(pgi.world_size):
                if target_rank == pgi.rank:
                    continue
                send_bytes = int(per_rank_send_bytes[target_rank].item())
                if send_bytes > 0:
                    nvshmem.put(
                        dst=nvshmem_recv_bufs[pgi.rank],
                        src=nvshmem_send_bufs[target_rank],
                        remote_pe=target_rank,
                        stream=cuda_stream
                    )

            cuda_stream.sync()
            nvshmem.collective.barrier(team, torch_stream_wrapped)
            e3.record(torch_stream)  # Communication complete

            _, _, _, _ = unpack_moe_data_from_buffers(
                buffers=nvshmem_recv_bufs,
                per_rank_recv_bytes=per_rank_recv_bytes,
                num_local_experts=num_local_experts,
                hidden_dim=moe.hidden_dim,
                world_size=pgi.world_size,
                device=device,
                x_dtype=moe.in_dtype,
                idx_dtype=topk_idx.dtype,
                weight_dtype=topk_weights.dtype,
            )
            e4.record(torch_stream)  # Method 2 complete

            # ========== Dense Baseline ==========
            torch.distributed.all_to_all_single(dense_a2a_out_tensor, dense_a2a_tensor)
            _ = dense_a2a_out_tensor.sum()
            e5.record(torch_stream)  # Dense complete

            if CUDA_AVAILABLE:
                # ========== Method 3: CUDA kernel pack/unpack + torch.distributed ==========
                # Pack already done outside loop, only communication + unpack here
                torch.distributed.all_to_all_single(
                    output=cuda_torch_recvbuf,
                    input=cuda_torch_sendbuf,
                    output_split_sizes=sparse_out_splits,
                    input_split_sizes=sparse_in_splits,
                )
                e6.record(torch_stream)  # Communication complete

                cuda_torch_recv_bufs = []
                offset = 0
                for i in range(pgi.world_size):
                    size = sparse_out_splits[i]
                    cuda_torch_recv_bufs.append(cuda_torch_recvbuf[offset:offset+size])
                    offset += size

                _, _, _, _ = unpack_moe_data_from_buffers_cuda(
                    buffers=cuda_torch_recv_bufs,
                    per_rank_recv_bytes=per_rank_recv_bytes,
                    num_local_experts=num_local_experts,
                    hidden_dim=moe.hidden_dim,
                    world_size=pgi.world_size,
                    device=device,
                    x_dtype=moe.in_dtype,
                    idx_dtype=topk_idx.dtype,
                    weight_dtype=topk_weights.dtype,
                )
                e7.record(torch_stream)  # Method 3 complete

                # ========== Method 4: CUDA kernel pack/unpack + NVSHMEM ==========
                # Pack already done outside loop, only communication + unpack here
                for target_rank in range(pgi.world_size):
                    if target_rank == pgi.rank:
                        continue
                    send_bytes = int(per_rank_send_bytes[target_rank].item())
                    if send_bytes > 0:
                        nvshmem.put(
                            dst=cuda_nvshmem_recv_bufs[pgi.rank],
                            src=cuda_nvshmem_send_bufs[target_rank],
                            remote_pe=target_rank,
                            stream=cuda_stream
                        )

                cuda_stream.sync()
                nvshmem.collective.barrier(team, torch_stream_wrapped)
                e8.record(torch_stream)  # Communication complete

                _, _, _, _ = unpack_moe_data_from_buffers_cuda(
                    buffers=cuda_nvshmem_recv_bufs,
                    per_rank_recv_bytes=per_rank_recv_bytes,
                    num_local_experts=num_local_experts,
                    hidden_dim=moe.hidden_dim,
                    world_size=pgi.world_size,
                    device=device,
                    x_dtype=moe.in_dtype,
                    idx_dtype=topk_idx.dtype,
                    weight_dtype=topk_weights.dtype,
                )
                e9.record(torch_stream)  # Method 4 complete

        torch_stream.synchronize()

        # Method 1: CPU + torch.dist (e0->e1: comm, e1->e2: total)
        sum_cpu_torch_comm_us = sum(event_list[0].elapsed_time(event_list[1]) * 1e3 for event_list in events)
        sum_cpu_torch_us = sum(event_list[0].elapsed_time(event_list[2]) * 1e3 for event_list in events)
        cpu_torch_comm_us = sum_cpu_torch_comm_us / num_samples
        cpu_torch_us = sum_cpu_torch_us / num_samples

        # Method 2: CPU + NVSHMEM (e2->e3: comm, e2->e4: total)
        sum_cpu_nvshmem_comm_us = sum(event_list[2].elapsed_time(event_list[3]) * 1e3 for event_list in events)
        sum_cpu_nvshmem_us = sum(event_list[2].elapsed_time(event_list[4]) * 1e3 for event_list in events)
        cpu_nvshmem_comm_us = sum_cpu_nvshmem_comm_us / num_samples
        cpu_nvshmem_us = sum_cpu_nvshmem_us / num_samples

        # Dense baseline (e4->e5)
        sum_dense_us = sum(event_list[4].elapsed_time(event_list[5]) * 1e3 for event_list in events)
        dense_us = sum_dense_us / num_samples

        # Calculate bandwidth
        cpu_torch_gbps = total_sparse_bytes / cpu_torch_us / 1e3
        cpu_nvshmem_gbps = total_sparse_bytes / cpu_nvshmem_us / 1e3
        dense_gbps = dense_a2a_bytes / dense_us / 1e3

        if CUDA_AVAILABLE:
            # Method 3: CUDA + torch.dist (e5->e6: comm, e5->e7: total)
            sum_cuda_torch_comm_us = sum(event_list[5].elapsed_time(event_list[6]) * 1e3 for event_list in events)
            sum_cuda_torch_us = sum(event_list[5].elapsed_time(event_list[7]) * 1e3 for event_list in events)
            cuda_torch_comm_us = sum_cuda_torch_comm_us / num_samples
            cuda_torch_us = sum_cuda_torch_us / num_samples

            # Method 4: CUDA + NVSHMEM (e7->e8: comm, e7->e9: total)
            sum_cuda_nvshmem_comm_us = sum(event_list[7].elapsed_time(event_list[8]) * 1e3 for event_list in events)
            sum_cuda_nvshmem_us = sum(event_list[7].elapsed_time(event_list[9]) * 1e3 for event_list in events)
            cuda_nvshmem_comm_us = sum_cuda_nvshmem_comm_us / num_samples
            cuda_nvshmem_us = sum_cuda_nvshmem_us / num_samples

            # Calculate bandwidth
            cuda_torch_gbps = total_sparse_bytes / cuda_torch_us / 1e3
            cuda_nvshmem_gbps = total_sparse_bytes / cuda_nvshmem_us / 1e3

            return (
                cpu_torch_us, cpu_torch_gbps, cpu_torch_comm_us, cpu_pack_time_us,         # Method 1: CPU + torch.dist
                cpu_nvshmem_us, cpu_nvshmem_gbps, cpu_nvshmem_comm_us, cpu_pack_time_us,   # Method 2: CPU + NVSHMEM
                cuda_torch_us, cuda_torch_gbps, cuda_torch_comm_us, cuda_pack_time_us,      # Method 3: CUDA + torch.dist
                cuda_nvshmem_us, cuda_nvshmem_gbps, cuda_nvshmem_comm_us, cuda_pack_time_us,# Method 4: CUDA + NVSHMEM
                dense_us, dense_gbps,                                                        # Dense baseline
            )
        else:
            return (
                cpu_torch_us, cpu_torch_gbps, cpu_torch_comm_us, cpu_pack_time_us,
                cpu_nvshmem_us, cpu_nvshmem_gbps, cpu_nvshmem_comm_us, cpu_pack_time_us,
                dense_us, dense_gbps,
            )

    # Warmup
    for _ in range(10):
        run()

    # Benchmark
    torch.distributed.barrier()
    result = torch.tensor([run() for _ in range(20)])

    # Print MoE output example
    if pgi.rank == 0:
        # Use output from last run
        torch_recv_bufs = []
        offset = 0
        for i in range(pgi.world_size):
            size = sparse_out_splits[i]
            torch_recv_bufs.append(torch_sparse_recvbuf[offset:offset+size])
            offset += size

        recv_x, _, _, recv_counts = unpack_moe_data_from_buffers(
            buffers=torch_recv_bufs,
            per_rank_recv_bytes=per_rank_recv_bytes,
            num_local_experts=num_local_experts,
            hidden_dim=moe.hidden_dim,
            world_size=pgi.world_size,
            device=device,
            x_dtype=moe.in_dtype,
            idx_dtype=topk_idx.dtype,
            weight_dtype=topk_weights.dtype,
        )
        print(f"[MoE Output Example - Rank 0]")
        print(f"  Received tokens: {recv_x.shape[0]}")
        print(f"  Tokens per expert: {recv_counts[:min(3, len(recv_counts))]}...")

    ret_data = (
        (total_sparse_bytes, dense_a2a_bytes),
        result,
    )

    # Cleanup
    del nvshmem_send_bufs
    del nvshmem_recv_bufs
    import gc
    gc.collect()
    torch.cuda.synchronize()

    return ret_data


def _worker_bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
) -> None:
    num_ranks = pgi.world_size
    global_rank = pgi.rank
    local_rank = pgi.local_rank

    dev = Device(local_rank)
    dev.set_current()

    nvshmem_init(
        global_rank=global_rank, local_rank=local_rank, world_size=num_ranks, device=dev
    )

    in_dtype = getattr(torch, in_dtype_str)
    out_dtype = getattr(torch, out_dtype_str)

    configs = [
        # V2-Lite:  64 Experts, 6 Experts per Token, 2048 Hidden Dim
        MoEConfig(64, 6, 2048, 1, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 4, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 8, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 16, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 32, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 64, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 128, in_dtype, out_dtype),
        # R1     : 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    if pgi.rank == 0:
        print("=" * 80)
        print("MoE Pack/Unpack API Benchmark")
        print("Comparing CPU vs CUDA Kernel Pack/Unpack with torch.dist & NVSHMEM")
        methods = "(1) CPU+Torch (2) CPU+NVSHMEM"
        if CUDA_AVAILABLE:
            methods += " (3) CUDA+Torch (4) CUDA+NVSHMEM"
        print(f"Methods: {methods}")
        print("=" * 80)

    header = [
        "E", "E/tok", "tok", "dim",
        "CPU+Torch_total", "CPU+Torch_lat", "CPU+Torch_bw", "CPU+Torch_comm", "CPU_pack",
        "CPU+NVSHMEM_total", "CPU+NVSHMEM_lat", "CPU+NVSHMEM_bw", "CPU+NVSHMEM_comm",
        "Dense_lat", "Dense_bw",
        "Sparse_bytes", "Dense_bytes",
    ]

    if CUDA_AVAILABLE:
        header.extend([
            "CUDA+Torch_total", "CUDA+Torch_lat", "CUDA+Torch_bw", "CUDA+Torch_comm", "CUDA_pack",
            "CUDA+NVSHMEM_total", "CUDA+NVSHMEM_lat", "CUDA+NVSHMEM_bw", "CUDA+NVSHMEM_comm",
            "Speedup_Torch", "Speedup_NVSHMEM",
        ])

    outpath = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_unified_moe_separated.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")
        f_out.write(f"EP={pgi.world_size} DP={pgi.world_size // dp_size}\n")
        f_out.write("\t".join(header) + "\n")

    for config in configs:
        if pgi.world_size > config.num_experts:
            continue

        result_data = bench_all_to_all(pgi, dp_size, config)
        if result_data[0] is None:
            continue

        meta, result = result_data
        sparse_bytes, dense_bytes = meta

        if pgi.rank == 0:
            # result columns:
            # 0: cpu_torch_us, 1: cpu_torch_gbps, 2: cpu_torch_comm_us, 3: cpu_pack_us
            # 4: cpu_nvshmem_us, 5: cpu_nvshmem_gbps, 6: cpu_nvshmem_comm_us, 7: cpu_pack_us
            # 8: cuda_torch_us, 9: cuda_torch_gbps, 10: cuda_torch_comm_us, 11: cuda_pack_us (if CUDA)
            # 12: cuda_nvshmem_us, 13: cuda_nvshmem_gbps, 14: cuda_nvshmem_comm_us, 15: cuda_pack_us (if CUDA)
            # 16: dense_us, 17: dense_gbps (if CUDA)

            cpu_torch_lat = result[:, 0].mean()
            cpu_torch_bw = result[:, 1].mean()
            cpu_torch_comm = result[:, 2].mean()
            cpu_pack = result[:, 3].mean()
            cpu_nvshmem_lat = result[:, 4].mean()
            cpu_nvshmem_bw = result[:, 5].mean()
            cpu_nvshmem_comm = result[:, 6].mean()
            # cpu_pack already extracted above (same for both methods)
            dense_lat = result[:, 8].mean()
            dense_bw = result[:, 9].mean()

            # Calculate CPU total time
            cpu_torch_total = cpu_torch_lat + cpu_pack
            cpu_nvshmem_total = cpu_nvshmem_lat + cpu_pack

            row = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                "CPU+Torch_total": f"{cpu_torch_total:.1f}μs",
                "CPU+Torch_lat": f"{cpu_torch_lat:.1f}μs",
                "CPU+Torch_bw": f"{cpu_torch_bw:.2f}GB/s",
                "CPU+Torch_comm": f"{cpu_torch_comm:.1f}μs",
                "CPU_pack": f"{cpu_pack:.1f}μs",
                "CPU+NVSHMEM_total": f"{cpu_nvshmem_total:.1f}μs",
                "CPU+NVSHMEM_lat": f"{cpu_nvshmem_lat:.1f}μs",
                "CPU+NVSHMEM_bw": f"{cpu_nvshmem_bw:.2f}GB/s",
                "CPU+NVSHMEM_comm": f"{cpu_nvshmem_comm:.1f}μs",
                "Dense_lat": f"{dense_lat:.1f}μs",
                "Dense_bw": f"{dense_bw:.2f}GB/s",
                "Sparse_bytes": f"{sparse_bytes:,}",
                "Dense_bytes": f"{dense_bytes:,}",
            }

            # Add CUDA kernel results if available
            if CUDA_AVAILABLE and result.shape[1] > 10:
                cuda_torch_lat = result[:, 8].mean()
                cuda_torch_bw = result[:, 9].mean()
                cuda_torch_comm = result[:, 10].mean()
                cuda_pack = result[:, 11].mean()
                cuda_nvshmem_lat = result[:, 12].mean()
                cuda_nvshmem_bw = result[:, 13].mean()
                cuda_nvshmem_comm = result[:, 14].mean()

                # Calculate CUDA total time
                cuda_torch_total = cuda_torch_lat + cuda_pack
                cuda_nvshmem_total = cuda_nvshmem_lat + cuda_pack

                # Speedup calculation (CPU vs CUDA) - use total time including pack
                speedup_torch = cpu_torch_total / cuda_torch_total if cuda_torch_total > 0 else 0
                speedup_nvshmem = cpu_nvshmem_total / cuda_nvshmem_total if cuda_nvshmem_total > 0 else 0

                row["CUDA+Torch_total"] = f"{cuda_torch_total:.1f}μs"
                row["CUDA+Torch_lat"] = f"{cuda_torch_lat:.1f}μs"
                row["CUDA+Torch_bw"] = f"{cuda_torch_bw:.2f}GB/s"
                row["CUDA+Torch_comm"] = f"{cuda_torch_comm:.1f}μs"
                row["CUDA_pack"] = f"{cuda_pack:.1f}μs"
                row["CUDA+NVSHMEM_total"] = f"{cuda_nvshmem_total:.1f}μs"
                row["CUDA+NVSHMEM_lat"] = f"{cuda_nvshmem_lat:.1f}μs"
                row["CUDA+NVSHMEM_bw"] = f"{cuda_nvshmem_bw:.2f}GB/s"
                row["CUDA+NVSHMEM_comm"] = f"{cuda_nvshmem_comm:.1f}μs"
                row["Speedup_Torch"] = f"{speedup_torch:.2f}x"
                row["Speedup_NVSHMEM"] = f"{speedup_nvshmem:.2f}x"

            line = "\t".join(row[h] for h in header)
            print(line)
            f_out.write(line + "\n")
            f_out.flush()

    if f_out:
        f_out.close()
        print(f"\n✅ Saved to {outpath}")

    nvshmem.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--in-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--out-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if "MASTER_ADDR" in os.environ:
        parallel_launch_from_env(_worker_bench_all_to_all, args.dp_size, args.in_dtype, args.out_dtype)
    else:
        world_size = torch.cuda.device_count()
        parallel_launch(world_size, _worker_bench_all_to_all, args.dp_size, args.in_dtype, args.out_dtype)


if __name__ == "__main__":
    main()
