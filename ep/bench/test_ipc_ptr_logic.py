#!/usr/bin/env python3
"""
Simple IPC test without RDMA components
"""

import torch
import torch.distributed as dist
import os
import sys

try:
    from uccl import uccl_ep as ep
except ImportError:
    sys.stderr.write("Failed to import uccl modules\n")
    sys.exit(1)

from utils import init_dist


def test_simple_ipc(rank: int, world_size: int, group: dist.ProcessGroup):
    """Test simple IPC without RDMA components."""

    device_index = torch.cuda.current_device()
    print(f"\n[Rank {rank}] === Simple IPC Test ===")
    print(f"[Rank {rank}] Device: {torch.cuda.get_device_name(device_index)}")
    print(f"[Rank {rank}] World size: {world_size}")

    # Create a simple buffer
    buffer_size = int(1e6)  # 1MB
    buffer = torch.zeros(buffer_size, dtype=torch.uint8, device="cuda")
    buffer_ptr = buffer.data_ptr()

    print(
        f"[Rank {rank}] Buffer created at 0x{buffer_ptr:x}, size: {buffer_size} bytes"
    )

    # Synchronize
    dist.barrier(group)

    # Test node locality detection
    gpus_per_node = torch.cuda.device_count()
    my_node_id = rank // gpus_per_node

    all_node_ids = [None] * world_size
    dist.all_gather_object(all_node_ids, my_node_id, group=group)

    all_ptrs = [None] * world_size
    dist.all_gather_object(all_ptrs, buffer_ptr, group=group)

    print(f"[Rank {rank}] My node ID: {my_node_id}")
    print(f"[Rank {rank}] All node IDs: {all_node_ids}")
    print(f"[Rank {rank}] All buffer ptrs: {[hex(p) for p in all_ptrs]}")

    # Test IPC logic manually
    for target_rank in range(world_size):
        target_node_id = all_node_ids[target_rank]
        target_ptr = all_ptrs[target_rank]
        is_same_node = my_node_id == target_node_id

        print(f"\n[Rank {rank}] Testing with rank {target_rank}:")
        print(f"[Rank {rank}]   Target node ID: {target_node_id}")
        print(f"[Rank {rank}]   Target ptr: 0x{target_ptr:x}")
        print(f"[Rank {rank}]   Same node? {is_same_node}")

        if rank == target_rank:
            print(f"[Rank {rank}]   -> Same rank: Would return local pointer")
        elif is_same_node:
            print(f"[Rank {rank}]   -> Same node: Would use IPC")
            # In real implementation, this would create IPC handle
            print(
                f"[Rank {rank}]   -> IPC handle would be created for 0x{target_ptr:x}"
            )
        else:
            print(f"[Rank {rank}]   -> Different node: Would use RDMA (return nullptr)")

    # Test offset calculations
    print(f"\n[Rank {rank}] Testing offset calculations:")
    test_offsets = [0, 1024, 4096]
    for offset in test_offsets:
        test_ptr = buffer_ptr + offset
        calculated_offset = test_ptr - buffer_ptr
        print(
            f"[Rank {rank}]   Offset {offset}: ptr=0x{test_ptr:x}, calc_offset={calculated_offset}"
        )

    dist.barrier(group)
    print(f"[Rank {rank}] Simple IPC test completed successfully!")


def main():
    if "LOCAL_RANK" not in os.environ:
        print("This test requires torchrun")
        print("Usage: torchrun --nnodes=1 --nproc_per_node=2 simple_ipc_test.py")
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    print(f"Initializing rank {local_rank}/{local_world_size}")

    rank, world_size, group = init_dist(local_rank, local_world_size)

    try:
        test_simple_ipc(rank, world_size, group)
        print(f"\n[Rank {rank}] ALL TESTS COMPLETED!")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
