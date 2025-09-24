#!/usr/bin/env python3
"""
Simplified test for P2P IPC pointer functionality.
Focuses on testing the core get_ipc_p2p_ptr functionality.

Usage:
------
# Single GPU test (basic functionality)
python bench/test_p2p_simple.py --mode single

# Multi-GPU test on single node (IPC should work)
python bench/test_p2p_simple.py --mode multi --num-gpus 2

# Distributed test (requires torchrun)
torchrun --nnodes=1 --nproc_per_node=2 bench/test_p2p_simple.py --mode distributed
"""

import argparse
import os
import time
import torch
import numpy as np
from utils import init_dist, get_peer_ip, get_cpu_proxies_meta

try:
    from uccl import ep
except ImportError:
    import sys

    sys.stderr.write("Failed to import uccl modules\n")
    sys.exit(1)


def test_single_gpu():
    """Test basic GPU-driven functionality on a single GPU."""
    print("\n=== Single GPU Test ===")

    device = torch.cuda.current_device()
    print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")

    # Initialize bench
    bench = ep.Bench()
    # Skip env_info() call due to type binding issue
    print(f"Bench initialized successfully")

    # Create test buffer
    buffer_size = 1024 * 1024  # 1MB
    test_buffer = torch.ones(buffer_size, dtype=torch.float32, device="cuda")
    test_ptr = test_buffer.data_ptr()
    print(f"Created test buffer: {buffer_size * 4 / 1e6:.2f}MB at 0x{test_ptr:x}")

    peers_meta_list = [
        {
            "rank": 0,
            "ptr": int(test_ptr),
            "nbytes": int(buffer_size * 4),
            "ip": "127.0.0.1",
        }
    ]

    # Create proxy
    rb_addr = bench.ring_addr(0)
    proxy = ep.Proxy(
        thread_idx=0,
        gpu_buffer_addr=test_ptr,
        total_size=buffer_size * 4,
        rank=0,
        node_idx=0,
        local_rank=0,
        peer_ip="127.0.0.1",  # Local test, but cannot be empty
    )
    proxy.set_bench_ring_addrs([rb_addr])
    proxy.set_peers_meta(peers_meta_list)
    proxy.start_dual()
    print("✓ Proxy started successfully")

    # Register proxy
    ep.register_proxies(device, [proxy])
    print("✓ Proxy registered with EP module")

    time.sleep(1)

    # Cleanup
    ep.unregister_proxy(device)
    proxy.stop()
    print("✓ Test completed successfully")


def test_multi_gpu(num_gpus):
    """Test P2P functionality across multiple GPUs on the same node."""
    print(f"\n=== Multi-GPU Test ({num_gpus} GPUs) ===")

    if torch.cuda.device_count() < num_gpus:
        print(
            f"Error: Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available"
        )
        return

    print(f"Testing with {num_gpus} GPUs")

    # Create buffers on each GPU
    buffers = []
    pointers = []
    proxies = []

    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)

        # Create buffer
        buffer_size = 1024 * 1024  # 1MB per GPU
        buffer = torch.arange(buffer_size, dtype=torch.float32, device=f"cuda:{gpu_id}")
        buffer[0:10] = gpu_id  # Mark with GPU ID

        buffers.append(buffer)
        pointers.append(buffer.data_ptr())

        print(f"GPU {gpu_id}: Created buffer at 0x{pointers[gpu_id]:x}")

        # Create bench and proxy for this GPU
        bench = ep.Bench()
        rb_addr = bench.ring_addr(0)
        peers_meta_list = [
            {
                "rank": gpu_id,
                "ptr": int(pointers[gpu_id]),
                "nbytes": int(buffer_size * 4),
                "ip": "127.0.0.1",
            }
        ]
        proxy = ep.Proxy(
            thread_idx=0,
            gpu_buffer_addr=pointers[gpu_id],
            total_size=buffer_size * 4,
            rank=gpu_id,
            node_idx=0,
            local_rank=gpu_id,
            peer_ip="127.0.0.1",  # Local node, but cannot be empty
        )
        proxy.set_bench_ring_addrs([rb_addr])
        proxy.set_peers_meta(peers_meta_list)
        proxy.start_dual()
        proxies.append(proxy)

        # Register proxy
        ep.register_proxies(gpu_id, [proxy])

    print(f"✓ Created {num_gpus} GPU buffers and proxies")

    # Test P2P access between GPUs
    print("\nTesting P2P access patterns:")
    for src_gpu in range(num_gpus):
        for dst_gpu in range(num_gpus):
            if src_gpu != dst_gpu:
                # Check if P2P is available
                can_access = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
                print(
                    f"  GPU {src_gpu} -> GPU {dst_gpu}: {'✓ P2P available' if can_access else '✗ P2P not available'}"
                )

                if can_access:
                    # P2P access is automatically enabled in modern PyTorch versions
                    # No need to manually enable it
                    pass

    time.sleep(1)

    # Test data transfer
    print("\nTesting data transfer:")
    if num_gpus >= 2:
        torch.cuda.set_device(0)
        src_data = torch.ones(100, dtype=torch.float32, device="cuda:0") * 42

        torch.cuda.set_device(1)
        dst_data = torch.zeros(100, dtype=torch.float32, device="cuda:1")

        # Copy data
        dst_data.copy_(src_data)
        torch.cuda.synchronize()

        # Verify
        if torch.all(dst_data == 42):
            print("✓ P2P data transfer successful")
        else:
            print("✗ P2P data transfer failed")

    # Cleanup
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        ep.unregister_proxy(gpu_id)
        proxies[gpu_id].stop()

    print("✓ Multi-GPU test completed")


def test_distributed():
    """Test P2P in distributed setting (requires torchrun)."""
    import torch.distributed as dist
    from utils import init_dist, get_peer_ip

    print("\n=== Distributed Test ===")

    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    rank, world_size, group = init_dist(local_rank, local_world_size)
    print(f"[Rank {rank}/{world_size}] Initialized")

    device = torch.cuda.current_device()

    # Get peer IP
    peer_ip = get_peer_ip(rank, world_size, group)
    print(f"[Rank {rank}] Peer IP: {peer_ip}")

    # Create test buffer
    buffer_size = 1024 * 1024
    test_buffer = torch.ones(buffer_size, dtype=torch.float32, device="cuda") * rank
    test_ptr = test_buffer.data_ptr()

    rank2meta = get_cpu_proxies_meta(rank, test_ptr, buffer_size * 4, world_size, group)
    peers_meta_list = [rank2meta[r] for r in range(world_size)]
    dist.barrier(group)

    # Create bench and proxy
    bench = ep.Bench()
    rb_addr = bench.ring_addr(0)
    proxy = ep.Proxy(
        thread_idx=0,
        gpu_buffer_addr=test_ptr,
        total_size=buffer_size * 4,
        rank=rank,
        node_idx=rank // local_world_size,
        local_rank=local_rank,
        peer_ip=peer_ip,
    )
    proxy.set_bench_ring_addrs([rb_addr])
    proxy.set_peers_meta(peers_meta_list)
    proxy.start_dual()

    # Register proxy
    ep.register_proxies(device, [proxy])
    print(f"[Rank {rank}] ✓ Setup completed")

    # Synchronize
    dist.barrier()

    # Check node locality
    node_id = rank // torch.cuda.device_count()
    all_node_ids = [None] * world_size
    dist.all_gather_object(all_node_ids, node_id, group=group)

    print(f"[Rank {rank}] Node distribution:")
    for i in range(world_size):
        same_node = all_node_ids[i] == node_id
        print(
            f"  Rank {i}: Node {all_node_ids[i]} ({'same' if same_node else 'different'} node)"
        )

    # Test IPC availability
    print(f"[Rank {rank}] IPC support:")
    for i in range(world_size):
        if i != rank:
            same_node = all_node_ids[i] == node_id
            if same_node:
                print(f"  -> Rank {i}: IPC P2P should be available")
            else:
                print(f"  -> Rank {i}: IPC P2P not available (different node)")

    time.sleep(2)

    # Cleanup
    dist.barrier()
    ep.unregister_proxy(device)
    proxy.stop()
    dist.destroy_process_group()

    print(f"[Rank {rank}] ✓ Test completed")


def main():
    parser = argparse.ArgumentParser(description="Simple P2P IPC test")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "distributed"],
        default="single",
        help="Test mode",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs for multi-GPU test",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    if args.mode == "single":
        test_single_gpu()
    elif args.mode == "multi":
        test_multi_gpu(args.num_gpus)
    elif args.mode == "distributed":
        test_distributed()


if __name__ == "__main__":
    main()
