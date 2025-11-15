#!/usr/bin/env python3
"""
Launch script for all-to-all RDMA benchmark using torch.distributed
Usage: torchrun --nnodes=<N> --node_rank=<0/1/../N-1> --nproc_per_node=8 --master_addr=<main_ip> launch.py
Eg:
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=10.1.18.53 launch.py
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=10.1.18.53 launch.py
"""

import os
import subprocess
import sys
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Error: Must run with torchrun")
        sys.exit(1)

    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    print(
        f"Rank {rank}/{world_size} on GPU {local_rank}, master: {master_addr}"
    )

    dist.barrier()

    benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark")
    if not os.path.exists(benchmark_path):
        if rank == 0:
            print(f"Error: benchmark executable not found at {benchmark_path}")
            print("Please run 'make' first")
        sys.exit(1)

    cmd = [benchmark_path, str(rank), str(world_size), master_addr]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Rank {rank} failed: {e}")
        sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
