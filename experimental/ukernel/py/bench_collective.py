"""Benchmark collective communication: ukernel_ccl vs NCCL."""

import os
import time
import torch

import ukernel_ccl as dist

import torch.distributed as nccl_dist


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def bench_allreduce_ukernel(pg, tensor, tile_bytes, num_flows, warmup, iters):
    for _ in range(warmup):
        pg.all_reduce(tensor, tile_bytes=tile_bytes, num_flows=num_flows)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        pg.all_reduce(tensor, tile_bytes=tile_bytes, num_flows=num_flows)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_allreduce_nccl(tensor, warmup, iters):
    for _ in range(warmup):
        nccl_dist.all_reduce(tensor, op=nccl_dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        nccl_dist.all_reduce(tensor, op=nccl_dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_alltoall_ukernel(pg, tensor, tile_bytes, num_flows, warmup, iters):
    for _ in range(warmup):
        pg.all_to_all_single(tensor, tensor, tile_bytes=tile_bytes, num_flows=num_flows)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        pg.all_to_all_single(tensor, tensor, tile_bytes=tile_bytes, num_flows=num_flows)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_alltoall_nccl(tensor, warmup, iters):
    for _ in range(warmup):
        nccl_dist.all_to_all_single(tensor, tensor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        nccl_dist.all_to_all_single(tensor, tensor)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 2)
    local_rank = env_int("LOCAL_RANK", rank)

    torch.cuda.set_device(local_rank)

    tile_bytes = 64 << 10
    num_flows = 2
    warmup = 3
    iters = 20

    sizes = [
        1 << 10,       # 4 KB
        1 << 14,       # 64 KB
        1 << 18,       # 1 MB
        1 << 22,       # 16 MB
        1 << 26,       # 256 MB
    ]

    if rank == 0:
        print(f"{'Size':>12s} | {'ukernel AR (ms)':>16s} | {'NCCL AR (ms)':>14s} | {'ukernel A2A (ms)':>17s} | {'NCCL A2A (ms)':>14s}")
        print("-" * 90)

    # --- Init ukernel_ccl once ---
    pg = dist.init_process_group(
        backend="ukernel",
        rank=rank,
        world_size=world,
        gpu_id=local_rank,
        exchanger_ip=os.getenv("MASTER_ADDR", "127.0.0.1"),
        exchanger_port=env_int("EXCHANGER_PORT", 29600),
        transport="auto",
    )

    # --- Init NCCL once ---
    nccl_dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    for size in sizes:
        if size % world != 0:
            size = (size // world) * world
        if size == 0:
            continue

        # ukernel_ccl
        t_uk_ar = torch.empty(size, device="cuda", dtype=torch.float32)
        t_uk_a2a = torch.empty(size, device="cuda", dtype=torch.float32)
        ar_time = bench_allreduce_ukernel(pg, t_uk_ar, tile_bytes, num_flows, warmup, iters)
        a2a_time = bench_alltoall_ukernel(pg, t_uk_a2a, tile_bytes, num_flows, warmup, iters)

        # NCCL
        t_nc_ar = torch.empty(size, device="cuda", dtype=torch.float32)
        t_nc_a2a = torch.empty(size, device="cuda", dtype=torch.float32)
        nc_ar_time = bench_allreduce_nccl(t_nc_ar, warmup, iters)
        nc_a2a_time = bench_alltoall_nccl(t_nc_a2a, warmup, iters)

        if rank == 0:
            ar_ms = ar_time / iters * 1000
            nc_ar_ms = nc_ar_time / iters * 1000
            a2a_ms = a2a_time / iters * 1000
            nc_a2a_ms = nc_a2a_time / iters * 1000
            label = f"{size * 4} B"
            print(f"{label:>12s} | {ar_ms:>16.3f} | {nc_ar_ms:>14.3f} | {a2a_ms:>17.3f} | {nc_a2a_ms:>14.3f}")

    dist.destroy_process_group()
    nccl_dist.destroy_process_group()


if __name__ == "__main__":
    main()
