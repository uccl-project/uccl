"""Benchmark P2P communication: ukernel_p2p vs NCCL send/recv."""

import os
import time
import torch
import ukernel_p2p as p2p

import torch.distributed as nccl_dist


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def bench_p2p_ukernel(comm, peer, size_bytes, warmup, iters):
    """Benchmark ukernel_p2p send/recv bandwidth."""
    n = size_bytes // 4  # float32
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    comm.pin_tensor(send_buf)
    comm.pin_tensor(recv_buf)

    rank = comm.rank
    # Server (rank 0) recv first, client (rank 1) send first
    if rank == 0:
        for _ in range(warmup):
            comm.recv(peer, recv_buf)
            comm.send(peer, send_buf)  # echo back
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            comm.recv(peer, recv_buf)
            comm.send(peer, send_buf)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            comm.send(peer, send_buf)
            comm.recv(peer, recv_buf)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            comm.send(peer, send_buf)
            comm.recv(peer, recv_buf)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    comm.unpin_tensor(send_buf)
    comm.unpin_tensor(recv_buf)
    return elapsed


def bench_p2p_nccl(warmup, iters, size_bytes):
    """Benchmark NCCL send/recv bandwidth via torch.distributed."""
    rank = nccl_dist.get_rank()
    world = nccl_dist.get_world_size()
    peer = 1 if rank == 0 else 0

    n = size_bytes // 4
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)

    if rank == 0:
        for _ in range(warmup):
            nccl_dist.recv(recv_buf, src=peer)
            nccl_dist.send(send_buf, dst=peer)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            nccl_dist.recv(recv_buf, src=peer)
            nccl_dist.send(send_buf, dst=peer)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            nccl_dist.send(send_buf, dst=peer)
            nccl_dist.recv(recv_buf, src=peer)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            nccl_dist.send(send_buf, dst=peer)
            nccl_dist.recv(recv_buf, src=peer)
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 2)
    local_rank = env_int("LOCAL_RANK", rank)

    if world != 2:
        if rank == 0:
            print("P2P benchmark requires exactly 2 ranks")
        return

    torch.cuda.set_device(local_rank)

    warmup = 3
    iters = 20

    sizes = [
        1 << 10,       # 4 KB
        1 << 12,       # 16 KB
        1 << 14,       # 64 KB
        1 << 16,       # 256 KB
        1 << 18,       # 1 MB
        1 << 20,       # 4 MB
        1 << 22,       # 16 MB
        1 << 24,       # 64 MB
        1 << 26,       # 256 MB
        1 << 28,       # 1 GB
    ]

    exchanger_port = env_int("EXCHANGER_PORT", 29610)

    if rank == 0:
        print(f"{'Size':>12s} | {'ukernel (ms)':>13s} | {'ukernel (GB/s)':>15s} | {'NCCL (ms)':>11s} | {'NCCL (GB/s)':>13s}")
        print("-" * 80)

    peer = 1 if rank == 0 else 0

    # --- Init ukernel_p2p once ---
    comm = p2p.Communicator(
        gpu_id=local_rank,
        rank=rank,
        world_size=world,
        exchanger_ip=os.getenv("MASTER_ADDR", "127.0.0.1"),
        exchanger_port=exchanger_port,
        transport=os.getenv("UK_P2P_TRANSPORT", "auto"),
    )
    if peer == 1:
        if not comm.connect_peer(peer):
            raise RuntimeError(f"connect_peer({peer}) failed")
    else:
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")

    if rank == 0:
        print(f"[ukernel] selected transport to peer {peer}: {comm.peer_transport(peer)}")

    # --- Init NCCL once ---
    nccl_dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    for size in sizes:
        if size % 4 != 0:
            size = (size // 4) * 4
        if size == 0:
            continue

        uk_time = bench_p2p_ukernel(comm, peer, size, warmup, iters)
        nc_time = bench_p2p_nccl(warmup, iters, size)

        if rank == 0:
            uk_ms = uk_time / iters * 1000
            nc_ms = nc_time / iters * 1000
            uk_bw = (2 * size / 1e9) / (uk_time / iters)
            nc_bw = (2 * size / 1e9) / (nc_time / iters)
            label = f"{size} B"
            print(f"{label:>12s} | {uk_ms:>13.3f} | {uk_bw:>15.2f} | {nc_ms:>11.3f} | {nc_bw:>13.2f}")

    del comm
    nccl_dist.destroy_process_group()


if __name__ == "__main__":
    main()
