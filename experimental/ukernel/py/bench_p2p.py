"""Benchmark P2P communication: ukernel_p2p vs uccl.p2p vs NCCL send/recv."""

import os
import time
import torch
import ukernel_p2p as p2p

import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def bench_p2p_ukernel(comm, peer, size_bytes, warmup, iters):
    """Benchmark ukernel_p2p send/recv bandwidth."""
    rank = comm.rank
    print(f"[rank {rank}] bench_p2p_ukernel start size={size_bytes}", flush=True)
    n = size_bytes // 4  # float32
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    send_buffer_id = 1
    recv_buffer_id = 2
    selected_transport = comm.peer_transport(peer)
    if not comm.reg_rdma(send_buffer_id, send_buf, publish=False):
        raise RuntimeError("reg_rdma(send) failed")
    if not comm.reg_rdma(recv_buffer_id, recv_buf, publish=True):
        raise RuntimeError("reg_rdma(recv) failed")
    ipc_registered = False
    if selected_transport == "ipc":
        if not comm.reg_ipc(recv_buffer_id, recv_buf, publish=True):
            raise RuntimeError("reg_ipc(recv) failed")
        if not comm.wait_ipc(peer, recv_buffer_id):
            raise RuntimeError("wait_ipc(peer recv buffer) failed")
        ipc_registered = True
    elif selected_transport == "uccl" or selected_transport == "rdma":
        # print(f"[rank {comm.rank}] wait_mr peer={peer} buf={recv_buffer_id}...", flush=True)
        if not comm.wait_mr(peer, recv_buffer_id):
            raise RuntimeError("wait_mr(peer recv buffer) failed")

    def do_send():
        comm.send(peer, send_buffer_id, remote_buffer_id=recv_buffer_id)

    def do_recv():
        comm.recv(peer, recv_buffer_id)

    rank = comm.rank
    # print(f"[rank {rank}] warmup loop start", flush=True)
    # Server (rank 0) recv first, client (rank 1) send first
    if rank == 0:
        for _ in range(warmup):
            do_recv()
            do_send()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            do_recv()
            do_send()
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            do_send()
            do_recv()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            do_send()
            do_recv()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if ipc_registered:
        comm.unreg_ipc(recv_buffer_id)
    comm.unreg_rdma(send_buffer_id)
    comm.unreg_rdma(recv_buffer_id)
    return elapsed


def bench_p2p_nccl(warmup, iters, size_bytes, nccl_group):
    """Benchmark NCCL send/recv bandwidth via torch.distributed."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    peer = 1 if rank == 0 else 0

    n = size_bytes // 4
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)

    if rank == 0:
        for _ in range(warmup):
            dist.recv(recv_buf, src=peer, group=nccl_group)
            dist.send(send_buf, dst=peer, group=nccl_group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.recv(recv_buf, src=peer, group=nccl_group)
            dist.send(send_buf, dst=peer, group=nccl_group)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            dist.send(send_buf, dst=peer, group=nccl_group)
            dist.recv(recv_buf, src=peer, group=nccl_group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.send(send_buf, dst=peer, group=nccl_group)
            dist.recv(recv_buf, src=peer, group=nccl_group)
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _send_i64(value: int, dst: int):
    t = torch.tensor([int(value)], dtype=torch.int64)
    dist.send(t, dst=dst)


def _recv_i64(src: int) -> int:
    t = torch.empty(1, dtype=torch.int64)
    dist.recv(t, src=src)
    return int(t.item())


def _send_bytes(payload: bytes, dst: int):
    _send_i64(len(payload), dst=dst)
    if payload:
        t = torch.tensor(list(payload), dtype=torch.uint8)
        dist.send(t, dst=dst)


def _recv_bytes(src: int) -> bytes:
    n = _recv_i64(src=src)
    if n == 0:
        return b""
    t = torch.empty(n, dtype=torch.uint8)
    dist.recv(t, src=src)
    return bytes(t.tolist())


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

    peer = 1 if rank == 0 else 0
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    uk_times = []
    uc_times = None
    nc_times = []

    # Phase 1: ukernel only
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
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")
    else:
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")
        if not comm.connect_peer(peer):
            raise RuntimeError(f"connect_peer({peer}) failed")

    if rank == 0:
        print(f"[ukernel] selected transport to peer {peer}: {comm.peer_transport(peer)}")

    for size in sizes:
        if size % 4 != 0:
            size = (size // 4) * 4
        if size == 0:
            uk_times.append(None)
            continue
        uk_times.append(bench_p2p_ukernel(comm, peer, size, warmup, iters))

    del comm
    dist.barrier()

    if rank == 0:
        print("[uccl] skip uccl column: two-sided uccl.p2p APIs were removed")
    dist.barrier()

    # Phase 3: NCCL only
    nccl_group = dist.new_group(ranks=[0, 1], backend="nccl")
    dist.barrier()
    for size in sizes:
        if size % 4 != 0:
            size = (size // 4) * 4
        if size == 0:
            nc_times.append(None)
            continue
        nc_times.append(bench_p2p_nccl(warmup, iters, size, nccl_group))
    dist.barrier()
    try:
        dist.destroy_process_group(nccl_group)
    except TypeError:
        # Older torch may not accept a group argument here.
        pass

    if rank == 0:
        print(
            f"{'Size':>12s} | {'ukernel (ms)':>13s} | {'ukernel (GB/s)':>15s} | "
            f"{'UCCL (ms)':>11s} | {'UCCL (GB/s)':>13s} | "
            f"{'NCCL (ms)':>11s} | {'NCCL (GB/s)':>13s}"
        )
        print("-" * 128)

        for i, size in enumerate(sizes):
            uk_time = uk_times[i]
            uc_time = uc_times[i] if uc_times is not None else None
            nc_time = nc_times[i]
            if uk_time is None or nc_time is None:
                continue
            uk_ms = uk_time / iters * 1000
            nc_ms = nc_time / iters * 1000
            uk_bw = (2 * size / 1e9) / (uk_time / iters)
            nc_bw = (2 * size / 1e9) / (nc_time / iters)
            if uc_time is not None:
                uc_ms = uc_time / iters * 1000
                uc_bw = (2 * size / 1e9) / (uc_time / iters)
            else:
                uc_ms = None
                uc_bw = None
            label = f"{size} B"
            uc_ms_str = f"{uc_ms:>11.3f}" if uc_ms is not None else f"{'N/A':>11s}"
            uc_bw_str = f"{uc_bw:>13.2f}" if uc_bw is not None else f"{'N/A':>13s}"
            print(
                f"{label:>12s} | {uk_ms:>13.3f} | {uk_bw:>15.2f} | "
                f"{uc_ms_str} | {uc_bw_str} | "
                f"{nc_ms:>11.3f} | {nc_bw:>13.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
