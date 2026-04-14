"""Benchmark P2P communication: ukernel_p2p vs uccl.p2p vs NCCL send/recv."""

import os
import time
import torch
import ukernel_p2p as p2p

import torch.distributed as dist

try:
    from uccl import p2p as uccl_p2p
except Exception:
    uccl_p2p = None


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def bench_p2p_ukernel(comm, peer, size_bytes, warmup, iters):
    """Benchmark ukernel_p2p send/recv bandwidth."""
    n = size_bytes // 4  # float32
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    comm.pin_tensor(send_buf)
    recv_mr_id = comm.pin_tensor(recv_buf)

    # Stable logical id for destination receive buffer in ping-pong.
    recv_buffer_id = 2
    if not comm.publish_mr(peer, recv_buffer_id, recv_mr_id):
        raise RuntimeError("publish_mr(recv) failed")
    # Wait for peer's next published recv-buffer mapping.
    comm.wait_mr(peer, recv_buffer_id)

    def do_send():
        # Sender writes into peer's recv buffer id.
        comm.send_buffer(
            peer,
            send_buf,
            recv_buffer_id,
            remote_offset=0,
        )

    def do_recv():
        comm.recv(peer, recv_buf)

    rank = comm.rank
    # Server (rank 0) recv first, client (rank 1) send first
    if rank == 0:
        for _ in range(warmup):
            do_recv()
            do_send()  # echo back
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
    comm.unpin_tensor(send_buf)
    comm.unpin_tensor(recv_buf)
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


def _exchange_uccl_metadata(local_metadata: bytes, local_gpu_idx: int, rank: int):
    if rank == 0:
        _send_bytes(local_metadata, dst=1)
        _send_i64(local_gpu_idx, dst=1)
        remote_metadata = _recv_bytes(src=1)
        remote_gpu_idx = _recv_i64(src=1)
    else:
        remote_metadata = _recv_bytes(src=0)
        remote_gpu_idx = _recv_i64(src=0)
        _send_bytes(local_metadata, dst=0)
        _send_i64(local_gpu_idx, dst=0)
    return remote_metadata, remote_gpu_idx


def _exchange_local_gpu_idx(local_gpu_idx: int, rank: int) -> int:
    if rank == 0:
        _send_i64(local_gpu_idx, dst=1)
        return _recv_i64(src=1)
    remote_gpu_idx = _recv_i64(src=0)
    _send_i64(local_gpu_idx, dst=0)
    return remote_gpu_idx


def _create_uccl_endpoint(local_gpu_idx: int, rank: int):
    # Support multiple p2p python binding signatures.
    for num_cpus in (rank, 0):
        try:
            return uccl_p2p.Endpoint(local_gpu_idx, num_cpus)
        except TypeError:
            pass
    try:
        return uccl_p2p.Endpoint(local_gpu_idx)
    except TypeError:
        return uccl_p2p.Endpoint(rank)


def bench_p2p_uccl(ep, send_conn_id, recv_conn_id, rank, size_bytes, warmup, iters, mode):
    """Benchmark uccl.p2p ping-pong bandwidth (rdma or ipc)."""
    peer = 1 if rank == 0 else 0
    _ = peer
    n = size_bytes // 4
    send_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    recv_buf = torch.empty(n, device="cuda", dtype=torch.float32)
    if mode == "rdma":
        ok, send_mr = ep.reg(send_buf.data_ptr(), size_bytes)
        assert ok, "[uccl] reg(send) failed"
        ok, recv_mr = ep.reg(recv_buf.data_ptr(), size_bytes)
        assert ok, "[uccl] reg(recv) failed"
    else:
        send_mr = None
        recv_mr = None

    def _send():
        if mode == "ipc":
            ok = ep.send_ipc(send_conn_id, send_buf.data_ptr(), size_bytes)
        else:
            ok = ep.send(send_conn_id, send_mr, send_buf.data_ptr(), size_bytes)
        assert ok, "[uccl] send failed"

    def _recv():
        if mode == "ipc":
            ok = ep.recv_ipc(recv_conn_id, recv_buf.data_ptr(), size_bytes)
        else:
            ok = ep.recv(recv_conn_id, recv_mr, recv_buf.data_ptr(), size_bytes)
        assert ok, "[uccl] recv failed"

    if rank == 0:
        for _ in range(warmup):
            _recv()
            _send()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _recv()
            _send()
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            _send()
            _recv()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _send()
            _recv()
        torch.cuda.synchronize()

    if mode == "rdma":
        ep.dereg(send_mr)
        ep.dereg(recv_mr)
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

    peer = 1 if rank == 0 else 0
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    uk_times = []
    uc_times = [] if uccl_p2p is not None else None
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
    else:
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")

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

    # Phase 2: uccl p2p only
    if uc_times is not None:
        uc_mode = os.getenv("UCCL_P2P_MODE", "ipc").strip().lower()
        if uc_mode not in ("rdma", "ipc"):
            raise RuntimeError(f"invalid UCCL_P2P_MODE={uc_mode}, expected rdma/ipc")

        uc_ep = None
        uc_send_conn_id = None
        uc_recv_conn_id = None
        uc_ep = _create_uccl_endpoint(local_rank, rank)
        if uc_mode == "ipc":
            required_ipc_api = ("connect_local", "accept_local", "send_ipc", "recv_ipc")
            missing = [name for name in required_ipc_api if not hasattr(uc_ep, name)]
            if missing:
                raise RuntimeError(
                    f"[uccl] ipc mode requested but endpoint lacks API: {missing}"
                )
            remote_gpu_idx = _exchange_local_gpu_idx(local_rank, rank)
            if rank == 0:
                ok, uc_send_conn_id = uc_ep.connect_local(remote_gpu_idx)
                assert ok, "[uccl] connect_local(0->1) failed"
                ok, _, uc_recv_conn_id = uc_ep.accept_local()
                assert ok, "[uccl] accept_local(1->0) failed"
            else:
                ok, _, uc_recv_conn_id = uc_ep.accept_local()
                assert ok, "[uccl] accept_local(0->1) failed"
                ok, uc_send_conn_id = uc_ep.connect_local(remote_gpu_idx)
                assert ok, "[uccl] connect_local(1->0) failed"
        else:
            local_meta = bytes(uc_ep.get_metadata())
            remote_meta, remote_gpu_idx = _exchange_uccl_metadata(local_meta, local_rank, rank)
            if rank == 0:
                ip, port, _ = uccl_p2p.Endpoint.parse_metadata(remote_meta)
                ok, uc_send_conn_id = uc_ep.connect(ip, remote_gpu_idx, remote_port=port)
                assert ok, "[uccl] connect(0->1) failed"
                ok, _, _, uc_recv_conn_id = uc_ep.accept()
                assert ok, "[uccl] accept(1->0) failed"
            else:
                ok, _, _, uc_recv_conn_id = uc_ep.accept()
                assert ok, "[uccl] accept(0->1) failed"
                ip, port, _ = uccl_p2p.Endpoint.parse_metadata(remote_meta)
                ok, uc_send_conn_id = uc_ep.connect(ip, remote_gpu_idx, remote_port=port)
                assert ok, "[uccl] connect(1->0) failed"
        dist.barrier()
        if rank == 0:
            print(
                f"[uccl] p2p endpoint connected mode={uc_mode} "
                f"(send_conn={uc_send_conn_id}, recv_conn={uc_recv_conn_id})"
            )
        for size in sizes:
            if size % 4 != 0:
                size = (size // 4) * 4
            if size == 0:
                uc_times.append(None)
                continue
            uc_times.append(
                bench_p2p_uccl(
                    uc_ep,
                    uc_send_conn_id,
                    uc_recv_conn_id,
                    rank,
                    size,
                    warmup,
                    iters,
                    uc_mode,
                )
            )
        del uc_ep
    else:
        if rank == 0:
            print("[uccl] p2p python module not found; skip uccl column")
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
