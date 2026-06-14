"""Bidirectional P2P data transfer using only send_put_async + poll."""

import os
import sys
import torch
import ukernel_p2p as p2p


N = 256 * 1024
SEND_BUF_ID = 100
RECV_BUF_ID = 200


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def run_rank() -> bool:
    rank = env_int("RANK", 0)
    peer = 1 - rank
    world = 2
    gpu_id = env_int("LOCAL_RANK", 0)
    exchanger_port = env_int("EXCHANGER_PORT", 29610)

    comm = p2p.Communicator(
        gpu_id=gpu_id,
        rank=rank,
        world_size=world,
        exchanger_ip="127.0.0.1",
        exchanger_port=exchanger_port,
        transport=os.getenv("UK_P2P_TRANSPORT", "auto"),
    )

    if rank == 0:
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")
        if not comm.connect_peer(peer):
            raise RuntimeError(f"connect_peer({peer}) failed")
    else:
        if not comm.connect_peer(peer):
            raise RuntimeError(f"connect_peer({peer}) failed")
        if not comm.accept_peer(peer):
            raise RuntimeError(f"accept_peer({peer}) failed")

    selected = comm.peer_transport(peer)
    print(f"[rank {rank}] connected to peer {peer}; transport={selected}", flush=True)

    if selected == "tcp":
        raise RuntimeError(
            "TCP transport requires signal/wait and is not supported by this test"
        )

    # Each rank fills its send buffer with a distinct pattern.
    start_val = 0 if rank == 0 else N
    send = torch.arange(start_val, start_val + N, device="cuda", dtype=torch.float32)
    recv = torch.empty(N, device="cuda", dtype=torch.float32)

    if not comm.reg_rdma(SEND_BUF_ID, send, publish=False):
        raise RuntimeError("reg_rdma(send) failed")
    if not comm.reg_rdma(RECV_BUF_ID, recv, publish=True):
        raise RuntimeError("reg_rdma(recv) failed")
    if selected == "ipc":
        if not comm.reg_ipc(RECV_BUF_ID, recv, publish=True):
            raise RuntimeError("reg_ipc(recv) failed")

    # Resolve the peer's receive buffer before issuing the PUT.
    if selected == "ipc":
        if not comm.wait_ipc(peer, RECV_BUF_ID):
            raise RuntimeError("wait_ipc(peer recv) failed")
    else:
        if not comm.wait_mr(peer, RECV_BUF_ID):
            raise RuntimeError("wait_mr(peer recv) failed")

    comm.barrier()

    # Issue one-sided PUT into peer's receive buffer.
    if rank == 0:
        rid = comm.send_put_async(
            peer, local_buf=SEND_BUF_ID, remote_buf=RECV_BUF_ID, remote_off=0
        )
        if rid == 0:
            raise RuntimeError("send_put_async returned 0")
        while rid not in comm.poll([rid]):
            pass
        comm.barrier()
    else:
        comm.barrier()
        rid = comm.send_put_async(
            peer, local_buf=SEND_BUF_ID, remote_buf=RECV_BUF_ID, remote_off=0
        )
        if rid == 0:
            raise RuntimeError("send_put_async returned 0")
        while rid not in comm.poll([rid]):
            pass
        comm.barrier()

    # Ensure GPU writes are visible before validating on the CPU.
    torch.cuda.synchronize()

    expected_start = 0 if peer == 0 else N
    expected = torch.arange(expected_start, expected_start + N, dtype=torch.float32)
    recv_cpu = recv.cpu()
    if not torch.equal(recv_cpu, expected):
        print(f"[rank {rank}] recv mismatch", flush=True)
        print(f"[rank {rank}] expected: {expected[:16]}...", flush=True)
        print(f"[rank {rank}] actual:   {recv_cpu[:16]}...", flush=True)
        return False

    # Cleanup.
    if selected == "ipc":
        comm.unreg_ipc(RECV_BUF_ID)
    comm.unreg_rdma(SEND_BUF_ID)
    comm.unreg_rdma(RECV_BUF_ID)

    print(f"[rank {rank}] data validated", flush=True)
    return True


def main() -> None:
    ok = run_rank()
    if ok:
        print("PASS", flush=True)
    else:
        print("FAIL", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
