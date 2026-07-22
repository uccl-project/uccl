"""Bidirectional P2P data transfer via send_put_async + signal + wait_signal_async."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch
import ukernel_p2p as p2p


N = 256 * 1024
SEND_BUF_ID = 100
RECV_BUF_ID = 200


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def poll_rid(comm, rid):
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if rid in comm.poll([rid]):
            return
    raise RuntimeError(f"timeout waiting for rid={rid}")


def run_rank() -> bool:
    rank = env_int("RANK", 0)
    peer = 1 - rank
    gpu_id = env_int("LOCAL_RANK", 0)
    exchanger_port = env_int("EXCHANGER_PORT", 29610)

    comm = p2p.Communicator(
        gpu_id=gpu_id, rank=rank, world_size=2,
        exchanger_ip="127.0.0.1", exchanger_port=exchanger_port,
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
    print(f"[rank {rank}] transport={selected}", flush=True)

    if selected == "tcp":
        raise RuntimeError("TCP not supported by this test")

    # Each rank fills its send buffer with a distinct pattern.
    start_val = 0 if rank == 0 else N
    send = torch.arange(start_val, start_val + N, device="cuda",
                        dtype=torch.float32)
    recv = torch.empty(N, device="cuda", dtype=torch.float32)

    if not comm.reg_rdma(SEND_BUF_ID, send, publish=False):
        raise RuntimeError("reg_rdma(send) failed")
    if not comm.reg_rdma(RECV_BUF_ID, recv, publish=True):
        raise RuntimeError("reg_rdma(recv) failed")
    if selected == "ipc":
        if not comm.reg_ipc(RECV_BUF_ID, recv, publish=True):
            raise RuntimeError("reg_ipc(recv) failed")

    # Resolve the peer's receive buffer.
    if selected == "ipc":
        if not comm.wait_ipc(peer, RECV_BUF_ID):
            raise RuntimeError("wait_ipc(peer recv) failed")
    else:
        if not comm.wait_mr(peer, RECV_BUF_ID):
            raise RuntimeError("wait_mr(peer recv) failed")

    # Tags: each direction uses a distinct tag.
    my_tag = 10 + rank      # tag I send to signal my put is done
    peer_tag = 10 + peer    # tag I wait for from peer

    # Issue PUT into peer's recv buffer.
    put_rid = comm.send_put_async(peer, local_buf=SEND_BUF_ID,
                                   remote_buf=RECV_BUF_ID)
    if put_rid == 0:
        raise RuntimeError("send_put_async returned 0")
    poll_rid(comm, put_rid)

    # Tell peer my data is ready.
    comm.signal(peer, my_tag)

    # Wait for peer's signal.
    wait_rid = comm.wait_signal_async(peer, peer_tag)
    if wait_rid == 0:
        raise RuntimeError("wait_signal_async returned 0")
    poll_rid(comm, wait_rid)

    torch.cuda.synchronize()

    expected_start = 0 if peer == 0 else N
    expected = torch.arange(expected_start, expected_start + N,
                            dtype=torch.float32)
    recv_cpu = recv.cpu()
    if not torch.equal(recv_cpu, expected):
        print(f"[rank {rank}] recv mismatch", flush=True)
        print(f"[rank {rank}] expected: {expected[:16]}...", flush=True)
        print(f"[rank {rank}] actual:   {recv_cpu[:16]}...", flush=True)
        return False

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
