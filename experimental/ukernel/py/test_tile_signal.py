"""Test out-of-order tag delivery via wait_signal + try_complete_signals API."""

import os
import time
import torch
import ukernel_p2p as p2p


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def run_sender() -> None:
    rank = 0
    world = 2
    exchanger_port = env_int("EXCHANGER_PORT", 29611)
    gpu_id = env_int("LOCAL_RANK", 0)

    comm = p2p.Communicator(
        gpu_id=gpu_id,
        rank=rank,
        world_size=world,
        exchanger_ip="127.0.0.1",
        exchanger_port=exchanger_port,
        transport=os.getenv("UK_P2P_TRANSPORT", "auto"),
    )

    if not comm.connect_peer(1):
        raise RuntimeError("connect_peer(1) failed")
    selected = comm.peer_transport(1)
    print(f"[sender] transport: {selected}")

    # Register a small buffer so the signal path is fully set up
    buf = torch.empty(16, device="cuda", dtype=torch.float32)
    buf_id = 100
    if not comm.reg_rdma(buf_id, buf, publish=False):
        raise RuntimeError("reg_rdma failed")
    if selected == "ipc":
        if not comm.reg_ipc(buf_id, buf, publish=False):
            raise RuntimeError("reg_ipc failed")

    # Send signals out of order: 30, 10, 20
    tags = [30, 10, 20]
    for tag in tags:
        comm.signal(1, tag)
        print(f"[sender] signaled tag={tag}")

    comm.unreg_rdma(buf_id)
    if selected == "ipc":
        comm.unreg_ipc(buf_id)
    print("[sender] done")


def run_receiver() -> None:
    rank = 1
    world = 2
    exchanger_port = env_int("EXCHANGER_PORT", 29611)
    gpu_id = env_int("LOCAL_RANK", 0)

    comm = p2p.Communicator(
        gpu_id=gpu_id,
        rank=rank,
        world_size=world,
        exchanger_ip="127.0.0.1",
        exchanger_port=exchanger_port,
        transport=os.getenv("UK_P2P_TRANSPORT", "auto"),
    )

    if not comm.accept_peer(0):
        raise RuntimeError("accept_peer(0) failed")
    selected = comm.peer_transport(0)
    print(f"[receiver] transport: {selected}")

    # Register a buffer to establish wait path
    buf = torch.empty(16, device="cuda", dtype=torch.float32)
    buf_id = 200
    if not comm.reg_rdma(buf_id, buf, publish=False):
        raise RuntimeError("reg_rdma failed")
    if selected == "ipc":
        if not comm.reg_ipc(buf_id, buf, publish=False):
            raise RuntimeError("reg_ipc failed")

    expected = {10, 20, 30}
    received = set()
    deadline = time.time() + 30  # 30 second timeout

    # Register async waits for each expected tag.
    rids = []
    rids_to_tags = {}
    for tag in expected:
        rid = comm.wait_signal_async(0, tag)
        if rid == 0:
            raise RuntimeError(f"wait_signal_async(0, {tag}) returned 0")
        rids.append(rid)
        rids_to_tags[rid] = tag

    while received != expected:
        if time.time() > deadline:
            missing = expected - received
            raise RuntimeError(f"Timeout waiting for tags {missing}")
        done = comm.poll(rids)
        for rid in done:
            tag = rids_to_tags.get(rid)
            if tag:
                print(f"[receiver] got tag={tag}")
                received.add(tag)

    comm.unreg_rdma(buf_id)
    if selected == "ipc":
        comm.unreg_ipc(buf_id)

    print(f"[receiver] all tags received: {sorted(received)}")
    print("[receiver] test PASSED")


def main() -> None:
    rank = env_int("RANK", 0)
    if rank == 0:
        run_sender()
    else:
        run_receiver()


if __name__ == "__main__":
    main()
