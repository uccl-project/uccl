import os
import torch
import ukernel_p2p as p2p


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def run_server() -> None:
    rank = 0
    world = 2
    exchanger_port = env_int("EXCHANGER_PORT", 29610)
    gpu_id = env_int("LOCAL_RANK", 0)

    comm = p2p.Communicator(
        gpu_id=gpu_id,
        rank=rank,
        world_size=world,
        exchanger_ip="127.0.0.1",
        exchanger_port=exchanger_port,
        transport="auto",
    )

    if not comm.accept_peer(1):
        raise RuntimeError("accept_peer(1) failed")
    print(f"[rank {rank}] accepted client")

    recv_buffer_id = 100
    recv = torch.empty(16, device="cuda", dtype=torch.float32)
    recv_mr_id = comm.pin_tensor(recv)
    if not comm.publish_mr(1, recv_buffer_id, recv_mr_id):
        raise RuntimeError("publish_mr(recv) failed")

    comm.recv(1, recv)
    print(f"[rank {rank}] received: {recv}")

    if recv.sum() == 0:
        raise RuntimeError("No data received!")
    comm.unpin_tensor(recv)
    print(f"[rank {rank}] P2P server test passed!")


def run_client() -> None:
    rank = 1
    world = 2
    exchanger_port = env_int("EXCHANGER_PORT", 29610)
    gpu_id = env_int("LOCAL_RANK", 0)

    comm = p2p.Communicator(
        gpu_id=gpu_id,
        rank=rank,
        world_size=world,
        exchanger_ip="127.0.0.1",
        exchanger_port=exchanger_port,
        transport="auto",
    )

    if not comm.connect_peer(0):
        raise RuntimeError("connect_peer(0) failed")
    print(f"[rank {rank}] connected to server")

    recv_buffer_id = 100
    send = torch.arange(0, 16, device="cuda", dtype=torch.float32)
    comm.pin_tensor(send)
    comm.wait_mr(0, recv_buffer_id)

    comm.send_buffer(0, send, recv_buffer_id, remote_offset=0)
    comm.unpin_tensor(send)
    print(f"[rank {rank}] sent: {send}")
    print(f"[rank {rank}] P2P client test passed!")


def main() -> None:
    rank = env_int("RANK", 0)
    if rank == 0:
        run_server()
    else:
        run_client()


if __name__ == "__main__":
    main()
