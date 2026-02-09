"""
Test script for UCCL P2P Engine local NVLink IPC path.

Run with:
torchrun --nproc_per_node=2 test_engine_nvlink.py
"""

import sys
import torch
import torch.distributed as dist

try:
    from uccl import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    sys.exit(1)


# Torch dist helper functions
def _send_int(value: int, dst: int):
    t = torch.tensor([int(value)], dtype=torch.uint64)
    dist.send(t, dst=dst)


def _recv_int(src: int) -> int:
    t = torch.empty(1, dtype=torch.uint64)
    dist.recv(t, src=src)
    return int(t.item())


def _send_bytes(payload: bytes, dst: int):
    n = len(payload)
    _send_int(n, dst)
    if n == 0:
        return
    buf = torch.frombuffer(memoryview(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    n = _recv_int(src)
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


# Main local IPC test
def test_local_ipc():
    """
    Two-process test:
      rank0 = server  → accept_local()
      rank1 = client  → connect_local()
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    # each rank binds its own visible GPU
    torch.cuda.set_device(0)

    ep = p2p.Endpoint(local_gpu_idx=rank, num_cpus=4)

    # Rank 0: server
    if rank == 0:
        ok, remote_gpu_idx, conn_id = ep.accept_local()
        assert ok
        print(f"[server] accepted from remote_gpu={remote_gpu_idx}, conn_id={conn_id}")

        # allocate GPU buffer
        tensor = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
        assert tensor.is_contiguous()

        # advertise fifo blob for client write_ipc
        ok, fifo_blob = ep.advertise_ipc(
            conn_id,
            tensor.data_ptr(),
            tensor.numel() * 4,
        )
        assert ok
        print("[server] advertised IPC fifo")

        _send_bytes(bytes(fifo_blob), dst=1)

        success = _recv_int(src=1)
        assert success

        assert tensor.allclose(torch.ones_like(tensor))
        print("[server] received correct data!")

    # Rank 1: client
    else:
        # connect to server
        ok, conn_id = ep.connect_local(remote_gpu_idx=0)
        assert ok
        print(f"[client] connected successfully: conn_id={conn_id}")

        fifo_blob = _recv_bytes(src=0)
        print("[client] received fifo blob")

        tensor = torch.ones(1024, dtype=torch.float32, device="cuda:0")
        assert tensor.is_contiguous()

        ok = ep.write_ipc(
            conn_id,
            tensor.data_ptr(),
            tensor.numel() * 4,
            fifo_blob,
        )
        assert ok
        print("[client] write_ipc done")

        # notify server
        _send_int(1, dst=0)


def main():
    dist.init_process_group("gloo")
    try:
        print(f"=== UCCL Local IPC Test (rank {dist.get_rank()}) ===")
        test_local_ipc()
        dist.barrier()
        if dist.get_rank() == 0:
            print("\nAll local IPC tests passed!")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
