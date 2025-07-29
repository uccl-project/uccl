#!/usr/bin/env python3
"""
Test script for the UCCL P2P Engine
using NVLink for inter-process communication.

Run with:
  OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 test_engine_nvlink.py
or:
  python -m torch.distributed.run --nproc_per_node=2 test_engine_nvlink.py
"""

import sys
import os
import time
import struct
import socket
import numpy as np

import torch
import torch.distributed as dist

try:
    from uccl import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)


def parse_metadata(metadata: bytes):
    if len(metadata) == 10:
        # IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_b, port_b, gpu_b = metadata[:4], metadata[4:6], metadata[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_b)
    elif len(metadata) == 22:
        # IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_b, port_b, gpu_b = metadata[:16], metadata[16:18], metadata[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_b)
    else:
        raise ValueError(f"Unexpected metadata length: {len(metadata)}")

    port = struct.unpack("!H", port_b)[0]
    remote_gpu_idx = struct.unpack("i", gpu_b)[0]
    return ip, port, remote_gpu_idx


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
    mv = memoryview(bytearray(payload))  # copies once, writable
    buf = torch.frombuffer(mv, dtype=torch.uint8)  # no warning
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    n = _recv_int(src)
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def test_local_dist():
    """Two-process local test: rank 0 = server, rank 1 = client."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if rank == 0:
        print("Running test_local (server)…")

        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        metadata = engine.get_endpoint_metadata()
        ip, port, remote_gpu_idx = parse_metadata(metadata)
        print(f"[server] Parsed IP: {ip}")
        print(f"[server] Parsed Port: {port}")
        print(f"[server] Parsed Remote GPU Index: {remote_gpu_idx}")

        _send_bytes(bytes(metadata), dst=1)

        conn_id = _recv_int(src=1)
        print(f"[server] Received conn_id={conn_id} from client")

        tensor = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
        assert tensor.is_contiguous()
        mr_id = 0
        ok, fifo_blob = engine.advertise(
            conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4
        )
        assert ok and isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
        print("[server] Buffer exposed for RDMA READ")

        _send_bytes(bytes(fifo_blob), dst=1)

        success = _recv_int(src=1)
        assert success

        assert tensor.allclose(torch.ones(1024, dtype=torch.float32, device="cuda:0"))
        print("[server] Received correct data")

    else:
        print("Running test_local (client)…")

        metadata = _recv_bytes(src=0)
        ip, port, remote_gpu_idx = parse_metadata(metadata)
        print(
            f"[client] Parsed server IP: {ip}, port: {port}, remote_gpu_idx: {remote_gpu_idx}"
        )

        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        success, conn_id = engine.connect(metadata)
        assert success
        print(f"[client] Connected successfully: conn_id={conn_id}")
        _send_int(conn_id, dst=0)

        tensor = torch.ones(1024, dtype=torch.float32, device="cuda:0")
        assert tensor.is_contiguous()

        fifo_blob = _recv_bytes(src=0)
        print("[client] Received FIFO blob from server")
        assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64

        mr_id = 0
        success = engine.send(
            conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4, fifo_blob
        )
        assert success
        print("[client] Sent data")

        _send_int(success, dst=0)


def main():
    dist.init_process_group(backend="gloo")
    try:
        print(f"=== UCCL P2P test (rank {dist.get_rank()}/{dist.get_world_size()}) ===")
        test_local_dist()
        dist.barrier()
        if dist.get_rank() == 0:
            print("\n=== All UCCL P2P Engine tests completed! ===")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
