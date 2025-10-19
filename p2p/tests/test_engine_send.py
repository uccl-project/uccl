#!/usr/bin/env python3
"""
Test script for the UCCL P2P Engine pybind11 extension
"""

import sys
import os
import numpy as np
import multiprocessing
import socket
import struct
import time
import torch
from typing import Tuple

try:
    from uccl import p2p
    from uccl.utils import get_tensor_id_by_tensor, create_tensor

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)


def parse_endpoint_meta(meta: bytes) -> Tuple[str, int, int]:
    """Return (ip, port, remote_gpu_idx)."""
    if len(meta) == 10:  # IPv4
        ip_b, port_b, gpu_b = meta[:4], meta[4:6], meta[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_b)
    elif len(meta) == 22:  # IPv6
        ip_b, port_b, gpu_b = meta[:16], meta[16:18], meta[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_b)
    else:
        raise ValueError(f"Unexpected endpoint-metadata length {len(meta)}")
    port = struct.unpack("!H", port_b)[0]
    gpu = struct.unpack("i", gpu_b)[0]
    return ip, port, gpu


def test_local():
    """Test the UCCL P2P Engine"""
    print("Running UCCL P2P Engine local test...")
    meta_parent, meta_child = multiprocessing.Pipe()

    def server_process(meta_q):
        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        meta_q.send(bytes(engine.get_metadata()))

        success, remote_ip_addr, remote_gpu_idx, conn_id = engine.accept()
        assert success
        print(
            f"Server accepted connection from {remote_ip_addr}, GPU {remote_gpu_idx}, conn_id={conn_id}"
        )

        tensor, t_id = create_tensor((1024,), dtype=torch.float32)
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert tensor.is_contiguous()
        assert mr_id == t_id

        success = engine.recv(
            conn_id, mr_id, tensor.data_ptr(), size=tensor.numel() * 8
        )
        assert success

        assert tensor.allclose(torch.ones(1024, dtype=torch.float32))
        return True

    def client_process(meta_q):
        engine = p2p.Endpoint(local_gpu_idx=1, num_cpus=4)
        ep_meta = meta_q.recv()

        ip, r_port, r_gpu = parse_endpoint_meta(ep_meta)
        success, conn_id = engine.connect(
            remote_ip_addr=ip, remote_gpu_idx=r_gpu, remote_port=r_port
        )
        assert success
        print(f"Client connected successfully: conn_id={conn_id}")

        tensor, t_id = create_tensor((1024,), dtype=torch.float32)
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert tensor.is_contiguous()
        assert mr_id == t_id

        success = engine.send(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4)
        assert success

        return True

    server = multiprocessing.Process(target=server_process, args=(meta_parent,))
    server.start()
    time.sleep(1)

    client = multiprocessing.Process(target=client_process, args=(meta_child,))
    client.start()

    try:
        server.join()
        client.join()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, terminating processes...")
        server.terminate()
        client.terminate()
        server.join()
        client.join()
        raise


def main():
    """Run all tests"""
    print("Running UCCL P2P Engine tests...")

    test_local()

    print("\n=== All UCCL P2P Engine tests completed! ===")


if __name__ == "__main__":
    main()
