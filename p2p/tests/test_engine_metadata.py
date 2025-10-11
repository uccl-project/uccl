#!/usr/bin/env python3
"""
Test script for the UCCL P2P Engine pybind11 extension
"""

import sys
import os
import numpy as np
import multiprocessing
import time
import torch
import socket
import struct

try:
    from uccl import p2p
    from uccl.utils import get_tensor_id_by_tensor, create_tensor

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)


# parse_metadata is now provided by the C++ layer via p2p.Endpoint.parse_metadata()


def test_local():
    """Test the UCCL P2P Engine local send/recv functionality"""
    print("Running test_local...")

    meta_parent, meta_child = multiprocessing.Pipe()

    def server_process(q):
        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        metadata = engine.get_metadata()
        ip, port, remote_gpu_idx = p2p.Endpoint.parse_metadata(metadata)
        print(f"Parsed IP: {ip}")
        print(f"Parsed Port: {port}")
        print(f"Parsed Remote GPU Index: {remote_gpu_idx}")
        q.send(bytes(metadata))  # ensure it's serialized as bytes

        success, remote_ip_addr, remote_gpu_idx, conn_id = engine.accept()
        assert success
        print(
            f"Server accepted connection from {remote_ip_addr}, GPU {remote_gpu_idx}, conn_id={conn_id}"
        )

        tensor, t_id = create_tensor((1024,), dtype=torch.float32, device=f"cpu")
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert tensor.is_contiguous()
        assert mr_id == t_id

        success = engine.recv(
            conn_id, mr_id, tensor.data_ptr(), size=tensor.numel() * 8
        )
        assert success

        assert tensor.allclose(torch.ones(1024, dtype=torch.float32))
        print("✓ Server received correct data")

    def client_process(q):
        metadata = q.recv()
        ip, port, remote_gpu_idx = p2p.Endpoint.parse_metadata(metadata)
        print(
            f"Client parsed server IP: {ip}, port: {port}, remote_gpu_idx: {remote_gpu_idx}"
        )

        engine = p2p.Endpoint(local_gpu_idx=1, num_cpus=4)
        success, conn_id = engine.connect(
            remote_ip_addr=ip, remote_gpu_idx=remote_gpu_idx, remote_port=port
        )
        assert success
        print(f"Client connected successfully: conn_id={conn_id}")

        tensor, t_id = create_tensor((1024,), dtype=torch.float32, device=f"cpu")
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert tensor.is_contiguous()
        assert mr_id == t_id

        success = engine.send(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4)
        assert success
        print("✓ Client sent data")

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
    print("=== Running UCCL P2P Engine tests ===\n")
    test_local()
    print("\n=== All UCCL P2P Engine tests completed! ===")


if __name__ == "__main__":
    main()
