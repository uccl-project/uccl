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

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)


def test_local():
    """Test the UCCL P2P Engine"""
    print("Running UCCL P2P Engine local test...")

    def server_process():
        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        success, remote_ip_addr, remote_gpu_idx, conn_id = engine.accept()
        assert success
        print(
            f"Server accepted connection from {remote_ip_addr}, GPU {remote_gpu_idx}, conn_id={conn_id}"
        )

        tensor = torch.zeros(1024, dtype=torch.float32)
        assert tensor.is_contiguous()

        success, mr_id = engine.reg(tensor.data_ptr(), tensor.numel() * 4)
        assert success

        success, recv_size = engine.recv(
            conn_id, mr_id, tensor.data_ptr(), max_size=tensor.numel() * 8
        )
        assert success
        assert recv_size == tensor.numel() * 4, f"recv_size={recv_size}"

        assert tensor.allclose(torch.ones(1024, dtype=torch.float32))
        return True

    def client_process():
        engine = uccl_p2p.Endpoint(local_gpu_idx=1, num_cpus=4)
        success, conn_id = engine.connect(
            remote_ip_addr="127.0.0.1", remote_gpu_idx=0
        )
        print(f"Client connected successfully: conn_id={conn_id}")

        tensor = torch.ones(1024, dtype=torch.float32)
        assert tensor.is_contiguous()

        success, mr_id = engine.reg(tensor.data_ptr(), tensor.numel() * 4)
        assert success

        success = engine.send(
            conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4
        )
        assert success

        return True

    server = multiprocessing.Process(target=server_process)
    server.start()
    time.sleep(1)

    client = multiprocessing.Process(target=client_process)
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
