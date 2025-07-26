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
    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)

def parse_metadata(metadata: bytes):
    if len(metadata) == 10:
        # IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:4]
        port_bytes = metadata[4:6]
        gpu_idx_bytes = metadata[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_bytes)
    elif len(metadata) == 22:
        # IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:16]
        port_bytes = metadata[16:18]
        gpu_idx_bytes = metadata[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_bytes)
    else:
        raise ValueError(f"Unexpected metadata length: {len(metadata)}")
    
    port = struct.unpack('!H', port_bytes)[0]
    remote_gpu_idx = struct.unpack('i', gpu_idx_bytes)[0]  # host byte order
    return ip, port, remote_gpu_idx

def test_local():
    """Test the UCCL P2P Engine local send/recv functionality"""
    print("Running test_local...")

    metadata_queue = multiprocessing.Queue()
    connection_id_queue = multiprocessing.Queue()

    def server_process(q):
        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        metadata = engine.get_endpoint_metadata()
        ip, port, remote_gpu_idx = parse_metadata(metadata)
        print(f"Parsed IP: {ip}")
        print(f"Parsed Port: {port}")
        print(f"Parsed Remote GPU Index: {remote_gpu_idx}")
        q.put(bytes(metadata))  # ensure it's serialized as bytes
        
        conn_id = connection_id_queue.get(timeout=5)  # wait for client to connect
        assert isinstance(conn_id, int)
        
        tensor = torch.zeros(1024, dtype=torch.float32, device='cuda:0')
        assert tensor.is_contiguous()

        success, mr_id = engine.reg(tensor.data_ptr(), tensor.numel() * 4)
        assert success

        ok, fifo_blob = engine.advertise(conn_id,
                                 mr_id,
                                 tensor.data_ptr(),
                                 tensor.numel() * 4)
        assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
        print("Buffer exposed for RDMA READ")

        q.put(bytes(fifo_blob))
        time.sleep(1)

        assert tensor.allclose(torch.ones(1024, dtype=torch.float32, device='cuda:0'))
        print("✓ Server received correct data")

    def client_process(q):
        metadata = q.get(timeout=5)
        ip, port, remote_gpu_idx = parse_metadata(metadata)
        print(f"Client parsed server IP: {ip}, port: {port}, remote_gpu_idx: {remote_gpu_idx}")

        engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        success, conn_id = engine.connect(metadata)
        assert success
        print(f"Client connected successfully: conn_id={conn_id}")
        connection_id_queue.put(conn_id)  # notify server of connection

        tensor = torch.ones(1024, dtype=torch.float32, device='cuda:0')
        assert tensor.is_contiguous()

        success, mr_id = engine.reg(tensor.data_ptr(), tensor.numel() * 4)
        assert success

        fifo_blob = q.get(timeout=10)
        print("Received FIFO blob from server")
        assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob)

        success = engine.send(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4, fifo_blob)
        assert success
        print("✓ Client sent data")

    server = multiprocessing.Process(target=server_process, args=(metadata_queue,))
    server.start()
    time.sleep(1)

    client = multiprocessing.Process(target=client_process, args=(metadata_queue,))
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