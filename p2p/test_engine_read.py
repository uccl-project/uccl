#!/usr/bin/env python3
"""
Local unit-test for UCCL P2P Engine — server pulls data with RDMA READ
"""

from __future__ import annotations
import sys, os, time, socket, struct, multiprocessing
from typing import Tuple

try:
    from uccl import p2p
except ImportError as e:
    sys.stderr.write(f"Failed to import p2p: {e}\n"
                     "Build the pybind11 extension first (make)\n")
    sys.exit(1)

import torch

def parse_metadata(meta: bytes) -> Tuple[str, int, int]:
    if len(meta) == 10:                               # IPv4
        ip_b, port_b, gpu_b = meta[:4], meta[4:6], meta[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_b)
    elif len(meta) == 22:                             # IPv6
        ip_b, port_b, gpu_b = meta[:16], meta[16:18], meta[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_b)
    else:
        raise ValueError(f"Unexpected metadata length {len(meta)}")
    port = struct.unpack("!H", port_b)[0]
    gpu  = struct.unpack("i", gpu_b)[0]
    return ip, port, gpu

def test_local():
    print("Running RDMA-READ local test")

    meta_q = multiprocessing.Queue()

    def server_proc(q):
        meta = q.get(timeout=5)
        ip, port, r_gpu = parse_metadata(meta)

        ep = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port); assert ok
        print(f"[Server] connected, conn_id={conn_id}")
        
        tensor = torch.zeros(1024, dtype=torch.float32)
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel()*4); assert ok

        print("[Server] Waiting for client to post receive descriptor …")
        ok = ep.read(conn_id, mr_id, tensor.data_ptr(), tensor.numel()*4)
        assert ok, "read failed"

        assert torch.allclose(tensor, torch.ones_like(tensor))
        print("✓ Server read data correctly")

    def client_proc(q):
        ep = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        q.put(bytes(ep.get_endpoint_metadata()))

        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "accept failed"
        print(f"[Client] accepted conn_id={conn_id} from {r_ip}, GPU{r_gpu}")
        
        tensor = torch.ones(1024, dtype=torch.float32)
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel()*4); assert ok

        # Advertise buffer so server can READ it
        print("[Client] Posting receive descriptor …")
        ok, recv_sz = ep.recv(conn_id, mr_id, tensor.data_ptr(),
                              max_size=tensor.numel()*4)
        assert ok and recv_sz == tensor.numel()*4
        print("✓ Client posted receive descriptor")

    # run the two processes
    srv = multiprocessing.Process(target=server_proc, args=(meta_q,))
    cli = multiprocessing.Process(target=client_proc,  args=(meta_q,))
    srv.start(); time.sleep(1); cli.start()
    srv.join(); cli.join()
    print("Local RDMA-READ test passed\n")

if __name__ == "__main__":
    try:
        test_local()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating…")
        sys.exit(1)