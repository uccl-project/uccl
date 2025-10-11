#!/usr/bin/env python3
"""
Local unit-test for UCCL P2P Engine — server writes data with RDMA-WRITE
using the new one-sided metadata handshake.
"""

from __future__ import annotations
import sys, os, time, socket, struct, multiprocessing
from typing import Tuple

# You must first import torch before importing uccl for AMD GPUs
import torch

os.environ["UCCL_RCMODE"] = "1"

try:
    from uccl import p2p
    from uccl.utils import get_tensor_id_by_tensor, create_tensor
except ImportError as e:
    sys.stderr.write(f"Failed to import p2p: {e}\n")
    raise


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
    print("Running RDMA-WRITE local test")
    meta_parent, meta_child = multiprocessing.Pipe()
    fifo_parent, fifo_child = multiprocessing.Pipe()

    def server_proc(ep_meta_q, fifo_meta_q):
        ep_meta = ep_meta_q.recv()
        ip, port, r_gpu = parse_endpoint_meta(ep_meta)

        ep = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, "connect failed"
        print(f"[Server] connected (conn_id={conn_id})")

        tensor, t_id = create_tensor((1024,), dtype=torch.float32, device="cuda:0")
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert mr_id == t_id

        fifo_meta = fifo_meta_q.recv()
        assert isinstance(fifo_meta, (bytes, bytearray)) and len(fifo_meta) == 64

        ok = ep.write(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4, fifo_meta)
        assert ok, "write failed"

        torch.cuda.synchronize()
        print("tensor[:8]:", tensor[:8])
        assert torch.allclose(tensor, torch.ones_like(tensor))
        print("✓ Server write data correctly")

    def client_proc(ep_meta_q, fifo_meta_q):
        ep = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
        ep_meta_q.send(bytes(ep.get_metadata()))

        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "accept failed"
        print(f"[Client] accepted (conn_id={conn_id})")

        tensor, t_id = create_tensor((1024,), dtype=torch.float32, device="cuda:0")
        print("data pointer hex", hex(tensor.data_ptr()))
        torch.cuda.synchronize()
        mr_id = get_tensor_id_by_tensor(tensor=tensor)
        assert mr_id == t_id

        time.sleep(0.1)
        print("advertise data pointer hex", hex(tensor.data_ptr()))
        ok, fifo_blob = ep.advertise(
            conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4
        )
        assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
        print("Buffer exposed for RDMA WRITE")

        fifo_meta_q.send(bytes(fifo_blob))
        time.sleep(1)

    srv = multiprocessing.Process(target=server_proc, args=(meta_parent, fifo_parent))
    cli = multiprocessing.Process(target=client_proc, args=(meta_child, fifo_child))
    srv.start()
    time.sleep(1)
    cli.start()
    srv.join()
    cli.join()
    print("Local RDMA-WRITE test passed\n")


if __name__ == "__main__":
    try:
        test_local()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating…")
        sys.exit(1)
