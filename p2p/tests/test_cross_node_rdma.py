#!/usr/bin/env python3
"""
Cross-node test for UCCL P2P Engine — verifies RDMA-WRITE / RDMA-READ between
two physical machines using the one-sided metadata handshake.

Run on the "acceptor" machine (owns the buffer being written to / read from):
    python3 test_cross_node_rdma.py --role acceptor --mode write
    python3 test_cross_node_rdma.py --role acceptor --mode read

Run on the "initiator" machine (drives the RDMA op), pointing at the
acceptor's rendezvous IP:
    python3 test_cross_node_rdma.py --role initiator --peer-ip <acceptor_ip> --mode write
    python3 test_cross_node_rdma.py --role initiator --peer-ip <acceptor_ip> --mode read

The acceptor and initiator must agree on --mode. A plain TCP "rendezvous"
socket (separate from the RDMA path) is used only to exchange the small
p2p endpoint/buffer metadata blobs; the actual data transfer goes over
UCCL's RDMA path.
"""

from __future__ import annotations
import argparse
import socket
import struct
import sys
import time
from typing import Tuple

# You must first import torch before importing uccl for AMD GPUs
import torch

try:
    from uccl import p2p
except ImportError as e:
    sys.stderr.write(f"Failed to import p2p: {e}\n")
    raise

DEFAULT_RENDEZVOUS_PORT = 29500


def parse_endpoint_meta(meta: bytes) -> Tuple[str, int, str]:
    """Return (ip, port, remote_gpu_bdf)."""
    return p2p.Endpoint.parse_metadata(meta)


def send_blob(sock: socket.socket, blob: bytes) -> None:
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def recv_blob(sock: socket.socket) -> bytes:
    (length,) = struct.unpack(">I", recv_exact(sock, 4))
    return recv_exact(sock, length)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Rendezvous socket closed unexpectedly")
        buf += chunk
    return buf


def run_acceptor(mode: str, rendezvous_port: int, gpu_idx: int) -> None:
    print(f"[Acceptor] starting, mode={mode}, listening for rendezvous on 0.0.0.0:{rendezvous_port}")

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("0.0.0.0", rendezvous_port))
    lsock.listen(1)
    conn, addr = lsock.accept()
    print(f"[Acceptor] rendezvous connection from {addr}")

    ep = p2p.Endpoint(local_gpu_idx=gpu_idx)
    send_blob(conn, bytes(ep.get_metadata()))

    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "accept failed"
    print(f"[Acceptor] p2p accepted from {r_ip} (conn_id={conn_id})")

    tensor = torch.ones(1024, dtype=torch.float32, device=f"cuda:{gpu_idx}")
    torch.cuda.synchronize()
    ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert ok, "reg failed"

    ok, fifo_blob = ep.advertise(mr_id, tensor.data_ptr(), tensor.numel() * 4)
    assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
    send_blob(conn, bytes(fifo_blob))
    print(f"[Acceptor] buffer exposed for RDMA {mode.upper()}, waiting for completion")

    time.sleep(5)
    torch.cuda.synchronize()
    print("tensor[:8]:", tensor[:8])
    if mode == "write":
        assert torch.allclose(tensor, torch.full_like(tensor, 7.0))
        print("✓ Acceptor buffer correctly written by remote RDMA-WRITE")
    else:
        print("✓ Acceptor buffer exposed for remote RDMA-READ (no local mutation expected)")

    conn.close()
    lsock.close()


def run_initiator(mode: str, peer_ip: str, rendezvous_port: int, gpu_idx: int) -> None:
    print(f"[Initiator] connecting rendezvous to {peer_ip}:{rendezvous_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((peer_ip, rendezvous_port))

    ep_meta = recv_blob(sock)
    ip, port, r_gpu = parse_endpoint_meta(ep_meta)
    print(f"[Initiator] acceptor metadata: ip={ip} port={port} remote_gpu_bdf={r_gpu}")

    ep = p2p.Endpoint(local_gpu_idx=gpu_idx)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "connect failed"
    print(f"[Initiator] p2p connected (conn_id={conn_id})")

    # write: initiator pushes 7.0 into the acceptor's buffer (acceptor starts at 1.0)
    # read: initiator pulls the acceptor's buffer (1.0) into its own, starting from
    #       a different sentinel (0.0) so the assertion proves data actually moved
    value = 7.0 if mode == "write" else 0.0
    tensor = torch.full((1024,), value, dtype=torch.float32, device=f"cuda:{gpu_idx}")
    torch.cuda.synchronize()
    ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert ok, "reg failed"

    fifo_meta = recv_blob(sock)
    assert isinstance(fifo_meta, (bytes, bytearray)) and len(fifo_meta) == 64

    if mode == "write":
        ok = ep.write(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4, fifo_meta)
        assert ok, "write failed"
        print("✓ Initiator RDMA-WRITE completed")
    else:
        ok = ep.read(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4, fifo_meta)
        assert ok, "read failed"
        torch.cuda.synchronize()
        print("tensor[:8]:", tensor[:8])
        assert torch.allclose(tensor, torch.full_like(tensor, 1.0))
        print("✓ Initiator RDMA-READ completed and data verified")

    time.sleep(2)
    sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-node UCCL P2P RDMA test")
    parser.add_argument("--role", choices=["acceptor", "initiator"], required=True)
    parser.add_argument("--mode", choices=["write", "read"], required=True)
    parser.add_argument("--peer-ip", default=None, help="Required for --role initiator")
    parser.add_argument("--rendezvous-port", type=int, default=DEFAULT_RENDEZVOUS_PORT)
    parser.add_argument("--gpu", type=int, default=0, help="Local GPU index to use")
    args = parser.parse_args()

    if args.role == "initiator" and not args.peer_ip:
        parser.error("--peer-ip is required when --role initiator")

    if args.role == "acceptor":
        run_acceptor(args.mode, args.rendezvous_port, args.gpu)
    else:
        run_initiator(args.mode, args.peer_ip, args.rendezvous_port, args.gpu)

    print("Cross-node RDMA test passed\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating…")
        sys.exit(1)
