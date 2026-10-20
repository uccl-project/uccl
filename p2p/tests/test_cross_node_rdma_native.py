#!/usr/bin/env python3
"""
Cross-node test for UCCL P2P Engine — torch-free variant.

Buffers are allocated via ctypes against the platform GPU runtime, so torch is
never imported (avoids dual-runtime conflicts with a native-linked uccl.p2p).
Runtime selection comes from uccl.p2p.backend_info(). UCCL_GPU_RT_LIB can
optionally override the runtime library path in a test installation.

Run on the "acceptor" machine (owns the buffer being written to / read from):
    python3 test_cross_node_rdma_native.py --role acceptor --mode write
    python3 test_cross_node_rdma_native.py --role acceptor --mode read

Run on the "initiator" machine, pointing at the acceptor's rendezvous IP:
    python3 test_cross_node_rdma_native.py --role initiator --peer-ip <acceptor_ip> --mode write
    python3 test_cross_node_rdma_native.py --role initiator --peer-ip <acceptor_ip> --mode read
"""

from __future__ import annotations
import argparse
import ctypes
import socket
import struct
import sys
import time
from typing import Tuple

try:
    from uccl import p2p
except ImportError as e:
    sys.stderr.write(f"Failed to import p2p: {e}\n")
    raise

DEFAULT_RENDEZVOUS_PORT = 29500
NUM_FLOATS = 1024
NBYTES = NUM_FLOATS * 4


from native_runtime import Rt as NativeRuntime


class Rt(NativeRuntime):
    def alloc_filled(self, value: float) -> ctypes.c_void_p:
        host_buf = (ctypes.c_float * NUM_FLOATS)(*([value] * NUM_FLOATS))
        ptr = self.malloc(NBYTES)
        self.memcpy_h2d(ptr, host_buf, NBYTES)
        return ptr

    def read_first8(self, ptr: ctypes.c_void_p) -> list:
        host_buf = (ctypes.c_float * 8)()
        self.memcpy_d2h(host_buf, ptr, ctypes.sizeof(host_buf))
        return list(host_buf)


def parse_endpoint_meta(meta: bytes) -> Tuple[str, int, str]:
    return p2p.Endpoint.parse_metadata(meta)


def send_blob(sock: socket.socket, blob: bytes) -> None:
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Rendezvous socket closed unexpectedly")
        buf += chunk
    return buf


def recv_blob(sock: socket.socket) -> bytes:
    (length,) = struct.unpack(">I", recv_exact(sock, 4))
    return recv_exact(sock, length)


def run_acceptor(mode: str, rendezvous_port: int, gpu_idx: int) -> None:
    print(
        f"[Acceptor] starting, mode={mode}, listening for rendezvous on 0.0.0.0:{rendezvous_port}"
    )
    rt = Rt()
    rt.set_device(gpu_idx)

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

    dev_ptr = rt.alloc_filled(1.0)
    ok, mr_id = ep.reg(dev_ptr.value, NBYTES)
    assert ok, "reg failed"

    ok, fifo_blob = ep.advertise(mr_id, dev_ptr.value, NBYTES)
    assert isinstance(fifo_blob, (bytes, bytearray)) and len(fifo_blob) == 64
    send_blob(conn, bytes(fifo_blob))
    print(f"[Acceptor] buffer exposed for RDMA {mode.upper()}, waiting for completion")

    time.sleep(5)
    values = rt.read_first8(dev_ptr)
    print("buffer[:8]:", values)
    if mode == "write":
        assert all(abs(v - 7.0) < 1e-5 for v in values), values
        print("✓ Acceptor buffer correctly written by remote RDMA-WRITE")
    else:
        print(
            "✓ Acceptor buffer exposed for remote RDMA-READ (no local mutation expected)"
        )

    rt.free(dev_ptr)
    conn.close()
    lsock.close()


def run_initiator(mode: str, peer_ip: str, rendezvous_port: int, gpu_idx: int) -> None:
    print(f"[Initiator] connecting rendezvous to {peer_ip}:{rendezvous_port}")
    rt = Rt()
    rt.set_device(gpu_idx)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((peer_ip, rendezvous_port))

    ep_meta = recv_blob(sock)
    ip, port, r_gpu = parse_endpoint_meta(ep_meta)
    print(f"[Initiator] acceptor metadata: ip={ip} port={port} remote_gpu_bdf={r_gpu}")

    ep = p2p.Endpoint(local_gpu_idx=gpu_idx)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "connect failed"
    print(f"[Initiator] p2p connected (conn_id={conn_id})")

    # write: push 7.0 into the acceptor's buffer (acceptor starts at 1.0)
    # read: pull the acceptor's buffer (1.0) into a sentinel-initialized local
    #       buffer (0.0) so the assertion proves data actually moved
    value = 7.0 if mode == "write" else 0.0
    dev_ptr = rt.alloc_filled(value)
    ok, mr_id = ep.reg(dev_ptr.value, NBYTES)
    assert ok, "reg failed"

    fifo_meta = recv_blob(sock)
    assert isinstance(fifo_meta, (bytes, bytearray)) and len(fifo_meta) == 64

    if mode == "write":
        ok = ep.write(conn_id, mr_id, dev_ptr.value, NBYTES, fifo_meta)
        assert ok, "write failed"
        print("✓ Initiator RDMA-WRITE completed")
    else:
        ok = ep.read(conn_id, mr_id, dev_ptr.value, NBYTES, fifo_meta)
        assert ok, "read failed"
        values = rt.read_first8(dev_ptr)
        print("buffer[:8]:", values)
        assert all(abs(v - 1.0) < 1e-5 for v in values), values
        print("✓ Initiator RDMA-READ completed and data verified")

    time.sleep(2)
    rt.free(dev_ptr)
    sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-node UCCL P2P RDMA test (torch-free)"
    )
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
