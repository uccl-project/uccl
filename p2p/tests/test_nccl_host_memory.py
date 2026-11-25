#!/usr/bin/env python3
"""Check NCCL one-sided reads/writes with pageable host and GPU buffers.

Run once on each node with UCCL_P2P_TRANSPORT=nccl and UCCL_P2P_DISABLE_IPC=1:
  python test_nccl_host_memory.py --role acceptor --port 24000
  python test_nccl_host_memory.py --role initiator --peer-ip <host> --port 24000
"""

import argparse
import ctypes
import os
import socket
import time

os.environ["UCCL_P2P_TRANSPORT"] = "nccl"
os.environ["UCCL_P2P_DISABLE_IPC"] = "1"

from test_cross_node_rdma_native import Rt, recv_blob, send_blob
from uccl import p2p

SIZE = 65536
CASES = [
    (mode, local, remote)
    for mode in ("read", "write")
    for local in ("host", "gpu")
    for remote in ("host", "gpu")
]


class Buffer:
    def __init__(self, runtime, kind, value):
        self.runtime = runtime
        self.kind = kind
        self.host = ctypes.create_string_buffer(bytes([value]) * SIZE, SIZE)
        self.ptr = (
            runtime.malloc(SIZE)
            if kind == "gpu"
            else ctypes.c_void_p(ctypes.addressof(self.host))
        )
        if kind == "gpu":
            runtime.memcpy_h2d(self.ptr, self.host, SIZE)

    def check(self, value):
        if self.kind == "gpu":
            self.runtime.memcpy_d2h(self.host, self.ptr, SIZE)
        assert self.host.raw == bytes([value]) * SIZE, "Transfer data mismatch"

    def close(self):
        if self.kind == "gpu":
            self.runtime.free(self.ptr)


def connect(peer, port):
    deadline = time.monotonic() + 30
    while True:
        try:
            return socket.create_connection((peer, port), timeout=30)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def run(args):
    runtime = Rt()
    runtime.set_device(args.gpu)
    endpoint = p2p.Endpoint(local_gpu_idx=args.gpu)
    acceptor = args.role == "acceptor"
    if acceptor:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("", args.port))
            listener.listen(1)
            sock, _ = listener.accept()
        sock.settimeout(30)
    else:
        sock = connect(args.peer_ip, args.port)
    with sock:
        if acceptor:
            send_blob(sock, bytes(endpoint.get_metadata()))
            ok, _, _, conn = endpoint.accept()
        else:
            ip, port, gpu = p2p.Endpoint.parse_metadata(recv_blob(sock))
            ok, conn = endpoint.connect(ip, gpu, remote_port=port)
        assert ok, "NCCL connection failed"
        for index, (mode, local, remote) in enumerate(CASES):
            value = index + 1
            source = (mode == "read") == acceptor
            buffer = Buffer(
                runtime, remote if acceptor else local, value if source else 0
            )
            ok, mr = endpoint.reg(buffer.ptr.value, SIZE)
            assert ok, "Buffer registration failed"
            try:
                if acceptor:
                    ok, metadata = endpoint.advertise(mr, buffer.ptr.value, SIZE)
                    assert ok
                    send_blob(sock, bytes(metadata))
                    assert recv_blob(sock) == b"done"
                    # The initiator's completion must include any host copyback.
                    buffer.check(value)
                    send_blob(sock, b"checked")
                else:
                    metadata = recv_blob(sock)
                    operation = endpoint.read if mode == "read" else endpoint.write
                    assert operation(conn, mr, buffer.ptr.value, SIZE, metadata)
                    buffer.check(value)
                    send_blob(sock, b"done")
                    assert recv_blob(sock) == b"checked"
                print(f"PASS {mode}: local={local}, remote={remote}", flush=True)
            finally:
                endpoint.dereg(mr)
                buffer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["acceptor", "initiator"], required=True)
    parser.add_argument("--peer-ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24000)
    parser.add_argument("--gpu", type=int, default=0)
    run(parser.parse_args())
