#!/usr/bin/env python3
"""
CXI payload check for one-sided READ and WRITE over the target lanes.

A CXI message above 512 KiB is split into pieces that go to the target's
lanes in turn. The split uses only the target's (acceptor's) lane count;
the initiator always posts on its lane 0. After each transfer every byte is
compared, so a missing or misaddressed piece fails the test:
  - source: a uint32 counter plus a salt that changes per transfer;
  - destination payload starts as 0xA5;
  - a 4 KiB guard (0x5A) before and after the payload, in the same MR, must
    not change.

Run the same revision on two CXI nodes; set UCCL_CXI_NUM_LANES (1 or 2,
default 2) on each side:
    python3 test_cxi_payload.py --role acceptor
    python3 test_cxi_payload.py --role initiator --peer-ip <acceptor>
"""

from __future__ import annotations
import argparse
import ctypes
import os
import socket
import struct
import sys
import time

import numpy as np

os.environ.setdefault("UCCL_P2P_TRANSPORT", "cxi")

from uccl import p2p
from native_runtime import Rt

GUARD = 4096
# Piece counts are for a 2-lane target; a 1-lane target never splits.
SIZES = [
    1,  # smallest message
    7,  # odd size, not a multiple of 4 bytes
    64,
    255,
    4096,  # one page
    4097,  # one page + 1 byte
    524288,  # 512 KiB: the largest size that is not split
    524289,  # 2 pieces, the last one shorter
    528385,  # 512 KiB + 4 KiB + 1: 2 pieces, the last one shorter
    1048577,  # 1 MiB + 1: 3 pieces
    7340035,  # 15 pieces, the last one shorter
    8388609,  # 8 MiB + 1: 16 pieces (the maximum), the last one shorter
    67112959,  # 64 MiB + 4095: 16 pieces of about 4 MiB
]
REPS = 2  # the first lane changes per request, so each size starts on each lane


def send_blob(sock, blob: bytes) -> None:
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def recv_blob(sock) -> bytes:
    def exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("rendezvous socket closed")
            buf += chunk
        return buf

    return exact(struct.unpack(">I", exact(4))[0])


def region_bytes(size: int, salt: int | None) -> np.ndarray:
    """Guards 0x5A; payload = the source pattern, or 0xA5 if salt is None."""
    # All 0x5A; the payload is overwritten below, so only the guards keep it.
    host = np.full(GUARD + size + GUARD, 0x5A, dtype=np.uint8)
    if salt is None:
        host[GUARD : GUARD + size] = 0xA5
    else:
        words = np.arange(salt, salt + size // 4 + 1, dtype=np.uint32)
        host[GUARD : GUARD + size] = words.view(np.uint8)[:size]
    return host


class Region:
    """One MR: guard, payload, guard."""

    def __init__(self, rt, ep, size: int):
        self.rt, self.ep, self.total = rt, ep, GUARD + size + GUARD
        self.base = rt.malloc(self.total)
        ok, self.mr_id = ep.reg(self.base.value, self.total)
        assert ok, "reg failed"
        self.ptr = self.base.value + GUARD

    def put(self, host: np.ndarray) -> None:
        self.rt.memcpy_h2d(self.base, host.ctypes.data_as(ctypes.c_void_p), self.total)

    def get(self) -> np.ndarray:
        host = np.empty(self.total, dtype=np.uint8)
        self.rt.memcpy_d2h(host.ctypes.data_as(ctypes.c_void_p), self.base, self.total)
        return host

    def free(self) -> None:
        assert self.ep.dereg(self.mr_id), "dereg failed"
        self.rt.free(self.base)


def diff(got: np.ndarray, want: np.ndarray) -> str:
    bad = np.flatnonzero(got != want)
    if not bad.size:
        return ""
    return f"{bad.size} bad bytes at region offsets {bad[0]}..{bad[-1]} (payload at {GUARD})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["acceptor", "initiator"], required=True)
    ap.add_argument("--peer-ip")
    ap.add_argument("--rendezvous-port", type=int, default=29510)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--modes", default="write,read")
    args = ap.parse_args()
    acceptor = args.role == "acceptor"
    if not acceptor and not args.peer_ip:
        ap.error("--peer-ip is required for the initiator")

    rt = Rt()
    rt.set_device(args.gpu)
    if acceptor:
        with socket.create_server(("0.0.0.0", args.rendezvous_port)) as lsock:
            sock, _ = lsock.accept()
        ep = p2p.Endpoint(local_gpu_idx=args.gpu)
        send_blob(sock, bytes(ep.get_metadata()))
        assert ep.accept()[0], "accept failed"
    else:
        deadline = time.monotonic() + 60
        while True:
            try:
                sock = socket.create_connection((args.peer_ip, args.rendezvous_port))
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(recv_blob(sock))
        ep = p2p.Endpoint(local_gpu_idx=args.gpu)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, "connect failed"

    # Both sides walk the same (mode, size, rep) list.
    failures = 0
    for mode in args.modes.split(","):
        for size in SIZES:
            region = Region(rt, ep, size)
            if acceptor:
                ok, fifo = ep.advertise(region.mr_id, region.ptr, size)
                assert ok, "advertise failed"
                send_blob(sock, bytes(fifo))
            else:
                fifo = recv_blob(sock)
            for rep in range(REPS):
                salt = (size * REPS + rep) * 2 + (mode == "read")
                src = region_bytes(size, salt)

                # write: the initiator is the source; read: the acceptor is.
                is_src = acceptor == (mode == "read")
                region.put(src if is_src else region_bytes(size, None))

                if acceptor:
                    send_blob(sock, b"ready")
                    recv_blob(sock)  # transfer done
                    send_blob(sock, b"" if is_src else diff(region.get(), src).encode())
                    continue
                recv_blob(sock)  # acceptor ready

                op = ep.write if mode == "write" else ep.read
                assert op(
                    conn_id, region.mr_id, region.ptr, size, fifo
                ), f"{mode} failed"

                err = "" if is_src else diff(region.get(), src)
                send_blob(sock, b"done")
                err = err or recv_blob(sock).decode()
                print(
                    f"{'FAIL' if err else 'PASS'} {mode:5s} size {size:>9} rep {rep} {err}"
                )

                failures += bool(err)
            region.free()
    if not acceptor:
        print(f"RESULT failures={failures}")
    sock.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
