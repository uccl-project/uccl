#!/usr/bin/env python3
"""Finite cross-node CUDA-compatible device-buffer qualification.

The host-staged path is explicit application staging, not an engine fallback.
See P800_TOPOLOGY_VALIDATION.md for setup, timing definitions and limitations.
Use an outer process timeout: native endpoint calls can block independently of
Python socket timeouts. No torch dependency and no runtime calls on import.
"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import struct
import time

GUARD_BYTES = 256
PROTOCOL_VERSION = 5


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def payload(size, sequence):
    # Use an extendable stream, not a repeated digest: moving or duplicating
    # whole digest-sized blocks must change the expected destination bytes.
    seed = f"uccl-p800-payload-v2:{size}:{sequence}".encode()
    return hashlib.shake_256(seed).digest(size)


def guarded(data, sequence, role):
    seed = f"uccl-p800-guards-v3:{len(data)}:{sequence}:{role}".encode()
    guards = hashlib.shake_256(seed).digest(2 * GUARD_BYTES)
    return guards[:GUARD_BYTES] + data + guards[GUARD_BYTES:]


def verify(actual, expected):
    require(actual == expected, "device buffer mismatch (full allocation checked)")
    return hashlib.sha256(actual).hexdigest()


def send(sock, data):
    require(len(data) <= 65536, "control frame too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive(sock):
    def exact(count):
        result = bytearray()
        while len(result) < count:
            chunk = sock.recv(count - len(result))
            require(bool(chunk), "peer closed before completion")
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    require(size <= 65536, "control frame too large")
    return exact(size)


class Runtime:
    def __init__(self, library, gpu):
        self.lib = ctypes.CDLL(library)
        signatures = {
            "cudaSetDevice": [ctypes.c_int],
            "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cudaFree": [ctypes.c_void_p],
            "cudaDeviceSynchronize": [],
            "cudaMemcpy": [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
        }
        for name, types in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes, func.restype = types, ctypes.c_int
        self.check(self.lib.cudaSetDevice(gpu), "cudaSetDevice")

    @staticmethod
    def check(code, name):
        require(code == 0, f"{name} failed: runtime error {code}")

    def alloc(self, size):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), size), "cudaMalloc")
        return ptr

    def copy(self, dst, src, size, direction):
        self.check(self.lib.cudaMemcpy(dst, src, size, direction), "cudaMemcpy")
        if direction == 1:
            # Pageable H2D may return after host staging, before device completion.
            # Finish initialization before RDMA and include H2D completion in timing.
            self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def read(self, ptr, size):
        host = ctypes.create_string_buffer(size)
        self.copy(host, ptr, size, 2)
        return host.raw

    def fill(self, ptr, data):
        host = ctypes.create_string_buffer(data, len(data))
        self.copy(ptr, host, len(data), 1)

    def free(self, ptr):
        self.check(self.lib.cudaFree(ptr), "cudaFree")


def run_pair(args, sock, rt, ep, p2p):
    plan = json.dumps(
        [
            PROTOCOL_VERSION,
            args.path,
            args.sizes,
            args.warmup,
            args.iterations,
            GUARD_BYTES,
        ],
        separators=(",", ":"),
    ).encode()
    send(sock, plan)
    require(receive(sock) == plan, "peer test plans differ")
    sender = args.role == "sender"
    if sender:
        ip, port, bdf = p2p.Endpoint.parse_metadata(receive(sock))
        ok, conn_id = ep.connect(ip, bdf, remote_port=port)
        require(ok, "endpoint connect failed")
    else:
        send(sock, bytes(ep.get_metadata()))
        require(ep.accept()[0], "endpoint accept failed")

    for size in args.sizes:
        allocation_size = size + 2 * GUARD_BYTES
        device = rt.alloc(allocation_size)
        device_payload = ctypes.c_void_p(device.value + GUARD_BYTES)
        # This is ordinary host memory registered by UCCL, not pinned CUDA memory.
        host = (
            ctypes.create_string_buffer(allocation_size)
            if args.path == "host-staged"
            else None
        )
        base = ctypes.addressof(host) if host is not None else device.value
        address = base + GUARD_BYTES
        ok, mr = ep.reg(base, allocation_size)
        require(ok, f"{args.path} registration failed; no automatic fallback")
        if sender:
            descriptor = receive(sock)
        else:
            ok, descriptor = ep.advertise(mr, address, size)
            require(ok, "advertise failed")
            send(sock, bytes(descriptor))
        require(len(descriptor) == 64, "invalid RDMA descriptor")
        transport_total = path_total = 0.0
        for sequence in range(args.warmup + args.iterations):
            expected = payload(size, sequence)
            source_expected = guarded(expected, sequence, "sender")
            destination_expected = guarded(expected, sequence, "receiver")
            if sender:
                rt.fill(device, source_expected)
                if host is not None:
                    source_poison = guarded(
                        bytes(b ^ 255 for b in expected), sequence, "sender"
                    )
                    ctypes.memmove(host, source_poison, allocation_size)
                require(receive(sock) == b"ready", "receiver not ready")
                start = time.perf_counter()
                if host is not None:
                    rt.copy(address, device_payload, size, 2)
                transfer_start = time.perf_counter()
                require(
                    ep.write(conn_id, mr, address, size, descriptor),
                    "RDMA write failed",
                )
                transfer_seconds = time.perf_counter() - transfer_start
                local_seconds = time.perf_counter() - start
                send(sock, b"written")
                receipt = json.loads(receive(sock))
                require(receipt["sequence"] == sequence, "stale completion")
                require(
                    receipt["sha256"]
                    == hashlib.sha256(destination_expected).hexdigest(),
                    "peer verification failed",
                )
                verify(rt.read(device, allocation_size), source_expected)
                if host is not None:
                    verify(host.raw, source_expected)
                if sequence >= args.warmup:
                    transport_total += transfer_seconds
                    path_total += local_seconds + receipt["staging_seconds"]
            else:
                # Poison all destination bytes before every transfer.
                poisoned = guarded(
                    bytes(b ^ 255 for b in expected), sequence, "receiver"
                )
                rt.fill(device, poisoned)
                if host is not None:
                    ctypes.memmove(host, poisoned, allocation_size)
                send(sock, b"ready")
                require(receive(sock) == b"written", "missing write completion")
                if host is not None:
                    # Check staging guards before copying only the payload.
                    verify(host.raw, destination_expected)
                start = time.perf_counter()
                if host is not None:
                    rt.copy(device_payload, address, size, 1)
                staging_seconds = (
                    time.perf_counter() - start if host is not None else 0.0
                )
                digest = verify(rt.read(device, allocation_size), destination_expected)
                send(
                    sock,
                    json.dumps(
                        dict(
                            sequence=sequence,
                            sha256=digest,
                            staging_seconds=staging_seconds,
                        )
                    ).encode(),
                )
        # No deregistration/free until both peers finish the case.
        send(sock, b"complete")
        require(receive(sock) == b"complete", "missing case completion")
        require(ep.dereg(mr), "deregistration failed")
        rt.free(device)
        send(sock, b"released")
        require(receive(sock) == b"released", "peer cleanup incomplete")
        if sender:
            measured_bytes = size * args.iterations
            print(
                json.dumps(
                    dict(
                        status="PASS",
                        protocol_version=PROTOCOL_VERSION,
                        transport="rdma",
                        compression="none",
                        path=args.path,
                        bytes=size,
                        allocation_bytes=allocation_size,
                        guard_bytes_per_side=GUARD_BYTES,
                        verified_transfers=args.warmup + args.iterations,
                        measured_transfers=args.iterations,
                        transport_seconds=transport_total,
                        device_path_seconds=path_total,
                        transport_GBps=measured_bytes / transport_total / 1e9,
                        device_path_GBps=measured_bytes / path_total / 1e9,
                    )
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--peer", help="Receiver's routable address")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--runtime-lib", required=True, help="Same libcudart used to link uccl.p2p"
    )
    parser.add_argument("--path", choices=["direct", "host-staged"], required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 1048576, 16777216]
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if (
        args.iterations <= 0
        or args.warmup < 0
        or args.timeout <= 0
        or args.gpu < 0
        or not all(0 < n <= 64 * 1024 * 1024 for n in args.sizes)
    ):
        parser.error("invalid count, timeout, GPU ordinal, or size (maximum 64 MiB)")
    if args.role == "sender" and not args.peer:
        parser.error("sender requires --peer")
    if os.environ.get("UCCL_P2P_TRANSPORT", "rdma") != "rdma":
        parser.error("this test requires UCCL_P2P_TRANSPORT=rdma")
    # DietGPU builds also enable compression for unknown nonempty values.
    # Reject inherited settings before native initialization can allocate buffers.
    if os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "").lower() not in (
        "",
        "none",
        "off",
        "0",
    ):
        parser.error(
            "this test requires uncompressed RDMA; set UCCL_P2P_COMPRESS_STRATEGY=none"
        )
    from uccl import p2p

    rt = Runtime(args.runtime_lib, args.gpu)
    ep = p2p.Endpoint(local_gpu_idx=args.gpu)
    if args.role == "receiver":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(args.timeout)
            listener.bind(("0.0.0.0", args.port))
            listener.listen(1)
            sock, _ = listener.accept()
    else:
        sock = socket.create_connection((args.peer, args.port), timeout=args.timeout)
    with sock:
        sock.settimeout(args.timeout)
        run_pair(args, sock, rt, ep, p2p)


if __name__ == "__main__":
    main()
