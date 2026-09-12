#!/usr/bin/env python3
"""Two-node CUDA RDMA CC regression; see README_cc.md for invocation."""

import argparse
import base64
import ctypes as C
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class CudaBuffer:
    def __init__(self, gpu, sizes):
        self.lib = C.CDLL(os.environ.get("UCCL_TEST_CUDART", "libcudart.so"))
        signatures = {
            "cudaSetDevice": [C.c_int],
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t],
            "cudaMemset": [C.c_void_p, C.c_int, C.c_size_t],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaDeviceSynchronize": [],
            "cudaFree": [C.c_void_p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.call("cudaSetDevice", gpu)
        self.ptr = C.c_void_p()
        self.call("cudaMalloc", C.byref(self.ptr), sum(sizes))
        self.sizes = sizes
        self.ptrs = []
        offset = 0
        for size in sizes:
            self.ptrs.append(self.ptr.value + offset)
            offset += size

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        require(status == 0, f"{name} failed: {status}")

    @staticmethod
    def pattern(index, iteration):
        return 1 + (index * 37 + iteration * 19) % 255

    def fill(self, iteration, source):
        for i, (ptr, size) in enumerate(zip(self.ptrs, self.sizes)):
            value = self.pattern(i, iteration) if source else 0
            self.call("cudaMemset", ptr, value, size)
        self.call("cudaDeviceSynchronize")

    def verify(self, iteration):
        # Read and compare every byte, with bounded host memory usage.
        host = C.create_string_buffer(1024 * 1024)
        for i, (ptr, size) in enumerate(zip(self.ptrs, self.sizes)):
            value = self.pattern(i, iteration)
            for offset in range(0, size, len(host)):
                length = min(len(host), size - offset)
                self.call("cudaMemcpy", host, ptr + offset, length, 2)
                require(
                    host.raw[:length] == bytes([value]) * length,
                    f"Data mismatch: iov={i}, offset={offset}, iteration={iteration}",
                )


def send(sock, value):
    blob = json.dumps(value).encode()
    sock.sendall(struct.pack("!I", len(blob)) + blob)


def receive(sock):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(size - len(result))
            require(chunk, "Peer closed the control connection")
            result.extend(chunk)
        return result

    size = struct.unpack("!I", exact(4))[0]
    require(size <= 1024 * 1024, "Control message too large")
    return json.loads(exact(size))


def connect_control(args):
    if args.role == "acceptor":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(args.timeout)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            print("Listening for test peer", flush=True)
            sock, _ = listener.accept()
    else:
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                sock = socket.create_connection((args.peer, args.port), timeout=2)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
    sock.settimeout(args.timeout)
    return sock


def worker(args):
    os.environ["UCCL_P2P_RDMA_CC"] = args.cc
    os.environ["UCCL_P2P_TRANSPORT"] = "ib"
    os.environ["UCCL_P2P_DISABLE_IPC"] = "1"
    os.environ["UCCL_P2P_COMPRESS_STRATEGY"] = "none"
    if args.module:
        spec = importlib.util.spec_from_file_location("p2p", args.module)
        p2p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p2p)
    else:
        from uccl import p2p

    sizes = args.sizes * args.iov_repeats
    gpu = CudaBuffer(args.gpu, sizes)
    ep = p2p.Endpoint(args.gpu)
    mrs, fifos = [], []
    for ptr, size in zip(gpu.ptrs, sizes):
        ok, mr = ep.reg(ptr, size)
        require(ok, "GPU memory registration failed")
        mrs.append(mr)
        ok, fifo = ep.advertise(mr, ptr, size)
        require(ok, "GPU memory advertisement failed")
        fifos.append(base64.b64encode(bytes(fifo)).decode())
    require(ep.start_passive_accept(), "Endpoint listener failed")
    initiator = args.role == "initiator"
    with connect_control(args) as sock:
        send(
            sock,
            {
                "sizes": sizes,
                "iterations": args.iterations,
                "cc": args.cc,
                "mode": args.mode,
                "fifos": fifos,
                "endpoint": base64.b64encode(bytes(ep.get_metadata())).decode(),
            },
        )
        peer = receive(sock)
        for key, value in [
            ("sizes", sizes),
            ("iterations", args.iterations),
            ("cc", args.cc),
            ("mode", args.mode),
        ]:
            require(peer[key] == value, f"Peers disagree on {key}")
        if initiator:
            ok, conn = ep.add_remote_endpoint(base64.b64decode(peer["endpoint"]))
            require(ok, "Endpoint connection failed")
        remote = [base64.b64decode(blob) for blob in peer["fifos"]]
        modes = ["write", "read"] if args.mode == "both" else [args.mode]
        for mode in modes:
            for iteration in range(args.iterations):
                source = initiator == (mode == "write")
                gpu.fill(iteration, source)
                send(sock, "ready")
                require(receive(sock) == "ready", "Missing ready handshake")
                print(
                    f"START {mode} iteration={iteration} bytes={sum(sizes)}", flush=True
                )
                start = time.monotonic()
                if initiator:
                    ok, transfer = getattr(ep, mode + "v_async")(
                        conn, mrs, gpu.ptrs, sizes, remote, len(sizes)
                    )
                    require(ok, "Transfer submission failed")
                    while True:
                        ok, done = ep.poll_async(transfer)
                        require(ok, "Transfer completion failed")
                        if done:
                            break
                        require(
                            time.monotonic() - start < args.timeout,
                            "Transfer completion timed out",
                        )
                    send(sock, "completed")
                else:
                    require(receive(sock) == "completed", "Missing completion")
                if not source:
                    gpu.verify(iteration)
                # The peer must verify before either side reuses the buffer.
                send(sock, "verified")
                require(receive(sock) == "verified", "Peer verification failed")
                print(
                    f"PASS {mode} iteration={iteration} (full buffer verified)",
                    flush=True,
                )
        # No further RDMA can reference these registrations after the barrier.
        for mr in mrs:
            require(ep.dereg(mr), "Memory deregistration failed")
        del ep
        gpu.call("cudaFree", gpu.ptr)
    print("CC RDMA regression passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["acceptor", "initiator"], required=True)
    parser.add_argument("--peer", help="Acceptor control address (initiator only)")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cc", choices=["none", "swift", "timely"], default="swift")
    parser.add_argument("--mode", choices=["write", "read", "both"], default="both")
    parser.add_argument(
        "--sizes",
        type=lambda s: [int(n) for n in s.split(",")],
        default=[131072],
        help="Comma-separated iov byte sizes",
    )
    parser.add_argument("--iov-repeats", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Hard deadline for the entire local test process, in seconds",
    )
    parser.add_argument("--module", help="Path to a locally built p2p extension")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role == "initiator" and not args.peer:
        parser.error("--peer is required for the initiator")
    if min(args.sizes + [args.iov_repeats, args.iterations, args.timeout]) <= 0:
        parser.error("Sizes, counts and timeout must be positive")
    if args.worker:
        worker(args)
        return 0
    # A Python deadline inside poll_async cannot stop a blocked native thread
    # or destructor. An independent parent enforces a process-wide deadline.
    try:
        result = subprocess.run(
            [sys.executable, "-u", __file__, *sys.argv[1:], "--worker"],
            timeout=args.timeout,
        )
        return result.returncode if result.returncode >= 0 else 1
    except subprocess.TimeoutExpired:
        print("FAIL: CC RDMA test exceeded its deadline", file=sys.stderr)
        return 124


if __name__ == "__main__":
    sys.exit(main())
