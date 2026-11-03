#!/usr/bin/env python3
"""Measure mixed-size RDMA vectors on one Endpoint and two independent peers.

No GPU libraries are imported by --help. See benchmark_proxy_fairness.md for
topology, separate-process A/B commands, and the measurement boundaries.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time

# Keep native owners alive until explicit cleanup, or process exit on error.
# Unwinding an Endpoint with failed/in-flight operations can otherwise block.
KEEPALIVE = []
PROTOCOL = 2
MAX_CONTROL_BYTES = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def send_message(sock, message):
    payload = json.dumps(message).encode("utf-8")
    require(len(payload) <= MAX_CONTROL_BYTES, "control message too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(sock):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            data = sock.recv(size - len(parts))
            require(data, "control peer disconnected")
            parts.extend(data)
        return bytes(parts)

    size = struct.unpack("!I", exact(4))[0]
    require(0 < size <= MAX_CONTROL_BYTES, "invalid control message size")
    message = json.loads(exact(size))
    require(isinstance(message, dict), "invalid control message")
    require(message.get("op") != "abort", "remote benchmark aborted")
    return message


class Runtime:
    """The same backend_info()/ctypes contract as tests/native_runtime.py."""

    def __init__(self, info, gpu):
        self.lib = ctypes.CDLL(
            os.environ.get("UCCL_GPU_RT_LIB") or info["runtime_library"]
        )
        prefix = info["runtime_prefix"]
        self.allocate = getattr(self.lib, prefix + "Malloc")
        self.release = getattr(self.lib, prefix + "Free")
        self.copy = getattr(self.lib, prefix + "Memcpy")
        select = getattr(self.lib, prefix + "SetDevice")
        self.allocate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.release.argtypes = [ctypes.c_void_p]
        self.copy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        select.argtypes = [ctypes.c_int]
        for function in (self.allocate, self.release, self.copy, select):
            function.restype = ctypes.c_int
        if info["runtime_init"]:
            initialize = getattr(self.lib, info["runtime_init"])
            initialize.argtypes = [ctypes.c_uint]
            initialize.restype = ctypes.c_int
            self.check(initialize(0), "runtime initialization")
        self.check(select(gpu), "select GPU")
        self.h2d = info["memcpy_h2d"]
        self.d2h = info["memcpy_d2h"]

    @staticmethod
    def check(status, operation):
        require(status == 0, f"{operation} failed: runtime status {status}")

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self.check(self.allocate(ctypes.byref(pointer), size), "allocate GPU buffer")
        return pointer.value

    def fill(self, pointer, size, value):
        chunk = min(size, 1024 * 1024)
        host = ctypes.create_string_buffer(bytes([value]) * chunk, chunk)
        for offset in range(0, size, chunk):
            self.check(
                self.copy(pointer + offset, host, min(chunk, size - offset), self.h2d),
                "initialize GPU buffer",
            )

    def verify(self, pointer, size, value):
        chunk = min(size, 1024 * 1024)
        host = ctypes.create_string_buffer(chunk)
        expected = bytes([value]) * chunk
        for offset in range(0, size, chunk):
            length = min(chunk, size - offset)
            self.check(self.copy(host, pointer + offset, length, self.d2h), "read back")
            require(
                host.raw[:length] == expected[:length],
                f"payload mismatch at allocation offset {offset}",
            )


def vector_sizes(size, iov_bytes):
    require(size > 0 and iov_bytes > 0, "buffer and IOV sizes must be positive")
    return [min(iov_bytes, size - offset) for offset in range(0, size, iov_bytes)]


class BufferSlot:
    def __init__(self, runtime, endpoint, size, iov_bytes, buffer_offset=0):
        self.runtime, self.endpoint, self.size = runtime, endpoint, size
        self.allocation = runtime.malloc(size + buffer_offset)
        self.pointer = self.allocation + buffer_offset
        ok, self.mr_id = endpoint.reg(self.allocation, size + buffer_offset)
        require(ok, "register memory failed")
        self.sizes = vector_sizes(size, iov_bytes)
        self.pointers = [self.pointer + i * iov_bytes for i in range(len(self.sizes))]
        self.mr_ids = [self.mr_id] * len(self.sizes)
        self.remote = None
        self.transfer = None
        self.started = None

    def advertise(self):
        ok, metadata = self.endpoint.advertisev(
            self.mr_ids, self.pointers, self.sizes, len(self.sizes)
        )
        require(ok, "advertisev failed")
        require(all(len(blob) == 64 for blob in metadata), "invalid FifoItem length")
        return [bytes(blob).hex() for blob in metadata]

    def close(self):
        require(self.transfer is None, "cannot free an in-flight buffer")
        require(self.endpoint.dereg(self.mr_id), "deregister memory failed")
        self.runtime.check(self.runtime.release(self.allocation), "free GPU buffer")


def check_remote_write_ranges(streams):
    """Match the scheduler's conservative numeric-address conflict domain.

    advertisev returns FifoItem, not XferDesc. common.cc serializes addr at byte
    0 and size at byte 8 with native-endian memcpy. Peer byte order is checked
    during the handshake before these bytes are decoded.
    """
    ranges = []
    for stream in streams:
        intervals = []
        for slot in stream["slots"]:
            for descriptor, length in zip(slot.remote, slot.sizes):
                require(len(descriptor) == 64, "invalid FifoItem length")
                address, capacity = struct.unpack_from("=QI", descriptor)
                require(
                    0 < length <= capacity, "write exceeds advertised FifoItem size"
                )
                require(
                    0 < address and address + length < 2**64,
                    "invalid remote address range",
                )
                intervals.append((address, address + length))
        require(intervals, "empty remote write vector")
        ranges.append(sorted(intervals))
    large, small = ranges
    left, right = 0, 0
    while left < len(large) and right < len(small):
        a, b = large[left], small[right]
        require(
            not (a[0] < b[1] and b[0] < a[1]),
            "WRITE benchmark would be serialized by conservative remote-address "
            f"conflict detection: large [{a[0]:#x}, {a[1]:#x}) overlaps small "
            f"[{b[0]:#x}, {b[1]:#x}), even though the peers are independent. "
            "Restart a peer with a different --buffer-offset BYTES and recheck, "
            "or use the default --mode read. Offsets do not guarantee disjoint VAs.",
        )
        if a[1] <= b[1]:
            left += 1
        else:
            right += 1
    return {
        "status": "cross-flow numeric remote write ranges are disjoint",
        "large": [[hex(start), hex(end)] for start, end in large],
        "small": [[hex(start), hex(end)] for start, end in small],
    }


def pattern(phase, stream, slot):
    return 1 + (phase * 31 + stream * 71 + slot * 3) % 251


def load_backend(args):
    required = {
        "UCCL_P2P_TRANSPORT": "rdma",
        "UCCL_P2P_DISABLE_IPC": "1",
        "UCCL_P2P_RDMA_CC": "none",
        "UCCL_P2P_COMPRESS_STRATEGY": "none",
    }
    for name, value in required.items():
        require(os.environ.get(name, value) == value, f"set {name}={value}")
        os.environ[name] = value
    require(
        os.environ.get("UCCL_P2P_COOPERATIVE_PROXY", "0") in ("0", "1"),
        "UCCL_P2P_COOPERATIVE_PROXY must be 0 or 1",
    )
    # Delay imports and all device access until after argument parsing.
    from uccl import p2p

    info = dict(p2p.backend_info())
    require(
        not info["staged_network"], "staged-network backend is outside this benchmark"
    )
    runtime = Runtime(info, args.gpu)
    endpoint = p2p.Endpoint(local_gpu_idx=args.gpu)
    KEEPALIVE.extend((runtime, endpoint))
    binary = Path(p2p.__file__)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    identity = {
        "hostname": socket.gethostname(),
        "gpu": args.gpu,
        "byteorder": sys.byteorder,
        "buffer_offset_bytes": args.buffer_offset,
        "backend": info,
        "binary": str(binary),
        "binary_sha256": digest,
        "build_label": args.build_label,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("UCCL_P2P_") or key == "UCCL_SOCKET_IFNAME"
        },
    }
    return endpoint, runtime, identity


def run_peer(args):
    with socket.create_server((args.listen_host, args.port)) as listener:
        listener.settimeout(args.timeout)
        print(f"Waiting for initiator at {args.listen_host}:{args.port}", flush=True)
        control, _ = listener.accept()
        with control:
            control.settimeout(args.timeout)
            hello = receive_message(control)
            require(hello.get("protocol") == PROTOCOL, "protocol version mismatch")
            require(hello.get("op") == "hello", "expected hello")
            spec, stream, mode = hello["spec"], hello["stream"], hello["mode"]
            require(stream in (0, 1) and mode in ("read", "write"), "invalid workload")
            require(1 <= spec["depth"] <= 64, "invalid depth")
            require(
                0 < spec["iov_bytes"] <= hello["chunk_bytes"],
                "IOV exceeds declared chunk size",
            )
            require(
                0
                < (spec["bytes"] + args.buffer_offset) * spec["depth"]
                <= args.max_buffer_mib * 1024**2,
                "requested buffers exceed peer --max-buffer-mib",
            )
            endpoint, runtime, identity = load_backend(args)
            slots = [
                BufferSlot(
                    runtime,
                    endpoint,
                    spec["bytes"],
                    spec["iov_bytes"],
                    args.buffer_offset,
                )
                for _ in range(spec["depth"])
            ]
            KEEPALIVE.extend(slots)
            # reg() initializes the lazy engine. Do it before accept().
            send_message(
                control,
                {
                    "op": "endpoint",
                    "metadata": bytes(endpoint.get_metadata()).hex(),
                    "identity": identity,
                },
            )
            ok, _, _, _ = endpoint.accept()
            require(ok, "RDMA accept failed")
            send_message(
                control, {"op": "ready", "slots": [slot.advertise() for slot in slots]}
            )
            prepared_phase = None
            while True:
                message = receive_message(control)
                operation = message.get("op")
                if operation == "prepare":
                    prepared_phase = message["phase"]
                    for index, slot in enumerate(slots):
                        value = (
                            pattern(prepared_phase, stream, index)
                            if mode == "read"
                            else 0
                        )
                        runtime.fill(slot.pointer, slot.size, value)
                    send_message(control, {"op": "prepared", "phase": prepared_phase})
                elif operation == "verify":
                    require(message["phase"] == prepared_phase, "phase mismatch")
                    for index, slot in enumerate(slots):
                        runtime.verify(
                            slot.pointer,
                            slot.size,
                            pattern(prepared_phase, stream, index),
                        )
                    send_message(control, {"op": "verified", "phase": prepared_phase})
                elif operation == "stop":
                    for slot in slots:
                        slot.close()
                    send_message(control, {"op": "stopped"})
                    return
                else:
                    raise RuntimeError(f"unexpected peer operation: {operation}")


def percentile(values, percent):
    ordered = sorted(values)
    require(ordered, "no latency samples")
    # Nearest-rank percentiles; include this definition in the result.
    return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]


def run_timed_phase(endpoint, streams, mode, seconds):
    submit = endpoint.writev_async if mode == "write" else endpoint.readv_async
    latencies = [[], []]
    submitted = [0, 0]
    cpu_start = time.process_time()
    start = time.perf_counter()
    deadline = start + seconds

    def post(stream, slot):
        slot.started = time.perf_counter()
        ok, transfer = submit(
            stream["connection"],
            slot.mr_ids,
            slot.pointers,
            slot.sizes,
            slot.remote,
            len(slot.sizes),
        )
        require(ok, f"{mode}v_async rejected a request")
        slot.transfer = transfer
        submitted[stream["index"]] += 1

    # Queue large requests first so the small flow encounters a loaded proxy.
    # Each outstanding request owns a separate allocation at both endpoints.
    for stream in streams:
        for slot in stream["slots"]:
            post(stream, slot)

    while any(
        slot.transfer is not None for stream in streams for slot in stream["slots"]
    ):
        # Observe the small flow first in BOTH modes. Python polling overhead is
        # part of submit-to-observed-completion latency, not a NIC timestamp.
        for stream in reversed(streams):
            for slot in stream["slots"]:
                if slot.transfer is None:
                    continue
                ok, done = endpoint.poll_async(slot.transfer)
                require(ok, "poll_async reported a failed transfer")
                if not done:
                    continue
                finished = time.perf_counter()
                slot.transfer = None  # poll_async destroys the completed handle.
                latencies[stream["index"]].append((finished - slot.started) * 1e6)
                if finished < deadline:
                    post(stream, slot)
    elapsed = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    completed = [len(values) for values in latencies]
    require(completed == submitted, "submitted/completed request count mismatch")
    byte_counts = [completed[i] * streams[i]["spec"]["bytes"] for i in (0, 1)]
    return {
        "requested_submission_seconds": seconds,
        "elapsed_seconds_including_drain": elapsed,
        "initiator_process_cpu_seconds": cpu,
        "initiator_process_cpu_cores_average": cpu / elapsed,
        "large_completed_requests": completed[0],
        "small_completed_requests": completed[1],
        "large_bytes": byte_counts[0],
        "small_bytes": byte_counts[1],
        "large_GB_per_second": byte_counts[0] / elapsed / 1e9,
        "total_GB_per_second": sum(byte_counts) / elapsed / 1e9,
        "small_p50_us": percentile(latencies[1], 0.50),
        "small_p99_us": percentile(latencies[1], 0.99),
        "small_latency_us": latencies[1],
    }


def run_initiator(args):
    endpoint, runtime, identity = load_backend(args)
    streams, controls, remote_identities, remote_endpoints = [], [], [], []
    for index, (peer, size, iov, depth) in enumerate(
        zip(
            args.peer,
            (args.large_bytes, args.small_bytes),
            (args.large_iov_bytes, args.small_iov_bytes),
            (args.large_depth, args.small_depth),
        )
    ):
        host, port = peer.rsplit(":", 1)
        control = socket.create_connection((host.strip("[]"), int(port)), args.timeout)
        controls.append(control)
        spec = {"bytes": size, "iov_bytes": iov, "depth": depth}
        slots = [BufferSlot(runtime, endpoint, size, iov) for _ in range(depth)]
        KEEPALIVE.extend(slots)
        send_message(
            control,
            {
                "op": "hello",
                "protocol": PROTOCOL,
                "stream": index,
                "mode": args.mode,
                "spec": spec,
                "chunk_bytes": args.chunk_bytes,
            },
        )
        response = receive_message(control)
        require(response.get("op") == "endpoint", "expected endpoint metadata")
        require(
            response["identity"]["byteorder"] == sys.byteorder,
            "native FifoItem descriptors require matching peer byte order",
        )
        remote = bytes.fromhex(response["metadata"])
        ip, port, bdf = endpoint.parse_metadata(remote)
        require(
            (ip, port) not in remote_endpoints,
            "large and small require independent peer Endpoints",
        )
        remote_endpoints.append((ip, port))
        remote_identities.append(response["identity"])
        ok, connection = endpoint.connect(ip, bdf, remote_port=port)
        require(ok, "RDMA connect failed")
        response = receive_message(control)
        require(
            response.get("op") == "ready" and len(response["slots"]) == depth,
            "invalid peer slots",
        )
        for slot, descriptors in zip(slots, response["slots"]):
            slot.remote = [bytes.fromhex(blob) for blob in descriptors]
            require(
                len(slot.remote) == len(slot.sizes)
                and all(len(blob) == 64 for blob in slot.remote),
                "invalid remote vector metadata",
            )
        streams.append(
            {"index": index, "spec": spec, "connection": connection, "slots": slots}
        )
    require(
        streams[0]["connection"] != streams[1]["connection"], "connections must differ"
    )
    # Separate GPU processes often receive identical numeric VAs. Different
    # peers alone do not imply independent WRITE tasks for this scheduler.
    address_check = (
        check_remote_write_ranges(streams)
        if args.mode == "write"
        else {"status": "READ destinations are separate local allocations"}
    )

    results = []
    for phase, seconds in enumerate((args.warmup_seconds, args.duration), start=1):
        for control in controls:
            send_message(control, {"op": "prepare", "phase": phase})
        for stream in streams:
            for index, slot in enumerate(stream["slots"]):
                value = (
                    pattern(phase, stream["index"], index)
                    if args.mode == "write"
                    else 0
                )
                runtime.fill(slot.pointer, slot.size, value)
        for control in controls:
            require(
                receive_message(control) == {"op": "prepared", "phase": phase},
                "prepare failed",
            )
        result = run_timed_phase(endpoint, streams, args.mode, seconds)
        # Data checking and all control traffic are outside the timed interval.
        for control in controls:
            send_message(control, {"op": "verify", "phase": phase})
        for stream in streams:
            for index, slot in enumerate(stream["slots"]):
                runtime.verify(
                    slot.pointer, slot.size, pattern(phase, stream["index"], index)
                )
        for control in controls:
            require(
                receive_message(control) == {"op": "verified", "phase": phase},
                "remote verification failed",
            )
        results.append(result)

    for control in controls:
        send_message(control, {"op": "stop"})
    for control in controls:
        require(receive_message(control).get("op") == "stopped", "peer cleanup failed")
        control.close()
    for stream in streams:
        for slot in stream["slots"]:
            slot.close()

    measurement = results[1]
    output = {
        "benchmark": "proxy-fairness-v1",
        "mode": args.mode,
        "cooperative_proxy": int(os.environ.get("UCCL_P2P_COOPERATIVE_PROXY", "0")),
        "initiator": identity,
        "peers": remote_identities,
        "remote_endpoints": remote_endpoints,
        "address_conflict_check": address_check,
        "workload": {
            "large": streams[0]["spec"],
            "small": streams[1]["spec"],
            "declared_build_chunk_bytes": args.chunk_bytes,
        },
        "measurement": measurement,
        "warmup_completed_requests": [
            results[0]["large_completed_requests"],
            results[0]["small_completed_requests"],
        ],
        "payload_validation": "full buffers after each drained phase; not per-transfer epoch validation",
        "latency_definition": "Python vector API entry to poll_async completion observation; nearest-rank percentiles",
        "throughput_definition": "completed payload bytes / interval including final drain; decimal GB/s",
        "cpu_definition": "initiator process CPU, including native threads and Python; excludes peers/setup/validation",
        "raw_path_requirement": "each IOV must fit this binary's actual chunk size; compression and CC disabled",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in measurement.items()
                if key != "small_latency_us"
            },
            indent=2,
        )
    )
    print(f"Verified payloads; results written to {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--role", choices=("peer", "initiator"), required=True)
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="Initiator: specify twice, large peer first, small peer second",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=29500, help="Peer control-plane port"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=("read", "write"), default="read")
    parser.add_argument(
        "--buffer-offset",
        type=int,
        default=0,
        help="Peer only: unused prefix bytes per allocation, to adjust advertised WRITE addresses",
    )
    parser.add_argument("--large-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--large-iov-bytes", type=int, default=512 * 1024)
    parser.add_argument("--small-bytes", type=int, default=64 * 1024)
    parser.add_argument("--small-iov-bytes", type=int, default=4 * 1024)
    parser.add_argument("--large-depth", type=int, default=2)
    parser.add_argument("--small-depth", type=int, default=1)
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=512 * 1024,
        help="Actual build's single-chunk limit; operator must verify against source",
    )
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Whole-process watchdog seconds, including setup and validation",
    )
    parser.add_argument(
        "--max-buffer-mib",
        type=int,
        default=2048,
        help="Peer allocation cap; includes all outstanding slots and buffer-offset prefixes",
    )
    parser.add_argument(
        "--build-label",
        required=True,
        help="Revision/build identifier recorded in JSON",
    )
    parser.add_argument(
        "--output", default="proxy-fairness.json", help="Initiator JSON result path"
    )
    args = parser.parse_args()
    if args.buffer_offset < 0 or (args.role != "peer" and args.buffer_offset != 0):
        parser.error(
            "--buffer-offset must be nonnegative and is only supported for --role peer"
        )
    if args.role == "initiator" and (len(args.peer) != 2 or len(set(args.peer)) != 2):
        parser.error("initiator requires two distinct --peer HOST:PORT endpoints")
    if any(
        value <= 0
        for value in (
            args.large_bytes,
            args.small_bytes,
            args.large_iov_bytes,
            args.small_iov_bytes,
            args.chunk_bytes,
            args.max_buffer_mib,
        )
    ):
        parser.error("sizes must be positive")
    if max(args.large_iov_bytes, args.small_iov_bytes) > args.chunk_bytes:
        parser.error("each IOV must fit the declared --chunk-bytes limit")
    if args.large_bytes <= args.small_bytes:
        parser.error("--large-bytes must exceed --small-bytes")
    if not all(1 <= depth <= 64 for depth in (args.large_depth, args.small_depth)):
        parser.error("queue depths must be between 1 and 64")
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.duration, args.warmup_seconds, args.timeout)
    ):
        parser.error("durations and timeout must be finite and positive")
    if args.timeout <= args.duration + args.warmup_seconds:
        parser.error("--timeout must cover setup, warmup, measurement and validation")
    return args


def main():
    args = parse_args()

    def expired():
        print(
            "Benchmark watchdog expired; terminating without freeing in-flight buffers",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)

    watchdog = threading.Timer(args.timeout, expired)
    watchdog.daemon = True
    watchdog.start()
    try:
        if args.role == "peer":
            run_peer(args)
        else:
            run_initiator(args)
        KEEPALIVE.clear()
        watchdog.cancel()
    except BaseException as error:
        print(
            f"Benchmark failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        # Native requests cannot be cancelled safely. OS process cleanup avoids
        # deregistering/freeing memory while a NIC may still access it.
        os._exit(1)


if __name__ == "__main__":
    main()
