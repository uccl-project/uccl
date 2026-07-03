#!/usr/bin/env python3
"""
Torch-free intra-node P2P bandwidth/latency benchmark for Cambricon MLU.

Reuses the same one-sided IPC engine surface as test_engine_onesided_ipc_cnrt.py:
server advertises a dst buffer on MLU 0, client issues repeated write_ipc_async
from MLU 1 and measures wall time -> bandwidth (large msgs) and per-op latency
(small msgs).  Memory via cnrtMalloc (ctypes against libcnrt.so), no torch_mlu.

    python3 tests/bench_ipc_cnrt.py
"""

import ctypes
import multiprocessing
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import p2p

NEUWARE_HOME = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
CNRT_LIB = os.path.join(NEUWARE_HOME, "lib64", "libcnrt.so")
CNRT_SUCCESS = 0
H2D = 0

# message sizes (bytes) and iteration counts
SIZES = [
    (1 << 12, 2000),  # 4 KB
    (1 << 14, 2000),  # 16 KB
    (1 << 16, 2000),  # 64 KB
    (1 << 18, 1000),  # 256 KB
    (1 << 20, 1000),  # 1 MB
    (1 << 22, 500),  # 4 MB
    (1 << 24, 200),  # 16 MB
    (1 << 26, 100),  # 64 MB
    (1 << 28, 50),  # 256 MB
]
MAX_SIZE = SIZES[-1][0]
WARMUP = 10


def load_cnrt():
    return ctypes.CDLL(CNRT_LIB)


def check(err, name):
    if err != CNRT_SUCCESS:
        raise RuntimeError(f"{name} returned error code {err}")


def cnrt_malloc(cnrt, gpu_idx, size):
    check(cnrt.cnrtSetDevice(gpu_idx), "cnrtSetDevice")
    dev_ptr = ctypes.c_void_p(0)
    check(cnrt.cnrtMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(size)), "cnrtMalloc")
    check(
        cnrt.cnrtMemset(dev_ptr, ctypes.c_int(0), ctypes.c_size_t(size)), "cnrtMemset"
    )
    return dev_ptr


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}"
        n /= 1024
    return f"{n}"


def human(n):
    if n >= 1 << 20:
        return f"{n >> 20}MB"
    if n >= 1 << 10:
        return f"{n >> 10}KB"
    return f"{n}B"


def server_proc(pipe):
    cnrt = load_cnrt()
    ep = p2p.Endpoint(local_gpu_idx=0)
    ok, remote_gpu_idx, conn_id = ep.accept_local()
    assert ok, "accept_local failed"
    dst = cnrt_malloc(cnrt, 0, MAX_SIZE)
    for size, _iters in SIZES:
        ok, info = ep.advertise_ipc(conn_id, dst.value, size)
        assert ok, "advertise_ipc failed"
        pipe.send(("blob", bytes(info)))
        assert pipe.recv() == "done"
    pipe.send(("results", "server_ok"))


def client_proc(pipe):
    cnrt = load_cnrt()
    ep = p2p.Endpoint(local_gpu_idx=1)
    ok, conn_id = ep.connect_local(remote_gpu_bdf=os.environ["SERVER_GPU_BDF"])
    assert ok, "connect_local failed"
    src = cnrt_malloc(cnrt, 1, MAX_SIZE)

    results = []
    for size, iters in SIZES:
        kind, info = pipe.recv()

        # warmup
        for _ in range(WARMUP):
            ok, tid = ep.write_ipc_async(conn_id, src.value, size, info)
            assert ok, "write_ipc_async failed"
            done = False
            while not done:
                ok, done = ep.poll_async(tid)
                assert ok

        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            ok, tid = ep.write_ipc_async(conn_id, src.value, size, info)
            assert ok, "write_ipc_async failed"
            done = False
            while not done:
                ok, done = ep.poll_async(tid)
                assert ok
        t1 = time.perf_counter()

        total = t1 - t0
        per_op = total / iters
        bw = size / per_op  # bytes/s
        results.append((size, iters, per_op, bw))
        pipe.send("done")

    pipe.send(("results", results))


def main():
    multiprocessing.set_start_method("spawn", force=True)
    cnrt = load_cnrt()
    bus_id = ctypes.create_string_buffer(64)
    check(
        cnrt.cnrtDeviceGetPCIBusId(bus_id, ctypes.c_int(64), ctypes.c_int(0)),
        "cnrtDeviceGetPCIBusId",
    )
    os.environ["SERVER_GPU_BDF"] = bus_id.value.decode().lower()
    print(f"Intra-node P2P write_ipc benchmark: MLU 1 -> MLU 0 (CNRT, torch-free)")
    print(f"Server MLU 0 bus id: {os.environ['SERVER_GPU_BDF']}\n")

    srv_parent, srv_child = multiprocessing.Pipe()
    cli_parent, cli_child = multiprocessing.Pipe()
    srv = multiprocessing.Process(target=server_proc, args=(srv_child,))
    cli = multiprocessing.Process(target=client_proc, args=(cli_child,))
    srv.start()
    time.sleep(1)
    cli.start()

    client_results = None
    while True:
        if srv_parent.poll(0.01):
            msg = srv_parent.recv()
            if msg[0] == "blob":
                cli_parent.send(("blob", msg[1]))
        if cli_parent.poll(0.01):
            msg = cli_parent.recv()
            if msg == "done":
                srv_parent.send("done")
            elif isinstance(msg, tuple) and msg[0] == "results":
                client_results = msg[1]
                break
        if not srv.is_alive() and not cli.is_alive():
            break

    srv.join(timeout=10)
    cli.join(timeout=10)
    assert cli.exitcode == 0, f"client exitcode={cli.exitcode}"
    assert srv.exitcode == 0, f"server exitcode={srv.exitcode}"

    print(f"{'size':>8} {'iters':>6} {'lat(us)':>10} {'BW(GB/s)':>10}")
    print("-" * 38)
    for size, iters, per_op, bw in client_results:
        print(f"{human(size):>8} {iters:>6} {per_op*1e6:>10.2f} {bw/1e9:>10.2f}")
    small = client_results[0]
    big = client_results[-1]
    print("\nSummary:")
    print(f"  small-msg latency ({human(small[0])}): {small[2]*1e6:.2f} us")
    print(f"  peak bandwidth     ({human(big[0])}): {big[3]/1e9:.2f} GB/s")


if __name__ == "__main__":
    main()
