#!/usr/bin/env python3
"""
Torch-free variant of test_engine_onesided_ipc.py for Cambricon MLU.

Exercises the same one-sided IPC surface (write_ipc / read_ipc, vectorized
writev_ipc / readv_ipc, and their _async + poll_async counterparts) between
two real MLUs (MLU 0 and MLU 1) on this node, using cnrtMalloc/cnrtMemcpy
(ctypes against libcnrt.so) instead of torch tensors since torch_mlu is not
installed in this environment.

Run with:
    python3 tests/test_engine_onesided_ipc_cnrt.py
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
# cnrtMemTransDir_t: HostToDev=0, DevToDev=1, DevToHost=2.
H2D, D2D, D2H = 0, 1, 2
NUM_IOVS = 4
BUF_ELEMS = 1024
SIZE_PER = BUF_ELEMS * 4


def load_cnrt():
    return ctypes.CDLL(CNRT_LIB)


def check(err, name):
    if err != CNRT_SUCCESS:
        raise RuntimeError(f"{name} returned error code {err}")


def cnrt_buf(cnrt, gpu_idx: int, fill_val: float):
    check(cnrt.cnrtSetDevice(gpu_idx), "cnrtSetDevice")
    dev_ptr = ctypes.c_void_p(0)
    check(
        cnrt.cnrtMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(SIZE_PER)), "cnrtMalloc"
    )
    host = (ctypes.c_float * BUF_ELEMS)(*([fill_val] * BUF_ELEMS))
    check(
        cnrt.cnrtMemcpy(dev_ptr, host, ctypes.c_size_t(SIZE_PER), ctypes.c_int(H2D)),
        "cnrtMemcpy H->D",
    )
    return dev_ptr


def cnrt_read(cnrt, gpu_idx: int, dev_ptr) -> list:
    check(cnrt.cnrtSetDevice(gpu_idx), "cnrtSetDevice")
    check(cnrt.cnrtSyncDevice(), "cnrtSyncDevice")
    host = (ctypes.c_float * BUF_ELEMS)()
    check(
        cnrt.cnrtMemcpy(host, dev_ptr, ctypes.c_size_t(SIZE_PER), ctypes.c_int(D2H)),
        "cnrtMemcpy D->H",
    )
    return list(host)


def allclose(vals, expected, tol=1e-5):
    return all(abs(v - expected) < tol for v in vals)


def poll_done(ep, transfer_id):
    is_done = False
    while not is_done:
        ok, is_done = ep.poll_async(transfer_id)
        assert ok, "poll_async failed"


PASS = []


def record(name, ok):
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


# ── server (rank 0, MLU 0) ───────────────────────────────────────────────────


def server_proc(pipe):
    cnrt = load_cnrt()
    ep = p2p.Endpoint(local_gpu_idx=0)
    ok, remote_gpu_idx, conn_id = ep.accept_local()
    assert ok, "accept_local failed"

    # write_ipc: client writes 1.0 into our zeroed MLU buffer.
    dst = cnrt_buf(cnrt, 0, 0.0)
    ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("write_ipc", allclose(cnrt_read(cnrt, 0, dst), 1.0))

    # write_ipc_async
    dst = cnrt_buf(cnrt, 0, 0.0)
    ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("write_ipc_async", allclose(cnrt_read(cnrt, 0, dst), 1.0))

    # read_ipc: server source filled with 1.0, client reads it.
    src = cnrt_buf(cnrt, 0, 1.0)
    ok, info = ep.advertise_ipc(conn_id, src.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("read_ipc (server side OK)", True)

    # read_ipc_async
    src = cnrt_buf(cnrt, 0, 1.0)
    ok, info = ep.advertise_ipc(conn_id, src.value, SIZE_PER)
    assert ok
    pipe.send(("blob", bytes(info)))
    assert pipe.recv() == "done"
    record("read_ipc_async (server side OK)", True)

    # writev_ipc: client writes [1,2,3,4] into our zeroed buffers.
    dsts = [cnrt_buf(cnrt, 0, 0.0) for _ in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(
        conn_id, [d.value for d in dsts], [SIZE_PER] * NUM_IOVS
    )
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    ok_all = all(
        allclose(cnrt_read(cnrt, 0, d), float(i + 1)) for i, d in enumerate(dsts)
    )
    record("writev_ipc", ok_all)

    # writev_ipc_async
    dsts = [cnrt_buf(cnrt, 0, 0.0) for _ in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(
        conn_id, [d.value for d in dsts], [SIZE_PER] * NUM_IOVS
    )
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    ok_all = all(
        allclose(cnrt_read(cnrt, 0, d), float(i + 1)) for i, d in enumerate(dsts)
    )
    record("writev_ipc_async", ok_all)

    # readv_ipc: server sources filled [1,2,3,4], client reads.
    srcs = [cnrt_buf(cnrt, 0, float(i + 1)) for i in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(
        conn_id, [s.value for s in srcs], [SIZE_PER] * NUM_IOVS
    )
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    record("readv_ipc (server side OK)", True)

    # readv_ipc_async
    srcs = [cnrt_buf(cnrt, 0, float(i + 1)) for i in range(NUM_IOVS)]
    ok, infos = ep.advertisev_ipc(
        conn_id, [s.value for s in srcs], [SIZE_PER] * NUM_IOVS
    )
    assert ok
    packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in infos)
    pipe.send(("blob", packed))
    assert pipe.recv() == "done"
    record("readv_ipc_async (server side OK)", True)

    pipe.send(("results", PASS))


# ── client (rank 1, MLU 1) ───────────────────────────────────────────────────


def client_proc(pipe):
    cnrt = load_cnrt()
    ep = p2p.Endpoint(local_gpu_idx=1)
    ok, conn_id = ep.connect_local(remote_gpu_bdf=os.environ["SERVER_GPU_BDF"])
    assert ok, "connect_local failed"

    # write_ipc
    kind, info = pipe.recv()
    src = cnrt_buf(cnrt, 1, 1.0)
    ok = ep.write_ipc(conn_id, src.value, SIZE_PER, info)
    assert ok, "write_ipc failed"
    pipe.send("done")

    # write_ipc_async
    kind, info = pipe.recv()
    src = cnrt_buf(cnrt, 1, 1.0)
    ok, tid = ep.write_ipc_async(conn_id, src.value, SIZE_PER, info)
    assert ok, "write_ipc_async failed"
    poll_done(ep, tid)
    pipe.send("done")

    # read_ipc
    kind, info = pipe.recv()
    dst = cnrt_buf(cnrt, 1, 0.0)
    ok = ep.read_ipc(conn_id, dst.value, SIZE_PER, info)
    assert ok, "read_ipc failed"
    record("read_ipc", allclose(cnrt_read(cnrt, 1, dst), 1.0))
    pipe.send("done")

    # read_ipc_async
    kind, info = pipe.recv()
    dst = cnrt_buf(cnrt, 1, 0.0)
    ok, tid = ep.read_ipc_async(conn_id, dst.value, SIZE_PER, info)
    assert ok, "read_ipc_async failed"
    poll_done(ep, tid)
    record("read_ipc_async", allclose(cnrt_read(cnrt, 1, dst), 1.0))
    pipe.send("done")

    # writev_ipc
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size : 4 + (i + 1) * blob_size] for i in range(n)]
    srcs = [cnrt_buf(cnrt, 1, float(i + 1)) for i in range(n)]
    ok = ep.writev_ipc(conn_id, [s.value for s in srcs], [SIZE_PER] * n, infos)
    assert ok, "writev_ipc failed"
    pipe.send("done")

    # writev_ipc_async
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size : 4 + (i + 1) * blob_size] for i in range(n)]
    srcs = [cnrt_buf(cnrt, 1, float(i + 1)) for i in range(n)]
    ok, tid = ep.writev_ipc_async(
        conn_id, [s.value for s in srcs], [SIZE_PER] * n, infos
    )
    assert ok, "writev_ipc_async failed"
    poll_done(ep, tid)
    pipe.send("done")

    # readv_ipc
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size : 4 + (i + 1) * blob_size] for i in range(n)]
    dsts = [cnrt_buf(cnrt, 1, 0.0) for _ in range(n)]
    ok = ep.readv_ipc(conn_id, [d.value for d in dsts], [SIZE_PER] * n, infos)
    assert ok, "readv_ipc failed"
    ok_all = all(
        allclose(cnrt_read(cnrt, 1, d), float(i + 1)) for i, d in enumerate(dsts)
    )
    record("readv_ipc", ok_all)
    pipe.send("done")

    # readv_ipc_async
    kind, packed = pipe.recv()
    n = struct.unpack_from("I", packed, 0)[0]
    blob_size = (len(packed) - 4) // n
    infos = [packed[4 + i * blob_size : 4 + (i + 1) * blob_size] for i in range(n)]
    dsts = [cnrt_buf(cnrt, 1, 0.0) for _ in range(n)]
    ok, tid = ep.readv_ipc_async(
        conn_id, [d.value for d in dsts], [SIZE_PER] * n, infos
    )
    assert ok, "readv_ipc_async failed"
    poll_done(ep, tid)
    ok_all = all(
        allclose(cnrt_read(cnrt, 1, d), float(i + 1)) for i, d in enumerate(dsts)
    )
    record("readv_ipc_async", ok_all)
    pipe.send("done")

    pipe.send(("results", PASS))


def main():
    multiprocessing.set_start_method("spawn", force=True)

    cnrt = load_cnrt()
    bus_id = ctypes.create_string_buffer(64)
    check(
        cnrt.cnrtDeviceGetPCIBusId(bus_id, ctypes.c_int(64), ctypes.c_int(0)),
        "cnrtDeviceGetPCIBusId",
    )
    os.environ["SERVER_GPU_BDF"] = bus_id.value.decode().lower()
    print(f"Server MLU 0 bus id: {os.environ['SERVER_GPU_BDF']}")

    srv_parent, srv_child = multiprocessing.Pipe()
    cli_parent, cli_child = multiprocessing.Pipe()

    srv = multiprocessing.Process(target=server_proc, args=(srv_child,))
    cli = multiprocessing.Process(target=client_proc, args=(cli_child,))
    srv.start()
    time.sleep(1)
    cli.start()

    # Relay messages between the two pipes (server <-> client) in this parent
    # process since accept_local/connect_local talk over a local control
    # channel, not directly via these pipes.
    results = {"server": None, "client": None}
    while True:
        if srv_parent.poll(0.01):
            msg = srv_parent.recv()
            if msg[0] == "blob":
                cli_parent.send(("blob", msg[1]))
            elif msg[0] == "results":
                results["server"] = msg[1]
        if cli_parent.poll(0.01):
            msg = cli_parent.recv()
            if msg == "done":
                srv_parent.send("done")
            elif isinstance(msg, tuple) and msg[0] == "results":
                results["client"] = msg[1]
        if results["server"] is not None and results["client"] is not None:
            break
        if not srv.is_alive() and not cli.is_alive():
            break

    srv.join(timeout=10)
    cli.join(timeout=10)

    all_results = (results["server"] or []) + (results["client"] or [])
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for name, ok in all_results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in all_results if not ok]
    assert srv.exitcode == 0, f"server exitcode={srv.exitcode}"
    assert cli.exitcode == 0, f"client exitcode={cli.exitcode}"
    assert not failed, f"failed tests: {failed}"
    assert len(all_results) >= 12, f"too few results: {len(all_results)}"
    print("\nAll one-sided IPC tests passed! (CNRT, torch-free)")


if __name__ == "__main__":
    main()
