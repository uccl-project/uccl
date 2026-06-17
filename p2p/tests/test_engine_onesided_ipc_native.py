#!/usr/bin/env python3
"""
Torch-free end-to-end test of UCCL one-sided IPC (write_ipc / read_ipc) on
Hygon DCU, exercising the *native DTK runtime* build of the p2p module.

Two processes on the same node:
  rank 0 = server/owner  -> DCU 0   (advertises GPU buffers via IPC)
  rank 1 = client        -> DCU 1   (write_ipc / read_ipc against rank 0)
=> cross-GPU zero-copy IPC.

Rendezvous and the tiny control-plane exchanges use multiprocessing.Pipe
instead of torch.distributed, and GPU buffers are allocated via ctypes against
libamdhip64.so instead of torch tensors -- so torch is NEVER imported. This
avoids both the dual HIP runtime conflict and torch's ROCm6.1 lib
incompatibility with the hydcu 6.2.31 driver on node2.

REQUIREMENTS (verified on node0 + node2):
  1. The `uccl.p2p` module MUST be the *native DTK* build (`make -f
     Makefile.dtk`), which links libgalaxyhip. A `dtk-torch` build links
     torch's libamdhip64.so.6 with an RUNPATH baked to torch/lib, so
     `from uccl import p2p` pulls torch's ROCm6.1 HIP + HSA runtime into the
     process EVEN IF this script never imports torch -- reintroducing the
     exact dual-runtime conflict this test is meant to avoid (symptom:
     `undefined symbol: hsa_amd_queue_intercept_register, version ROCR_1`).
  2. LD_LIBRARY_PATH must cover the full native DTK runtime chain. On node0,
     /opt/dtk/lib alone is insufficient: DTK's libamdhip64.so.4 depends on
     libNanoLog.so.1 which lives in /opt/hyhal/lib, and the HSA runtime is in
     /opt/dtk/hsa/lib. Missing them makes the loader fall back to the system
     hsa-runtime (no ROCR_1 symbols) and the import fails.

Run (paths below work on both node0 and node2):
    PYTHONPATH=~/uccl \
    LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/hsa/lib:/opt/hyhal/lib \
        python3 p2p/tests/test_engine_onesided_ipc_native.py
"""
import ctypes
import multiprocessing
import sys
import traceback

HIP_LIB = "/opt/dtk/lib/libamdhip64.so"
HIP_SUCCESS = 0
H2D = 1
D2H = 2
BUF_ELEMS = 1024
SIZE = BUF_ELEMS * 4  # float32 bytes


def _hip():
    lib = ctypes.CDLL(HIP_LIB)
    lib.hipInit(ctypes.c_uint(0))
    return lib


def _chk(lib, err, name):
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{name} failed err={err}")


def alloc_fill(lib, val):
    ptr = ctypes.c_void_p(0)
    _chk(lib, lib.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(SIZE)), "hipMalloc")
    host = (ctypes.c_float * BUF_ELEMS)(*([val] * BUF_ELEMS))
    _chk(lib, lib.hipMemcpy(ptr, host, ctypes.c_size_t(SIZE), ctypes.c_int(H2D)),
         "hipMemcpy H2D")
    return ptr


def read_back(lib, ptr):
    host = (ctypes.c_float * BUF_ELEMS)()
    _chk(lib, lib.hipMemcpy(host, ptr, ctypes.c_size_t(SIZE), ctypes.c_int(D2H)),
         "hipMemcpy D2H")
    return list(host)


def all_close(vals, target, tol=1e-5):
    return all(abs(v - target) < tol for v in vals)


def worker(rank, pipe, result_q):
    try:
        from uccl import p2p
        lib = _hip()
        _chk(lib, lib.hipSetDevice(rank), f"hipSetDevice({rank})")
        ep = p2p.Endpoint(local_gpu_idx=rank)

        # Rendezvous: rank 1 needs rank 0's GPU BDF for connect_local.
        my_meta = bytes(ep.get_metadata())
        if rank == 0:
            pipe.send(my_meta)
            ok, _bdf, conn_id = ep.accept_local()
            assert ok, "accept_local failed"
        else:
            meta0 = pipe.recv()
            _, _, bdf0 = ep.parse_metadata(meta0)
            ok, conn_id = ep.connect_local(remote_gpu_bdf=bdf0)
            assert ok, "connect_local failed"
        print(f"[rank{rank}] connected (conn_id={conn_id})", flush=True)

        # --- write_ipc: client writes 1.0 into server's zero buffer ---
        if rank == 0:
            dst = alloc_fill(lib, 0.0)
            ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE)
            assert ok, "advertise_ipc(write) failed"
            pipe.send(bytes(info))
            pipe.recv()  # wait for client to finish writing
            vals = read_back(lib, dst)
            assert all_close(vals, 1.0), f"write_ipc dst mismatch sample={vals[:4]}"
            print("[rank0] write_ipc PASS  (server dst == 1.0)", flush=True)
        else:
            src = alloc_fill(lib, 1.0)
            info = pipe.recv()
            ok = ep.write_ipc(conn_id, src.value, SIZE, info)
            assert ok, "write_ipc failed"
            pipe.send(b"done")

        # --- read_ipc: client reads server's 1.0 buffer into its zero buffer ---
        if rank == 0:
            src = alloc_fill(lib, 1.0)
            ok, info = ep.advertise_ipc(conn_id, src.value, SIZE)
            assert ok, "advertise_ipc(read) failed"
            pipe.send(bytes(info))
            pipe.recv()  # wait for client to finish reading
            print("[rank0] read_ipc PASS  (buffer exposed for remote READ)",
                  flush=True)
        else:
            dst = alloc_fill(lib, 0.0)
            info = pipe.recv()
            ok = ep.read_ipc(conn_id, dst.value, SIZE, info)
            assert ok, "read_ipc failed"
            vals = read_back(lib, dst)
            assert all_close(vals, 1.0), f"read_ipc dst mismatch sample={vals[:4]}"
            print("[rank1] read_ipc PASS  (client dst == 1.0)", flush=True)
            pipe.send(b"done")

        result_q.put((rank, "OK"))
    except Exception as e:
        result_q.put((rank, f"FAIL: {e}\n{traceback.format_exc()}"))


def main():
    multiprocessing.set_start_method("spawn", force=True)
    p0, p1 = multiprocessing.Pipe()
    rq = multiprocessing.Queue()
    procs = []
    for rank, pipe in ((0, p0), (1, p1)):
        pr = multiprocessing.Process(target=worker, args=(rank, pipe, rq))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join(120)

    results = {}
    while not rq.empty():
        r, s = rq.get()
        results[r] = s

    print("=" * 52)
    ok = True
    for r in (0, 1):
        s = results.get(r, "NO RESULT (timeout/crash)")
        print(f"rank {r}: {s.splitlines()[0]}")
        if not s.startswith("OK"):
            ok = False
            if "FAIL" in s:
                print(s)
    print("=" * 52)
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
