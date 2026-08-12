#!/usr/bin/env python3
"""
Torch-free end-to-end test of UCCL one-sided IPC (write_ipc / read_ipc) across
two GPUs on one node: rank 0 (GPU 0) advertises buffers, rank 1 (GPU 1) runs
write_ipc / read_ipc against them. The control plane uses multiprocessing.Pipe
and buffers are allocated via ctypes, so torch is never imported.

Runtime backend is auto-detected: Hygon DCU / HIP by default, Cambricon MLU /
CNRT when neuware is present. Force it with UCCL_GPU_RT=hip|cnrt, or point at a
specific shared library with UCCL_GPU_RT_LIB.

Requirements (Hygon): build uccl.p2p with `make -f Makefile.dtk`, and
  LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/hsa/lib:/opt/hyhal/lib
Requirements (Cambricon): build with `make -f Makefile.cnrt`.

Run:
    PYTHONPATH=~/uccl python3 p2p/tests/test_engine_onesided_ipc_native.py
"""

import ctypes
import multiprocessing
import os
import sys
import traceback

RT_SUCCESS = 0
BUF_ELEMS = 1024
SIZE = BUF_ELEMS * 4  # float32 bytes


def _detect_backend():
    kind = os.environ.get("UCCL_GPU_RT", "").lower()
    lib = os.environ.get("UCCL_GPU_RT_LIB", "")
    if not kind:
        if lib:
            kind = "cnrt" if "cnrt" in lib else "hip"
        elif os.path.isdir("/usr/local/neuware") and not os.path.exists(
            "/opt/dtk/lib/libamdhip64.so"
        ):
            kind = "cnrt"
        else:
            kind = "hip"
    if not lib:
        lib = (
            "/usr/local/neuware/lib64/libcnrt.so"
            if kind == "cnrt"
            else "/opt/dtk/lib/libamdhip64.so"
        )
    return kind, lib


class Rt:
    """Minimal GPU-runtime shim over HIP (hip*) or Cambricon CNRT (cnrt*)."""

    def __init__(self):
        self.kind, path = _detect_backend()
        self.lib = ctypes.CDLL(path)
        if self.kind == "cnrt":
            self._h2d, self._d2h = 0, 2  # cnrtMemTransDir_t
            self._malloc, self._memcpy = self.lib.cnrtMalloc, self.lib.cnrtMemcpy
            self._set = self.lib.cnrtSetDevice
        else:
            self.lib.hipInit(ctypes.c_uint(0))
            self._h2d, self._d2h = 1, 2  # hipMemcpyKind
            self._malloc, self._memcpy = self.lib.hipMalloc, self.lib.hipMemcpy
            self._set = self.lib.hipSetDevice

    def _chk(self, err, name):
        if err != RT_SUCCESS:
            raise RuntimeError(f"{name} failed err={err}")

    def set_device(self, idx):
        self._chk(self._set(idx), f"set_device({idx})")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p(0)
        self._chk(self._malloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "malloc")
        return ptr

    def memcpy_h2d(self, dst, host, nbytes):
        self._chk(
            self._memcpy(dst, host, ctypes.c_size_t(nbytes), ctypes.c_int(self._h2d)),
            "memcpy H2D",
        )

    def memcpy_d2h(self, host, src, nbytes):
        self._chk(
            self._memcpy(host, src, ctypes.c_size_t(nbytes), ctypes.c_int(self._d2h)),
            "memcpy D2H",
        )


def alloc_fill(rt, val):
    ptr = rt.malloc(SIZE)
    host = (ctypes.c_float * BUF_ELEMS)(*([val] * BUF_ELEMS))
    rt.memcpy_h2d(ptr, host, SIZE)
    return ptr


def read_back(rt, ptr):
    host = (ctypes.c_float * BUF_ELEMS)()
    rt.memcpy_d2h(host, ptr, SIZE)
    return list(host)


def all_close(vals, target, tol=1e-5):
    return all(abs(v - target) < tol for v in vals)


def worker(rank, pipe, result_q):
    try:
        from uccl import p2p

        rt = Rt()
        rt.set_device(rank)
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
            dst = alloc_fill(rt, 0.0)
            ok, info = ep.advertise_ipc(conn_id, dst.value, SIZE)
            assert ok, "advertise_ipc(write) failed"
            pipe.send(bytes(info))
            pipe.recv()  # wait for client to finish writing
            vals = read_back(rt, dst)
            assert all_close(vals, 1.0), f"write_ipc dst mismatch sample={vals[:4]}"
            print("[rank0] write_ipc PASS  (server dst == 1.0)", flush=True)
        else:
            src = alloc_fill(rt, 1.0)
            info = pipe.recv()
            ok = ep.write_ipc(conn_id, src.value, SIZE, info)
            assert ok, "write_ipc failed"
            pipe.send(b"done")

        # --- read_ipc: client reads server's 1.0 buffer into its zero buffer ---
        if rank == 0:
            src = alloc_fill(rt, 1.0)
            ok, info = ep.advertise_ipc(conn_id, src.value, SIZE)
            assert ok, "advertise_ipc(read) failed"
            pipe.send(bytes(info))
            pipe.recv()  # wait for client to finish reading
            print("[rank0] read_ipc PASS  (buffer exposed for remote READ)", flush=True)
        else:
            dst = alloc_fill(rt, 0.0)
            info = pipe.recv()
            ok = ep.read_ipc(conn_id, dst.value, SIZE, info)
            assert ok, "read_ipc failed"
            vals = read_back(rt, dst)
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
