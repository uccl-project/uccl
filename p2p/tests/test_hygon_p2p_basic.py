#!/usr/bin/env python3
"""
Basic Hygon DCU P2P validation — no torch required.

Tests via ctypes directly against libamdhip64.so:
  1. GPU enumeration
  2. hipDeviceCanAccessPeer between DCU 0 and DCU 1
  3. hipIpcGetMemHandle + hipIpcOpenMemHandle (the P2P IPC path)
  4. hipMemcpyPeer between DCU 0 and DCU 1
  5. p2p module load + Endpoint creation
"""

import ctypes
import ctypes.util
import multiprocessing
import os
import sys

HIP_LIB = "/opt/dtk/lib/libamdhip64.so"
HIP_SUCCESS = 0


def load_hip():
    try:
        lib = ctypes.CDLL(HIP_LIB)
    except OSError as e:
        sys.exit(f"Cannot load {HIP_LIB}: {e}")
    return lib


def check(err, name):
    if err != HIP_SUCCESS:
        sys.exit(f"[FAIL] {name} returned error code {err}")


# ── 1. GPU enumeration ────────────────────────────────────────────────────────

def test_device_count(hip):
    count = ctypes.c_int(0)
    check(hip.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
    n = count.value
    print(f"[PASS] hipGetDeviceCount → {n} DCU(s)")
    if n < 1:
        sys.exit("No Hygon DCUs found")
    return n


# ── 2. Peer access capability ─────────────────────────────────────────────────

def test_peer_access(hip, n):
    if n < 2:
        print("[SKIP] peer access check — only 1 DCU")
        return
    can = ctypes.c_int(0)
    check(hip.hipDeviceCanAccessPeer(ctypes.byref(can), 0, 1),
          "hipDeviceCanAccessPeer(0→1)")
    print(f"[{'PASS' if can.value else 'WARN'}] hipDeviceCanAccessPeer(0→1) = {bool(can.value)}")


# ── 3. IPC memory handle round-trip ──────────────────────────────────────────

IPC_HANDLE_SIZE = 64  # sizeof(hipIpcMemHandle_t)


class hipIpcMemHandle_t(ctypes.Structure):
    # Use c_uint8 (not c_char) to avoid null-termination truncation
    # when converting to/from Python bytes.
    _fields_ = [("reserved", ctypes.c_uint8 * 64)]


def _ipc_child(child_conn):
    """Open the IPC handle sent from the parent on DCU 1 and verify data."""
    # Must load HIP fresh in a spawned process (no inherited context).
    hip = load_hip()
    # hipInit(0) is needed before any other HIP call in a spawned process.
    hip.hipInit(ctypes.c_uint(0))
    check(hip.hipSetDevice(1), "hipSetDevice(1) [child]")

    raw = child_conn.recv()  # bytes: handle (64 bytes)
    handle = hipIpcMemHandle_t()
    ctypes.memmove(ctypes.addressof(handle), raw, IPC_HANDLE_SIZE)

    ptr = ctypes.c_void_p(0)
    err = hip.hipIpcOpenMemHandle(ctypes.byref(ptr),
                                  handle,
                                  ctypes.c_uint(0))  # hipIpcMemLazyEnablePeerAccess=0
    if err != HIP_SUCCESS:
        child_conn.send(f"FAIL:hipIpcOpenMemHandle err={err}")
        return

    # Copy the DCU-0 buffer to a local host buffer to inspect
    HOST_BYTES = 4 * 4  # 4 floats
    host_buf = (ctypes.c_float * 4)()
    err = hip.hipMemcpy(host_buf, ptr,
                        ctypes.c_size_t(HOST_BYTES),
                        ctypes.c_int(2))  # hipMemcpyDeviceToHost
    if err != HIP_SUCCESS:
        child_conn.send(f"FAIL:hipMemcpy err={err}")
        return

    values = list(host_buf)
    hip.hipIpcCloseMemHandle(ptr)
    child_conn.send(f"OK:{values}")


def test_ipc_handles(hip, n):
    if n < 2:
        print("[SKIP] IPC handle test — only 1 DCU")
        return

    check(hip.hipSetDevice(0), "hipSetDevice(0)")

    # Allocate 4 floats on DCU 0 and fill with 1.0
    HOST_BYTES = 4 * 4
    src_host = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    dev_ptr = ctypes.c_void_p(0)
    check(hip.hipMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(HOST_BYTES)),
          "hipMalloc")
    check(hip.hipMemcpy(dev_ptr, src_host,
                        ctypes.c_size_t(HOST_BYTES),
                        ctypes.c_int(1)),  # hipMemcpyHostToDevice
          "hipMemcpy H→D")

    # Export IPC handle (hipIpcGetMemHandle takes a pointer to the struct)
    handle = hipIpcMemHandle_t()
    check(hip.hipIpcGetMemHandle(ctypes.byref(handle), dev_ptr), "hipIpcGetMemHandle")
    print("[PASS] hipIpcGetMemHandle succeeded on DCU 0")

    # Send raw bytes (bytearray preserves all 64 bytes including zeros)
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_ipc_child, args=(child_conn,))
    proc.start()
    parent_conn.send(bytes(bytearray(handle.reserved)))
    result = parent_conn.recv()
    proc.join()

    if result.startswith("OK:"):
        values = eval(result[3:])
        ok = all(abs(v - 1.0) < 1e-5 for v in values)
        print(f"[{'PASS' if ok else 'FAIL'}] hipIpcOpenMemHandle on DCU 1 → values={values}")
    else:
        print(f"[FAIL] {result}")

    hip.hipFree(dev_ptr)


# ── 4. hipMemcpyPeer ──────────────────────────────────────────────────────────

def test_memcpy_peer(hip, n):
    if n < 2:
        print("[SKIP] hipMemcpyPeer — only 1 DCU")
        return

    NBYTES = 4 * 4
    src_host = (ctypes.c_float * 4)(2.0, 2.0, 2.0, 2.0)
    dst_host = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)

    check(hip.hipSetDevice(0), "hipSetDevice(0)")
    src_dev = ctypes.c_void_p(0)
    check(hip.hipMalloc(ctypes.byref(src_dev), ctypes.c_size_t(NBYTES)), "hipMalloc src")
    check(hip.hipMemcpy(src_dev, src_host, ctypes.c_size_t(NBYTES), ctypes.c_int(1)),
          "hipMemcpy H→D src")

    check(hip.hipSetDevice(1), "hipSetDevice(1)")
    dst_dev = ctypes.c_void_p(0)
    check(hip.hipMalloc(ctypes.byref(dst_dev), ctypes.c_size_t(NBYTES)), "hipMalloc dst")

    err = hip.hipMemcpyPeer(dst_dev, ctypes.c_int(1),
                             src_dev, ctypes.c_int(0),
                             ctypes.c_size_t(NBYTES))
    if err != HIP_SUCCESS:
        print(f"[WARN] hipMemcpyPeer failed (err={err}) — peer access may not be enabled")
    else:
        check(hip.hipMemcpy(dst_host, dst_dev, ctypes.c_size_t(NBYTES), ctypes.c_int(2)),
              "hipMemcpy D→H dst")
        ok = all(abs(dst_host[i] - 2.0) < 1e-5 for i in range(4))
        print(f"[{'PASS' if ok else 'FAIL'}] hipMemcpyPeer DCU0→DCU1 → {list(dst_host)}")

    hip.hipFree(src_dev)
    hip.hipFree(dst_dev)


# ── 5. p2p module Endpoint creation ──────────────────────────────────────────

def test_p2p_endpoint():
    p2p_so = os.path.join(os.path.dirname(__file__), "..", "p2p.abi3.so")
    p2p_so = os.path.realpath(p2p_so)
    if not os.path.exists(p2p_so):
        print(f"[SKIP] {p2p_so} not found")
        return

    sys.path.insert(0, os.path.dirname(p2p_so))
    try:
        import p2p as _p2p
        ep = _p2p.Endpoint(local_gpu_idx=0)
        meta = bytes(ep.get_metadata())
        print(f"[PASS] p2p.Endpoint(gpu=0) created, metadata len={len(meta)}")
        del ep
    except Exception as e:
        print(f"[FAIL] p2p.Endpoint: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use spawn so child processes start with a clean HIP context.
    multiprocessing.set_start_method("spawn", force=True)

    print("=" * 60)
    print("  Hygon DCU P2P Basic Test")
    print("=" * 60)

    hip = load_hip()
    n = test_device_count(hip)
    test_peer_access(hip, n)
    test_ipc_handles(hip, n)
    test_memcpy_peer(hip, n)
    test_p2p_endpoint()

    print("=" * 60)
    print("  Done")
    print("=" * 60)
