"""
Test script for UCCL P2P Engine one-sided IPC operations.

Each test fills the source buffer with a known pattern and verifies the
destination buffer contains the same data after the operation completes.

  write_ipc / writev_ipc : client fills source → writes → server checks destination
  read_ipc  / readv_ipc  : server fills source → client reads → client checks destination

Vectorized tests use a distinct fill value per iov (1.0, 2.0, …) so that
misrouted copies are caught even if total byte counts match.

Run with:
    torchrun --nproc_per_node=2 tests/test_engine_onesided_ipc.py
"""

import struct
import sys
import torch
import torch.distributed as dist

try:
    from uccl import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    sys.exit(1)

NUM_IOVS = 4
BUF_ELEMS = 1024  # float32 elements → 4 KB per buffer


# ── torch.distributed helpers ─────────────────────────────────────────────────


def _send_int(value: int, dst: int):
    t = torch.tensor([int(value)], dtype=torch.uint64)
    dist.send(t, dst=dst)


def _recv_int(src: int) -> int:
    t = torch.empty(1, dtype=torch.uint64)
    dist.recv(t, src=src)
    return int(t.item())


def _send_bytes(payload: bytes, dst: int):
    n = len(payload)
    _send_int(n, dst)
    if n == 0:
        return
    buf = torch.frombuffer(memoryview(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    n = _recv_int(src)
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def _poll_done(ep, transfer_id):
    is_done = False
    while not is_done:
        ok, is_done = ep.poll_async(transfer_id)
        assert ok, "poll_async failed"


def _unpack_info_blobs(packed: bytes, num_iovs: int):
    blob_size = (len(packed) - 4) // num_iovs
    return [packed[4 + i * blob_size: 4 + (i + 1) * blob_size] for i in range(num_iovs)]


# ── scalar write_ipc (sync) ───────────────────────────────────────────────────


def test_write_ipc(ep, conn_id, rank):
    """
    Client fills source with 1.0 and writes it into server's zero-filled buffer.
    Server verifies its destination buffer now contains 1.0 everywhere.
    """
    size = BUF_ELEMS * 4
    if rank == 0:  # server — owns destination
        dst = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        ok, info_blob = ep.advertise_ipc(conn_id, dst.data_ptr(), size)
        assert ok, "advertise_ipc failed"
        _send_bytes(bytes(info_blob), dst=1)
        _recv_int(src=1)  # wait for write to complete
        expected = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        assert dst.allclose(expected), f"write_ipc: destination mismatch\n{dst}"
        print("[server] test_write_ipc PASSED")
    else:  # client — source data
        src = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        info_blob = _recv_bytes(src=0)
        ok = ep.write_ipc(conn_id, src.data_ptr(), size, info_blob)
        assert ok, "write_ipc failed"
        _send_int(1, dst=0)


# ── scalar write_ipc_async ────────────────────────────────────────────────────


def test_write_ipc_async(ep, conn_id, rank):
    """Same as test_write_ipc but uses the async API + poll_async."""
    size = BUF_ELEMS * 4
    if rank == 0:
        dst = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        ok, info_blob = ep.advertise_ipc(conn_id, dst.data_ptr(), size)
        assert ok, "advertise_ipc failed"
        _send_bytes(bytes(info_blob), dst=1)
        _recv_int(src=1)
        expected = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        assert dst.allclose(expected), f"write_ipc_async: destination mismatch\n{dst}"
        print("[server] test_write_ipc_async PASSED")
    else:
        src = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        info_blob = _recv_bytes(src=0)
        ok, transfer_id = ep.write_ipc_async(conn_id, src.data_ptr(), size, info_blob)
        assert ok, "write_ipc_async failed"
        _poll_done(ep, transfer_id)
        _send_int(1, dst=0)


# ── scalar read_ipc (sync) ────────────────────────────────────────────────────


def test_read_ipc(ep, conn_id, rank):
    """
    Server fills its buffer with 1.0. Client reads it into a zero-filled local
    buffer and verifies the destination now contains 1.0 everywhere.
    """
    size = BUF_ELEMS * 4
    if rank == 0:  # server — owns source
        src = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        ok, info_blob = ep.advertise_ipc(conn_id, src.data_ptr(), size)
        assert ok, "advertise_ipc failed"
        _send_bytes(bytes(info_blob), dst=1)
        _recv_int(src=1)
        print("[server] test_read_ipc PASSED")
    else:  # client — destination
        dst = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        info_blob = _recv_bytes(src=0)
        ok = ep.read_ipc(conn_id, dst.data_ptr(), size, info_blob)
        assert ok, "read_ipc failed"
        expected = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        assert dst.allclose(expected), f"read_ipc: destination mismatch\n{dst}"
        print("[client] test_read_ipc PASSED")
        _send_int(1, dst=0)


# ── scalar read_ipc_async ─────────────────────────────────────────────────────


def test_read_ipc_async(ep, conn_id, rank):
    """Same as test_read_ipc but uses the async API + poll_async."""
    size = BUF_ELEMS * 4
    if rank == 0:
        src = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        ok, info_blob = ep.advertise_ipc(conn_id, src.data_ptr(), size)
        assert ok, "advertise_ipc failed"
        _send_bytes(bytes(info_blob), dst=1)
        _recv_int(src=1)
        print("[server] test_read_ipc_async PASSED")
    else:
        dst = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        info_blob = _recv_bytes(src=0)
        ok, transfer_id = ep.read_ipc_async(conn_id, dst.data_ptr(), size, info_blob)
        assert ok, "read_ipc_async failed"
        _poll_done(ep, transfer_id)
        expected = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
        assert dst.allclose(expected), f"read_ipc_async: destination mismatch\n{dst}"
        print("[client] test_read_ipc_async PASSED")
        _send_int(1, dst=0)


# ── vectorized writev_ipc (sync) ──────────────────────────────────────────────


def test_writev_ipc(ep, conn_id, rank):
    """
    Client fills iov i with float(i+1) and writes all NUM_IOVS buffers into
    the server's zero-filled destination buffers. Server verifies each
    destination[i] contains float(i+1), catching any misrouted copies.
    """
    size_per = BUF_ELEMS * 4
    if rank == 0:  # server — owns destinations
        dsts = [torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
                for _ in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in dsts]
        ok, info_blobs = ep.advertisev_ipc(conn_id, ptrs, [size_per] * NUM_IOVS)
        assert ok, "advertisev_ipc failed"
        packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in info_blobs)
        _send_bytes(packed, dst=1)
        _recv_int(src=1)
        for i, dst in enumerate(dsts):
            expected = torch.full((BUF_ELEMS,), float(i + 1),
                                  dtype=torch.float32, device="cuda:0")
            assert dst.allclose(expected), \
                f"writev_ipc: destination[{i}] mismatch (expected {i+1}.0)\n{dst}"
        print(f"[server] test_writev_ipc PASSED (num_iovs={NUM_IOVS})")
    else:  # client — source data
        srcs = [torch.full((BUF_ELEMS,), float(i + 1), dtype=torch.float32, device="cuda:0")
                for i in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in srcs]
        packed = _recv_bytes(src=0)
        info_blobs = _unpack_info_blobs(packed, NUM_IOVS)
        ok = ep.writev_ipc(conn_id, ptrs, [size_per] * NUM_IOVS, info_blobs)
        assert ok, "writev_ipc failed"
        _send_int(1, dst=0)


# ── vectorized writev_ipc_async ───────────────────────────────────────────────


def test_writev_ipc_async(ep, conn_id, rank):
    """Same as test_writev_ipc but uses the async API + poll_async."""
    size_per = BUF_ELEMS * 4
    if rank == 0:
        dsts = [torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
                for _ in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in dsts]
        ok, info_blobs = ep.advertisev_ipc(conn_id, ptrs, [size_per] * NUM_IOVS)
        assert ok, "advertisev_ipc failed"
        packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in info_blobs)
        _send_bytes(packed, dst=1)
        _recv_int(src=1)
        for i, dst in enumerate(dsts):
            expected = torch.full((BUF_ELEMS,), float(i + 1),
                                  dtype=torch.float32, device="cuda:0")
            assert dst.allclose(expected), \
                f"writev_ipc_async: destination[{i}] mismatch (expected {i+1}.0)\n{dst}"
        print(f"[server] test_writev_ipc_async PASSED (num_iovs={NUM_IOVS})")
    else:
        srcs = [torch.full((BUF_ELEMS,), float(i + 1), dtype=torch.float32, device="cuda:0")
                for i in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in srcs]
        packed = _recv_bytes(src=0)
        info_blobs = _unpack_info_blobs(packed, NUM_IOVS)
        ok, transfer_id = ep.writev_ipc_async(conn_id, ptrs, [size_per] * NUM_IOVS, info_blobs)
        assert ok, "writev_ipc_async failed"
        _poll_done(ep, transfer_id)
        _send_int(1, dst=0)


# ── vectorized readv_ipc (sync) ───────────────────────────────────────────────


def test_readv_ipc(ep, conn_id, rank):
    """
    Server fills source[i] with float(i+1). Client reads all NUM_IOVS buffers
    into zero-filled local buffers and verifies destination[i] == float(i+1),
    catching any misrouted copies.
    """
    size_per = BUF_ELEMS * 4
    if rank == 0:  # server — owns sources
        srcs = [torch.full((BUF_ELEMS,), float(i + 1), dtype=torch.float32, device="cuda:0")
                for i in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in srcs]
        ok, info_blobs = ep.advertisev_ipc(conn_id, ptrs, [size_per] * NUM_IOVS)
        assert ok, "advertisev_ipc failed"
        packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in info_blobs)
        _send_bytes(packed, dst=1)
        _recv_int(src=1)
        print(f"[server] test_readv_ipc PASSED (num_iovs={NUM_IOVS})")
    else:  # client — destinations
        dsts = [torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
                for _ in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in dsts]
        packed = _recv_bytes(src=0)
        info_blobs = _unpack_info_blobs(packed, NUM_IOVS)
        ok = ep.readv_ipc(conn_id, ptrs, [size_per] * NUM_IOVS, info_blobs)
        assert ok, "readv_ipc failed"
        for i, dst in enumerate(dsts):
            expected = torch.full((BUF_ELEMS,), float(i + 1),
                                  dtype=torch.float32, device="cuda:0")
            assert dst.allclose(expected), \
                f"readv_ipc: destination[{i}] mismatch (expected {i+1}.0)\n{dst}"
        print(f"[client] test_readv_ipc PASSED (num_iovs={NUM_IOVS})")
        _send_int(1, dst=0)


# ── vectorized readv_ipc_async ────────────────────────────────────────────────


def test_readv_ipc_async(ep, conn_id, rank):
    """Same as test_readv_ipc but uses the async API + poll_async."""
    size_per = BUF_ELEMS * 4
    if rank == 0:
        srcs = [torch.full((BUF_ELEMS,), float(i + 1), dtype=torch.float32, device="cuda:0")
                for i in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in srcs]
        ok, info_blobs = ep.advertisev_ipc(conn_id, ptrs, [size_per] * NUM_IOVS)
        assert ok, "advertisev_ipc failed"
        packed = struct.pack("I", NUM_IOVS) + b"".join(bytes(b) for b in info_blobs)
        _send_bytes(packed, dst=1)
        _recv_int(src=1)
        print(f"[server] test_readv_ipc_async PASSED (num_iovs={NUM_IOVS})")
    else:
        dsts = [torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
                for _ in range(NUM_IOVS)]
        ptrs = [t.data_ptr() for t in dsts]
        packed = _recv_bytes(src=0)
        info_blobs = _unpack_info_blobs(packed, NUM_IOVS)
        ok, transfer_id = ep.readv_ipc_async(conn_id, ptrs, [size_per] * NUM_IOVS, info_blobs)
        assert ok, "readv_ipc_async failed"
        _poll_done(ep, transfer_id)
        for i, dst in enumerate(dsts):
            expected = torch.full((BUF_ELEMS,), float(i + 1),
                                  dtype=torch.float32, device="cuda:0")
            assert dst.allclose(expected), \
                f"readv_ipc_async: destination[{i}] mismatch (expected {i+1}.0)\n{dst}"
        print(f"[client] test_readv_ipc_async PASSED (num_iovs={NUM_IOVS})")
        _send_int(1, dst=0)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "This test requires exactly 2 processes"

    torch.cuda.set_device(0)
    ep = p2p.Endpoint(local_gpu_idx=rank, num_cpus=4)

    print(f"=== UCCL One-Sided IPC Tests (rank {rank}) ===")

    if rank == 0:
        ok, remote_gpu_idx, conn_id = ep.accept_local()
        assert ok, "accept_local failed"
    else:
        ok, conn_id = ep.connect_local(remote_gpu_idx=0)
        assert ok, "connect_local failed"

    tests = [
        ("write_ipc",        test_write_ipc),
        ("write_ipc_async",  test_write_ipc_async),
        ("read_ipc",         test_read_ipc),
        ("read_ipc_async",   test_read_ipc_async),
        ("writev_ipc",       test_writev_ipc),
        ("writev_ipc_async", test_writev_ipc_async),
        ("readv_ipc",        test_readv_ipc),
        ("readv_ipc_async",  test_readv_ipc_async),
    ]

    for name, fn in tests:
        dist.barrier()
        if rank == 0:
            print(f"\n--- {name} ---")
        fn(ep, conn_id, rank)

    dist.barrier()
    if rank == 0:
        print("\nAll one-sided IPC tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
