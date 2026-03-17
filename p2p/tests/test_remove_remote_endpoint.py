#!/usr/bin/env python3
"""
Test script for remove_remote_endpoint API in UCCL P2P Engine.

Single-process tests (no torch.distributed required):
  - test_remove_invalid_conn_id
  - test_remove_double_remove

Two-process tests (requires torchrun --nproc_per_node=2):
  - test_remove_after_add
  - test_remove_and_readd
  - test_remove_then_transfer_fails
"""

from __future__ import annotations

import sys
import torch

try:
    import torch.distributed as dist
except ImportError:
    dist = None

try:
    from uccl import p2p
except ImportError as exc:
    sys.stderr.write("Failed to import p2p\n")
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send_bytes(payload: bytes, dst: int):
    n = len(payload)
    t_size = torch.tensor([n], dtype=torch.int64)
    dist.send(t_size, dst=dst)
    if n == 0:
        return
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    t_size = torch.empty(1, dtype=torch.int64)
    dist.recv(t_size, src=src)
    n = int(t_size.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


def test_remove_invalid_conn_id():
    """remove_remote_endpoint with a bogus conn_id should return False."""
    print("=" * 60)
    print("Test: remove_remote_endpoint with invalid conn_id")
    print("=" * 60)

    ep = p2p.Endpoint(0)
    ep.start_passive_accept()

    result = ep.remove_remote_endpoint(999999)
    assert result is False, f"Expected False for invalid conn_id, got {result}"

    print("PASSED: correctly returned False for invalid conn_id")
    return True


def test_remove_double_remove():
    """Removing the same conn_id twice should fail the second time."""
    print("\n" + "=" * 60)
    print("Test: double remove_remote_endpoint")
    print("=" * 60)

    if not dist.is_initialized() or dist.get_world_size() < 2:
        print("Skipping: requires 2 processes")
        return True

    rank = dist.get_rank()
    peer = 1 - rank

    ep = p2p.Endpoint(0)
    ep.start_passive_accept()

    local_meta = ep.get_metadata()
    if rank == 0:
        _send_bytes(local_meta, dst=peer)
        remote_meta = _recv_bytes(src=peer)
    else:
        remote_meta = _recv_bytes(src=peer)
        _send_bytes(local_meta, dst=peer)

    success, conn_id = ep.add_remote_endpoint(remote_meta)
    assert success, "Failed to add remote endpoint"

    dist.barrier()

    # First remove should succeed
    ok1 = ep.remove_remote_endpoint(conn_id)
    assert ok1 is True, f"First remove failed, got {ok1}"

    # Second remove should fail (already removed)
    ok2 = ep.remove_remote_endpoint(conn_id)
    assert ok2 is False, f"Second remove should fail, got {ok2}"

    dist.barrier()
    print("PASSED: double remove correctly fails the second time")
    return True


# ---------------------------------------------------------------------------
# Two-process tests
# ---------------------------------------------------------------------------


def test_remove_after_add():
    """Add then remove a remote endpoint; verify remove returns True."""
    print("\n" + "=" * 60)
    print("Test: add then remove remote endpoint")
    print("=" * 60)

    if not dist.is_initialized() or dist.get_world_size() < 2:
        print("Skipping: requires 2 processes")
        return True

    rank = dist.get_rank()
    peer = 1 - rank

    ep = p2p.Endpoint(0)
    ep.start_passive_accept()

    local_meta = ep.get_metadata()
    if rank == 0:
        _send_bytes(local_meta, dst=peer)
        remote_meta = _recv_bytes(src=peer)
    else:
        remote_meta = _recv_bytes(src=peer)
        _send_bytes(local_meta, dst=peer)

    success, conn_id = ep.add_remote_endpoint(remote_meta)
    assert success, "add_remote_endpoint failed"
    print(f"  [rank {rank}] added conn_id={conn_id}")

    dist.barrier()

    ok = ep.remove_remote_endpoint(conn_id)
    assert ok is True, f"remove_remote_endpoint failed, got {ok}"
    print(f"  [rank {rank}] removed conn_id={conn_id}")

    dist.barrier()
    print("PASSED: add then remove works correctly")
    return True


def test_remove_and_readd():
    """Remove then re-add the same remote endpoint; should get a new conn_id."""
    print("\n" + "=" * 60)
    print("Test: remove then re-add remote endpoint")
    print("=" * 60)

    if not dist.is_initialized() or dist.get_world_size() < 2:
        print("Skipping: requires 2 processes")
        return True

    rank = dist.get_rank()
    peer = 1 - rank

    ep = p2p.Endpoint(0)
    ep.start_passive_accept()

    local_meta = ep.get_metadata()
    if rank == 0:
        _send_bytes(local_meta, dst=peer)
        remote_meta = _recv_bytes(src=peer)
    else:
        remote_meta = _recv_bytes(src=peer)
        _send_bytes(local_meta, dst=peer)

    # First add
    success, conn_id1 = ep.add_remote_endpoint(remote_meta)
    assert success, "First add failed"
    print(f"  [rank {rank}] first conn_id={conn_id1}")

    dist.barrier()

    # Remove
    ok = ep.remove_remote_endpoint(conn_id1)
    assert ok is True, "remove failed"

    dist.barrier()

    # Re-add should succeed and return a new conn_id
    success, conn_id2 = ep.add_remote_endpoint(remote_meta)
    assert success, "Re-add failed"
    print(f"  [rank {rank}] second conn_id={conn_id2}")
    assert conn_id2 != conn_id1, (
        f"Expected different conn_id after remove+readd, " f"got {conn_id1} both times"
    )

    dist.barrier()

    # Clean up
    ep.remove_remote_endpoint(conn_id2)

    dist.barrier()
    print("PASSED: remove then re-add produces new conn_id")
    return True


def test_remove_then_transfer_fails():
    """After removing a remote endpoint, transfer using the old conn_id should fail."""
    print("\n" + "=" * 60)
    print("Test: transfer after remove_remote_endpoint should fail")
    print("=" * 60)

    if not dist.is_initialized() or dist.get_world_size() < 2:
        print("Skipping: requires 2 processes")
        return True

    rank = dist.get_rank()
    peer = 1 - rank

    ep = p2p.Endpoint(0)
    ep.start_passive_accept()

    local_meta = ep.get_metadata()
    if rank == 0:
        _send_bytes(local_meta, dst=peer)
        remote_meta = _recv_bytes(src=peer)
    else:
        remote_meta = _recv_bytes(src=peer)
        _send_bytes(local_meta, dst=peer)

    success, conn_id = ep.add_remote_endpoint(remote_meta)
    assert success, "add_remote_endpoint failed"

    # Register some memory
    pin = torch.cuda.is_available()
    buf = torch.ones(256, dtype=torch.float32, pin_memory=pin)
    local_descs = ep.register_memory([buf])

    dist.barrier()

    # Exchange descriptors so we have remote_descs
    local_serialized = ep.get_serialized_descs(local_descs)
    if rank == 0:
        _send_bytes(local_serialized, dst=peer)
        remote_serialized = _recv_bytes(src=peer)
    else:
        remote_serialized = _recv_bytes(src=peer)
        _send_bytes(local_serialized, dst=peer)
    remote_descs = ep.deserialize_descs(remote_serialized)

    dist.barrier()

    # Remove the remote endpoint
    ok = ep.remove_remote_endpoint(conn_id)
    assert ok is True, "remove failed"

    dist.barrier()

    # Now try to transfer using the removed conn_id — should raise invalid conn_id
    try:
        ep.transfer(conn_id, "write", local_descs, remote_descs)
        raise AssertionError(
            "Expected transfer to raise RuntimeError after removing remote endpoint"
        )
    except RuntimeError as e:
        assert "Invalid conn_id" in str(e), f"Expected Invalid conn_id error, got: {e}"

    ep.deregister_memory(local_descs)

    dist.barrier()
    print("PASSED: transfer correctly fails after remove_remote_endpoint")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("UCCL P2P Engine - remove_remote_endpoint Tests")
    print("=" * 60)

    if dist is not None and not dist.is_initialized():
        try:
            dist.init_process_group(backend="gloo")
        except Exception:
            pass  # single-process run

    tests = [
        ("remove_invalid_conn_id", test_remove_invalid_conn_id),
        ("remove_after_add", test_remove_after_add),
        ("remove_and_readd", test_remove_and_readd),
        ("remove_double_remove", test_remove_double_remove),
        ("remove_then_transfer_fails", test_remove_then_transfer_fails),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            if func():
                passed += 1
            else:
                failed += 1
                print(f"FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"FAILED: {name} with {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(1)
