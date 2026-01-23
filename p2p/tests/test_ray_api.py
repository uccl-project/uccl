#!/usr/bin/env python3
"""
Test script for register_memory API in UCCL P2P Engine.
This script tests the register_memory method that accepts a list of PyTorch tensors.
"""

from __future__ import annotations

import sys
import os
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


def test_register_memory_basic():
    """Test basic register_memory functionality with single tensor."""
    print("=" * 60)
    print("Test 1: Basic register_memory with single tensor")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create a single tensor
    tensor = torch.ones(1024, dtype=torch.float32, device="cuda:0")
    print(
        f"Created tensor: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
    )
    print(f"Tensor data pointer: {tensor.data_ptr()}")
    print(f"Tensor size in bytes: {tensor.numel() * tensor.element_size()}")

    # Register memory
    descriptors = ep.register_memory([tensor])
    print(f"register_memory returned {len(descriptors)} descriptor(s)")

    # Verify descriptor structure
    assert len(descriptors) == 1, f"Expected 1 descriptor, got {len(descriptors)}"
    desc = descriptors[0]

    assert "addr" in desc, "Descriptor missing 'addr' field"
    assert "size" in desc, "Descriptor missing 'size' field"
    assert "lkeys" in desc, "Descriptor missing 'lkeys' field"
    assert "rkeys" in desc, "Descriptor missing 'rkeys' field"

    print(f"Descriptor addr: {desc['addr']}")
    print(f"Descriptor size: {desc['size']}")
    print(f"Descriptor lkeys: {desc['lkeys']}")
    print(f"Descriptor rkeys: {desc['rkeys']}")

    # Verify address matches
    expected_addr = tensor.data_ptr()
    assert (
        desc["addr"] == expected_addr
    ), f"Address mismatch: {desc['addr']} != {expected_addr}"

    # Verify size matches
    expected_size = tensor.numel() * tensor.element_size()
    assert (
        desc["size"] == expected_size
    ), f"Size mismatch: {desc['size']} != {expected_size}"

    # Verify lkeys and rkeys are lists
    assert isinstance(desc["lkeys"], list), "lkeys should be a list"
    assert isinstance(desc["rkeys"], list), "rkeys should be a list"
    assert len(desc["lkeys"]) > 0, "lkeys should not be empty"
    assert len(desc["rkeys"]) > 0, "rkeys should not be empty"
    assert len(desc["lkeys"]) == len(
        desc["rkeys"]
    ), "lkeys and rkeys should have same length"

    print("✓ Test 1 passed: Basic register_memory works correctly")
    return True


def test_register_memory_multiple():
    """Test register_memory with multiple tensors."""
    print("\n" + "=" * 60)
    print("Test 2: register_memory with multiple tensors")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create multiple tensors with different sizes
    tensors = [
        torch.ones(512, dtype=torch.float32, device="cuda:0"),
        torch.zeros(1024, dtype=torch.float32, device="cuda:0"),
        torch.randn(256, dtype=torch.float32, device="cuda:0"),
        torch.ones(2048, dtype=torch.float16, device="cuda:0"),
    ]

    print(f"Created {len(tensors)} tensors:")
    for i, t in enumerate(tensors):
        size_bytes = t.numel() * t.element_size()
        print(
            f"  Tensor {i}: shape={t.shape}, dtype={t.dtype}, size={size_bytes} bytes, ptr={t.data_ptr()}"
        )

    # Register memory
    descriptors = ep.register_memory(tensors)
    print(f"\nregister_memory returned {len(descriptors)} descriptor(s)")

    # Verify we got one descriptor per tensor
    assert len(descriptors) == len(
        tensors
    ), f"Expected {len(tensors)} descriptors, got {len(descriptors)}"

    # Verify each descriptor
    for i, (tensor, desc) in enumerate(zip(tensors, descriptors)):
        expected_addr = tensor.data_ptr()
        expected_size = tensor.numel() * tensor.element_size()

        assert (
            desc["addr"] == expected_addr
        ), f"Tensor {i}: Address mismatch: {desc['addr']} != {expected_addr}"
        assert (
            desc["size"] == expected_size
        ), f"Tensor {i}: Size mismatch: {desc['size']} != {expected_size}"
        assert len(desc["lkeys"]) > 0, f"Tensor {i}: lkeys should not be empty"
        assert len(desc["rkeys"]) > 0, f"Tensor {i}: rkeys should not be empty"

        print(
            f"  Descriptor {i}: addr={desc['addr']}, size={desc['size']}, "
            f"lkeys={len(desc['lkeys'])}, rkeys={len(desc['rkeys'])}"
        )

    print("✓ Test 2 passed: Multiple tensor registration works correctly")
    return True


def test_register_memory_empty_list():
    """Test register_memory with empty list."""
    print("\n" + "=" * 60)
    print("Test 3: register_memory with empty list")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Register empty list
    descriptors = ep.register_memory([])
    print(f"register_memory returned {len(descriptors)} descriptor(s)")

    # Should return empty list
    assert (
        len(descriptors) == 0
    ), f"Expected empty list, got {len(descriptors)} descriptors"

    print("✓ Test 3 passed: Empty list handling works correctly")
    return True


def test_register_memory_invalid_input():
    """Test register_memory with invalid input (non-tensor object)."""
    print("\n" + "=" * 60)
    print("Test 4: register_memory with invalid input")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Try to register a non-tensor object
    try:
        descriptors = ep.register_memory([123])  # Not a tensor
        print("ERROR: Should have raised an exception for non-tensor input")
        return False
    except (RuntimeError, AttributeError) as e:
        print(
            f"✓ Correctly raised exception for invalid input: {type(e).__name__}: {e}"
        )
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception type: {type(e).__name__}: {e}")
        return False


def test_register_memory_mixed_valid_invalid():
    """Test register_memory with mixed valid and invalid inputs."""
    print("\n" + "=" * 60)
    print("Test 5: register_memory with mixed valid/invalid inputs")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create valid tensor
    tensor = torch.ones(1024, dtype=torch.float32, device="cuda:0")

    # Try to register mixed list
    try:
        descriptors = ep.register_memory([tensor, "not a tensor"])
        print("ERROR: Should have raised an exception for mixed input")
        return False
    except (RuntimeError, AttributeError) as e:
        print(f"✓ Correctly raised exception for mixed input: {type(e).__name__}: {e}")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception type: {type(e).__name__}: {e}")
        return False


def test_register_memory_different_dtypes():
    """Test register_memory with tensors of different dtypes."""
    print("\n" + "=" * 60)
    print("Test 6: register_memory with different dtypes")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create tensors with different dtypes
    tensors = [
        torch.ones(1024, dtype=torch.float32, device="cuda:0"),
        torch.ones(1024, dtype=torch.float16, device="cuda:0"),
        torch.ones(1024, dtype=torch.int32, device="cuda:0"),
        torch.ones(1024, dtype=torch.int64, device="cuda:0"),
    ]

    print(f"Created {len(tensors)} tensors with different dtypes:")
    for i, t in enumerate(tensors):
        size_bytes = t.numel() * t.element_size()
        print(f"  Tensor {i}: dtype={t.dtype}, size={size_bytes} bytes")

    # Register memory
    descriptors = ep.register_memory(tensors)
    print(f"\nregister_memory returned {len(descriptors)} descriptor(s)")

    # Verify we got one descriptor per tensor
    assert len(descriptors) == len(
        tensors
    ), f"Expected {len(tensors)} descriptors, got {len(descriptors)}"

    # Verify each descriptor has correct size
    for i, (tensor, desc) in enumerate(zip(tensors, descriptors)):
        expected_size = tensor.numel() * tensor.element_size()
        assert (
            desc["size"] == expected_size
        ), f"Tensor {i}: Size mismatch: {desc['size']} != {expected_size}"
        print(f"  Descriptor {i}: dtype={tensor.dtype}, size={desc['size']} bytes")

    print("✓ Test 6 passed: Different dtype handling works correctly")
    return True


def test_get_serialized_descs():
    """Test get_serialized_descs functionality."""
    print("\n" + "=" * 60)
    print("Test 7: get_serialized_descs")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create tensors
    tensors = [
        torch.ones(1024, dtype=torch.float32, device="cuda:0"),
        torch.zeros(512, dtype=torch.float32, device="cuda:0"),
    ]

    # Register memory
    descriptors = ep.register_memory(tensors)
    print(f"Registered {len(descriptors)} tensor(s)")

    # Serialize descriptors
    serialized = ep.get_serialized_descs(descriptors)
    print(f"Serialized to {len(serialized)} bytes")

    # Verify serialized data is not empty
    assert len(serialized) > 0, "Serialized data should not be empty"
    assert isinstance(serialized, bytes), "Serialized data should be bytes"

    print(f"✓ Test 7 passed: get_serialized_descs works correctly")
    return True


def test_deserialize_descs():
    """Test deserialize_descs functionality."""
    print("\n" + "=" * 60)
    print("Test 8: deserialize_descs")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create tensors
    tensors = [
        torch.ones(1024, dtype=torch.float32, device="cuda:0"),
        torch.zeros(512, dtype=torch.float32, device="cuda:0"),
        torch.randn(256, dtype=torch.float32, device="cuda:0"),
    ]

    # Register memory and get descriptors
    original_descriptors = ep.register_memory(tensors)
    print(f"Registered {len(original_descriptors)} tensor(s)")

    # Serialize descriptors
    serialized = ep.get_serialized_descs(original_descriptors)
    print(f"Serialized to {len(serialized)} bytes")

    # Deserialize descriptors
    deserialized_descriptors = ep.deserialize_descs(serialized)
    print(f"Deserialized {len(deserialized_descriptors)} descriptor(s)")

    # Verify count matches
    assert len(deserialized_descriptors) == len(
        original_descriptors
    ), f"Descriptor count mismatch: {len(deserialized_descriptors)} != {len(original_descriptors)}"

    # Verify each descriptor matches
    for i, (orig, deser) in enumerate(
        zip(original_descriptors, deserialized_descriptors)
    ):
        assert (
            orig["addr"] == deser["addr"]
        ), f"Descriptor {i}: addr mismatch: {orig['addr']} != {deser['addr']}"
        assert (
            orig["size"] == deser["size"]
        ), f"Descriptor {i}: size mismatch: {orig['size']} != {deser['size']}"
        assert (
            orig["lkeys"] == deser["lkeys"]
        ), f"Descriptor {i}: lkeys mismatch: {orig['lkeys']} != {deser['lkeys']}"
        assert (
            orig["rkeys"] == deser["rkeys"]
        ), f"Descriptor {i}: rkeys mismatch: {orig['rkeys']} != {deser['rkeys']}"
        print(
            f"  Descriptor {i}: verified (addr={orig['addr']}, size={orig['size']}, "
            f"keys={len(orig['lkeys'])})"
        )

    print(f"✓ Test 8 passed: deserialize_descs works correctly")
    return True


def test_serialize_deserialize_roundtrip():
    """Test roundtrip serialization and deserialization."""
    print("\n" + "=" * 60)
    print("Test 9: serialize/deserialize roundtrip")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    # Create endpoint
    ep = p2p.Endpoint(4, True)
    print(f"Created Endpoint with 4 CPUs")

    # Create tensors with different sizes and dtypes
    # Note: Use ones/zeros instead of randn for integer types (randn doesn't support int types)
    tensors = [
        torch.ones(1024, dtype=torch.float32, device="cuda:0"),
        torch.zeros(512, dtype=torch.float16, device="cuda:0"),
        torch.ones(256, dtype=torch.int32, device="cuda:0"),
    ]

    # Register memory
    original_descriptors = ep.register_memory(tensors)
    print(f"Registered {len(original_descriptors)} tensor(s)")

    # Serialize
    serialized = ep.get_serialized_descs(original_descriptors)
    print(f"Serialized to {len(serialized)} bytes")

    # Deserialize
    deserialized = ep.deserialize_descs(serialized)
    print(f"Deserialized {len(deserialized)} descriptor(s)")

    # Verify roundtrip
    assert len(deserialized) == len(
        original_descriptors
    ), "Roundtrip failed: descriptor count mismatch"

    for i, (orig, deser) in enumerate(zip(original_descriptors, deserialized)):
        assert orig == deser, f"Roundtrip failed for descriptor {i}: {orig} != {deser}"

    # Test with empty list
    empty_serialized = ep.get_serialized_descs([])
    empty_deserialized = ep.deserialize_descs(empty_serialized)
    assert len(empty_deserialized) == 0, "Empty list roundtrip failed"

    print(f"✓ Test 9 passed: serialize/deserialize roundtrip works correctly")
    return True


def _send_bytes(payload: bytes, dst: int):
    """Send bytes via PyTorch distributed."""
    n = len(payload)
    t_size = torch.tensor([n], dtype=torch.int64)
    dist.send(t_size, dst=dst)
    if n == 0:
        return
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    """Receive bytes via PyTorch distributed."""
    t_size = torch.empty(1, dtype=torch.int64)
    dist.recv(t_size, src=src)
    n = int(t_size.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def test_transfer():
    """Test transfer functionality with two processes."""
    print("\n" + "=" * 60)
    print("Test 10: transfer")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return True

    if dist is None:
        print("Skipping test: torch.distributed not available")
        return True

    # Initialize distributed if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        print("Skipping test: requires at least 2 processes")
        return True

    # rank 0 = client, rank 1 = server
    if rank == 0:
        # Client side
        print("[Client] Creating endpoint...")
        ep = p2p.Endpoint(4, True)

        # Get local metadata
        local_metadata = ep.get_metadata()
        print(f"[Client] Got local metadata, size={len(local_metadata)}")

        # Exchange metadata with server
        _send_bytes(local_metadata, dst=1)
        remote_metadata = _recv_bytes(src=1)
        print(f"[Client] Received remote metadata, size={len(remote_metadata)}")

        # Add remote endpoint (connect to server)
        success, conn_id = ep.add_remote_endpoint(remote_metadata)
        assert success, "Failed to add remote endpoint"
        print(f"[Client] Connected to server, conn_id={conn_id}")

        # Create source tensor with "hello uccl" content
        message = "hello uccl"
        message_bytes = message.encode("utf-8")
        # Create a tensor large enough to hold the message
        # Use uint8 to store bytes
        src_tensor = torch.zeros(256, dtype=torch.uint8, device="cuda:0")
        # Copy message bytes to tensor (convert to tensor first, then copy)
        message_tensor = torch.tensor(
            list(message_bytes), dtype=torch.uint8, device="cuda:0"
        )
        src_tensor[: len(message_bytes)] = message_tensor
        src_tensors = [src_tensor]
        print(
            f"[Client] Created source tensor with message: '{message}' ({len(message_bytes)} bytes)"
        )

        # Register local memory
        local_descs = ep.register_memory(src_tensors)
        print(f"[Client] Registered {len(local_descs)} local descriptor(s)")

        # Wait for server to send remote descriptors
        remote_descs_serialized = _recv_bytes(src=1)
        print(
            f"[Client] Received remote descriptors, size={len(remote_descs_serialized)}"
        )

        # Deserialize remote descriptors
        remote_descs = ep.deserialize_descs(remote_descs_serialized)
        print(f"[Client] Deserialized {len(remote_descs)} remote descriptor(s)")

        # Start transfer (WRITE operation)
        xfer_handle = ep.trasnfer(conn_id, "WRITE", local_descs, remote_descs)
        assert xfer_handle is not None, "Failed to start transfer"
        print(f"[Client] Started WRITE transfer")

        # Check transfer state until complete
        max_wait = 100
        wait_count = 0
        while wait_count < max_wait:
            is_done = ep.check_xfer_state(xfer_handle)
            if is_done:
                print(f"[Client] Transfer completed")
                break
            wait_count += 1
            import time

            time.sleep(0.01)

        assert is_done, "Transfer did not complete in time"
        print("✓ Test 10 passed: transfer works correctly")

    elif rank == 1:
        # Server side
        print("[Server] Creating endpoint...")
        ep = p2p.Endpoint(4, True)  # passive_accept=True

        # Get local metadata
        local_metadata = ep.get_metadata()
        print(f"[Server] Got local metadata, size={len(local_metadata)}")

        # Exchange metadata with client
        remote_metadata = _recv_bytes(src=0)
        _send_bytes(local_metadata, dst=0)
        print(f"[Server] Exchanged metadata with client")

        # Server doesn't need to call accept (passive_accept=True handles it)
        # Wait a bit for connection to establish
        import time

        time.sleep(0.5)  # Give time for connection to establish
        print("[Server] Connection should be established (passive accept)")

        # Create destination tensor for receiving
        # Use uint8 to match source tensor
        dst_tensor = torch.zeros(256, dtype=torch.uint8, device="cuda:0")
        dst_tensors = [dst_tensor]
        print(f"[Server] Created destination tensor for receiving")

        # Register remote memory
        remote_descs = ep.register_memory(dst_tensors)
        print(f"[Server] Registered {len(remote_descs)} remote descriptor(s)")

        # Serialize and send remote descriptors to client
        remote_descs_serialized = ep.get_serialized_descs(remote_descs)
        _send_bytes(remote_descs_serialized, dst=0)
        print(
            f"[Server] Sent remote descriptors to client, size={len(remote_descs_serialized)}"
        )

        # Wait for transfer to complete (client will poll and complete)
        time.sleep(3)

        # Verify data was written correctly
        # Read the content from GPU memory
        received_data = dst_tensor.cpu().numpy()
        # Extract the message bytes (first 10 bytes for "hello uccl")
        expected_message = "hello uccl"
        message_length = len(expected_message)
        received_message_bytes = bytes(received_data[:message_length])
        received_message = received_message_bytes.decode("utf-8")

        assert (
            received_message == expected_message
        ), f"Message mismatch: expected '{expected_message}', got '{received_message}' (bytes: {received_message_bytes})"
        print(f"[Server] Verified data was written correctly: '{received_message}'")
        print("✓ Test 10 passed: transfer works correctly")

    return True


def main():
    """Run all tests."""
    print("UCCL P2P Engine - Ray API Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Some tests will be skipped.")
        print("This test requires CUDA-enabled PyTorch.")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")

    print("\n")

    tests = [
        # ("Basic register_memory", test_register_memory_basic),
        # ("Multiple tensors", test_register_memory_multiple),
        # ("Empty list", test_register_memory_empty_list),
        # ("Invalid input", test_register_memory_invalid_input),
        # ("Mixed valid/invalid", test_register_memory_mixed_valid_invalid),
        # ("Different dtypes", test_register_memory_different_dtypes),
        # ("get_serialized_descs", test_get_serialized_descs),
        # ("deserialize_descs", test_deserialize_descs),
        # ("serialize/deserialize roundtrip", test_serialize_deserialize_roundtrip),
        ("transfer", test_transfer),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Test aborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Fatal Error] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
