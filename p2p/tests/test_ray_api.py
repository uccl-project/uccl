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
    ep = p2p.Endpoint(4)
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
    ep = p2p.Endpoint(4)
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
    ep = p2p.Endpoint(4)
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
    ep = p2p.Endpoint(4)
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
    ep = p2p.Endpoint(4)
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
    ep = p2p.Endpoint(4)
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


def main():
    """Run all tests."""
    print("UCCL P2P Engine - register_memory API Test")
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
        ("Basic register_memory", test_register_memory_basic),
        ("Multiple tensors", test_register_memory_multiple),
        ("Empty list", test_register_memory_empty_list),
        ("Invalid input", test_register_memory_invalid_input),
        ("Mixed valid/invalid", test_register_memory_mixed_valid_invalid),
        ("Different dtypes", test_register_memory_different_dtypes),
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
