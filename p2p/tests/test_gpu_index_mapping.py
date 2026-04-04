#!/usr/bin/env python3
"""
Tests for GPU identity via PCI Bus ID (BDF).

Verifies that Endpoint uses PCI Bus ID strings for cross-process GPU identity,
and that cudaIpcOpenMemHandle works across different CUDA_VISIBLE_DEVICES.

Run:  python test_gpu_index_mapping.py          (needs >= 2 GPUs)
"""

import os
import re
import subprocess
import sys
import time

import torch

try:
    from uccl import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    sys.exit(1)


NUM_GPUS = torch.cuda.device_count()

BDF_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip(reason: str):
    print(f"SKIP: {reason}")
    return True


def _parse_metadata_bdf(metadata: bytes) -> str:
    """Extract the GPU BDF string from Endpoint metadata."""
    _, _, bdf = p2p.Endpoint.parse_metadata(metadata)
    return bdf


def _run_in_subprocess(
    func_name: str, env_override: dict = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run a test helper function in an isolated subprocess with custom env."""
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    cmd = [sys.executable, __file__, func_name]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


# ---------------------------------------------------------------------------
# Test 1: Metadata contains a valid PCI Bus ID (BDF) string
# ---------------------------------------------------------------------------


def test_metadata_contains_bdf():
    """Endpoint.get_metadata() should embed a valid PCI Bus ID."""
    if NUM_GPUS < 1:
        return _skip("No GPU available")

    print("Running test_metadata_contains_bdf...")

    ep = p2p.Endpoint(local_gpu_idx=0)
    metadata = ep.get_metadata()
    bdf = _parse_metadata_bdf(metadata)

    assert BDF_PATTERN.match(bdf), f"Expected BDF format, got '{bdf}'"
    print(f"  metadata bdf = {bdf} ✓")

    if NUM_GPUS >= 2:
        ep2 = p2p.Endpoint(local_gpu_idx=1)
        bdf2 = _parse_metadata_bdf(ep2.get_metadata())
        assert BDF_PATTERN.match(bdf2), f"Expected BDF format, got '{bdf2}'"
        assert bdf != bdf2, f"GPU 0 and 1 should have different BDFs: {bdf} vs {bdf2}"
        print(f"  metadata bdf (GPU 1) = {bdf2} ✓")

    print("✓ test_metadata_contains_bdf passed")
    return True


# ---------------------------------------------------------------------------
# Test 2: BDF is consistent under CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------


def _subprocess_bdf_check():
    """Subprocess: Endpoint(0) should report BDF matching the visible GPU."""
    from uccl import p2p as _p2p

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected_bdf = os.environ.get("TEST_EXPECTED_BDF", "")

    ep = _p2p.Endpoint(0)
    metadata = ep.get_metadata()
    _, _, bdf = _p2p.Endpoint.parse_metadata(metadata)

    if expected_bdf:
        assert (
            bdf == expected_bdf
        ), f"CUDA_VISIBLE_DEVICES={cvd}: expected BDF={expected_bdf}, got {bdf}"
    print(f"OK: CUDA_VISIBLE_DEVICES={cvd}, Endpoint(0) → BDF={bdf}")


def test_bdf_under_cvd():
    """Endpoint(0) should report the correct BDF when CUDA_VISIBLE_DEVICES is set."""
    if NUM_GPUS < 2:
        return _skip("Need >= 2 GPUs")

    print("Running test_bdf_under_cvd...")

    # Get BDFs of all visible GPUs
    bdfs = []
    for i in range(min(NUM_GPUS, 4)):
        ep = p2p.Endpoint(local_gpu_idx=i)
        bdf = _parse_metadata_bdf(ep.get_metadata())
        bdfs.append(bdf)

    # Run subprocess with only GPU 1 visible, check it reports GPU 1's BDF
    r = _run_in_subprocess(
        "_subprocess_bdf_check",
        {"CUDA_VISIBLE_DEVICES": "1", "TEST_EXPECTED_BDF": bdfs[1]},
    )
    assert r.returncode == 0, f"Failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    print(f"  CUDA_VISIBLE_DEVICES=1, Endpoint(0) → BDF={bdfs[1]} ✓")

    if NUM_GPUS >= 4:
        r = _run_in_subprocess(
            "_subprocess_bdf_check",
            {"CUDA_VISIBLE_DEVICES": "3", "TEST_EXPECTED_BDF": bdfs[3]},
        )
        assert r.returncode == 0, f"Failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
        print(f"  CUDA_VISIBLE_DEVICES=3, Endpoint(0) → BDF={bdfs[3]} ✓")

    print("✓ test_bdf_under_cvd passed")
    return True


# ---------------------------------------------------------------------------
# Test 3: BDF uniqueness — different GPUs produce different BDFs
# ---------------------------------------------------------------------------


def test_bdf_uniqueness():
    """Endpoints on different GPUs should have unique BDF strings."""
    if NUM_GPUS < 2:
        return _skip("Need >= 2 GPUs")

    print("Running test_bdf_uniqueness...")

    bdfs = set()
    for i in range(min(NUM_GPUS, 4)):
        ep = p2p.Endpoint(local_gpu_idx=i)
        bdf = _parse_metadata_bdf(ep.get_metadata())
        assert BDF_PATTERN.match(bdf), f"GPU {i}: invalid BDF '{bdf}'"
        bdfs.add(bdf)

    expected = min(NUM_GPUS, 4)
    assert (
        len(bdfs) == expected
    ), f"Expected {expected} unique BDFs, got {len(bdfs)}: {bdfs}"
    print(f"  {len(bdfs)} endpoints with unique BDFs: {sorted(bdfs)} ✓")
    print("✓ test_bdf_uniqueness passed")
    return True


# ---------------------------------------------------------------------------
# Test 4: parse_metadata round-trip
# ---------------------------------------------------------------------------


def test_parse_metadata_roundtrip():
    """parse_metadata should return the BDF string embedded by get_metadata."""
    if NUM_GPUS < 1:
        return _skip("No GPU available")

    print("Running test_parse_metadata_roundtrip...")

    ep = p2p.Endpoint(local_gpu_idx=0)
    metadata = ep.get_metadata()
    ip, port, bdf = p2p.Endpoint.parse_metadata(metadata)

    assert isinstance(ip, str) and len(ip) > 0, f"Bad IP: {ip}"
    assert isinstance(port, int) and port > 0, f"Bad port: {port}"
    assert isinstance(bdf, str) and BDF_PATTERN.match(bdf), f"Bad BDF: {bdf}"
    print(f"  parse_metadata → ip={ip}, port={port}, bdf={bdf} ✓")
    print("✓ test_parse_metadata_roundtrip passed")
    return True


# ---------------------------------------------------------------------------
# Test 5: Two restricted-visibility processes communicate via IPC
# ---------------------------------------------------------------------------


def _subprocess_ipc_server():
    """Subprocess entry: server that accepts and receives data via IPC."""
    from uccl import p2p as _p2p
    import torch as _torch

    meta_file = os.environ["TEST_META_FILE"]

    ep = _p2p.Endpoint(0)
    metadata = ep.get_metadata()

    with open(meta_file, "wb") as f:
        f.write(bytes(metadata))

    success, remote_ip, remote_gpu_idx, conn_id = ep.accept()
    assert success, "Server accept failed"

    tensor = _torch.zeros(1024, dtype=_torch.float32, device="cuda:0")
    success, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert success, "Server reg failed"

    success = ep.recv(conn_id, mr_id, tensor.data_ptr(), size=tensor.numel() * 4)
    assert success, "Server recv failed"

    expected = _torch.ones(1024, dtype=_torch.float32, device="cuda:0")
    assert tensor.allclose(
        expected
    ), f"Data mismatch! got sum={tensor.sum().item()}, expected 1024.0"
    print("OK: Server received correct data")


def _subprocess_ipc_client():
    """Subprocess entry: client that connects and sends data via IPC."""
    from uccl import p2p as _p2p
    import torch as _torch

    meta_file = os.environ["TEST_META_FILE"]

    for _ in range(50):
        if os.path.exists(meta_file) and os.path.getsize(meta_file) > 0:
            break
        time.sleep(0.1)

    with open(meta_file, "rb") as f:
        metadata = f.read()
    ip, port, remote_bdf = _p2p.Endpoint.parse_metadata(metadata)

    ep = _p2p.Endpoint(0)

    success, conn_id = ep.connect(remote_ip_addr=ip, remote_gpu_bdf=remote_bdf, remote_port=port)
    assert success, f"Client connect failed to {ip}:{port}"

    tensor = _torch.ones(1024, dtype=_torch.float32, device="cuda:0")
    success, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * 4)
    assert success, "Client reg failed"

    success = ep.send(conn_id, mr_id, tensor.data_ptr(), tensor.numel() * 4)
    assert success, "Client send failed"
    print("OK: Client sent data")


def test_cross_process_disjoint_cvd():
    """Two processes with disjoint CUDA_VISIBLE_DEVICES should communicate."""
    if NUM_GPUS < 2:
        return _skip("Need >= 2 GPUs")

    print("Running test_cross_process_disjoint_cvd...")

    import tempfile

    meta_file = tempfile.mktemp(prefix="uccl_test_meta_")

    try:
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = "0"
        server_env["TEST_META_FILE"] = meta_file

        server = subprocess.Popen(
            [sys.executable, __file__, "_subprocess_ipc_server"],
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(5)

        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = "1"
        client_env["TEST_META_FILE"] = meta_file

        client = subprocess.Popen(
            [sys.executable, __file__, "_subprocess_ipc_client"],
            env=client_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        client_out, client_err = client.communicate(timeout=60)
        server_out, server_err = server.communicate(timeout=60)

        if server.returncode != 0:
            raise AssertionError(
                f"Server failed (rc={server.returncode}):\n"
                f"stdout: {server_out.decode()}\nstderr: {server_err.decode()}"
            )
        if client.returncode != 0:
            raise AssertionError(
                f"Client failed (rc={client.returncode}):\n"
                f"stdout: {client_out.decode()}\nstderr: {client_err.decode()}"
            )
    except subprocess.TimeoutExpired:
        server.kill()
        client.kill()
        server.communicate()
        client.communicate()
        raise AssertionError("Timed out waiting for subprocesses")
    finally:
        if os.path.exists(meta_file):
            os.unlink(meta_file)

    print("  Server & client exchanged data with disjoint CUDA_VISIBLE_DEVICES ✓")
    print("✓ test_cross_process_disjoint_cvd passed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"=== GPU BDF Identity Tests (NUM_GPUS={NUM_GPUS}) ===\n")

    tests = [
        test_parse_metadata_roundtrip,
        test_metadata_contains_bdf,
        test_bdf_uniqueness,
        test_bdf_under_cvd,
        test_cross_process_disjoint_cvd,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if result is True:
                passed += 1
            print()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}\n")
            failed += 1

    total = passed + failed
    print(
        f"=== Results: {passed}/{total} passed"
        f"{f', {failed} failed' if failed else ''} ==="
    )
    sys.exit(1 if failed else 0)


# Subprocess dispatch
if __name__ == "__main__":
    if len(sys.argv) > 1:
        func = globals().get(sys.argv[1])
        if func and callable(func):
            func(*sys.argv[2:])
        else:
            print(f"Unknown function: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    else:
        main()
