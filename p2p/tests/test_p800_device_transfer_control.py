"""CPU tests for the qualification protocol; these do not qualify any GPU path."""

import contextlib
import ctypes
import io
import json
import socket
import struct
import sys
import threading
import types
import unittest
from unittest import mock

import p800_device_transfer as probe


class HostRuntime:
    def __init__(self, drop_copy=None):
        self.buffers = {}
        self.copies = []
        self.drop_copy = drop_copy

    def alloc(self, size):
        buf = ctypes.create_string_buffer(size)
        ptr = ctypes.c_void_p(ctypes.addressof(buf))
        self.buffers[ptr.value] = buf
        return ptr

    def fill(self, ptr, data):
        ctypes.memmove(ptr, data, len(data))

    def read(self, ptr, size):
        return ctypes.string_at(ptr, size)

    def copy(self, dst, src, size, direction):
        self.copies.append(direction)
        if direction != self.drop_copy:
            ctypes.memmove(dst, src, size)

    def free(self, ptr):
        del self.buffers[ptr.value]


class AliasRuntime(HostRuntime):
    """Separate device identifiers from host storage used by the fake endpoint."""

    def alloc(self, size):
        pointer = ctypes.c_void_p(0x4000000000 + len(self.buffers) * 0x100000)
        self.buffers[pointer.value] = ctypes.create_string_buffer(size)
        return pointer

    def mapped_alias(self, ptr, gpu):
        return ctypes.addressof(self.buffers[ptr.value])

    def fill(self, ptr, data):
        ctypes.memmove(self.mapped_alias(ptr, 0), data, len(data))

    def read(self, ptr, size):
        return ctypes.string_at(self.mapped_alias(ptr, 0), size)


class MappedAliasTests(unittest.TestCase):
    def query(self, memory_type=2, gpu=0, alias=0x2000, error=0):
        def query(output, pointer):
            attrs = ctypes.cast(output, ctypes.POINTER(probe.PointerAttributes))[0]
            attrs.type, attrs.device = memory_type, gpu
            attrs.devicePointer, attrs.hostPointer = pointer.value, alias
            return error

        runtime = object.__new__(probe.Runtime)
        runtime.lib = types.SimpleNamespace(cudaPointerGetAttributes=query)
        return runtime.mapped_alias(ctypes.c_void_p(0x1000), 0)

    def test_device_alias_is_explicit(self):
        self.assertEqual(self.query(), 0x2000)

    def test_missing_or_identical_alias_rejected(self):
        for alias in (None, 0x1000):
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(
                    RuntimeError, "missing distinct vendor alias"
                ):
                    self.query(alias=alias)

    def test_wrong_memory_type_or_device_rejected(self):
        for memory_type, gpu in ((1, 0), (2, 1)):
            with self.subTest(memory_type=memory_type, gpu=gpu):
                with self.assertRaisesRegex(RuntimeError, "not the assigned device"):
                    self.query(memory_type=memory_type, gpu=gpu)

    def test_query_error_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "cudaPointerGetAttributes.*13"):
            self.query(error=13)


class DeferredCopyLibrary:
    """Model pageable H2D returning after staging, before device completion."""

    def __init__(self, sync_error=0):
        self.host = HostRuntime()
        self.pending = []
        self.sync_error = sync_error
        self.completed_copies = 0
        # ctypes function objects accept signature attributes; bound methods do not.
        for name in ("SetDevice", "Malloc", "Free", "Memcpy", "DeviceSynchronize"):
            setattr(
                self,
                "cuda" + name,
                lambda *args, method=name: getattr(self, method)(*args),
            )

    def SetDevice(self, device):
        return 0

    def Malloc(self, output, size):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = self.host.alloc(size)
        return 0

    def Free(self, ptr):
        self.host.free(ptr)
        return 0

    def Memcpy(self, dst, src, size, direction):
        if direction == 1:
            # Snapshot the pageable source just as a runtime staging copy would.
            self.pending.append((dst, ctypes.string_at(src, size)))
        else:
            # A later blocking D2H can hide a missing H2D completion wait.
            self.complete()
            ctypes.memmove(dst, src, size)
        return 0

    def complete(self):
        for dst, data in self.pending:
            ctypes.memmove(dst, data, len(data))
            self.completed_copies += 1
        self.pending.clear()

    def DeviceSynchronize(self):
        if not self.sync_error:
            self.complete()
        return self.sync_error


def deferred_runtime(library=None):
    library = library or DeferredCopyLibrary()
    with mock.patch.object(probe.ctypes, "CDLL", return_value=library):
        runtime = probe.Runtime("cpu-test-library", 0)
    return runtime


class RuntimeCompletionTests(unittest.TestCase):
    def test_pageable_h2d_completes_before_copy_returns(self):
        library = DeferredCopyLibrary()
        runtime = deferred_runtime(library)
        source = ctypes.create_string_buffer(b"copy-done", 9)
        destination = ctypes.create_string_buffer(9)
        runtime.copy(destination, source, 9, 1)
        self.assertEqual(destination.raw, b"copy-done")
        self.assertEqual(library.pending, [])
        self.assertEqual(library.completed_copies, 1)

    def test_fill_finishes_before_external_device_write(self):
        runtime = deferred_runtime()
        device = runtime.alloc(8)
        try:
            runtime.fill(device, b"poisoned")
            # Simulate RDMA writing only after initialization has returned.
            ctypes.memmove(device, b"received", 8)
            self.assertEqual(runtime.read(device, 8), b"received")
        finally:
            runtime.free(device)

    def test_completion_error_is_not_a_successful_copy(self):
        runtime = deferred_runtime(DeferredCopyLibrary(sync_error=13))
        source = ctypes.create_string_buffer(b"data", 4)
        destination = ctypes.create_string_buffer(4)
        with self.assertRaisesRegex(RuntimeError, "cudaDeviceSynchronize.*13"):
            runtime.copy(destination, source, 4, 1)


class HostEndpoint:
    def __init__(self, corrupt=False, fail_cleanup=False, fault=None):
        self.registered = set()
        self.corrupt = corrupt
        self.fail_cleanup = fail_cleanup
        self.fault = fault
        self.previous = None

    def get_metadata(self):
        return b"host-test"

    @staticmethod
    def parse_metadata(meta):
        return "unused", 0, "0"

    def connect(self, *args, **kwargs):
        return True, 0

    def accept(self):
        return True, "unused", 0, 0

    def reg(self, address, size):
        self.registered.add(address)
        return True, address

    def advertise(self, mr, address, size):
        return True, struct.pack("!Q", address) + bytes(56)

    def write(self, conn, mr, address, size, descriptor):
        destination = struct.unpack("!Q", descriptor[:8])[0]
        if self.fault == "ignore-source-offset":
            address = mr
        elif self.fault == "ignore-destination-offset":
            destination -= probe.GUARD_BYTES
        data = ctypes.string_at(address, size)
        if self.fault == "swap-blocks":
            data = data[32:64] + data[:32] + data[64:]
        elif self.fault == "repeat-block":
            data = (data[:32] * ((size + 31) // 32))[:size]
        elif self.fault == "rotate-blocks":
            data = data[32:] + data[:32]
        elif self.fault == "stale" and self.previous is not None:
            data = self.previous
        self.previous = ctypes.string_at(address, size)
        if self.fault != "drop":
            ctypes.memmove(destination, data, size)
        if self.corrupt:
            tail = ctypes.c_ubyte.from_address(destination + size - 1)
            tail.value ^= 1
        if self.fault in ("overwrite-prefix", "overwrite-suffix"):
            offset = -1 if self.fault == "overwrite-prefix" else size
            guard = ctypes.c_ubyte.from_address(destination + offset)
            guard.value ^= 1
        if self.fault in ("source-prefix", "source-suffix"):
            offset = -1 if self.fault == "source-prefix" else size
            guard = ctypes.c_ubyte.from_address(address + offset)
            guard.value ^= 1
        return True

    def dereg(self, mr):
        if self.fail_cleanup:
            return False
        self.registered.remove(mr)
        return True


class ConfigurationTests(unittest.TestCase):
    def run_main(self, path, strategy):
        argv = [
            "p800_device_transfer.py",
            "--role",
            "sender",
            "--peer",
            "127.0.0.1",
            "--runtime-lib",
            "unused-cpu-test-library",
            "--path",
            path,
        ]
        endpoint = mock.Mock()
        module = types.SimpleNamespace(p2p=types.SimpleNamespace(Endpoint=endpoint))
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(sys.modules, {"uccl": module}),
            mock.patch.dict(
                probe.os.environ,
                {
                    "UCCL_P2P_TRANSPORT": "rdma",
                    "UCCL_P2P_COMPRESS_STRATEGY": strategy,
                },
            ),
            mock.patch.object(probe, "Runtime") as runtime,
            mock.patch.object(probe.socket, "create_connection") as connection,
            mock.patch.object(probe, "run_pair") as run,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            try:
                probe.main()
            except SystemExit as exc:
                return (
                    exc.code,
                    stderr.getvalue(),
                    runtime.call_count,
                    endpoint.call_count,
                    run.call_count,
                )
        return (
            0,
            stderr.getvalue(),
            runtime.call_count,
            endpoint.call_count,
            run.call_count,
        )

    def test_compression_rejected_before_runtime_or_endpoint(self):
        # In DietGPU builds, unknown values also select split+encode.
        for path in ["direct", "host-staged", "mapped-alias"]:
            for strategy in [
                "split",
                "split_only",
                "encode",
                "split_encode",
                "full",
                "1",
                "SPLIT",
                "typo",
            ]:
                with self.subTest(path=path, strategy=strategy):
                    code, error, runtime, endpoint, run = self.run_main(path, strategy)
                    self.assertEqual(code, 2)
                    self.assertIn("uncompressed RDMA", error)
                    self.assertEqual((runtime, endpoint, run), (0, 0, 0))

    def test_disabled_compression_spellings_reach_protocol(self):
        for strategy in ["", "none", "off", "0", "NONE", "Off"]:
            with self.subTest(strategy=strategy):
                code, _, runtime, endpoint, run = self.run_main("direct", strategy)
                self.assertEqual(code, 0)
                self.assertEqual((runtime, endpoint, run), (1, 1, 1))


class ProtocolTests(unittest.TestCase):
    def run_pair(
        self,
        path,
        corrupt=False,
        mismatch=False,
        fail_cleanup=False,
        fault=None,
        drop_copy=None,
        runtime_factory=None,
    ):
        errors = []
        if runtime_factory is None and path == "mapped-alias":
            runtime_factory = AliasRuntime
        runtimes = [
            runtime_factory() if runtime_factory else HostRuntime(drop_copy)
            for _ in range(2)
        ]
        endpoints = [
            HostEndpoint(corrupt, fault=fault),
            HostEndpoint(fail_cleanup=fail_cleanup),
        ]
        pair = socket.socketpair()
        for sock in pair:
            sock.settimeout(2)

        def worker(index):
            args = types.SimpleNamespace(
                role="sender" if index == 0 else "receiver",
                path=path,
                gpu=0,
                sizes=[257, 1024, 65536],
                warmup=1,
                iterations=3 + int(mismatch and index == 1),
            )
            try:
                with pair[index]:
                    probe.run_pair(
                        args,
                        pair[index],
                        runtimes[index],
                        endpoints[index],
                        types.SimpleNamespace(Endpoint=HostEndpoint),
                    )
            except Exception as exc:
                errors.append(exc)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            threads = [
                threading.Thread(target=worker, args=(i,), daemon=True)
                for i in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            self.assertFalse(any(t.is_alive() for t in threads))
        return (
            errors,
            [json.loads(line) for line in out.getvalue().splitlines()],
            runtimes,
            endpoints,
        )

    def test_all_paths_complete_and_release(self):
        for path in ["direct", "host-staged", "mapped-alias"]:
            with self.subTest(path=path):
                errors, results, runtimes, endpoints = self.run_pair(path)
                self.assertEqual(errors, [])
                self.assertEqual([r["bytes"] for r in results], [257, 1024, 65536])
                for result in results:
                    self.assertEqual(result["protocol_version"], probe.PROTOCOL_VERSION)
                    self.assertEqual(result["transport"], "rdma")
                    self.assertEqual(result["compression"], "none")
                    self.assertEqual(result["verified_transfers"], 4)
                    self.assertEqual(result["measured_transfers"], 3)
                    self.assertEqual(result["guard_bytes_per_side"], probe.GUARD_BYTES)
                    self.assertEqual(
                        result["allocation_bytes"],
                        result["bytes"] + 2 * probe.GUARD_BYTES,
                    )
                    self.assertGreater(result["transport_GBps"], 0)
                    self.assertGreaterEqual(
                        result["device_path_seconds"], result["transport_seconds"]
                    )
                self.assertTrue(all(not rt.buffers for rt in runtimes))
                self.assertTrue(all(not ep.registered for ep in endpoints))
                self.assertEqual(
                    runtimes[0].copies, [2] * 12 if path == "host-staged" else []
                )
                self.assertEqual(
                    runtimes[1].copies, [1] * 12 if path == "host-staged" else []
                )

    def test_both_paths_with_deferred_pageable_copies(self):
        for path in ["direct", "host-staged"]:
            with self.subTest(path=path):
                errors, results, runtimes, endpoints = self.run_pair(
                    path, runtime_factory=deferred_runtime
                )
                self.assertEqual(errors, [])
                self.assertEqual([r["bytes"] for r in results], [257, 1024, 65536])
                self.assertTrue(all(not rt.lib.host.buffers for rt in runtimes))
                self.assertTrue(all(not rt.lib.pending for rt in runtimes))
                self.assertTrue(all(not ep.registered for ep in endpoints))

    def test_tail_corruption_fails_both_peers(self):
        for path in ["direct", "host-staged", "mapped-alias"]:
            errors, results, _, _ = self.run_pair(path, corrupt=True)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("mismatch" in str(e) for e in errors))
            self.assertEqual(results, [])

    def test_different_plans_rejected_before_registration(self):
        errors, results, runtimes, endpoints = self.run_pair("direct", mismatch=True)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("plans differ" in str(e) for e in errors))
        self.assertEqual(results, [])
        self.assertTrue(all(not rt.buffers for rt in runtimes))
        self.assertTrue(all(not ep.registered for ep in endpoints))

    def test_old_protocol_rejected_before_allocation(self):
        args = types.SimpleNamespace(
            role="sender", path="direct", sizes=[1024], warmup=1, iterations=3
        )
        runtime, endpoint = HostRuntime(), HostEndpoint()
        a, b = socket.socketpair()
        with a, b:
            a.settimeout(2)
            # Version 1 peers sent an unversioned plan and repeated 32-byte data.
            probe.send(b, b'["direct",[1024],1,3]')
            with self.assertRaisesRegex(RuntimeError, "plans differ"):
                probe.run_pair(args, a, runtime, endpoint, None)
        self.assertEqual(runtime.buffers, {})
        self.assertEqual(endpoint.registered, set())

    def test_cleanup_failure_prevents_success_report(self):
        errors, results, _, _ = self.run_pair("host-staged", fail_cleanup=True)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("deregistration failed" in str(e) for e in errors))
        self.assertEqual(results, [])

    def test_guard_overwrites_and_ignored_offsets_fail(self):
        for path in ["direct", "host-staged", "mapped-alias"]:
            for fault in [
                "overwrite-prefix",
                "overwrite-suffix",
                "source-prefix",
                "source-suffix",
                "ignore-source-offset",
                "ignore-destination-offset",
            ]:
                with self.subTest(path=path, fault=fault):
                    errors, results, _, _ = self.run_pair(path, fault=fault)
                    self.assertEqual(len(errors), 2)
                    self.assertTrue(any("mismatch" in str(e) for e in errors))
                    self.assertEqual(results, [])

    def test_staging_copies_cannot_be_skipped(self):
        for direction in [1, 2]:
            with self.subTest(direction=direction):
                errors, results, _, _ = self.run_pair(
                    "host-staged", drop_copy=direction
                )
                self.assertEqual(len(errors), 2)
                self.assertTrue(any("mismatch" in str(e) for e in errors))
                self.assertEqual(results, [])

    def test_earlier_qualification_protocols_rejected_before_allocation(self):
        args = types.SimpleNamespace(
            role="sender", path="direct", sizes=[1024], warmup=1, iterations=3
        )
        for plan in (
            b'[2,"direct",[1024],1,3]',
            b'[3,"direct",[1024],1,3,256]',
            b'[4,"direct",[1024],1,3,256]',
            b'[5,"direct",[1024],1,3,256]',
        ):
            runtime, endpoint = HostRuntime(), HostEndpoint()
            a, b = socket.socketpair()
            with a, b:
                a.settimeout(2)
                probe.send(b, plan)
                with self.assertRaisesRegex(RuntimeError, "plans differ"):
                    probe.run_pair(args, a, runtime, endpoint, None)
            self.assertEqual(runtime.buffers, {})
            self.assertEqual(endpoint.registered, set())

    def test_payload_changes_and_full_tail_check(self):
        expected = probe.payload(1025, 4)
        self.assertNotEqual(expected, probe.payload(1025, 5))
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            probe.verify(expected[:-1] + bytes([expected[-1] ^ 1]), expected)

    def test_block_reordering_repetition_stale_and_dropped_writes_fail(self):
        for path in ["direct", "host-staged", "mapped-alias"]:
            for fault in [
                "swap-blocks",
                "repeat-block",
                "rotate-blocks",
                "stale",
                "drop",
            ]:
                with self.subTest(path=path, fault=fault):
                    errors, results, _, _ = self.run_pair(path, fault=fault)
                    self.assertEqual(len(errors), 2)
                    self.assertTrue(any("mismatch" in str(e) for e in errors))
                    self.assertEqual(results, [])

    def test_payload_depends_on_size_sequence_and_block_position(self):
        # Block-sized addressing errors must not preserve the expected bytes.
        for size in [1024, 1048576]:
            original = probe.payload(size, 4)
            self.assertEqual(original, probe.payload(size, 4))
            self.assertNotEqual(original, probe.payload(size + 1, 4)[:size])
            self.assertNotEqual(original, probe.payload(size, 5))
            for width in [32, 256, 512]:
                with self.subTest(size=size, block_bytes=width):
                    swapped = (
                        original[width : 2 * width]
                        + original[:width]
                        + original[2 * width :]
                    )
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        probe.verify(swapped, original)

    def test_oversize_control_frame_rejected(self):
        a, b = socket.socketpair()
        with a, b:
            a.sendall(struct.pack("!I", 65537))
            with self.assertRaisesRegex(RuntimeError, "too large"):
                probe.receive(b)


if __name__ == "__main__":
    unittest.main()
