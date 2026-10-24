"""CPU tests for the qualification protocol; these do not qualify any GPU path."""

import contextlib
import ctypes
import io
import json
import socket
import struct
import threading
import types
import unittest

import p800_device_transfer as probe


class HostRuntime:
    def __init__(self):
        self.buffers = {}
        self.copies = []

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
        ctypes.memmove(dst, src, size)

    def free(self, ptr):
        del self.buffers[ptr.value]


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
        return True

    def dereg(self, mr):
        if self.fail_cleanup:
            return False
        self.registered.remove(mr)
        return True


class ProtocolTests(unittest.TestCase):
    def run_pair(
        self, path, corrupt=False, mismatch=False, fail_cleanup=False, fault=None
    ):
        errors = []
        runtimes = [HostRuntime(), HostRuntime()]
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

    def test_both_paths_complete_and_release(self):
        for path in ["direct", "host-staged"]:
            with self.subTest(path=path):
                errors, results, runtimes, endpoints = self.run_pair(path)
                self.assertEqual(errors, [])
                self.assertEqual([r["bytes"] for r in results], [257, 1024, 65536])
                for result in results:
                    self.assertEqual(result["verified_transfers"], 4)
                    self.assertEqual(result["measured_transfers"], 3)
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

    def test_tail_corruption_fails_both_peers(self):
        for path in ["direct", "host-staged"]:
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

    def test_payload_changes_and_full_tail_check(self):
        expected = probe.payload(1025, 4)
        self.assertNotEqual(expected, probe.payload(1025, 5))
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            probe.verify(expected[:-1] + bytes([expected[-1] ^ 1]), expected)

    def test_block_reordering_repetition_stale_and_dropped_writes_fail(self):
        for path in ["direct", "host-staged"]:
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
