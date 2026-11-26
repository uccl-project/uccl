"""CPU-only checks of the benchmark's real native-descriptor preflight.

No UCCL extension, GPU runtime, sockets, or synthetic performance measurements
are used. These cases keep the previously executed descriptor checks repeatable.
"""

import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_proxy_fairness.py"
SPEC = importlib.util.spec_from_file_location("proxy_fairness_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def stream(address, length=64, capacity=64):
    # Native FifoItem: uint64 addr at byte 0, uint32 size at byte 8; the rest is
    # irrelevant to range validation but the complete wire record is 64 bytes.
    descriptor = struct.pack("=QI", address, capacity) + bytes(52)
    return {"slots": [SimpleNamespace(remote=[descriptor], sizes=[length])]}


class ProxyFairnessDescriptorTests(unittest.TestCase):
    def test_native_addresses_decode_and_adjacent_ranges_are_independent(self):
        result = BENCHMARK.check_remote_write_ranges([stream(4096), stream(4160)])
        self.assertEqual(result["large"], [["0x1000", "0x1040"]])
        self.assertEqual(result["small"], [["0x1040", "0x1080"]])

    def test_overlapping_peer_addresses_reject_a_serialized_benchmark(self):
        with self.assertRaisesRegex(RuntimeError, "serialized"):
            BENCHMARK.check_remote_write_ranges([stream(4096), stream(4128)])

    def test_requested_write_cannot_exceed_advertised_capacity(self):
        with self.assertRaisesRegex(RuntimeError, "advertised FifoItem size"):
            BENCHMARK.check_remote_write_ranges([stream(4096, 65), stream(8192)])

    def test_end_address_must_fit_the_adapter_uint64_range(self):
        for start in (2**64 - 32, 2**64 - 64):
            with self.subTest(start=start):
                with self.assertRaisesRegex(RuntimeError, "invalid remote address"):
                    BENCHMARK.check_remote_write_ranges([stream(start), stream(8192)])


if __name__ == "__main__":
    unittest.main()
