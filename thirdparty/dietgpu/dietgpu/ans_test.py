# Copyright (c) (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import random
import unittest

import torch

torch.ops.load_library("//dietgpu:dietgpu")


def run_test(dev, ts, checksum, temp_mem=None):
    comp, sizes, _ = torch.ops.dietgpu.compress_data(False, ts, checksum, temp_mem)
    for s, t in zip(sizes, ts):
        t_bytes = t.numel() * t.element_size()
        print(f"{t_bytes} bytes -> {s.item()} bytes ({s.item() / t_bytes}x)")

    # Truncate the output data to exactly the sizes that are used
    # (i.e., we are testing that the byte sizes we report in compression are accurate)
    truncated_comp = []
    for size, t in zip(sizes, [*comp]):
        truncated_t = t.narrow(0, 0, size.item()).clone()
        truncated_comp.append(truncated_t)

    out_ts = []
    for t in ts:
        out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

    if temp_mem is not None:
        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        torch.ops.dietgpu.decompress_data(
            False, truncated_comp, out_ts, checksum, temp_mem, out_status, out_sizes
        )

        for t, status, size in zip(ts, out_status, out_sizes):
            assert status.item()
            assert t.numel() * t.element_size() == size.item()
    else:
        torch.ops.dietgpu.decompress_data(False, truncated_comp, out_ts, checksum)

    for a, b in zip(ts, out_ts):
        assert torch.equal(a, b)


class TestANSCodec(unittest.TestCase):
    def test_codec(self):
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

        for dt in [torch.float32]:
            for tm in [False, True]:
                for checksum in [False, True]:
                    ts = [
                        torch.normal(0, 1.0, [10000], dtype=dt, device=dev),
                        torch.normal(0, 1.0, [100000], dtype=dt, device=dev),
                        torch.normal(0, 1.0, [1000000], dtype=dt, device=dev),
                    ]
                    if tm:
                        run_test(dev, ts, checksum, temp_mem)
                    else:
                        run_test(dev, ts, checksum)

    def test_empty(self):
        dev = torch.device("cuda:0")
        ts = [torch.empty([0], dtype=torch.uint8, device=dev)]
        comp_ts = torch.ops.dietgpu.compress_data_simple(False, ts, True)

        # should have a header
        assert comp_ts[0].numel() > 0

        decomp_ts = torch.ops.dietgpu.decompress_data_simple(False, comp_ts, True)
        assert torch.equal(ts[0], decomp_ts[0])

    def test_split_compress(self):
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

        for tries in range(5):
            batch_size = random.randrange(1, 15)
            sizes = []

            sum_sizes = 0
            max_size = 0
            for i in range(batch_size):
                size = random.randrange(1, 10000)
                # meet required alignment
                size += 4 - (size % 4)

                sizes.append(size)
                sum_sizes += size
                if size > max_size:
                    max_size = size

            t = torch.randint(0, 65, [sum_sizes], dtype=torch.uint8, device=dev)
            sizes_t = torch.IntTensor(sizes)
            splits = torch.split(t, sizes)

            comp_ts, _, _ = torch.ops.dietgpu.compress_data_split_size(
                False, t, sizes_t, True, temp_mem
            )
            decomp_ts = torch.ops.dietgpu.decompress_data_simple(False, comp_ts, True)

            for orig, decomp in zip(splits, decomp_ts):
                assert torch.equal(orig, decomp)

    def test_split_decompress(self):
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

        for tries in range(5):
            batch_size = random.randrange(1, 15)
            sizes = []

            sum_sizes = 0
            for i in range(batch_size):
                size = random.randrange(1, 10000)
                # meet required alignment
                size += 4 - (size % 4)

                sizes.append(size)
                sum_sizes += size

            t = torch.randint(0, 65, [sum_sizes], dtype=torch.uint8, device=dev)
            sizes_t = torch.IntTensor(sizes)

            splits = torch.split(t, sizes)
            comp_ts = torch.ops.dietgpu.compress_data_simple(False, splits, True)

            decomp_t = torch.empty([sum_sizes], dtype=torch.uint8, device=dev)
            torch.ops.dietgpu.decompress_data_split_size(
                False, comp_ts, decomp_t, sizes_t, True, temp_mem
            )

            assert torch.equal(t, decomp_t)
