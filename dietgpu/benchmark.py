# Copyright (c) (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Simple benchmarking script for both float and raw byte-wise ANS codecs in
# PyTorch using the asynchronous API, as applied to floating point data
# ~ N(0, 1)

import torch


torch.ops.load_library("/home/yangzhou/shuangma/uccl/uccl/dietgpu.cpython-314-x86_64-linux-gnu.so")
dev = torch.device("cuda:0")


def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0

    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s

    return total_input_size, total_comp_size, total_comp_size / total_input_size


def get_float_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_float_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            True, ts, False, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        if i > 0:
            comp_time += start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            True, comp_ts, out_ts, False, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        if i > 0:
            decomp_time += start.elapsed_time(end)

        # validate
        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    comp_time /= num_runs
    decomp_time /= num_runs

    return comp_time, decomp_time, total_size, comp_size


def get_any_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            False, ts, False, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        comp_time = start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            False, comp_ts, out_ts, False, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        decomp_time = start.elapsed_time(end)

        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    return comp_time, decomp_time, total_size, comp_size


# for dt in [torch.bfloat16, torch.float16, torch.float32]:
#     # Non-batched
#     ts = []
#     ts.append(torch.normal(0, 1.0, [128 * 512 * 1024], dtype=dt, device=dev))

#     c, dc, total_size, comp_size = get_float_comp_timings(ts)
#     ratio = comp_size / total_size
#     c_bw = (total_size / 1e9) / (c * 1e-3)
#     dc_bw = (total_size / 1e9) / (dc * 1e-3)

#     print(f"Float codec non-batched perf [128 * 512 * 1024] {dt}")
#     print(
#         "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
#             c, c_bw, total_size, comp_size, ratio
#         )
#     )
#     print(f"decomp time {dc:.3f} ms B/W {dc_bw:.1f} GB/s")

#     # Batched
#     ts = []
#     for i in range(128):
#         ts.append(torch.normal(0, 1.0, [512 * 1024], dtype=dt, device=dev))

#     c, dc, total_size, comp_size = get_float_comp_timings(ts)
#     ratio = comp_size / total_size
#     bw = (total_size / 1e9) / (c * 1e-3)
#     dc_bw = (total_size / 1e9) / (dc * 1e-3)

#     print(f"Float codec batched perf [128, [512 * 1024]] {dt}")
#     print(
#         "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
#             c, c_bw, total_size, comp_size, ratio
#         )
#     )
#     print(f"decomp time {dc:.3f} ms B/W {dc_bw:.1f} GB/s")

print("\n")

for dt in [torch.bfloat16, torch.float16, torch.float32]:
    # Non-batched
    ts = []
    ts.append(torch.normal(0, 1.0, [128 * 512 * 1024], dtype=dt, device=dev))

    c, dc, total_size, comp_size = get_any_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print(f"Raw ANS byte-wise non-batched perf [128 * 512 * 1024] {dt}")
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print(f"decomp time {dc:.3f} ms B/W {dc_bw:.1f} GB/s")

    # Batched
    ts = []
    for i in range(128):
        ts.append(torch.normal(0, 1.0, [512 * 1024], dtype=dt, device=dev))

    c, dc, total_size, comp_size = get_any_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print(f"Raw ANS byte-wise batched perf [128, [512 * 1024]] {dt}")
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print(f"decomp time {dc:.3f} ms B/W {dc_bw:.1f} GB/s")
