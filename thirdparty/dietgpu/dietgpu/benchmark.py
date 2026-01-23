# Copyright (c) (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Simple benchmarking script for both float and raw byte-wise ANS codecs in
# PyTorch using the asynchronous API, as applied to floating point data
# ~ N(0, 1)

import torch
import csv
from datetime import datetime

torch.ops.load_library("/home/yangzhou/shuangma/uccl/thirdparty/dietgpu/build/lib.linux-x86_64-cpython-314/p2p_dietgpu.cpython-314-x86_64-linux-gnu.so")
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


def run_benchmark_suite():
    """Run benchmarks for different tensor sizes and data types"""
    # Define test configurations
    test_sizes_kb = [256, 512, 1024, 2048,4*1024, 10240,16*1024,32*1024,100*1024]  # 246KB, 512KB, 1MB, 2MB
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    dtype_names = ['bfloat16', 'float16', 'float32']

    results = []

    print("=" * 80)
    print("Running Benchmark Suite: Testing Different Tensor Sizes and Float Types")
    print("=" * 80)

    for size_kb in test_sizes_kb:
        for dt, dt_name in zip(dtypes, dtype_names):
            # Calculate number of elements based on dtype
            bytes_total = size_kb * 1024
            if dt in [torch.bfloat16, torch.float16]:
                num_elements = bytes_total // 2
            else:  # float32
                num_elements = bytes_total // 4

            print(f"\nTesting: Size={size_kb}KB, Dtype={dt_name}, Elements={num_elements}")

            # Test Float codec
            ts = []
            ts.append(torch.normal(0, 1.0, [num_elements], dtype=dt, device=dev))

            c, dc, total_size, comp_size = get_float_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            results.append({
                'Codec Type': 'Float',
                'Size (KB)': size_kb,
                'Data Type': dt_name,
                'Elements': num_elements,
                'Original Size (bytes)': total_size,
                'Compressed Size (bytes)': comp_size,
                'Compression Ratio': f"{ratio:.4f}",
                'Compress Time (ms)': f"{c:.3f}",
                'Compress BW (GB/s)': f"{c_bw:.2f}",
                'Decompress Time (ms)': f"{dc:.3f}",
                'Decompress BW (GB/s)': f"{dc_bw:.2f}"
            })

            print(f"  Float Codec - Comp: {c:.3f}ms ({c_bw:.2f} GB/s), "
                  f"Decomp: {dc:.3f}ms ({dc_bw:.2f} GB/s), Ratio: {ratio:.4f}")

            # Test Raw ANS byte-wise codec
            ts = []
            ts.append(torch.normal(0, 1.0, [num_elements], dtype=dt, device=dev))

            c, dc, total_size, comp_size = get_any_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            results.append({
                'Codec Type': 'Raw ANS',
                'Size (KB)': size_kb,
                'Data Type': dt_name,
                'Elements': num_elements,
                'Original Size (bytes)': total_size,
                'Compressed Size (bytes)': comp_size,
                'Compression Ratio': f"{ratio:.4f}",
                'Compress Time (ms)': f"{c:.3f}",
                'Compress BW (GB/s)': f"{c_bw:.2f}",
                'Decompress Time (ms)': f"{dc:.3f}",
                'Decompress BW (GB/s)': f"{dc_bw:.2f}"
            })

            print(f"  Raw ANS Codec - Comp: {c:.3f}ms ({c_bw:.2f} GB/s), "
                  f"Decomp: {dc:.3f}ms ({dc_bw:.2f} GB/s), Ratio: {ratio:.4f}")

    return results


def save_results_to_csv(results, filename=None):
    """Save benchmark results to CSV file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.csv"

    fieldnames = [
        'Codec Type', 'Size (KB)', 'Data Type', 'Elements',
        'Original Size (bytes)', 'Compressed Size (bytes)', 'Compression Ratio',
        'Compress Time (ms)', 'Compress BW (GB/s)',
        'Decompress Time (ms)', 'Decompress BW (GB/s)'
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {filename}")
    print(f"{'=' * 80}")
    return filename


def print_summary_table(results):
    """Print a formatted summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Group by size and dtype
    print(f"\n{'Codec':<10} {'Size (KB)':<8} {'DType':<10} {'Comp(ms)':<12} {'Decomp(ms)':<12} {'Ratio':<10} {'Comp BW':<12} {'DeComp BW':<12}")
    print("-" * 94)

    for result in results:
        print(f"{result['Codec Type']:<10} "
              f"{result['Size (KB)']:<8} "
              f"{result['Data Type']:<10} "
              f"{result['Compress Time (ms)']:<12} "
              f"{result['Decompress Time (ms)']:<12} "
              f"{result['Compression Ratio']:<10} "
              f"{result['Compress BW (GB/s)']:<12} "
              f"{result['Decompress BW (GB/s)']:<12}")


# Run the benchmark suite
results = run_benchmark_suite()

# Print summary table
print_summary_table(results)

# Save to CSV
csv_filename = save_results_to_csv(results)
