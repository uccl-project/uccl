#!/usr/bin/env python3
"""
Benchmark script for cudaMemcpyPeerAsync performance with multiple streams.

This script measures the performance of GPU-to-GPU memory copies using
cudaMemcpyPeerAsync with message sizes ranging from 1KB to 1GB, doubling
at each step. Uses multiple CUDA streams for concurrent operations.

Requirements:
- PyTorch with CUDA support
- At least 2 CUDA GPUs
- CUDA peer access enabled between GPUs
"""

import argparse
import time
import torch
import torch.cuda
import sys
import concurrent.futures
from typing import List, Tuple
import threading


def check_cuda_availability() -> bool:
    """Check if CUDA is available and we have at least 2 GPUs."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"❌ Need at least 2 GPUs, found {num_gpus}")
        return False
    
    print(f"✅ Found {num_gpus} CUDA GPUs")
    return True


def enable_peer_access(src_gpu: int, dst_gpu: int) -> bool:
    """Enable peer access between two GPUs."""
    try:
        # Set device to source GPU
        torch.cuda.set_device(src_gpu)
        
        # Check if peer access is possible
        can_access = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
        if not can_access:
            print(f"❌ Peer access not supported between GPU {src_gpu} and GPU {dst_gpu}")
            return False
        
        # Enable peer access
        torch.cuda.device(src_gpu).__enter__()
        try:
            # This will enable peer access if not already enabled
            test_tensor = torch.randn(10, device=f"cuda:{src_gpu}")
            test_tensor_peer = test_tensor.to(f"cuda:{dst_gpu}")
            print(f"✅ Peer access enabled between GPU {src_gpu} and GPU {dst_gpu}")
            return True
        except Exception as e:
            print(f"❌ Failed to enable peer access: {e}")
            return False
    except Exception as e:
        print(f"❌ Error setting up peer access: {e}")
        return False


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def format_bandwidth(bytes_per_sec: float) -> str:
    """Format bandwidth in bytes/sec to human readable format."""
    for unit in ['B/s', 'KB/s', 'MB/s', 'GB/s']:
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.2f} {unit}"
        bytes_per_sec /= 1024
    return f"{bytes_per_sec:.2f} TB/s"


def benchmark_memcpy_peer_multistream(
    src_gpu: int, 
    dst_gpu: int, 
    size: int, 
    num_streams: int = 4,
    num_warmup: int = 10, 
    num_iters: int = 100
) -> Tuple[float, float, float]:
    """
    Benchmark cudaMemcpyPeerAsync for a specific message size using multiple streams.
    
    Returns:
        Tuple of (average_time_ms, single_stream_bandwidth_gbps, aggregate_bandwidth_gbps)
    """
    # Calculate per-stream size
    per_stream_size = max(size // num_streams, 4)  # Minimum 4 bytes per stream
    total_actual_size = per_stream_size * num_streams
    
    # Allocate memory on source GPU
    torch.cuda.set_device(src_gpu)
    src_tensors = []
    for i in range(num_streams):
        tensor = torch.randn(per_stream_size // 4, dtype=torch.float32, device=f"cuda:{src_gpu}")
        src_tensors.append(tensor)
    
    # Allocate memory on destination GPU
    torch.cuda.set_device(dst_gpu)
    dst_tensors = []
    for i in range(num_streams):
        tensor = torch.empty(per_stream_size // 4, dtype=torch.float32, device=f"cuda:{dst_gpu}")
        dst_tensors.append(tensor)
    
    # Create CUDA streams
    streams = []
    for i in range(num_streams):
        stream = torch.cuda.Stream(device=src_gpu)
        streams.append(stream)
    
    # Warmup iterations
    torch.cuda.set_device(src_gpu)
    for _ in range(num_warmup):
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                dst_tensors[i].copy_(src_tensors[i], non_blocking=True)
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
    
    # Benchmark iterations - measure aggregate performance
    torch.cuda.synchronize()
    
    aggregate_times = []
    individual_times = [[] for _ in range(num_streams)]
    
    for iter_idx in range(num_iters):
        # Start timing for aggregate measurement
        torch.cuda.set_device(src_gpu)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Launch copies on all streams concurrently
        stream_events = []
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                stream_start = torch.cuda.Event(enable_timing=True)
                stream_end = torch.cuda.Event(enable_timing=True)
                
                stream_start.record(streams[i])
                dst_tensors[i].copy_(src_tensors[i], non_blocking=True)
                stream_end.record(streams[i])
                
                stream_events.append((stream_start, stream_end, streams[i]))
        
        # Wait for all streams to complete
        for stream in streams:
            stream.synchronize()
        
        end_event.record()
        torch.cuda.synchronize()
        
        # Record aggregate time
        aggregate_time = start_event.elapsed_time(end_event)
        aggregate_times.append(aggregate_time)
        
        # Record individual stream times
        for i, (stream_start, stream_end, stream) in enumerate(stream_events):
            individual_time = stream_start.elapsed_time(stream_end)
            individual_times[i].append(individual_time)
    
    # Calculate statistics
    avg_aggregate_time_ms = sum(aggregate_times) / len(aggregate_times)
    avg_individual_times = [sum(times) / len(times) for times in individual_times]
    avg_individual_time_ms = sum(avg_individual_times) / len(avg_individual_times)
    
    # Calculate bandwidths
    single_stream_bandwidth_bytes_per_sec = per_stream_size / (avg_individual_time_ms / 1000.0)
    single_stream_bandwidth_gbps = single_stream_bandwidth_bytes_per_sec / (1024**3)
    
    aggregate_bandwidth_bytes_per_sec = total_actual_size / (avg_aggregate_time_ms / 1000.0)
    aggregate_bandwidth_gbps = aggregate_bandwidth_bytes_per_sec / (1024**3)
    
    return avg_individual_time_ms, single_stream_bandwidth_gbps, aggregate_bandwidth_gbps


def run_benchmark(
    src_gpu: int, 
    dst_gpu: int, 
    num_streams: int = 4,
    start_size: int = 1024, 
    end_size: int = 1024**3,
    num_warmup: int = 10,
    num_iters: int = 100
) -> List[Tuple[int, float, float, float]]:
    """
    Run benchmark across all message sizes.
    
    Returns:
        List of (size, single_time_ms, single_bandwidth_gbps, aggregate_bandwidth_gbps) tuples
    """
    results = []
    current_size = start_size
    
    print(f"\n🚀 Benchmarking GPU-to-GPU memory copy (GPU {src_gpu} → GPU {dst_gpu})")
    print(f"📊 Streams: {num_streams}, Warmup: {num_warmup}, Iterations: {num_iters}")
    print("-" * 95)
    print(f"{'Size':>10} {'Single Time':>12} {'Single BW':>12} {'Aggregate BW':>15} {'Speedup':>10}")
    print(f"{'':>10} {'(ms)':>12} {'':>12} {'':>15} {'':>10}")
    print("-" * 95)
    
    while current_size <= end_size:
        try:
            single_time_ms, single_bandwidth_gbps, aggregate_bandwidth_gbps = benchmark_memcpy_peer_multistream(
                src_gpu, dst_gpu, current_size, num_streams, num_warmup, num_iters
            )
            
            results.append((current_size, single_time_ms, single_bandwidth_gbps, aggregate_bandwidth_gbps))
            
            # Calculate speedup
            speedup = aggregate_bandwidth_gbps / single_bandwidth_gbps if single_bandwidth_gbps > 0 else 0
            
            # Print progress
            size_str = format_size(current_size)
            single_bw_str = format_bandwidth(single_bandwidth_gbps * (1024**3))
            aggregate_bw_str = format_bandwidth(aggregate_bandwidth_gbps * (1024**3))
            speedup_str = f"{speedup:.1f}x"
            
            print(f"{size_str:>10} {single_time_ms:>9.3f} {single_bw_str:>12} {aggregate_bw_str:>15} {speedup_str:>10}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Out of memory at size {format_size(current_size)}")
            break
        except Exception as e:
            print(f"❌ Error at size {format_size(current_size)}: {e}")
            break
        
        # Double the size for next iteration
        current_size *= 2
    
    return results


def print_summary(results: List[Tuple[int, float, float, float]], src_gpu: int, dst_gpu: int, num_streams: int):
    """Print benchmark summary statistics."""
    if not results:
        print("❌ No successful benchmark results")
        return
    
    print("\n" + "=" * 95)
    print(f"📈 BENCHMARK SUMMARY (GPU {src_gpu} → GPU {dst_gpu}, {num_streams} streams)")
    print("=" * 95)
    
    # Find peak bandwidths
    max_single_bandwidth_result = max(results, key=lambda x: x[2])
    max_aggregate_bandwidth_result = max(results, key=lambda x: x[3])
    
    max_single_size, _, max_single_bw, _ = max_single_bandwidth_result
    max_agg_size, _, _, max_agg_bw = max_aggregate_bandwidth_result
    
    print(f"🏆 Peak single stream bandwidth: {format_bandwidth(max_single_bw * (1024**3))} at {format_size(max_single_size)}")
    print(f"🚀 Peak aggregate bandwidth: {format_bandwidth(max_agg_bw * (1024**3))} at {format_size(max_agg_size)}")
    
    # Calculate average speedup for large messages
    large_results = [r for r in results if r[0] >= 1024 * 1024]  # >= 1MB
    if large_results:
        avg_speedup = sum(r[3] / r[2] for r in large_results if r[2] > 0) / len(large_results)
        efficiency = (avg_speedup / num_streams) * 100
        print(f"⚡ Average speedup (large messages): {avg_speedup:.1f}x ({efficiency:.1f}% efficiency)")
    
    # Find fastest small message
    small_results = [r for r in results if r[0] <= 64 * 1024]  # <= 64KB
    if small_results:
        fastest_small = min(small_results, key=lambda x: x[1])
        small_size, small_time, _, _ = fastest_small
        print(f"⏱️  Fastest small message latency: {small_time:.3f}ms at {format_size(small_size)}")
    
    print(f"📏 Message sizes tested: {len(results)}")
    print(f"📐 Size range: {format_size(results[0][0])} to {format_size(results[-1][0])}")
    print(f"🔄 Streams used: {num_streams}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cudaMemcpyPeerAsync performance with multiple streams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--src-gpu", 
        type=int, 
        default=0, 
        help="Source GPU index"
    )
    
    parser.add_argument(
        "--dst-gpu", 
        type=int, 
        default=1, 
        help="Destination GPU index"
    )
    
    parser.add_argument(
        "--streams", 
        type=int, 
        default=4, 
        help="Number of CUDA streams to use"
    )
    
    parser.add_argument(
        "--start-size", 
        type=int, 
        default=1024, 
        help="Starting message size in bytes"
    )
    
    parser.add_argument(
        "--end-size", 
        type=str, 
        default="1G", 
        help="Ending message size (supports K, M, G suffixes)"
    )
    
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=10, 
        help="Number of warmup iterations"
    )
    
    parser.add_argument(
        "--iters", 
        type=int, 
        default=100, 
        help="Number of benchmark iterations per size"
    )
    
    parser.add_argument(
        "--csv", 
        type=str, 
        help="Save results to CSV file"
    )
    
    args = parser.parse_args()
    
    # Validate streams parameter
    if args.streams < 1:
        print("❌ Number of streams must be at least 1")
        sys.exit(1)
    if args.streams > 32:
        print("❌ Number of streams should not exceed 32 for reasonable performance")
        sys.exit(1)
    
    # Parse end size
    end_size_str = args.end_size.upper()
    if end_size_str.endswith('K'):
        end_size = int(end_size_str[:-1]) * 1024
    elif end_size_str.endswith('M'):
        end_size = int(end_size_str[:-1]) * 1024 * 1024
    elif end_size_str.endswith('G'):
        end_size = int(end_size_str[:-1]) * 1024 * 1024 * 1024
    else:
        end_size = int(end_size_str)
    
    print("🔧 CUDA Peer Memory Copy Benchmark (Multi-Stream)")
    print("=" * 60)
    
    # Check CUDA availability
    if not check_cuda_availability():
        sys.exit(1)
    
    # Validate GPU indices
    num_gpus = torch.cuda.device_count()
    if args.src_gpu >= num_gpus or args.dst_gpu >= num_gpus:
        print(f"❌ Invalid GPU indices. Available GPUs: 0-{num_gpus-1}")
        sys.exit(1)
    
    if args.src_gpu == args.dst_gpu:
        print("❌ Source and destination GPUs must be different")
        sys.exit(1)
    
    # Enable peer access
    if not enable_peer_access(args.src_gpu, args.dst_gpu):
        sys.exit(1)
    
    # Print GPU information
    torch.cuda.set_device(args.src_gpu)
    src_name = torch.cuda.get_device_name(args.src_gpu)
    dst_name = torch.cuda.get_device_name(args.dst_gpu)
    
    print(f"📱 Source GPU {args.src_gpu}: {src_name}")
    print(f"📱 Destination GPU {args.dst_gpu}: {dst_name}")
    print(f"🔄 Number of streams: {args.streams}")
    print(f"📏 Size range: {format_size(args.start_size)} to {format_size(end_size)}")
    
    try:
        # Run benchmark
        results = run_benchmark(
            args.src_gpu, 
            args.dst_gpu, 
            args.streams,
            args.start_size, 
            end_size,
            args.warmup,
            args.iters
        )
        
        # Print summary
        print_summary(results, args.src_gpu, args.dst_gpu, args.streams)
        
        # Save to CSV if requested
        if args.csv:
            import csv
            with open(args.csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'Size (bytes)', 'Size (human)', 'Single Stream Time (ms)', 
                    'Single Stream BW (GB/s)', 'Aggregate BW (GB/s)', 'Speedup', 'Efficiency (%)'
                ])
                for size, single_time_ms, single_bw_gbps, agg_bw_gbps in results:
                    speedup = agg_bw_gbps / single_bw_gbps if single_bw_gbps > 0 else 0
                    efficiency = (speedup / args.streams) * 100
                    writer.writerow([
                        size, 
                        format_size(size), 
                        f"{single_time_ms:.3f}", 
                        f"{single_bw_gbps:.3f}",
                        f"{agg_bw_gbps:.3f}",
                        f"{speedup:.2f}",
                        f"{efficiency:.1f}"
                    ])
            print(f"💾 Results saved to {args.csv}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 