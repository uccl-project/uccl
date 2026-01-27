import torch
import numpy as np
from pack_unpack_triton import (
    pack_moe_data_to_buffers_triton,
    unpack_moe_data_from_buffers_triton,
    count_expert_tokens_triton,
)

# Try to import CUDA backend
try:
    from pack_unpack_cuda import (
        pack_moe_data_to_buffers_cuda,
        unpack_moe_data_from_buffers_cuda,
        CUDA_AVAILABLE,
    )
except ImportError:
    CUDA_AVAILABLE = False
    pack_moe_data_to_buffers_cuda = None
    unpack_moe_data_from_buffers_cuda = None


# Test configurations (DeepSeek-V3 style)
configs = [
    {"tokens": 128, "hidden": 7168, "experts": 256, "topk": 8, "name": "Decode batch"},
    {
        "tokens": 4096,
        "hidden": 7168,
        "experts": 256,
        "topk": 8,
        "name": "Prefill batch",
    },
]


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def verify_correctness(
    x, topk_idx, topk_weights, recv_x, recv_idx, recv_weights, expert_counts
):
    """Verify that pack/unpack roundtrip preserves data."""
    # Count expected items (excluding -1 padding)
    expected_items = (topk_idx >= 0).sum().item()
    actual_items = recv_x.shape[0]

    if expected_items == 0:
        return True, "No items to verify (all padding)"

    # Check total items match
    if actual_items < expected_items:
        return (
            False,
            f"Item count mismatch: expected {expected_items}, got {actual_items}",
        )

    # Check expert counts sum to total items
    total_counted = sum(expert_counts)
    if total_counted != expected_items:
        return False, f"Expert counts sum {total_counted} != expected {expected_items}"

    return True, "✓ Correctness verified"


def benchmark_backend(
    backend_name,
    pack_fn,
    unpack_fn,
    x,
    topk_idx,
    topk_weights,
    buffers,
    cfg,
    world_size,
    num_local_experts,
    num_iterations=100,
):
    """Benchmark a single backend."""
    # Clone inputs to avoid modifying originals
    x_copy = x.clone()
    topk_idx_copy = topk_idx.clone()
    topk_weights_copy = topk_weights.clone()
    buffers_copy = [b.clone() for b in buffers]

    # ========== Pack Benchmark ==========
    torch.cuda.synchronize()
    for _ in range(50):
        per_rank_bytes = pack_fn(
            x_copy,
            topk_idx_copy,
            topk_weights_copy,
            cfg["experts"],
            world_size,
            torch.device("cuda"),
            buffers_copy,
        )
    torch.cuda.synchronize()

    pack_events_start = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]
    pack_events_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]

    # Store packed buffers and per_rank_bytes from last iteration for unpack
    packed_buffers = None
    packed_per_rank_bytes = None

    for i in range(num_iterations):
        # Reset buffers for each iteration
        buffers_copy = [b.clone() for b in buffers]
        pack_events_start[i].record()
        per_rank_bytes = pack_fn(
            x_copy,
            topk_idx_copy,
            topk_weights_copy,
            cfg["experts"],
            world_size,
            torch.device("cuda"),
            buffers_copy,
        )
        pack_events_end[i].record()

        # Save last iteration's results for unpack benchmark
        if i == num_iterations - 1:
            packed_buffers = buffers_copy
            packed_per_rank_bytes = per_rank_bytes.clone()

    torch.cuda.synchronize()
    pack_times = np.array(
        [s.elapsed_time(e) for s, e in zip(pack_events_start, pack_events_end)]
    )

    # ========== Unpack Benchmark ==========
    # Use the packed buffers from the last pack iteration
    assert packed_buffers is not None, "No packed buffers available"
    assert packed_per_rank_bytes is not None, "No per_rank_bytes available"

    torch.cuda.synchronize()
    for _ in range(50):
        recv_x, recv_idx, recv_weights, expert_counts = unpack_fn(
            packed_buffers,
            packed_per_rank_bytes,
            num_local_experts,
            cfg["hidden"],
            world_size,
            torch.device("cuda"),
            torch.bfloat16,
            torch.int64,
            torch.float32,
        )
    torch.cuda.synchronize()

    unpack_events_start = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]
    unpack_events_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
    ]

    for i in range(num_iterations):
        unpack_events_start[i].record()
        recv_x, recv_idx, recv_weights, expert_counts = unpack_fn(
            packed_buffers,
            packed_per_rank_bytes,
            num_local_experts,
            cfg["hidden"],
            world_size,
            torch.device("cuda"),
            torch.bfloat16,
            torch.int64,
            torch.float32,
        )
        unpack_events_end[i].record()

    torch.cuda.synchronize()
    unpack_times = np.array(
        [s.elapsed_time(e) for s, e in zip(unpack_events_start, unpack_events_end)]
    )

    # Calculate statistics
    pack_stats = {
        "mean": np.mean(pack_times),
        "min": np.min(pack_times),
        "max": np.max(pack_times),
        "std": np.std(pack_times),
        "p50": np.percentile(pack_times, 50),
        "p99": np.percentile(pack_times, 99),
    }

    unpack_stats = {
        "mean": np.mean(unpack_times),
        "min": np.min(unpack_times),
        "max": np.max(unpack_times),
        "std": np.std(unpack_times),
        "p50": np.percentile(unpack_times, 50),
        "p99": np.percentile(unpack_times, 99),
    }

    # Calculate throughput
    total_items = (topk_idx_copy >= 0).sum().item()
    total_bytes = packed_per_rank_bytes.sum().item()

    pack_stats["tokens_per_sec"] = (cfg["tokens"] / pack_stats["mean"]) * 1000
    pack_stats["gb_per_sec"] = (total_bytes / pack_stats["mean"]) / 1e6
    pack_stats["total_bytes"] = total_bytes
    pack_stats["total_items"] = total_items

    unpack_stats["tokens_per_sec"] = (recv_x.shape[0] / unpack_stats["mean"]) * 1000
    unpack_stats["gb_per_sec"] = (total_bytes / unpack_stats["mean"]) / 1e6
    unpack_stats["items_processed"] = recv_x.shape[0]

    # Correctness check
    is_correct, correctness_msg = verify_correctness(
        x_copy,
        topk_idx_copy,
        topk_weights_copy,
        recv_x,
        recv_idx,
        recv_weights,
        expert_counts,
    )

    return {
        "backend": backend_name,
        "pack": pack_stats,
        "unpack": unpack_stats,
        "correctness": (is_correct, correctness_msg),
    }


for cfg in configs:
    print(f"Testing: {cfg['name']} ({cfg['tokens']} tokens, {cfg['hidden']} hidden)")
    print("=" * 70)

    # Setup
    x = torch.randn(cfg["tokens"], cfg["hidden"], dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0,
        cfg["experts"],
        (cfg["tokens"], cfg["topk"]),
        dtype=torch.int64,
        device="cuda",
    )
    topk_weights = torch.randn(
        cfg["tokens"], cfg["topk"], dtype=torch.float32, device="cuda"
    )

    world_size = 8
    num_local_experts = cfg["experts"] // world_size
    bytes_per_item = cfg["hidden"] * 2 + 8 + 4  # bfloat16=2, int64=8, float32=4
    max_items = cfg["tokens"] * cfg["topk"]
    buffers = [
        torch.zeros(max_items * bytes_per_item, dtype=torch.uint8, device="cuda")
        for _ in range(world_size)
    ]

    results = []

    # Benchmark Triton backend
    print("\n[1/2] Benchmarking Triton backend...")
    try:
        triton_result = benchmark_backend(
            "Triton",
            pack_moe_data_to_buffers_triton,
            unpack_moe_data_from_buffers_triton,
            x,
            topk_idx,
            topk_weights,
            buffers,
            cfg,
            world_size,
            num_local_experts,
        )
        results.append(triton_result)
    except Exception as e:
        print(f"  ❌ Triton backend failed: {e}")
        results.append(None)

    # Benchmark CUDA backend
    print("[2/2] Benchmarking CUDA backend...")
    if CUDA_AVAILABLE and pack_moe_data_to_buffers_cuda is not None:
        try:
            cuda_result = benchmark_backend(
                "CUDA",
                pack_moe_data_to_buffers_cuda,
                unpack_moe_data_from_buffers_cuda,
                x,
                topk_idx,
                topk_weights,
                buffers,
                cfg,
                world_size,
                num_local_experts,
            )
            results.append(cuda_result)
        except Exception as e:
            print(f"  ❌ CUDA backend failed: {e}")
            results.append(None)
    else:
        print("  ⚠️  CUDA backend not available (extension not built)")
        results.append(None)

    # ========== Display Results ==========
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Pack comparison
    print("\n📦 Pack Performance:")
    print("-" * 70)
    print(
        f"{'Backend':<12} {'Mean (ms)':<12} {'Min':<10} {'Max':<10} {'Std':<10} {'P99':<10} {'Tokens/s':<12} {'GB/s':<10}"
    )
    print("-" * 70)

    for result in results:
        if result is None:
            continue
        p = result["pack"]
        print(
            f"{result['backend']:<12} {p['mean']:>11.3f}  {p['min']:>9.3f}  {p['max']:>9.3f}  "
            f"{p['std']:>9.3f}  {p['p99']:>9.3f}  {p['tokens_per_sec']:>11.1f}  {p['gb_per_sec']:>9.2f}"
        )

    # Unpack comparison
    print("\n📤 Unpack Performance:")
    print("-" * 70)
    print(
        f"{'Backend':<12} {'Mean (ms)':<12} {'Min':<10} {'Max':<10} {'Std':<10} {'P99':<10} {'Tokens/s':<12} {'GB/s':<10}"
    )
    print("-" * 70)

    for result in results:
        if result is None:
            continue
        u = result["unpack"]
        print(
            f"{result['backend']:<12} {u['mean']:>11.3f}  {u['min']:>9.3f}  {u['max']:>9.3f}  "
            f"{u['std']:>9.3f}  {u['p99']:>9.3f}  {u['tokens_per_sec']:>11.1f}  {u['gb_per_sec']:>9.2f}"
        )

    # Total time comparison
    print("\n⏱️  Total Time (Pack + Unpack):")
    print("-" * 70)
    for result in results:
        if result is None:
            continue
        total = result["pack"]["mean"] + result["unpack"]["mean"]
        print(f"  {result['backend']:<12}: {total:>7.3f} ms")

    # Speedup comparison
    if len(results) == 2 and results[0] is not None and results[1] is not None:
        print("\n🚀 Speedup (CUDA vs Triton):")
        print("-" * 70)
        triton_total = results[0]["pack"]["mean"] + results[0]["unpack"]["mean"]
        cuda_total = results[1]["pack"]["mean"] + results[1]["unpack"]["mean"]

        pack_speedup = results[0]["pack"]["mean"] / results[1]["pack"]["mean"]
        unpack_speedup = results[0]["unpack"]["mean"] / results[1]["unpack"]["mean"]
        total_speedup = triton_total / cuda_total

        print(f"  Pack:   {pack_speedup:.2f}x")
        print(f"  Unpack: {unpack_speedup:.2f}x")
        print(f"  Total:  {total_speedup:.2f}x")

        if total_speedup > 1.0:
            print(f"  → CUDA is {total_speedup:.2f}x faster")
            if total_speedup > 10:
                print(
                    f"  ⚠️  Large speedup expected: Triton uses Python loops, CUDA uses GPU kernels"
                )
        elif total_speedup < 1.0:
            print(f"  → Triton is {1/total_speedup:.2f}x faster")
        else:
            print(f"  → Performance is similar")

    # Correctness
    print("\n✅ Correctness:")
    print("-" * 70)
    for result in results:
        if result is None:
            continue
        is_correct, msg = result["correctness"]
        status = "✓" if is_correct else "✗"
        print(f"  {result['backend']:<12}: {status} {msg}")

    print("\n" + "=" * 70 + "\n")

print("Benchmark complete!")
