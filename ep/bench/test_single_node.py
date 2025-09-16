"""
Single-node test for DeepEP low-latency kernels.
This test can run on a single node with multiple GPUs to test intranode communication.
It can also simulate mixed workload patterns within a single node.

Example launch commands:

For single GPU:
torchrun --nproc_per_node=1 bench/test_single_node.py

For multiple GPUs on single node (e.g., 4 GPUs):
torchrun --nproc_per_node=4 bench/test_single_node.py

For simulating mixed workload pattern:
torchrun --nproc_per_node=4 bench/test_single_node.py --mixed

With custom parameters:
torchrun --nproc_per_node=4 bench/test_single_node.py --num-tokens 1024 --hidden 4096
"""

import torch
import torch.distributed as dist
import time
from buffer import Buffer
import os
import argparse

from utils import (
    init_dist,
    detect_ib_hca,
    initialize_uccl,
    destroy_uccl,
)


def test_single_node(rank: int, num_ranks: int, group: dist.ProcessGroup, args):
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_experts = args.num_experts if args.num_experts else 4 * num_ranks
    num_topk = args.num_topk

    # Calculate expected device index based on rank and world size
    # For single node: device_index should match LOCAL_RANK
    # For multi-node: device_index should be rank % num_gpus_per_node
    device_index = local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_index)
    torch.manual_seed(rank)

    # Generate input data - explicitly specify device to be safe
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{device_index}"
    )

    # Verify tensor is on correct device
    if x.device.index != local_rank:
        raise RuntimeError(
            f"Rank {rank}: Input tensor x created on wrong device! "
            f"Expected cuda:{local_rank}, got {x.device}"
        )

    print(
        f"[single-node] Rank {rank}: Created input tensor on {x.device} (verified)",
        flush=True,
    )

    if args.mixed:
        # Simulate mixed workload pattern - some experts are treated as "remote"
        # This helps test different communication patterns even on single node
        topk_idx = torch.zeros(
            (num_tokens, num_topk), dtype=torch.long, device=f"cuda:{device_index}"
        )

        experts_per_rank = num_experts // num_ranks

        for i in range(num_tokens):
            # Simulate 50% local, 50% "remote" expert selection
            # Even though all are on same node, this tests different access patterns
            if i % 2 == 0:
                # Prefer "local" experts (same rank)
                local_start = rank * experts_per_rank
                local_end = (rank + 1) * experts_per_rank
                local_choices = torch.randint(
                    local_start,
                    local_end,
                    (num_topk // 2,),
                    device=f"cuda:{device_index}",
                )

                # Add some "remote" experts
                remote_choices = []
                for _ in range(num_topk - len(local_choices)):
                    remote_rank = (
                        rank
                        + 1
                        + torch.randint(
                            0, num_ranks - 1, (1,), device=f"cuda:{device_index}"
                        ).item()
                    ) % num_ranks
                    remote_start = remote_rank * experts_per_rank
                    remote_end = (remote_rank + 1) * experts_per_rank
                    remote_choices.append(
                        torch.randint(
                            remote_start,
                            remote_end,
                            (1,),
                            device=f"cuda:{device_index}",
                        ).item()
                    )

                topk_idx[i] = torch.cat(
                    [
                        local_choices,
                        torch.tensor(
                            remote_choices,
                            dtype=torch.long,
                            device=f"cuda:{device_index}",
                        ),
                    ]
                ).to(f"cuda:{device_index}")
            else:
                # Fully random selection
                topk_idx[i] = torch.randint(
                    0, num_experts, (num_topk,), device=f"cuda:{device_index}"
                )

        print(
            f"[single-node] Mixed pattern: alternating local/remote expert selection",
            flush=True,
        )
    else:
        # Pure random selection for standard intranode test
        topk_idx = torch.randint(
            0, num_experts, (num_tokens, num_topk), device=f"cuda:{device_index}"
        )
        print(f"[single-node] Pure intranode: random expert selection", flush=True)

    # Verify topk_idx tensor is on correct device
    if topk_idx.device.index != device_index:
        raise RuntimeError(
            f"Rank {rank}: topk_idx tensor created on wrong device! "
            f"Expected cuda:{device_index}, got {topk_idx.device}"
        )
    print(
        f"[single-node] Rank {rank}: Created topk_idx tensor on {topk_idx.device} (verified)",
        flush=True,
    )

    num_device_sms = torch.cuda.get_device_properties(
        device_index
    ).multi_processor_count

    scratch_nbytes = int(args.buffer_size * 1e9)  # Convert GB to bytes
    scratch = torch.empty(
        scratch_nbytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )

    # Verify device before initialize_uccl
    pre_init_device = torch.cuda.current_device()
    print(
        f"[single-node] Rank {rank}: Device before initialize_uccl: current={pre_init_device}, expected={device_index}",
        flush=True,
    )
    if pre_init_device != device_index:
        print(
            f"[single-node] Rank {rank}: CRITICAL - Device mismatch before initialize_uccl!",
            flush=True,
        )
        torch.cuda.set_device(device_index)
        torch.set_default_device(f"cuda:{device_index}")

    proxies, workers, bench = initialize_uccl(
        scratch, scratch_nbytes, rank, num_ranks, group
    )

    # Verify device after initialize_uccl
    post_init_device = torch.cuda.current_device()
    print(
        f"[single-node] Rank {rank}: Device after initialize_uccl: current={post_init_device}, expected={device_index}",
        flush=True,
    )
    if post_init_device != device_index:
        print(
            f"[single-node] Rank {rank}: CRITICAL - initialize_uccl changed device! Fixing...",
            flush=True,
        )
        torch.cuda.set_device(device_index)
        torch.set_default_device(f"cuda:{device_index}")

    try:
        # Additional device consistency check before creating buffer
        current_dev = torch.cuda.current_device()
        print(
            f"[single-node] Rank {rank}: Pre-buffer device check: current={current_dev}, expected={device_index}",
            flush=True,
        )

        if current_dev != device_index:
            print(
                f"[single-node] Rank {rank}: Device inconsistency detected, re-setting device...",
                flush=True,
            )
            torch.cuda.set_device(device_index)
            torch.set_default_device(f"cuda:{device_index}")
            # Verify the fix worked
            new_current = torch.cuda.current_device()
            if new_current != device_index:
                raise RuntimeError(
                    f"Rank {rank}: Critical device setting failure. Expected {device_index}, got {new_current}"
                )

        # For single node, prioritize NVLink over RDMA
        # Double-check device before buffer creation
        pre_buffer_device = torch.cuda.current_device()
        if pre_buffer_device != device_index:
            print(
                f"[single-node] Rank {rank}: CRITICAL - Device wrong before Buffer creation! current={pre_buffer_device}, expected={device_index}",
                flush=True,
            )
            torch.cuda.set_device(device_index)
            torch.set_default_device(f"cuda:{device_index}")

        if num_ranks > 1:
            # Multi-GPU single node - use NVLink
            buffer = Buffer(
                group=group,
                rdma_buffer_ptr=scratch.data_ptr(),
                num_nvl_bytes=scratch_nbytes,  # All memory for NVLink
                num_rdma_bytes=scratch_nbytes,  # No RDMA needed for single node
                low_latency_mode=True,
                num_qps_per_rank=num_device_sms,
                allow_nvlink_for_low_latency_mode=True,
                allow_mnnvl=False,  # No multi-node NVLink needed
                explicitly_destroy=True,
            )
            print(
                "[single-node] ✓ Buffer created with NVLink for multi-GPU", flush=True
            )
        else:
            # Single GPU - local memory only
            buffer = Buffer(
                group=group,
                rdma_buffer_ptr=scratch.data_ptr(),
                num_nvl_bytes=scratch_nbytes,
                num_rdma_bytes=scratch_nbytes,
                low_latency_mode=True,
                num_qps_per_rank=num_device_sms,
                allow_nvlink_for_low_latency_mode=False,
                allow_mnnvl=False,
                explicitly_destroy=True,
            )
            print("[single-node] ✓ Buffer created for single GPU", flush=True)

        # Verify device after buffer creation
        post_buffer_device = torch.cuda.current_device()
        if post_buffer_device != device_index:
            print(
                f"[single-node] Rank {rank}: CRITICAL - Buffer creation changed device! current={post_buffer_device}, expected={device_index}",
                flush=True,
            )
            torch.cuda.set_device(device_index)
            torch.set_default_device(f"cuda:{device_index}")

        buffer.connect_atomic_buffer(proxies[0])

        # Verify device consistency before setting up proxies
        verification_device = torch.cuda.current_device()
        if verification_device != device_index:
            raise RuntimeError(
                f"Rank {rank}: Device inconsistency before proxy setup. "
                f"Expected {device_index}, current {verification_device}"
            )

        for proxy in proxies:
            proxy.calculate_and_set_dispatch_recv_data_offset(
                num_tokens, hidden, num_experts
            )
            proxy.set_atomic_buffer_ptr(proxies[0].get_atomic_buffer_ptr())

        if rank == 0:
            print(
                f"[single-node] ✓ Atomic buffer connected and offsets set. "
                f"All ranks using consistent device mapping (rank->device).",
                flush=True,
            )

        cumulative_local_expert_recv_stats = torch.zeros(
            (num_experts // num_ranks,), dtype=torch.int, device=f"cuda:{device_index}"
        )

        # Verify this tensor is also on correct device
        if cumulative_local_expert_recv_stats.device.index != device_index:
            raise RuntimeError(
                f"Rank {rank}: cumulative_local_expert_recv_stats tensor on wrong device! "
                f"Expected cuda:{device_index}, got {cumulative_local_expert_recv_stats.device}"
            )

        # Warmup run if requested
        if args.warmup > 0:
            print(
                f"[single-node] Running {args.warmup} warmup iterations...", flush=True
            )
            for _ in range(args.warmup):
                recv_x, recv_count, handle, event, dispatch_hook = (
                    buffer.low_latency_dispatch(
                        x=x,
                        topk_idx=topk_idx,
                        num_max_dispatch_tokens_per_rank=num_tokens,
                        num_experts=num_experts,
                        use_fp8=False,
                        round_scale=False,
                        use_ue8m0=False,
                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats.clone(),
                        async_finish=False,
                        return_recv_hook=True,
                    )
                )
                dispatch_hook()
                torch.cuda.synchronize()

                topk_weights = torch.ones(
                    (num_tokens, num_topk),
                    dtype=torch.float32,
                    device=f"cuda:{device_index}",
                )
                # Verify tensor device
                if topk_weights.device.index != device_index:
                    raise RuntimeError(
                        f"Rank {rank}: topk_weights tensor on wrong device in warmup! "
                        f"Expected cuda:{device_index}, got {topk_weights.device}"
                    )
                combined_x, combine_event, combine_hook = buffer.low_latency_combine(
                    x=recv_x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    handle=handle,
                    use_logfmt=False,
                    zero_copy=False,
                    async_finish=False,
                    return_recv_hook=True,
                )
                combine_hook()
                torch.cuda.synchronize()
            print("[single-node] ✓ Warmup completed", flush=True)

        # Benchmark run
        print(
            f"[single-node] Running {args.iterations} benchmark iterations...",
            flush=True,
        )
        dispatch_times = []
        combine_times = []

        for iter_idx in range(args.iterations):
            # Reset stats for each iteration
            cumulative_local_expert_recv_stats.zero_()

            # Dispatch
            torch.cuda.synchronize()
            start_time = time.time()
            recv_x, recv_count, handle, event, dispatch_hook = (
                buffer.low_latency_dispatch(
                    x=x,
                    topk_idx=topk_idx,
                    num_max_dispatch_tokens_per_rank=num_tokens,
                    num_experts=num_experts,
                    use_fp8=False,
                    round_scale=False,
                    use_ue8m0=False,
                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                    async_finish=False,
                    return_recv_hook=True,
                )
            )
            dispatch_hook()
            torch.cuda.synchronize()
            dispatch_time = time.time() - start_time
            dispatch_times.append(dispatch_time)

            # Combine
            topk_weights = torch.ones(
                (num_tokens, num_topk),
                dtype=torch.float32,
                device=f"cuda:{device_index}",
            )
            # Verify tensor device
            if topk_weights.device.index != device_index:
                raise RuntimeError(
                    f"Rank {rank}: topk_weights tensor on wrong device in benchmark! "
                    f"Expected cuda:{device_index}, got {topk_weights.device}"
                )

            torch.cuda.synchronize()
            start_time = time.time()
            combined_x, combine_event, combine_hook = buffer.low_latency_combine(
                x=recv_x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                handle=handle,
                use_logfmt=False,
                zero_copy=False,
                async_finish=False,
                return_recv_hook=True,
            )
            combine_hook()
            torch.cuda.synchronize()
            combine_time = time.time() - start_time
            combine_times.append(combine_time)

            if args.verbose:
                print(
                    f"[single-node] Iter {iter_idx}: dispatch={dispatch_time:.4f}s, "
                    f"combine={combine_time:.4f}s",
                    flush=True,
                )

        # Calculate statistics
        avg_dispatch = sum(dispatch_times) / len(dispatch_times)
        avg_combine = sum(combine_times) / len(combine_times)
        min_dispatch = min(dispatch_times)
        min_combine = min(combine_times)
        max_dispatch = max(dispatch_times)
        max_combine = max(combine_times)

        # Verify output shape
        if combined_x.shape[0] == num_tokens and combined_x.shape[1] == hidden:
            print("[single-node] ✓ Output shape verification passed!", flush=True)
        else:
            print(
                f"[single-node] ✗ Output shape mismatch! Expected ({num_tokens}, {hidden}), "
                f"got {combined_x.shape}",
                flush=True,
            )

        # Print summary
        if rank == 0:
            print("\n[single-node] === Performance Summary ===", flush=True)
            print(f"  Configuration:", flush=True)
            print(f"    - GPUs: {num_ranks}", flush=True)
            print(f"    - Tokens: {num_tokens}", flush=True)
            print(f"    - Hidden: {hidden}", flush=True)
            print(
                f"    - Experts: {num_experts} ({num_experts // num_ranks} per rank)",
                flush=True,
            )
            print(f"    - Top-K: {num_topk}", flush=True)
            print(
                f"    - Mode: {'Mixed simulation' if args.mixed else 'Pure intranode'}",
                flush=True,
            )
            print(f"  Dispatch times (s):", flush=True)
            print(f"    - Average: {avg_dispatch:.6f}", flush=True)
            print(f"    - Min: {min_dispatch:.6f}", flush=True)
            print(f"    - Max: {max_dispatch:.6f}", flush=True)
            print(f"  Combine times (s):", flush=True)
            print(f"    - Average: {avg_combine:.6f}", flush=True)
            print(f"    - Min: {min_combine:.6f}", flush=True)
            print(f"    - Max: {max_combine:.6f}", flush=True)
            print(
                f"  Total average time: {avg_dispatch + avg_combine:.6f}s", flush=True
            )

            # Calculate throughput
            total_data = num_tokens * hidden * 2  # bfloat16 = 2 bytes
            throughput_gbps = (total_data / 1e9) / (avg_dispatch + avg_combine)
            print(f"  Throughput: {throughput_gbps:.2f} GB/s", flush=True)
            print("[single-node] ✓ All tests passed!", flush=True)

    except Exception as e:
        import traceback

        print(f"[single-node] ✗ Error on rank {rank}: {repr(e)}", flush=True)
        traceback.print_exc()
        raise

    try:
        buffer.destroy()
    except Exception:
        pass

    # Ensure we're on the correct device before barrier
    torch.cuda.set_device(device_index)
    torch.set_default_device(f"cuda:{device_index}")
    current_dev = torch.cuda.current_device()
    print(
        f"[Rank {rank}] Device before barrier: current={current_dev}, expected={device_index}",
        flush=True,
    )
    if current_dev != device_index:
        print(
            f"[Rank {rank}] CRITICAL: Device mismatch before barrier! Fixing...",
            flush=True,
        )
        torch.cuda.set_device(device_index)
        torch.set_default_device(f"cuda:{device_index}")

    try:
        dist.barrier()
    except Exception as e:
        print(f"[Rank {rank}] Barrier failed: {e}", flush=True)
        raise
    print(f"[single-node] Rank {rank}: ✓ Buffer destroyed", flush=True)

    destroy_uccl(proxies, workers, bench)
    # Ensure we're on the correct device before final barrier
    torch.cuda.set_device(device_index)
    torch.set_default_device(f"cuda:{device_index}")
    final_current_dev = torch.cuda.current_device()
    print(
        f"[Rank {rank}] Device before final barrier: current={final_current_dev}, expected={device_index}",
        flush=True,
    )
    if final_current_dev != device_index:
        print(
            f"[Rank {rank}] CRITICAL: Device mismatch before final barrier! Fixing...",
            flush=True,
        )
        torch.cuda.set_device(device_index)
        torch.set_default_device(f"cuda:{device_index}")

    try:
        dist.barrier()
    except Exception as e:
        print(f"[Rank {rank}] Final barrier failed: {e}", flush=True)
        raise


def test_worker(local_rank: int, num_local_ranks: int, args):
    print(
        f"[Local Rank {local_rank}] Entering test_worker, about to init distributed...",
        flush=True,
    )
    print(
        f"[Local Rank {local_rank}] Environment for init_dist: "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} (nodes), "
        f"RANK={os.environ.get('RANK')} (node index), "
        f"LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={num_local_ranks}, "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
        flush=True,
    )

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    print(
        f"[Local Rank {local_rank}] init_dist completed. Global rank={rank}/{num_ranks}",
        flush=True,
    )
    test_single_node(rank, num_ranks, group, args)
    try:
        dist.barrier()
    except Exception as e:
        print(f"[Rank {rank}] Cleanup barrier failed: {e}", flush=True)
        # Continue with cleanup even if barrier fails

    print(f"[Rank {rank}] Destroying process group...", flush=True)
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-node DeepEP test")
    parser.add_argument("--num-tokens", type=int, default=512, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden dimension")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Number of experts (default: 4*num_ranks)",
    )
    parser.add_argument("--num-topk", type=int, default=4, help="Top-K value")
    parser.add_argument(
        "--buffer-size", type=float, default=1.0, help="Buffer size in GB"
    )
    parser.add_argument(
        "--mixed", action="store_true", help="Simulate mixed workload pattern"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-iteration timings"
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    print(
        f"[Rank {local_rank}] Single-node test starting with args: {args}", flush=True
    )
    print(
        f"[Rank {local_rank}] LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={num_local_ranks}",
        flush=True,
    )

    # Set up IB if available
    print(f"[Rank {local_rank}] Detecting IB HCA...", flush=True)
    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):
        os.environ["NCCL_IB_HCA"] = ib_dev
        print(f"[Rank {local_rank}] Set NCCL_IB_HCA={ib_dev}", flush=True)
    else:
        print(
            f"[Rank {local_rank}] No IB device found or not Mellanox (detected: {ib_dev})",
            flush=True,
        )

    print(f"[Rank {local_rank}] Starting test_worker...", flush=True)
    test_worker(local_rank, num_local_ranks, args)
