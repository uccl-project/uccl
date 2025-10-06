# ruff: noqa: T201

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import nvshmem.core as nvshmem
import torch
from cuda.core.experimental import Device
from nvshmem.core import Teams

from all_to_all_utils import (
    ProcessGroupInfo,
    PyTorchStreamWrapper, 
    nvshmem_init,
    MoEConfig, 
    RankTestData,
    parallel_launch,
    parallel_launch_from_env,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
) -> tuple[tuple[int, int], torch.Tensor]:
    device = pgi.device
    num_dp = pgi.world_size // dp_size
    dp_rank = pgi.rank // dp_size

    # Generate the same rank data for each DP group
    rng = torch.Generator()
    rng.manual_seed(dp_rank + 1)
    rank_data = RankTestData(moe, rng, use_max_tokens=True)

    num_local_experts = moe.num_experts // pgi.world_size

    hidden_dim_scale_bytes = (
        0
        if moe.in_dtype.itemsize != 1
        else (
            (moe.hidden_dim + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )
    )

    hidden_dim_bytes_with_scale = moe.hidden_dim // dp_size * moe.in_dtype.itemsize
    if moe.in_dtype.itemsize == 1:
        hidden_dim_bytes_with_scale += (
            (moe.hidden_dim // dp_size + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )

    # Calculate per-rank send counts for sparse communication
    per_rank_token_counts = torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
    for i in range(rank_data.num_tokens):
        for j in range(moe.experts_per_token):
            expert_id = rank_data.indices[i, j].item()
            target_rank = expert_id // num_local_experts
            per_rank_token_counts[target_rank] += 1

    # Calculate send/recv sizes
    per_rank_send_bytes = per_rank_token_counts * hidden_dim_bytes_with_scale
    per_rank_recv_bytes = torch.zeros_like(per_rank_send_bytes)
    torch.distributed.all_to_all_single(per_rank_recv_bytes, per_rank_send_bytes)

    total_sparse_bytes = int(per_rank_send_bytes.sum().item())

    # Build sparse buffers for torch all_to_all_single
    sparse_in_splits = per_rank_send_bytes.tolist()
    sparse_out_splits = per_rank_recv_bytes.tolist()
    
    total_send_bytes = int(per_rank_send_bytes.sum().item())
    sparse_sendbuf = torch.randint(0, 256, (max(total_send_bytes, 1),), dtype=torch.uint8, device=device)
    
    total_recv_bytes = int(per_rank_recv_bytes.sum().item())
    sparse_recvbuf = torch.empty(max(total_recv_bytes, 1), dtype=torch.uint8, device=device)

    # NVSHMEM sparse implementation - use nvshmem.tensor() for symmetric memory
    # Note: nvshmem.tensor() creates proper symmetric memory across all PEs

    # Strategy: Create world_size x world_size symmetric buffers
    # All ranks create the same buffer structure to ensure symmetry
    # send_bufs[i][j] = buffer for rank i to send to rank j
    # recv_bufs[i][j] = buffer for rank i to receive from rank j

    # Gather all send/recv sizes so all ranks know all buffer sizes
    per_rank_send_bytes_all = [torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
                                for _ in range(pgi.world_size)]
    per_rank_recv_bytes_all = [torch.zeros(pgi.world_size, dtype=torch.int32, device=device)
                                for _ in range(pgi.world_size)]
    torch.distributed.all_gather(per_rank_send_bytes_all, per_rank_send_bytes)
    torch.distributed.all_gather(per_rank_recv_bytes_all, per_rank_recv_bytes)

    # Create symmetric send buffers - world_size buffers, one for each target
    # All ranks create buffers in same order with same sizes -> symmetric
    nvshmem_send_bufs = []
    for target_rank in range(pgi.world_size):
        # Find max send size across all ranks for this target
        max_send_size = 0
        for sender_rank in range(pgi.world_size):
            send_size = int(per_rank_send_bytes_all[sender_rank][target_rank].item())
            max_send_size = max(max_send_size, send_size)

        # All ranks create buffer with same size -> symmetric memory
        send_buf = nvshmem.tensor((max(max_send_size, 1),), dtype=torch.uint8)
        nvshmem_send_bufs.append(send_buf)

        # Fill with random data (only the portion we'll actually send)
        actual_send_size = int(per_rank_send_bytes[target_rank].item())
        if actual_send_size > 0:
            rand_data = torch.randint(0, 256, (actual_send_size,), dtype=torch.uint8, device=device)
            send_buf[:actual_send_size].copy_(rand_data)

    # Create symmetric recv buffers - world_size buffers, one for each sender
    nvshmem_recv_bufs = []
    for sender_rank in range(pgi.world_size):
        # Find max recv size across all ranks from this sender
        max_recv_size = 0
        for receiver_rank in range(pgi.world_size):
            recv_size = int(per_rank_recv_bytes_all[receiver_rank][sender_rank].item())
            max_recv_size = max(max_recv_size, recv_size)

        # All ranks create buffer with same size -> symmetric memory
        recv_buf = nvshmem.tensor((max(max_recv_size, 1),), dtype=torch.uint8)
        nvshmem_recv_bufs.append(recv_buf)

    # Dense tensor for comparison
    a2a_shape = (
        pgi.world_size,
        num_local_experts,
        moe.max_num_tokens * hidden_dim_bytes_with_scale,
    )
    dense_a2a_tensor = torch.empty(a2a_shape, dtype=torch.uint8, device=device)
    dense_a2a_out_tensor = torch.empty_like(dense_a2a_tensor)

    # Compute stats
    dense_a2a_bytes = dense_a2a_tensor.numel() * dense_a2a_tensor.element_size()
    sparse_a2a_bytes = total_sparse_bytes

    # Create CUDA stream for NVSHMEM operations
    cuda_dev = Device(device.index)
    cuda_stream = cuda_dev.create_stream()

    # ============ Validation: Test NVSHMEM PUT correctness ============
    if pgi.rank == 0:
        print("\n[Validation] Testing NVSHMEM PUT correctness...")

    # Fill send buffers with a deterministic pattern for validation
    for target_rank in range(pgi.world_size):
        send_size = int(per_rank_send_bytes[target_rank].item())
        if send_size > 0:
            # Use a unique pattern: (sender_rank * world_size + target_rank) % 256
            pattern = (pgi.rank * pgi.world_size + target_rank) % 256
            nvshmem_send_bufs[target_rank][:send_size].fill_(pattern)

    # Clear recv buffers with a different pattern (255)
    for buf in nvshmem_recv_bufs:
        buf.fill_(255)

    # Execute one round of PUT operations
    torch_stream_ = torch.cuda.current_stream()
    torch_stream_wrapped = PyTorchStreamWrapper(torch_stream_)
    team = Teams.TEAM_WORLD
    nvshmem.collective.barrier(team, torch_stream_wrapped)

    for target_rank in range(pgi.world_size):
        if target_rank == pgi.rank:
            continue
        send_bytes = int(per_rank_send_bytes[target_rank].item())
        if send_bytes > 0:
            nvshmem.put(
                dst=nvshmem_recv_bufs[pgi.rank],
                src=nvshmem_send_bufs[target_rank],
                remote_pe=target_rank,
                stream=cuda_stream
            )

    cuda_stream.sync()
    nvshmem.collective.barrier(team, torch_stream_wrapped)

    # Verify received data
    validation_passed = True
    for sender_rank in range(pgi.world_size):
        if sender_rank == pgi.rank:
            continue
        recv_size = int(per_rank_recv_bytes[sender_rank].item())
        if recv_size > 0:
            # Expected pattern: sender sent (sender * world_size + pgi.rank) % 256
            expected = (sender_rank * pgi.world_size + pgi.rank) % 256
            actual = nvshmem_recv_bufs[sender_rank][0].item()
            if actual != expected:
                print(f"[Rank {pgi.rank}] FAILED: recv[{sender_rank}][0]={actual}, expected={expected}")
                validation_passed = False

    # Gather validation results from all ranks
    validation_result = torch.tensor([1 if validation_passed else 0], dtype=torch.int32, device=device)
    validation_results = [torch.zeros(1, dtype=torch.int32, device=device)
                          for _ in range(pgi.world_size)]
    torch.distributed.all_gather(validation_results, validation_result)

    if pgi.rank == 0:
        if all(r.item() == 1 for r in validation_results):
            print("[Validation] ✓ All ranks PASSED\n")
        else:
            failed_ranks = [i for i, r in enumerate(validation_results) if r.item() == 0]
            print(f"[Validation] ✗ FAILED on ranks: {failed_ranks}\n")

    # Restore random data in send buffers for benchmarking
    for target_rank in range(pgi.world_size):
        send_size = int(per_rank_send_bytes[target_rank].item())
        if send_size > 0:
            rand_data = torch.randint(0, 256, (send_size,), dtype=torch.uint8, device=device)
            nvshmem_send_bufs[target_rank][:send_size].copy_(rand_data)

    # Clear recv buffers
    for buf in nvshmem_recv_bufs:
        buf.zero_()

    nvshmem.collective.barrier(team, torch_stream_wrapped)
    # ============ End of Validation ============

    # Benchmark launcher
    def run() -> tuple[float, ...]:
        num_samples = 10
        events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for _ in range(num_samples)
        ]

        torch_stream_ = torch.cuda.current_stream()
        torch_stream_wrapped = PyTorchStreamWrapper(torch_stream_)
        team = Teams.TEAM_WORLD

        for e0, e1, e2, e3 in events:
            # Use NVSHMEM collective barrier
            nvshmem.collective.barrier(team, torch_stream_wrapped)

            e0.record(torch_stream_)

            # Torch Sparse All-to-All using all_to_all_single with split_sizes
            torch.distributed.all_to_all_single(
                output=sparse_recvbuf,
                input=sparse_sendbuf,
                output_split_sizes=sparse_out_splits,
                input_split_sizes=sparse_in_splits,
            )

            e1.record(torch_stream_)

            # NVSHMEM Sparse: Use PUT with correct API signature
            # put(dst, src, remote_pe, stream)
            # Logic: Put my send_buf[target_rank] to target_rank's recv_bufs[pgi.rank]
            for target_rank in range(pgi.world_size):
                if target_rank == pgi.rank:
                    continue

                send_bytes = int(per_rank_send_bytes[target_rank].item())
                if send_bytes > 0:
                    # PUT: Send local data to remote PE's symmetric buffer
                    # nvshmem_recv_bufs[pgi.rank] is the symmetric buffer on target_rank
                    # that is designated to receive data from pgi.rank

                    # Use full tensors, not slices
                    # If send_bytes < buffer size, we only copy the valid portion
                    nvshmem.put(
                        dst=nvshmem_recv_bufs[pgi.rank],  # Symmetric buffer for data from pgi.rank
                        src=nvshmem_send_bufs[target_rank],  # Local send buffer for target_rank
                        remote_pe=target_rank,
                        stream=cuda_stream
                    )
            
            # Wait for all PUTs to complete
            cuda_stream.sync()  # Use sync() not synchronize()
            nvshmem.collective.barrier(team, torch_stream_wrapped)

            e2.record(torch_stream_)

            # Dense All-to-All (baseline)
            torch.distributed.all_to_all_single(dense_a2a_out_tensor, dense_a2a_tensor)

            e3.record(torch_stream_)

        # Get latency
        torch_stream_.synchronize()
        sum_torch_sparse_us = 0.0
        sum_nvshmem_sparse_us = 0.0
        sum_dense_us = 0.0
        
        for e0, e1, e2, e3 in events:
            sum_torch_sparse_us += e0.elapsed_time(e1) * 1e3
            sum_nvshmem_sparse_us += e1.elapsed_time(e2) * 1e3
            sum_dense_us += e2.elapsed_time(e3) * 1e3
        
        torch_sparse_us = sum_torch_sparse_us / num_samples
        nvshmem_sparse_us = sum_nvshmem_sparse_us / num_samples
        dense_us = sum_dense_us / num_samples
        
        torch_sparse_gbps = sparse_a2a_bytes / torch_sparse_us / 1e3
        nvshmem_sparse_gbps = sparse_a2a_bytes / nvshmem_sparse_us / 1e3
        dense_gbps = dense_a2a_bytes / dense_us / 1e3
        
        return (
            torch_sparse_us,
            nvshmem_sparse_us,
            dense_us,
            torch_sparse_gbps,
            nvshmem_sparse_gbps,
            dense_gbps,
        )

    # Warmup
    num_warmup = 10
    with torch.cuda.nvtx.range("warmup"):
        for _ in range(num_warmup):
            run()

    # Benchmark
    torch.distributed.barrier()
    num_repeat = 20
    with torch.cuda.nvtx.range("bench"):
        result = torch.tensor([run() for _ in range(num_repeat)])

    # Save results before cleanup
    ret_data = (
        (sparse_a2a_bytes, dense_a2a_bytes),
        result,
    )

    # Cleanup NVSHMEM buffers BEFORE returning
    # Delete the run function first to release closures
    del run

    # Then delete NVSHMEM objects
    del nvshmem_send_bufs
    del nvshmem_recv_bufs

    # Force garbage collection and synchronization
    import gc
    gc.collect()
    torch.cuda.synchronize()

    return ret_data


def _worker_bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
) -> None:
    num_ranks = pgi.world_size
    global_rank = pgi.rank
    local_rank = pgi.local_rank

    dev = Device(local_rank)
    dev.set_current()

    nvshmem_init(
        global_rank=global_rank, local_rank=local_rank, world_size=num_ranks, device=dev
    )

    in_dtype = getattr(torch, in_dtype_str)
    out_dtype = getattr(torch, out_dtype_str)
    assert isinstance(in_dtype, torch.dtype)
    
    configs = [
        # V2-Lite:  64 Experts, 6 Experts per Token, 2048 Hidden Dim
        MoEConfig(64, 6, 2048, 1, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 4, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 8, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 16, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 32, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 64, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 128, in_dtype, out_dtype),
        # R1: 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    if pgi.rank == 0:
        print("=" * 80)
        print("Sparse All-to-All Comparison: Torch vs NVSHMEM")
        print("NOTE: NVSHMEM does not have variable-size alltoall (alltoallv)")
        print("      Using PUT-based implementation for sparse communication")
        print("=" * 80)
        print()

    header = [
        "E",
        "E/tok",
        "tok",
        "dim",
        "Torch_Sparse_lat",
        "Torch_Sparse_bw",
        "NVSHMEM_Sparse_lat",
        "NVSHMEM_Sparse_bw",
        "Dense_lat",
        "Dense_bw",
        "Sparse_bytes",
        "Dense_bytes",
        "Torch_vs_Dense",
        "NVSHMEM_vs_Dense",
        "NVSHMEM_vs_Torch",
        "Bytes_Reduction",
    ]

    outpath = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sparse_comparison.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")

        line = f"EP={pgi.world_size} DP={pgi.world_size // dp_size}"
        print(line)
        f_out.write(line + "\n")

        line = "\t".join(header)
        print(line)
        f_out.write(line + "\n")

    for config in configs:
        if pgi.world_size > config.num_experts:
            continue
        meta, result = bench_all_to_all(pgi, dp_size, config)
        sparse_a2a_bytes, dense_a2a_bytes = meta
        
        if pgi.rank == 0:
            torch_sparse_lat = result[:, 0].mean()
            nvshmem_sparse_lat = result[:, 1].mean()
            dense_lat = result[:, 2].mean()
            
            torch_vs_dense = dense_lat / torch_sparse_lat if torch_sparse_lat > 0 else 0
            nvshmem_vs_dense = dense_lat / nvshmem_sparse_lat if nvshmem_sparse_lat > 0 else 0
            nvshmem_vs_torch = torch_sparse_lat / nvshmem_sparse_lat if nvshmem_sparse_lat > 0 else 0
            bytes_reduction = dense_a2a_bytes / sparse_a2a_bytes if sparse_a2a_bytes > 0 else 0
            
            row: dict[str, str] = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                "Torch_Sparse_lat": f"{result[:, 0].mean():.1f}μs ± {result[:, 0].std():.1f}",
                "Torch_Sparse_bw": f"{result[:, 3].mean():.2f}GB/s",
                "NVSHMEM_Sparse_lat": f"{result[:, 1].mean():.1f}μs ± {result[:, 1].std():.1f}",
                "NVSHMEM_Sparse_bw": f"{result[:, 4].mean():.2f}GB/s",
                "Dense_lat": f"{result[:, 2].mean():.1f}μs ± {result[:, 2].std():.1f}",
                "Dense_bw": f"{result[:, 5].mean():.2f}GB/s",
                "Sparse_bytes": f"{sparse_a2a_bytes / 1e6:.2f}MB",
                "Dense_bytes": f"{dense_a2a_bytes / 1e6:.2f}MB",
                "Torch_vs_Dense": f"{torch_vs_dense:.2f}x",
                "NVSHMEM_vs_Dense": f"{nvshmem_vs_dense:.2f}x",
                "NVSHMEM_vs_Torch": f"{nvshmem_vs_torch:.2f}x",
                "Bytes_Reduction": f"{bytes_reduction:.2f}x",
            }
            assert list(row.keys()) == header
            line = "\t".join(row[h] for h in header)
            print(line)
            assert f_out is not None
            f_out.write(line + "\n")
            f_out.flush()

    if f_out is not None:
        f_out.close()
        print("Saved to", outpath)

    nvshmem.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--in-dtype",
        choices=["bfloat16", "float16", "float8_e4m3fn"],
        default="float8_e4m3fn",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    args = parser.parse_args()
    dp_size = int(args.dp_size)
    in_dtype = str(args.in_dtype)
    out_dtype = str(args.out_dtype)

    if "MASTER_ADDR" in os.environ:
        parallel_launch_from_env(_worker_bench_all_to_all, dp_size, in_dtype, out_dtype)
    else:
        world_size = torch.cuda.device_count()
        parallel_launch(
            world_size, _worker_bench_all_to_all, dp_size, in_dtype, out_dtype
        )


if __name__ == "__main__":
    main()