# ruff: noqa: T201

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch

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

    # Calculate total sparse bytes
    total_sparse_bytes = int(per_rank_send_bytes.sum().item())

    # Build sparse sendbuf using all_to_all_single with split_sizes (like reference implementation)
    # Calculate which tokens go to which rank
    topk_idx = rank_data.indices  # [num_tokens, experts_per_token]
    dest_rank_per_routing = topk_idx // num_local_experts  # which rank each expert is on
    local_expert_per_routing = topk_idx % num_local_experts
    
    # Build contiguous sendbuf by concatenating data for each destination rank
    sparse_in_splits = per_rank_send_bytes.tolist()
    sparse_out_splits = per_rank_recv_bytes.tolist()
    
    total_send_bytes = int(per_rank_send_bytes.sum().item())
    sparse_sendbuf = torch.empty(max(total_send_bytes, 1), dtype=torch.uint8, device=device)
    
    # Fill sendbuf: for each dest rank, pack all tokens going to that rank
    offset = 0
    for dest in range(pgi.world_size):
        dest_bytes = int(per_rank_send_bytes[dest].item())
        if dest_bytes > 0:
            # Find all (token, expert) pairs going to this destination
            mask = (dest_rank_per_routing == dest)  # [num_tokens, experts_per_token]
            num_routings = mask.sum().item()
            
            # Create dummy data for this destination
            # In real implementation, this would be actual token embeddings
            # Use torch.randint for uint8 data
            chunk = torch.randint(0, 256, (dest_bytes,), dtype=torch.uint8, device=device)
            sparse_sendbuf[offset:offset + dest_bytes] = chunk
            offset += dest_bytes
    
    # Allocate contiguous recv buffer
    total_recv_bytes = int(per_rank_recv_bytes.sum().item())
    sparse_recvbuf = torch.empty(max(total_recv_bytes, 1), dtype=torch.uint8, device=device)

    # For comparison: old all_to_all with list of tensors
    sparse_send_tensors = []
    sparse_recv_tensors = []
    for rank in range(pgi.world_size):
        send_size = int(per_rank_send_bytes[rank].item())
        recv_size = int(per_rank_recv_bytes[rank].item())
        sparse_send_tensors.append(
            torch.empty(max(send_size, 1), dtype=torch.uint8, device=device)
        )
        sparse_recv_tensors.append(
            torch.empty(max(recv_size, 1), dtype=torch.uint8, device=device)
        )

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

    # Benchmark launcher
    def run() -> tuple[float, ...]:
        num_samples = 10
        events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for _ in range(num_samples)
        ]

        torch_stream_ = torch.cuda.current_stream()

        for e0, e1, e2, e3 in events:
            torch.distributed.barrier()

            e0.record(torch_stream_)

            # Sparse All-to-All using all_to_all_single with split_sizes (optimized)
            torch.distributed.all_to_all_single(
                output=sparse_recvbuf,
                input=sparse_sendbuf,
                output_split_sizes=sparse_out_splits,
                input_split_sizes=sparse_in_splits,
            )

            e1.record(torch_stream_)

            # Sparse All-to-All using all_to_all with list (original method for comparison)
            torch.distributed.all_to_all(sparse_recv_tensors, sparse_send_tensors)

            e2.record(torch_stream_)

            # Dense All-to-All using all_to_all_single (baseline)
            torch.distributed.all_to_all_single(dense_a2a_out_tensor, dense_a2a_tensor)

            e3.record(torch_stream_)

        # Get latency
        torch_stream_.synchronize()
        sum_sparse_opt_us = 0.0
        sum_sparse_list_us = 0.0
        sum_dense_a2a_us = 0.0
        for e0, e1, e2, e3 in events:
            sum_sparse_opt_us += e0.elapsed_time(e1) * 1e3
            sum_sparse_list_us += e1.elapsed_time(e2) * 1e3
            sum_dense_a2a_us += e2.elapsed_time(e3) * 1e3
        sparse_opt_us = sum_sparse_opt_us / num_samples
        sparse_list_us = sum_sparse_list_us / num_samples
        dense_a2a_us = sum_dense_a2a_us / num_samples
        
        sparse_opt_gbps = sparse_a2a_bytes / sparse_opt_us / 1e3
        sparse_list_gbps = sparse_a2a_bytes / sparse_list_us / 1e3
        dense_a2a_gbps = dense_a2a_bytes / dense_a2a_us / 1e3
        return (
            sparse_opt_us,
            sparse_list_us,
            dense_a2a_us,
            sparse_opt_gbps,
            sparse_list_gbps,
            dense_a2a_gbps,
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

    return (
        (sparse_a2a_bytes, dense_a2a_bytes),
        result,
    )


def _worker_bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
) -> None:
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
        # R1     : 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    header = [
        "E",
        "E/tok",
        "tok",
        "dim",
        "Sparse_Opt_lat",
        "Sparse_Opt_bw",
        "Sparse_List_lat",
        "Sparse_List_bw",
        "Sparse_bytes",
        "Dense_lat",
        "Dense_bw",
        "Dense_bytes",
        "Opt_vs_Dense",
        "List_vs_Dense",
        "Bytes_Reduction",
    ]

    outpath = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_all_to_all_only.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")

        line = f"EP={pgi.world_size} DP={pgi.world_size // dp_size} (Sparse: Optimized vs List vs Dense)"
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
            # Calculate metrics
            sparse_opt_lat = result[:, 0].mean()
            sparse_list_lat = result[:, 1].mean()
            dense_lat = result[:, 2].mean()
            
            opt_vs_dense = dense_lat / sparse_opt_lat if sparse_opt_lat > 0 else 0
            list_vs_dense = dense_lat / sparse_list_lat if sparse_list_lat > 0 else 0
            bytes_reduction = dense_a2a_bytes / sparse_a2a_bytes if sparse_a2a_bytes > 0 else 0
            
            row: dict[str, str] = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                "Sparse_Opt_lat": f"{result[:, 0].mean():.1f}μs ± {result[:, 0].std():.1f}",
                "Sparse_Opt_bw": f"{result[:, 3].mean():.2f}GB/s",
                "Sparse_List_lat": f"{result[:, 1].mean():.1f}μs ± {result[:, 1].std():.1f}",
                "Sparse_List_bw": f"{result[:, 4].mean():.2f}GB/s",
                "Sparse_bytes": f"{sparse_a2a_bytes:,}",
                "Dense_lat": f"{result[:, 2].mean():.1f}μs ± {result[:, 2].std():.1f}",
                "Dense_bw": f"{result[:, 5].mean():.2f}GB/s",
                "Dense_bytes": f"{dense_a2a_bytes:,}",
                "Opt_vs_Dense": f"{opt_vs_dense:.2f}x",
                "List_vs_Dense": f"{list_vs_dense:.2f}x",
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