"""
This is the same test_low_latency.py test in DeepEP's repo.
On first node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=1 --num-experts=28

On second node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=1 --num-experts=28
"""

import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

from buffer import Buffer
from utils import (
    init_dist,
    bench,
    bench_kineto,
    calc_diff,
    hash_tensor,
    per_token_cast_back,
    get_cpu_proxies_meta,
    initialize_uccl,
    destroy_uccl,
    get_peer_ip,
    detect_ib_hca,
)

# UCCL import
try:
    from uccl import ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def peek_slot_from_handle(packed_recv_x, handle, le, src_rank, n_words=4):
    rl = handle[1][le]  # recv_layout_range for that expert
    int_mask = (1 << 32) - 1
    begin = int((rl[src_rank] >> 32).item())
    cnt = int((rl[src_rank] & int_mask).item())
    if cnt == 0:
        print(f"[peek] le={le} src={src_rank} has no tokens")
        return
    slot = begin  # first filled slot for this src_rank

    elt_size = packed_recv_x.element_size()
    hidden = packed_recv_x.size(-1)
    slots_per_expert = packed_recv_x.size(1)
    byte_off = ((le * slots_per_expert + slot) * hidden) * elt_size

    nbytes_total = packed_recv_x.numel() * elt_size
    u8 = torch.cuda.ByteTensor().set_(
        packed_recv_x.untyped_storage(), 0, (nbytes_total,), (1,)
    )
    torch.cuda.synchronize()
    chunk = u8[byte_off : byte_off + n_words * 4].cpu().numpy().view("<u4")
    dev_addr = packed_recv_x.data_ptr() + byte_off
    print(
        f"[host peek] le={le} src={src_rank} slot={slot} @ {hex(dev_addr)} " f"words=",
        [hex(int(x)) for x in chunk],
    )


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    use_logfmt: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(
            torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            * 0.5
            * random.random()
        )
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    )

    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = (
            -1
        )

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False,):
                    for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                        for use_ue8m0 in (False, True) if round_scale else (False,):
                            print(
                                "Start experiment with settings:"
                                f" return_recv_hook={return_recv_hook}"
                                f" dispatch_use_fp8={dispatch_use_fp8}"
                                f" round_scale={round_scale}"
                                f" use_ue8m0={use_ue8m0}",
                                flush=True,
                            )
                            num_times += 1
                            for i in range((num_times % 2) + 1):
                                cumulative_local_expert_recv_stats = torch.zeros(
                                    (num_local_experts,), dtype=torch.int, device="cuda"
                                )
                                (
                                    packed_recv_x,
                                    packed_recv_count,
                                    handle,
                                    event,
                                    hook,
                                ) = buffer.low_latency_dispatch(
                                    current_x,
                                    topk_idx,
                                    num_tokens,
                                    num_experts,
                                    use_fp8=dispatch_use_fp8,
                                    round_scale=round_scale,
                                    use_ue8m0=use_ue8m0,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook,
                                    return_recv_hook=return_recv_hook,
                                )
                                (
                                    hook()
                                    if return_recv_hook
                                    else event.current_stream_wait()
                                )
                            packed_recv_x = (
                                (packed_recv_x[0], packed_recv_x[1].contiguous())
                                if dispatch_use_fp8
                                else packed_recv_x
                            )
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, hidden // 128),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8
                                else packed_recv_x.clone()
                            )
                            all_topk_idx = torch.empty(
                                (num_ranks, num_tokens, num_topk),
                                dtype=topk_idx.dtype,
                                device="cuda",
                            )
                            dist.all_gather_into_tensor(
                                all_topk_idx, topk_idx, group=group
                            )
                            for i in range(num_local_experts if do_check else 0):
                                expert_id = rank * num_local_experts + i
                                recv_x = (
                                    per_token_cast_back(
                                        packed_recv_x[0][i], packed_recv_x[1][i]
                                    )
                                    if dispatch_use_fp8
                                    else packed_recv_x[i]
                                )
                                recv_count, recv_src_info, recv_layout_range = (
                                    packed_recv_count[i],
                                    handle[0][i],
                                    handle[1][i],
                                )

                                # Check expert indices
                                int_mask = (2**32) - 1
                                num_valid_tokens = recv_count.item()
                                assert (
                                    cumulative_local_expert_recv_stats[i].item()
                                    == num_valid_tokens
                                ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                                assert (
                                    num_valid_tokens
                                    == (recv_layout_range & int_mask).sum().item()
                                ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                                assert (
                                    num_valid_tokens
                                    == (all_topk_idx == expert_id).sum().item()
                                ), f"{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}"
                                if num_valid_tokens == 0:
                                    continue
                                # Check received data
                                if current_x is x:
                                    recv_x = recv_x[:num_valid_tokens]
                                    recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                    recv_src_info = recv_src_info[:num_valid_tokens]
                                    assert torch.equal(
                                        recv_x_amin, recv_x[:, :-128].amax(dim=-1)
                                    )
                                    if round_scale:
                                        assert (
                                            calc_diff(
                                                recv_x[:, -1], recv_src_info.view(-1)
                                            )
                                            < 0.007
                                        )
                                    else:
                                        assert (
                                            recv_x[:, -128:]
                                            - recv_src_info.view(-1, 1) % num_tokens
                                        ).sum().item() == 0
                                    for j in range(num_ranks):
                                        begin_idx, count = (
                                            recv_layout_range[j] >> 32
                                        ).item(), (
                                            recv_layout_range[j] & int_mask
                                        ).item()
                                        if not round_scale:
                                            assert (
                                                recv_x_amin == j - rank_offset
                                            ).sum().item() == (
                                                all_topk_idx[j] == expert_id
                                            ).sum().item()
                                            assert (
                                                recv_x[
                                                    begin_idx : begin_idx + count, :-128
                                                ]
                                                - j
                                                + rank_offset
                                            ).sum().item() == 0
                                if dispatch_use_fp8:
                                    hash_value ^= hash_tensor(
                                        packed_recv_x[0][i, :num_valid_tokens]
                                    )
                                    hash_value ^= hash_tensor(
                                        packed_recv_x[1][i, :num_valid_tokens]
                                    )
                                else:
                                    hash_value ^= hash_tensor(
                                        packed_recv_x[i, :num_valid_tokens]
                                    )
                            # Check combine correctness
                            for zero_copy in (False,) if use_logfmt else (False, True):
                                if zero_copy:
                                    buffer.get_next_low_latency_combine_buffer(handle)[
                                        :, :, :
                                    ] = simulated_gemm_x
                                out = torch.empty(
                                    (num_tokens, hidden),
                                    dtype=torch.bfloat16,
                                    device="cuda",
                                )
                                combined_x, event, hook = buffer.low_latency_combine(
                                    simulated_gemm_x,
                                    topk_idx,
                                    topk_weights,
                                    handle,
                                    use_logfmt=use_logfmt,
                                    async_finish=not return_recv_hook,
                                    zero_copy=zero_copy,
                                    return_recv_hook=return_recv_hook,
                                    out=out,
                                )
                                (
                                    hook()
                                    if return_recv_hook
                                    else event.current_stream_wait()
                                )
                                if do_check:
                                    diff = calc_diff(
                                        current_x
                                        * topk_weights.masked_fill(topk_idx == -1, 0)
                                        .sum(dim=1)
                                        .view(-1, 1),
                                        combined_x,
                                    )
                                    assert torch.isnan(combined_x).sum().item() == 0
                                    assert diff < (
                                        9e-4 if dispatch_use_fp8 else 1e-5
                                    ), f"Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}"
                                    hash_value ^= hash_tensor(combined_x)

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            current_x,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=return_recv_hook,
        )
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            use_logfmt=use_logfmt,
            return_recv_hook=return_recv_hook,
        )
        large_gemm_with_hook(hook) if return_recv_hook else None

    print("✓ All correctness tests passed!", flush=True)
    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (
            num_logfmt10_bytes if use_logfmt else num_bf16_bytes
        ) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e6 / avg_t:.2f} MB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
        flush=True,
    )
    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(
            partial(test_func, return_recv_hook=return_recv_hook),
            kernel_names=("dispatch", "combine"),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            num_kernels_per_period=2 if return_recv_hook else 1,
        )
        if not return_recv_hook:
            print(
                f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e6 / dispatch_t:.2f} MB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                f"Combine bandwidth: {num_combine_comm_bytes / 1e6 / combine_t:.2f} MB/s, avg_t={combine_t * 1e6:.2f} us",
                flush=True,
            )
        else:
            print(
                f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                flush=True,
            )
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )

    # UCCL new code for initialization
    device_index = int(os.environ["LOCAL_RANK"])
    scratch = torch.zeros(
        num_rdma_bytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    proxies, workers = initialize_uccl(scratch, num_rdma_bytes, rank, num_ranks, group)

    if local_rank == 0:
        print(f"Allocating buffer size: {num_rdma_bytes / 1e6} MB ...", flush=True)
    buffer = Buffer(
        group,
        rdma_buffer_ptr=scratch.data_ptr(),
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
        explicitly_destroy=True,
        allow_mnnvl=args.allow_mnnvl,
    )

    buffer.connect_atomic_buffer(proxies[0])

    for proxy in proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            num_tokens, hidden, num_experts
        )
        proxy.set_atomic_buffer_ptr(proxies[0].get_atomic_buffer_ptr())

    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        group,
        buffer,
        use_logfmt=args.use_logfmt,
        seed=1,
    )

    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        ref_hash = test_main(
            num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            use_logfmt=args.use_logfmt,
            seed=seed,
        )
        for i in range(20):
            assert (
                test_main(
                    num_tokens,
                    hidden,
                    num_experts,
                    num_topk,
                    rank,
                    num_ranks,
                    group,
                    buffer,
                    use_logfmt=args.use_logfmt,
                    seed=seed,
                )
                == ref_hash
            ), f"Error: seed={seed}"

    # Destroy the buffer runtime and communication group
    group.barrier()
    buffer.destroy()
    dist.barrier()
    destroy_uccl(proxies, workers)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`

    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):  # Mellanox IB devices show up like mlx5_0
        os.environ["NCCL_IB_HCA"] = ib_dev
        print(f"Set NCCL_IB_HCA={ib_dev}")
    else:
        print(f"Skipping NCCL_IB_HCA export (detected {ib_dev})")

    parser = argparse.ArgumentParser(description="Test low-latency EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=128, help="Number of tokens (default: 128)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=288, help="Number of experts (default: 288)"
    )
    parser.add_argument(
        "--allow-mnnvl", action="store_true", help="Allow MNNVL for communication"
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_true",
        help="Whether to disable NVLink for testing",
    )
    parser.add_argument(
        "--use-logfmt", action="store_true", help="Whether to test LogFMT combine"
    )
    parser.add_argument(
        "--pressure-test", action="store_true", help="Whether to do pressure test"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    # NOTE: modified from deep_ep
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    test_loop(local_rank, num_local_ranks, args)
