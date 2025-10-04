"""
This is the same test_internode.py test in DeepEP's repo.

On first node:
export OMP_NUM_THREADS=4
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.1.227.34 --master_port=12355 \
  bench/test_internode.py --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=256 --test-ll-compatibility

On second node:
export OMP_NUM_THREADS=4
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr=10.1.227.34 --master_port=12355 \
  bench/test_internode.py --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=256 --test-ll-compatibility

This benchmark verifies:
  * Dispatch and combine correctness for BF16/FP8
  * Top-k routing and per-expert token distribution
  * Compatibility with cached dispatch and low-latency kernels
  * Performance tuning for NVL and RDMA chunk sizes
  
It is currently still under development. 
"""

import argparse
import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences

from utils import (
    init_dist,
    bench,
    bench_kineto,
    calc_diff,
    create_grouped_scores,
    inplace_unique,
    per_token_cast_to_fp8,
    per_token_cast_back,
    initialize_uccl,
    destroy_uccl,
    init_dist_under_torchrun,
    detect_ib_hca,
)

# Test compatibility with low latency functions
import test_low_latency
from buffer import Buffer

try:
    from uccl.ep import Config
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise


# Paste inside test_main, after you compute rdma_rank_idx / inplace_unique
def channel_token_range(num_tokens, num_channels, channel_id):
    # matches typical get_channel_task_range partitioning
    start = (num_tokens * channel_id) // num_channels
    end = (num_tokens * (channel_id + 1)) // num_channels
    return start, end


def expected_per_dst_for_channel(
    rdma_rank_idx, num_nodes, num_tokens, num_channels, channel_id
):
    start, end = channel_token_range(num_tokens, num_channels, channel_id)
    # Does token t go to dst d?
    # rdma_rank_idx[t] holds up to num_topk entries in 0..num_nodes-1 or -1
    token_goes_to = [
        (rdma_rank_idx == d).any(dim=1) for d in range(num_nodes)
    ]  # list of [num_tokens] bool
    per_dst_counts = [int(mask[start:end].sum().item()) for mask in token_goes_to]
    return per_dst_counts  # length = num_nodes


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: Buffer,
    group: dist.ProcessGroup,
):
    # Setting
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = (
        args.num_topk_groups,
        args.num_topk,
        args.num_experts,
    )

    # assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}",
            flush=True,
        )

    # Deterministic toy input: no randomness, uniform routing

    # Input activations
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = x.clone()  # keep shape compatibility with original code

    # Build uniform topk_idx: token t → expert e = (t % num_experts)
    topk_idx = torch.full((num_tokens, num_topk), -1, dtype=torch.long, device="cuda")
    for t in range(num_tokens):
        for k in range(num_topk):
            e = (t * num_topk + k) % num_experts
            topk_idx[t, k] = e

    # Uniform weights: each token has weight=1.0
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )

    # Also define a "pure random" placeholder if you want to keep compatibility
    topk_weights_pure_rand = topk_weights.clone()

    # Also define a "pure random" placeholder if you want to keep compatibility
    topk_weights_pure_rand = topk_weights.clone()
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)

    if rank == 0:
        num_channels = num_sms // 2
        for ch in range(num_channels):
            counts = expected_per_dst_for_channel(
                rdma_rank_idx, num_nodes, num_tokens, num_channels, ch
            )
            print(f"[host] channel {ch}: per-dst RDMA counts = {counts}", flush=True)

    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    previous_mode = False
    async_mode = False
    current_x = x_pure_rand
    with_topk = False

    if local_rank == 0:
        print(
            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
            flush=True,
            end="",
        )
    dispatch_args = {
        "x": current_x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "async_finish": async_mode,
    }
    if with_topk:
        dispatch_args.update(
            {
                "topk_idx": topk_idx,
                "topk_weights": (
                    topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                ),
            }
        )
    if previous_mode:
        dispatch_args.update({"previous_event": buffer.capture()})
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(**dispatch_args)
    event.current_stream_wait() if async_mode else ()
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

    # Checks
    recv_gbl_rank_prefix_sum = handle[-4]
    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
        0
    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
    assert (
        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        == recv_num_tokens_per_expert_list
    )
    if current_x is not x_pure_rand:
        check_data(recv_x, recv_gbl_rank_prefix_sum)
    if with_topk:
        # Check `topk_idx`
        assert (
            recv_topk_idx.eq(-1)
            | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
        ).sum().item() == recv_topk_idx.numel()
        for i, count in enumerate(recv_num_tokens_per_expert_list):
            assert recv_topk_idx.eq(i).sum().item() == count

        # Check `topk_weights`
        if current_x is not x_pure_rand:
            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                dim=1, keepdim=True
            ).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

    # NOTE(MaoZiming): for debug
    print("passed!", flush=True, end=" ")
    print("Before dist.barrier", flush=True)
    dist.barrier()
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(
    local_rank: int, num_local_ranks: int, num_nodes: int, args: argparse.Namespace
):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms = 2
    num_qps_per_rank = max(
        num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0
    )
    num_rdma_bytes = int(2e9)
    num_nvlink_bytes = int(1e9)

    # UCCL new code for initialization
    device_index = int(os.environ["LOCAL_RANK"])
    scratch = torch.zeros(
        num_rdma_bytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    proxies, workers = initialize_uccl(
        scratch, num_rdma_bytes, rank, num_ranks, group, num_experts=args.num_experts
    )

    buffer = Buffer(
        group,
        scratch.data_ptr(),
        num_nvlink_bytes,
        num_rdma_bytes,
        low_latency_mode=args.test_ll_compatibility,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True,
    )
    buffer.connect_atomic_buffer(proxies[0])

    for proxy in proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            args.num_tokens, args.hidden, args.num_experts
        )
        proxy.set_atomic_buffer_ptr(proxies[0].get_atomic_buffer_ptr())

    # assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for i in (num_sms,):
        test_main(
            args,
            i,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
        )
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    group.barrier()
    buffer.destroy()
    dist.barrier()
    destroy_uccl(proxies, workers)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):  # Mellanox IB devices show up like mlx5_0
        os.environ["NCCL_IB_HCA"] = ib_dev
        print(f"Set NCCL_IB_HCA={ib_dev}")
    parser = argparse.ArgumentParser(description="Test internode EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk-groups",
        type=int,
        default=None,
        help="Number of top-k groups (default: `min(num_nodes, 4)`)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256"
    )
    parser.add_argument(
        "--test-ll-compatibility",
        action="store_true",
        help="whether to test compatibility with low-latency kernels",
    )
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    num_nodes = world_size // local_world_size

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes
    if num_processes != 8:
        raise ValueError("Only --num-processes=8 is supported for this test.")
    # NOTE: modified from deep_ep
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    test_loop(local_rank, num_local_ranks, num_nodes, args)
