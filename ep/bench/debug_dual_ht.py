"""Debug harness: run only the HT internode phase from test_dual_mode on either
a pure-HT buffer (--buffer ht) or the unified dual buffer (--buffer dual),
printing diagnostics instead of asserting on the topk_weights check.

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<R> \
        --master_addr=172.31.77.96 --master_port=12366 \
        debug_dual_ht.py --buffer ht|dual
"""

import argparse
import os

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun, calc_diff
from test_dual_mode import _build_ht_layout, make_unified_buffer


def make_ht_buffer(group, num_ranks, hidden):
    def _align(size, margin=1.2, alignment=128):
        return ((int(size * margin) + alignment - 1) // alignment) * alignment

    hidden_bytes = hidden * 2
    dispatch_cfg = Buffer.get_dispatch_config(num_ranks)
    combine_cfg = Buffer.get_combine_config(num_ranks)
    nvl_bytes = max(
        cfg.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
        for cfg in (dispatch_cfg, combine_cfg)
    )
    rdma_bytes = _align(
        max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
            for cfg in (dispatch_cfg, combine_cfg)
        )
    )
    return Buffer(
        group,
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=False,
        num_qps_per_rank=24,
        explicitly_destroy=True,
    )


def main(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 256

    if args.buffer == "dual":
        buffer = make_unified_buffer(group, num_ranks, num_tokens, hidden, num_experts)
    else:
        buffer = make_ht_buffer(group, num_ranks, hidden)
    dist.barrier(group)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )

    (
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        gbl_num_tokens_per_rank,
        gbl_num_tokens_per_expert,
        num_tokens_per_rdma_rank,
    ) = _build_ht_layout(
        topk_idx.clone(), num_tokens, num_ranks, num_experts, group, num_local_ranks
    )

    config = Buffer.get_dispatch_config(num_ranks)
    dispatch_kwargs = {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
    }
    recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
        **dispatch_kwargs
    )

    combined_x, combined_topk_weights, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config,
    )
    counts = is_token_in_rank.sum(dim=1).unsqueeze(1)
    check_x = combined_x.float() / counts
    x_diff = calc_diff(check_x, x)
    w_div = combined_topk_weights / counts
    w_diff_div = calc_diff(w_div, topk_weights)
    w_diff_raw = calc_diff(combined_topk_weights, topk_weights)

    ratio = (
        (combined_topk_weights / topk_weights.clamp_min(1e-9))[:4, :4]
        if rank > 0
        else torch.zeros(4, 4)
    )
    print(
        f"[dbg rank {rank}] x_diff={x_diff:.3e} w_diff_div={w_diff_div:.3e} "
        f"w_diff_raw={w_diff_raw:.3e} counts[:4]={counts[:4].flatten().tolist()} "
        f"combined/expected[:4,:4]={ratio.tolist()}",
        flush=True,
    )

    dist.barrier(group)
    buffer.destroy()
    dist.barrier(group)
    if rank == 0:
        print(f"[dbg] DONE buffer={args.buffer}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", choices=["ht", "dual"], default="ht")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    main(local_rank, num_local_ranks, args)
