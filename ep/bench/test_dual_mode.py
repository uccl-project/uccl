"""Functional test: one dual-mode Buffer runs both high-throughput
dispatch/combine and low-latency dispatch/combine on the same proxy thread
pool.

This is the "unified buffer" path: a single Buffer constructed with
``low_latency_mode=True, dual_mode=True`` allocates the NVLink buffer for HT
kernels plus an RDMA buffer sized for the larger of the HT and LL layouts,
then switches with ``clean_low_latency_buffer()`` between phases.

The dual-mode proxy pool serves both encodings: WRITE/ATOMIC commands carry a
mode bit (cmd_type bit[5]) so the proxy decodes offsets/unions and picks QPs
per command, and incoming completions are classified by receiving QP (HT
senders target the per-channel data QPs, LL senders the base QP). Proxy init
creates the superset of resources: all inter-node peers, per-channel data QPs,
and the LocalBarrier shm (``_dual_`` namespace).

Run on a single node (8 ranks, intranode):

    cd ep
    OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 \\
        bench/test_dual_mode.py --num-tokens 128 --hidden 7168 \\
        --num-topk 8 --num-experts 256

Internode (two nodes x 8 ranks). Set ``UCCL_SOCKET_IFNAME`` / ``UCCL_IB_GID_INDEX``
(and matching NCCL vars) before launch:

    # node 0
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=<head_ip> --master_port=12366 \\
        bench/test_dual_mode.py --num-tokens 128 --hidden 7168 \\
        --num-topk 8 --num-experts 256

    # node 1
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \\
        --master_addr=<head_ip> --master_port=12366 \\
        bench/test_dual_mode.py --num-tokens 128 --hidden 7168 \\
        --num-topk 8 --num-experts 256
"""

import argparse
import os
import random

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import (
    init_dist_under_torchrun,
    calc_diff,
    inplace_unique,
)

_SUPPORTED_LL_HIDDEN = (2048, 2560, 4096, 5120, 6144, 7168, 8192)


def make_unified_buffer(
    group: dist.ProcessGroup,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
) -> Buffer:
    """Single dual-mode buffer sized for both HT (NVL + RDMA) and LL (RDMA)."""

    def _align(size: int, margin: float = 1.2, alignment: int = 128) -> int:
        return ((int(size * margin) + alignment - 1) // alignment) * alignment

    hidden_bytes = hidden * 2  # bf16
    dispatch_cfg = Buffer.get_dispatch_config(num_ranks)
    combine_cfg = Buffer.get_combine_config(num_ranks)
    nvl_bytes = max(
        cfg.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
        for cfg in (dispatch_cfg, combine_cfg)
    )
    ht_rdma_bytes = _align(
        max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
            for cfg in (dispatch_cfg, combine_cfg)
        )
    )
    ll_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    return Buffer(
        group,
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=max(ht_rdma_bytes, ll_rdma_bytes),
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        dual_mode=True,
    )


def _build_ht_layout(
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_ranks: int,
    num_experts: int,
    group: dist.ProcessGroup,
    num_local_ranks: int,
):
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
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
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    num_tokens_per_rdma_rank = None
    if num_ranks > num_local_ranks:
        num_nodes = num_ranks // num_local_ranks
        rdma_rank_idx = rank_idx // num_local_ranks
        rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
        inplace_unique(rdma_rank_idx, num_nodes)
        num_tokens_per_rdma_rank = torch.empty(
            (num_nodes,), dtype=torch.int, device="cuda"
        )
        for i in range(num_nodes):
            num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()

    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)
    return (
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        gbl_num_tokens_per_rank,
        gbl_num_tokens_per_expert,
        num_tokens_per_rdma_rank,
    )


def test_high_throughput(
    buffer: Buffer,
    group: dist.ProcessGroup,
    rank: int,
    num_ranks: int,
    num_local_ranks: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    local_rank: int,
) -> None:
    """Minimal HT dispatch + combine correctness on the unified buffer."""
    assert num_experts % num_ranks == 0
    is_internode = buffer.runtime.get_num_rdma_ranks() > 1

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

    if is_internode:
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

    config = Buffer.get_dispatch_config(num_ranks)
    if local_rank == 0:
        mode = "internode" if is_internode else "intranode"
        print(
            f"[unified-buffer/ht/{mode}] dispatch+combine "
            f"tokens={num_tokens} hidden={hidden} experts={num_experts}",
            flush=True,
        )

    dispatch_kwargs = {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
    }
    if is_internode:
        dispatch_kwargs["num_tokens_per_rdma_rank"] = num_tokens_per_rdma_rank

    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, _ = (
        buffer.dispatch(**dispatch_kwargs)
    )

    if is_internode:
        recv_gbl_rank_prefix_sum = handle[-6]
    else:
        rank_prefix_matrix = handle[0]

    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0)
    assert (
        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        == recv_num_tokens_per_expert_list
    )

    if recv_x.size(0) > 0:
        assert torch.allclose(recv_x.amin(dim=1), recv_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            if is_internode:
                check_end = recv_gbl_rank_prefix_sum[i].item()
            else:
                check_end = rank_prefix_matrix[i][rank].item()
            assert (recv_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    combined_x, combined_topk_weights, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config,
    )
    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
    assert calc_diff(check_x, x) < 5e-6
    # Unlike combined_x (summed over the token's rank copies), this fork's
    # combine returns topk_weights deduplicated in both the intranode and
    # internode kernels, so compare raw (verified empirically on a pure-HT
    # buffer: combined == original exactly, x scaled by copy count).
    assert calc_diff(combined_topk_weights, topk_weights) < 1e-9

    if local_rank == 0:
        print("[unified-buffer/ht] PASS", flush=True)


def test_low_latency(
    buffer: Buffer,
    group: dist.ProcessGroup,
    rank: int,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    local_rank: int,
) -> None:
    """Minimal LL dispatch + combine correctness on the same unified buffer."""
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    rank_offset = 128
    assert num_ranks - rank_offset < 257

    torch.manual_seed(42 + rank)
    random.seed(42 + rank)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()
    for _ in range(3):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = (
            -1
        )

    if local_rank == 0:
        print(
            f"[unified-buffer/ll] low_latency_dispatch+combine "
            f"tokens={num_tokens} hidden={hidden} experts={num_experts}",
            flush=True,
        )

    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int, device="cuda"
    )
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        use_fp8=False,
        async_finish=True,
        return_recv_hook=False,
    )
    event.current_stream_wait()

    all_topk_idx = torch.empty(
        (num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device="cuda"
    )
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    int_mask = (2**32) - 1
    for le in range(num_local_experts):
        expert_id = rank * num_local_experts + le
        recv_x = packed_recv_x[le]
        recv_count = packed_recv_count[le]
        recv_layout_range = handle[1][le]
        num_valid_tokens = recv_count.item()
        assert cumulative_local_expert_recv_stats[le].item() == num_valid_tokens
        assert num_valid_tokens == (recv_layout_range & int_mask).sum().item()
        assert num_valid_tokens == (all_topk_idx == expert_id).sum().item()
        if num_valid_tokens == 0:
            continue
        recv_x = recv_x[:num_valid_tokens]
        recv_x_amin = recv_x[:, :-128].amin(dim=-1)
        assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
        recv_src_info = handle[0][le][:num_valid_tokens]
        assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
        for src_rank in range(num_ranks):
            begin_idx = (recv_layout_range[src_rank] >> 32).item()
            count = (recv_layout_range[src_rank] & int_mask).item()
            assert (
                recv_x_amin[begin_idx : begin_idx + count] == src_rank - rank_offset
            ).sum().item() == (all_topk_idx[src_rank] == expert_id).sum().item()

    simulated_gemm_x = packed_recv_x.clone()
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    combined_x, combine_event, combine_hook = buffer.low_latency_combine(
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=True,
        zero_copy=False,
        return_recv_hook=False,
        out=out,
    )
    combine_event.current_stream_wait()

    expected = x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
    diff = calc_diff(expected, combined_x)
    assert torch.isnan(combined_x).sum().item() == 0
    assert diff < 1e-5, f"combine diff too large: {diff}"

    if local_rank == 0:
        print("[unified-buffer/ll] PASS", flush=True)


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts or max(num_ranks * 2, 16)
    if num_experts % num_ranks:
        num_experts = ((num_experts // num_ranks) + 1) * num_ranks

    assert hidden % 128 == 0, "hidden must be divisible by 128 for low-latency kernels"
    assert hidden in _SUPPORTED_LL_HIDDEN, (
        f"hidden={hidden} is not supported by low-latency kernels; "
        f"choose one of {_SUPPORTED_LL_HIDDEN}"
    )

    if rank == 0:
        from uccl import ep as uccl_ep

        num_nodes = num_ranks // num_local_ranks
        print(
            f"[unified-buffer] starting: world={num_ranks} nodes={num_nodes} "
            f"tokens={num_tokens} hidden={hidden} topk={num_topk} "
            f"experts={num_experts} proxy_threads={uccl_ep.get_num_proxy_threads()}",
            flush=True,
        )

    buffer = make_unified_buffer(group, num_ranks, num_tokens, hidden, num_experts)
    dist.barrier(group)
    if rank == 0:
        print(
            f"[unified-buffer] created single buffer "
            f"(low_latency_mode=True, nvl={buffer.num_nvl_bytes}, "
            f"rdma={buffer.num_rdma_bytes})",
            flush=True,
        )

    # Alternate prefill-style HT and decode-style LL phases on the same
    # buffer + dual proxy pool, the way an application toggling on sequence
    # length would. HT kernels dirty the LL RDMA layout and the shared atomic
    # buffer; clean before each LL phase.
    num_rounds = 2
    for round_idx in range(num_rounds):
        test_high_throughput(
            buffer,
            group,
            rank,
            num_ranks,
            num_local_ranks,
            num_tokens,
            hidden,
            num_topk,
            num_experts,
            local_rank,
        )
        dist.barrier(group)

        if rank == 0:
            print(
                f"[unified-buffer] round {round_idx}: clean_low_latency_buffer() "
                "before LL phase",
                flush=True,
            )
        buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)
        torch.cuda.synchronize()
        dist.barrier(group)

        test_low_latency(
            buffer,
            group,
            rank,
            num_ranks,
            num_tokens,
            hidden,
            num_topk,
            num_experts,
            local_rank,
        )
        dist.barrier(group)

    buffer.destroy()
    dist.barrier(group)
    if rank == 0:
        print(
            f"[unified-buffer] PASS ({num_rounds}x (HT -> clean -> LL) on one buffer)",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Functional unified-buffer test: HT then LL on one Buffer"
    )
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Must divide world size; default 256",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    main(local_rank, num_local_ranks, args)
