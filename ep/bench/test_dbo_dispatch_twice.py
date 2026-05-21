"""
Minimal repro for the "DBO dispatch-twice" issue:
under DBO/TBO, sglang issues two dispatch() calls on the SAME deep_ep Buffer
(one per micro-batch, with different inputs). This script tries to reproduce
the failure with no sglang dependency, by:

  baseline_A, baseline_B: separate dispatches of two different micro-batches
                          (each run alone with sync) -> ground truth recv tensors

  pattern_A: same two micro-batches dispatched back-to-back on default stream
             (async_finish=True so they queue without intermediate sync)
             -> compare to baselines, exact equality required

  pattern_B: same two micro-batches dispatched on two different CUDA streams,
             overlapped (most TBO-like)
             -> compare to baselines, exact equality required

  pattern_C: full dispatch -> combine -> dispatch -> combine cycle for both
             micro-batches with overlap (TBO Cycle).

If "calling dispatch twice on the buffer" corrupts the buffer state, the
pattern_A/B/C recv tensors will differ from their baselines.

Launch (2 nodes x 8 ranks):
  node 0:  bash scripts/run_dbo_repro.sh 0
  node 1:  bash scripts/run_dbo_repro.sh 1
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    init_dist_under_torchrun,
    inplace_unique,
    create_grouped_scores,
    per_token_cast_to_fp8,
)
from buffer import Buffer
from uccl.ep import Config


def gen_workload(num_tokens, hidden, num_experts, num_topk, num_topk_groups, num_nodes,
                 num_ranks, num_local_ranks, rank, seed):
    """Mirror test_internode.py's data + layout generation."""
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=g)

    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda",
                    generator=g).abs() + 1
    )
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda",
                               generator=g)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

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

    return dict(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
    )


def gen_empty_workload(hidden, num_experts, num_topk, num_nodes, num_ranks):
    """Construct an explicitly-empty workload (num_tokens == 0).

    Mirrors what a DP-attention rank with no local tokens passes into
    buffer.dispatch(): zero-row x / topk tensors (data_ptr() == 0), the
    per-rank / per-expert count arrays zeroed but properly sized, and
    is_token_in_rank shaped (0, num_ranks).
    """
    x = torch.empty((0, hidden), dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.empty((0, num_topk), dtype=torch.int64, device="cuda")
    topk_weights = torch.empty((0, num_topk), dtype=torch.float32, device="cuda")
    num_tokens_per_rank = torch.zeros((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank = torch.zeros((num_nodes,), dtype=torch.int, device="cuda")
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    is_token_in_rank = torch.zeros((0, num_ranks), dtype=torch.bool, device="cuda")
    return dict(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
    )


def to_fp8(wl):
    """Replace wl['x'] (bf16) with the (fp8, scales) tuple sglang would dispatch.
    Mirrors sglang's `sglang_per_token_group_quant_fp8` with column-major scales.
    Handles empty x (num_tokens == 0)."""
    x = wl["x"]
    m, n = x.shape
    if m == 0:
        # per_token_cast_to_fp8's `view(m, -1, 128)` chokes on m==0; build
        # the empty (fp8, scales) tuple directly.
        fp8 = torch.empty((0, n), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((0, n // 128), dtype=torch.float32, device=x.device)
    else:
        fp8, scales = per_token_cast_to_fp8(x)
    # sglang uses transposed-contiguous scales:
    scales = scales.T.contiguous().T
    wl2 = dict(wl)
    wl2["x"] = (fp8, scales)
    return wl2


def dispatch_args(wl, config, async_finish, previous_event=None,
                  allocate_on_comm_stream=False, expert_alignment=1):
    args = dict(
        x=wl["x"],
        topk_idx=wl["topk_idx"],
        topk_weights=wl["topk_weights"],
        num_tokens_per_rank=wl["num_tokens_per_rank"],
        num_tokens_per_rdma_rank=wl["num_tokens_per_rdma_rank"],
        num_tokens_per_expert=wl["num_tokens_per_expert"],
        is_token_in_rank=wl["is_token_in_rank"],
        config=config,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        expert_alignment=expert_alignment,
    )
    if previous_event is not None:
        args["previous_event"] = previous_event
    return args


def fingerprint(t):
    if isinstance(t, tuple):
        t = t[0]
    if t.numel() == 0:
        return (0, 0, 0.0, 0.0)
    fp = (int(t.size(0)), int(t.numel()),
          float(t.float().sum().item()),
          float(t.float().abs().sum().item()))
    return fp


def fp_eq(a, b):
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-topk-groups", type=int, default=None)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--fp8", action="store_true",
                        help="dispatch FP8 (x, scales) tuple like sglang TBO does")
    parser.add_argument("--layers", type=int, default=1,
                        help="repeat full dispatch+combine cycle this many times "
                             "per iter to simulate multi-MoE-layer state")
    parser.add_argument("--dp-attention-style", action="store_true",
                        help="Emulate TBO + DP-attention: half the ranks have a "
                             "zero-token sub-batch for batch B. Reproduces the "
                             "dispatch-CPU timeout deadlock that sglang TBO hits.")
    parser.add_argument("--all-empty-b", action="store_true",
                        help="All ranks have a zero-token batch B "
                             "(TBO warmup pattern when load balancer puts "
                             "everything into batch A).")
    parser.add_argument("--moe-layers", type=int, default=1,
                        help="Number of (dispatch_A, dispatch_B, combine_A, combine_B) "
                             "cycles within a single iter, simulating Qwen3-30B's "
                             "48 MoE layers in one forward pass.")
    parser.add_argument("--only-empty-pattern", action="store_true",
                        help="Skip patterns A/B/C and only run the empty-rank "
                             "pattern. Use with --dp-attention-style.")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    num_nodes = world_size // num_local_ranks
    if args.num_topk_groups is None:
        args.num_topk_groups = min(num_nodes, 4)

    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    is_master = (local_rank == 0 and rank == 0)
    if is_master:
        print(f"[setup] world_size={world_size}, num_nodes={num_nodes}, "
              f"num_local_ranks={num_local_ranks}", flush=True)

    num_sms = 24
    hidden_bytes = args.hidden * 2
    cfg_for_size = Config(num_sms, 8, 512, 16, 512)
    align = lambda s: ((int(s * 1.2) + 127) // 128) * 128
    num_nvlink_bytes = align(cfg_for_size.get_nvl_buffer_size_hint(hidden_bytes, num_ranks))
    num_rdma_bytes = align(cfg_for_size.get_rdma_buffer_size_hint(hidden_bytes, num_ranks))
    if is_master:
        print(f"[buffer] num_nvlink_bytes={num_nvlink_bytes/1e9:.2f} GB, "
              f"num_rdma_bytes={num_rdma_bytes/1e9:.2f} GB", flush=True)

    buffer = Buffer(
        group, num_nvlink_bytes, num_rdma_bytes,
        low_latency_mode=False, num_qps_per_rank=num_sms,
        explicitly_destroy=True,
    )
    config = Config(num_sms, 8, 512, 16, 512)

    total_fail = 0

    for it in range(args.iters):
        # Two DIFFERENT micro-batches (different seeds -> different x/topk).
        wl_A = gen_workload(args.num_tokens, args.hidden, args.num_experts,
                            args.num_topk, args.num_topk_groups, num_nodes,
                            num_ranks, num_local_ranks, rank,
                            seed=100000 * rank + 2 * it + 1)
        wl_B = gen_workload(args.num_tokens, args.hidden, args.num_experts,
                            args.num_topk, args.num_topk_groups, num_nodes,
                            num_ranks, num_local_ranks, rank,
                            seed=100000 * rank + 2 * it + 2)

        # Optional: emulate TBO + DP-attention by giving some ranks an
        # empty wl_B. This is the missing ingredient that made the previous
        # microbench miss the hang.
        empty_b_for_this_rank = args.all_empty_b or (
            args.dp_attention_style and (rank % 2 == 1)
        )
        if empty_b_for_this_rank:
            wl_B = gen_empty_workload(
                args.hidden, args.num_experts, args.num_topk,
                num_nodes, num_ranks,
            )

        if args.fp8:
            wl_A = to_fp8(wl_A)
            wl_B = to_fp8(wl_B)

        if args.only_empty_pattern:
            # Skip baselines so we go straight to the back-to-back dispatch
            # pattern below; it's the only one that reproduces the hang.
            fp_A_base = fp_B_base = None
        else:
            # ---- BASELINES: each dispatch alone, with full sync between -------
            torch.cuda.synchronize(); group.barrier()
            recv_A_base, _, _, _, hA, _ = buffer.dispatch(
                **dispatch_args(wl_A, config, async_finish=False))
            torch.cuda.synchronize()
            fp_A_base = fingerprint(recv_A_base)
            del recv_A_base, hA

            torch.cuda.synchronize(); group.barrier()
            recv_B_base, _, _, _, hB, _ = buffer.dispatch(
                **dispatch_args(wl_B, config, async_finish=False))
            torch.cuda.synchronize()
            fp_B_base = fingerprint(recv_B_base)
            del recv_B_base, hB

            if is_master:
                print(f"[iter {it}] baseline_A={fp_A_base}", flush=True)
                print(f"[iter {it}] baseline_B={fp_B_base}", flush=True)

        # ---- PATTERN A: back-to-back dispatch+combine, async_finish=True ----
        # Run --moe-layers cycles of (dispatch_A, dispatch_B, combine_A, combine_B)
        # within a single iter, mimicking N MoE layers in one forward pass.
        # This mirrors sglang's TBO `_dispatch_core`: previous_event chain
        # (Buffer.capture()), allocate_on_comm_stream=True, and the FP8
        # expert_alignment=128 path.
        for moe_layer in range(args.moe_layers):
            torch.cuda.synchronize(); group.barrier()
            if is_master and moe_layer == 0:
                empty_ranks_msg = (
                    " (rank%2==1 ranks have empty B)" if args.dp_attention_style else
                    " (ALL ranks have empty B)" if args.all_empty_b else ""
                )
                print(f"[iter {it}] pattern_A back-to-back{empty_ranks_msg}: "
                      f"running {args.moe_layers} MoE-layer cycles...", flush=True)
            prev_A = buffer.capture()
            recv_A, _, _, _, handle_A, evA = buffer.dispatch(
                **dispatch_args(wl_A, config, async_finish=True,
                                previous_event=prev_A,
                                allocate_on_comm_stream=True,
                                expert_alignment=128 if args.fp8 else 1))
            prev_B = buffer.capture()
            recv_B, _, _, _, handle_B, evB = buffer.dispatch(
                **dispatch_args(wl_B, config, async_finish=True,
                                previous_event=prev_B,
                                allocate_on_comm_stream=True,
                                expert_alignment=128 if args.fp8 else 1))
            evA.current_stream_wait(); evB.current_stream_wait()

            # combine() expects bf16; if dispatch produced fp8 tuples,
            # take just the fp8 component and upcast.
            out_A = recv_A if not isinstance(recv_A, tuple) else recv_A[0]
            out_B = recv_B if not isinstance(recv_B, tuple) else recv_B[0]
            if out_A.dtype != torch.bfloat16: out_A = out_A.to(torch.bfloat16)
            if out_B.dtype != torch.bfloat16: out_B = out_B.to(torch.bfloat16)
            combined_A, _, ev_cA = buffer.combine(out_A, handle_A,
                                                  async_finish=True, config=config)
            combined_B, _, ev_cB = buffer.combine(out_B, handle_B,
                                                  async_finish=True, config=config)
            ev_cA.current_stream_wait(); ev_cB.current_stream_wait()
            torch.cuda.synchronize()

            if is_master and (moe_layer % 8 == 0 or moe_layer == args.moe_layers - 1):
                fp_A_a = fingerprint(recv_A); fp_B_a = fingerprint(recv_B)
                print(f"[iter {it} layer {moe_layer}/{args.moe_layers}] "
                      f"A={fp_A_a} B={fp_B_a}", flush=True)
            del recv_A, recv_B, combined_A, combined_B

        if args.only_empty_pattern:
            continue

        # ---- PATTERN B: two CUDA streams, overlapped (most TBO-like) -------
        torch.cuda.synchronize(); group.barrier()
        sA = torch.cuda.Stream(); sB = torch.cuda.Stream()
        with torch.cuda.stream(sA):
            recv_A, _, _, _, _, evA = buffer.dispatch(
                **dispatch_args(wl_A, config, async_finish=True))
        with torch.cuda.stream(sB):
            recv_B, _, _, _, _, evB = buffer.dispatch(
                **dispatch_args(wl_B, config, async_finish=True))
        sA.synchronize(); sB.synchronize(); torch.cuda.synchronize()
        fp_A_b = fingerprint(recv_A); fp_B_b = fingerprint(recv_B)
        ok_A_b = fp_eq(fp_A_b, fp_A_base); ok_B_b = fp_eq(fp_B_b, fp_B_base)
        if not (ok_A_b and ok_B_b):
            total_fail += 1
        if is_master:
            print(f"[iter {it}] pattern_B 2-stream:    "
                  f"A={fp_A_b} okA={ok_A_b} B={fp_B_b} okB={ok_B_b}", flush=True)
        del recv_A, recv_B

        # ---- PATTERN C: repeated dispatch->combine cycles (multi-layer-ish)
        # Mimics N MoE layers of TBO: dispatch_a, dispatch_b, combine_a, combine_b
        # all on the same Buffer with different inputs, looped to expose
        # cumulative state drift.
        for L in range(args.layers):
            torch.cuda.synchronize(); group.barrier()
            prev = buffer.capture()
            (recv_A, _, _, _, handle_A, evA) = buffer.dispatch(
                **dispatch_args(wl_A, config, async_finish=True, previous_event=prev))
            (recv_B, _, _, _, handle_B, evB) = buffer.dispatch(
                **dispatch_args(wl_B, config, async_finish=True, previous_event=prev))
            evA.current_stream_wait(); evB.current_stream_wait()

            out_A = recv_A if not isinstance(recv_A, tuple) else recv_A[0]
            out_B = recv_B if not isinstance(recv_B, tuple) else recv_B[0]
            # combine expects bf16
            if out_A.dtype != torch.bfloat16:
                out_A = out_A.to(torch.bfloat16)
            if out_B.dtype != torch.bfloat16:
                out_B = out_B.to(torch.bfloat16)

            combined_A, _, ev_cA = buffer.combine(out_A, handle_A,
                                                  async_finish=True, config=config)
            combined_B, _, ev_cB = buffer.combine(out_B, handle_B,
                                                  async_finish=True, config=config)
            ev_cA.current_stream_wait(); ev_cB.current_stream_wait()
            torch.cuda.synchronize()

            fp_A_c = fingerprint(recv_A); fp_B_c = fingerprint(recv_B)
            ok_A_c = fp_eq(fp_A_c, fp_A_base); ok_B_c = fp_eq(fp_B_c, fp_B_base)
            if not (ok_A_c and ok_B_c):
                total_fail += 1
            if is_master:
                print(f"[iter {it} layer {L}] pattern_C dispatch+combine: "
                      f"A={fp_A_c} okA={ok_A_c} B={fp_B_c} okB={ok_B_c} "
                      f"cA={tuple(combined_A.shape)} cB={tuple(combined_B.shape)}",
                      flush=True)

        group.barrier()

    if is_master:
        print(f"[summary] total mismatch events across ranks/iters/patterns: {total_fail}",
              flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
