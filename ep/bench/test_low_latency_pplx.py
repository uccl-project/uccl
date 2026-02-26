"""
This is the same test_low_latency.py test in DeepEP's repo.
On first node:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/test_low_latency_pplx.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=288

On second node:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/test_low_latency_pplx.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=288
"""

import argparse
import random
import time
import os
import torch
import torch.distributed as dist
from typing import Optional

from buffer import Buffer
from utils import (
    init_dist,
    init_dist_under_torchrun,
    calc_diff,
    hash_tensor,
    per_token_cast_back,
    initialize_uccl,
    destroy_uccl,
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
    dispatch_use_fp8: bool = True,
    seed: int = 0,
    skip_benchmark: bool = False,
    debug_hash: bool = False,
    num_warmup: int = 10000,
    num_repeats: int = 10000,
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
    hash_details = {} if debug_hash else None

    def _record_hash(label: str, t: torch.Tensor, include_in_overall: bool = True):
        nonlocal hash_value
        if not t.is_contiguous():
            t = t.contiguous()
        hv = hash_tensor(t)
        if include_in_overall:
            hash_value ^= hv
        if hash_details is not None:
            # Preserve the XOR aggregation behavior at per-label granularity.
            hash_details[label] = hash_details.get(label, 0) ^ hv

    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8_case in (False, True):
                for round_scale in (False,):
                    for round_scale in (
                        (False, True) if dispatch_use_fp8_case else (False,)
                    ):
                        for use_ue8m0 in (False, True) if round_scale else (False,):
                            print(
                                "Start experiment with settings:"
                                f" return_recv_hook={return_recv_hook}"
                                f" dispatch_use_fp8={dispatch_use_fp8_case}"
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
                                    use_fp8=dispatch_use_fp8_case,
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
                                if dispatch_use_fp8_case
                                else packed_recv_x
                            )
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, hidden // 128),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8_case
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
                                    if dispatch_use_fp8_case
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
                                if dispatch_use_fp8_case:
                                    tag = (
                                        f"x={'x' if current_x is x else 'rand'}"
                                        f"|hook={return_recv_hook}"
                                        f"|fp8={dispatch_use_fp8_case}"
                                        f"|rs={round_scale}"
                                        f"|ue={use_ue8m0}"
                                        f"|le={i}"
                                        f"|nvt={num_valid_tokens}"
                                    )
                                    _record_hash(
                                        f"dispatch_fp8_data|{tag}",
                                        packed_recv_x[0][i, :num_valid_tokens],
                                    )
                                    _record_hash(
                                        f"dispatch_fp8_scale|{tag}",
                                        packed_recv_x[1][i, :num_valid_tokens],
                                    )
                                else:
                                    tag = (
                                        f"x={'x' if current_x is x else 'rand'}"
                                        f"|hook={return_recv_hook}"
                                        f"|fp8={dispatch_use_fp8_case}"
                                        f"|rs={round_scale}"
                                        f"|ue={use_ue8m0}"
                                        f"|le={i}"
                                        f"|nvt={num_valid_tokens}"
                                    )
                                    _record_hash(
                                        f"dispatch_bf16|{tag}",
                                        packed_recv_x[i, :num_valid_tokens],
                                    )
                                _record_hash(
                                    f"dispatch_meta_count|{tag}",
                                    packed_recv_count[i],
                                    include_in_overall=False,
                                )
                                _record_hash(
                                    f"dispatch_meta_src_info|{tag}",
                                    recv_src_info[:num_valid_tokens],
                                    include_in_overall=False,
                                )
                                _record_hash(
                                    f"dispatch_meta_layout_range|{tag}",
                                    recv_layout_range,
                                    include_in_overall=False,
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
                                        9e-4 if dispatch_use_fp8_case else 1e-5
                                    ), f"Error: {diff=}, {dispatch_use_fp8_case=}, {zero_copy=}"
                                    tag = (
                                        f"x={'x' if current_x is x else 'rand'}"
                                        f"|hook={return_recv_hook}"
                                        f"|fp8={dispatch_use_fp8_case}"
                                        f"|rs={round_scale}"
                                        f"|ue={use_ue8m0}"
                                        f"|zc={zero_copy}"
                                        f"|logfmt={use_logfmt}"
                                    )
                                    _record_hash(f"combine_out|{tag}", combined_x)

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
            use_fp8=dispatch_use_fp8,
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

    if skip_benchmark:
        return (hash_value, hash_details) if debug_hash else hash_value

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    # For pplx-style benchmark, each token routes to exactly num_topk experts.
    # Use benchmark-route bytes (not correctness masked routes) for apples-to-apples bandwidth.
    num_dispatch_comm_bytes = (
        num_tokens * num_topk * (num_fp8_bytes if dispatch_use_fp8 else num_bf16_bytes)
    )
    num_combine_comm_bytes = (
        num_tokens * num_topk * (num_logfmt10_bytes if use_logfmt else num_bf16_bytes)
    )

    # Benchmark with the same timing structure as pplx/benchmarks/bench_all_to_all.py
    out_dummy = torch.empty((1,), dtype=torch.float32, device="cuda")
    gemm = torch.empty(
        (2048, 2048) if num_tokens <= 128 else (8192, 8192),
        dtype=torch.float32,
        device="cuda",
    )
    rng = torch.Generator(device="cuda")
    rng.manual_seed(rank + seed + 123)

    pending_dispatch_hook = None
    pending_combine_hook = None
    pending_recv_x = None
    pending_handle = None
    bench_topk_idx = topk_idx

    def wait():
        # Same "wait" structure as bench_all_to_all.py
        dist.all_reduce(out_dummy, group=group)
        _ = gemm @ gemm
        dist.all_reduce(out_dummy, group=group)

    def _rand_topk_idx() -> torch.Tensor:
        scores = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.float32,
            device="cuda",
            generator=rng,
        )
        scores = scores.abs() + 1
        return torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]

    def dispatch(do_send: bool, do_recv: bool):
        nonlocal pending_dispatch_hook
        nonlocal pending_recv_x
        nonlocal pending_handle
        if do_send:
            recv_x, _, handle, _, hook = buffer.low_latency_dispatch(
                current_x,
                bench_topk_idx,
                num_tokens,
                num_experts,
                cumulative_local_expert_recv_stats=None,
                use_fp8=dispatch_use_fp8,
                async_finish=False,
                return_recv_hook=not do_recv,
            )
            if do_recv:
                return recv_x, handle
            pending_dispatch_hook = hook
            pending_recv_x = recv_x
            pending_handle = handle
            return None, None
        assert do_recv, "Invalid dispatch mode"
        assert pending_dispatch_hook is not None
        pending_dispatch_hook()
        out = (pending_recv_x, pending_handle)
        pending_dispatch_hook = None
        pending_recv_x = None
        pending_handle = None
        return out

    def materialize_for_combine(recv_x):
        if dispatch_use_fp8:
            return per_token_cast_back(
                recv_x[0].view(-1, hidden),
                recv_x[1].contiguous().view(-1, hidden // 128),
            ).view(recv_x[0].shape)
        return recv_x

    def combine(simulated_x, handle, do_send: bool, do_recv: bool):
        nonlocal pending_combine_hook
        if do_send:
            _, _, hook = buffer.low_latency_combine(
                simulated_x,
                bench_topk_idx,
                topk_weights,
                handle,
                use_logfmt=use_logfmt,
                return_recv_hook=not do_recv,
            )
            if not do_recv:
                pending_combine_hook = hook
            return
        assert do_recv, "Invalid combine mode"
        assert pending_combine_hook is not None
        pending_combine_hook()
        pending_combine_hook = None

    events = []
    for _ in range(num_warmup + num_repeats):
        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)
        dispatch_send_start = torch.cuda.Event(enable_timing=True)
        dispatch_send_end = torch.cuda.Event(enable_timing=True)
        dispatch_recv_start = torch.cuda.Event(enable_timing=True)
        dispatch_recv_end = torch.cuda.Event(enable_timing=True)
        combine_send_start = torch.cuda.Event(enable_timing=True)
        combine_send_end = torch.cuda.Event(enable_timing=True)
        combine_recv_start = torch.cuda.Event(enable_timing=True)
        combine_recv_end = torch.cuda.Event(enable_timing=True)
        dispatch_start.record()
        dispatch_end.record()
        combine_start.record()
        combine_end.record()
        dispatch_send_start.record()
        dispatch_send_end.record()
        dispatch_recv_start.record()
        dispatch_recv_end.record()
        combine_send_start.record()
        combine_send_end.record()
        combine_recv_start.record()
        combine_recv_end.record()
        events.append(
            (
                dispatch_start,
                dispatch_end,
                combine_start,
                combine_end,
                dispatch_send_start,
                dispatch_send_end,
                dispatch_recv_start,
                dispatch_recv_end,
                combine_send_start,
                combine_send_end,
                combine_recv_start,
                combine_recv_end,
            )
        )

    last_report_time = time.time()
    profiler_started = False
    for i in range(num_warmup + num_repeats):
        if i + 1 == num_warmup and num_warmup > 0:
            torch.cuda.profiler.start()
            profiler_started = True
        now = time.time()
        if rank == 0 and (
            now - last_report_time > 1 or i + 1 == num_warmup + num_repeats
        ):
            print(
                f"[pplx][rank 0] Iteration {i + 1}/{num_warmup + num_repeats}",
                flush=True,
            )
            last_report_time = now

        (
            dispatch_start,
            dispatch_end,
            combine_start,
            combine_end,
            dispatch_send_start,
            dispatch_send_end,
            dispatch_recv_start,
            dispatch_recv_end,
            combine_send_start,
            combine_send_end,
            combine_recv_start,
            combine_recv_end,
        ) = events[i]

        bench_topk_idx = _rand_topk_idx()

        # Send + recv back-to-back
        wait()
        dispatch_start.record()
        recv_x, handle = dispatch(do_send=True, do_recv=True)
        dispatch_end.record()
        simulated_x = materialize_for_combine(recv_x)

        wait()
        combine_start.record()
        combine(simulated_x, handle, do_send=True, do_recv=True)
        combine_end.record()

        # Send and recv split by long kernels
        wait()
        dispatch_send_start.record()
        dispatch(do_send=True, do_recv=False)
        dispatch_send_end.record()

        wait()
        dispatch_recv_start.record()
        recv_x, handle = dispatch(do_send=False, do_recv=True)
        dispatch_recv_end.record()
        simulated_x = materialize_for_combine(recv_x)

        wait()
        combine_send_start.record()
        combine(simulated_x, handle, do_send=True, do_recv=False)
        combine_send_end.record()

        wait()
        combine_recv_start.record()
        combine(None, None, do_send=False, do_recv=True)
        combine_recv_end.record()

    torch.cuda.synchronize()
    if profiler_started:
        torch.cuda.profiler.stop()

    dispatch_times_us = []
    dispatch_send_times_us = []
    dispatch_recv_times_us = []
    combine_times_us = []
    combine_send_times_us = []
    combine_recv_times_us = []
    for (
        dispatch_st,
        dispatch_en,
        combine_st,
        combine_en,
        dispatch_send_st,
        dispatch_send_en,
        dispatch_recv_st,
        dispatch_recv_en,
        combine_send_st,
        combine_send_en,
        combine_recv_st,
        combine_recv_en,
    ) in events[num_warmup:]:
        dispatch_times_us.append(dispatch_st.elapsed_time(dispatch_en) * 1000.0)
        combine_times_us.append(combine_st.elapsed_time(combine_en) * 1000.0)
        dispatch_send_times_us.append(
            dispatch_send_st.elapsed_time(dispatch_send_en) * 1000.0
        )
        dispatch_recv_times_us.append(
            dispatch_recv_st.elapsed_time(dispatch_recv_en) * 1000.0
        )
        combine_send_times_us.append(
            combine_send_st.elapsed_time(combine_send_en) * 1000.0
        )
        combine_recv_times_us.append(
            combine_recv_st.elapsed_time(combine_recv_en) * 1000.0
        )

    gathered = [None for _ in range(num_ranks)]
    dist.all_gather_object(gathered, dispatch_times_us, group=group)
    dispatch_times_us = [v for per_rank in gathered for v in per_rank]
    dist.all_gather_object(gathered, dispatch_send_times_us, group=group)
    dispatch_send_times_us = [v for per_rank in gathered for v in per_rank]
    dist.all_gather_object(gathered, dispatch_recv_times_us, group=group)
    dispatch_recv_times_us = [v for per_rank in gathered for v in per_rank]
    dist.all_gather_object(gathered, combine_times_us, group=group)
    combine_times_us = [v for per_rank in gathered for v in per_rank]
    dist.all_gather_object(gathered, combine_send_times_us, group=group)
    combine_send_times_us = [v for per_rank in gathered for v in per_rank]
    dist.all_gather_object(gathered, combine_recv_times_us, group=group)
    combine_recv_times_us = [v for per_rank in gathered for v in per_rank]

    def _p50(values):
        # Match pplx Statistics.create percentile behavior exactly.
        if not values:
            return 0.0
        xs = sorted(float(v) for v in values)
        n = len(xs)
        index = int(n * 0.5)
        if n * 0.5 == index or index + 1 >= n:
            return xs[index]
        return (xs[index] + xs[index + 1]) / 2.0

    if rank == 0:
        dispatch_p50_s = _p50(dispatch_times_us) / 1e6
        combine_p50_s = _p50(combine_times_us) / 1e6
        dispatch_bw = num_dispatch_comm_bytes / 1e9 / dispatch_p50_s
        combine_bw = num_combine_comm_bytes / 1e9 / combine_p50_s

        print(
            f"[pplx][rank 0] Dispatch both p50: {_p50(dispatch_times_us):.2f} us, {dispatch_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"[pplx][rank 0] Dispatch send p50: {_p50(dispatch_send_times_us):.2f} us",
            flush=True,
        )
        print(
            f"[pplx][rank 0] Dispatch recv p50: {_p50(dispatch_recv_times_us):.2f} us",
            flush=True,
        )
        print(
            f"[pplx][rank 0] Combine both p50: {_p50(combine_times_us):.2f} us, {combine_bw:.2f} GB/s",
            flush=True,
        )
        print(
            f"[pplx][rank 0] Combine send p50: {_p50(combine_send_times_us):.2f} us",
            flush=True,
        )
        print(
            f"[pplx][rank 0] Combine recv p50: {_p50(combine_recv_times_us):.2f} us",
            flush=True,
        )
    return (hash_value, hash_details) if debug_hash else hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )

    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
        explicitly_destroy=True,
        allow_mnnvl=args.allow_mnnvl,
    )

    for seed in range(int(1e9)):
        if local_rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        torch.manual_seed(rank + seed)
        ref_out = test_main(
            num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            use_logfmt=args.use_logfmt,
            dispatch_use_fp8=args.dispatch_use_fp8,
            seed=seed,
            skip_benchmark=args.pressure_test_mode == 1,
            debug_hash=args.debug_hash,
            num_warmup=args.num_warmup,
            num_repeats=args.num_repeats,
        )
        if args.debug_hash:
            ref_hash, ref_hash_details = ref_out
        else:
            ref_hash, ref_hash_details = ref_out, None
        if args.pressure_test_mode == 0:
            break

        if local_rank == 0:
            print(f"{ref_hash=}")
            print("", flush=True)

        for _ in range(20):
            torch.manual_seed(rank + seed)
            cur_out = test_main(
                num_tokens,
                hidden,
                num_experts,
                num_topk,
                rank,
                num_ranks,
                group,
                buffer,
                use_logfmt=args.use_logfmt,
                dispatch_use_fp8=args.dispatch_use_fp8,
                seed=seed,
                skip_benchmark=args.pressure_test_mode == 1,
                debug_hash=args.debug_hash,
                num_warmup=args.num_warmup,
                num_repeats=args.num_repeats,
            )
            if args.debug_hash:
                current_hash, current_hash_details = cur_out
            else:
                current_hash, current_hash_details = cur_out, None

            if current_hash != ref_hash:
                print(
                    f"[rank {rank} local_rank {local_rank}] NON-DETERMINISM: "
                    f"seed={seed} current_hash={current_hash} ref_hash={ref_hash}",
                    flush=True,
                )
                if args.debug_hash and ref_hash_details and current_hash_details:
                    diffs = []
                    keys = set(ref_hash_details.keys()) | set(
                        current_hash_details.keys()
                    )
                    for k in sorted(keys):
                        a = ref_hash_details.get(k, 0)
                        b = current_hash_details.get(k, 0)
                        if a != b:
                            diffs.append((k, a, b))
                    if diffs:
                        k0, a0, b0 = diffs[0]
                        print(
                            f"[rank {rank}] First differing tensor: {k0}\n"
                            f"  ref={a0} cur={b0}\n"
                            f"[rank {rank}] Total differing labels: {len(diffs)}",
                            flush=True,
                        )
                        for k, a, b in diffs[:10]:
                            print(f"[rank {rank}] DIFF {k} ref={a} cur={b}", flush=True)
                    else:
                        print(
                            f"[rank {rank}] Hash differs but no per-label diffs "
                            f"(possible XOR collision).",
                            flush=True,
                        )
                # assert current_hash == ref_hash, f"Error: seed={seed}"

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
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
        "--dispatch-use-fp8",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether dispatch path uses FP8 casting (default: true).",
    )
    parser.add_argument(
        "--pressure-test-mode",
        type=int,
        default=0,
        help="Pressure test mode. 0: don't do pressure test, 1: do pressure test without benchmarks, 2: do pressure test with benchmarks",
    )
    parser.add_argument(
        "--debug-hash",
        action="store_true",
        help="Print per-tensor hash breakdown when non-determinism is detected.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=200,
        help="Number of warmup iterations for pplx-style measurement.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=500,
        help="Number of measured iterations for pplx-style measurement.",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    # NOTE: modified from deep_ep
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    test_loop(local_rank, num_local_ranks, args)
