"""
A/B of the default low-latency kernels and the per-rank dedup kernels
(`low_latency_dispatch_dedup` / `low_latency_combine_dedup`) in one process.

Checks:
  - dispatch: for every local expert and source rank, the dedup path delivers
    the same tokens (by source index) with bitwise-identical payloads and
    scales as the default path;
  - combine: the dedup result matches the default combine (calc_diff).
Then times both paths: dispatch + combine (CUDA events, L2 flushed), per-kernel
times (kineto), and send/recv kernel times with a GEMM in between (hooks).
Prints one JSON line per rank and path.

torchrun --nnodes=2 --nproc_per_node=1 ... bench/test_ll_dedup.py \
  --num-tokens 1024 --hidden 2048 --num-topk 8 --num-experts 128
"""

import argparse
import json
import os
import random
import time
from functools import partial

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import (
    bench,
    bench_kineto,
    calc_diff,
    init_dist_under_torchrun,
    per_token_cast_back,
)


def unpack_range(layout_range):
    return (layout_range & 0xFFFFFFFF).to(torch.int64), (layout_range >> 32).to(
        torch.int64
    )


def expert_scale(global_expert):
    """Each simulated expert scales its input by 1, 1.5, ..., 4 (spread over
    neighbouring ids), so a weight paired with the wrong expert's row moves
    that token's output by several percent."""
    return 1.0 + 0.5 * ((37 * global_expert) % 7)


def fp8_roundtrip(x, round_scale=False):
    """The dispatch FP8 cast (per 128 channels, scale 448 / amax, or a power
    of two with `round_scale`) and back."""
    xf = x.float().view(x.size(0), -1, 128)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    if round_scale:
        scale_inv = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
        scale = 1.0 / scale_inv
    else:
        # One rounded division, as the kernel does: `448.0 / amax` would be
        # reciprocal(amax) * 448 in torch, a scale that can differ by one ulp
        # and round BF16 inputs that sit exactly on an FP8 tie the other way.
        scale, scale_inv = torch.full_like(amax, 448.0) / amax, amax * (1.0 / 448.0)
    q = (xf * scale).to(torch.float8_e4m3fn)
    return (q.float() * scale_inv).view_as(x).to(torch.bfloat16)


def zip_bytes(rows):
    """Bytes of the per-row window code for BF16 rows [N, H] as the kernel
    packs them (window from the mean nonzero exponent; raw when packing is
    not smaller)."""
    n, h = rows.shape
    e = (rows.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    cnt = (e != 0).sum(dim=1, keepdim=True)
    base = ((e.sum(dim=1, keepdim=True) + cnt // 2) // cnt.clamp_min(1) - 4).clamp(
        0, 255 - 6
    )
    base = torch.where(cnt == 0, torch.zeros_like(base), base)
    esc = ((e < base) | (e > base + 6)).sum(dim=1)
    words = ((h // 256 + 3) // 4) * 3 * 32 * 4
    packed = ((16 + h + words + esc + 15) // 16) * 16
    max_esc = (16 + 2 * h) - 16 - (16 + h + words)
    return torch.where(esc > max_esc, torch.full_like(packed, 16 + 2 * h), packed)


# Per-token tolerance: the reference reproduces the expert inputs and outputs
# bit for bit, so only the combine's own BF16 sums differ (well under 1%),
# while a lost row or a weight on the wrong expert moves a token by more.
ROW_TOL = 1e-2


def row_errors(out, ref):
    """Per-token relative error of a combine output; a token with any
    non-finite value counts as infinitely wrong (NaN would compare false
    against the tolerance and pass)."""
    out, ref = out.float(), ref.float()
    err = (out - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)
    bad = ~torch.isfinite(out).all(dim=1) | ~torch.isfinite(err)
    return torch.where(bad, torch.full_like(err, float("inf")), err)


def simulated_expert_out(recv_x, hidden, use_fp8, rank):
    if use_fp8:
        scales = recv_x[1].contiguous()  # float per 128 channels, or packed UE8M0
        y = per_token_cast_back(
            recv_x[0].view(-1, hidden), scales.view(-1, scales.size(-1))
        ).view(recv_x[0].shape)
    else:
        y = recv_x.clone()
    num_local_experts = y.size(0)
    scale = expert_scale(
        torch.arange(num_local_experts, device=y.device) + rank * num_local_experts
    ).float()
    return (y.float() * scale.view(-1, 1, 1)).to(y.dtype)


def combine_reference(x, topk_idx, topk_weights, use_fp8=True, round_scale=False):
    """sum_k w_k * expert_k(x_hat) with the same FP8 and BF16 roundings as
    the expert inputs and outputs (only the combine's own sums differ)."""
    x_hat = fp8_roundtrip(x, round_scale) if use_fp8 else x
    scale = expert_scale(topk_idx.clamp_min(0)).float()
    y = (x_hat.float().unsqueeze(1) * scale.unsqueeze(-1)).to(torch.bfloat16).float()
    w = topk_weights.masked_fill(topk_idx == -1, 0)
    return (y * w.unsqueeze(-1)).sum(dim=1)


def check_dispatch(a, b, num_ranks, use_fp8):
    """Compare default (a) and dedup (b) dispatch outputs; return mismatch counts."""
    (xa, ca, sa, la), (xb, cb, sb, lb) = a, b
    count_mismatch = int((ca != cb).sum().item())
    row_mismatch = 0
    num_a, begin_a = unpack_range(la)
    num_b, begin_b = unpack_range(lb)
    count_mismatch += int((num_a != num_b).sum().item())
    data_a = xa[0] if use_fp8 else xa
    data_b = xb[0] if use_fp8 else xb
    for e in range(data_a.size(0)):
        for s in range(num_ranks):
            n = int(num_a[e, s].item())
            if n != int(num_b[e, s].item()):
                continue
            ba, bb = int(begin_a[e, s].item()), int(begin_b[e, s].item())
            src_a, ia = torch.sort(sa[e, ba : ba + n])
            src_b, ib = torch.sort(sb[e, bb : bb + n])
            row_mismatch += int((src_a != src_b).sum().item())
            ra = data_a[e, ba : ba + n][ia].view(torch.uint8)
            rb = data_b[e, bb : bb + n][ib].view(torch.uint8)
            row_mismatch += int((ra != rb).any(dim=1).sum().item())
            if use_fp8:
                qa = xa[1][e, ba : ba + n][ia]
                qb = xb[1][e, bb : bb + n][ib]
                row_mismatch += int((qa != qb).any(dim=1).sum().item())
    return count_mismatch, row_mismatch


def graph_measure(fn, iters=50):
    """Per-step time when dispatch + combine replay from a CUDA graph (decode
    runs this way), which removes host issue cost from the step."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        return {"graph_us": start.elapsed_time(end) / iters * 1e3}
    except Exception as e:  # report, do not hide, a capture failure
        return {"graph_error": repr(e)[:300]}


NIC_COUNTERS = (
    "hw_counters/out_of_buffer",
    "hw_counters/out_of_sequence",
    "hw_counters/packet_seq_err",
    "hw_counters/local_ack_timeout_err",
    "hw_counters/duplicate_request",
    "hw_counters/implied_nak_seq_err",
    "hw_counters/rnr_nak_retry_err",
    "hw_counters/req_transport_retries_exceeded",
    "counters/port_xmit_data",
    "counters/port_rcv_data",
    "counters/port_xmit_wait",
    "counters/port_xmit_discards",
    "counters/port_rcv_errors",
)


def nic_counters():
    """Read (never reset) the port counters of every RDMA device on this node."""
    out = {}
    base = "/sys/class/infiniband"
    for dev in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        for c in NIC_COUNTERS:
            try:
                with open(f"{base}/{dev}/ports/1/{c}") as f:
                    out[f"{dev}:{c.split('/')[-1]}"] = int(f.read())
            except OSError:
                pass
    return out


def sustain(name, fn, seconds, period_s=0.0):
    """Replay dispatch + combine from a CUDA graph for about `seconds`, either
    back to back or paced to one step per `period_s` (same offered token rate
    for every path). Prints wall-clock phase bounds and NIC counter deltas so
    an external latency probe can be split by phase. Both ranks replay the
    same number of steps."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t) / 20
    n = torch.tensor([int(seconds / max(step_s, period_s))], device="cuda")
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    n = int(n.item())
    dist.barrier()
    c0 = nic_counters()
    w0 = time.time()
    t0 = time.perf_counter()
    for i in range(n):
        g.replay()
        if period_s > 0:
            torch.cuda.synchronize()
            while time.perf_counter() < t0 + (i + 1) * period_s:
                pass
        elif i % 100 == 99:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    w1 = time.time()
    c1 = nic_counters()
    dist.barrier()
    return {
        "phase": name,
        "wall_start": w0,
        "wall_end": w1,
        "steps": n,
        "paced_us": period_s * 1e6,
        "step_us": (t1 - t0) / n * 1e6,
        "free_step_us": step_s * 1e6,
        "nic_delta": {k: c1[k] - c0[k] for k in c0 if k in c1},
    }


def interference_measure(fn, steps=20):
    """SM cost seen by a concurrent GEMM: time of a BF16 GEMM alone, of the
    EP steps alone, and of both issued on two streams together. The EP steps
    replay a CUDA graph, so neither path has host-induced idle gaps."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    fn = graph.replay
    # Concurrent partner (UCCL_TEST_PARTNER), about 2.5 ms of work on B300:
    #   dense:  4 x 8192^3 BF16 GEMM (compute and HBM)
    #   small:  2048^3 BF16 GEMMs that stay in L2 (compute, little HBM)
    #   copy:   1 GiB device copies (HBM streaming, no compute)
    #   expert: batched expert FFN GEMMs, 32 experts x 512 rows x 2048 -> 768
    partner = os.environ.get("UCCL_TEST_PARTNER", "dense")
    if partner == "small":
        a = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")
        reps = 256
    elif partner == "copy":
        a = torch.empty((1 << 30,), dtype=torch.uint8, device="cuda")
        b = torch.empty_like(a)
        reps = 8
    elif partner == "expert":
        a = torch.randn((32, 512, 2048), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((32, 2048, 768), dtype=torch.bfloat16, device="cuda")
        reps = 64
    else:
        a = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
        reps = 4

    def work():
        for _ in range(reps):
            if partner == "copy":
                b.copy_(a)
            elif partner == "expert":
                torch.bmm(a, b)
            else:
                a @ b

    gemm_stream = torch.cuda.Stream()

    def timed(run):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6

    def gemms():
        with torch.cuda.stream(gemm_stream):
            work()

    def ep_steps():
        for _ in range(steps):
            fn()

    gemms(), ep_steps()
    gemm_alone = min(timed(gemms) for _ in range(3))
    ep_alone = min(timed(ep_steps) for _ in range(3))

    def both():
        gemms()
        ep_steps()

    together = min(timed(both) for _ in range(3))
    # GEMM time when overlapped, measured with events on its own stream
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(gemm_stream)
    gemms()
    e.record(gemm_stream)
    ep_steps()
    torch.cuda.synchronize()
    gemm_overlapped = s.elapsed_time(e) * 1e3
    out = {
        "partner": partner,
        "ep_graph_step_us": ep_alone / steps,
        "gemm_alone_us": gemm_alone,
        "gemm_overlapped_us": gemm_overlapped,
        "gemm_slowdown": gemm_overlapped / gemm_alone,
        "ep_alone_us": ep_alone,
        "together_us": together,
        "overlap_saving": 1 - together / (gemm_alone + ep_alone),
    }
    # Issue-order / priority controls: the EP kernels are cooperative launches
    # (every block resident at once), so they can wait behind GEMM blocks that
    # already fill the SMs. Same work, EP issued first and/or on a
    # high-priority stream.
    hi = torch.cuda.Stream(priority=-1)

    def ep_on(stream):
        with torch.cuda.stream(stream):
            ep_steps()

    variants = {
        "ep_first": lambda: (ep_steps(), gemms()),
        "gemm_first_hi": lambda: (gemms(), ep_on(hi)),
        "ep_first_hi": lambda: (ep_on(hi), gemms()),
    }
    for name, run in variants.items():
        t = min(timed(run) for _ in range(3))
        out[f"together_{name}_us"] = t
        out[f"overlap_saving_{name}"] = 1 - t / (gemm_alone + ep_alone)
    out["stream_priority_range"] = list(torch.cuda.Stream.priority_range())
    # Spatial partition: with an EP SM budget, keep the GEMM off those SMs too
    # (cuBLAS carveout), so neither side waits for the other's blocks.
    budget = int(os.environ.get("UCCL_EP_LL_NUM_SMS", "0"))
    if budget > 0 and hasattr(torch._C, "_set_sm_carveout_experimental"):
        prev = torch._C._get_sm_carveout_experimental()
        torch._C._set_sm_carveout_experimental(budget)
        try:
            gemms()
            g_alone = min(timed(gemms) for _ in range(3))
            out["gemm_alone_carveout_us"] = g_alone
            for name, run in (
                ("carveout", both),
                ("carveout_ep_first", variants["ep_first"]),
            ):
                t = min(timed(run) for _ in range(3))
                out[f"together_{name}_us"] = t
                out[f"overlap_saving_{name}"] = 1 - t / (g_alone + ep_alone)
        finally:
            torch._C._set_sm_carveout_experimental(prev)
    # The carveout is only a cuBLAS heuristic hint; a green context is a hard
    # partition: the GEMM runs on (all SMs - budget), the EP kernels keep the rest.
    gcs = getattr(torch.cuda, "green_contexts", None)
    if budget > 0 and gcs is not None and gcs.SUPPORTED:
        total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        green = gcs.GreenContext.create(total - budget, torch.cuda.current_device())

        def gemms_green():
            green.set_context()
            work()
            green.pop_context()

        def timed_green(run):
            torch.cuda.synchronize()
            green.set_context()
            torch.cuda.synchronize()
            green.pop_context()
            dist.barrier()
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            green.set_context()
            torch.cuda.synchronize()
            green.pop_context()
            return (time.perf_counter() - t0) * 1e6

        timed_green(gemms_green)
        g_alone = min(timed_green(gemms_green) for _ in range(3))
        out["green_sms"] = total - budget
        out["gemm_alone_green_us"] = g_alone
        # Entering / leaving the green context chains its stream to the then
        # current stream, so the GEMMs get a parent stream of their own and
        # the EP steps run on another stream: no ordering between the two.
        ep_stream = torch.cuda.Stream()

        def gemms_green_split():
            with torch.cuda.stream(gemm_stream):
                gemms_green()

        for name, run in (
            ("green", lambda: (gemms_green(), ep_steps())),
            ("green_ep_first", lambda: (ep_steps(), gemms_green())),
            ("green_split", lambda: (gemms_green_split(), ep_on(ep_stream))),
            ("green_split_ep_first", lambda: (ep_on(ep_stream), gemms_green_split())),
        ):
            t = min(timed_green(run) for _ in range(3))
            out[f"together_{name}_us"] = t
            out[f"overlap_saving_{name}"] = 1 - t / (g_alone + ep_alone)
        # EP's own time while the partner runs on the other SMs (events on
        # the EP stream), against EP alone
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timed_green(lambda: None)
        gemms_green_split()
        e0.record(ep_stream)
        ep_on(ep_stream)
        e1.record(ep_stream)
        timed_green(lambda: None)
        out["ep_under_partner_us"] = e0.elapsed_time(e1) * 1e3
        out["ep_slowdown_under_partner"] = out["ep_under_partner_us"] / ep_alone
    return out


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    T, H, K, E = args.num_tokens, args.hidden, args.num_topk, args.num_experts
    use_fp8 = args.dispatch_use_fp8
    num_local_experts = E // num_ranks
    default_bytes = Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
    dedup_bytes = Buffer.get_low_latency_dedup_rdma_size_hint(T, H, num_ranks, E)
    ab_bytes = Buffer.get_low_latency_ab_rdma_size_hint(T, H, num_ranks, E)
    buffer = Buffer(
        group,
        num_rdma_bytes=dedup_bytes if args.dedup_only else ab_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    def make_inputs(seed):
        torch.manual_seed(seed * 1000 + rank)
        random.seed(seed * 1000 + rank)
        x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda") * 0.1
        x[:, :128] = float(rank)
        scores = torch.randn((T, E), dtype=torch.float32, device="cuda").abs() + 1
        if args.skew > 0:
            # Incast: a fraction of every rank's tokens routes all top-k to
            # rank 0's experts (hot experts)
            hot = torch.rand((T,), device="cuda") < args.skew
            scores[hot, : E // num_ranks] += 10
        topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
        topk_weights = torch.randn((T, K), dtype=torch.float32, device="cuda").abs()
        for _ in range(10):
            topk_idx[random.randint(0, T - 1), random.randint(0, K - 1)] = -1
        return x, topk_idx, topk_weights

    # Correctness over several routings
    results = {
        "rank": rank,
        "tokens": T,
        "hidden": H,
        "experts": E,
        "topk": K,
        "fp8": use_fp8,
        "skew": args.skew,
        "rdma_bytes_default_hint": default_bytes,
        "rdma_bytes_dedup_hint": dedup_bytes,
        "dedup_only": args.dedup_only,
    }
    worst_diff, worst_ref, count_bad, row_bad = 0.0, 0.0, 0, 0
    worst_row, rows_bad, zip_mismatch = 0.0, 0, 0
    seed_diag = []
    for seed in range(args.num_checks if args.dedup_only else 0):
        # Without the default path, check against every rank's inputs
        x, topk_idx, topk_weights = make_inputs(seed)
        all_x = torch.empty((num_ranks, T, H), dtype=x.dtype, device="cuda")
        all_topk = torch.empty((num_ranks, T, K), dtype=topk_idx.dtype, device="cuda")
        dist.all_gather_into_tensor(all_x, x, group=group)
        dist.all_gather_into_tensor(all_topk, topk_idx, group=group)
        rb, cb, hb, _, _ = buffer.low_latency_dispatch_dedup(
            x,
            topk_idx,
            topk_weights,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
        )
        torch.cuda.synchronize()
        num_b, begin_b = unpack_range(hb[1])
        for e in range(num_local_experts):
            ge = rank * num_local_experts + e
            hit = (all_topk == ge).any(dim=-1)  # [ranks, T]
            count_bad += int(cb[e].item() != int(hit.sum().item()))
            data = per_token_cast_back(rb[0][e], rb[1][e]) if use_fp8 else rb[e]
            for src in range(num_ranks):
                n, b0 = int(num_b[e, src].item()), int(begin_b[e, src].item())
                want = torch.nonzero(hit[src]).flatten()
                if n != want.numel():
                    count_bad += 1
                    continue
                got, order = torch.sort(hb[0][e, b0 : b0 + n].to(torch.int64))
                row_bad += int((got != want).sum().item())
                if n:
                    d = calc_diff(
                        data[b0 : b0 + n][order].float(), all_x[src, want].float()
                    )
                    row_bad += int(d > 1e-3)
        yb = simulated_expert_out(rb, H, use_fp8, rank)
        out_b, _, _ = buffer.low_latency_combine_dedup(yb, topk_idx, hb)
        torch.cuda.synchronize()
        ref = combine_reference(x, topk_idx, topk_weights, use_fp8, args.round_scale)
        worst_diff = max(worst_diff, calc_diff(ref, out_b.float()))
        rows = row_errors(out_b, ref)
        worst_row = max(worst_row, rows.max().item())
        rows_bad += int((rows > ROW_TOL).sum().item())
    for seed in range(0 if args.dedup_only else args.num_checks):
        x, topk_idx, topk_weights = make_inputs(seed)
        ra, ca, ha, _, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
        )
        a = (
            tuple(t.clone() for t in ra) if use_fp8 else ra.clone(),
            ca.clone(),
            ha[0].clone(),
            ha[1].clone(),
        )
        ya = simulated_expert_out(ra, H, use_fp8, rank)
        out_a, _, _ = buffer.low_latency_combine(ya, topk_idx, topk_weights, ha)
        out_a = out_a.clone()
        rb, cb, hb, _, _ = buffer.low_latency_dispatch_dedup(
            x,
            topk_idx,
            topk_weights,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
        )
        b = (
            tuple(t.clone() for t in rb) if use_fp8 else rb.clone(),
            cb.clone(),
            hb[0].clone(),
            hb[1].clone(),
        )
        yb = simulated_expert_out(rb, H, use_fp8, rank)
        out_b, _, _ = buffer.low_latency_combine_dedup(yb, topk_idx, hb)
        out_b = out_b.clone()
        if args.compress:
            out_z, _, _ = buffer.low_latency_combine_dedup(
                yb, topk_idx, hb, compress=True
            )
            torch.cuda.synchronize()
            zip_mismatch += int(
                (out_z.view(torch.int16) != out_b.view(torch.int16))
                .any(dim=1)
                .sum()
                .item()
            )
        torch.cuda.synchronize()
        cm, rm = check_dispatch(a, b, num_ranks, use_fp8)
        count_bad += cm
        row_bad += rm
        worst_diff = max(worst_diff, calc_diff(out_a.float(), out_b.float()))
        ref = combine_reference(x, topk_idx, topk_weights, use_fp8, args.round_scale)
        worst_ref = max(
            worst_ref, calc_diff(ref, out_a.float()), calc_diff(ref, out_b.float())
        )
        seed_diag.append(
            {
                "seed": seed,
                "ref_vs_default": calc_diff(ref, out_a.float()),
                "ref_vs_dedup": calc_diff(ref, out_b.float()),
                "nonfinite_default": int((~torch.isfinite(out_a.float())).sum().item()),
                "nonfinite_dedup": int((~torch.isfinite(out_b.float())).sum().item()),
                "nonfinite_ref": int((~torch.isfinite(ref)).sum().item()),
                "absmax_default": out_a.float().abs().max().item(),
                "absmax_dedup": out_b.float().abs().max().item(),
            }
        )
        rows = torch.maximum(row_errors(out_b, ref), row_errors(out_b, out_a))
        worst_row = max(worst_row, rows.max().item())
        rows_bad += int((rows > ROW_TOL).sum().item())
    results.update(
        {
            "check_seeds": args.num_checks,
            "dispatch_count_mismatch": count_bad,
            "dispatch_row_mismatch": row_bad,
            "combine_calc_diff_max": worst_diff,
            "combine_ref_diff_max": worst_ref,
            "combine_row_err_max": worst_row,
            "combine_rows_bad": rows_bad,
            "seed_diag": seed_diag,
            "zip_rows_not_bit_identical": zip_mismatch if args.compress else None,
            "combine_pass": worst_diff
            < ((9e-4 if use_fp8 else 1e-5) if args.dedup_only else 1e-5),
        }
    )

    # Repeated steps with a fresh routing each time, alternating normal and
    # hook launches; every token of every combine is checked against the
    # reference and, when the buffer also fits the default kernels, against
    # the default path. This runs the double-buffered signaling through many
    # cycles.
    stress_bad, stress_worst, stress_oracle_worst = 0, 0.0, 0.0
    for it in range(args.stress):
        x, topk_idx, topk_weights = make_inputs(10_000 + it)
        hooks = it % 2 == 1
        rb, _, hb, _, hook = buffer.low_latency_dispatch_dedup(
            x,
            topk_idx,
            topk_weights,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
            return_recv_hook=hooks,
        )
        if hooks:
            hook()
        yb = simulated_expert_out(rb, H, use_fp8, rank)
        out_b, _, hook = buffer.low_latency_combine_dedup(
            yb, topk_idx, hb, return_recv_hook=hooks, compress=args.compress
        )
        if hooks:
            hook()
        out_b = out_b.clone()
        if args.inject_fault:
            # Positive control: one token off by 3%, one token lost
            out_b[0] *= 1.03
            out_b[1] = 0
        rows = row_errors(
            out_b,
            combine_reference(x, topk_idx, topk_weights, use_fp8, args.round_scale),
        )
        if not args.dedup_only:
            ra, _, ha, _, _ = buffer.low_latency_dispatch(
                x,
                topk_idx,
                T,
                E,
                use_fp8=use_fp8,
                round_scale=args.round_scale,
                use_ue8m0=args.ue8m0,
            )
            out_a, _, _ = buffer.low_latency_combine(
                simulated_expert_out(ra, H, use_fp8, rank), topk_idx, topk_weights, ha
            )
            oracle = row_errors(out_b, out_a)
            stress_oracle_worst = max(stress_oracle_worst, oracle.max().item())
            rows = torch.maximum(rows, oracle)
        stress_worst = max(stress_worst, rows.max().item())
        stress_bad += int((rows > ROW_TOL).sum().item()) + int(
            torch.isnan(out_b).any().item()
        )
    if args.stress:
        results.update(
            stress_steps=args.stress,
            stress_bad_rows=stress_bad,
            stress_row_err_max=stress_worst,
            stress_oracle_row_err_max=stress_oracle_worst,
            stress_oracle="default" if not args.dedup_only else "none",
        )

    # Two micro-batches interleaved as in two-batch overlap: dispatch A,
    # dispatch B, combine A, combine B, each with compute between its send and
    # its receive hook, and both handles alive. (The double-buffered signaling
    # requires each call's receive before the next call's send: a send clears
    # the other buffer's signals, for the default kernels as well.)
    dbo_bad, dbo_worst = 0, 0.0
    filler = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")
    for it in range(args.stress_dbo):
        mb = [make_inputs(20_000 + 2 * it + j) for j in range(2)]
        disp = []
        for xm, tm, wm in mb:
            d = buffer.low_latency_dispatch_dedup(
                xm,
                tm,
                wm,
                T,
                E,
                use_fp8=use_fp8,
                round_scale=args.round_scale,
                use_ue8m0=args.ue8m0,
                return_recv_hook=True,
            )
            filler @ filler
            d[4]()
            disp.append(d)
        outs = []
        for d, (xm, tm, wm) in zip(disp, mb):
            c = buffer.low_latency_combine_dedup(
                simulated_expert_out(d[0], H, use_fp8, rank),
                tm,
                d[2],
                return_recv_hook=True,
                compress=args.compress,
            )
            filler @ filler
            c[2]()
            outs.append(c[0])
        for out, (xm, tm, wm) in zip(outs, mb):
            rows = row_errors(
                out, combine_reference(xm, tm, wm, use_fp8, args.round_scale)
            )
            dbo_worst = max(dbo_worst, rows.max().item())
            dbo_bad += int((rows > ROW_TOL).sum().item()) + int(
                torch.isnan(out).any().item()
            )
    if args.stress_dbo:
        results.update(
            dbo_pairs=args.stress_dbo, dbo_bad_rows=dbo_bad, dbo_row_err_max=dbo_worst
        )

    # Bytes that cross to other ranks per step
    x, topk_idx, topk_weights = make_inputs(1000)
    dst = torch.where(topk_idx >= 0, topk_idx // num_local_experts, -1)
    remote = (dst >= 0) & (dst != rank)
    msg = 16 + (H + H // 128 * 4 if use_fp8 else 2 * H)
    distinct_remote = sum(
        int(((dst == r).any(dim=1)).sum().item()) for r in range(num_ranks) if r != rank
    )
    results["remote_msgs_default"] = int(remote.sum().item())
    results["remote_msgs_dedup"] = distinct_remote
    results["remote_dispatch_bytes_default"] = int(remote.sum().item()) * msg
    results["remote_dispatch_bytes_dedup"] = distinct_remote * (msg + 48)
    results["remote_combine_bytes_default"] = int(remote.sum().item()) * 2 * H
    results["remote_combine_bytes_dedup"] = distinct_remote * 2 * H
    if args.compress:
        # Rows this rank's experts send back are partial sums for other ranks'
        # tokens; emulate them from every rank's routing and inputs.
        all_x = torch.empty((num_ranks, T, H), dtype=x.dtype, device="cuda")
        all_t = torch.empty((num_ranks, T, K), dtype=topk_idx.dtype, device="cuda")
        all_w = torch.empty((num_ranks, T, K), dtype=topk_weights.dtype, device="cuda")
        dist.all_gather_into_tensor(all_x, x, group=group)
        dist.all_gather_into_tensor(all_t, topk_idx, group=group)
        dist.all_gather_into_tensor(all_w, topk_weights, group=group)
        raw_b = zip_b = 0
        for src in range(num_ranks):
            if src == rank:
                continue
            ti, tw = all_t[src], all_w[src]
            mine = (ti >= 0) & (ti // num_local_experts == rank)
            sel = mine.any(dim=1)
            x_hat = (
                fp8_roundtrip(all_x[src][sel], args.round_scale)
                if use_fp8
                else all_x[src][sel]
            )
            sc = expert_scale(ti[sel].clamp_min(0)).float()
            y = (
                (x_hat.float().unsqueeze(1) * sc.unsqueeze(-1))
                .to(torch.bfloat16)
                .float()
            )
            w = tw[sel].masked_fill(~mine[sel], 0)
            part = (y * w.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
            raw_b += part.shape[0] * (16 + 2 * H)
            zip_b += int(zip_bytes(part).sum().item())
        results["zip_combine_ratio_est"] = zip_b / max(raw_b, 1)

    ya = None
    if not args.dedup_only:
        ra, _, ha, _, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
        )
        ya = simulated_expert_out(ra, H, use_fp8, rank)
    rb, _, hb, _, _ = buffer.low_latency_dispatch_dedup(
        x,
        topk_idx,
        topk_weights,
        T,
        E,
        use_fp8=use_fp8,
        round_scale=args.round_scale,
        use_ue8m0=args.ue8m0,
    )
    yb = simulated_expert_out(rb, H, use_fp8, rank)
    torch.cuda.synchronize()

    def gemm_then(hook):
        a = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
        a @ a
        hook()

    def run_default(hooks):
        _, _, h, _, hook = buffer.low_latency_dispatch(
            x,
            topk_idx,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
            return_recv_hook=hooks,
        )
        gemm_then(hook) if hooks else None
        _, _, hook = buffer.low_latency_combine(
            ya, topk_idx, topk_weights, h, return_recv_hook=hooks
        )
        gemm_then(hook) if hooks else None

    def run_dedup(hooks, compress=False):
        _, _, h, _, hook = buffer.low_latency_dispatch_dedup(
            x,
            topk_idx,
            topk_weights,
            T,
            E,
            use_fp8=use_fp8,
            round_scale=args.round_scale,
            use_ue8m0=args.ue8m0,
            return_recv_hook=hooks,
        )
        gemm_then(hook) if hooks else None
        _, _, hook = buffer.low_latency_combine_dedup(
            yb, topk_idx, h, return_recv_hook=hooks, compress=compress
        )
        gemm_then(hook) if hooks else None

    class TimedRuntime:
        """Delegates to the C++ runtime and times its LL entry points."""

        def __init__(self, rt):
            self._rt, self.ns = rt, {}

        def __getattr__(self, name):
            f = getattr(self._rt, name)
            if not name.startswith("low_latency_"):
                return f

            def timed(*a, **k):
                t0 = time.perf_counter_ns()
                out = f(*a, **k)
                self.ns[name] = self.ns.get(name, 0) + time.perf_counter_ns() - t0
                return out

            return timed

    def host_breakdown(name, steps=50):
        """Host time of each wrapper call (no sync inside the loop), and the
        part of it spent inside the C++ runtime call."""
        d_ns = c_ns = 0
        real_rt = buffer.runtime
        buffer.runtime = timed_rt = TimedRuntime(real_rt)
        torch.cuda.synchronize()
        for _ in range(steps):
            t0 = time.perf_counter_ns()
            if name == "default":
                _, _, h, _, _ = buffer.low_latency_dispatch(
                    x,
                    topk_idx,
                    T,
                    E,
                    use_fp8=use_fp8,
                    round_scale=args.round_scale,
                    use_ue8m0=args.ue8m0,
                )
            else:
                _, _, h, _, _ = buffer.low_latency_dispatch_dedup(
                    x,
                    topk_idx,
                    topk_weights,
                    T,
                    E,
                    use_fp8=use_fp8,
                    round_scale=args.round_scale,
                    use_ue8m0=args.ue8m0,
                )
            t1 = time.perf_counter_ns()
            if name == "default":
                buffer.low_latency_combine(ya, topk_idx, topk_weights, h)
            else:
                buffer.low_latency_combine_dedup(yb, topk_idx, h)
            t2 = time.perf_counter_ns()
            d_ns += t1 - t0
            c_ns += t2 - t1
        torch.cuda.synchronize()
        buffer.runtime = real_rt
        rt = {k: v / steps / 1e3 for k, v in timed_rt.ns.items()}
        return {
            "host_dispatch_call_us": d_ns / steps / 1e3,
            "host_combine_call_us": c_ns / steps / 1e3,
            "host_runtime_dispatch_us": rt.get(
                (
                    "low_latency_dispatch_dedup"
                    if name == "dedup"
                    else "low_latency_dispatch"
                ),
                0.0,
            ),
            "host_runtime_combine_us": rt.get(
                (
                    "low_latency_combine_dedup"
                    if name == "dedup"
                    else "low_latency_combine"
                ),
                0.0,
            ),
        }

    def profile_host(name, steps=20):
        """Host-side op and CUDA API times of the dispatch call (rank 0 prints)."""
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                if name == "default":
                    _, _, h, _, _ = buffer.low_latency_dispatch(
                        x,
                        topk_idx,
                        T,
                        E,
                        use_fp8=use_fp8,
                        round_scale=args.round_scale,
                        use_ue8m0=args.ue8m0,
                    )
                    buffer.low_latency_combine(ya, topk_idx, topk_weights, h)
                else:
                    _, _, h, _, _ = buffer.low_latency_dispatch_dedup(
                        x,
                        topk_idx,
                        topk_weights,
                        T,
                        E,
                        use_fp8=use_fp8,
                        round_scale=args.round_scale,
                        use_ue8m0=args.ue8m0,
                    )
                    buffer.low_latency_combine_dedup(yb, topk_idx, h)
            torch.cuda.synchronize()
        if rank == 0:
            print(
                f"PROFILE {name}\n"
                + prof.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=14
                ),
                flush=True,
            )
            # Host op sequence of a few steps in the middle of the window
            evs = sorted(
                (
                    e
                    for e in prof.events()
                    if e.device_type.name == "CPU"
                    and e.name != "Activity Buffer Request"
                ),
                key=lambda e: e.time_range.start,
            )
            slow = sorted(
                range(len(evs)), key=lambda i: -evs[i].time_range.elapsed_us()
            )[:12]
            print(
                f"SLOWEST {name} n={len(evs)} "
                + " | ".join(
                    f"#{i}:{evs[i].name}:{evs[i].time_range.elapsed_us():.0f}"
                    for i in sorted(slow)
                ),
                flush=True,
            )
            mid = len(evs) // 2
            print(
                f"SEQUENCE {name} "
                + " | ".join(
                    f"{e.name}:{e.time_range.elapsed_us():.0f}"
                    for e in evs[mid : mid + 40]
                ),
                flush=True,
            )

    order = [("default", run_default), ("dedup", run_dedup)]
    if args.sustain > 0:
        # Congestion plane: idle, default back to back, then the dedup paths
        # paced to the default step (same token rate) and back to back.
        def idle_phase(name):
            dist.barrier()
            c0, w0 = nic_counters(), time.time()
            time.sleep(args.sustain)
            c1, w1 = nic_counters(), time.time()
            dist.barrier()
            return {
                "phase": name,
                "wall_start": w0,
                "wall_end": w1,
                "nic_delta": {k: c1[k] - c0[k] for k in c0 if k in c1},
            }

        dz = partial(run_dedup, compress=True)
        phases = [idle_phase("idle0")]
        d = sustain("default", partial(run_default, False), args.sustain)
        phases.append(d)
        period = torch.tensor([d["step_us"]], device="cuda")
        dist.all_reduce(period, op=dist.ReduceOp.MAX)
        period_s = float(period.item()) / 1e6
        phases.append(idle_phase("idle1"))
        phases.append(
            sustain("dedupz_paced", partial(dz, False), args.sustain, period_s)
        )
        phases.append(
            sustain("dedup_paced", partial(run_dedup, False), args.sustain, period_s)
        )
        phases.append(sustain("dedupz_free", partial(dz, False), args.sustain))
        phases.append(idle_phase("idle2"))
        for ph in phases:
            print(
                "SUSTAIN "
                + json.dumps(dict(ph, rank=rank, tokens=T, hidden=args.hidden)),
                flush=True,
            )
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return
    if args.reverse:
        order.reverse()
    if args.dedup_only:
        order = [("dedup", run_dedup)]
    if args.compress:
        order.append(("dedupz", partial(run_dedup, compress=True)))
    for name, fn in order:
        r = dict(results, impl=name)
        avg_t, min_t, max_t = bench(partial(fn, False))
        r.update(avg_us=avg_t * 1e6, min_us=min_t * 1e6, max_us=max_t * 1e6)
        # Host issue time per step (no sync inside the loop) against wall time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn(False)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        r.update(host_issue_us=(t1 - t0) / 50 * 1e6, host_wall_us=(t2 - t0) / 50 * 1e6)
        r.update(host_breakdown(name))
        if args.profile_host:
            profile_host(name)
        if args.py_profile:
            import cProfile
            import io
            import pstats

            torch.cuda.synchronize()
            prof = cProfile.Profile()
            prof.enable()
            for _ in range(50):
                fn(False)
            prof.disable()
            torch.cuda.synchronize()
            if rank == 0:
                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(14)
                print(f"PYPROFILE {name}\n" + buf.getvalue(), flush=True)
        d_t, c_t = bench_kineto(
            partial(fn, False),
            kernel_names=("dispatch", "combine"),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
        )
        r.update(dispatch_us=d_t * 1e6, combine_us=c_t * 1e6)
        d_t, c_t = bench_kineto(
            partial(fn, True),
            kernel_names=("dispatch", "combine"),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            num_kernels_per_period=2,
        )
        if not isinstance(d_t, (int, float)):
            r.update(
                dispatch_send_us=d_t[0] * 1e6,
                dispatch_recv_us=d_t[1] * 1e6,
                combine_send_us=c_t[0] * 1e6,
                combine_recv_us=c_t[1] * 1e6,
            )
        if args.graph:
            r.update(graph_measure(partial(fn, False)))
        if args.interference:
            r.update(interference_measure(partial(fn, False)))
        print("RESULT " + json.dumps(r), flush=True)

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-checks", type=int, default=3)
    parser.add_argument(
        "--stress", type=int, default=0, help="checked dedup steps with fresh routings"
    )
    parser.add_argument(
        "--stress-dbo",
        type=int,
        default=0,
        help="checked pairs of interleaved micro-batches (two-batch overlap order)",
    )
    parser.add_argument(
        "--compress", action="store_true", help="also run the compressed dedup combine"
    )
    parser.add_argument(
        "--round-scale", action="store_true", help="power-of-2 FP8 scales"
    )
    parser.add_argument(
        "--ue8m0", action="store_true", help="UE8M0 scales (needs --round-scale)"
    )
    parser.add_argument(
        "--inject-fault",
        action="store_true",
        help="corrupt two tokens per stress step (checker control)",
    )
    parser.add_argument("--reverse", action="store_true", help="time dedup first")
    parser.add_argument(
        "--graph", action="store_true", help="also time CUDA graph replay"
    )
    parser.add_argument(
        "--profile-host", action="store_true", help="print host op/API profile"
    )
    parser.add_argument(
        "--py-profile", action="store_true", help="print a cProfile of 50 eager steps"
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="fraction of tokens routed entirely to rank 0's experts (incast)",
    )
    parser.add_argument(
        "--dedup-only",
        action="store_true",
        help="size the buffer for the dedup kernels only and check against all ranks' inputs",
    )
    parser.add_argument(
        "--interference", action="store_true", help="also time a concurrent BF16 GEMM"
    )
    parser.add_argument(
        "--sustain",
        type=int,
        default=0,
        help="seconds per phase of the congestion run (idle / default / dedup paced / free)",
    )
    parser.add_argument(
        "--dispatch-use-fp8",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    main(parser.parse_args())
