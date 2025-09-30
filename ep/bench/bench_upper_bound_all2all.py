"""
=== Benchmark: Upper-Bound All-to-All with Metadata Headers ===

This script measures upper-bound all-to-all bandwidth and latency
using NCCL's `all_to_all_single`, while simulating MoE-style token
dispatch. It includes optional breakdown of preprocessing, communication,
and reorganization stages.

----------------------------------------------------------------------
Message Format (per row in sendbuf / recvbuf):

Each row represents a single (token → expert) routing decision and is laid
out as:

    [Header Bytes][Quantized Payload Bytes][Scale Bytes]

1. Header Bytes (>= 4 bytes)
   - Bytes [0:2] : local_expert_id (uint16, little-endian)
       • Expert index relative to the destination rank
       • Computed as: expert_idx % experts_per_rank
   - Bytes [2:4] : token_id (uint16, little-endian)
       • Global token index at the sender (0 ≤ token_id < num_tokens)
   - Any extra header bytes are zero-padded (reserved for future use).

   Together, the header allows reconstruction of (src_rank, local_expert_id, token_id).

2. Quantized Payload Bytes
   - The hidden vector for the token, quantized block-wise into uint8
     using a fake FP8-like scheme.
   - Each group = 128 elements, quantized separately.
   - Layout per row = num_groups * 128 bytes.

3. Scale Bytes
   - For each 128-element group, 4 scale bytes are reserved.
   - In this benchmark, each scale is normalized to [0,255], cast to uint8,
     and repeated 4x.
   - Layout per row = num_groups * 4 bytes.

----------------------------------------------------------------------
At the receiver:
- recvbuf is reorganized into CSR-like groups keyed by (src_rank, local_expert_id),
  so you can slice each group without Python dictionaries/lists.

export MASTER_ADDR=10.1.227.34
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --nnodes=2 --nproc_per_node=8  --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT   bench_upper_bound_all2all.py   --num-tokens 4096 --hidden 7168 --num-experts 256 --num-topk 8   --use-real-payload --include-processing --verify --remote-only

torchrun --nnodes=2 --nproc_per_node=8   --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT   bench_upper_bound_all2all.py   --num-tokens 4096 --hidden 7168 --num-experts 256 --num-topk 8   --use-real-payload --include-processing --verify --remote-only

"""

import os, math, argparse, socket, hashlib
import torch
import torch.distributed as dist


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


@torch.no_grad()
def timeit(fn, iters=50, warmup=10):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
        dist.barrier()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)  # sec
    return sum(times) / len(times), min(times), max(times)


@torch.no_grad()
def timeit_breakdown(build_fn, comm_fn, reorg_fn, iters=50, warmup=10):
    """Return (build_avg/min/max), (comm_avg/min/max), (reorg_avg/min/max) in seconds."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    def measure(fn):
        s.record()
        fn()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) / 1000.0  # sec

    # Warmup
    for _ in range(warmup):
        build_fn()
        comm_fn()
        reorg_fn()
        torch.cuda.synchronize()
        dist.barrier()

    bt, ct, rt = [], [], []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        bt.append(measure(build_fn))
        ct.append(measure(comm_fn))
        rt.append(measure(reorg_fn))

    def stats(lst):
        return (sum(lst) / len(lst), min(lst), max(lst))

    return stats(bt), stats(ct), stats(rt)


def splits_from_topk(topk_idx: torch.Tensor, num_experts: int, world: int):
    experts_per_rank = num_experts // world
    dest = (topk_idx // experts_per_rank).reshape(-1)
    counts = torch.bincount(dest, minlength=world)
    return counts


def pack_current_x_linear_fp8_like(
    current_x: torch.Tensor, header_bytes: int = 16
) -> torch.Tensor:
    """Pack rows of current_x into bytes with fake FP8-like layout."""
    assert current_x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    x = current_x.to(torch.float32)
    T, H = x.shape
    group = 128
    G = (H + group - 1) // group
    pad = G * group - H
    if pad:
        x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)

    # [T, G, 128]
    xg = x.view(T, G, group)

    maxabs = torch.amax(torch.abs(xg), dim=-1)  # [T, G]
    scale = torch.where(maxabs > 0, maxabs / 127.0, torch.ones_like(maxabs))
    inv_scale = 1.0 / scale

    q = torch.clamp(torch.round(xg * inv_scale.unsqueeze(-1)) + 128.0, 0, 255).to(
        torch.uint8
    )  # [T,G,128]

    row_max = torch.amax(scale, dim=1, keepdim=True) + 1e-12
    scale_u8 = (
        torch.clamp((scale / row_max) * 255.0, 0, 255).round().to(torch.uint8)
    )  # [T,G]
    scale_bytes = scale_u8.unsqueeze(-1).expand(-1, -1, 4).contiguous()  # [T,G,4]

    bytes_per_token = header_bytes + (G * group) + 4 * G
    out = torch.empty((T, bytes_per_token), dtype=torch.uint8, device=current_x.device)
    if header_bytes:
        out[:, :header_bytes].zero_()

    q_bytes = q.contiguous().view(T, G * group)
    out[:, header_bytes : header_bytes + G * group] = q_bytes
    out[:, header_bytes + G * group :] = scale_bytes.view(T, 4 * G)
    return out


def _bytes_per_token_from_hidden(hidden: int, header_bytes: int) -> int:
    group = 128
    G = (hidden + group - 1) // group
    return header_bytes + G * (group + 4)


def build_sendbuf(args, rank, world, device, current_x, scores, topk_idx):
    # Match the packer layout
    bytes_per_token = _bytes_per_token_from_hidden(args.hidden, args.header_bytes)

    if args.use_real_payload:
        packed = pack_current_x_linear_fp8_like(
            current_x, header_bytes=args.header_bytes
        )
    else:
        packed = torch.empty(
            (args.num_tokens, bytes_per_token), dtype=torch.uint8, device=device
        )
        if args.header_bytes:
            packed[:, : args.header_bytes].zero_()

    experts_per_rank = args.num_experts // world
    dest_rank = topk_idx // experts_per_rank
    local_expert = topk_idx % experts_per_rank  # local expert id on that rank

    per_dest_slices, per_dest_bytes = [], []
    for dst in range(world):
        mask = dest_rank == dst  # [T, K]
        counts = mask.sum(dim=1)  # per-token multiplicity for this dst

        if counts.sum().item() == 0:
            per_dest_slices.append(
                torch.empty((0, bytes_per_token), dtype=torch.uint8, device=device)
            )
            per_dest_bytes.append(0)
            continue

        # repeat token indices for however many experts they map to on this dst
        idxs = torch.repeat_interleave(
            torch.arange(args.num_tokens, device=device), counts
        )
        rows = packed.index_select(0, idxs).clone()

        # flatten matching local expert ids and token ids
        expert_ids = local_expert[mask].reshape(-1).to(torch.int16)
        token_ids = idxs.to(torch.int16)

        # encode metadata into the first 4 header bytes (little endian)
        if args.num_tokens >= 65536:
            raise ValueError(
                "token_id won't fit in uint16; raise header size or change format."
            )
        eid = expert_ids.to(torch.int32)
        tid = token_ids.to(torch.int32)
        rows[:, 0] = (eid & 0xFF).to(torch.uint8)
        rows[:, 1] = ((eid >> 8) & 0xFF).to(torch.uint8)
        rows[:, 2] = (tid & 0xFF).to(torch.uint8)
        rows[:, 3] = ((tid >> 8) & 0xFF).to(torch.uint8)

        per_dest_slices.append(rows)
        per_dest_bytes.append(int(rows.shape[0] * bytes_per_token))

    if args.remote_only:
        per_dest_slices[rank] = torch.empty(
            (0, bytes_per_token), dtype=torch.uint8, device=device
        )
        per_dest_bytes[rank] = 0

    # flatten into sendbuf
    sendbuf = torch.empty((sum(per_dest_bytes),), dtype=torch.uint8, device=device)
    offset = 0
    for slc, nbytes in zip(per_dest_slices, per_dest_bytes):
        if nbytes:
            sb = slc.view(-1)
            sendbuf[offset : offset + nbytes].copy_(sb)
        offset += nbytes

    # communicate sizes
    send_splits_t = torch.tensor(per_dest_bytes, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(send_splits_t) for _ in range(world)]
    dist.all_gather(gathered, send_splits_t)
    recv_splits = [int(gathered[src][rank].item()) for src in range(world)]
    recvbuf = torch.empty((sum(recv_splits),), dtype=torch.uint8, device=device)

    return (
        sendbuf,
        recvbuf,
        per_dest_bytes,
        recv_splits,
        bytes_per_token,
        dest_rank,
        local_expert,
    )


@torch.no_grad()
def reorganize_recvbuf_fast(
    recvbuf: torch.Tensor,
    recv_splits: list[int],
    bytes_per_token: int,
    world: int,
    header_bytes: int = 16,
    return_token_ids: bool = False,
):
    """
    Returns a compact grouped view of rows by (src_rank, local_expert_id),
    without Python loops.

    Output (all on GPU):
      rows_sorted:   [N, bytes_per_token] uint8
      group_ptr:     [G+1] int32      (CSR pointer; slice i as rows_sorted[group_ptr[i]:group_ptr[i+1]])
      group_src:     [G]   int16      (source rank per group)
      group_eid:     [G]   int16      (local expert id per group)
      token_ids_sorted (optional): [N] int16 (aligned with rows_sorted)
    """
    device = recvbuf.device
    assert header_bytes >= 4, "Need at least 4 header bytes (eid[0:2], tok[2:4])."

    # Per-source row counts
    recv_splits = [int(x) for x in recv_splits]
    rows_per_src = [s // bytes_per_token for s in recv_splits]
    total_rows = sum(rows_per_src)
    if total_rows == 0:
        empty = torch.empty((0, bytes_per_token), dtype=torch.uint8, device=device)
        gp = torch.zeros(1, dtype=torch.int32, device=device)
        gs = torch.empty((0,), dtype=torch.int16, device=device)
        ge = torch.empty((0,), dtype=torch.int16, device=device)
        if return_token_ids:
            tids = torch.empty((0,), dtype=torch.int16, device=device)
            return empty, gp, gs, ge, tids
        return empty, gp, gs, ge

    # View as rows
    rows = recvbuf.view(-1, bytes_per_token).contiguous()  # [N, B]

    # Build src_rank vector for all rows (ragged → flat)
    src_ids = torch.arange(world, device=device, dtype=torch.int32)
    src_vec = torch.repeat_interleave(
        src_ids, torch.tensor(rows_per_src, device=device, dtype=torch.int32)
    )  # [N]

    # Parse 4-byte header: [0:2] local_expert_id (u16 LE), [2:4] token_id (u16 LE)
    h = rows[:, :4].contiguous()  # [N, 4] uint8
    eid = h[:, 0].to(torch.int32) | (h[:, 1].to(torch.int32) << 8)  # [N]
    tid = h[:, 2].to(torch.int32) | (h[:, 3].to(torch.int32) << 8)  # [N]

    # Grouping key = (src << 16) | eid
    keys = (src_vec << 16) | (eid & 0xFFFF)

    # Sort once by key (stable kw only on newer torch)
    try:
        order = torch.argsort(keys, stable=True)
    except TypeError:
        order = torch.argsort(keys)

    rows_sorted = rows.index_select(0, order)
    keys_sorted = keys.index_select(0, order)
    src_sorted = src_vec.index_select(0, order).to(torch.int16)
    eid_sorted = (eid.index_select(0, order) & 0xFFFF).to(torch.int16)
    if return_token_ids:
        tid_sorted = (tid.index_select(0, order) & 0xFFFF).to(torch.int16)

    # Unique groups and counts
    uniq_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    group_ptr = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=device)
    group_ptr[0] = 0
    group_ptr[1:] = torch.cumsum(counts.to(torch.int32), dim=0)

    # Decode uniq_keys → (src, local_eid) per group
    group_src = (uniq_keys >> 16).to(torch.int16)
    group_eid = (uniq_keys & 0xFFFF).to(torch.int16)

    if return_token_ids:
        return rows_sorted, group_ptr, group_src, group_eid, tid_sorted
    else:
        return rows_sorted, group_ptr, group_src, group_eid


def _compute_local_export_counts(
    topk_idx: torch.Tensor,
    num_experts: int,
    world: int,
    experts_per_rank: int,
    remote_only: bool,
    rank: int,
    device,
):
    """
    Returns a [world, experts_per_rank] int64 tensor of counts that THIS SOURCE will send to each destination.
    Row d is a histogram of local_eid for destination d.
    """
    dest_rank = topk_idx // experts_per_rank
    local_eid = topk_idx % experts_per_rank

    counts = torch.zeros((world, experts_per_rank), dtype=torch.int64, device=device)
    for d in range(world):
        mask = dest_rank == d
        if mask.any():
            eids = local_eid[mask].reshape(-1).long()
            ones = torch.ones_like(eids, dtype=torch.int64)
            counts[d].index_add_(0, eids, ones)
    if remote_only:
        counts[rank].zero_()  # source won't send to itself in remote-only mode
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--header-bytes", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--remote-only", action="store_true")
    ap.add_argument("--use-real-payload", action="store_true")
    ap.add_argument(
        "--include-processing",
        action="store_true",
        help="Include preprocessing, comm, reorg with per-stage timings",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check received headers and per-(src,eid) counts against source-side expectations",
    )
    ap.add_argument(
        "--debug-print",
        action="store_true",
        help="Print per-(src,eid) token counts on each rank",
    )
    ap.add_argument("--nic-gbps", type=float, default=400.0)
    ap.add_argument("--nic-count", type=int, default=1)
    ap.add_argument("--efficiency", type=float, default=0.92)
    args = ap.parse_args()

    rank, world, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    assert args.num_experts % world == 0
    experts_per_rank = args.num_experts // world

    current_x = (
        torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
        * 0.1
    )
    scores = (
        torch.randn(
            (args.num_tokens, args.num_experts), dtype=torch.float32, device=device
        ).abs()
        + 1
    )
    topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=True)[1]

    if args.include_processing:
        # Stage variables reused across closures
        sendbuf = recvbuf = None
        in_splits = out_splits = None
        bytes_per_token = 0
        recv_total_holder = {"bytes": 0}

        def build_fn():
            nonlocal sendbuf, recvbuf, in_splits, out_splits, bytes_per_token, dest_rank, local_expert
            (
                sendbuf,
                recvbuf,
                in_splits,
                out_splits,
                bytes_per_token,
                dest_rank,
                local_expert,
            ) = build_sendbuf(args, rank, world, device, current_x, scores, topk_idx)
            recv_total_holder["bytes"] = sum(out_splits)

        def comm_fn():
            dist.all_to_all_single(
                output=recvbuf,
                input=sendbuf,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits,
                async_op=False,
            )

        def reorg_fn():
            # Do the grouping pass (discarded for timing; we re-run once after timing if verifying)
            _ = reorganize_recvbuf_fast(
                recvbuf,
                out_splits,
                bytes_per_token,
                world=world,
                header_bytes=args.header_bytes,
                return_token_ids=False,
            )

        (b_avg, b_min, b_max), (c_avg, c_min, c_max), (r_avg, r_min, r_max) = (
            timeit_breakdown(
                build_fn, comm_fn, reorg_fn, iters=args.iters, warmup=args.warmup
            )
        )

        # Approx overall = sum of stage avgs (mins/maxes are per-stage mins, not same iter)
        avg_s = b_avg + c_avg + r_avg
        min_s = b_min + c_min + r_min
        max_s = b_max + c_max + r_max
        recv_total = recv_total_holder["bytes"]

    else:
        (
            sendbuf,
            recvbuf,
            in_splits,
            out_splits,
            bytes_per_token,
            dest_rank,
            local_expert,
        ) = build_sendbuf(args, rank, world, device, current_x, scores, topk_idx)

        def do_all2all():
            dist.all_to_all_single(
                output=recvbuf,
                input=sendbuf,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits,
                async_op=False,
            )

        avg_s, min_s, max_s = timeit(do_all2all, args.iters, args.warmup)
        recv_total = sum(out_splits)
        # dummy stage stats for print compatibility
        b_avg = b_min = b_max = 0.0
        c_avg, c_min, c_max = avg_s, min_s, max_s
        r_avg = r_min = r_max = 0.0

    # ---- Throughput (decimal units), with network-only and NIC-only (inter-node) breakdowns ----
    if recv_total:
        # Comm-only timing
        comm_time = c_avg if args.include_processing else avg_s

        # Bytes breakdown on this destination rank
        local_bytes = out_splits[rank] if rank < len(out_splits) else 0
        remote_bytes = int(recv_total - local_bytes)  # excludes self

        # Decimal throughput (GB/s = bytes/s / 1e9)
        total_GBps_dec = (recv_total / comm_time) / 1e9
        remote_GBps_dec = (remote_bytes / comm_time) / 1e9
        local_GBps_dec = (local_bytes / comm_time) / 1e9

        # End-to-end (decimal, total bytes)
        if args.include_processing:
            total_time = b_avg + c_avg + r_avg
            e2e_total_GBps_dec = (recv_total / total_time) / 1e9
        else:
            e2e_total_GBps_dec = 0.0

        # Theoretical NIC ceiling (decimal) and ideal times
        theo_GBps_dec = (args.nic_gbps / 8.0) * args.efficiency * args.nic_count
        ideal_remote_s_dec = (
            (remote_bytes / (theo_GBps_dec * 1e9)) if remote_bytes > 0 else float("inf")
        )

        # ----- NIC-only (inter-node) bytes: exclude self + same-host peers -----
        # Identify hosts
        my_host = socket.gethostname()
        my_host_id = int.from_bytes(
            hashlib.sha1(my_host.encode()).digest()[:8], "little"
        )
        host_id_tensor = torch.tensor([my_host_id], dtype=torch.long, device=device)
        gathered_host_ids = [torch.zeros_like(host_id_tensor) for _ in range(world)]
        dist.all_gather(gathered_host_ids, host_id_tensor)
        host_ids = [int(t.item()) for t in gathered_host_ids]  # host id per src rank

        # Bytes from each source rank into THIS destination
        per_src_bytes = [
            int(b) for b in out_splits
        ]  # out_splits[src] == bytes from src→this rank

        inter_node_bytes = 0
        for src in range(world):
            if src == rank:
                continue  # exclude self
            if host_ids[src] != my_host_id:
                inter_node_bytes += per_src_bytes[src]

        nic_only_GBps_dec = (
            (inter_node_bytes / comm_time) / 1e9 if inter_node_bytes > 0 else 0.0
        )
    else:
        total_GBps_dec = remote_GBps_dec = local_GBps_dec = 0.0
        e2e_total_GBps_dec = 0.0
        theo_GBps_dec = (args.nic_gbps / 8.0) * args.efficiency * args.nic_count
        ideal_remote_s_dec = float("inf")
        nic_only_GBps_dec = 0.0

    # Optional verification / debug
    if recv_total and (args.verify or args.debug_print):
        rows_sorted, group_ptr, group_src, group_eid, tok_ids = reorganize_recvbuf_fast(
            recvbuf,
            out_splits,
            bytes_per_token,
            world=world,
            header_bytes=args.header_bytes,
            return_token_ids=True,
        )

        # Observed per-(src,eid) counts from received buffer
        group_counts = (group_ptr[1:] - group_ptr[:-1]).to(torch.int64)  # [G]
        flat_idx = group_src.to(torch.int64) * experts_per_rank + group_eid.to(
            torch.int64
        )
        obs_counts_flat = torch.zeros(
            world * experts_per_rank, dtype=torch.int64, device=device
        )
        obs_counts_flat.index_add_(0, flat_idx, group_counts)
        obs_counts = obs_counts_flat.view(world, experts_per_rank)  # [world, EPR]

        # Expected counts from source-side topk for this dest rank
        local_export = _compute_local_export_counts(
            topk_idx,
            args.num_experts,
            world,
            experts_per_rank,
            remote_only=args.remote_only,
            rank=rank,
            device=device,
        )  # [world, EPR] from THIS source
        # Gather all sources' export matrices
        gathered = [torch.empty_like(local_export) for _ in range(world)]
        dist.all_gather(gathered, local_export)
        # expected on THIS destination rank = take row [rank] from each source's export
        expected = torch.stack([g[rank] for g in gathered], dim=0)

        if args.debug_print and rank == 0:
            print(
                f"[rank{rank}] Observed counts per (src,eid): shape={tuple(obs_counts.shape)}"
            )
            to_show_src = min(world, 4)
            to_show_eid = min(experts_per_rank, 8)
            print("[rank0] obs_counts sample:")
            print(obs_counts[:to_show_src, :to_show_eid].cpu())
            print(
                f"[rank{rank}] Expected counts per (src,eid): shape={tuple(expected.shape)}"
            )
            print("[rank0] expected sample:")
            print(expected[:to_show_src, :to_show_eid].cpu())

        # Header sanity checks
        eid_from_rows = rows_sorted[:, 0].to(torch.int32) | (
            rows_sorted[:, 1].to(torch.int32) << 8
        )
        eid_expanded = group_eid.repeat_interleave(group_counts)
        assert torch.equal(
            eid_from_rows.to(torch.int16), eid_expanded
        ), f"[rank{rank}] Header EID mismatch against grouped EIDs."
        assert (tok_ids >= 0).all() and (
            tok_ids < args.num_tokens
        ).all(), f"[rank{rank}] Token IDs out of range."

        # Final equality check
        if not torch.equal(obs_counts, expected):
            diff = (obs_counts - expected).abs()
            max_diff = int(diff.max().item())
            total_diff = int(diff.sum().item())
            mism_idx = (diff > 0).nonzero(as_tuple=False)
            preview = mism_idx[:10].tolist()
            raise AssertionError(
                f"[rank{rank}] VERIFY FAILED: observed vs expected per-(src,eid) counts differ. "
                f"max_diff={max_diff}, sum_diff={total_diff}, first_bad_indices={preview}"
            )
        else:
            if rank == 0:
                print("[VERIFY] Per-(src,eid) counts match across ranks. ✔️")

    # --------- Prints ---------
    if rank == 0:
        print("=== Upper-Bound All-to-All (uint8) ===")
        print(
            f"use_real_payload={args.use_real_payload}  remote_only={args.remote_only}  include_processing={args.include_processing}"
        )
        print(
            f"world={world}  experts={args.num_experts}  experts/rank={experts_per_rank}"
        )
        print(f"bytes_per_token={bytes_per_token} B")
        if recv_total:
            print(f"per-rank recv_total={recv_total/1e6:.2f} MB (decimal)")

        # Overall timing summary
        print(f"avg={avg_s*1e6:.1f} us  min={min_s*1e6:.1f} us  max={max_s*1e6:.1f} us")

        if recv_total:
            print(
                f"achieved (comm-only, TOTAL bytes incl. self) ~ {total_GBps_dec:.2f} GB/s (decimal)"
            )
            print(
                f"achieved (NIC-ONLY, inter-node only)        ~ {nic_only_GBps_dec:.2f} GB/s (decimal)"
            )
            if args.include_processing:
                print(
                    f"achieved (end-to-end, TOTAL bytes)         ~ {e2e_total_GBps_dec:.2f} GB/s (decimal)"
                )
            print(
                f"theoretical NIC ceiling (decimal)           ~ {theo_GBps_dec:.2f} GB/s"
            )
            print(
                f"theoretical network-only time          ~ {ideal_remote_s_dec*1e6:.1f} us"
            )

        # Breakdown (when processing included)
        if recv_total and args.include_processing:
            print("\n--- Latency Breakdown ---")
            print(
                f"Preprocess:  avg={b_avg*1e6:.1f} us  min={b_min*1e6:.1f} us  max={b_max*1e6:.1f} us"
            )
            print(
                f"All2All:     avg={c_avg*1e6:.1f} us  min={c_min*1e6:.1f} us  max={c_max*1e6:.1f} us"
            )
            print(
                f"Reorganize:  avg={r_avg*1e6:.1f} us  min={r_min*1e6:.1f} us  max={r_max*1e6:.1f} us"
            )
            print(
                f"Overall:     avg={avg_s*1e6:.1f} us  min={min_s*1e6:.1f} us  max={max_s*1e6:.1f} us"
            )

    # Clean shutdown
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
