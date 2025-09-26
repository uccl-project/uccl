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
- recvbuf is reorganized into a dict keyed by (src_rank, local_expert_id),
  grouping all rows belonging to the same expert on a given source.

This makes it possible to simulate expert-parallel dispatch with correct
indexing while benchmarking communication throughput and latency.
"""

import os, math, argparse
import torch
import torch.distributed as dist

def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank

@torch.no_grad()
def timeit(fn, iters=50, warmup=10):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn(); torch.cuda.synchronize(); dist.barrier()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); dist.barrier()
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)  # sec
    return sum(times)/len(times), min(times), max(times)

@torch.no_grad()
def timeit_breakdown(build_fn, comm_fn, reorg_fn, iters=50, warmup=10):
    """Return (build_avg/min/max), (comm_avg/min/max), (reorg_avg/min/max) in seconds."""
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)

    def measure(fn):
        s.record(); fn(); e.record(); e.synchronize()
        return s.elapsed_time(e) / 1000.0  # sec

    # Warmup
    for _ in range(warmup):
        build_fn(); comm_fn(); reorg_fn()
        torch.cuda.synchronize(); dist.barrier()

    bt, ct, rt = [], [], []
    for _ in range(iters):
        torch.cuda.synchronize(); dist.barrier()
        bt.append(measure(build_fn))
        ct.append(measure(comm_fn))
        rt.append(measure(reorg_fn))

    def stats(lst):
        return (sum(lst)/len(lst), min(lst), max(lst))

    return stats(bt), stats(ct), stats(rt)

def splits_from_topk(topk_idx: torch.Tensor, num_experts: int, world: int):
    experts_per_rank = num_experts // world
    dest = (topk_idx // experts_per_rank).reshape(-1)
    counts = torch.bincount(dest, minlength=world)
    return counts

def pack_current_x_linear_fp8_like(current_x: torch.Tensor, header_bytes: int = 16) -> torch.Tensor:
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

    maxabs = torch.amax(torch.abs(xg), dim=-1)                   # [T, G]
    scale = torch.where(maxabs > 0, maxabs / 127.0, torch.ones_like(maxabs))
    inv_scale = 1.0 / scale

    q = torch.clamp(torch.round(xg * inv_scale.unsqueeze(-1)) + 128.0,
                    0, 255).to(torch.uint8)  # [T,G,128]

    row_max = torch.amax(scale, dim=1, keepdim=True) + 1e-12
    scale_u8 = torch.clamp((scale / row_max) * 255.0, 0, 255).round().to(torch.uint8)  # [T,G]
    scale_bytes = scale_u8.unsqueeze(-1).expand(-1, -1, 4).contiguous()               # [T,G,4]

    bytes_per_token = header_bytes + (G * group) + 4 * G
    out = torch.empty((T, bytes_per_token), dtype=torch.uint8, device=current_x.device)
    if header_bytes:
        out[:, :header_bytes].zero_()

    q_bytes = q.contiguous().view(T, G * group)
    out[:, header_bytes:header_bytes + G * group] = q_bytes
    out[:, header_bytes + G * group:] = scale_bytes.view(T, 4 * G)
    return out

def build_sendbuf(args, rank, world, device, current_x, scores, topk_idx):

    num_groups = math.ceil(args.hidden / 128)
    bytes_per_token = args.hidden + 4 * num_groups + args.header_bytes

    if args.use_real_payload:
        packed = pack_current_x_linear_fp8_like(current_x,
                                                header_bytes=args.header_bytes)
    else:
        packed = torch.empty((args.num_tokens, bytes_per_token),
                             dtype=torch.uint8, device=device)

    experts_per_rank = args.num_experts // world
    dest_rank = (topk_idx // experts_per_rank)
    local_expert = (topk_idx % experts_per_rank)  # local expert id on that rank

    per_dest_slices, per_dest_bytes = [], []
    for dst in range(world):
        mask = (dest_rank == dst)   # [T, K]
        counts = mask.sum(dim=1)    # per-token multiplicity for this dst

        if counts.sum().item() == 0:
            per_dest_slices.append(torch.empty((0, bytes_per_token),
                                               dtype=torch.uint8, device=device))
            per_dest_bytes.append(0)
            continue

        # repeat token indices for however many experts they map to on this dst
        idxs = torch.repeat_interleave(torch.arange(args.num_tokens, device=device), counts)
        rows = packed.index_select(0, idxs).clone()

        # also flatten out the matching expert ids for those repeats
        expert_ids = local_expert[mask].reshape(-1).to(torch.int16)
        token_ids = idxs.to(torch.int16)

        # encode metadata into the first 4 header bytes
        # [0:2] = local expert id, [2:4] = token id
        rows[:, 0:2] = expert_ids.view(-1, 1).to(torch.uint8).repeat(1, 2)  # naive little-endian
        rows[:, 2:4] = token_ids.view(-1, 1).to(torch.uint8).repeat(1, 2)

        per_dest_slices.append(rows)
        per_dest_bytes.append(int(rows.shape[0] * bytes_per_token))

    if args.remote_only:
        per_dest_slices[rank] = torch.empty((0, bytes_per_token),
                                            dtype=torch.uint8, device=device)
        per_dest_bytes[rank] = 0

    # flatten into sendbuf
    sendbuf = torch.empty((sum(per_dest_bytes),), dtype=torch.uint8, device=device)
    offset = 0
    for slc, nbytes in zip(per_dest_slices, per_dest_bytes):
        if nbytes:
            sb = slc.view(-1)
            sendbuf[offset:offset+nbytes].copy_(sb)
        offset += nbytes

    # communicate sizes
    send_splits_t = torch.tensor(per_dest_bytes, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(send_splits_t) for _ in range(world)]
    dist.all_gather(gathered, send_splits_t)
    recv_splits = [int(gathered[src][rank].item()) for src in range(world)]
    recvbuf = torch.empty((sum(recv_splits),), dtype=torch.uint8, device=device)

    return sendbuf, recvbuf, per_dest_bytes, recv_splits, bytes_per_token


def reorganize_recvbuf(recvbuf, recv_splits, bytes_per_token, rank, world, header_bytes=16):
    offset = 0
    per_src = {}
    for src in range(world):
        nbytes = recv_splits[src]
        if nbytes == 0:
            per_src[src] = {}
        else:
            rows = nbytes // bytes_per_token
            chunk = recvbuf[offset:offset+nbytes].view(rows, bytes_per_token)

            # Ensure contiguous before viewing
            local_expert_ids = chunk[:, 0:2].contiguous().view(-1).view(-1,2).to(torch.uint8)
            token_ids        = chunk[:, 2:4].contiguous().view(-1).view(-1,2).to(torch.uint8)

            # Convert 2×uint8 → int16 (little endian)
            local_expert_ids = (local_expert_ids[:,0].to(torch.int16) |
                                (local_expert_ids[:,1].to(torch.int16) << 8))
            token_ids = (token_ids[:,0].to(torch.int16) |
                         (token_ids[:,1].to(torch.int16) << 8))

            # Group rows by local expert
            per_src[src] = {}
            for eid in torch.unique(local_expert_ids):
                mask = (local_expert_ids == eid)
                per_src[src][int(eid.item())] = chunk[mask]
        offset += nbytes
    return per_src

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
    ap.add_argument("--include-processing", action="store_true",
                    help="Include preprocessing, comm, reorg with per-stage timings")
    ap.add_argument("--nic-gbps", type=float, default=400.0)
    ap.add_argument("--nic-count", type=int, default=1)
    ap.add_argument("--efficiency", type=float, default=0.92)
    args = ap.parse_args()

    rank, world, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    assert args.num_experts % world == 0
    current_x = torch.randn((args.num_tokens, args.hidden),
                            dtype=torch.bfloat16, device=device) * 0.1
    scores = (torch.randn((args.num_tokens, args.num_experts),
                          dtype=torch.float32, device=device).abs() + 1)
    topk_idx = torch.topk(scores, args.num_topk, dim=-1,
                          largest=True, sorted=True)[1]
    if args.include_processing:
        # Stage variables reused across closures
        sendbuf = recvbuf = None
        in_splits = out_splits = None
        bytes_per_token = 0
        recv_total_holder = {"bytes": 0}

        def build_fn():
            nonlocal sendbuf, recvbuf, in_splits, out_splits, bytes_per_token
            sendbuf, recvbuf, in_splits, out_splits, bytes_per_token = \
                build_sendbuf(args, rank, world, device, current_x, scores, topk_idx)
            recv_total_holder["bytes"] = sum(out_splits)

        def comm_fn():
            dist.all_to_all_single(
                output=recvbuf, input=sendbuf,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits,
                async_op=False,
            )

        def reorg_fn():
            _ = reorganize_recvbuf(recvbuf, out_splits, bytes_per_token, rank, world)

        (b_avg, b_min, b_max), (c_avg, c_min, c_max), (r_avg, r_min, r_max) = \
            timeit_breakdown(build_fn, comm_fn, reorg_fn, iters=args.iters, warmup=args.warmup)

        # Approx overall = sum of stage avgs (mins/maxes are per-stage mins, not same iter)
        avg_s = b_avg + c_avg + r_avg
        min_s = b_min + c_min + r_min
        max_s = b_max + c_max + r_max
        recv_total = recv_total_holder["bytes"]

    else:
        sendbuf, recvbuf, in_splits, out_splits, bytes_per_token = \
            build_sendbuf(args, rank, world, device, current_x, scores, topk_idx)

        def do_all2all():
            dist.all_to_all_single(
                output=recvbuf, input=sendbuf,
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

    # Throughput (comm vs end-to-end)
    bw_comm_GBps = bw_comm_MBps = 0.0
    bw_e2e_GBps = bw_e2e_MBps = 0.0
    if recv_total:
        # Comm-only
        comm_time = c_avg if args.include_processing else avg_s
        bw_comm_MBps = (recv_total / 1e6) / comm_time
        bw_comm_GBps = bw_comm_MBps / 1024.0

        # End-to-end (preprocess + comm + reorg)
        if args.include_processing:
            total_time = b_avg + c_avg + r_avg
            bw_e2e_MBps = (recv_total / 1e6) / total_time
            bw_e2e_GBps = bw_e2e_MBps / 1024.0

    nic_GBps = (args.nic_gbps / 8.0) * args.efficiency
    theo_GBps = nic_GBps * args.nic_count
    ideal_s = (recv_total / (theo_GBps * 1e9)) if recv_total else float("inf")

    if rank == 0:
        print("=== Upper-Bound All-to-All (uint8) ===")
        print(f"use_real_payload={args.use_real_payload}  remote_only={args.remote_only}  include_processing={args.include_processing}")
        print(f"world={world}  experts={args.num_experts}  experts/rank={args.num_experts//world}")
        print(f"bytes_per_token={bytes_per_token} B")
        if recv_total:
            print(f"per-rank recv_total={recv_total/1e6:.2f} MB")

        # Overall timing summary
        print(f"avg={avg_s*1e6:.1f} us  min={min_s*1e6:.1f} us  max={max_s*1e6:.1f} us")

        if recv_total:
            print(f"achieved (comm-only)   ~ {bw_comm_GBps:.2f} GB/s ({bw_comm_MBps:.0f} MB/s)")
            if args.include_processing:
                print(f"achieved (end-to-end) ~ {bw_e2e_GBps:.2f} GB/s ({bw_e2e_MBps:.0f} MB/s)")
            print(f"theoretical link ceiling ≈ {theo_GBps:.1f} GB/s  -> ideal time ≈ {ideal_s*1e6:.1f} us")

        # Breakdown (when processing included)
        if args.include_processing:
            print("\n--- Latency Breakdown ---")
            print(f"Preprocess:  avg={b_avg*1e6:.1f} us  min={b_min*1e6:.1f} us  max={b_max*1e6:.1f} us")
            print(f"All2All:     avg={c_avg*1e6:.1f} us  min={c_min*1e6:.1f} us  max={c_max*1e6:.1f} us")
            print(f"Reorganize:  avg={r_avg*1e6:.1f} us  min={r_min*1e6:.1f} us  max={r_max*1e6:.1f} us")


if __name__ == "__main__":
    main()
