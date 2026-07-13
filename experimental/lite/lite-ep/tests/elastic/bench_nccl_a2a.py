# NCCL all-to-all reference for EP dispatch/combine bandwidth.
#
# Two modes match the per-iteration byte volume of DeepEPv2's `legacy_bytes
# = num_tokens * num_topk * hidden * 2`:
#
#   - balanced (default): all_to_all_single splits legacy_bytes evenly across
#     all peers (including self). Closest to a fully load-balanced top-k.
#   - fanout: each rank sends `num_topk * hidden * 2` bytes to each of
#     `num_topk` random peers (or all peers if num_topk >= world_size). More
#     representative of EP fan-out where a token only reaches its top-k
#     destination ranks.
#
# Reports per-rank GB/s = legacy_bytes / latency and the bottleneck (max
# latency across ranks) for direct comparison with our EP `legacy: X GB/s`
# benchmark line.
#
# Usage:
#   $PYTHON_BIN tests/elastic/bench_nccl_a2a.py \
#       --num-tokens=128 --hidden=7168 --num-topk=8 \
#       --warmup=20 --iters=50 --num-processes=$LOCAL_WORLD_SIZE
#
# Multi-node: launch with the same MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE env
# vars used by `tests/elastic/test_ep.py` (init_dist reads them).

import argparse
import os

import torch
import torch.distributed as dist


def init_dist(local_rank: int, num_local_ranks: int):
    """Standalone init (mirrors deep_ep.utils.envs.init_dist) without
    importing deep_ep, so this script does not trigger the NCCL-version
    sanity check that conflicts with LD_PRELOAD."""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def bench_balanced(send_buf, recv_buf, warmup, iters, group):
    for _ in range(warmup):
        dist.all_to_all_single(recv_buf, send_buf, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_to_all_single(recv_buf, send_buf, group=group)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us


def bench_fanout(rank, world_size, peers, msg_elems, warmup, iters, group):
    # Each rank sends `msg_elems` bf16 to each peer in `peers` (its top-k
    # destinations) and receives one msg from each peer that picked it. We
    # use isend/irecv pairs to model EP's fan-out shape.
    send_buf = torch.empty(len(peers) * msg_elems, dtype=torch.bfloat16,
                           device='cuda').normal_()
    recv_buf = torch.empty_like(send_buf)

    # Inverse mapping: who sends to me? In a balanced top-k random
    # assignment, expected in-degree ~= top-k. For a deterministic mirror
    # benchmark, assume every rank picks the same set offset relative to
    # itself: rank r picks peers (r+1, r+2, ..., r+k). Then rank q is
    # picked by (q-1, q-2, ..., q-k).
    in_peers = [(rank - i) % world_size for i in range(1, len(peers) + 1)]

    def round_trip():
        ops = []
        for i, p in enumerate(peers):
            ops.append(dist.P2POp(dist.isend,
                                  send_buf[i * msg_elems:(i + 1) * msg_elems],
                                  p, group=group))
        for i, p in enumerate(in_peers):
            ops.append(dist.P2POp(dist.irecv,
                                  recv_buf[i * msg_elems:(i + 1) * msg_elems],
                                  p, group=group))
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    for _ in range(warmup):
        round_trip()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        round_trip()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us


def worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)

    bytes_per_token = args.hidden * 2  # bf16
    legacy_bytes = args.num_tokens * args.num_topk * bytes_per_token

    if args.mode == 'balanced':
        bytes_per_peer = legacy_bytes // world_size
        elems_per_peer = bytes_per_peer // 2
        send_buf = torch.empty(world_size * elems_per_peer,
                               dtype=torch.bfloat16, device='cuda').normal_()
        recv_buf = torch.empty_like(send_buf)
        if rank == 0:
            print(f'[ref/balanced] world_size={world_size}, '
                  f'legacy_bytes/iter={legacy_bytes:,}, '
                  f'bytes/peer={bytes_per_peer:,}', flush=True)
        t_disp = bench_balanced(send_buf, recv_buf, args.warmup, args.iters, group)
        t_comb = bench_balanced(recv_buf, send_buf, args.warmup, args.iters, group)
    else:  # fanout
        k = min(args.num_topk, world_size - 1)
        peers = [(rank + i) % world_size for i in range(1, k + 1)]
        # Each peer receives num_tokens / world_size tokens worth of data
        # in expectation; model as msg_elems = legacy_bytes / k / 2.
        bytes_per_peer = legacy_bytes // k
        msg_elems = bytes_per_peer // 2
        if rank == 0:
            print(f'[ref/fanout] world_size={world_size}, k={k}, '
                  f'legacy_bytes/iter={legacy_bytes:,}, '
                  f'bytes/peer={bytes_per_peer:,}', flush=True)
        t_disp = bench_fanout(rank, world_size, peers, msg_elems,
                              args.warmup, args.iters, group)
        t_comb = bench_fanout(rank, world_size, peers, msg_elems,
                              args.warmup, args.iters, group)

    bw_disp = legacy_bytes / (t_disp * 1e-6) / 1e9
    bw_comb = legacy_bytes / (t_comb * 1e-6) / 1e9
    print(f'[ref/{args.mode}] rank={rank}/{world_size} | '
          f'dispatch: {bw_disp:6.1f} GB/s @ {t_disp:7.1f} us | '
          f'combine: {bw_comb:6.1f} GB/s @ {t_comb:7.1f} us', flush=True)

    metrics = torch.tensor([t_disp, t_comb], device='cuda')
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        bw_d = legacy_bytes / (metrics[0].item() * 1e-6) / 1e9
        bw_c = legacy_bytes / (metrics[1].item() * 1e-6) / 1e9
        print(f'[ref/{args.mode}] BOTTLENECK | '
              f'dispatch: {bw_d:6.1f} GB/s @ {metrics[0].item():7.1f} us | '
              f'combine: {bw_c:6.1f} GB/s @ {metrics[1].item():7.1f} us',
              flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--mode', choices=['balanced', 'fanout'],
                        default='balanced')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--num-processes', type=int,
                        default=int(os.environ.get('LOCAL_WORLD_SIZE', 1)))
    args = parser.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args),
                                nprocs=args.num_processes)
