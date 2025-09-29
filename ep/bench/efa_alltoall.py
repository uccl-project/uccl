import os
import argparse
import torch
import torch.distributed as dist


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


@torch.no_grad()
def bench_alltoall(msg_size, iters=50, warmup=10):
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    sendbuf = torch.empty(world * msg_size, dtype=torch.uint8, device=device)
    recvbuf = torch.empty_like(sendbuf)
    send_splits = [msg_size] * world
    recv_splits = [msg_size] * world

    for _ in range(warmup):
        dist.all_to_all_single(
            recvbuf,
            sendbuf,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

    torch.cuda.synchronize()
    dist.barrier()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        s.record()
        dist.all_to_all_single(
            recvbuf,
            sendbuf,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)  # sec

    avg = sum(times) / len(times)

    # Per-rank bytes exchanged = sent to (world-1) peers
    total_bytes_per_rank = msg_size * (world - 1)
    bw_per_rank_GBps = (total_bytes_per_rank / 1e9) / avg

    return avg, bw_per_rank_GBps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
            2097152,
            4194304,
            8388608,
            16777216,
            33554432,
            67108864,
        ],
    )
    args = ap.parse_args()

    rank, world, _ = init_dist()

    if rank == 0:
        print(f"=== EFA All-to-All Per-Rank Throughput Test ===")
        print(f"world={world}, iters={args.iters}, warmup={args.warmup}\n")

    for sz in args.sizes:
        avg, bw_per_rank = bench_alltoall(sz, args.iters, args.warmup)
        if rank == 0:
            print(
                f"size={sz:9d} B : avg={avg*1e6:10.1f} us   "
                f"per-rank throughput≈ {bw_per_rank:8.2f} GB/s"
            )


if __name__ == "__main__":
    main()
