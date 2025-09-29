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
def bench_oneway(msg_size, iters=50, warmup=10):
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    sendbuf = torch.empty(msg_size, dtype=torch.uint8, device=device)
    recvbuf = torch.empty_like(sendbuf)

    # Warmup
    for _ in range(warmup):
        if rank == 0:
            dist.send(sendbuf, dst=1)
        elif rank == 1:
            dist.recv(recvbuf, src=0)
    dist.barrier()

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(iters):
        if rank == 0:
            dist.send(sendbuf, dst=1)
        elif rank == 1:
            dist.recv(recvbuf, src=0)
    e.record()
    e.synchronize()

    elapsed = s.elapsed_time(e) / 1000.0  # sec
    avg = elapsed / iters

    # Each iteration transfers msg_size bytes in one direction
    bytes_per_iter = msg_size
    bw_GBps = (bytes_per_iter / 1e9) / avg
    return avg, bw_GBps


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
    assert world == 2, "This benchmark assumes exactly 2 ranks"

    if rank == 0:
        print(f"=== Point-to-Point One-Way Throughput ===")
        print(f"world={world}, iters={args.iters}, warmup={args.warmup}\n")

    for sz in args.sizes:
        avg, bw = bench_oneway(sz, args.iters, args.warmup)
        if rank == 0:
            print(
                f"size={sz:9d} B : avg={avg*1e6:10.1f} us   "
                f"throughput≈ {bw:8.2f} GB/s"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
