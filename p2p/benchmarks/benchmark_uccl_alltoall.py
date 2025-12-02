import argparse
import os

import csv
import numpy as np
import nvtx
import torch
import torch.distributed as dist
import time
from util import setup_seed, get_fcp_comm_plans, sync_all, Metrics
from uccl import collective

"""
Benchmark UCCL Collective for Alltoall
NCCL_IB_GID_INDEX=3 UCCL_ENTROPY=2 UCCL_CHUNK_SIZE_KB=64 torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master_addr=10.21.9.41 --master_port=19999 benchmark_uccl_alltoall.py --block-size 1024 4096 16384 65536 264114 --num-qo-heads 32 --gqa-group-size 4 --head-dim 128 --num-iters 100
"""


def warmup_all2all_check(
    chunk: int = 4 * 1024,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
):
    print("\n🔧 Running warmup_all2all_check...")
    print(chunk)
    # chunk = 1024 * 1024
    if device == None:
        print("No device specified for warmup_all2all_check, using current device")
        device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # time.sleep(20)
    send_chunks_tensor = torch.full(
        (world_size, 1, chunk),
        fill_value=float(rank + 1),
        dtype=dtype,
        device=device,
    )

    recv_chunks_tensor = torch.full(
        (world_size, 1, chunk),
        fill_value=float(888),
        dtype=dtype,
        device=device,
    )

    send_chunks = [send_chunks_tensor[i] for i in range(world_size)]
    recv_chunks = [recv_chunks_tensor[i] for i in range(world_size)]
    # recv_chunks = torch.empty_like(send_chunks, device=device)

    sync_all()

    send_ids, recv_ids = [], []
    registered_tensors = []

    # send_chunks = send_tensor.view(world_size, -1)
    # recv_chunks = recv_tensor.view(world_size, -1)
    # send
    print(f"[Rank {rank}] send_chunks: {send_chunks}")
    collective.register_tensor(send_chunks_tensor)
    print(send_chunks_tensor.data_ptr())
    print(send_chunks_tensor.size())
    collective.register_tensor(recv_chunks_tensor)


    for r in range(world_size):
        if r == rank:
            recv_chunks[r].copy_(send_chunks[r].contiguous())
        else:
            print(f"send_chunks[r]:{send_chunks[r].data_ptr()}")
            print(send_chunks[r].size())
            tid = collective.isend(send_chunks[r], r)

            send_ids.append(tid)
    # sync_all()
    # # recv

    for r in range(world_size):
        if not r == rank:
            # print(f"[Rank {rank}] : posting irecv from rank before {r} {recv_chunks[r].contiguous()}")
            tid = collective.irecv(recv_chunks[r], r)
            # print(f"[Rank {rank}] : posting irecv from rank {r} {recv_chunks[r].contiguous()}")
            recv_ids.append(tid)
        # print(f"[Rank {rank}] : posted irecv from rank {r} {recv_chunks[r].contiguous()}")
    print(f"[Rank {rank}] : waiting for all sends and recvs")
    collective.wait_all(send_ids + recv_ids)
    print(f"[Rank {rank}] : all sends and recvs done")
    sync_all()
    if rank == 0:
        ok = True
        for src in range(world_size):
            expected_val = float(src + 1)
            print(
                f"[Rank {rank}] : checking chunk from rank {src}, expected {expected_val}"
            )
            if torch.allclose(
                recv_chunks[src], torch.full_like(recv_chunks[src], expected_val)
            ):
                print(f"[Rank {rank}] : chunk from rank {src} PASSED")
            else:
                print(recv_chunks)
                print(
                    f"[Rank {rank}] : ERROR in chunk from rank {src}, expected {expected_val}"
                )
                ok = False
            assert torch.allclose(
                recv_chunks[src], torch.full_like(recv_chunks[src], expected_val)
            ), f"[Rank {rank}] : ERROR in chunk from rank {src}, expected {expected_val}"
        print(f"[Rank {rank}] : verification PASSED")
    print(f"[Rank {rank}] : warmup all2all done")
    # for t in registered_tensors:
    #     collective.deregister_tensor(t)
    collective.deregister_tensor(send_chunks_tensor)
    collective.deregister_tensor(recv_chunks_tensor)


def run_fcp_p2p(
    block_size: int,
    num_qo_heads: int,
    gqa_group_size: int,
    head_dim: int,
    num_iters: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> Metrics:
    if device == None:
        device = torch.cuda.current_device()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # init tensors
    num_kv_heads = num_qo_heads // gqa_group_size
    send_tensor = torch.randn(
        block_size,
        2,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    recv_tensor = torch.empty_like(send_tensor, device=device)

    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    sync_all()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = []

    plans = get_fcp_comm_plans(world_size, num_iters)
    start_event.record()
    for idx in range(num_iters):
        send_rank = plans[idx][0][global_rank]
        recv_rank = plans[idx][1][global_rank]

        with nvtx.annotate(f"iter {idx}"):
            send_req = collective.isend(send_tensor, send_rank)
            recv_req = collective.irecv(recv_tensor, recv_rank)
            collective.wait_all([send_req, recv_req])
            print(f"FCP P2P Alltoall Iteration {idx+1}/{num_iters}" + " completed")
    print(f"Completed {num_iters} iterations of FCP P2P Alltoall")
    end_event.record()
    sync_all()
    elapsed_time.append(start_event.elapsed_time(end_event))

    data = Metrics(
        avg_time=np.mean(elapsed_time) / num_iters,
        total_flops=0,
        mem_buckets=np.zeros(world_size),
        flops_buckets=np.zeros(world_size),
        seq_lens=np.zeros(world_size),
    )
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return data


def run_ring_p2p(
    block_size: int,
    num_qo_heads: int,
    gqa_group_size: int,
    head_dim: int,
    num_iters: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> Metrics:
    if device == None:
        device = torch.cuda.current_device()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # init tensors
    num_kv_heads = num_qo_heads // gqa_group_size
    send_tensor = torch.randn(
        block_size,
        2,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    send_tensor.fill_(float(global_rank + 1))
    recv_tensor = torch.empty_like(send_tensor, device=device)
    recv_tensor.fill_(0.0)
    send_rank = (global_rank + 1) % world_size
    recv_rank = (global_rank - 1) % world_size

    sync_all()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = []

    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    send_req = collective.isend(send_tensor, dst=send_rank)
    recv_req = collective.irecv(recv_tensor, src=recv_rank)
    collective.wait_all([send_req, recv_req])

    start_event.record()
    for idx in range(num_iters):
        with nvtx.annotate(f"iter {idx}"):
            send_req = collective.isend(send_tensor, send_rank)
            recv_req = collective.irecv(recv_tensor, recv_rank)
            collective.wait_all([send_req, recv_req])

    end_event.record()
    sync_all()
    elapsed_time.append(start_event.elapsed_time(end_event))

    data = Metrics(
        avg_time=np.mean(elapsed_time) / num_iters,
        total_flops=0,
        mem_buckets=np.zeros(world_size),
        flops_buckets=np.zeros(world_size),
        seq_lens=np.zeros(world_size),
    )

    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return data


def main():
    p = argparse.ArgumentParser(description="Benchmark UCCL Collective for Alltoall")
    p.add_argument("--num-cpus", type=int, default=4, help="#CPU threads for RDMA ops")

    # 修改为支持多个 block-size
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[4*1024],
        help="List of block sizes per GPU",
    )
    p.add_argument("--num-qo-heads", type=int, default=32, help="#QO heads")
    p.add_argument("--gqa-group-size", type=int, default=4, help="GQA group size")
    p.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p.add_argument("--num-iters", type=int, default=10, help="#Iterations")
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64"],
        help="Data type for tensors",
    )
    args = p.parse_args()

    setup_seed(330)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    results = []

    try:
        collective.init_collective(args.num_cpus)
        print(f"[Rank {rank}] UCCL Collective initialized successfully")
        dist.barrier()
        global_rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        for block_size in args.block_sizes:
            print(f"\n🚀 Running benchmark with block_size={block_size}")
            # warmup
            # warmup_all2all_check(
            #     chunk=block_size
            #     * 2
            #     * args.num_qo_heads
            #     // args.gqa_group_size
            #     * args.head_dim,
            #     dtype=dtype,
            #     device=device,
            # )
            print("warmup_all2all_check")
            run_fcp_p2p(
                block_size=block_size,
                num_qo_heads=args.num_qo_heads,
                gqa_group_size=args.gqa_group_size,
                head_dim=args.head_dim,
                num_iters=10,
                dtype=dtype,
                device=device,
            )
            print("run_fcp_p2p")
            data = run_ring_p2p(
                block_size=block_size,
                num_qo_heads=args.num_qo_heads,
                gqa_group_size=args.gqa_group_size,
                head_dim=args.head_dim,
                num_iters=args.num_iters,
                dtype=dtype,
                device=device,
            )

            size_in_bytes = torch.tensor([], dtype=dtype).element_size()
            msg_sz = (
                block_size
                * 2
                * args.num_qo_heads
                // args.gqa_group_size
                * args.head_dim
                * size_in_bytes
                / 1024
                / 1024
            )
            avg_bw = msg_sz / data.avg_time

            if global_rank == 0:
                results.append(
                    {
                        "block_size": block_size,
                        "msg_sz_MB": msg_sz,
                        "avg_bw_GBs": avg_bw,
                    }
                )
                print(
                    f"block_size={block_size//1024}K, msg_sz={msg_sz:.2f}MB, avg_bw={avg_bw:.2f}GB/s"
                )
            sync_all()

        if global_rank == 0:
            print("\n================ Summary =================")
            for r in results:
                print(
                    f"Block={r['block_size']//1024}K | Msg={r['msg_sz_MB']:.2f}MB | BW={r['avg_bw_GBs']:.2f}GB/s"
                )
        print("=========================================\n")

        csv_file = "uccl_benchmark_results.csv"
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "num_qo_heads",
                        "gqa_group_size",
                        "head_dim",
                        "num_iters",
                        "block_size",
                        "msg_size_MB",
                        "avg_bw_GBs",
                    ]
                )

            for r in results:
                writer.writerow(
                    [
                        args.num_qo_heads,
                        args.gqa_group_size,
                        args.head_dim,
                        args.num_iters,
                        r["block_size"],
                        r["msg_sz_MB"],
                        r["avg_bw_GBs"],
                    ]
                )
        print(f"✅ Results saved to {csv_file}")

        sync_all()
    finally:
        collective.finalize_collective()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
