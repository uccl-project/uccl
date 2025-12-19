import os
import time
import torch
import torch.distributed as dist


def main():
    # 1) Initialize the process group (using NCCL for GPU).
    dist.init_process_group(backend="nccl")

    # 2) Get global rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3) Determine which GPU to use based on local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 4) Create a tensor on this GPU
    # ResNet-50 fp16 gradient size is 51MB with 50 layers.
    # Each fp16 = 2 bytes.
    MB = 1024**2
    # tensor_size = 51 * MB / 50
    tensor_size = 1000 * MB
    num_floats = int(tensor_size / 2)
    x = torch.full(
        (num_floats,), float(rank), device="cuda", dtype=torch.float16
    )

    # 5) Optional warm-up: do a few all-reduces to stabilize timing
    for _ in range(3):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

    # 6) Time the all-reduce over several iterations
    #    to get an average duration
    num_warmup = 10
    num_iters = 10
    total_time = 0.0

    for i in range(num_warmup + num_iters):
        # Refill x with rank value (so each iteration is consistent)
        x.fill_(float(rank))
        torch.cuda.synchronize()  # ensure fill completes

        start = time.time()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()  # ensure all-reduce finishes
        end = time.time()

        elapsed = end - start
        if i >= num_warmup:
            total_time += elapsed

        if rank == 0:
            print(f"[Iteration {i}] Elapsed: {elapsed*1000:.2f} ms")

    avg_time = total_time / num_iters

    # 7) Calculate approximate bandwidth (per rank, naive measure)
    #    - Our tensor is 1GB on each GPU. We'll define:
    #      data_size_bytes = x.numel() * x.element_size() = 1GB
    #    - Effective bandwidth = data_size_bytes / time
    #    - We'll convert bytes -> gigabytes by dividing by (1024^3).
    data_size_gb = (x.numel() * x.element_size()) / (1024**3 * 1.0)
    bandwidth_gb_s = data_size_gb / avg_time

    # 8) Print results
    if rank == 0:
        print(f"\n--- All-Reduce Performance (Rank {rank}) ---")
        print(f"World size: {world_size}")
        print(f"Tensor size: {data_size_gb:.4f} GB per rank")
        print(f"Average time over {num_iters} iters: {avg_time:.4f} s")
        print(f"Approx. bandwidth (algo): {bandwidth_gb_s:.2f} GB/s (per rank)\n")

    # 9) Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()