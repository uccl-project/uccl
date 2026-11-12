import os
import threading
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters", type=int, default=100, help="Iterations per collective"
    )
    parser.add_argument(
        "--tensor_size",
        type=int,
        default=1024,
        help="Number of float elements per tensor in collective",
    )
    return parser.parse_args()


def setup(rank: int, world_size: int):
    """Initialize default process group."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def create_process_groups(world_size: int):
    """Create two overlapping process groups.

    Group A : All ranks (equivalent to dist.group.WORLD)
    Group B : Even ranks only (0,2,4,...)
    Group C : Odd ranks only (1,3,5,...)
    Returns dictionaries mapping group names to process group handles.
    Ranks not part of a group receive `None` for that handle.
    """
    even_ranks = list(range(0, world_size, 2))
    odd_ranks = list(range(1, world_size, 2))

    # IMPORTANT: Every rank must invoke dist.new_group **in the same order** even
    # if it doesn't belong to that subgroup.  The call returns `None` for ranks
    # outside the provided list but still needs to be executed to keep the
    # rendezvous logic consistent across the job.

    # 1. BIG group (all ranks)
    pg_big = dist.new_group(ranks=list(range(world_size)))

    # 2. EVEN group
    pg_even = dist.new_group(ranks=even_ranks)

    # 3. ODD group
    pg_odd = dist.new_group(ranks=odd_ranks)

    # Return group handles – for ranks not in a subgroup, the handle will be
    # `None`.  This is expected and can be checked with `is None` by callers.
    return {
        "world": dist.group.WORLD,
        "big": pg_big,
        "even": pg_even if dist.get_rank() in even_ranks else None,
        "odd": pg_odd if dist.get_rank() in odd_ranks else None,
    }


def run_allreduce(pg, tensor, iters: int, name: str):
    for _ in range(iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
    print(f"[Rank {dist.get_rank()}] Finished all_reduce on {name}")


def run_broadcast(pg, tensor, iters: int, src: int, name: str):
    for _ in range(iters):
        dist.broadcast(tensor, src=src, group=pg)
    print(f"[Rank {dist.get_rank()}] Finished broadcast on {name}")


def run_group_collectives(groups, iters: int, tensor_size: int):
    """Launch four collectives concurrently using different CUDA/HIP streams.

    We avoid Python threading/GIL issues by:
      1. Launching each collective with ``async_op=True`` so the call is
         non-blocking and returns a ``Work`` handle.
      2. Using a dedicated CUDA/HIP stream per collective, which lets the GPU
         runtime (and RCCL/NCCL) overlap their execution.
      3. Issuing the collectives **in exactly the same order on every rank** to
         guarantee progress.
    """

    device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
    torch.cuda.set_device(device)

    # Allocate tensors
    tensor_world = torch.ones(tensor_size, device=device) * dist.get_rank()
    tensor_even = torch.ones(tensor_size, device=device) * dist.get_rank()
    tensor_odd = torch.ones(tensor_size, device=device) * dist.get_rank()
    tensor_big = torch.ones(tensor_size, device=device) * (
        dist.get_rank() + 0.1
    )

    # Prepare gather list for BIG group all_gather
    if groups["big"] is not None:
        gather_big = [
            torch.empty_like(tensor_big)
            for _ in range(dist.get_world_size(group=groups["big"]))
        ]
    else:
        gather_big = None

    # Create separate streams
    stream_world = torch.cuda.Stream(device=device)
    stream_even = (
        torch.cuda.Stream(device=device) if groups["even"] is not None else None
    )
    stream_odd = (
        torch.cuda.Stream(device=device) if groups["odd"] is not None else None
    )
    stream_big = torch.cuda.Stream(device=device)

    # Launch ops per iteration to avoid overwhelming NCCL with too many queued
    # operations.  Each iteration launches one op per group then waits for all
    # four to finish before moving on.  This pattern keeps ordering identical
    # across ranks and limits outstanding work items.

    for i in range(iters):
        works_iter = []

        # Reset tensors to per-rank values so all_reduce/broadcast results are
        # predictable and verification after the loop remains valid.
        tensor_world.fill_(float(dist.get_rank()))
        tensor_even.fill_(float(dist.get_rank()))
        tensor_odd.fill_(float(dist.get_rank()))
        tensor_big.fill_(float(dist.get_rank()) + 0.1)

        if dist.get_rank() == 0:
            print(f"[Rank {dist.get_rank()}] Starting iteration {i}")

        with torch.cuda.stream(stream_world):
            works_iter.append(
                dist.all_reduce(
                    tensor_world,
                    op=dist.ReduceOp.SUM,
                    group=groups["world"],
                    async_op=True,
                )
            )

        if groups["even"] is not None:
            with torch.cuda.stream(stream_even):
                works_iter.append(
                    dist.broadcast(
                        tensor_even, src=0, group=groups["even"], async_op=True
                    )
                )

        if groups["odd"] is not None:
            with torch.cuda.stream(stream_odd):
                works_iter.append(
                    dist.all_reduce(
                        tensor_odd,
                        op=dist.ReduceOp.SUM,
                        group=groups["odd"],
                        async_op=True,
                    )
                )

        # Re-create gather list each iteration to avoid pointer aliasing issues
        gather_big_iter = [
            torch.empty_like(tensor_big)
            for _ in range(dist.get_world_size(group=groups["big"]))
        ]
        with torch.cuda.stream(stream_big):
            works_iter.append(
                dist.all_gather(
                    gather_big_iter,
                    tensor_big,
                    group=groups["big"],
                    async_op=True,
                )
            )

        # Wait for this iteration's ops to finish
        for w in works_iter:
            w.wait()
        if dist.get_rank() == 0:
            print(
                f"[Rank {dist.get_rank()}] All operations completed for iteration {i}"
            )

        # Ensure every rank finishes all collectives before next iteration. This
        # provides a clear ordering point across the entire world group and
        # helps prevent connection aborts seen with heavy concurrent traffic.
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"[Rank {dist.get_rank()}] Barrier completed for iteration {i}"
            )

        # Ensure GPU kernels have completed before entering global barrier
        torch.cuda.synchronize(device)

    # Ensure all streams are done before verification
    torch.cuda.synchronize(device)

    # Verification for WORLD all_reduce
    expected_sum = sum(range(dist.get_world_size()))
    if torch.allclose(
        tensor_world, torch.full_like(tensor_world, expected_sum)
    ):
        print(
            f"[Rank {dist.get_rank()}] Verification passed for WORLD all_reduce"
        )
    else:
        print(
            f"[Rank {dist.get_rank()}] Verification FAILED for WORLD all_reduce"
        )


def main():
    args = parse_args()

    # Env vars set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)

    groups = create_process_groups(world_size)

    run_group_collectives(groups, args.iters, args.tensor_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
