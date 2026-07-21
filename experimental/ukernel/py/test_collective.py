import os
import torch

import ukernel_ccl as dist


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 2)
    local_rank = env_int("LOCAL_RANK", rank)

    torch.cuda.set_device(local_rank)

    pg = dist.init_process_group(
        backend="ukernel",
        rank=rank,
        world_size=world,
        gpu_id=local_rank,
        exchanger_ip=os.getenv("MASTER_ADDR", "127.0.0.1"),
        exchanger_port=env_int("EXCHANGER_PORT", 29600),
    )

    # allreduce (element count must be divisible by world_size)
    x = torch.arange(0, 1024 * world, device="cuda", dtype=torch.float32)
    x = x + rank * 1000
    dist.allreduce(x, group=pg)
    print(f"[rank {rank}] allreduce ok: {x[:8]}")

    # equal-split alltoall (inplace)
    a2a = torch.arange(0, 12 * world, device="cuda", dtype=torch.float32)
    a2a = a2a + rank * 10000
    dist.alltoall(a2a, group=pg)
    print(f"[rank {rank}] alltoall ok: {a2a[:8]}")

    # variable-split alltoallv
    base = 4
    input_splits = [base + ((rank + peer) % 2) for peer in range(world)]
    output_splits = [base + ((src + rank) % 2) for src in range(world)]
    send_v = torch.empty(sum(input_splits), device="cuda", dtype=torch.float32)
    cursor = 0
    for dst, split in enumerate(input_splits):
        send_v[cursor: cursor + split] = rank * 10000 + dst * 100
        cursor += split
    recv_v = torch.empty(sum(output_splits), device="cuda", dtype=torch.float32)
    dist.alltoallv(
        recv_v, send_v,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=pg,
    )
    print(f"[rank {rank}] alltoallv ok: {recv_v[:8]}")

    dist.barrier(group=pg)
    if rank == 0:
        print("all tests passed")


if __name__ == "__main__":
    main()
