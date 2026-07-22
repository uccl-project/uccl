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

    # async API smoke: submit / poll / wait / status / release
    x2 = torch.full((256 * world,), float(rank + 1),
                    device="cuda", dtype=torch.float32)
    h = pg.allreduce_submit(x2)
    while not pg.poll(h):
        pass
    assert pg.wait(h), f"[rank {rank}] async allreduce failed"
    assert pg.status(h) == dist.CollectiveOpStatus.Completed
    pg.release(h)
    expected_sum = float(world * (world + 1) // 2)
    assert bool((x2 == expected_sum).all().item()), (
        f"[rank {rank}] async allreduce mismatch: {x2[:8]} != {expected_sum}")
    print(f"[rank {rank}] async allreduce ok: {x2[:8]}")

    dist.barrier(group=pg)
    if rank == 0:
        print("all tests passed")


if __name__ == "__main__":
    main()
