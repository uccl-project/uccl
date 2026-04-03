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
        transport="auto",
    )

    x = torch.arange(0, 1024 * world + 1, device="cuda", dtype=torch.float32)
    x = x + rank * 1000
    work = dist.all_reduce(
        x, group=pg, async_op=True, tile_bytes=64 << 10, num_flows=2
    )
    work.wait()
    print(f"[rank {rank}] allreduce ok: {x[:8]}")

    send = torch.arange(0, 12 * world, device="cuda", dtype=torch.float32)
    send = send + rank * 10000
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=pg, tile_bytes=64 << 10, num_flows=2)
    print(f"[rank {rank}] alltoall ok: {recv[:8]}")

    base = 4
    input_splits = [base + ((rank + peer) % 2) for peer in range(world)]
    output_splits = [base + ((src + rank) % 2) for src in range(world)]
    send_v = torch.empty(sum(input_splits), device="cuda", dtype=torch.float32)
    cursor = 0
    for dst, split in enumerate(input_splits):
        send_v[cursor : cursor + split] = rank * 10000 + dst * 100
        cursor += split
    recv_v = torch.empty(sum(output_splits), device="cuda", dtype=torch.float32)
    dist.all_to_all_single(
        recv_v,
        send_v,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=pg,
        tile_bytes=64 << 10,
        num_flows=2,
    )
    print(f"[rank {rank}] alltoallv ok: {recv_v[:8]}")

    dist.barrier(group=pg)


if __name__ == "__main__":
    main()
