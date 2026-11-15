"""Stock ProcessGroupNCCL fallback, split groups, P2P, and graph replay."""

import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    for dtype in (torch.float32, torch.bfloat16):
        value = torch.full((33,), rank + 1, device="cuda", dtype=dtype)
        dist.all_reduce(value)
        assert torch.all(value == world * (world + 1) // 2)
    group = dist.new_group(list(range(world)), backend="nccl")
    value = torch.full((33,), rank + 1, device="cuda", dtype=torch.float32)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize()
    assert torch.all(value == world * (world + 1) // 2)
    send = torch.tensor([rank], device="cuda")
    recv = torch.empty_like(send)
    requests = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send, (rank + 1) % world, group),
        dist.P2POp(dist.irecv, recv, (rank - 1) % world, group),
    ])
    for request in requests:
        request.wait()
    assert recv.item() == (rank - 1) % world
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        dist.all_reduce(value, group=group)
    for i in range(3):
        value.fill_(rank + 1 + i)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(value == world * (world + 1) // 2 + world * i)
    del graph
    torch.cuda.synchronize()
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    if rank == 0:
        print(f"PASS: {world}-rank fallback, split communicator, grouped P2P, native graph replay")


if __name__ == "__main__":
    main()
