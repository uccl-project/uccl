"""Eight-process PyTorch NCCL interception check; run via make check-torch."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    for count in (8, 1024, 16384):
        value = torch.full((count,), rank + 1, dtype=torch.bfloat16, device="cuda")
        gathered = torch.empty((count * 8,), dtype=torch.bfloat16, device="cuda")
        dist.all_gather_into_tensor(gathered, value)
        for source in range(8):
            expected = source + 1
            assert torch.all(gathered[source * count : (source + 1) * count] == expected)

        contribution = torch.full(
            (count * 8,), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        reduced = torch.empty((count,), dtype=torch.bfloat16, device="cuda")
        dist.reduce_scatter_tensor(reduced, contribution, op=dist.ReduceOp.SUM)
        assert torch.all(reduced == 36)

        value = torch.full((count,), rank + 1, dtype=torch.bfloat16, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        assert torch.all(value == 36)

    # Unsupported dtype must pass through the real NCCL library.
    fp32 = torch.full((8,), rank + 1, dtype=torch.float32, device="cuda")
    fp32_gathered = torch.empty((64,), dtype=torch.float32, device="cuda")
    dist.all_gather_into_tensor(fp32_gathered, fp32)
    for source in range(8):
        assert torch.all(fp32_gathered[source * 8 : (source + 1) * 8] == source + 1)

    dist.all_reduce(fp32, op=dist.ReduceOp.SUM)
    assert torch.all(fp32 == 36)

    torch.cuda.synchronize()
    dist.destroy_process_group()
    print(f"rank {rank}: torch AG/RS/AR passed", flush=True)


if __name__ == "__main__":
    main()
