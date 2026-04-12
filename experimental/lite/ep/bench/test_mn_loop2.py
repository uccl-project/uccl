"""Copy of test_internode_simple.py with a loop added"""
import torch
import torch.distributed as dist
import time
from buffer import Buffer
import os
import sys
from utils import init_dist, detect_ib_hca


def test_simple_internode(rank, num_ranks, group):
    num_tokens = 512
    hidden = 2048
    num_experts = 3 * num_ranks
    num_topk = 4
    device_index = int(os.environ["LOCAL_RANK"])
    print(f"[simple-test] Running on device {device_index}", flush=True)

    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{device_index}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{device_index}")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=f"cuda:{device_index}")
    num_device_sms = torch.cuda.get_device_properties(device_index).multi_processor_count

    buffer = Buffer(
        group=group,
        num_nvl_bytes=0,
        num_rdma_bytes=int(1e9),
        low_latency_mode=True,
        num_qps_per_rank=num_device_sms,
        allow_nvlink_for_low_latency_mode=True,
        allow_mnnvl=False,
        explicitly_destroy=True,
    )
    print(f"[simple-test] ✓ Buffer created", flush=True)

    for proxy in buffer.proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(num_tokens, hidden, num_experts)
    print(f"[simple-test] ✓ dispatch_recv_data_offset set", flush=True)

    cumulative = torch.zeros(
        (num_experts // num_ranks,), dtype=torch.int, device=f"cuda:{device_index}"
    )

    for iteration in range(10):
        recv_x, recv_count, handle, event, dispatch_hook = buffer.low_latency_dispatch(
            x=x, topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            use_fp8=False, round_scale=False, use_ue8m0=False,
            cumulative_local_expert_recv_stats=cumulative,
            async_finish=False, return_recv_hook=True,
        )
        dispatch_hook()

        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            x=recv_x, topk_idx=topk_idx, topk_weights=topk_weights,
            handle=handle,
            use_logfmt=False, zero_copy=False,
            async_finish=False, return_recv_hook=True,
        )
        combine_hook()
        torch.cuda.synchronize()
        print(f"[simple-test] ✓ Iteration {iteration}", flush=True)

    print(f"[simple-test] ✓ All 10 iterations passed!", flush=True)
    time.sleep(2)

    try:
        buffer.destroy()
    except:
        pass


if __name__ == "__main__":
    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):
        os.environ["NCCL_IB_HCA"] = ib_dev
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    try:
        test_simple_internode(rank, num_ranks, group)
    except:
        import traceback
        traceback.print_exc()
    finally:
        try:
            dist.destroy_process_group()
        except:
            pass
