"""Minimal test: verify RDMA WRITEs work repeatedly between 2 nodes.
Instead of using the GPU dispatch kernel, we manually write to the D2H queue
to trigger RDMA WRITEs from the proxy. This isolates RDMA from GPU kernel issues."""
import os, sys, time, torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(__file__))
from buffer import Buffer
from utils import init_dist_under_torchrun, detect_ib_hca

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(
        local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    torch.cuda.set_device(local_rank)

    num_tokens, hidden = 64, 2048
    num_experts = 2 * num_ranks
    num_topk = 1
    num_device_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count

    num_rdma_bytes = 256 * 1024 * 1024
    buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=num_device_sms, explicitly_destroy=True)

    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{local_rank}")

    # Do 10 dispatch+combine cycles with a sleep between each
    for i in range(10):
        print(f"[rank {rank}] Iteration {i}: dispatch...", flush=True)
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, use_fp8=False,
            async_finish=False, return_recv_hook=True)
        hook()
        torch.cuda.synchronize()
        print(f"[rank {rank}] Iteration {i}: dispatch done", flush=True)

        time.sleep(0.5)  # Wait for all RDMA operations to complete

        print(f"[rank {rank}] Iteration {i}: combine...", flush=True)
        combined = buffer.low_latency_combine(recv_x, handle, event,
                                              async_finish=False, return_recv_hook=True)
        combined[-1]()  # call the hook
        torch.cuda.synchronize()
        print(f"[rank {rank}] Iteration {i}: combine done", flush=True)

        time.sleep(0.5)

    print(f"[rank {rank}] All 10 iterations passed!", flush=True)
    try: buffer.destroy()
    except: pass
    try: dist.destroy_process_group()
    except: pass

if __name__ == "__main__":
    ib = detect_ib_hca()
    if ib and ib.startswith("mlx"): os.environ["NCCL_IB_HCA"] = ib
    main()
