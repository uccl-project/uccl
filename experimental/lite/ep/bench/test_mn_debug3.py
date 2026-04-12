"""Minimal 2-iteration multi-node debug test with clean_low_latency_buffer"""
import os, torch, torch.distributed as dist, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from buffer import Buffer
from utils import init_dist_under_torchrun, detect_ib_hca

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    torch.cuda.set_device(local_rank)
    
    num_tokens, hidden = 64, 2048
    num_experts = 2 * num_ranks
    num_topk = 1
    num_local_experts = num_experts // num_ranks
    num_device_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    
    num_rdma_bytes = 256 * 1024 * 1024
    buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=num_device_sms, explicitly_destroy=True)
    
    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{local_rank}")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=f"cuda:{local_rank}")
    cumulative = torch.zeros((num_local_experts,), dtype=torch.int, device=f"cuda:{local_rank}")
    
    for iteration in range(5):
        print(f"[rank {rank}] Starting iteration {iteration}...", flush=True)
        
        # Clean before every iteration starting from iteration 1
        if iteration > 0:
            buffer.clean_low_latency_buffer(x, topk_idx, num_tokens, num_experts, use_fp8=False)
            torch.cuda.synchronize()
        
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, use_fp8=False,
            cumulative_local_expert_recv_stats=cumulative,
            async_finish=False, return_recv_hook=True)
        hook()
        print(f"[rank {rank}] Dispatch {iteration} done", flush=True)
        
        combined_x, ce, ch = buffer.low_latency_combine(
            recv_x, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=True)
        ch()
        torch.cuda.synchronize()
        print(f"[rank {rank}] ✓ Iteration {iteration} complete", flush=True)
    
    print(f"[rank {rank}] All 5 iterations passed!", flush=True)
    try: buffer.destroy()
    except: pass
    try: dist.destroy_process_group()
    except: pass

if __name__ == "__main__":
    ib = detect_ib_hca()
    if ib and ib.startswith("mlx"): os.environ["NCCL_IB_HCA"] = ib
    main()
