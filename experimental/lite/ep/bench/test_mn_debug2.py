"""Minimal 2-iteration multi-node debug test
Focus: identify exactly which operation fails and on which side"""
import os, torch, torch.distributed as dist, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from buffer import Buffer
from utils import init_dist_under_torchrun, detect_ib_hca

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    torch.cuda.set_device(local_rank)
    
    num_tokens, hidden = 64, 256  # TINY to minimize data
    num_experts = 2 * num_ranks  # Just 2 experts per rank
    num_topk = 1  # Single top-k
    num_local_experts = num_experts // num_ranks
    num_device_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    
    num_rdma_bytes = 256 * 1024 * 1024  # 256MB
    buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=num_device_sms, explicitly_destroy=True)
    
    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{local_rank}")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=f"cuda:{local_rank}")
    cumulative = torch.zeros((num_local_experts,), dtype=torch.int, device=f"cuda:{local_rank}")
    
    # Try the clean_low_latency_buffer before first iteration
    print(f"[rank {rank}] Starting iteration 0...", flush=True)
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts, use_fp8=False,
        cumulative_local_expert_recv_stats=cumulative,
        async_finish=False, return_recv_hook=True)
    hook()
    print(f"[rank {rank}] Dispatch 0 done", flush=True)
    
    combined_x, ce, ch = buffer.low_latency_combine(
        recv_x, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=True)
    ch()
    torch.cuda.synchronize()
    print(f"[rank {rank}] ✓ Iteration 0 complete", flush=True)
    
    # Wait and sync
    time.sleep(2)
    dist.barrier()
    time.sleep(2)
    print(f"[rank {rank}] After barrier, starting iteration 1...", flush=True)
    
    # Clean the buffer between iterations
    buffer.clean_low_latency_buffer(x, topk_idx, num_tokens, num_experts, use_fp8=False)
    torch.cuda.synchronize()
    print(f"[rank {rank}] Buffer cleaned, proceeding with dispatch 1...", flush=True)
    time.sleep(1)
    
    recv_x2, recv_count2, handle2, event2, hook2 = buffer.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts, use_fp8=False,
        cumulative_local_expert_recv_stats=cumulative,
        async_finish=False, return_recv_hook=True)
    hook2()
    print(f"[rank {rank}] Dispatch 1 done", flush=True)
    
    combined_x2, ce2, ch2 = buffer.low_latency_combine(
        recv_x2, topk_idx, topk_weights, handle2, async_finish=False, return_recv_hook=True)
    ch2()
    torch.cuda.synchronize()
    print(f"[rank {rank}] ✓ Iteration 1 complete!", flush=True)
    
    try: buffer.destroy()
    except: pass
    try: dist.destroy_process_group()
    except: pass

if __name__ == "__main__":
    ib = detect_ib_hca()
    if ib and ib.startswith("mlx"): os.environ["NCCL_IB_HCA"] = ib
    main()
