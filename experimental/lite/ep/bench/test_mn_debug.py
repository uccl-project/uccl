"""Debug multi-node: exactly 2 dispatch+combine cycles"""
import os, torch, torch.distributed as dist, sys
sys.path.insert(0, os.path.dirname(__file__))
from buffer import Buffer
from utils import init_dist_under_torchrun, detect_ib_hca

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    
    num_tokens, hidden = 128, 2048
    num_experts = 3 * num_ranks  # Same as simple test
    num_topk = 4
    num_local_experts = num_experts // num_ranks
    num_device_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    print(f"[rank {rank}] num_rdma_bytes={num_rdma_bytes}, num_qps_per_rank={num_device_sms}", flush=True)
    
    buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=num_device_sms, explicitly_destroy=True)
    
    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{local_rank}")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=f"cuda:{local_rank}")
    cumulative = torch.zeros((num_local_experts,), dtype=torch.int, device=f"cuda:{local_rank}")
    
    for iteration in range(5):
        print(f"[rank {rank}] === Iteration {iteration} ===", flush=True)
        
        # Dispatch
        print(f"[rank {rank}] Starting dispatch...", flush=True)
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, use_fp8=False,
            cumulative_local_expert_recv_stats=cumulative,
            async_finish=False, return_recv_hook=True)
        hook()
        print(f"[rank {rank}] ✓ Dispatch done", flush=True)
        
        # Combine
        print(f"[rank {rank}] Starting combine...", flush=True)
        combined_x, ce, ch = buffer.low_latency_combine(
            recv_x, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=True)
        ch()
        print(f"[rank {rank}] ✓ Combine done", flush=True)
        
        torch.cuda.synchronize()
        dist.barrier(group)
        print(f"[rank {rank}] ✓ Barrier passed", flush=True)
    
    print(f"[rank {rank}] All 5 iterations complete!", flush=True)
    
    try: buffer.destroy()
    except: pass
    try: dist.destroy_process_group()
    except: pass

if __name__ == "__main__":
    ib = detect_ib_hca()
    if ib and ib.startswith("mlx"): os.environ["NCCL_IB_HCA"] = ib
    main()
