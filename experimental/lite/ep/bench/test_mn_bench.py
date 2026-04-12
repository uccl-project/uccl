"""Minimal multi-node dispatch+combine benchmark"""
import os, time, torch, torch.distributed as dist, sys
sys.path.insert(0, os.path.dirname(__file__))
from buffer import Buffer
from utils import init_dist_under_torchrun, detect_ib_hca

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    
    num_tokens, hidden, num_experts, num_topk = 128, 2048, 8 * num_ranks, 4
    num_local_experts = num_experts // num_ranks
    
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=num_local_experts, explicitly_destroy=True)
    
    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=f"cuda:{local_rank}")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=f"cuda:{local_rank}")
    
    cumulative = torch.zeros((num_local_experts,), dtype=torch.int, device=f"cuda:{local_rank}")
    
    print(f"[rank {rank}] Starting dispatch...", flush=True)
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts, use_fp8=False,
        cumulative_local_expert_recv_stats=cumulative,
        async_finish=False, return_recv_hook=True)
    hook()
    print(f"[rank {rank}] ✓ Dispatch done, recv shape: {recv_x.shape}", flush=True)
    
    combined_x, combine_event, combine_hook = buffer.low_latency_combine(
        recv_x, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=True)
    combine_hook()
    print(f"[rank {rank}] ✓ Combine done", flush=True)
    
    # Benchmark
    def test_fn():
        r, rc, h, e, hk = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, use_fp8=False,
            cumulative_local_expert_recv_stats=cumulative, async_finish=False, return_recv_hook=True)
        hk()
        cx, ce, ch = buffer.low_latency_combine(r, topk_idx, topk_weights, h, async_finish=False, return_recv_hook=True)
        ch()
    
    # Warmup
    for _ in range(3):
        test_fn()
    torch.cuda.synchronize()
    
    N = 20
    start = time.time()
    for _ in range(N):
        test_fn()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_us = elapsed / N * 1e6
    print(f"[rank {rank}] Dispatch+Combine: {avg_us:.1f} us avg over {N} iters", flush=True)
    
    try:
        buffer.destroy()
    except: pass
    try:
        dist.destroy_process_group()
    except: pass

if __name__ == "__main__":
    ib = detect_ib_hca()
    if ib and ib.startswith("mlx"): os.environ["NCCL_IB_HCA"] = ib
    main()
