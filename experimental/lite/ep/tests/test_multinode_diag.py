"""
Diagnostic multi-node test for DeepEP low-latency kernels.
Tests both return_recv_hook=True and return_recv_hook=False separately
to isolate which path hangs.

Usage (2 nodes, 1 GPU each):
  Node 0 (l40):
    conda activate uccl
    NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=2 --nproc_per_node=1 \
      --node_rank=0 --master_addr=4.14.153.89 --master_port=12357 \
      tests/test_multinode_diag.py

  Node 1 (l41):
    source ~/zhongjie/zj_py/bin/activate
    NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=2 --nproc_per_node=1 \
      --node_rank=1 --master_addr=4.14.153.89 --master_port=12357 \
      tests/test_multinode_diag.py
"""

import torch
import torch.distributed as dist
import time
import os
import sys
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
from buffer import Buffer
from utils import init_dist, detect_ib_hca

TIMEOUT = 15  # seconds per operation


def alarm_handler(signum, frame):
    print(f"[rank {os.environ.get('RANK','?')}] TIMEOUT after {TIMEOUT}s!", flush=True)
    os._exit(1)


def test_dispatch_combine(buffer, rank, num_ranks, group, return_recv_hook, label):
    """Run one dispatch+combine with given return_recv_hook setting."""
    num_tokens = 64
    hidden = 2048
    num_experts = 4 * num_ranks
    num_topk = 2
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=device)

    cumulative = torch.zeros(
        (num_experts // num_ranks,), dtype=torch.int, device=device
    )

    print(f"  [{label}] Calling low_latency_dispatch...", flush=True)
    signal.alarm(TIMEOUT)
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        use_fp8=False,
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=cumulative,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
    )
    if return_recv_hook:
        print(f"  [{label}] Calling dispatch hook...", flush=True)
        hook()
    else:
        print(f"  [{label}] Waiting on event...", flush=True)
        event.current_stream_wait()
    signal.alarm(0)
    print(f"  [{label}] Dispatch done! recv_x.shape={recv_x.shape}", flush=True)

    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device=device
    )

    print(f"  [{label}] Calling low_latency_combine...", flush=True)
    signal.alarm(TIMEOUT)
    combined_x, combine_event, combine_hook = buffer.low_latency_combine(
        x=recv_x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        handle=handle,
        use_logfmt=False,
        zero_copy=False,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
    )
    if return_recv_hook:
        print(f"  [{label}] Calling combine hook...", flush=True)
        combine_hook()
    else:
        print(f"  [{label}] Waiting on combine event...", flush=True)
        combine_event.current_stream_wait()
    signal.alarm(0)
    print(f"  [{label}] Combine done! combined_x.shape={combined_x.shape}", flush=True)
    return True


def main():
    signal.signal(signal.SIGALRM, alarm_handler)

    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):
        os.environ["NCCL_IB_HCA"] = ib_dev

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[rank {rank}] Starting diagnostic test (num_ranks={num_ranks})", flush=True)

    num_tokens = 64
    hidden = 2048
    num_experts = 4 * num_ranks
    scratch_nbytes = int(1e9)

    buffer = Buffer(
        group=group,
        num_nvl_bytes=0,
        num_rdma_bytes=scratch_nbytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=True,
        allow_mnnvl=False,
        explicitly_destroy=True,
    )
    print(f"[rank {rank}] Buffer created", flush=True)

    for proxy in buffer.proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            num_tokens, hidden, num_experts
        )
    print(f"[rank {rank}] Offsets calculated", flush=True)

    # Test 1: return_recv_hook=True (separate send/recv kernels)
    print(f"\n[rank {rank}] === Test 1: return_recv_hook=True ===", flush=True)
    try:
        ok = test_dispatch_combine(buffer, rank, num_ranks, group, True, "hook=True")
        print(f"[rank {rank}] Test 1 PASSED ✓", flush=True)
    except Exception as e:
        print(f"[rank {rank}] Test 1 FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    # Sync between tests
    dist.barrier()
    buffer.runtime.clean()
    time.sleep(0.5)

    # Test 2: return_recv_hook=False (combined send+recv kernel)
    print(f"\n[rank {rank}] === Test 2: return_recv_hook=False ===", flush=True)
    try:
        ok = test_dispatch_combine(buffer, rank, num_ranks, group, False, "hook=False")
        print(f"[rank {rank}] Test 2 PASSED ✓", flush=True)
    except Exception as e:
        print(f"[rank {rank}] Test 2 FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    print(f"\n[rank {rank}] All diagnostic tests complete!", flush=True)

    try:
        buffer.destroy()
    except Exception:
        pass
    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
