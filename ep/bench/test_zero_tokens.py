"""
Test that dispatch+combine work correctly when many destination ranks receive
zero tokens.  This reproduces the vLLM profile_run() scenario where DeepSeek-V3
routing leaves most ranks with 0 incoming tokens.

Build:
  export OMP_NUM_THREADS=6
  export MAKE_NORMAL_MODE=1
  make clean && make -j install

On first node:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<NODE0_IP> --master_port=12355 \
    bench/test_zero_tokens.py

On second node:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<NODE0_IP> --master_port=12355 \
    bench/test_zero_tokens.py
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist

from utils import (
    inplace_unique,
    create_grouped_scores,
)
from buffer import Buffer

try:
    from uccl.ep import Config
except ImportError:
    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def make_sparse_routing(num_tokens, num_experts, num_ranks, num_topk,
                         num_local_ranks, num_nodes, target_ranks, rank):
    """Create a topk_idx that routes tokens ONLY to `target_ranks`.

    All other ranks receive 0 tokens.  This is the pattern that triggers
    the vLLM profile_run() hang.
    """
    num_experts_per_rank = num_experts // num_ranks
    # Pick experts that belong exclusively to target_ranks
    expert_pool = []
    for r in target_ranks:
        base = r * num_experts_per_rank
        expert_pool.extend(range(base, base + num_experts_per_rank))

    # For each token, pick num_topk experts uniformly from the pool
    idx = torch.randint(0, len(expert_pool), (num_tokens, num_topk), device="cuda")
    topk_idx = torch.tensor(expert_pool, device="cuda", dtype=torch.int64)[idx]

    return topk_idx


def run_test(rank, num_ranks, num_local_ranks, num_nodes, group, buffer,
             num_tokens, hidden, num_experts, num_topk, target_ranks,
             dispatch_config, combine_config, test_name, timeout_s=60):
    """Run a single dispatch+combine cycle and report success/failure."""
    local_rank = rank % num_local_ranks

    topk_idx = make_sparse_routing(
        num_tokens, num_experts, num_ranks, num_topk,
        num_local_ranks, num_nodes, target_ranks, rank,
    )

    # Derive layout from topk_idx
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    torch.cuda.synchronize()

    # Log what this rank sees
    ntp = num_tokens_per_rank.tolist()
    zero_count = sum(1 for v in ntp if v == 0)
    if local_rank == 0:
        print(f"  rank {rank}: num_tokens_per_rank={ntp}  ({zero_count}/{num_ranks} zeros)",
              flush=True)

    # x data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda")

    dist.barrier(group=group)

    # ---- Dispatch ----
    # NOTE: We deliberately do NOT call torch.cuda.synchronize() between
    # dispatch and combine because vLLM doesn't either.  Each rank proceeds
    # independently once its CPU returns from dispatch, so zero-token ranks
    # reach combine immediately while token-receiving ranks are delayed
    # by MoE computation.  This replicates the Bug 2 NVL-barrier hang.
    if local_rank == 0:
        print(f"  rank {rank}: starting dispatch ...", flush=True)

    t_start = time.time()
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=dispatch_config,
        async_finish=True,
    )

    # Make the compute stream wait on dispatch (vLLM does
    # event.current_stream_wait() here).  CPU does NOT block.
    event.current_stream_wait()

    num_recv_tokens = recv_x.shape[0]

    # ---- Simulate MoE compute on token-receiving ranks ----
    # vLLM skips the entire MoE path when num_recv_tokens == 0 (the
    # `M_full == 0` early-return in modular_kernel.py).  Ranks that *do*
    # receive tokens spend real GPU time on grouped GEMMs.  We mimic this
    # asymmetry so zero-token ranks arrive at combine first while
    # token-receiving ranks are still busy.  Use a LONG delay
    # (several seconds) so the NVL-barrier timeout (100s) can trigger if
    # Bug 2 is present.
    if num_recv_tokens > 0:
        fake_w = torch.randn((hidden, hidden), dtype=torch.bfloat16,
                             device="cuda")
        fake_out = recv_x
        # 200 iterations × ~5ms/iter ≈ 1s of GPU compute per test
        for _ in range(200):
            fake_out = fake_out @ fake_w
        # Keep the result alive so the compiler doesn't elide the work.
        recv_x = recv_x + 0.0 * fake_out[:num_recv_tokens]

    # ---- Combine ----
    # No sync here — zero-token ranks will call combine immediately.
    if local_rank == 0:
        print(f"  rank {rank}: starting combine  "
              f"(recv_x.shape={recv_x.shape}) ...", flush=True)

    combined_x, combined_topk_weights, event = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=combine_config,
        async_finish=True,
    )
    event.current_stream_wait()
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    if local_rank == 0:
        print(f"  rank {rank}: combine done, total {elapsed:.3f}s, "
              f"combined_x.shape={combined_x.shape}", flush=True)

    dist.barrier(group=group)

    if local_rank == 0:
        print(f"  [{test_name}] PASSED  (total={elapsed:.3f}s)", flush=True)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test zero-token dispatch/combine")
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{local_rank}")

    # Force NCCL to use the right interface for EFA
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp71s0")

    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = dist.new_group(list(range(num_ranks)))
    num_nodes = num_ranks // num_local_ranks

    if local_rank == 0:
        print(f"rank={rank} num_ranks={num_ranks} num_local_ranks={num_local_ranks} "
              f"num_nodes={num_nodes}", flush=True)

    # Buffer setup — match vLLM's exact settings for DeepSeek-V3-0324
    # vLLM uses: num_nvl_bytes=num_rdma_bytes=1GB, num_qps_per_rank=10
    num_nvlink_bytes = 1073741824  # 1 GB, same as vLLM
    num_rdma_bytes = 1073741824    # 1 GB, same as vLLM

    if local_rank == 0:
        print(f"num_nvlink_bytes={num_nvlink_bytes / 1e6:.1f} MB, "
              f"num_rdma_bytes={num_rdma_bytes / 1e6:.1f} MB", flush=True)

    buffer = Buffer(
        group,
        num_nvlink_bytes,
        num_rdma_bytes,
        low_latency_mode=False,
        num_qps_per_rank=10,  # same as vLLM
        explicitly_destroy=True,
    )

    # Use vLLM's default dispatch config for 16 ranks:
    # Config(num_sms=20, 36, 288, 20, 128) → num_channels=10
    dispatch_config = Buffer.get_dispatch_config(num_ranks)
    combine_config = Buffer.get_combine_config(num_ranks)
    if local_rank == 0:
        print(f"dispatch_config: num_sms={dispatch_config.num_sms}  "
              f"combine_config: num_sms={combine_config.num_sms}", flush=True)

    # ========================================================================
    # Test cases — exact vLLM profile_run() pattern first
    # ========================================================================
    tests = []

    # Test 1 (EXACT vLLM pattern): tokens routed to ranks 0,1,5,10,11,12
    # vLLM saw: [1024, 1024, 0, 0, 0, 1024, 0, 0, 0, 0, 1024, 1024, 1024, 0, 0, 0]
    if num_ranks >= 16:
        tests.append(("vllm_exact_pattern", [0, 1, 5, 10, 11, 12]))

    # Test 2: all tokens go to rank 0 only (15/16 ranks get 0 tokens)
    tests.append(("only_rank0", [0]))

    # Test 3: tokens go to 2 ranks on different nodes (14/16 zeros)
    if num_ranks > 8:
        tests.append(("cross_node_2ranks", [0, 8]))

    # Test 4: tokens go to exactly 1 rank on the remote node (15/16 zeros)
    if num_ranks > 8:
        tests.append(("only_remote_rank8", [8]))

    # Test 5: all tokens go to ranks on the OTHER node only
    if num_ranks > 8:
        tests.append(("all_remote", list(range(8, min(num_ranks, 16)))))

    if local_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {len(tests)} zero-token test cases", flush=True)
        print(f"{'='*60}\n", flush=True)

    for test_name, target_ranks in tests:
        if local_rank == 0:
            print(f"\n--- {test_name}: tokens routed to ranks {target_ranks} ---",
                  flush=True)

        torch.manual_seed(42 + rank)
        try:
            run_test(
                rank, num_ranks, num_local_ranks, num_nodes, group, buffer,
                args.num_tokens, args.hidden, args.num_experts, args.num_topk,
                target_ranks, dispatch_config, combine_config, test_name,
            )
        except Exception as e:
            if local_rank == 0:
                print(f"  [{test_name}] FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if local_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print("All tests completed.", flush=True)
        print(f"{'='*60}\n", flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
