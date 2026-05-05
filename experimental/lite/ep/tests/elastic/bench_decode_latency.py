"""Small-batch decode-style latency scan for ElasticBuffer.

For each `num_tokens` value, this script:
  1. (Re)allocates an ElasticBuffer (sized either per-iter for `num_tokens` or
     once at the largest batch in the scan, depending on `--reuse-buffer`).
  2. Runs a one-shot warm dispatch (with topk_idx) to populate `EPHandle`.
  3. Repeatedly invokes `buffer.dispatch(handle=cached_handle)` and
     `buffer.combine(handle=cached_handle)` while measuring per-call latency
     via CUDA events. This is the same code path vLLM/SGLang would hit during
     decode (gating decisions cached, no CPU sync, fully async).

The output is a per-rank table with median / p99 latencies for dispatch,
combine, and end-to-end (dispatch + combine).

Run intra-node:

    CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$(pwd) ... \
      python3 tests/elastic/bench_decode_latency.py \
        --num-processes 2 --num-tokens-list 1,4,16,64,128

Run multi-node via run_multinode.sh:

    NCCL_ROOT=...  TORCH_NVSHMEM_STUB=...  EP_JIT_CACHE_DIR=...  \
      GPUS_PER_NODE=4 GPU_LIST=0,1,2,3 NODE1_HOST=l41  \
      MASTER_ADDR=... \
      TEST_SCRIPT=tests/elastic/bench_decode_latency.py \
      TEST_ARGS="--num-tokens-list 1,4,16,64,128 --num-iters 30" \
      bash run_multinode.sh

If a previous run was killed mid-flight, clean up leaked SHM segments before
re-running:

    ls /dev/shm/ | grep uccl_deepepv2_ | xargs -I {} rm -f /dev/shm/{}
"""
import argparse
import os
import statistics
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, init_seed
from deep_ep.utils.gate import get_unbalanced_scores


def time_call(fn, num_iters: int, num_warmup: int = 10) -> list:
    """Time `fn` `num_iters` times. Returns list of per-call ms."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def fmt_us(times_ms: list) -> str:
    times_us = sorted(t * 1000 for t in times_ms)
    n = len(times_us)
    median = times_us[n // 2]
    p99 = times_us[min(n - 1, int(n * 0.99))]
    mean = statistics.mean(times_us)
    return f'med={median:7.1f} p99={p99:7.1f} mean={mean:7.1f} us'


def make_buffer(group, num_max_tokens_per_rank, hidden, num_topk, args):
    return deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        allow_hybrid_mode=args.allow_hybrid_mode,
        allow_multiple_reduction=args.allow_multiple_reduction,
        explicitly_destroy=True,
    )


@torch.inference_mode()
def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)

    num_tokens_list = sorted(int(x) for x in args.num_tokens_list.split(','))
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    buffer_max = max(num_tokens_list) if args.reuse_buffer else None

    if rank_idx == 0:
        mode = f'reuse buffer max={buffer_max}' if args.reuse_buffer else 'fresh buffer per scan point'
        print(f'\n=== Decode latency scan ({num_ranks} ranks, hidden={hidden}, '
              f'num_topk={num_topk}, num_experts={num_experts}, num_iters={args.num_iters}, '
              f'{mode}) ===',
              flush=True)

    init_seed(args.seed)
    rows = []
    shared_buffer = None
    if args.reuse_buffer:
        shared_buffer = make_buffer(group, buffer_max, hidden, num_topk, args)

    for num_tokens in num_tokens_list:
        if args.reuse_buffer:
            buffer = shared_buffer
            buf_max = buffer_max
        else:
            buffer = make_buffer(group, num_tokens, hidden, num_topk, args)
            buf_max = num_tokens

        num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)
        num_qps = buffer.get_theoretical_num_qps(num_sms)

        # Mimic test_ep: per-rank skew (rank 0 has the full count).
        per_rank_num_tokens = max(1, num_tokens - rank_idx)
        x = torch.randn((per_rank_num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        scores = get_unbalanced_scores(per_rank_num_tokens, num_experts, num_ranks, num_topk,
                                       ratio=1.0, precise=False)
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)
        topk_weights = topk_weights.to(torch.float32)
        topk_idx = topk_idx.to(deep_ep.topk_idx_t)

        recv_x, _, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=buf_max,
            expert_alignment=128,
            num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=False,
        )

        if isinstance(recv_x, tuple):
            from deep_ep.utils.math import per_token_cast_back
            combine_x = per_token_cast_back(recv_x[0], recv_x[1])
        else:
            combine_x = recv_x.clone()

        def cached_dispatch():
            buffer.dispatch(x, handle=handle, num_sms=num_sms, num_qps=num_qps,
                            async_with_compute_stream=False)

        def cached_combine():
            buffer.combine(combine_x, handle=handle, num_sms=num_sms, num_qps=num_qps,
                           async_with_compute_stream=False)

        def cached_both():
            buffer.dispatch(x, handle=handle, num_sms=num_sms, num_qps=num_qps,
                            async_with_compute_stream=False)
            buffer.combine(combine_x, handle=handle, num_sms=num_sms, num_qps=num_qps,
                           async_with_compute_stream=False)

        d_times = time_call(cached_dispatch, args.num_iters, num_warmup=args.num_warmup)
        c_times = time_call(cached_combine, args.num_iters, num_warmup=args.num_warmup)
        e2e_times = time_call(cached_both, args.num_iters, num_warmup=args.num_warmup)

        if rank_idx == 0:
            print(f'\nnum_tokens={num_tokens:4d}:', flush=True)
            print(f'  dispatch: {fmt_us(d_times)}', flush=True)
            print(f'  combine : {fmt_us(c_times)}', flush=True)
            print(f'  d+c     : {fmt_us(e2e_times)}', flush=True)

        rows.append((num_tokens, sorted(d_times), sorted(c_times), sorted(e2e_times)))

        del cached_dispatch, cached_combine, cached_both, combine_x, recv_x, handle
        torch.cuda.synchronize()
        if not args.reuse_buffer:
            buffer.destroy()
            del buffer
            torch.cuda.empty_cache()

    if rank_idx == 0:
        print('\n=== Summary (median µs) ===', flush=True)
        print(f'{"num_tokens":>10} {"dispatch":>10} {"combine":>10} {"d+c":>10}', flush=True)
        for nt, dt, ct, et in rows:
            n = len(dt)
            print(f'{nt:>10d} '
                  f'{dt[n//2]*1000:>10.1f} '
                  f'{ct[n//2]*1000:>10.1f} '
                  f'{et[n//2]*1000:>10.1f}', flush=True)

    if shared_buffer is not None:
        shared_buffer.destroy()
    dist.destroy_process_group()
    if int(os.environ.get('EP_FORCE_PROCESS_EXIT', '0')):
        os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-tokens-list', type=str, default='1,4,16,64,128',
                        help='Comma-separated list of num_tokens to scan')
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-iters', type=int, default=30)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-hybrid-mode', type=int, default=0)
    parser.add_argument('--allow-multiple-reduction', type=int, default=1)
    parser.add_argument('--reuse-buffer', action='store_true',
                        help='Allocate one buffer at max(num_tokens_list) and reuse it for '
                             'all scan points (mirrors vLLM/SGLang dynamic-batch usage).')
    # Tolerate stray test_ep-style args from default TEST_ARGS in run_multinode.sh
    args, _ = parser.parse_known_args()

    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
