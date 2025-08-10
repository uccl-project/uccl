"""
Simple internode test for DeepEP low-latency kernels.
This test avoids the IPC handle issues by focusing only on low-latency functionality.
"""

import argparse
import torch
import torch.distributed as dist

from buffer import Buffer

# import deep_ep as ep
try:
    from uccl import gpu_driven
    from uccl import uccl_ep as ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl_ep\n")
    raise


from utils import init_dist, get_peer_ip


def test_simple_internode(rank: int, num_ranks: int, group: dist.ProcessGroup):
    num_tokens = 512
    hidden = 2048
    num_experts = 64
    num_topk = 4
    device_index = 0

    peer_ip = get_peer_ip(rank, num_ranks, group)

    if rank == 0:
        print(
            f"[simple-test] Testing with {num_tokens} tokens, {hidden} hidden, {num_experts} experts, peer IP: {peer_ip}",
            flush=True,
        )
        print(
            f"[simple-test] Running on {num_ranks} ranks across {num_ranks} nodes, peer_ip: {peer_ip}",
            flush=True,
        )
    torch.manual_seed(rank)
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{device_index}"
    )
    x_bytes = x.numel() * x.element_size()
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), device=f"cuda:{device_index}"
    )
    num_device_sms = torch.cuda.get_device_properties(0).multi_processor_count

    bench = gpu_driven.Bench()
    x_ptr = x.data_ptr()
    proxies = []
    for i in range(bench.blocks()):
        proxy = gpu_driven.Proxy(
            rb_addr=bench.ring_addr(i),
            block_idx=i,
            gpu_buffer_addr=x_ptr,
            total_size=x_bytes,
            rank=rank,
            peer_ip=peer_ip,
        )
        proxy.start_dual()
        proxies.append(proxy)
    ep.register_proxies(device_index, proxies)

    try:
        buffer = Buffer(
            group=group,
            num_nvl_bytes=0,
            num_rdma_bytes=int(
                1e9
            ),  # TODO(MaoZiming): How does num_rdma_bytes relate to the size of x?
            low_latency_mode=True,
            num_qps_per_rank=num_device_sms,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=False,
            explicitly_destroy=True,
        )

        if rank == 0:
            print("[simple-test] ✓ Buffer created successfully", flush=True)

        cumulative_local_expert_recv_stats = torch.zeros(
            (num_experts // num_ranks,), dtype=torch.int, device="cuda"
        )
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            use_fp8=False,
            round_scale=False,
            use_ue8m0=False,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            async_finish=False,
            return_recv_hook=True,
        )
        hook()

        if rank == 0:
            print("[simple-test] ✓ Low-latency dispatch completed", flush=True)
            print(f"[simple-test] Received tensor shape: {recv_x.shape}", flush=True)

        topk_weights = torch.ones(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        )
        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            x=recv_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=True,
        )
        combine_hook()

        if rank == 0:
            print("[simple-test] ✓ Low-latency combine completed", flush=True)
            print(
                f"[simple-test] Combined tensor shape: {combined_x.shape}", flush=True
            )
            print("[simple-test] ✓ All tests passed!", flush=True)

        try:
            buffer.destroy()
        except Exception:
            pass
        dist.barrier()
        print("[simple-test] ✓ Buffer destroyed", flush=True)

        try:
            for p in proxies:
                p.stop()
        except Exception:
            pass

        if rank == 0:
            print("[simple-test] ✓ Proxy stopped", flush=True)
            try:
                ep.unregister_proxy(device_index)
            except Exception:
                pass
            print("[simple-test] ✓ Command ring freed", flush=True)

        dist.barrier()

    except Exception as e:
        if rank == 0:
            import traceback

            print(f"[simple-test] ✗ Error: {repr(e)}", flush=True)
            traceback.print_exc()

        raise


def test_worker(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    try:
        test_simple_internode(rank, num_ranks, group)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple DeepEP internode test")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes per node (default: 1)",
    )
    args = parser.parse_args()

    print("Simple internode test starting...")
    torch.multiprocessing.spawn(
        test_worker, args=(args.num_processes, args), nprocs=args.num_processes
    )
