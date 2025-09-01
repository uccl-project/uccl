"""
Simple internode test for DeepEP low-latency kernels.
This test avoids the IPC handle issues by focusing only on low-latency functionality.
On first node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.141.1.1 --master_port=12356 \
  bench/test_internode_simple.py

On second node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.141.1.1 --master_port=12356 \
  bench/test_internode_simple.py
"""

import torch
import torch.distributed as dist
import time
from buffer import Buffer
import os

# import deep_ep as ep
try:
    from uccl import gpu_driven
    from uccl import uccl_ep as ep
except ImportError as exc:
    import sys

    sys.stderr.write("Failed to import uccl_ep\n")
    raise


from utils import init_dist, get_peer_ip, detect_ib_hca, get_cpu_proxies_meta


def test_simple_internode(rank: int, num_ranks: int, group: dist.ProcessGroup):
    num_tokens = 512
    hidden = 2048
    num_experts = 3 * num_ranks
    num_topk = 4
    device_index = int(os.environ["LOCAL_RANK"])
    print(f"[simple-test] Running on device {device_index}", flush=True)

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
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), device=f"cuda:{device_index}"
    )
    num_device_sms = torch.cuda.get_device_properties(
        device_index
    ).multi_processor_count

    bench = gpu_driven.Bench()
    # x_ptr = x.data_ptr()
    proxies = []

    scratch_nbytes = int(1e9)  # 256 MB
    scratch = torch.empty(
        scratch_nbytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    scratch_ptr = scratch.data_ptr()
    scratch_bytes = scratch.numel() * scratch.element_size()

    rank2meta = get_cpu_proxies_meta(rank, scratch_ptr, scratch_bytes, num_ranks, group)
    peers_meta_list = [rank2meta[r] for r in range(num_ranks)]

    for i in range(bench.num_proxies()):
        proxy = gpu_driven.Proxy(
            rb_addr=bench.ring_addr(i),
            block_idx=i,
            gpu_buffer_addr=scratch_ptr,
            total_size=scratch_bytes,
            rank=rank,
            peer_ip=peer_ip,
        )
        proxy.set_peers_meta(peers_meta_list)
        proxies.append(proxy)
    ep.register_proxies(device_index, proxies)

    dist.barrier(group)
    for i in range(bench.num_proxies()):
        proxies[i].start_dual()

    workers = None
    if hasattr(gpu_driven, "PeerCopyManager"):
        try:
            workers = gpu_driven.PeerCopyManager(src_device=device_index)
            workers.start_for_proxies(proxies)
            if rank == 0:
                print("[simple-test] ✓ PeerCopyManager started", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[simple-test] PeerCopyManager unavailable: {e}", flush=True)

    time.sleep(1)

    try:
        buffer = Buffer(
            group=group,
            rdma_buffer_ptr=scratch_ptr,
            num_nvl_bytes=0,
            num_rdma_bytes=int(scratch_nbytes),
            low_latency_mode=True,
            num_qps_per_rank=num_device_sms,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=False,
            explicitly_destroy=True,
        )

        if rank == 0:
            print("[simple-test] ✓ Buffer created successfully", flush=True)

        for proxy in proxies:
            proxy.calculate_and_set_dispatch_recv_data_offset(
                num_tokens, hidden, num_experts
            )

        if rank == 0:
            print(
                "[simple-test] ✓ dispatch_recv_data_offset calculated and set by CPU proxy",
                flush=True,
            )

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

        time.sleep(10)

        print("[simple-test] ✓ before destory!", flush=True)

        try:
            buffer.destroy()
        except Exception:
            pass
        dist.barrier()
        print("[simple-test] ✓ Buffer destroyed", flush=True)

        if workers is not None:
            try:
                workers.stop()
            except Exception:
                pass

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


def test_worker(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    try:
        test_simple_internode(rank, num_ranks, group)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):  # Mellanox IB devices show up like mlx5_0
        os.environ["NCCL_IB_HCA"] = ib_dev
        print(f"Set NCCL_IB_HCA={ib_dev}")
    else:
        print(f"Skipping NCCL_IB_HCA export (detected {ib_dev})")

    print("Simple internode test starting...")
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    test_worker(local_rank, num_local_ranks)
