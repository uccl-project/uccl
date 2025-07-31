import time
import threading
import torch
import pyproxy


def test_bench():
    bench = pyproxy.Bench()

    bench.start_local_proxies(rank=0, peer_ip="", pin_thread=True)

    bench.launch_gpu_issue_batched_commands()
    bench.sync_stream()
    bench.join_proxies()

    bench.print_block_latencies()
    stats = bench.compute_stats()
    bench.print_summary(stats)
    bench.print_summary_last()

    print("elapsed_ms:", bench.last_elapsed_ms())


def run_thread_with_kernel(block_idx, gpu_tensor, stream_ptr, rbs_ptr, num_blocks):
    proxy, rb_addr = (
        pyproxy.Proxy(
            rb_addr=pyproxy.alloc_cmd_ring(),
            block_idx=block_idx,
            gpu_buffer_addr=gpu_tensor.data_ptr(),
            total_size=gpu_tensor.numel(),
            rank=0,
            peer_ip="",
            pin_thread=True,
        ),
        pyproxy.alloc_cmd_ring(),
    )

    proxy.start_local()

    if block_idx == 0:
        pyproxy.launch_gpu_issue_kernel(num_blocks, 1, stream_ptr, rbs_ptr)
        pyproxy.sync_stream()

    proxy.stop()
    print(f"[Block {block_idx}] WRs completed: {proxy.completed_wr()}")
    pyproxy.free_cmd_ring(rb_addr)


def run_proxy_thread(block_idx, gpu_tensor):
    num_blocks = 4
    gpu_tensor = torch.empty(1 << 24, dtype=torch.uint8, device="cuda")

    bench = pyproxy.Bench()
    env = bench.env_info()
    stream_ptr = env["stream_addr"]
    rbs_ptr = env["rbs_addr"]

    threads = []
    for i in range(num_blocks):
        t = threading.Thread(
            target=run_thread_with_kernel,
            args=(i, gpu_tensor, stream_ptr, rbs_ptr, num_blocks),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def test_proxy():
    bench = pyproxy.Bench()
    env = bench.env_info()
    num_blocks = int(env["blocks"])
    stream_ptr = env["stream_addr"]
    rbs_ptr = env["rbs_addr"]

    nbytes = 1 << 24
    gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    gpu_addr = gpu.data_ptr()

    proxies = []
    for i in range(num_blocks):
        rb_i = bench.ring_addr(i)
        p = pyproxy.Proxy(
            rb_addr=rb_i,
            block_idx=i,
            gpu_buffer_addr=gpu_addr,
            total_size=nbytes,
            rank=0,
            peer_ip="",
            pin_thread=True,
        )
        p.start_local()
        proxies.append(p)
    bench.timing_start()
    pyproxy.launch_gpu_issue_kernel(
        num_blocks, int(env["threads_per_block"]), stream_ptr, rbs_ptr
    )
    pyproxy.sync_stream()
    bench.timing_stop()

    for p in proxies:
        p.stop()

    bench.print_block_latencies()
    stats = bench.compute_stats()
    bench.print_summary(stats)
    print("elapsed_ms:", bench.last_elapsed_ms())


def main():
    """Run all tests"""
    print("Running UCCL GPU-driven benchmark tests...")
    test_bench()
    print("Running UCCL GPU-driven proxy tests...")
    test_proxy()
    print("\n=== All UCCL GPU-driven tests completed! ===")


if __name__ == "__main__":
    main()
