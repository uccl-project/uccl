import threading
import torch

try:
    from uccl import gpu_driven
except ImportError as exc:
    sys.stderr.write("Failed to import gpu_driven\n")
    raise


def test_bench():
    bench = gpu_driven.Bench()
    bench.start_local_proxies()
    bench.launch_gpu_issue_batched_commands()
    bench.sync_stream()
    bench.join_proxies()

    bench.print_block_latencies()
    stats = bench.compute_stats()
    bench.print_summary(stats)
    bench.print_summary_last()
    print("elapsed_ms:", bench.last_elapsed_ms())


def test_proxy():
    bench = gpu_driven.Bench()
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
        p = gpu_driven.Proxy(
            rb_addr=rb_i,
            block_idx=i,
            gpu_buffer_addr=gpu_addr,
            total_size=nbytes,
        )
        p.start_local()
        proxies.append(p)
    bench.timing_start()
    gpu_driven.launch_gpu_issue_kernel(
        num_blocks, int(env["threads_per_block"]), stream_ptr, rbs_ptr
    )
    gpu_driven.sync_stream()
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
