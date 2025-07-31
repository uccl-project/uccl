import pyproxy


def test_local():
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


def main():
    """Run all tests"""
    test_local()


if __name__ == "__main__":
    main()
