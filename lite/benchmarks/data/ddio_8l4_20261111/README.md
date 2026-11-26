# GPU-write cache residency and CPU reduction probe (8xL4)

This microbenchmark asks whether host inputs written by the GPUs are faster
for the CPU to reduce before those inputs are evicted from cache. It uses
the production `CpuExecutor::sumBf16` function on four BF16 sources and one
output. GPUs 0–3 write NUMA-0 pinned buffers; GPUs 4–7 write NUMA-1 pinned
buffers. One CPU thread is pinned to CPU 0 or 24, respectively. The host
reports an Intel Xeon CPU at 2.20 GHz, CUDA 12.9, driver 580.178.04, and
eight NVIDIA L4s. Each source
contains the constant 1, 2, 3, or 4; every output element is checked against
the exact BF16 sum 10. The host has 1 MiB L2 per core and about 38.5 MiB
L3 per NUMA node.

The six conditions use the **same** host input buffers and reduction code:

| Condition | Preparation before timing the CPU reduction |
| --- | --- |
| `cpu_hot` | CPU reads each input cache line. |
| `cpu_flushed` | `CLFLUSH` evicts input and output lines. |
| `gpu_sm_write` | Four GPU SM kernels write their mapped host buffers, then synchronize. |
| `gpu_sm_write_flushed` | Same GPU writes, then `CLFLUSH` inputs. |
| `gpu_copy_engine` | Four device-to-host copies, then synchronize. |
| `gpu_copy_engine_flushed` | Same copies, then `CLFLUSH` inputs. |

The CPU reduction alone is timed with a host clock; GPU writes, synchronization,
cache preparation, and output verification are outside timing. At each size,
the six conditions rotate order. Each of three independent process runs per
NUMA node has five warmups and 21 timed repetitions per condition. The
[summary](summary.csv) is the median of the three run medians. All six raw
runs end in correctness PASS:

| NUMA | Run 1 | Run 2 | Run 3 |
| --- | --- | --- | --- |
| 0 | [samples](node0_run1.csv) | [samples](node0_run2.csv) | [samples](node0_run3.csv) |
| 1 | [samples](node1_run1.csv) | [samples](node1_run2.csv) | [samples](node1_run3.csv) |

| NUMA | Source bytes each | CPU hot | CPU flushed | GPU SM write | GPU write, then flush |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 KiB | 165.7 µs | 229.1 µs | 175.5 µs | 231.0 µs |
| 1 | 512 KiB | 163.9 µs | 225.5 µs | 173.0 µs | 227.3 µs |
| 0 | 2 MiB | 610.7 µs | 883.2 µs | 751.7 µs | 885.4 µs |
| 1 | 2 MiB | 595.4 µs | 876.4 µs | 732.6 µs | 879.0 µs |

At 512 KiB per source, matching the large AllReduce router's maximum input
tile, reduction immediately after GPU SM writes is about **1.31× faster**
than after evicting those same writes on both NUMA nodes. At 2 MiB per
source it is **1.18–1.20× faster**. Copy-engine writes show a similar
pattern. At 8 MiB per source, the four inputs plus output exceed one node's
L3 capacity and the gap shrinks to about 5%. The first NUMA-0 process run
had elevated absolute times at several sizes; it remains in the raw data.

This confirms that cache residency is useful for the CPU reduction and that
GPU-written host inputs retain a measurable cache-sensitive advantage. It
does **not** prove which PCIe writes allocated in the LLC or that DDIO was
enabled: this virtualized host exposes neither DDIO/uncore counters nor an
`IO_LLC_WAYS` control. Intel documents that inbound I/O writes may allocate
in only a configured subset of LLC ways, so the full L3 size is not a
guaranteed DDIO budget. The 64-KiB post-GPU-flush condition was slower than
the standalone cold condition; that small-size anomaly is retained in the
summary and is not used to infer DDIO behavior. One CPU thread and a
synchronize-before-reduce schedule isolate cache effects; they do not prove
an end-to-end gain for the concurrent multi-worker packet router.

Build and rerun from `experimental/lite`:

```bash
make ddio-reduce-probe CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0
./build/uccl_lite_ddio_reduce_probe --node 0 --cpu 0 \
  --trials 21 --warmups 5 > build/ddio-node0-new.csv
./build/uccl_lite_ddio_reduce_probe --node 1 --cpu 24 \
  --trials 21 --warmups 5 > build/ddio-node1-new.csv
python3 scripts/summarize_ddio_reduce_probe.py
```

[Intel DDIO behavior](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2024-1/effective-utilization-of-intel-ddio-technology.html)
and [LLC-way controls](https://www.intel.com/content/www/us/en/developer/articles/technical/ddio-analysis-performance-monitoring.html)
provide the hardware context. The timing claim above comes from this probe,
not from a DDIO counter.
