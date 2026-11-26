# Host DRAM bandwidth on the four-L4 machine (2026-11-11)

This measures **CPU host-memory bandwidth**, the relevant resource for the
NUMA-local CPU router on `mibura-sky-test-02`. The host has two Xeon Silver
4410Y sockets, each with 12 physical cores, 24 logical CPUs, and 30 MiB of
L3 cache. `numactl --membind` keeps each process's arrays on its local NUMA
node. The OpenMP threads are pinned to 12 physical cores or to all 24
logical CPUs on that node. See [hardware/software identity](hardware_software.txt)
and the exact [isolated](commands_isolated.json) and
[concurrent](commands_concurrent.json) commands.

The standalone [STREAM-style triad](stream_triad.cc) computes
`c[i] = a[i] + 3*b[i]` over three 512-MiB double arrays: **1.5 GiB per
NUMA process**, much larger than the socket's LLC. Its effective rate counts
two useful reads and one useful write, `24 × elements / seconds`, in decimal
GB/s. Each run has three warmups and ten timed passes; the table uses the
median of three independent run medians. `summary.csv` retains the full
run-median ranges. Two-process runs initialize both nodes first and then
release a shared start gate. `concurrent_aggregate.csv` records the sum of
both nodes' simultaneously measured useful rates in each run.

| Workload | NUMA 0 | NUMA 1 | Aggregate |
| --- | ---: | ---: | ---: |
| Isolated, 12 physical cores/node | 87.12 GB/s | 87.21 GB/s | — |
| Isolated, 24 logical threads/node | 89.71 GB/s | 89.82 GB/s | — |
| Concurrent, 12 physical cores/node | 87.24 GB/s | 87.33 GB/s | **174.56 GB/s** |
| Concurrent, 24 logical threads/node | 87.60 GB/s | 89.73 GB/s | **177.31 GB/s** |

The aggregate is the sum of per-node useful bandwidths, not a memory-controller
counter. SMT increases the concurrent aggregate by about **1.6%** for this
triad. These measurements do not establish the DIMMs' theoretical peak or
the bandwidth of every CPU-router operation. Cached stores can create memory
traffic beyond the three useful arrays counted here.

The older [hardware-profiler record](../hardware_profile_4l4_20261110_summary.csv)
measured a different workload with four workers and 8 MiB per worker:
CPU copy 69.50/48.72 GB/s (NUMA 0/1) and four-source BF16 reduction
47.56/48.98 GB/s. Those numbers describe copy/reduction kernel throughput,
not the large-working-set STREAM triad. They are retained rather than
substituted into the new DRAM table. The separate
[CPU-router worker sweep](../cpu_router_4l4_workers_20261111/README.md)
shows that fewer workers did not improve every collective, so DRAM bandwidth
alone does not determine the router's latency.

All 24 raw logs and 240 timed passes are checked by `python3 summarize.py`,
which regenerates [summary.csv](summary.csv) and
[concurrent_aggregate.csv](concurrent_aggregate.csv). The sampled checksum
also verifies that the triad's result was computed. To rebuild the probe
from `experimental/lite`:

```bash
make host-memory-bw-triad
```
