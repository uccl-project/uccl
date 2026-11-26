# Selected two-node AllGather figure (2026-11-11)

[`selected_summary.csv`](selected_summary.csv) is the source of the main
two-node AllGather SVG. It has 21 FP32 complete-output sizes per GPU from
1 KiB to 1 GiB and records the UCCL path, paired NCCL configuration, MPI
process count, three-run latency range, and source for each point. Rebuild it
with `python3 scripts/select_multinode_host_allgather.py` from
`experimental/lite`, then run `python3 scripts/plot_optimized_collectives.py`.

At **1–256 KiB**, the selected UCCL path is the two-NUMA-group host multicast
AllGather with UCX automatic protocol. Each point uses its NCCL 2.28.7
control from the **same four-MPI-process/two-GPU-per-process harness** and
the same benchmark window. The [raw logs, per-call samples, protocol
evidence, and full method](../ucx_eager_nccl_2x4l4_20261111/README.md)
are retained. UCX eager is confirmed at 1 KiB; the automatic protocol can
change at larger sizes, so the plotted path is labeled `host-multicast-ucx-auto`.
The forced-eager measurements remain in that archive as a negative candidate
at 32–256 KiB. NCCL uses its default algorithm/protocol at 1–64 KiB and
`RING+SIMPLE` at 128/256 KiB to avoid the LL spike.

At **512 KiB–1 GiB**, the chart retains the previous eight-MPI-process
direct-GPU UCCL and paired NCCL controls from
[the tuned NCCL record](../nccl_tuned_2x4l4_20261111/README.md). Every
plotted comparison matches eight L4 GPUs, FP32 dtype, complete-output bytes,
host enqueue-to-completion timing, 20 warmups, three alternating paired
rounds of ten calls, and complete output checks. The MPI process layout
changes after 256 KiB because the selected UCCL implementation changes;
the CSV marks this explicitly. Adjacent points across that boundary are
not one continuous single-implementation sweep.

| Complete output/GPU | Selected host path (µs) | Paired NCCL (µs) | Earlier direct UCCL (µs) |
| ---: | ---: | ---: | ---: |
| 1 KiB | 25.54 | 37.17 | 33.09 |
| 64 KiB | 30.58 | 60.62 | 34.34 |
| 128 KiB | 34.78 | 90.50 | 37.43 |
| 256 KiB | 47.42 | 95.86 | 43.38 |

The earlier direct UCCL column is a **separate eight-process sweep**, shown
only to expose the candidate tradeoff; its numbers are not used in the figure.
At 256 KiB, the requested host path regresses versus that direct candidate by
about 9.3%. The 1-KiB selected host result has a high outlier round
(50.18 µs); at 128 KiB the host run range is 34.27–59.03 µs, still below
its paired NCCL range of 89.71–91.57 µs. No code dispatcher changed: this
is an explicit figure selection, not automatic runtime selection.
