# Four-GPU peer AllGather and ReduceScatter

## November 13 full-range follow-up

The [new full-size study](../benchmarks/data/small_medium_4l4_20261113/README.md)
adds eager fused host AllGather (`mini-host`) for the medium range and a phased
LL ReduceScatter hierarchy for 16–32 KiB. At 1 / 4 MiB AG takes 66.43 / 188.79 µs;
at 32 KiB RS takes 19.47 µs. Complete curves now include 16 B–1 GiB, with every
candidate, range and negative result retained. The separate prepared resident
mode is input-ready and host-blocking; its timings must not be mixed into these
eager GPU-event curves. The README links the current full-size tables.


## Archived November 11 four-L4 AllGather figure

The [owner-host FP32 paired sweep](../benchmarks/data/fp32_packet_ag_per_source_4l4_20261111/README.md)
keeps `push-serial` at 1 KiB–2 MiB and the direct peer path at 4–8 MiB,
then uses the per-source GPU-progress host-packet router at 16 MiB–1 GiB.
Every size uses the
same four GPUs, FP32 dtype, out-of-place placement, CUDA-event timer and
paired NCCL 2.28.7 control. The older l40 direct-peer sweep below remains
archived. The main AllReduce and ReduceScatter figures retain their prior
FP32 implementations and values.

## Archived FP32 four-L4 AllGather

The [November 11 paired sweep](../benchmarks/data/ag_midrange_4l4_20261111/README.md)
adds a single-CTA source-side push. Each GPU writes its shard to all four
rank-ordered output slices, visiting one peer at a time; the four sources
rotate destination order. This removes the many copy submissions and peer
completion waits from the small-message path. The default benchmark includes
this candidate through 2 MiB complete output bytes per GPU; larger messages
use the measured direct peer-copy path. The optional vectorized pull and CPU
router remain separate candidates in the detailed data.

On l40, the selected non-CPU-router path takes 15.85 versus 32.36 µs at
64 KiB and 72.88 versus 99.92 µs at 1 MiB (UCCL Lite versus NCCL 2.28.7).
The 4-MiB point still loses: 239.43 versus 234.39 µs. A second four-L4 host
reproduces the 16 KiB–1 MiB gains. Both sweeps use FP32, full output checks
on every call, 20 warmups, three rotated runs of ten calls, and the same
per-call CUDA-event timer for each paired backend. The archived FP32 figure
below uses only the l40 sweep, with one selected UCCL path and NCCL.

## Archived FP32 four-L4 ReduceScatter

The [new paired sweep and candidate data](../benchmarks/data/rs_ar_serial_4l4_20261111/README.md)
uses one source-side CTA per GPU to visit peer scratch regions sequentially.
Through 1 MiB, the destination's same kernel waits for system-scope source
flags and reduces the four contributions. An LL16 packet form is selected at
8 KiB. From 2–8 MiB, a sequential push followed by the destination reduction
wins; larger sizes retain the peer-copy path. The archived FP32 figure uses the l40
selected path and paired NCCL values.

On l40, the selected path takes 21.76 versus 29.43 µs at 32 KiB and 85.57
versus 90.48 µs at 1 MiB. It wins at 16 of 21 sizes; the five points from
64 MiB through 1 GiB still lose. Both four-L4 hosts passed complete output
checks on every warmup and timed call, plus non-power-of-two tail checks.

## Host-staged AllGather candidate

The optional `--router-ag` benchmark also measures a pipelined pinned-host
AllGather on four L4 GPUs. Each GPU uploads its shard in chunks to two reusable
host slots bound to its PCI NUMA node; every GPU downloads all four shards in
rank order. Cross-device CUDA events protect slot reuse while uploads of the
next chunk overlap the current fan-out. The [duplex-optimized 21-size sweep,
raw samples, and Nsight timelines](../benchmarks/data/cpu_router_ag_duplex_20261111/README.md)
use 4 MiB chunks and compare against direct peer paths and NCCL 2.28.7.
The [original CUDA-allocated host-slot sweep](../benchmarks/data/cpu_router_ag_4l4_20261111/README.md)
is retained as an ablation.
This is an experimental host-staged candidate, separate from the BF16 CPU
router module and from the archived FP32 direct-peer curve in this document.

## Previous pull and peer-copy sweep

The previous four-L4 eager benchmark on l40 compared three correct paths in
one process: the original source-side peer copies, destination-side peer
reads, and NCCL 2.28.7. Every backend uses FP32 out-of-place buffers,
slowest-rank per-call CUDA-event time, five warmups, ten timed calls, and
three rotated repetitions. A GPU fill kernel publishes changed inputs and
a GPU check kernel inspects every output element; both run outside the timed
call for all backends. `cudaDeviceSynchronize` establishes the input-ready
boundary before peer reads. This is a standalone eager contract, not a
validation of arbitrary application streams.

The selected UCCL Lite path reads peers directly through 32 KiB. Above
that, remote-read bandwidth collapses and the source-side peer-write path
wins. For peer writes, one leader consolidates the four source completion
events when AG is at most 256 KiB or RS is at most 64 KiB; larger messages
retain the original waits. The selected path is chosen by correct UCCL
medians without consulting NCCL.
For RS at 256 KiB, 1–8 MiB, and at least 32 MiB, the destination reduction
reads its own input shard directly and skips the local copy into scratch.
The tested 128-KiB, 512-KiB, and 16-MiB exceptions retain that copy because
the direct-self variant regressed there on at least one host.

![Optimized four-L4 AllGather latency and bus throughput](assets/collectives-4l4-allgather-optimized.svg)

![Optimized four-L4 ReduceScatter latency and bus throughput](assets/collectives-4l4-reducescatter-optimized.svg)

At 2 KiB, AG takes **14.07 versus 26.57 µs** and RS **13.73 versus
26.34 µs** (UCCL Lite versus NCCL). At 1 GiB, AG takes **45.489 versus
47.599 ms**, while RS still loses at **48.579 versus 46.567 ms**. AG wins
at 13 of 21 sizes and RS at 7 of 21; medium-size regressions remain in the
full sweep.

The [optimization record](../benchmarks/data/optimization_20261110/README.md)
links the complete raw log, 3-run samples, every UCCL candidate, and the
Nsight/ablation evidence. An unfenced peer-read prototype failed RS
correctness at 1 MiB and is retained there as a negative result.

## November 10 refreshed sweep

The current-source sweep on `mibura-sky-test-02` uses four L4s in 2+2 NUMA
order, driver 595.58.03, CUDA 12.9 and NCCL 2.28.7. Both direct-peer and
NCCL run in one process with the same FP32 out-of-place buffers, complete
bytes per rank, slowest-rank per-call CUDA-event timer, and alternating order.
Each of three repetitions has five warmups and ten timed calls. Every warmup
and timed output is fully checked with a changed input generation. A separate
1008-byte non-power-of-two tail check passes.

![Fresh four-L4 AllGather latency and bus throughput](assets/collectives-4l4-allgather-refresh.svg)

![Fresh four-L4 ReduceScatter latency and bus throughput](assets/collectives-4l4-reducescatter-refresh.svg)

| Complete bytes/rank | AG peer / NCCL (µs) | RS peer / NCCL (µs) |
| ---: | ---: | ---: |
| 1 KiB | 90.666 / 40.282 | 99.914 / 32.938 |
| 1 MiB | 117.491 / 109.014 | 164.496 / 100.739 |
| 32 MiB | 1717.338 / 1769.373 | 1560.730 / 1714.086 |
| 1 GiB | 45644.624 / 49371.273 | 51001.610 / 49011.466 |

The 1-GiB AG result is 1.082× faster; RS loses by about 4%. Small-message
losses remain substantial. The [complete summary](../benchmarks/data/refresh_20261110/single_peer_summary.csv),
[2,520 timed call samples](../benchmarks/data/refresh_20261110/single_peer_samples.csv),
[raw log](../benchmarks/data/refresh_20261110/single_peer_ag_rs_raw.log),
and [tail check](../benchmarks/data/refresh_20261110/single_peer_tail_1008.log)
retain every size and repetition. The older sweep below is historical and
uses a different per-run batch timing method.

[peer_ag_rs.cu](../src/collectives/peer_ag_rs.cu) is a standalone float32
implementation and matched NCCL benchmark for the four-GPU PCIe host. It
requires CUDA peer access among all four devices and uses one process. It is
separate from the BF16 CPU router and the historical eight-GPU no-P2P result.

Each source GPU sends its AllGather shard to a distinct output region on every
destination. For ReduceScatter, each source sends the destination's input shard
to that destination's scratch. A destination kernel sums four received arrays
in FP32 and owns every output element. Cross-device CUDA events make each
destination wait for all source transfers. Before reusing ReduceScatter
scratch, sources wait for the preceding destination reduction. The source
tensors remain immutable until completion. The rotating enqueue order spreads
first transfers across peers; CUDA is free to overlap later phases.

The harness validates every output element against a position-sensitive
reference for three changed input generations, including one after timing.
A separate 1008-byte run checks the non-power-of-two tail. The supported byte
count is a multiple of 16, because the four ranks each own an integral number
of float32 elements. Both operations are out of place.

```bash
make peer_ag_rs NCCL_HOME=/path/to/nccl/build
./build/uccl_lite_peer_ag_rs -b 1024 -e 67108864 -w 5 -n 10 -r 3
./build/uccl_lite_peer_ag_rs -b 1008 -e 1008 -w 2 -n 10 -r 3
```

## Measured result

The [complete 17-size sweep](../benchmarks/data/peer_ag_rs_4l4_20261110.txt)
was run November 10, 2026 on four NVIDIA L4s, PCIe Gen4 x16, NUMA pairs
`(0,1)` and `(2,3)`, driver 595.58.03, CUDA 12.9 and NCCL 2.28.7. Each
backend uses the same process, buffers, streams, dtype, size, and timing
method. The order alternates across three repetitions. Each repetition has
five warmups and ten timed calls; the recorded latency is the slowest-rank
CUDA-event interval divided by ten. The table shows medians of repetition
values. Allocation, fill and correctness checking are outside timing. The
complete file also retains each backend's min/max run values. No per-size
variant selection is used.

Small-message latency (peer / NCCL, lower is better):

| Complete bytes/rank | AllGather | ReduceScatter |
| ---: | ---: | ---: |
| 1 KiB | 56.42 / 19.43 µs | 79.97 / 19.66 µs |
| 1 MiB | 77.33 / 83.34 µs | 108.95 / 76.49 µs |

Large-message bus bandwidth (peer / NCCL, higher is better):

| Complete bytes/rank | AllGather | ReduceScatter |
| ---: | ---: | ---: |
| 8 MiB | 16.93 / 15.84 GB/s | 16.19 / 15.04 GB/s |
| 16 MiB | 17.81 / 16.36 GB/s | 17.25 / 15.33 GB/s |
| 32 MiB | 18.19 / 16.46 GB/s | 17.56 / 15.42 GB/s |
| 64 MiB | 17.87 / 16.57 GB/s | 16.18 / 16.42 GB/s |

For four ranks, bus bandwidth is complete bytes per rank divided by time,
multiplied by `3/4`. It is a useful-byte collective metric, not one PCIe
link's wire rate.

AG has a lower median latency on 7/17 sizes, from 1 through 64 MiB, with
non-overlapping recorded timing ranges. RS has a lower median on 4/17 sizes;
its 4-MiB advantage is only 1.2%, while 8–32 MiB exceed 3% with
non-overlapping ranges. The small-message and 64-MiB RS losses remain in the
data. This benchmark does not install a runtime dispatch policy or establish
performance on another topology, dtype, in-place layout, or node count.
