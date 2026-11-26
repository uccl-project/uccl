# NCCL 256 KiB–1 MiB latency spike on two 4×L4 nodes

This diagnosis uses the same eight FP32 ranks, GPU/NIC placement, NCCL
2.28.7 library, and custom checked control as the [paired two-node sweep](../multinode_serial_20261111/README.md).
Each operation is timed from host enqueue through completion using the
slowest rank. All output elements are checked on every invocation.

## Confirmed protocol crossover

With `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`, the default control selects
`RING + LL` on two channels at the spike sizes. For AllGather and
ReduceScatter, the tuning log reports the **one-rank shard bytes**, which are
one eighth of the full bytes per rank used in our figures. AllReduce reports
the full bytes per rank. `tuning_selection.csv` retains the exact mapping.

| Collective | Default RING+LL through full bytes/rank | Switch to RING+SIMPLE |
| --- | ---: | ---: |
| AllReduce | 1 MiB | 2 MiB |
| AllGather | 512 KiB | 1 MiB |
| ReduceScatter | 512 KiB | 1 MiB |

Three alternating default/SIMPLE rounds at the exact same size use 20 warmups
and ten checked timed calls per round. Both configurations select `RING` on
channels `0..1`; only the protocol differs. Medians of the three run means:

| Full bytes/rank | Collective | Default LL | Forced SIMPLE |
| ---: | --- | ---: | ---: |
| 512 KiB | AllReduce | 1495.23 µs | 232.21 µs |
| 512 KiB | AllGather | 2083.72 µs | 100.37 µs |
| 512 KiB | ReduceScatter | 2226.22 µs | 90.42 µs |
| 1 MiB | AllReduce | 693.71 µs | 206.87 µs |

`paired/` contains all 24 raw logs. `paired_samples.csv` contains the 240
timed calls; `paired_summary.csv` contains run ranges and medians. Recreate
both CSVs and `tuning_selection.csv` with
`python3 scripts/summarize_nccl_spike.py benchmarks/data/nccl_spike_2x4l4_20261111`
from `experimental/lite`.

## Interpretation and limits

The immediately demonstrated cause is the **default protocol choice**:
this NCCL build keeps the low-latency `LL` protocol into a size range where
`SIMPLE` is much faster on these two nodes. The algorithm and channel count
stay fixed in the controlled comparison. NCCL's [tuning source](https://github.com/NVIDIA/nccl/blob/ae7aed194dc63c65d1bf5c0385ba3d68d3b64c8c/src/graph/tuning.cc#L584-L599)
uses an estimated latency plus bytes/bandwidth cost; this host's measured LL
cost is much higher than that selection implies. NVIDIA describes LL as
latency-oriented and SIMPLE as bandwidth-oriented in its
[NCCL tuning guide](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/).

The exact internal LL stall is **not isolated**. Its [implementation](https://github.com/NVIDIA/nccl/blob/ae7aed194dc63c65d1bf5c0385ba3d68d3b64c8c/src/device/prims_ll.h#L40-L145)
uses flag-bearing FIFO lines and polls for the expected flags; this makes
progress and buffer reuse plausible contributors, but the logs do not prove
which wait dominates. In one AllGather 512-KiB run, LL warmups 2–10 were
mostly about 115–150 µs; later calls became 1.8–2.4 ms. SIMPLE remained
near 100 µs throughout. Increasing `NCCL_LL_BUFFSIZE` to 4 MiB sometimes
helped and sometimes regressed (`buffer_paired/`), so buffer size alone is
not a validated fix. One-run LL probes remained slow with GDR disabled,
IB replaced by sockets, or local P2P disabled (`transport/`); those probes
do not establish a transport-independent root cause.

The [later tuned NCCL policy](../nccl_tuned_2x4l4_20261111/README.md)
freshly pairs RING+SIMPLE at the affected sizes and supplies the current
main figures. This diagnosis and the complete default-NCCL sweep preserve
the original spikes. Hardware, software versions, binary/library hashes,
and the original baseline launch configuration are in the linked two-node
record.
