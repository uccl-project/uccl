# Topology and hardware bandwidth profiler

For measured two-host GPU/NIC paths, see
[network calibration](network-calibration.md). That probe keeps effective path
rates separate from the advertised NIC capacities described below.

The runtime's passive topology discovery is also available independently of
CPU-router, MSCCL++ and NCCL:

```bash
make topology topology-check
./build/uccl_lite_topology_check
./build/uccl_lite_topology 0 1 2 3
```

Device arguments define rank order. The probe reports shared PCIe paths, NUMA
placement and InfiniBand port state/rate. PCIe and NIC rates are advertised
capabilities; memory bandwidth and effective transfer rates remain unknown
until calibrated. This command neither measures bandwidth nor starts routing
workers. See [runtime migration status](topology-runtime-refactor.md) for the
remaining planner and execution integration work.

[hardware_profiler.cu](../benchmarks/probes/hardware_profiler.cu) is a standalone
CUDA probe for planning local collective schedules. It reports each selected
GPU's PCI ID, sysfs path, NUMA node, current and maximum PCIe link speed/width,
and a CUDA peer-access matrix. Current PCIe speed can reflect an idle power
state; maximum link speed describes capability, not measured throughput.

The probe then runs all selected GPUs concurrently in each direction using
separate CUDA streams and pinned host buffers. It reports aggregate H2D and D2H
useful-byte bandwidth from a common host-clock interval. It measures each
directed GPU peer link separately, then launches a concurrent source-side
peer-copy ring in the specified order. It also pins CPU
workers to each NUMA node and measures concurrent memory copy and four-source
BF16 reduction. CPU buffers are allocated and first-touched on the pinned
workers. The reduction reads four BF16 arrays, accumulates in FP32, and writes
one rounded BF16 array. It is a hardware probe, not the CPU-router executor or
a prediction of collective latency.

```bash
make hardware_profiler
./build/uccl_lite_hardware_profiler
./build/uccl_lite_hardware_profiler --devices 0,1,2,3 \
  --peer-order 0,1,3,2 \
  --gpu-bytes 16777216 --cpu-bytes 8388608 \
  --iterations 10 --warmups 2 --repeats 3 --threads-per-node 4
```

`--gpu-bytes` is the buffer size per GPU and `--cpu-bytes` is one CPU output
buffer per worker. H2D and D2H are separate concurrent sweeps. Directed P2P
copies are measured one at a time; the ring has one source and one destination
per GPU concurrently. Peer copies use enabled CUDA peer access and separate
destination buffers. The probe does not measure simultaneous bidirectional
host traffic or arbitrary combinations of peer transfers.
It uses CUDA's pinned allocation for host copy buffers, whose NUMA placement is
not forced. CPU copy and reduction use first-touch node placement. CPU copy
counts one read plus one write; BF16 reduction counts four reads plus one write.
These are useful bytes per elapsed host time, not memory-controller counters.
Allocation, warmup, and checks are excluded. The common interval includes
submission and synchronization. Each row gives the minimum, median, and maximum
of complete repetitions. No per-size winner selection is performed.

## Recorded four-L4 result

On November 10, 2026, the command above ran on four NVIDIA L4s, PCIe Gen4
x16, two GPUs per NUMA node, dual Intel Xeon Silver 4410Y, CUDA runtime 12.9,
driver 595.58.03. It used 16 MiB per GPU, 8 MiB per CPU worker, four workers
per node, two warmups, ten timed iterations, three repetitions per run, and
three independent complete runs. There is no comparison baseline. The
[validated summary](../benchmarks/data/hardware_profile_4l4_20261110_summary.csv)
reports the median of the three run medians and the full min–max range. The
[raw runs 1](../benchmarks/data/hardware_profile_4l4_20261110_run1.txt),
[2](../benchmarks/data/hardware_profile_4l4_20261110_run2.txt), and
[3](../benchmarks/data/hardware_profile_4l4_20261110_run3.txt) include PCI
paths, peer matrices, every directed link, and per-run ranges.

| Concurrent workload | Median useful bandwidth |
| --- | ---: |
| Four-GPU H2D | 99.60 GB/s aggregate |
| Four-GPU D2H | 55.42 GB/s aggregate |
| Four-GPU peer ring `0→1→3→2→0` | 82.04 GB/s aggregate |
| Directed P2P links, separately | 22.63–22.64 GB/s each |
| CPU copy, NUMA node 0 / 1 | 69.50 / 48.72 GB/s |
| Four-source BF16 CPU reduction, NUMA node 0 / 1 | 47.56 / 48.98 GB/s |

The CPU copy ranges include slower first repetitions and a lower node-1 rate
than an earlier single run. These numbers characterize these runs, buffer
placement and thread count; they do not prove a router or
collective crossover. Use the paired collective calibration in
[CPU-router selection](cpu-router.md) to decide whether to route a workload.

A later [1.5-GiB-per-NUMA STREAM-style triad](../benchmarks/data/host_memory_bw_4l4_20261111/README.md)
measures sustained host DRAM useful bandwidth at about 87 GB/s per socket
with 12 physical-core threads, or about 175 GB/s for both sockets running
simultaneously. It uses a different kernel and larger working set than the
copy and BF16-reduction rows above.

Regenerate the summary without GPU work using:

```bash
python3 scripts/summarize_hardware_profile.py \
  benchmarks/data/hardware_profile_4l4_20261110_summary.csv \
  benchmarks/data/hardware_profile_4l4_20261110_run{1,2,3}.txt
```

## Persisted GPU resource calibration

```bash
make calibrate-hardware calibration-check
./build/uccl_lite_calibration_check
./build/uccl_lite_calibrate_hardware build/gpu-resources.profile 0 1 2 3
UCCL_LITE_HARDWARE_PROFILE=build/gpu-resources.profile ./build/uccl_lite_joint_peer_check 4 direct
```

The resource probe runs only at an explicit idle preparation boundary. It uses
private uint4 buffers and validates every copied value. GPU vector-copy memory
rates count reads plus writes; peer-copy and remote-read path rates count
one-way useful bytes. Each resource is measured separately at 16 bytes and
64 MiB, with two warmups, five iterations and three complete repetitions.
GPU events delimit the timed calls; allocation, warmup and validation are
excluded. The saved bandwidth/intercept is an affine fit of the two medians;
it is a workload-specific model, not a memory-controller counter or intrinsic
link latency. The file retains all repetition samples and probe build identity.
There is no baseline or best-variant envelope in this measurement.

Hardware profiles are applied only to matching discovery/software identity;
backend policy identity and paired collective admission remain separate.
Physical PCIe bridges and NIC line rates remain advertised, and CPU memory and
NUMA interconnects remain unknown until separately measured. In particular,
end-to-end P2P bandwidth is never assigned to every bridge along that path.
Peer preparation now automatically fills incomplete GPU resource data and
qualifies candidate algorithms on private buffers for the exact call. The
resource model supplies a scheduling prior; a candidate must beat the direct
baseline with a stable local timing advantage and pass full-buffer checks.
The fitted per-workload residual includes unmodeled execution costs and is not
an intrinsic launch-latency measurement. Enqueue never runs calibration.
Set PeerTuning.automaticCalibration=false only for explicit model/diagnostic
use; without sufficient supplied data that mode retains the direct baseline.

## Worker-scoped CPU and NUMA copy calibration

```bash
make calibrate-host host-calibration-check
./build/uccl_lite_host_calibration_check
./build/uccl_lite_calibrate_host build/host-copy.profile 0 1 2 3
```

The CPU probe uses private anonymous mappings with enforced source/destination
NUMA policy (`mbind`) and verifies resident placement before timing. Worker
threads run on explicitly recorded destination-node CPUs; the caller's affinity
is preserved. Each repetition compares every output byte and verifies input
preservation outside the common timing interval. Worker failures cancel waits
and join all threads.

Defaults are four workers per destination node (capped by permitted CPUs),
32/64 MiB per worker, two warmups, five iterations and three repetitions.
All host-copy path rates count **one-way payload bytes**, consistently for local
and cross-NUMA paths. Local memory read+write traffic is twice that amount.
The two-size model fits nonnegative fixed cost and positive time per byte;
raw repetitions, CPU IDs, probe build identity, memcpy build identity and libc
version are retained. It has no comparison baseline or per-size winner curve.
Run it without concurrent collective/performance jobs when collecting a profile.

Call `registerHostCopyPaths(topology, options)` with the same configuration
before applying a persisted host profile. CPU IDs are part of resource names,
so a result for one worker allocation cannot silently describe another one.
These end-to-end copy measurements do not populate generic memory-controller
bandwidth or physical NUMA-link bandwidth. Linux NUMA distances are recorded
as relative topology metadata, never interpreted as GB/s or microseconds.
CPU-router admission still requires its exact paired collective measurements;
copy probes alone do not enable it.
