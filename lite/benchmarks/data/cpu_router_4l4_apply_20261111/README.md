# Applying the 8×L4 BF16 CPU router on this 4×L4 host (2026-11-11)

The [worker-count follow-up](../cpu_router_4l4_workers_20261111/README.md)
reduces the AllGather-only configuration to eight physical-core workers per
NUMA node and rechecks the AR/RS losses. The measurements below remain the
original 23-worker-per-NUMA calibration.
The [archived BF16 figures](../cpu_router_4l4_figures_20261111/README.md)
show the router at every size, including losses. NCCL is only a comparison
curve; the automatic replay below is historical diagnostic data.

This is a fresh single-node experiment on `mibura-sky-test-02`: four NVIDIA
L4 GPUs, GPU 0/1 on NUMA 0 and GPU 2/3 on NUMA 1. Unlike the eight-L4
reference host, **every directed GPU pair has CUDA peer access**, including
across NUMA nodes. The complete [hardware and software record](hardware_software.txt)
includes the topology, driver, CUDA, binary hash, and NCCL 2.28.7 library hash.

The current 8×L4 CPU-router module already supports four ranks. We rebuilt
`make adaptive-collectives router-policy-check router-checks
local-collectives-check` and ran its BF16 AllReduce, AllGather, and
ReduceScatter through the one-process/four-GPU `uccl_lite_collectives`
harness. The paired control is NCCL 2.28.7 with its default local configuration
and no `NCCL_*` overrides. This is a **BF16** experiment; the main four-L4
FP32 figures are not a dtype-matched comparison and remain unchanged.

Both out-of-place and in-place sweeps cover every power-of-two **complete
tensor byte count per GPU** from 1 KiB to 1 GiB. Each backend gets five
warmups and three backend-alternating rounds of ten timed calls per size.
The timer is the maximum across four GPU CUDA-event durations from stream
start to stream completion; fill and full-output verification are outside
the timer. Every warmup and timed call checked every output element. The
router defaults are 23 CPU workers per NUMA group, 64 transport blocks,
512 KiB packet cap, 128 KiB small AR/RS packets, GPU-progress RS at exactly
128 KiB, and 2 MiB AG packets from 4 MiB upward. `--calibrate` forces both
backends for measurement; it does not silently select the router.

## Paired performance

All times are µs, median of three run medians. `summary.csv` retains the
three-run min/max for every backend and size. NCCL/router is greater than 1
when the router is faster. These are representative points; the CSV has all
21 sizes per operation and placement.

| Placement | Collective | Complete bytes/GPU | CPU router | NCCL | NCCL/router |
| --- | --- | ---: | ---: | ---: | ---: |
| Out of place | AllReduce | 1 KiB | 42.90 | 44.03 | 1.03× |
| Out of place | AllReduce | 1 MiB | 471.04 | 154.62 | 0.33× |
| Out of place | AllReduce | 1 GiB | 86,841.34 | 86,480.38 | 1.00× |
| Out of place | AllGather | 1 KiB | 43.01 | 38.91 | 0.90× |
| Out of place | AllGather | 512 KiB | 79.87 | 88.06 | 1.10× |
| Out of place | AllGather | 64 MiB | 2,807.30 | 3,110.91 | 1.11× |
| Out of place | AllGather | 1 GiB | 42,879.49 | 48,318.46 | 1.13× |
| Out of place | ReduceScatter | 1 MiB | 172.03 | 94.21 | 0.55× |
| Out of place | ReduceScatter | 1 GiB | 54,724.10 | 48,282.75 | 0.88× |
| In place | AllGather | 512 KiB | 79.36 | 88.06 | 1.11× |
| In place | AllGather | 1 GiB | 42,874.88 | 47,197.69 | 1.10× |
| In place | AllReduce | 1 MiB | 394.75 | 151.55 | 0.38× |
| In place | ReduceScatter | 1 MiB | 154.62 | 94.21 | 0.61× |

The router has a stable, at least 3% AllGather gain with non-overlapping
run-median ranges at **512 KiB and 16/32/64/128/256/512 MiB and 1 GiB**
in both placements: eight of 21 sizes. Its out-of-place 1-GiB AllGather
bus throughput is 18.78 GB/s versus NCCL's 16.67 GB/s, using the four-rank
AllGather bus factor `3/4`. No AllReduce or ReduceScatter size meets that
stable-gain rule. The 1-KiB out-of-place AR median is slightly lower than
NCCL, but its run ranges overlap and the in-place result loses. Small AG
also loses; the older FP32 four-L4 host-staged candidate is a different
implementation and dtype.

## Tried 8×L4 tuning variants

These are separate, paired **out-of-place** runs with 20 warmups and the
same three rounds of ten checked calls. Their NCCL control is measured in
each variant run. The complete negative results are in `summary.csv`.

| Variant | Representative result | Finding |
| --- | --- | --- |
| Four workers/NUMA, 16 blocks, compact AR | 256 KiB: 248.32 vs NCCL 88.06 µs | Loses; the 8×L4 compact setting does not transfer |
| GPU reduction/progress AR | 256 KiB: 90.11 vs NCCL 89.09 µs | Near tie; no stable gain |
| GPU reduction/progress RS | 256 KiB: 67.58 vs NCCL 63.49 µs | Loses |
| Interleaved GPU scatter RS | 256 KiB: 80.90 vs NCCL 64.51 µs | Loses |

CUDA reports peer access between all four GPUs, but the later
[NCCL transport probes](../cpu_router_4l4_workers_20261111/README.md)
show that NCCL selected `SHM/direct/direct` for every Ring edge in this
harness. Peer capability alone therefore does not explain the AR/RS loss.
Both paths use host memory, with different packet schedules; the router also
does CPU arithmetic on its CPU-reduction paths. A profiler would be needed
to split their critical-path costs precisely. Large AG benefits from the
router's upload-once host packets.

## Replay and artifacts

The fresh out-of-place and in-place calibration profiles were replayed in
new processes. [`auto_selection.csv`](auto_selection.csv) validates all 126
operation/size/placement keys: the automatic policy chooses the CPU router
at the same eight AllGather sizes in both placements and NCCL at every AR/RS
size. Replay times use one call per round and are **correctness/selection
checks**, not performance estimates.

- `out_of_place.log`, `in_place.log`: full paired raw logs; matching
  `*_samples.csv` files contain 3,780 timed calls each, and `*.profile` files
  contain the exact local calibration.
- `ar_compact.*`, `ar_gpu_progress.*`, `rs_gpu_progress.*`, and
  `rs_striped.*`: negative variant logs, samples and profiles.
- `pilot_1k.*`: initial 1-KiB correctness/performance check.
- `out_auto.*`, `in_auto.*`: fresh-process automatic replay logs and
  per-call samples.
- `check_*.log`: router policy, dispatcher, core, four-GPU eager,
  four-GPU graph, and GPU-progress correctness checks; all exit successfully.

Run `python3 summarize.py` here to validate 8,940 paired timed samples,
rebuild [`summary.csv`](summary.csv), and validate the 126 replay decisions.
To reproduce a fresh out-of-place sweep from `experimental/lite` after
building, choose new output paths and run:

```bash
./build/uccl_lite_collectives --gpus 4 --fallback nccl --op all \
  --begin 1024 --end 1073741824 --warmup 5 --iterations 10 --repeats 3 \
  --large-ag-packet-bytes 2097152 \
  --calibrate build/new-4l4-out.profile \
  --samples build/new-4l4-out-samples.csv
```

Add `--in-place` and use distinct output paths for the second placement.
This original calibration did not change the main README at the time. The
later [archived BF16 figures](../cpu_router_4l4_figures_20261111/README.md) use fresh
paired runs and show the router at every size.
