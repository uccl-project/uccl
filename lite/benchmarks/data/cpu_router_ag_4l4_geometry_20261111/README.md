# Four-L4 BF16 AllGather: CPU-router transfer geometry (2026-11-11)

The [archived BF16 figure](../cpu_router_4l4_figures_20261111/README.md)
plots the CPU router at **every size**, including the negative points below.
NCCL is a paired external control only. The older automatic replay retained
here is diagnostic and is not used to select the plotted UCCL path.

This follow-up diagnoses why the [original host-packet CPU router](../cpu_router_4l4_workers_20261111/README.md)
only modestly beat NCCL SHM Ring at 1 GiB on `mibura-sky-test-02`. The tested
algorithm is unchanged: each of four GPUs uploads its BF16 contribution once
to NUMA-local mapped host memory, then each GPU reads the other three
contributions. For this Copy/Concatenate path, the CPU worker pool may contain
eight threads per NUMA group, but the router dispatches **one active copy
task per NUMA group** per call; those workers publish readiness and do not
copy the payload. This is separate from the main four-L4 FP32 figure.

All accepted performance results use the same one-process/four-GPU harness
and paired NCCL 2.28.7 default control. Each size and backend has 20 warmups,
three backend-alternating rounds of ten timed calls, complete BF16 output
checks on every call, and slowest-GPU CUDA-event timing. Out-of-place and
in-place results are separate. [Hardware/software identity](hardware_software.txt)
records the exact executable/NCCL hashes, topology and versions.
`commands*.json` retain launch commands and order. The
[per-size summary](summary.csv) is rebuilt by `python3 summarize.py` from
**3,900 paired timed calls**; the same script validates all 126 fresh-process
automatic decisions. Nsight traces below are diagnostics, not performance
numbers.

## Confirmed bottleneck and geometry changes

1. The AllGather `packetCapacity` helper returns **16-bit storage units**.
   Its `packet_cap / (2 × 4 ranks)` result therefore corresponds to
   `packet_cap / 4` bytes per source/tile. At 1 GiB complete output/GPU,
   each source contributes 256 MiB. The old 2 MiB cap yields
   **512 KiB/source/tile and 512 tiles**; the new 4 MiB cap yields
   **1 MiB/source/tile and 256 tiles**. A 1-GiB, 64-block screen measured
   **44.93 ms at a 1 MiB cap**,
   **42.73 ms at 2 MiB**, **42.31 ms at
   4 MiB**, and **42.13 ms at 8 MiB**. Packet size alone saved less than
   1 ms relative to the old 2-MiB cap. The 16/32-MiB caps regressed in the
   `--transport-blocks 248` screen, so the final packet cap is 4 MiB.
2. The GPU transfer grid was the larger issue. At 8-MiB packet cap with GPU
   progress, raising `--transport-blocks` from 16 to 64 to 128 to 248
   changed the paired 1-GiB router medians from **47.89 → 41.97 → 41.23 →
   38.91 ms**. At the 248-block cap, a separate packet screen found a stable
   **36.88 ms** at 4 MiB. A CPU-progress check took **36.92 ms** with the
   same 248-block/4-MiB geometry, so GPU progress itself was not the main
   improvement. A one-worker-per-NUMA probe at the older 64-block geometry
   did not materially improve latency either.
3. Nsight Systems shows the router's persistent `dispatchTransport` GPU
   kernel at about **43.0 ms** with an actual 64-block grid/8 MiB cap and
   **37.7 ms** with an actual **180-block grid**/4 MiB cap. The configured
   `--transport-blocks 248` is an upper bound; occupancy and packet geometry
   limit the final launch to 180 blocks. The separately profiled NCCL kernel is
   `ncclDevKernel_AllGather_RING_LL`; the [transport trace](../cpu_router_4l4_workers_20261111/nccl_shm_ring_simple_probe.log)
   confirms SHM links. Nsight profiler timings are not substituted into the
   checked paired comparisons. These measurements place the main adjustable
   cost in the GPU transport kernel, not in CPU payload copying.

## Full-sweep result and automatic selection

The final **CPU-progress, 8-workers/NUMA, 248-block upper bound, 4-MiB-cap** configuration
was freshly paired with NCCL from 1 KiB–1 GiB in both placements. The full
sweeps passed complete output checks. Numbers below are out-of-place
run-median latencies; in-place is in `summary.csv`.

| Complete output/GPU | Router | NCCL | NCCL/router | Router outcome |
| ---: | ---: | ---: | ---: | --- |
| 4 MiB | 309.76 µs | 240.74 µs | 0.78× | Slower |
| 8 MiB | 487.94 µs | 418.30 µs | 0.86× | Slower |
| 16 MiB | 777.73 µs | 780.29 µs | 1.00× | Tied |
| 32 MiB | 1.373 ms | 1.545 ms | 1.13× | Faster |
| 64 MiB | 2.546 ms | 3.064 ms | 1.20× | Faster |
| 128 MiB | 4.898 ms | 6.067 ms | 1.24× | Faster |
| 1 GiB | **37.577 ms** | **48.120 ms** | **1.28×** | Faster |

The in-place 1-GiB point is **37.708 vs 48.614 ms**, also **1.29×**.
The original eight-worker, 64-block/2-MiB-cap AG-only profile reported
42.819 ms out of place and 42.922 ms in place at 1 GiB, with same-version
NCCL controls at 48.112/49.431 ms. These are separate benchmark windows,
not an interleaved old-versus-new comparison; the new out-of-place router
median is about **12% lower** and its paired NCCL control is essentially
unchanged.

The archived fresh-process [`auto_selection.csv`](auto_selection.csv) is
an earlier policy diagnostic. It would have selected the router at 512 KiB
and 32–1024 MiB out of place, with 16 MiB added in place; it is **not the
UCCL execution or figure policy**. The current figure always runs the
router, including 4/8 MiB losses. The paired measurement profiles are
`ag_full_cpu_blocks248_packet4m_out.profile` and
`ag_full_cpu_blocks248_packet4m_in.profile`. They are valid only for the
recorded exact host/software/configuration key; the module's global default
and archived FP32 figures are unchanged.

## Interpretation and artifacts

Both router and SHM Ring still move about `3B` host→GPU for a complete
AllGather output `B` per GPU. The router reduces GPU→host payload from
`3B` to `B`, but that upload can overlap the common host→GPU read.
At 1 GiB, the final router's logical 3 GiB host→GPU payload over 37.577 ms
is **85.7 GB/s aggregate**. The earlier four-GPU H2D copy-engine probe
measured 99.6 GB/s aggregate with a different transfer mechanism and
working set. Thus the remaining host-read/PCIe ceiling is a plausible
limit; it is **an inference**, not a hardware-counter attribution.

The data directory retains every packet, block-count, CPU/GPU-progress,
one-worker, and full-sweep paired log, sample CSV, and exact calibration
profile, including losses. It also retains the raw Nsight `.nsys-rep`
reports, SQLite exports, and [`nsys_kernel_summary.csv`](nsys_kernel_summary.csv).
The full-sweep logs and per-call samples are
`ag_full_cpu_blocks248_packet4m_{out,in}.*`; automatic replays use the same
prefix with `_auto_`. Run `python3 summarize.py` here to check and regenerate
the summaries. For a fresh out-of-place calibration from `experimental/lite`:

```bash
./build/uccl_lite_collectives --gpus 4 --fallback nccl --op allgather \
  --begin 1024 --end 1073741824 --warmup 20 --iterations 10 --repeats 3 \
  --workers-per-node 8 --transport-blocks 248 \
  --large-ag-packet-bytes 4194304 \
  --calibrate build/new-ag-geometry.profile \
  --samples build/new-ag-geometry-samples.csv
```
