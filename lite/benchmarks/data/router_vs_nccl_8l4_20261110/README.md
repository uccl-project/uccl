# Eight-L4 BF16 CPU router versus NCCL 2.28.9

This is the direct paired control for the current eight-GPU AllReduce and
AllGather README results. Its ReduceScatter data is retained as the earlier
control; the current RS figure uses a [new paired rerun](../reducescatter_8l4_20261110_v2/README.md).
The UCCL Lite candidate is the CPU router: the separate
[local/router sweep](../local_fallback_8l4_20261110_v3/README.md) measured it
faster than the project-owned local path at every 1 MiB–1 GiB AR/AG/RS size
in both placements. NCCL is an external control in this run, not the default
fallback of the production-style local dispatcher.

## Machine and method

- `wxz-l4`: eight NVIDIA L4s, GPUs 0–3 on NUMA node 0 and 4–7 on node 1;
  peer access within groups and unavailable across groups. Intel Xeon CPU
  reported as `@ 2.20GHz`, CUDA 12.9, driver 580.178.04, NCCL 2.28.9.
- BF16 AllReduce, AllGather, and ReduceScatter; in-place and out-of-place
  layouts measured separately, 1 MiB–1 GiB doubling complete bytes per rank.
  Each placement has 33 workload keys × two backends × three alternating
  paired repetitions × ten timed calls = **1,980 checked samples**. Each pair
  gets five warmups. Fill and full-output verification are outside timing;
  the latency is the slowest-rank CUDA-event interval.
- AllReduce through 4 MiB uses 128 KiB router packets; larger AllReduce and
  AG/RS use 512 KiB packets. Router allocation and NCCL communicator setup
  occur before timing. Min/max bound three run medians, not confidence limits.

| Placement | Operation | 1 MiB UCCL / NCCL (µs) | 1 GiB UCCL / NCCL (ms) |
| --- | --- | ---: | ---: |
| In place | AllReduce | 494.592 / 594.432 | 276.811 / 380.392 |
| In place | AllGather | 314.368 / 359.936 | 182.090 / 188.494 |
| In place | ReduceScatter | 394.240 / 351.232 | 170.489 / 202.710 |
| Out of place | AllReduce | 503.808 / 587.776 | 277.207 / 378.281 |
| Out of place | AllGather | 320.512 / 359.424 | 181.621 / 188.345 |
| Out of place | ReduceScatter | 346.624 / 337.920 | 171.896 / 202.479 |

ReduceScatter loses at 1 MiB in both placements; the complete CSVs retain
all sizes and losses. The README uses in-place AllReduce and out-of-place
AllGather/ReduceScatter, matching the conventional layouts of the other
topology records. Every calibration invocation passed the output check.

The `*.profile` files here identify NCCL as the measured fallback. They are
**not** valid evidence for selecting against the project-owned local fallback;
use the [local profile](../local_fallback_8l4_20261110_v3/README.md) for that
dispatcher.

## Reproduce

From `experimental/lite`, with new output paths:

```bash
make -j2 adaptive-collectives \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback nccl --op all \
  --in-place --begin 1048576 --end 1073741824 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/router-nccl-in-place-samples.csv \
  --calibrate build/router-nccl-in-place.profile
# Omit --in-place for the out-of-place sweep; use distinct output paths.
```
