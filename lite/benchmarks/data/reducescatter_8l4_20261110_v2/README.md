# Eight-L4 BF16 ReduceScatter rerun

This rerun checks the ReduceScatter points that previously trailed NCCL in the
single-node figure. The host has eight NVIDIA L4s: GPUs 0–3 are on NUMA node 0
and 4–7 on node 1. CUDA 12.9, NVIDIA driver 580.178.04, and NCCL 2.28.9 are
the same as in the earlier eight-L4 record. Each GPU has a complete BF16 input
tensor and a distinct output shard; all sizes below are complete input bytes
per rank.

The earlier default CPU-reduction router was repeatedly slower than NCCL at
128 KiB and 1–4 MiB. The [five-repetition default diagnostic](diagnostics/default.csv)
measured 237.568 / 173.056 µs at 128 KiB and 972.800 / 934.912 µs at 4 MiB
(router / NCCL). The run-median ranges do not overlap at 128 KiB. Existing
[GPU progress](diagnostics/gpu_progress_128.csv) cut the 128-KiB router to
151.040 µs versus 167.424 µs NCCL. Existing
[128-KiB packets](diagnostics/packet128.csv) reduced the 4-MiB router to
913.408 µs versus 926.720 µs NCCL. Each diagnostic checked every output.

The adaptive dispatcher now uses GPU reduction and GPU progress at exactly
128 KiB, and 128-KiB packets with CPU reduction from 256 KiB through 4 MiB.
It retains the 512-KiB packet limit elsewhere. A full-size
[ablation](ablation_all_small_packets/nccl.csv) showed no reliable paired gain
from smaller packets at 64 KiB, and its local comparison was slower there;
the final rule leaves smaller messages alone. The mode and packet settings
are included in the exact profile identity; old profiles cannot select this
changed path.

## Final paired measurement

The final [local/router calibration](local.csv) and [router/NCCL control](nccl.csv)
each cover 21 doubling sizes from 1 KiB to 1 GiB. Every size has five warmups
and five alternating paired repetitions of ten timed, output-checked calls per
backend: **2,100 checked samples per comparison**, or 4,200 total. Fill,
allocation, verification, and communicator setup are outside timing. Latency
is the slowest-rank CUDA-event interval; min/max bound five run medians and
are not confidence intervals. The [per-call local samples](local_samples.csv)
and [per-call NCCL-control samples](nccl_samples.csv) are retained. The final
`*.profile` files identify their different fallbacks; only `local.profile`
is valid for the project-owned local fallback.

| Complete bytes/rank | Final router / NCCL | Earlier router / NCCL |
| ---: | ---: | ---: |
| 128 KiB | 151.040 / 165.376 µs | 191.488 / 161.792 µs |
| 1 MiB | 340.480 / 353.280 µs | 346.624 / 337.920 µs |
| 2 MiB | 542.720 / 565.760 µs | 546.816 / 530.944 µs |
| 4 MiB | 916.480 / 978.944 µs | 976.896 / 911.872 µs |
| 1 GiB | 172.255 / 220.849 ms | 171.896 / 202.479 ms |

The earlier 128-KiB README curve selected the local path at 214.528 µs;
its paired router control was 191.488 µs. The final local comparison selects
the router at all 21 sizes. The final router beats NCCL at 20 of 21 sizes.
At 64 KiB, it measures 144.384 versus 143.872 µs, with overlapping five-run
ranges (142.336–148.480 versus 138.752–145.408 µs). That 0.5-µs difference
is retained as a near-tie: one paired repetition favored the router, one tied,
and three favored NCCL. It is not counted as a stable win. The full selected-path
result is in [the README summary](../collectives_8l4_20261110.csv).
An [automatic replay](auto.csv) with `local.profile` selected the router at
each affected 128 KiB–4 MiB size, checked every output, and initialized the
router at the explicit `prepare()` boundary.
The host policy and dispatcher checks passed. On this eight-GPU topology,
the 128-KiB-packet CPU router and GPU-progress router each passed 732 eager
mixed-operation checks and 1,000 captured mixed collectives across five graph
replays. The local fallback passed 61 eight-GPU BF16 cases.

## Reproduce

From `experimental/lite`, with new output paths:

```bash
make -j2 adaptive-collectives \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback local \
  --op reducescatter --begin 1024 --end 1073741824 \
  --warmup 5 --iterations 10 --repeats 5 \
  --samples build/rs-local-samples.csv --calibrate build/rs-local.profile
# Repeat with --fallback nccl and distinct output paths.
```
