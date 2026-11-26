# Eight-L4 NCCL control calibration — November 10, 2026

This is the NCCL-backed, single-node BF16 rerun at commit `d12a9562` on
`wxz-l4`: eight NVIDIA L4 GPUs, four per NUMA node, Intel Xeon CPU reported as
`@ 2.20GHz`, driver 580.178.04, CUDA toolkit/runtime 12.9, and NCCL 2.28.9
(`+cuda13.0` library). The shell's preconfigured `NCCL_NET=gIB` was cleared;
the fallback was NCCL with its default local transport. The CUDA peer matrix
allows peers within each four-GPU NUMA group and none across groups. The sysfs
PCIe speed/width fields were unavailable, so measured bandwidth is the link
evidence. This setup differs from the older eight-L4 no-P2P archive.
The installed NCCL 2.27.5 build failed initialization with a CUDA named-symbol
error on this driver; it produced no timing rows and was not used as a baseline.
The adaptive example now defaults to a [UCCL Lite local fallback](../local_fallback_8l4_20261110/README.md).
These NCCL-backed profiles cannot select a backend against it; the timings
remain an external control.

## Files

| File | Contents |
| --- | --- |
| [hardware_profile.txt](hardware_profile.txt) | Eight-GPU peer matrix, directed copies, concurrent H2D/D2H, and NUMA CPU copy/BF16 reduction; 16 MiB per GPU, 8 MiB per CPU worker, four workers per node, two warmups, ten timed iterations, three repetitions. |
| [in_place_calibration.csv](in_place_calibration.csv), [out_of_place_calibration.csv](out_of_place_calibration.csv) | Paired NCCL/router min/median/max run-level times, speedup and correctness result for all three collectives, 1 MiB–1 GiB doubling sizes. |
| [in_place_samples.csv](in_place_samples.csv), [out_of_place_samples.csv](out_of_place_samples.csv) | Every slowest-rank CUDA-event time: 11 sizes × three collectives × two backends × three repetitions × ten timed calls = 1,980 rows per placement. |
| [in_place.profile](in_place.profile), [out_of_place.profile](out_of_place.profile) | Exact local policy records produced by calibration; placement and environment identities are separate. |
| [in_place_auto.csv](in_place_auto.csv), [out_of_place_auto.csv](out_of_place_auto.csv) | Fresh-process automatic selection and correctness replay for all 33 workload keys per placement. The replay uses one timed call per repetition; its times are not performance summaries. |
| [in_place_1g_ag_auto.csv](in_place_1g_ag_auto.csv) | Fresh-process 1-GiB in-place AllGather fallback check: NCCL selected with `dispatcher_router_initialized=0`. |

The `.log` companions record selection reasons and any errors. All completed
calibrations and replays end with `# correctness: PASS (every invocation, all
ranks)`. The paired benchmark alternates backend order by repetition, reports
the median of three ten-call medians, and times the slowest GPU rank using CUDA
events. Fill and verification run outside each timed interval. The benchmark
checks every result but is still a standalone eager collective test, not an
application benchmark. The CSV's `bytes` is the complete per-rank tensor for
AllGather/ReduceScatter and the per-rank array for AllReduce. No per-size
algorithm variant is chosen when presenting router and NCCL timings.

The [six-panel bandwidth figure](../../../docs/assets/cpu-router-8l4.svg)
uses these NCCL median times as a dashed external control. It converts
latencies to decimal bus bandwidth using `14/8` for AllReduce and `7/8` for
AllGather/ReduceScatter. Blue and teal curves come from the separate paired
UCCL Lite local/router rerun. Regenerate it with
`python3 scripts/plot_cpu_router_8l4.py` from `experimental/lite`.

| Placement | AllReduce router choices | AllGather router choices | ReduceScatter router choices | Total |
| --- | ---: | ---: | ---: | ---: |
| In place | 9/11 | 7/11 | 8/11 | 24/33 |
| Out of place | 9/11 | 4/11 | 7/11 | 20/33 |

The policy requires at least a 1.03× median speedup and the router's slowest
run below NCCL's fastest. A fresh replay of each profile produced the counts
above. In a combined replay, a fallback row may show an initialized router if
an earlier size selected it; the single-key 1-GiB AllGather replay confirms a
fallback-only process does not initialize router workers.

## Reproduction

The installed NCCL 2.28.9 package had `libnccl.so.2` but no linker-name
symlink. The local `build/nccl-2.28.9` shim supplied the matching headers and
linker name. From `experimental/lite`:

```bash
NCCL_PKG=/data/zhongjie/vllm-dsv4-venv/lib/python3.12/site-packages/nvidia/nccl
mkdir -p build/nccl-2.28.9/lib
ln -s "$NCCL_PKG/include" build/nccl-2.28.9/include
ln -s "$NCCL_PKG/lib/libnccl.so.2" build/nccl-2.28.9/lib/libnccl.so
ln -s "$NCCL_PKG/lib/libnccl.so.2" build/nccl-2.28.9/lib/libnccl.so.2
make -B -j2 hardware_profiler cpu-router adaptive-collectives router-policy-check \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9

./build/uccl_lite_hardware_profiler --devices 0,1,2,3,4,5,6,7 \
  --gpu-bytes 16777216 --cpu-bytes 8388608 \
  --iterations 10 --warmups 2 --repeats 3 --threads-per-node 4

# Use new output paths; --calibrate and --samples reject existing files.
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback nccl --op all --in-place \
  --begin 1048576 --end 1073741824 --warmup 5 --iterations 10 --repeats 3 \
  --samples build/new-in-place-samples.csv --calibrate build/new-in-place.profile

# Omit --in-place and choose distinct files for the out-of-place sweep.
# For selection replay, replace --calibrate with --profile and use
# --warmup 1 --iterations 1 --repeats 3.
```

The profiler ran before calibration. Its median aggregate H2D/D2H rates were
49.30/50.98 GB/s; directed within-node P2P rates spanned 9.89–10.27 GB/s.
The default cross-node peer ring was skipped because GPU 3 cannot peer with
GPU 4. The profiler measures hardware paths, while paired collective timing
provides the selection evidence.
