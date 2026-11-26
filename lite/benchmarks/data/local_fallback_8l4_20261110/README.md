# Eight-L4 UCCL Lite fallback calibration — November 10, 2026

This rerun changes the standalone BF16 adaptive example's **fallback** from
NCCL to the project-owned [two-NUMA local collective](../../../include/uccl_lite/local_collectives.hpp).
The dispatcher still requires paired local evidence before choosing the CPU
router. An older NCCL-backed profile cannot match the new fallback identity.
NCCL 2.28.9 remains a [separate comparison](../cpu_router_8l4_20261110/README.md),
never the automatic fallback in this run.

## Machine and method

- `wxz-l4`, eight NVIDIA L4s, GPUs 0–3 on NUMA node 0 and 4–7 on node 1;
  peer access exists within each group and is unavailable across groups.
- Intel Xeon CPU reported as `@ 2.20GHz`, driver 580.178.04, CUDA 12.9,
  CUTLASS 4.5.0. The optional control library is NCCL 2.28.9.
- BF16 AllReduce, AllGather, and ReduceScatter; in-place and out-of-place
  layouts measured separately, 1 MiB–1 GiB doubling complete bytes per rank.
- For each placement: 33 workload keys × two UCCL Lite backends × three
  alternating paired repetitions × ten timed calls = **1,980** checked timing
  samples, with five warmups before each pair. The benchmark reports the
  slowest-rank CUDA-event time; fill and verification are outside timing.
- Each reported median is the median of three ten-call medians. `min_us` and
  `max_us` bound those three run medians, not confidence intervals. The figure
  draws both measured backends at every size and does not select a fastest
  variant for its curves.

The local fallback stages AllGather shards through NUMA-pinned host buffers
and reads them directly on each destination GPU. AllReduce/ReduceScatter first
sum within each four-GPU peer group, exchange group partials through pinned
host memory, then distribute results. This eager implementation completes
submitted streams before returning. CUDA-event intervals include those host
submission gaps. `prepare()` reserves two full-tensor pinned host buffers and,
on the two group leaders, five full-tensor GPU buffers each (2 GiB pinned host
plus 10 GiB GPU scratch at a 1-GiB maximum). Allocation is outside timing.
This is not a CUDA Graph backend.

| File | Contents |
| --- | --- |
| [in_place_calibration.csv](in_place_calibration.csv), [out_of_place_calibration.csv](out_of_place_calibration.csv) | Paired local/router run-level min/median/max, speedup and correctness, all sizes and collectives. `fallback` means UCCL Lite local. |
| [in_place_samples.csv](in_place_samples.csv), [out_of_place_samples.csv](out_of_place_samples.csv) | Every slowest-rank timed call, 1,980 rows per placement. |
| [in_place.profile](in_place.profile), [out_of_place.profile](out_of_place.profile) | Exact local policy evidence, bound to the new fallback identity and placement. |
| [in_place_auto.csv](in_place_auto.csv), [out_of_place_auto.csv](out_of_place_auto.csv) | Fresh-process automatic selection and checked replay, all 33 keys per placement. Their 1 × 3 timing runs check routing, not performance. |
| [in_place_1g_ag_auto.csv](in_place_1g_ag_auto.csv) | Fresh-process 1-GiB in-place AllGather fallback check after the final build: UCCL Lite local selected with `dispatcher_router_initialized=0`. |

The `.log` files contain selection reasons. Every completed calibration and
replay ends with `# correctness: PASS (every invocation, all ranks)`.
The final rebuild added input and capture guards, safe retry after a failed
prepare, and changed the CLI default to eight GPUs after the full sweeps; none
changed the timed transfer schedule. The separate 1-GiB AllGather replay used that
rebuild and accepted the saved
profile. The dedicated correctness check also used the final build and verified
capture rejection.

| Placement | Router choices | Local fallback choices |
| --- | ---: | ---: |
| In place | 30/33 | AllGather 256 MiB, 512 MiB, 1 GiB |
| Out of place | 31/33 | AllGather 512 MiB, 1 GiB |

At 1 GiB, the paired median local/router times are 1500.25/275.28 ms for
in-place AllReduce, 178.53/181.22 ms for AllGather, and 951.08/168.61 ms for
ReduceScatter. The out-of-place counterparts are 1498.57/277.21,
179.32/180.60, and 953.43/169.88 ms. The local AllReduce and ReduceScatter
are intentionally retained as negative results; the router wins all measured
sizes for both. The [six-panel figure](../../../docs/assets/cpu-router-8l4.svg)
adds the separate NCCL control for context. At 1 MiB, the selected router
AllReduce remains slower than the NCCL control, so this work does not claim a
universal speedup.

## Reproduce

The NCCL 2.28.9 package on this machine was exposed to the Makefile through
the `build/nccl-2.28.9` header/library symlinks documented in the
[control record](../cpu_router_8l4_20261110/README.md). From
`experimental/lite`, with new output paths:

```bash
make -j2 cpu-router local-collectives adaptive-collectives \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9

env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --in-place --begin 1048576 --end 1073741824 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/new-in-place-samples.csv --calibrate build/new-in-place.profile

# Omit --in-place and choose distinct files for the out-of-place sweep.
# For selection replay, replace --calibrate with --profile and use
# --warmup 1 --iterations 1 --repeats 3.
python3 scripts/plot_cpu_router_8l4.py
```

The calibration profiles are exact to this host, fallback algorithm version,
placement and software identity. Recalibrate after changing the local path,
router geometry or execution environment.
