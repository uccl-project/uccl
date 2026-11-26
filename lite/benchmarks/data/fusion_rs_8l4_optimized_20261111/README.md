# Eight-L4 GEMM + ReduceScatter after prefill scheduling fixes

BF16, eight L4 GPUs (0–3 and 4–7 on separate NUMA nodes), driver 580.178.04,
CUDA 12.9.41, CUTLASS 4.5.0, local NCCL 2.28.9 shim. All methods use 512 KiB
router packets where applicable. Each sample is the slowest GPU's CUDA-event
latency for GEMM plus ReduceScatter. The benchmark checks every output element
on all eight GPUs after each call, outside timing.
The model-derived `(N,K)` pairs are 9B `(4096,1536)`, 27B `(5120,2176)`,
Gemma `(5376,2688)`, and 32B `(5120,3200)`.

Each shape has two runs in opposite shape order, with two warmup rounds and
50 measured calls per method per run. Values below are medians of the two
per-run medians. The same automatic fused schedule is used for every shape;
the chart does not select a best variant per point.

| Model | M | Sequential CPU router (ms) | Fused CPU router (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 2048 | 3.156 | 3.078 | 1.025× |
| Qwen3.5-9B | 4096 | 6.100 | 6.050 | 1.008× |
| Qwen3.5-9B | 8192 | 12.036 | 11.794 | 1.020× |
| Qwen3.5-27B | 2048 | 4.094 | 3.916 | 1.045× |
| Qwen3.5-27B | 4096 | 7.864 | 7.581 | 1.037× |
| Qwen3.5-27B | 8192 | 15.650 | 14.793 | 1.058× |
| Gemma 4 31B IT | 2048 | 4.330 | 4.157 | 1.042× |
| Gemma 4 31B IT | 4096 | 8.456 | 8.158 | 1.036× |
| Gemma 4 31B IT | 8192 | 16.773 | 15.698 | 1.068× |
| Qwen3-32B | 2048 | 4.390 | 4.031 | 1.089× |
| Qwen3-32B | 4096 | 8.332 | 7.707 | 1.081× |
| Qwen3-32B | 8192 | 16.445 | 14.933 | 1.101× |

Fused beats sequential CPU router in both runs at all 12 prefill shapes.
It also beats the sequential router in both runs at all 12 decode shapes
(`M=1,32,64`); NCCL wins some decode points.

The original 512 KiB fused prefill path assigned up to 56 of 58 resident CTAs
to transport, leaving two for GEMM. The optimized path uses a 128×256 GEMM
tile, caps upload and download crews by shape, and uses eight CPU reduction
lanes for short prefill. The [historical sweep](../fusion_8l4_models_ag_rs_20261111/README.md)
remains separate.

The run logs' `active CPU<=8` line describes scratch sizing, not the regular
CPU lane cap; the label was corrected after measurement without changing
scheduling.

[Prefill bars](../../../docs/assets/fusion-8l4-model-gemm-rs-prefill-optimized-labeled-bars.svg) ·
[decode bars](../../../docs/assets/fusion-8l4-model-gemm-rs-decode-optimized-labeled-bars.svg) ·
[summary CSV](summary_gemm_rs.csv) ·
[run 1 samples](samples_gemm_rs_run1.csv) ·
[run 2 samples](samples_gemm_rs_run2.csv) ·
[run 1 log](gemm_rs_run1.log) ·
[run 2 log](gemm_rs_run2.log)

From `experimental/lite`, reproduce with:

```bash
make -j2 build/uccl_lite_router_fusion_8l4 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
python3 benchmarks/data/fusion_rs_8l4_optimized_20261111/run.py --op gemm-rs --run 1
python3 benchmarks/data/fusion_rs_8l4_optimized_20261111/run.py --op gemm-rs --run 2
python3 scripts/plot_router_fusion_8l4_models_ag_rs.py --op gemm-rs \
  --data-dir benchmarks/data/fusion_rs_8l4_optimized_20261111 --tag optimized
```
