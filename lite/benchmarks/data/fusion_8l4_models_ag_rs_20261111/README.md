# Eight-L4 model-size AG+GEMM and GEMM+RS sweep (2026-11-11)

This sweep reuses the four model-derived `N,K` pairs from the
[GEMM+AR sweep](../fusion_8l4_models_20261111/README.md) to compare fusion
operations at the same tensor dimensions. These isolated operations are not
full model layers. Each BF16 run uses eight L4 GPUs split across two NUMA
groups (GPUs 0–3 and 4–7), driver 580.178.04, CUDA 12.9.41, CUTLASS 4.5.0,
and the local NCCL 2.28.9 shim. The router uses 512 KiB packets.

- `gemm-rs`: each GPU computes `A[M,K] × B[K,N]`; ReduceScatter returns
  `M × N / 8` output elements per GPU. Decode `M=1,32,64` and prefill
  `M=2048,4096,8192`.
- `ag-gemm`: each GPU supplies `M/8` rows of `A`; AllGather constructs
  `A[M,K]` before GEMM. Decode `M=8,32,64` and prefill
  `M=2048,4096,8192`. The `M=1` GEMM+AR point is replaced by `M=8`
  because this eight-rank AllGather requires an integral input shard.

The three methods are cuBLAS + NCCL, cuBLAS + UCCL Lite CPU router, and the
resident CUTLASS BF16 GEMM fused with the CPU router. The benchmark checks
every output element on all eight GPUs after every call, outside timing.
Each sample is the slowest-GPU CUDA-event latency. Each shape has two
independent runs with two warmup rounds and five measured rounds of ten calls
per method. Method order rotates each round; shape order reverses in run 2.
The reported value is the median of the two per-run sample medians. Thus
there are 50 samples per method/shape/run, or 7,200 measured calls across the
two operations. The fused method also changes the GEMM kernel.

## Results

| Operation | Decode best path (12 shapes) | Prefill best path (12 shapes) |
| --- | --- | --- |
| GEMM + RS | Fused CPU router at all 12 aggregate medians | Sequential CPU router at all 12 |
| AG + GEMM | Sequential CPU router at all 12 | Fused CPU router at all 12 |

AG + GEMM has the same winner in both runs at every shape. For GEMM + RS,
Qwen3.5-9B `M=64` has an aggregate fused win but NCCL wins its second run;
the other 23 shapes have the same winning method in both runs. At prefill
sizes, fused GEMM + RS is **2.99×–5.03× slower** than sequential CPU router.
All negative results remain in the samples. Prefill panels use independent
linear latency scales so the small bars remain legible; compare their labeled
values across models.

## Data

| Operation | Run 1 | Run 2 | Summary | Decode chart | Prefill chart |
| --- | --- | --- | --- | --- | --- |
| GEMM + RS | [CSV](samples_gemm_rs_run1.csv) | [CSV](samples_gemm_rs_run2.csv) | [CSV](summary_gemm_rs.csv) | [SVG](../../../docs/assets/fusion-8l4-model-gemm-rs-decode-labeled-bars.svg) | [SVG](../../../docs/assets/fusion-8l4-model-gemm-rs-prefill-labeled-bars.svg) |
| AG + GEMM | [CSV](samples_ag_gemm_run1.csv) | [CSV](samples_ag_gemm_run2.csv) | [CSV](summary_ag_gemm.csv) | [SVG](../../../docs/assets/fusion-8l4-model-ag-gemm-decode-labeled-bars.svg) | [SVG](../../../docs/assets/fusion-8l4-model-ag-gemm-prefill-labeled-bars.svg) |

The run logs record each exact command and the benchmark output:
[RS run 1](gemm_rs_run1.log), [RS run 2](gemm_rs_run2.log),
[AG run 1](ag_gemm_run1.log), [AG run 2](ag_gemm_run2.log).

## Reproduce

From `experimental/lite`:

```bash
make -j2 build/uccl_lite_router_fusion_8l4 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
python3 benchmarks/data/fusion_8l4_models_ag_rs_20261111/run.py --op gemm-rs --run 1
python3 benchmarks/data/fusion_8l4_models_ag_rs_20261111/run.py --op gemm-rs --run 2
python3 benchmarks/data/fusion_8l4_models_ag_rs_20261111/run.py --op ag-gemm --run 1
python3 benchmarks/data/fusion_8l4_models_ag_rs_20261111/run.py --op ag-gemm --run 2
python3 scripts/plot_router_fusion_8l4_models_ag_rs.py
```
