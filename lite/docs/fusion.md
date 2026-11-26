# Communication–compute fusion

Fusion is useful when the first completed piece can be consumed before the
entire operation finishes. UCCL Lite exposes readiness at the granularity that
preserves GEMM efficiency: input groups, output shards, or channel stripes.

The [eight-GPU CPU-router experiments](../README.md#fusion-on-single-8xl4-gpu)
measure GEMM + AllReduce, GEMM + ReduceScatter, and AllGather + GEMM
on the two-NUMA host. The refreshed four-GPU direct-peer results
below use a different testbed and timing scope.

The [November 15 eight-L4 review](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/fusion_review_8l4_20261115/README.md)
adds fresh sequential controls using the optimized standalone collective
schedule. Historical fusion figures do not validate those later changes or
establish a gain over the new sequential control. GPU progress and host-ring RS
cannot simply be enabled on the resident GEMM path; the review records the
API restrictions, checked measurements, and remaining regressions.

## November 10 four-L4 refresh

The new FP16 dense comparison uses l40 (`mibura-sky-test-01`), four L4s in a
2+2 NUMA layout, driver 580.178.04, CUDA 12.9, NCCL 2.28.7, and the same
MSCCL++ binary for each UCCL path. Each shape ran three independent times,
with the shape order reversed in the second run. Each executable invocation
uses five warmups, twenty timed iterations, and correctness checking. The
plotted values are medians of run-level complete-operation latency. AG uses
the harness's recorded UCCL Lite dispatch path; RS and AR use the ready-ring
fusion path. The baselines are respectively NCCL AllGather + cuBLAS,
cuBLAS + NCCL ReduceScatter, and cuBLAS + NCCL AllReduce.

![Fresh complete-operation latency bars for every dense shape](assets/dense-fusion-4l4-refresh.svg)

At the large reference shape, AG+GEMM is 15.631 versus 22.850 ms, GEMM+RS
18.423 versus 22.746 ms, and GEMM+AR 19.964 versus 24.515 ms (UCCL Lite
versus baseline). Small shapes are retained: at Llama 8B M=128, GEMM+RS is
0.166 versus 0.145 ms and GEMM+AR is 0.302 versus 0.201 ms. The
selected path is faster at 19 of 24 measured dense operation/shape pairs;
the five losses remain in the full sweep. These counts are descriptive, not
significance tests. The
[per-shape summary](../benchmarks/data/refresh_20261110/fusion_dense_summary.csv),
[all variant samples](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/refresh_20261110/fusion_dense_samples.csv),
and [24 complete run logs](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/refresh_20261110/fusion_dense_l40)
record the other paths and negative results.

The BF16 full-layer MoE refresh uses the same host, CUDA and NCCL versions.
It measures NCCL + grouped cuBLAS, UCCL Lite + cuBLAS, and the fused path in
one harness. For each token count, a 2.5-second settling period precedes
twelve interleaved rounds of four iterations per variant; each round records
the slowest rank. The minimum of those round times per variant is reported,
matching this harness's power-stability method. Routing-plan construction
and framework overhead remain outside the timed layer. At eight tokens per
rank, fusion takes 2.459 versus 1.370 ms; at 32,768 it takes 84.861 versus
129.830 ms.
Fusion wins at five of thirteen token counts, beginning at 2,048 tokens per
rank; all smaller measured batches lose to the complete-layer baseline.

![Fresh complete-layer latency bars for every measured MoE token count](assets/moe-fusion-4l4-refresh.svg)

The [round samples](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/refresh_20261110/fusion_moe_samples.csv),
[full summary](../benchmarks/data/refresh_20261110/fusion_moe_summary.csv), and
[complete log](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/refresh_20261110/fusion_moe_l40.log) retain
all small-batch regressions and the unfused UCCL Lite control. The older
tables below remain historical measurements and are not substituted for this
refresh.

## Dense tensor parallelism

[dense_fusion.cu](../benchmarks/fusion/dense_fusion.cu) contains FP16 benchmarks
and their cuBLAS+NCCL controls in one executable.

- **AllGather + GEMM:** compute local rows while gathering the intra-socket
  peer, then compute those rows while gathering the other socket. Small shapes
  use gated CUTLASS kernels; the M=512 policy uses a non-overlapped hierarchy.
- **GEMM + ReduceScatter:** produce destination shards in the order consumed
  by the ring. A low-occupancy communication grid consumes each ready shard
  while cuBLAS computes later shards.
- **GEMM + AllReduce:** follow each stripe's ReduceScatter with AllGather,
  allowing earlier AllGather traffic to overlap later GEMM work. The large
  reference shape uses two stripes; more stripes lose GEMM efficiency.

| Large reference operation | cuBLAS + NCCL | UCCL Lite fused | Time saved |
| --- | ---: | ---: | ---: |
| AG + GEMM, `(4096,49152,12288)` | 22.871 ms | 15.578 ms | 7.293 ms |
| GEMM + RS, `(4096,12288,49152)` | 22.954 ms | 18.461 ms | 4.493 ms |
| GEMM + AR, `(4096,12288,49152)` | 25.658 ms | 19.774 ms | 5.884 ms |

[Archived figure: Paired latency bars for dense fusion and cuBLAS plus NCCL across eight model shapes per operation](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/assets/dense-fusion-shapes.svg)

The paired bars show absolute latency for every shape. The two columns use
different labeled linear ranges so the sub-millisecond cases remain legible;
longer bars mean slower operations. The large reference shapes are favorable,
not averages. The retained
[shape sweep](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/dense_fusion.txt) includes regressions, including
small GEMM+AllReduce cases. Flux uses a separate harness with different timing
aggregation; approximately 1–2% differences should be treated as ties.

The executable also retains a persistent CUTLASS epilogue/readiness experiment.
Do not attribute its timings to the cuBLAS ready-ring results above.

## Expert-parallel MoE

[fused_moe.cu](../benchmarks/fusion/fused_moe.cu) measures dispatch + gate/up GEMM.
[fused_moe_layer.cu](../benchmarks/fusion/fused_moe_layer.cu) adds SwiGLU, down
projection, weighted reduction, and combine. The reusable implementation lives
in [megamoe_core.cuh](../include/uccl_lite/megamoe_core.cuh).

```mermaid
flowchart LR
    D[Ragged dispatch] -->|arrival flags| U[Grouped gate/up GEMM]
    U --> A[SwiGLU]
    A --> G[Down projection]
    G --> R[Weighted reduction]
    R --> C[Combine to token owners]
```

Dispatch de-duplicates tokens per destination. Gather-A GEMM reads received
rows through an index list; a problem's arrival group determines when its tiles
may execute. The dispatcher occupies a block of the GEMM grid to avoid a
separate dispatcher being scheduled behind workers waiting for its data.

The default shape is DeepSeek-V2-Lite: H=2048, I=1408, 64 experts, top-6, EP=4,
BF16 inputs/outputs with FP32 GEMM accumulation. The Qwen build uses I=512,
256 experts, top-8. At 2,048 tokens/rank, the full layer takes **5.893 ms**
versus **8.241 ms** for NCCL + grouped cuBLAS; at 32,768 tokens/rank it takes
**85.283 versus 129.770 ms**, or **384k versus 253k tokens/s/rank**. At small
token counts, grouped scheduling overhead exceeds
the communication that can be hidden and fusion loses.

[Archived figure: Paired full-layer latency bars for expert-parallel MoE at each recorded token count](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/assets/moe-fusion.svg)

The standalone harness constructs routing metadata outside primary timing.
It is a kernel/layer benchmark, not complete model execution. The separate
[moe_plan.cuh](../include/uccl_lite/moe_plan.cuh) provides GPU histogram, scan,
scatter, and inverse-routing kernels. Callers must supply the metadata exchanges
and persistent workspace; importing the header does not make the standalone
benchmark charge routing-plan construction.

## Tensor-parallel MoE device kernels

[tp_moe.cuh](../include/uccl_lite/tp_moe.cuh) targets TP=4, EP=1: all ranks own
intermediate-channel shards of every expert, with replicated tokens and routes.
The cooperative down-projection grid also performs ordered weighted combination,
one complete AllReduce, and output restoration. Communication can start on a
ready channel stripe while later down-GEMM tiles continue.

[tp_moe_decode.cuh](../include/uccl_lite/tp_moe_decode.cuh) adds a direct sparse
decode path that avoids grouped routing-plan overhead. It is intended for very
small token counts; increasing expert reuse makes grouped computation preferable.

These headers are retained as reusable kernel building blocks. The host
extension, framework baseline tuner, and serving integration are deliberately
not included. `make headers` checks compilation, not a distributed numerical
run. A caller must implement bootstrap, workspace sizing, cooperative occupancy
limits, buffer/flag initialization, and rank-consistent launch ordering before
using them. Do not present header compilation as performance validation.

## Synchronization contract

Writers must publish data before readiness flags; consumers must acquire those
flags before reading or issuing DMA. In-flight source buffers must remain
immutable until the corresponding flush completes. Cooperative launches must
fit the resident grid capacity.

Host-proxy progress also depends on the CUDA submission path. Lazy module loads
or blocking allocations while a resident kernel waits for proxy-delivered data
can deadlock. Preload kernels and preallocate storage before launching such
work. Arbitrary application allocators and concurrent streams are outside this
standalone project's validated scope.
