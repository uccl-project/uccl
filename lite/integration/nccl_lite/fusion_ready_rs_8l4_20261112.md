# SGLang EP8 row-ready combine plus ReduceScatter (2026-11-12)

This opt-in path connects the GLM-4.5-Air-FP8 Triton MoE combine stage to a
dedicated UCCL Lite packet RS stream. It targets the BF16 decode shape
`[8 tokens, 8 routed experts, 4096 hidden]` on eight L4s. The ordinary
SGLang/packet paths handle other shapes, capture modes, and layouts.

The producer Triton kernel sums eight expert outputs into each token row,
then publishes a system-scope release flag for that row. A separate
non-cooperative eight-CTA CUDA kernel waits on the flags with system-scope
acquire. Each CTA uploads one destination's routed-plus-shared BF16 row; the
owner CTA can reduce its row as soon as all eight ranks publish that
destination. SGLang's current stream waits on a CUDA event before consuming
the RS output. A second packet Handle keeps this transport stream separate
from SGLang's ordinary NCCL communicator. The producer and consumer modules
and the Handle are prepared before the first resident wait. The original
SGLang MoE forward runs unchanged outside these exact-shape hooks.

This schedule permits **combine+RS overlap**: the ready consumer is enqueued
before combine on a separate stream and may upload earlier rows while later
rows are still being combined. A direct GPU kernel timeline was not captured;
the available Nsight Systems attempt collected CUDA API events but no GPU
kernel intervals. The FP8 down GEMM itself completes before combine starts
on the same stream, so these measurements do not prove down GEMM+RS overlap.

## Numerical and ordering checks

- On one L4, the readiness combine output matched SGLang's
  `moe_sum_reduce_torch_compile` bitwise for random signed BF16 input at
  `M=8` and `M=296`; all row flags became ready. A separate-stream consumer
  waited on those flags and copied the eight output rows exactly.
- The eight-process `check-ready-rs` launches the transport consumer before
  a separate producer stream writes deliberately staggered BF16 rows and
  release flags. It ran three consecutive calls and compared every RS element
  bitwise with materialized BF16 add plus ordinary packet RS. It passed.
- In the actual SGLang EP8 warmup, all eight ranks logged `RS_READY_SUM2` and
  used eight row flags. An optional shadow run compared the first 200 combine
  outputs and 4096-element ready RS results per rank with their ordinary
  counterparts; both were bitwise equal. Shadow operations and their
  synchronizations were excluded from latency measurements.
- Eight deterministic concurrent prompts matched a prior packet output set
  in seven cases in one diagnostic run. Repeated packet-only prompt sets
  themselves matched six to eight cases, so generated text equality is not a
  stable model-level correctness test for this setup.

The first standalone attempt hung because a waiting transport kernel was
launched before the producer CUDA module was loaded. The test now preloads
the producer, and the SGLang hook prewarms both modules and its dedicated
Handle before its first real call. The resident consumer uses a normal CUDA
launch; it has no grid-wide barrier.

## End-to-end measurements

Hardware and server flags match the
[AG/RS integration report](fusion_ag_rs_8l4_20261112.md): 8 NVIDIA L4s,
driver 580.178.04, proxy built with CUDA 12.9, PyTorch 2.9.1+cu128/NCCL
2.27.5, GLM-4.5-Air-FP8 TP8/DP8/EP8, eager execution, packet AG/RS/AR,
`mem_fraction_static=0.93`, no radix cache, and the same CPU affinity. The
effective running-request limit was four. All compared servers used seed
264301865. The deterministic client submits each whole batch to `/generate`
and includes queueing and HTTP/client overhead. Values are three-repeat
whole-batch medians in seconds; no variant was selected per individual size.
The [135 raw rows](fusion_ag_rs_e2e_20261112.jsonl) include these phases and
the earlier variants.

### Initial source revision

| Requests:input:output | Packet | Ready combine + ordinary RS | Ready combine + ready RS |
| --- | ---: | ---: | ---: |
| 8:64:16 | 2.685 | 2.643 | **2.557** |
| 8:64:256 | 37.052 before / 37.427 after | 36.942 | **35.314** |
| 64:256:32 | 14.739 | Not measured | **14.453** |

For `8:64:256`, ready RS reduced whole-batch time by 4.7% and 5.6% versus
the two packet controls. Individual ready RS runs were 35.338, 35.290, and
35.314 s; the three packet runs before were 36.876, 37.077, and 37.052 s,
and the three after were 37.427, 37.475, and 37.353 s. The producer-only
path at 36.942 s was effectively tied with packet, consistent with the
separate ready consumer contributing to the measured gain. For the original
`64:256:32` workload, ready RS was 1.9% below the packet median, but one
packet run was a slow 15.780 s outlier; treat this as tentative. These are
single-host measurements, not a significance test or a model-quality eval.

### GPU-only Handle screening

A later optional `UCCL_LITE_READY_RS_GPU_ONLY=1` mode reduced the dedicated
Handle's per-rank input capacity from 32 MiB to 64 KiB and omitted its CPU
packet worker. It passed the eight-process readiness check and the first
200 real SGLang combine/RS shadow comparisons per rank. On one server it
took 2.679 s for `8:64:16` and 35.969 s for `8:64:256` (three-run medians),
versus 2.557 and 35.314 s for the earlier full-Handle ready path. The
resource-saving mode was slower in those separate server runs. The default
keeps the full Handle; the comparison does not establish a causal worker or
memory effect.

### Current source revision

The optional-mode change initially exposed an incorrect guard that rejected
the default full Handle before enqueue. The guard was fixed, and both
full-Handle and GPU-only eight-process readiness checks passed. The restored
default also completed a SGLang smoke request. A later same-seed three-run
`8:64:256` comparison gave **38.094 s** for full-Handle ready RS (37.342,
38.094, 38.579 s) versus **38.491 s** for packet (39.536, 38.491, 38.364
s), only 1.0% lower. Both paths were slower than the initial measurements;
GPU clocks and system load were not recorded well enough to attribute the
change. The current-source result does not reproduce the earlier 4.7%–5.6%
gain, so that earlier figure must remain a historical measurement rather
than a validated current performance claim.

A further sequential packet-then-ready retest with the same source, model,
seed, and client measured the following three-run medians:

| Requests:input:output | Packet | Ready combine + ready RS | Difference |
| --- | ---: | ---: | ---: |
| 8:64:16 | 2.694 s | 2.710 s | 0.6% higher |
| 8:64:256 | 37.354 s | 36.472 s | 2.4% lower |

The long-output ready runs were 36.472, 36.141, and 36.884 s, below each of
the three packet runs (37.290, 37.354, and 37.467 s). This retest supports
a workload-specific E2E gain, while the short-output result does not. It is
one host and one server start per variant, so the size of the gain remains
uncertain.

## Reproduce

```bash
make -C lite/integration/nccl_lite check-ready-rs \
  CUDA_HOME=/usr/local/cuda-12.9 \
  NCCL_HOME=$PWD/lite/build/nccl-2.28.9 \
  REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2
UCCL_LITE_REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2 \
  ./lite/integration/nccl_lite/run_sglang_ready_rs_8l4.sh \
  ./lite/integration/nccl_lite/launch_glm45_air.sh --mem-fraction-static 0.93
```

`run_sglang_ready_combine_8l4.sh` enables the producer without the separate
RS consumer for the control above. The ready RS path is opt-in and does not
change the default NCCL compatibility layer. The GPU-only Handle is a
separate experiment enabled by `UCCL_LITE_READY_RS_GPU_ONLY=1`.

## Remaining down GEMM overlap

The installed SGLang Triton runner writes the routed expert outputs to
`intermediate_cache3` with `fused_moe_kernel`. Its programs are organized by
expert-sorted token blocks and output-channel tiles. The separate combine
kernel cannot begin any row until that down GEMM launch completes on its
stream. To overlap the actual FP8 down GEMM with RS, a per-token completion
counter must account for every selected expert and output-channel tile,
including the nonlocal-expert zero-fill branch. Only the last completion for
a token may publish system-scope row readiness to a resident combine
consumer; the ready RS consumer can then consume that row. The installed
`down_gemm_overlap_args` machinery is used by other MoE runners, not by this
Triton FP8 path. This ready-combine implementation remains combine+RS
scheduling; a down GEMM producer must be evaluated separately.

A subsequent [version-pinned down GEMM producer prototype](fusion_down_gemm_rs_8l4_20261112.md)
added those completion counters and passed the measured decode-shape
numerical checks, but did not show an incremental end-to-end gain. It
remains opt-in while destination-oriented scheduling and counter cost are
investigated.

For this installed L4 build, the fallback Triton FP8 MoE config reports
`BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, and `BLOCK_SIZE_K=128`. With 4096
output channels and top-8 routing, a token needs completion of up to
`8 × ceil(4096/64) = 512` expert/channel tile writes, including zero-filled
nonlocal experts. A GPU-scope completion counter with a final system-scope
release would be required; its atomic overhead has not been measured. The
Triton down GEMM source and launch API need a guarded change before that
counter can be published safely.
