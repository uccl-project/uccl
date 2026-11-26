# Experimental FP8 down GEMM to ready RS pipeline (2026-11-12)

This opt-in experiment extends the
[row-ready combine+RS path](fusion_ready_rs_8l4_20261112.md) with completion
signals from SGLang's Triton FP8 MoE down GEMM. Its kernel variant is a
version-pinned copy of the installed `fused_moe_kernel` function, guarded by
the original source SHA-256. Other MoE kernel calls continue using the
installed function. The experiment is enabled only by
`run_sglang_down_rs_8l4.sh`; it does not change the default serving path.

The down GEMM writes `intermediate_cache3[8,8,4096]`. For the installed L4
config, output channels use 64-wide tiles. A tile CTA increments an
expert-block counter after its BF16 output writes. The last of 64 channel
tiles in that expert block increments one counter for each valid flattened
token/expert row it wrote, including the nonlocal-expert zero-fill branch.
A separate resident combine kernel waits for eight expert-row completions
per token, reduces them, and publishes a system-scope row flag. The ready RS
consumer on a third CUDA stream waits for that flag and transfers the
routed-plus-shared BF16 row. Stream events protect flag initialization and
the downstream result.

The first diagnostic attempt used the down kernel's `top_k=1` parameter to
group sorted IDs. That parameter indexes flattened expert rows, while the
model's original top-k is eight. It produced 64 completions in each of the
first eight counters and could write beyond the counter buffer. The run
failed at the diagnostic assertion and its server was stopped; it is not a
correctness or performance result. The guarded variant now divides sorted
IDs by the explicit model top-k of eight and masks counter indices to 0–7.

## Checks

On a real GLM-4.5-Air-FP8 TP8/DP8/EP8 SGLang warmup, every rank's first
eligible call reported eight completions for each of the eight tokens.
Active expert-block counts differed by rank (one to four), as expected for
expert parallelism. The first combine and RS shadow outputs per rank matched
their ordinary counterparts bitwise for both the direct-per-tile counter
and the two-level counter variants. The two-level server completed its warmup
and eight concurrent deterministic prompts; that prompt set matched the
packet control in all eight cases in this single run. These checks cover the
observed decode shape, not other token counts or a model-quality evaluation.

## End-to-end result

The test host, software, BF16 communication, GLM-4.5-Air-FP8 model, packet
backend, eager flags, CPU affinity, seed 264301865, and effective four
running-request limit match the
[AG/RS integration report](fusion_ag_rs_8l4_20261112.md). The client times
complete `/generate` request batches including queueing and HTTP overhead.
Each value is the median of three completed runs. The
[135 raw rows](fusion_ag_rs_e2e_20261112.jsonl) include these variants and
all earlier packet/fusion controls; no fastest variant was selected per
individual size.

| Requests:input:output | Per-token tile counter | Two-level expert counter, group 8 | Two-level expert counter, group 1 | Earlier ready combine+RS |
| --- | ---: | ---: | ---: | ---: |
| 8:64:16 | 2.741 s | 2.801 s | 2.649 s | 2.557 s |
| 8:64:256 | 37.538 s | 38.090 s | 38.335 s | 35.314 s |

The two-level counter reduced token-level atomics from 512 to eight per
token for this shape, but it did not improve the measured end-to-end time.
Setting `UCCL_LITE_SGLANG_DOWN_GROUP_M=1` changes only the down GEMM CTA
swizzle. On one further server start, all eight ranks reached the expected
eight completions per token, and the checked combine and RS outputs were
bitwise equal to the ordinary path. Its shorter-request median improved
relative to group 8, while the longer decode-heavy request was slower.
There is no consistent E2E gain from this swizzle.
The runs were on different server starts, and a later same-source packet
control had itself slowed from 37.052 to 38.491 s, so the table is not a
causal estimate of counter overhead. It establishes no reproducible
incremental gain from down GEMM overlap yet. These are isolated single-host
measurements, not a significance test.

One likely limit, inferred from the installed kernel layout, is that its
`BLOCK_SIZE_M=64` groups most or all of the 64 flattened expert rows in a
block. A token's last required output tile may then become ready only near
the end of the down GEMM. A destination-oriented tile order or smaller
expert-row blocks would need a matching sorted-token alignment and a
measured GEMM-efficiency tradeoff; simply changing `BLOCK_SIZE_M` after
alignment would index the wrong expert metadata.

## Reproduce the opt-in diagnostic

```bash
UCCL_LITE_REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2 \
  UCCL_LITE_SGLANG_DOWN_COUNTER_VERIFY=1 \
  UCCL_LITE_SGLANG_READY_COMBINE_VERIFY=1 \
  UCCL_LITE_SGLANG_RS_SUM2_VERIFY=1 \
  UCCL_LITE_SGLANG_VERIFY_CALLS=2 \
  ./lite/integration/nccl_lite/run_sglang_down_rs_8l4.sh \
  ./lite/integration/nccl_lite/launch_glm45_air.sh --mem-fraction-static 0.93
```

The three verification flags add GPU work and synchronization. Leave them
unset for timed runs. The down GEMM kernel variant is not suitable for
other SGLang versions until its hash, shapes, indexing, and numerical checks
have been reviewed again.
