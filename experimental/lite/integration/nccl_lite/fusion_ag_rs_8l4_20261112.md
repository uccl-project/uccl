# SGLang EP8 AG+gate GEMM and MoE+RS integration (2026-11-12)

This experiment adds two opt-in layer-boundary operations for
GLM-4.5-Air-FP8 TP8/DP8/EP8 on eight L4s:

- **AG+gate:** defer the sparse layer's one-row-per-rank BF16 AllGather until
  its gate weight is available. The integration API fills `A[8,4096]` and
  computes BF16 gate logits `C[8,128]`. The default SGLang launcher uses
  packet AG followed by cuBLAS in one C++ call, matching the installed
  `F.linear` output in the measured checks. The optional
  `UCCL_LITE_AG_GEMM_BACKEND=kernel` uses a single cooperative AG+GEMM kernel;
  it changes some gate BF16 logits and remains experimental.
- **MoE epilogue+RS:** add routed and shared BF16 outputs during GPU packet
  ReduceScatter upload, retaining the local shard on its GPU. This is the
  [RS fusion path](fusion_rs_8l4_20261112.md).

The hooks live under `sglang_hooks/` and are activated by
`run_sglang_fusion_8l4.sh`. They check the installed SGLang method source
hashes. Unsupported shapes and layouts use the original SGLang methods.
`run_sglang_ag_gate_8l4.sh` and `run_sglang_rs_sum2_8l4.sh` allow independent
comparison. No installed SGLang package or model weight was edited.

## Integration boundary and initial probe

In the recorded SGLang build, `Glm4MoeDecoderLayer.forward` gathers through
`CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual`
and `dp_gather_partial` before MoE/MLP, then scatters through
`CommunicateSummableTensorPairFn._scatter_hidden_states` and
`dp_reduce_scatter_tensor`. Fusion needs the layer's weights and token layout;
the NCCL symbol interposer alone cannot identify the GEMM producer or consumer.
The routed experts use `CompressedTensorsW8A8Fp8MoE` with dynamic per-token FP8
activations. The BF16 standalone GEMM and single-process `PacketPipeline`
cannot directly replace that arithmetic or ownership. Expert quantization and
routing remain in SGLang; the gate weight is BF16 `[128,4096]`.

The initial single-process, synthetic-BF16 AG+gate probe used eight L4s,
CUDA 12.9, NCCL 2.28.9 and `router_fusion_8l4`: one warmup round and two
measured rounds of five calls per method. Slowest-GPU CUDA-event medians from
the [30 samples per shape](fusion_ag_gate_probe_20261112.tsv) were:

| Total gathered rows | cuBLAS + NCCL | cuBLAS + router | fused + router |
| ---: | ---: | ---: | ---: |
| 8 (one row/rank decode) | 0.2519 ms | 0.1904 ms | 0.1331 ms |
| 296 (37 rows/rank prefill) | 0.6252 ms | 0.5371 ms | 0.6221 ms |

These are isolated operation timings, not SGLang results or a per-size selected
curve. The prefill probe did not beat cuBLAS plus router. A cold eight-request
64-input/16-output-token shape trace observed BF16 element counts AG
4096/18944/132608/151552, RS 4096, and AR 4096/28672/32768/262144/1839104.
These were first occurrences, not frequencies; its 4.85 s cold request was
not a paired performance comparison. The implemented decode path above
supersedes the initial candidate inventory.

## Numerical checks

The eight-process `check-ag-gemm` compares every gathered BF16 element and
all 1024 gate logits with NCCL-compatible packet AG plus cuBLAS. With the
cuBLAS backend, all logits matched bitwise. It also queues AG+GEMM, ordinary
AG, and RS_SUM2 on one stream and checks RS placement. `check-sum2` compares
every BF16 RS element against materialized BF16 add plus packet RS for
positive and signed cancellation inputs; both checks passed on eight L4s.

In SGLang warmup, all eight ranks logged `AG_GEMM` and `RS_SUM2`. An optional
shadow run compared the deferred AG buffer, gate logits, and RS output with
their ordinary counterparts for the first 1800 calls per rank; those three
outputs matched bitwise in that diagnostic run. The shadow calls add
communication and synchronizations and were excluded from timed runs.

The cooperative AG+GEMM kernel's first real-weight gate check differed in
109 of 1024 BF16 logits (maximum absolute difference 0.015625), though the
raw top-eight expert sets matched on that input. Its deterministic eight-prompt
output check matched the packet control in five cases. The cuBLAS AG-only
path matched all eight in one check. For the safe combined path, repeated
eight-prompt runs matched 7–8 cases; packet-only repeats matched 6–8 cases.
Thus exact generated text is not stable for this concurrent workload even
without fusion. These observations do not prove model-level equivalence on
other prompts or configurations.

## End-to-end results

The host has 8 NVIDIA L4s, driver 580.178.04. The proxy was built with CUDA
12.9 and forwarded to the same NCCL 2.27.5 binary used by SGLang's PyTorch
2.9.1+cu128. SGLang used eager execution, packet AG/RS/AR, TP8/DP8/EP8,
`mem_fraction_static=0.93`, disabled radix cache, and the same CPU affinity.
The command requested 32 maximum running requests, but startup logs report
an **effective limit of four** for each compared server. The fixed client
uses temperature zero and records whole-batch time including queueing and
HTTP/client overhead. Each value below is a median of three completed runs;
the [135 raw rows](fusion_ag_rs_e2e_20261112.jsonl) retain all repeats and
earlier experimental variants. The current packet, safe combined, and
RS-only comparison used random seed 264301865.

| Requests:input:output | Packet | Safe AG+gate + RS | RS-only |
| --- | ---: | ---: | ---: |
| 8:64:16 | 2.685 s | **2.604 s** (3.0% lower) | Not remeasured after stream recording |
| 32:512:64 | **14.486 s** | 14.577 s (0.6% higher) | Not remeasured after stream recording |
| 64:256:32 | 14.739 s | 14.764 s (tied) | **14.519 s** (1.5% lower) |

These are small, single-host effects. The last packet `64:256:32` repeat was
15.780 s versus 14.739 and 14.496 s for its other two repeats, so the
RS-only 1.5% median gain is especially sensitive to run variation. The
eight-request safe AG+gate plus RS_SUM2 result is an end-to-end gain for the
measured shape; the 32- and 64-request workloads do not show a gain for that
specific path. The earlier
cooperative AG+GEMM variant reached 2.543 s for `8:64:16` but is excluded
from the recommended result because its output differed from the packet
control on some deterministic prompts. The earlier scalar RS upload also
regressed on the short workload; both negative results remain in the raw
rows.
The cooperative AG+GEMM plus RS path was also rerun on `64:256:32` with the
same seed: 14.993, 14.694, and 14.511 s, or a 14.694 s median. That is only
0.3% below the packet median and does not justify its gate-logit changes for
the original workload.

A decode-heavy `8:64:256` check with the same seed used packet, cooperative
AG+GEMM plus RS, safe AG/cuBLAS plus RS, then packet again. Three-run
whole-batch medians were **37.052, 34.381, 35.962, and 37.427 s** in that
order. The safe path was 2.9% and 3.9% below its two packet controls;
its individual runs were 37.053, 35.120, and 35.962 s. The experimental
single-kernel path was 7.2% below the first packet median, with individual
runs of 34.967, 34.381, and 34.331 s. All timings include complete HTTP
request batches. Median batch-tail TPOT was 143.5 ms for the first packet
control, 139.2 ms for the safe path, and 132.7 ms for the single-kernel path.
The single-kernel result must be read with its gate-logit numerical difference
above. The safe path's first run was near the packet median, so independent
repeats are still needed before treating a 2.9%–3.9% gain as general.

A later [row-ready MoE combine plus RS experiment](fusion_ready_rs_8l4_20261112.md)
uses a separate transport stream and is reported on its own; its samples
are also retained in the raw rows above.
An [opt-in FP8 down GEMM readiness experiment](fusion_down_gemm_rs_8l4_20261112.md)
also uses the ready consumer, but showed no reproducible incremental
end-to-end gain over combine+RS.

### Safe AG+gate with row-ready RS

The opt-in `run_sglang_ag_ready_rs_8l4.sh` composes the numerically matching
AG+gate cuBLAS path with ready combine and the separate-stream ready RS.
The hook installation order matters: ready combine wraps the original MoE
forward first, then AG+gate wraps ready combine, so the deferred AG path
still reaches the ready RS producer. On all eight ranks, the first checked
AG buffer, gate logits, combine output, and RS output matched their ordinary
counterparts bitwise. Verification flags remained enabled for the first two
matching calls per rank, so early timed requests may include their shadow
work; the three-run medians are reported without correcting for it.

In a sequential same-seed packet, ready-RS-only, then combined comparison,
the three-run whole-batch medians were as follows. The final combined
server had verification disabled, matching the packet and ready controls;
the earlier combined run retained two-call verification for diagnosis.

| Requests:input:output | Packet | Ready RS only | Combined with verification | Combined, timed without verification |
| --- | ---: | ---: | ---: | ---: |
| 8:64:16 | 2.694 s | 2.710 s | 2.672 s | 2.543 s |
| 8:64:256 | 37.354 s | 36.472 s | 36.373 s | 36.349 s |

The clean combined path was 5.6% and 2.7% lower than packet on the short-
and long-output workloads, respectively. Its 0.3% difference from ready RS
alone on the long-output workload is too small to attribute an incremental
AG gain from this one server start per variant. The short-output gain was
larger, but a first-request cold effect and server-start variation remain.
This is a working combined SGLang
integration with a measured E2E gain for one workload, not evidence that
the experimental single-kernel AG+GEMM or FP8 down GEMM readiness variants
are faster or numerically interchangeable.

The original [README end-to-end result](README.md#best-end-to-end-result)
used a different effective concurrency and earlier source, so its 7.1%
packet-versus-NCCL gain must not be added to these percentages.

## Run

```bash
UCCL_LITE_AG_GEMM_BACKEND=cublas make -C experimental/lite/integration/nccl_lite check-ag-gemm \
  CUDA_HOME=/usr/local/cuda-12.9 \
  NCCL_HOME=$PWD/experimental/lite/build/nccl-2.28.9 \
  REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2
make -C experimental/lite/integration/nccl_lite check-sum2 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  NCCL_HOME=$PWD/experimental/lite/build/nccl-2.28.9 \
  REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2
UCCL_LITE_REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2 \
  ./experimental/lite/integration/nccl_lite/run_sglang_fusion_8l4.sh \
  ./experimental/lite/integration/nccl_lite/launch_glm45_air.sh --mem-fraction-static 0.93
# For the combined AG+gate and row-ready RS path:
UCCL_LITE_REAL_NCCL=/path/to/sglang-pytorch/nvidia/nccl/lib/libnccl.so.2 \
  ./experimental/lite/integration/nccl_lite/run_sglang_ag_ready_rs_8l4.sh \
  ./experimental/lite/integration/nccl_lite/launch_glm45_air.sh --mem-fraction-static 0.93
```

`run_sglang_fusion_8l4.sh` defaults to the cuBLAS AG backend; set
`UCCL_LITE_AG_GEMM_BACKEND=kernel` only to reproduce the experimental
single-kernel path. `UCCL_LITE_REAL_NCCL` must point to the NCCL binary
loaded by SGLang's PyTorch ProcessGroup.
