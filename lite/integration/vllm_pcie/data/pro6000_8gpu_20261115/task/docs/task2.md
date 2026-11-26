# PCIeCCL integration and vLLM serving benchmark

## What is integrated

The local `vllm-0.26.0` checkout adds PCIeCCL ahead of PyNccl in vLLM's TP
all-reduce dispatch chain. It is enabled only with `VLLM_USE_PCIECCL=1` and
falls back to NCCL for unsupported dtype, layout, size, or process-group
configurations.

The extension adapts the existing eight-rank topology schedule to persistent
vLLM workers:

```text
ReduceScatter: xor 1 (PIX) -> xor 2 (NODE) -> xor 4 (SYS)
AllGather:     xor 4 (SYS) -> xor 2 (NODE) -> xor 1 (PIX)
```

It supports BF16 and FP32 and selects the previously tuned transports by
message size: LL through 32 KiB, SM put below 4 MiB, and copy engine from
4 MiB upward in eager execution. CUDA graph capture uses the pure-GPU
`ar8Put` path for messages of 4 MiB and larger because proxy-driven copy-engine
submission cannot safely occur inside a captured graph. The configured 64 MiB
maximum covers MiniMax-M2.7's largest TP hidden-state reduction with 8,192
scheduled tokens. Each rank owns about 448 MiB of registered PCIeCCL staging
memory.

The kernels are asynchronous on vLLM's current CUDA stream. LL sequence flags
advance on the device so captured graphs remain valid across replays. Proxy
threads are pinned to distinct CPU cores on their GPU's NUMA socket without
restricting the vLLM worker's main thread.

## Build and install

The environment comes from the reusable initialization script in `common/`:

```bash
cd /workspace/research_pcie_optim/dev/adhoc-PCIeCCL
../common/scripts/setup-uv-vllm-0.26.0.sh \
  --uv-config-path environments/vllm-pcieccl \
  --uv-env-name vllm-pcieccl
cd /workspace/research_pcie_optim/dev/adhoc-PCIeCCL/vllm-0.26.0
bash csrc/pcieccl/install.sh
```

`install.sh` builds only the small SM120 PCIeCCL extension and installs the
three modified Python modules into the isolated vLLM 0.26.0 environment. It
does not rebuild vLLM's upstream CUDA extensions.

## Correctness gate

```bash
cd /workspace/research_pcie_optim/dev/adhoc-PCIeCCL
bash tests/run_pcieccl_correctness.sh
```

This compares PCIeCCL with NCCL on all eight GPUs for BF16 and FP32 at 128 B,
64 KiB, and 4 MiB, covering LL, SM-put, and eager CE execution. It also
captures and replays the corresponding graph-safe paths three times. The
current gate passes with BF16 maximum absolute differences of 0.0625, 0.09375,
and 0.125 and FP32 maximum differences no larger than 1.91e-6. These
differences are the expected result of a different reduction tree.

The serving runner adds a greedy token diagnostic before each measurement
matrix. It records 64 output token IDs from an 8,192-token request and compares
NCCL with PCIeCCL. A mismatch is preserved in `parity_comparison.json` with
`status: warning`, but never blocks benchmark or report generation. Request,
response-shape, tensor correctness, and collective failures remain hard errors.

Exact token identity is not used as a quality gate. PCIeCCL's different
reduction tree changes BF16 rounding order; with FP8 model weights and
near-tied logits, that can change a greedy token and cause later autoregressive
tokens to diverge. Requiring bitwise-equivalent reductions would constrain the
optimized implementation and risk giving back substantial performance. The
relevant serving-quality gate is a task-level evaluation.

## Inference quality

The committed quality evaluation uses the full official GSM8K test split with
vLLM 0.26.0's evaluator: 1,319 questions, 5-shot prompts, temperature 0, seed
42, 256 maximum output tokens, and concurrency 64. It uses the same
MiniMax-M2.7 TP=8/EP configuration for both backends.

```bash
# Start the NCCL or PCIeCCL server described below, then run:
bash benchmarks/run_gsm8k_eval.sh nccl 31000
bash benchmarks/run_gsm8k_eval.sh pcieccl 31001
```

NCCL answered 1,215/1,319 correctly (92.115%); PCIeCCL answered 1,218/1,319
correctly (92.343%). Both produced zero invalid responses. The +0.227
percentage-point difference (+3/1,319 questions) is not evidence of a PCIeCCL
quality loss. Their total generations differ by 425 tokens because a changed
greedy choice changes the later autoregressive trajectory; this is not forced
to match. Raw evaluator summaries are committed under `results/quality/`.

## Serving benchmark

The two TOML files differ only in port, result directory, and the PCIeCCL
enable variable. Both use vLLM 0.26.0, TP=8, expert parallel across those eight
ranks, NCCL `naive` all-to-all, prefix caching, ISL=8,192, OSL=1,024, and a 70%
constructed shared prefix. Built-in vLLM custom and symmetric-memory
all-reduces are disabled in both, making the baseline standard PyNccl.

Run the complete fine benchmark:

```bash
cd /workspace/research_pcie_optim/dev/adhoc-PCIeCCL
bash benchmarks/run_task2.sh
```

Each of the fourteen cases has a 60-second warm-up and 180-second measurement
window. Including two MiniMax-M2.7 model loads, allow roughly 60-90 minutes.
The runner is resumable: completed cases with the same configuration signature
are skipped.

Raw manifests, server logs, per-case JSON, and summaries are written below
`results/task2/{nccl,pcieccl}`. After both matrices pass, the analysis script
generates `results/task2/report.md` with TTFT, TPOT, output throughput, speedups,
and bottleneck analysis for every batch size.

The NCCL environment normalization is explicit and identical in both configs:
`NCCL_IB_DISABLE=1` and `NCCL_NET=Socket`. Without it, this host attempts to
initialize an unsupported bonded mlx5 queue pair and NCCL exits before the
comparison starts.
