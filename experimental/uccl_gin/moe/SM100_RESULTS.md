# SM100-family results

Measured target: **B300 / SM103**; CUDA 13.2, NCCL 2.30.4, CUTLASS 4.3.4,
PyTorch 2.10.0+cu130. Speeds are medians of paired ratios; greater than 1 favors
the candidate. CUDA spans include launch gaps and synchronization; host timings
include full output validation.

| Comparison | CUDA speed | Host speed | Status |
|---|---:|---:|---|
| Complete Qwen: native PyTorch → CUTLASS+HT+sorted routes | **1.453x** | **1.195x** | Correctness PASS; faster in 4/4 processes |
| Same model: PyTorch+HT → CUTLASS+HT+sorted routes | 1.832x | 1.462x | Correctness PASS; arithmetic and routing combined |
| Eight-card staged-copy: active 8/18 → 4/37, resident pool 8 | 1.065x | 1.027x | Correctness PASS; faster in 5/5 holdout pairs |
| Finer quotas vs retained 4/37 | No gain | No gain | 18 jobs PASS; conditional holdout NOT_RUN |
| SM90 new four-method model comparison | — | — | 7 qualification waves PASS; performance NOT_RUN |

Complete-model scope: Qwen3-30B-A3B at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, all 48 layers, BF16,
one GPU and one HT group. B64/C128, two serial macro batches, prompt 128,
four greedy tokens, fixed math attention. Four balanced fresh processes contain
48 measured waves; with qualification, all **83 eligible waves** match native
tokens and complete BF16 vocabulary-logit hashes bitwise. An externally
contended original session is excluded and replaced with a fresh identical-order
process.

The separate eight-card fixture passes 46 jobs / 688 rank reports, including the
five-pair holdout. Finer quotas pass another 18 jobs / 144 rank reports but do not
meet the frozen improvement gate; retain **4 active groups / 37 CTAs / resident 8**.
These staged-copy results do not measure expert GEMM or full-model inference.

Earlier unsupported-LSA and sanitizer cache-teardown failures were corrected.
Additional PTXAS TMEM guardrail instrumentation is unavailable on the tested
SM103 target. Multiple active model groups, cross-GPU/RDMA model execution and
production serving performance remain unqualified. The PR contains no run logs
or result archives; [reproduction commands](README.md) generate fresh results.
