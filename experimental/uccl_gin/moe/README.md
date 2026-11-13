# SM100 expert MMA and Qwen3 forward HT

This experimental path runs the trained **Qwen3-30B-A3B** model through forward HT and real BF16 expert GEMMs. The SM100-family measurements use one physical B300 (SM103), batch 64, 128 queued clients, 128 prompt tokens and four greedy output tokens. One macro batch is active at a time. Separate eight-card experiments qualify staged-copy traffic scheduling with a fixed resident pool; their timings are reported separately from the model.

## Shared HT path, two expert implementations

```text
                 One MoE layer: token hidden states
                                  |
                   same router + top-k8 weights
                                  |
                          same HT dispatch
                                  |
                       same local expert rows
                    +-------------+-------------+
                    |                           |
                    v                           v
           PyTorch experts               CUTLASS experts
              (torch-ht)                   (cutlass-ht)
           gate / up linear              gate / up BF16 GEMM
                SwiGLU                     BF16 SwiGLU
             down linear                  down BF16 GEMM
                    |                           |
                    +-------------+-------------+
                                  |
                  same BF16 weighting + accumulation
                                  |
                          same HT combine
                                  |
                 residual / next layer -> norm / head
```

This comparison shares routing, transport and accumulation. Native PyTorch is
an independent complete-model reference and bypasses HT. The sorted-route
variant changes the grouping step separately, as shown below.

`qwen_moe.py` retains the actual128 expert weights, H2048 and intermediate768. Router probabilities have the same BF16 rounding as the native model before being carried as FP32 through HT. Expert gate/up outputs, activation and down output remain BF16. A stream owns one CUTLASS workspace and one HT group with at most one live routing handle. Combine receives already weighted values. ABI2 exposes callback allocation counts; completed windows retain the initialized group's allocations, and group teardown returns this count to zero.

The methods share the model, CPU mask, one CUDA stream, one resident group and attention settings. The B300 cohort uses CPU8–15. Native HF bypasses HT; `torch-ht` uses native expert arithmetic inside the HT wrapper; `cutlass-ht` changes the expert GEMMs and fuses BF16 SwiGLU. That comparison isolates the arithmetic implementation within the shared wrapper. `cutlass-ht-sorted` additionally changes route grouping. Native-to-HT also changes packing and aggregation work, so its ratio is not an isolated communication measurement.

## Stable expert segments

```text
                 Received expert IDs and probabilities
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
                where control               sorted candidate
            CUDA where per expert       stable argsort + counts
            up to 128 host syncs        one 129-int64 host copy
                    |                           |
                    +-------------+-------------+
                                  |
                  same expert / row / slot order
                                  |
                     same CUTLASS expert math
```


`expert_routes.py` adds an opt-in `sorted` strategy. The control calls CUDA `where` for each local expert; this form is equivalent to `nonzero`, which synchronizes the host with the device. The candidate stably sorts flattened top-k IDs, counts into a fixed-size integer array and copies that array once to the host. For128 local experts the count transfer is1,032 bytes. The zero bucket holds unused `-1` slots. [PyTorch where](https://docs.pytorch.org/docs/2.10/generated/torch.where.html), [CUDA nonzero](https://docs.pytorch.org/docs/2.10/generated/torch.nonzero.html)

Stable sorting preserves the original row/slot order within each expert, and experts still run in ascending order. BF16 probabilities and weighted accumulation therefore keep their rounding order. Gathered hidden states remain one expert at a time; the candidate materializes index/weight metadata rather than expanding all top-k hidden states. [Stable argsort](https://docs.pytorch.org/docs/2.10/generated/torch.argsort.html)

The routing ablation uses four fresh processes with balanced method positions and adjacent method pairs within each round: native, PyTorch+HT, CUTLASS+HT with `where`, and CUTLASS+HT with `sorted`. Each process first records an untimed CUTLASS/where reference. Every subsequent CUTLASS wave must match its greedy tokens and SHA256 of the complete BF16 vocabulary logits, in addition to the native-model oracle. This comparison holds expert kernels, CPU mask, group count, stage policy and CTA budget fixed. The existing HT prepare/drain synchronization protects buffer lifetimes; the routing change preserves that ownership contract.

## Resource contract

CUTLASS4.3.4 builds tile128×128×64/cluster1 on both architectures: WGMMA on SM90 and TMEM MMA on the SM100 family. The initial Blackwell integration treated `hw_info.sm_count` as a CTA cap. Its default CLC scheduler instead launched1024 logical blocks for M8192/N2048/K768 despite a requested18. Explicit static persistence fixes this integration contract. Every arithmetic qualification now checks the queried grid against the requested cap. [CUTLASS scheduler source](https://github.com/NVIDIA/cutlass/blob/v4.3.4/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp)

The same `KernelResources` and `group_capacity` in `thirdparty/nccl-ep/device/ht_scheduler.hpp` apply to communication and MMA:

`capacity = SMs × min(CUDA API residency, floor(SMEM per SM / actual CTA SMEM)) / actual launch blocks`.

The shared scheduler takes the minimum across stages, requested/resident groups, queue and measured complete-group memory. Model runs currently hold one group. Their maximum-row resource probes and initialized-context memory delta are recorded; this delta includes query-cache allocation and is not a peak-group calibration. API residency and this feasibility rule do not guarantee mixed-kernel co-residency.

GH200 retains automatic staging. The tested B300 static/four-stage build uses164,864 bytes of SMEM per CTA, compared with230,400 in the initial CLC/automatic-stage build; both report one resident CTA/SM. This is a measured build comparison, not an isolated stage-count performance ablation. All18/37/74/148 limits pass on B300. The drained traffic-profile and refinement results are in [SM100_RESULTS.md](SM100_RESULTS.md).

CUTLASS uses its standard epilogue for MMA output. The register-to-TMEM reverse-staging prototype has separate evidence and is not fused with the model's expert epilogue. The model cohort measures single-GPU loopback HT; the eight-card traffic cohort measures staged copies. Distributed model RDMA, simultaneous active model groups, fused reverse-TMEM expert epilogues and production serving remain unqualified.

## Reproduction

Use CUDA13.2, CUTLASS4.3.4 headers, PyTorch2.10.0+cu130, Transformers4.57.1, safetensors0.6.2 and the qualified tensor-array NCCL-EP runtime with NCCL2.30.4. The checkpoint revision is `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; all16 original shards have verified SHA256 receipts. Build dependencies stay outside this source tree.

Prepare an existing checkpoint at that revision with the standard Hugging Face downloader, then verify all 16 official LFS shard checksums:

```sh
python experimental/uccl_gin/moe/verify_checkpoint.py --weights "$WEIGHTS"
# For the one-card model build, instantiate a separate LSA1 runtime.
cmake -S thirdparty/nccl-ep -B "$EP_ONE_BUILD" \
  -DCMAKE_CUDA_ARCHITECTURES=103a -DNCCL_DIR="$NCCL_DIR" \
  -DNCCL_EP_USE_UCCL_GIN=ON -DNCCL_EP_UCCL_EFA=OFF \
  -D_NCCL_EP_LSA_TEAM_SIZE_MIN=1 -D_NCCL_EP_LSA_TEAM_SIZE_MAX=1 \
  -D_NCCL_EP_NUM_LSA_TEAMS_LIST=1
cmake --build "$EP_ONE_BUILD" --target nccl_ep_shared -j "$BUILD_JOBS"
make -C experimental/uccl_gin/moe CUDA_HOME="$CUDA_DIR" CUTLASS_DIR="$CUTLASS_DIR" \
  NCCL_DIR="$NCCL_DIR" UCCL_EP_DIR="$EP_ONE_BUILD" \
  ARCH=sm_103a STAGES=4 OUT="$MODEL_BUILD"
export LD_PRELOAD="$CUDA_DIR/lib64/libcudart.so.13:$NCCL_DIR/lib/libnccl.so.2"
export NCCL_IB_DISABLE=1 UCCL_PROXY_USE_TASKSET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 taskset -c 8-15 \
  python experimental/uccl_gin/moe/model_suite.py --weights "$WEIGHTS" \
  --mma-library "$MODEL_BUILD/libexpert_mma.so" \
  --ht-library "$MODEL_BUILD/libht_model_bridge.so" \
  --ctas 148 --routing-ablation --results "$MODEL_RESULTS"
```

The routing ablation starts four fresh processes with balanced positions and adjacent method pairs, one warmup and three measured waves per method. Without `--routing-ablation`, the three-method suite uses all six permutations. Run on an agreed quiet GPU/CPU window. Set build and result variables to fresh owned directories. A single-rank runtime must instantiate LSA1; keep it separate from the four/eight-rank runtime used by the traffic fixture. For SM90 use `ARCH=sm_90a`, automatic staging and a CTA target of132; the new SM90 routing path currently has correctness qualification only.

Native auto-SDPA decode replay changed logits even with identical input. Fixed math SDPA and deterministic operations restore bitwise native replay; both settings are common to all timed methods. Their individual contributions to the earlier instability were not separated. This benchmark therefore measures controlled math-attention inference. [PyTorch2.10 SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)

`sanitize_expert_mma.py` uses an independent CPU FP32 oracle, CPU RNG and explicit tensor/cache teardown. Full-API memcheck with full leak checks and synccheck each pass15 arithmetic cases on B300. The first GPU-oracle memcheck passed arithmetic but retained Torch/CUBLAS caches at exit and failed11 leak reports; that original is preserved. Additional PTXAS TMEM guardrail instrumentation remains unavailable in the tested toolchain.

[Compact measured results](SM100_RESULTS.md)

For traffic-policy analysis, `thirdparty/nccl-ep/tests/measurement.py` validates
fixture receipt hashes, complete payload/probability checks, drained group
lifetimes and resource admission before computing timing summaries.
`autotune_balance.py train` selects from the fixed traffic policies;
`autotune_balance.py holdout --model <training-policy.json>` evaluates fresh
independently ordered pairs. Training and holdout remain separate.
