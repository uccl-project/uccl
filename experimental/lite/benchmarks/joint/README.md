# Joint peer/router evaluation (EAT setup baseline)

This file records the historical frozen experiment protocol. For the integrated
branch, current build targets, four/eight-rank router fixtures, measured losses
and reproduction commands, see [retained optimizations](../../docs/retained-optimizations.md).

This experiment starts from committed UCCL Lite `5b9a7f86`, separately from the earlier CPU-router-only EAT task. The original checkout's uncommitted experiments are excluded. Integrating the two methods and adding BF16/rank support creates a **new baseline**, not an optimization win.

The direct-peer adapter in `src/collectives/joint_collectives.cu` generalizes the destination-pull patterns of `peer_ag_rs.cu` from four-rank FP32 to two/four-rank BF16, with full-tensor reduction for AR. It is not an unchanged historical FP32 benchmark or a claim about a separately named PCIeCCL project. It requires all-pairs CUDA peer access. The router method uses the current upstream `CpuRouterOptions` defaults, which include an optional GPU-progress reduction for the exact small-RS case; “router” denotes the routed transport, not exclusively CPU arithmetic.

Build `make joint-bench joint-nccl-reference` with the node's CUDA/NCCL/CUTLASS paths. `build/uccl_lite_joint_nccl_reference` is compiled with `NCCL_REFERENCE_ONLY`: candidate headers, dispatch/policy implementations and router/peer objects are excluded. It provides the immutable NCCL reference. NCCL default is primary on **both** nodes; `--shm-only` is a separate diagnostic and must not replace that primary comparison. The wrapper's peer/router/joint execution never calls NCCL.

Common arguments:

```
--backend router|peer|joint|nccl|calibrate
--gpus 2|4 --begin 1024 --end 1073741824 --op all|allreduce|allgather|reducescatter
--warmup 100 --repeats 5 --iters 20 --samples NEW_CSV
--source-identity REVISION --profile PROFILE --seed INTEGER [--reverse] [--shm-only]
```

All collectives are BF16 and out of place. Bytes means the AR tensor, complete AG output, or complete RS input. The scored grid has 63 equal-weight cases: three collectives and 21 powers of two from 1 KiB to 1 GiB. Preparation, communicator creation, allocations, input initialization and correctness checking are outside the event interval. Every invocation measures the maximum across GPUs of the CUDA event interval around enqueue, including host enqueue gaps. Every invocation is independently initialized and checked, including warmups. Reported case latency is the median of five repeat medians, each containing 20 invocations; 100 warmups precede them. This is an eager collective benchmark, not application inference throughput or a CUDA-graph comparison.

Caller producers are queued on nonblocking caller streams without a host synchronization before enqueue. The implementation must queue output and input-reader completion on those same streams. Every invocation checks the full output, unchanged input and 256-byte guards on both sides, outside timing. Values are rank/index/generation-dependent signed dyadic BF16 numbers whose two/four-rank sums are exactly representable, avoiding reduction-order ambiguity in scored tests.

`--backend calibrate` uses **separate invocations**, normally 20 warmups and 3 x 10 samples, with a distinct generation seed from held-out measurements. It measures router and peer in separate object lifetimes, destroys CPU workers before peer timing, alternates measurement order, and writes a new profile only after every case passes. It never uses NCCL measurements. `--backend joint` loads that profile in a later process and measures the actual dispatch choice and enqueue cost. Each record matches exact operation, BF16, size, out-of-place placement, pointer alignment, hardware/software identity, supplied source revision, executable identity and router options. Missing/stale records safely choose supported direct peer; unavailable peer selects a capable router or rejects unsupported topology. No route is chosen from the held-out timing output. Recalibrate every changed binary.

Standard output contains a summary CSV followed by `# correctness: PASS`; stderr contains configuration and decision records. Summary columns are `operation,bytes,backend,selected_backend,minimum_us,median_us,maximum_us,correct`. Raw CSV columns are `operation,bytes,backend,selected_backend,repeat,iteration,slowest_rank_us,generation`. Calibration writes two rows per case/repeat/iteration, distinguished by `selected_backend`. NCCL SHM rows explicitly carry `nccl_shm`; default rows carry `nccl_default`.

`--selftest --gpus N --source-identity REVISION` additionally checks missing and stale profile identities, mismatched size/placement/alignment, unsupported peer capability, delayed producers, twelve asynchronous back-to-back calls with input reuse, repeated instantiation, and eager-only capture rejection/recovery. An independent host double oracle covers broad finite BF16 reductions with a forward error bound allowing intermediate BF16 tree rounding; AG checks arbitrary BF16 bit patterns including signed zero, subnormals, infinities and NaNs exactly. Odd 257-element contributions test tails outside the scored grid. Existing router policy, dispatcher, eager and graph tests remain separate required checks.

The benchmark, this protocol, reference code and test sources are frozen after setup and before optimization iteration 1. Short diagnostics may override repetition counts but are labeled diagnostic; they cannot establish final NCCL parity or override promotion gates. Preserve failures and rejected trials, report separate node scores, and distinguish setup integration, optimization improvement, per-case wins and aggregate NCCL parity.

## Protocol amendment before scored trials

The user was informed before evaluation continuation: actual joint and NCCL control curves retain100 warmups and5x20 measurements across all63cases. Separate pure-router/peer diagnostic curves and independent calibration use3 warmups and3x5 measurements. The initial scalar peer adapter is exceptionally slow for large cross-NUMA transfers and is not the upstream selected best FP32 implementation. This amendment avoids exhausting the run on that known weak unselected path. Pure-path scores are diagnostic; official score remains full measured joint/NCCLdefault. Calibration remains a separate process with a distinct generation seed. Pre-amendment baseline logs are preserved and labeled.
