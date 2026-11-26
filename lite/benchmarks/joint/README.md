# Joint peer/router benchmark

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


Frozen experiment-specific source copies and the original optimization journal are in [the archive](../../docs/archived-experiments.md). Current capability checks and fresh exact-request qualification remain required.
