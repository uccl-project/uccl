# Four-L4 NCCL ABI replacement

UCCL Lite provides a `libnccl.so.2` replacement for one host with four L4 GPUs,
one GPU per MPI process. It reuses the multi-node strict ABI wrapper, symbol
version script, host consensus and identity-checked launcher. The local backend
uses MSCCL++ bootstrap, CUDA IPC and UCCL Lite kernels. It does not link or
forward collective calls to native NCCL.

**Qualified on the recorded four-L4 configuration with core2 placement.**
One fixed library passes all 126 performance points: AG 42/42, AR 42/42,
RS 42/42. The default Makefile policy is `four-l4-fp32`. A fresh default build
is byte-identical to the measured library, and its stock nccl-tests confirmation
passes. See the [qualification report](archived-experiments.md#local-only-records)
and [progress record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/local-strict-progress.md), which retains negative results.

The supported test scope is FP32 AllGather and FP32 SUM AllReduce/ReduceScatter,
eager calls, one live communicator per process, and serialized host submission.
The tested message range is powers of two from 1 KiB through 1 GiB, both layouts.
Caller buffers may be float-aligned; changing/recycled buffers and streams are
checked separately. Empty calls and empty groups are supported. Other datatypes,
reductions, nonempty groups, user graph capture, registration and other
collectives are refused explicitly. Internal CUDA graph reuse does not imply
support for capture by the caller.

Build from `lite`:

```bash
make nccl-local-strict BUILD=build/local_candidate -j4
```

This produces `build/local_candidate/nccl_local/libuccl_lite_nccl_local_strict.so`
and its `strict/libnccl.so.2` loader alias. Dependencies remain external: CUDA,
MSCCL++, NCCL headers and Open MPI for the clients. The Makefile builds the
independently linked CPU-router archive needed by the retained support code;
the selected gather path uses GPU progress and starts no CPU copy workers.
All binaries, dependency files and CUDA intermediates stay under `BUILD`.
The named build uses a fixed nvcc random seed and stable intermediate filenames.
Two independent builds were byte-identical. Build signatures track compiler,
include, policy and link inputs so changes do not silently reuse stale objects.

Run unchanged nccl-tests through the identity-checked launcher:

```bash
python3 -B tests/nccl_mscclpp/local.py \
  --library build/local_candidate/nccl_local/libuccl_lite_nccl_local_strict.so \
  --real-nccl "$HOME/third_party/nccl/build/lib/libnccl.so.2" \
  --tests build/nccl-tests-mpi \
  --mpirun /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  --cpu-placement core2 \
  --out build/local_candidate_check --max-bytes 1073741824 --unalign 1
```

Use a new output directory for each run. The driver records the profile,
hardware/software identity, commands, executable hashes and logs. It runs all
three stock binaries with four MPI ranks and one GPU each, verifies every output
row, checks every rank/size route, and requires `native_nccl_loaded=0` and
`fallback=0`. `--real-nccl` supplies version/identity information and the separate
comparison baseline; it is not a collective fallback in the replacement process.

`core2` assigns two distinct physical cores per rank on the GPU's NUMA node,
including allowed SMT siblings. The current testbed masks are `[4,6,28,30]`,
`[8,10,32,34]`, `[5,7,29,31]`, and `[9,11,33,35]`. Masks are derived from recorded
topology and the parent allowed set, fingerprinted and checked before exec.
Unknown or insufficient topology is refused. `unbound`, `numa`, and `core` remain
explicit alternatives with separate identities and measurements. A core2 profile
cannot silently be launched through a different placement mode.

Every run requires an explicit exact-shape profile and a matching current
identity. GPU count alone never admits a route. The first eligible collective
performs preparation; rejected shapes and empty operations do not prepare the
backend. Profiles for the two-host backend have a different magic and are not
accepted by the local library.

Performance qualification uses frozen original UCCL source `0fc3714b`, with the
fastest correct original variant at each size as the baseline. AR measures
maximum-rank per-call host enqueue plus stream completion (five warmups, ten
samples, three repetitions). AG/RS measure maximum-GPU per-call CUDA events,
including enqueue gaps (twenty warmups, ten samples, three repetitions), bracketed
by original sweeps. No samples are removed and any ratio above one fails. One
fixed library and placement must pass all 126 size/layout/operation points.
Stock nccl-tests correctness timings and profiler captures are not substitutes
for these gates. AG sizes denote full output bytes; RS sizes denote full input
bytes; AR sizes denote input/output bytes per rank.

The backend includes guarded ABI checks, delayed-rank queues, stream reuse,
forced counter resets, mixed legacy/stream-ordered allocations, and small-capacity
preparation checks. Three graph ranks plus one eager UCCL rank pass the mixed
allocation test. Those eager allocation-compatibility calls also use UCCL Lite;
no native NCCL path is available.

The qualified binary SHA256 is
`9b5d25d4a701f11c7c6dd1642c37f9b7d06365aee0c2450218bd61e606e953d7`.
The profile remains explicit; qualification does not enable automatic routing on
other four-GPU machines. Historical experimental commands can select
`UCCL_LITE_NCCL_LOCAL_POLICY=base` and a fresh `BUILD` directory.
