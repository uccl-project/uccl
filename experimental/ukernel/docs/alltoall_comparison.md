# AllToAll comparison plan (vs user-space MoE implementations)

## Decision

Do **not** benchmark AllToAll against native NCCL: NCCL has no dedicated
AllToAll primitive (only `ncclSend`/`ncclRecv` groups, which is how
engines and nccl-tests build it themselves). Training/inference engines
implement AllToAll (e.g., MoE dispatch/combine) on their own, so our
AllToAll competes with **their user-space implementations**, not with a
NCCL primitive.

## Our current position

- Native ukernel AllToAll (`CollKind::AllToAllPairwise`), measured via
  `test_perf_spray_allreduce --kind=alltoall`: 256MB 2.88ms (93 GB/s,
  8 blocks) on the A40 pair.
- The shim's `ncclAllToAll` is the same algorithm but is not reachable
  through nccl-tests (`alltoall_perf` uses `ncclSend`/`ncclRecv`, which
  the shim does not implement).

## Reference: `thirdparty/DeepEP`

The repo carries DeepEP (MoE dispatch/combine all-to-all):

- normal kernels: NVLink intranode, RDMA internode, asymmetric
  NVLink→RDMA domain forwarding;
- low-latency kernels;
- **SM number control** (matches our "fewer SMs, comparable
  performance" goal);
- FP8/BF16 support; DeepSeek-V3/R1 production shapes (4096 tokens ×
  7168 hidden, top-4/top-8 experts, FP8 dispatch + BF16 combine).

## Comparison plan

1. **Define common shapes** from DeepSeek-V3 production settings:
   prefill 4096×7168, decode 128×7168, top-4/top-8 experts; also a
   general pairwise all-to-all shape (per-expert token groups).
2. **Map our AllToAll to dispatch/combine semantics**: each rank holds
   a set of experts; tokens are grouped by destination expert and
   exchanged pairwise (dispatch), then results returned (combine).
3. **Harness**: same GPUs, same shapes, same iteration counts, same
   stream-sync timing; report latency and throughput per shape. Our
   side runs at the executor level (spray) or a small harness — not
   nccl-tests, since the shim lacks Send/Recv.
4. **Also compare SM usage** at matched performance (DeepEP exposes SM
   count; we sweep `UK_CCL_DEV_BLOCKS`).
5. **Known gaps to close before a fair run**:
   - FP8 dispatch: our device reduce path does not yet support Fp8;
   - internode AllToAll: our AllToAll is validated same-node only
     (internode would go through the RDMA adapter).

Open question: whether to compare at the library API level (our
pairwise AllToAll vs DeepEP's dispatch/combine on the same token-expert
layout) or at the kernel level (our spray AllToAll vs DeepEP kernels on
equivalent buffers).
