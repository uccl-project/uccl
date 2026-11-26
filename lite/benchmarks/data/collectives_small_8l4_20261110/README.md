# Eight-L4 BF16 small-message collectives

This historical record fills 1 KiB–512 KiB for the eight-L4 AllReduce and
AllGather figures and retains the earlier ReduceScatter and All-to-All runs.
The current ReduceScatter figure uses its [new paired rerun](../reducescatter_8l4_20261110_v2/README.md).
The 1 MiB point in each figure comes from the matching large-message run, so
both plot panels share that measured boundary point. GPUs 0–3 are on NUMA node 0 and
4–7 on node 1. CUDA 12.9, NVIDIA driver 580.178.04, and NCCL 2.28.9 match
the large-message record.

All sizes are complete BF16 tensor bytes per rank. AllReduce is in place;
AllGather, ReduceScatter, and All-to-All are out of place. Each run uses five
warmups and three alternating repetitions of ten timed calls per backend.
Each call checks every output element on all ranks. Fill, allocation,
verification, and communicator setup are outside timing; latency is the
slowest-rank CUDA-event interval. The five sweeps contain 5,700 checked timed
calls in total. The final line of each summary records the correctness pass.

| Files | Scope |
| --- | --- |
| `ar_local.csv`, `out_local.csv` | Paired CPU router versus project-owned local path; all three operations in the out-of-place run. |
| `ar_nccl.csv`, `out_nccl.csv` | Paired CPU router versus NCCL using the same placements. |
| `a2a.csv` | NUMA host staging, peer+host, and NCCL grouped send/recv. |
| `*_samples.csv` | Per-call slowest-rank CUDA-event times. |
| `*.profile`, `*.log` | Calibration identities and selection diagnostics; the NCCL profiles are external controls, not local-fallback policy evidence. |

The [selected-path summary](../collectives_8l4_20261110.csv) uses the
AllReduce and AllGather measurements here, choosing the faster UCCL Lite
median from each local comparison without consulting NCCL. For CPU-router
winners, it reports the router and NCCL timings from their direct paired run.
At 128 KiB, local wins for AllReduce, so that UCCL timing comes from the
separate local comparison. The earlier ReduceScatter and All-to-All data
remain here as experimental records, including their losses.

## Reproduce

From `experimental/lite`, with new output paths:

```bash
make -j2 adaptive-collectives eight-gpu-alltoall \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback local \
  --op all_reduce --in-place --begin 1024 --end 524288 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/ar-local-samples.csv --calibrate build/ar-local.profile
# Repeat with --fallback nccl and distinct output paths.
# For AG/RS, use --op all without --in-place and distinct output paths.
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_alltoall_8l4 --begin 1024 --end 524288 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/a2a-small-samples.csv
```
