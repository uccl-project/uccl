# Eight-L4 BF16 All-to-All comparison

This standalone experiment fills the eight-GPU All-to-All gap in the current
single-node README. It measures two project-owned paths against grouped NCCL
send/recv calls in one process. It is an eager benchmark, not an automatic
runtime selector or an NCCL compatibility layer.

## Machine and method

- `wxz-l4`: eight NVIDIA L4s, GPUs 0–3 on NUMA node 0 and 4–7 on node 1;
  CUDA peer access exists within each group. CUDA 12.9, NVIDIA driver
  580.178.04, NCCL 2.28.9. The CSV's `driver=13000` field is the CUDA driver
  API version reported by `cudaDriverGetVersion`, not the NVIDIA driver build.
- BF16, out-of-place, one complete input and output tensor per GPU. Each input
  holds eight equal destination chunks; each output holds eight rank-ordered
  source chunks. Sizes double from 1 MiB to 1 GiB per rank.
- `uccl-host` stages one destination chunk per source in NUMA-pinned host
  memory. `uccl-hybrid` uses CUDA peer copies inside each four-GPU group and
  host staging between groups. NCCL uses grouped `ncclSend`/`ncclRecv` pairs
  for nonself chunks and a local copy for the self chunk. Both UCCL paths use
  a collision-free rank-shift schedule.
- Five warmups and ten timed calls per backend in each of three repetitions;
  backend order alternates. The 11 sizes × three backends × three repetitions
  × ten calls give **990 timed samples** in [samples.csv](samples.csv). Every
  warmup and timed call verifies every output element on all eight GPUs. Fill,
  verification, allocation, and NCCL communicator setup are outside timing.
  Latency is the slowest-rank CUDA-event interval; min/max bound the three
  ten-call medians.

| Complete bytes/rank | UCCL host | UCCL peer+host | NCCL |
| ---: | ---: | ---: | ---: |
| 1 MiB | 0.849 ms | 1.127 ms | 0.439 ms |
| 64 MiB | 20.193 ms | 50.202 ms | 13.242 ms |
| 1 GiB | 306.545 ms | 791.372 ms | 186.989 ms |

Host staging is the faster UCCL Lite candidate at **all 11 sizes**, and NCCL
is faster at all 11. The [complete summary](summary.csv) retains both UCCL
paths and their negative results. The final build also passed a checked
1 MiB + 128 B tail case.

## Reproduce

From `experimental/lite`, with new output paths:

```bash
make -j2 eight-gpu-alltoall \
  CUDA_HOME=/usr/local/cuda-12.9 NCCL_HOME=build/nccl-2.28.9
env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_alltoall_8l4 --begin 1048576 --end 1073741824 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/alltoall-8l4-new-samples.csv
```
