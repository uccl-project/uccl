# UCCL-GIN vs NCCL-GIN microbench (standalone, Rail path)

Exercises the UCCL-GIN Rail ops (**put + red_add_rel**) in isolation and
compares against native **NCCL-GIN** running the same
paired-remote workload in the same MPI program.

This is the standalone "test those few APIs vs direct NCCL GIN" step. The
reusable device ops live in `experimental/uccl_gin/include/uccl_gin/`.

## What it measures

- **Workload (paired-remote)**: 2 nodes × `local_world` ranks; rank `r` streams
  `iters` messages of `bytes` to its remote pair `(r+local_world)%world`, then
  ordered `red_add` completion counters. The binary checks recv data and
  payload-before-tail ordering before printing bandwidth.
- **Two paths in one binary**:
  - `NCCL-GIN`: `ncclMemAlloc` + window + `ncclDevComm`(GIN) + `ncclGin.put(SignalInc)`
    (reference, modeled on `nccl/docs/examples/06_device_api/02_alltoall_gin`).
  - `UCCL-GIN`: `UcclProxy` (EFA) + D2H ring + `uccl_gin::rail_put / rail_red_add`.
- Size sweep (default 4 KiB … 16 MiB), prints avg per-rank GB/s for both.
  The UCCL path is timed with CPU wall clock and waits until the receiver-side
  ordered atomic counter is observed locally, so it includes GPU enqueue,
  D2H-ring drain, proxy posting, EFA completion, and receiver atomic handling.

## Build

```bash
cd experimental/uccl_gin
make NUM_PROXY_THS=4 CUDA_HOME=/usr/local/cuda-13.0 PYTHON="$VIRTUAL_ENV/bin/python" SM=90 -j
```

## Run (EP16, 8 GPUs/node, 2 nodes)

Launch one MPI rank per GPU, 8 per node (contiguous ranks per node):

```bash
# common env
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:$NCCL_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_NET_PLUGIN=ofi NCCL_SOCKET_IFNAME=enp71s0
export LOCAL_WORLD_SIZE=8

mpirun --host p5en_0:8,p5en_1:8 -npernode 8 \
       -x LD_LIBRARY_PATH -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA \
       -x OFI_NCCL_FORCE_NUM_RAILS -x NCCL_NET_PLUGIN -x NCCL_SOCKET_IFNAME -x LOCAL_WORLD_SIZE \
       ./build/uccl_gin_microbench --iters 50 --sizes 4096,16384,65536,262144,1048576,8388608
```

Flags: `--no-nccl` / `--no-uccl` to run one path; `--ifname`, `--iters`, `--warmup`, `--sizes`.

## Status / known gaps

This is a **first version**. Likely needs compile/iteration on the server:

1. **NCCL device linking**: verified on p5en with the `nvidia-nccl-cu13` wheel
   by putting the wheel include path first and linking `-l:libnccl.so.2`.
2. **UCCL proxy bootstrap**: `UcclProxy` ctor args, `set_peers_meta`, `start_dual`,
   `get_d2h_channel_device_addrs` follow `uccl_proxy.hpp`; the PeerMeta OOB
   exchange here uses MPI_Allgather + `enp71s0` IP (instead of the Python
   `all_gather_object`). Confirm the listen-port handshake completes (the Python
   smoke slept ~1s after `start_dual`; we use an MPI barrier — may need a settle).
3. **Timing semantics**: UCCL now waits for the receiver counter before stopping
   the clock. NCCL is still event-timed inside its kernel path, so compare
   directionally until the two paths share an identical end-to-end timing harness.
4. **Correctness check is active**: recv region + completion counters are checked
   before bandwidth is printed.
5. **Transport snapshot**: this directory owns a copied proxy/RDMA snapshot. It is
   intentionally independent, but still needs further symbol and comment cleanup.

## Expected reference points (AGENTS.md)

Reference from previous p5en runs: with 2 lanes/NICs per GPU, UCCL-GIN reached
about 46 GB/s per rank at large sizes and significantly beat NCCL-GIN on small
messages after correctness/order checks passed.
