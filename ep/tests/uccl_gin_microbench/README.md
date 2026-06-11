# UCCL-GIN vs NCCL-GIN microbench (standalone, Rail path)

Exercises the UCCL-GIN Rail ops (**put + red_add_rel**) in isolation — **not**
wired into DeepEP V2 — and compares against native **NCCL-GIN** running the same
paired-remote workload in the same MPI program.

This is the standalone "test those few APIs vs direct NCCL GIN" step from
`ep/docs/uccl_gin_plan.md` (P0/P1 验证). The reusable device ops live in
`ep/include/uccl_gin/uccl_gin_rail.cuh`.

## What it measures

- **Workload (paired-remote)**: 2 nodes × `local_world` ranks; rank `r` streams
  `iters` messages of `bytes` to its remote pair `(r+local_world)%world`, then one
  ordered `red_add` to a per-peer counter. Per-rank achievable BW + (TODO)
  correctness of the recv region and the counter.
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
# 1) build the UCCL ep transport objects (this microbench links ../../src/*.o)
cd ep && make install PYTHON="$VIRTUAL_ENV/bin/python" CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j
# 2) build the microbench
cd tests/uccl_gin_microbench
make NUM_PROXY_THS=4 CUDA_HOME=/usr/local/cuda-13.0
```

## Run (EP16, 8 GPUs/node, 2 nodes)

Launch one MPI rank per GPU, 8 per node (contiguous ranks per node):

```bash
# common env (same as the DeepEP V2 runs)
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:$NCCL_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_NET_PLUGIN=ofi NCCL_SOCKET_IFNAME=enp71s0
export LOCAL_WORLD_SIZE=8

mpirun --host p5en_0:8,p5en_1:8 -npernode 8 \
       -x LD_LIBRARY_PATH -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA \
       -x OFI_NCCL_FORCE_NUM_RAILS -x NCCL_NET_PLUGIN -x NCCL_SOCKET_IFNAME -x LOCAL_WORLD_SIZE \
       ./microbench --iters 50 --sizes 4096,16384,65536,262144,1048576,16777216
```

Flags: `--no-nccl` / `--no-uccl` to run one path; `--ifname`, `--iters`, `--warmup`, `--sizes`.

## ⚠️ Status / known gaps (written without a local GPU; verify on p5en)

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
4. **Correctness check** is still stubbed (TODO in `main`): copy back recv region +
   counter and compare to rank-tagged send data + expected red_add count.
5. **Single-stream**: both paths use one CTA / one D2H lane for an apples-to-apples
   single-stream number. Multi-lane / multi-CTA fan-out is a later sweep (and is
   where coalescing — see plan §4.1 — will matter for UCCL).

## Expected reference points (AGENTS.md)

NCCL-GIN proxy on this EFA HW: large messages ~44 GB/s/rank; small (16 KiB) ~8.8,
(32 KiB) ~12.5 GB/s/rank single-stream. The UCCL-GIN per-token (uncoalesced) path
was ~30 GB/s in the DeepEP integration; this microbench isolates the same ops so
the gap can be attributed cleanly (op overhead vs coalescing vs proxy rate).
