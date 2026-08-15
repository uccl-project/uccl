# Benchmarks

All project-owned benchmarks live in [`bench/`](../bench/). They cover
four layers: transport (point-to-point), device (reduce kernel), shim
(nccl-tests collectives), and standalone microbenchmarks. Perf-test
procedures against nccl-tests live in
[perf_test_procedure.md](perf_test_procedure.md).

## Transport and GDRCopy (`make bench`)

```bash
cd experimental/ukernel
make bench             # bench_transport + bench_gdrcopy
make transport_bench   # bench_transport only
```

### bench_transport

Transport performance benchmark over the Communicator: latency, one-way
throughput, bidirectional throughput, and payload-correctness checks
after each phase.

Flags: `--rank --peer-rank --gpu-id --msg-size --iterations --warmup
--ip --port --transport`. `--transport` accepts
`auto | ipc | uccl | tcp | rdma`.

IPC (same node, both GPUs visible to both processes):

```bash
CUDA_VISIBLE_DEVICES=6,7 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 \
  --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --port 6979
CUDA_VISIBLE_DEVICES=6,7 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 \
  --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc \
  --ip 127.0.0.1 --port 6979
```

RDMA (same node or cross-node):

```bash
# node A
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 \
  --iterations 1000 --warmup 100 --transport rdma --port 6981
# node B (replace IP with node A's address)
./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 \
  --iterations 1000 --warmup 100 --transport rdma --ip <NODE_A_IP> --port 6981
```

TCP:

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 \
  --iterations 1000 --warmup 100 --transport tcp --port 6982
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 \
  --iterations 1000 --warmup 100 --transport tcp --ip 127.0.0.1 --port 6982
```

Give each concurrent pair a unique `--port` so the bootstrap exchangers
do not collide. Reported numbers are environment-specific (GPU model,
link topology, NIC) — compare like for like on the same setup.

### bench_gdrcopy

Microbenchmark for GDRCopy host↔GPU mapping paths (pin/map bandwidth and
latency used by the device FIFOs and zero-copy signal rings):

```bash
cd experimental/ukernel
./bench/bench_gdrcopy
```

## Device reduce kernel (`bench_device_reduce_blocks.sh`)

Measures device reduce-kernel throughput vs block count and threads per
block, using the persistent-worker dispatch bench
(`src/device/benchmarks/bench_device_launch_vs_worker`). This is how we
pick `blocks_per_worker` so the reduce kernel keeps up with the IPC put
bandwidth at the fewest SMs.

```bash
cd experimental/ukernel
bash bench/bench_device_reduce_blocks.sh
# env overrides: BLOCKS THREADS SIZES ROUNDS WARMUP SMEM
```

Output rows: `blocks|threads|bytes|task_us|GB/s`, where GB/s counts
payload bytes reduced per second (read+write traffic is 2x that).

## Shim parameter sweep (`bench_shim_param_sweep.sh`)

Sweeps the shim's environment knobs against nccl-tests to find the best
(`UK_CCL_LARGE_TILES`, `UK_CCL_IPC_BATCH`, `UK_CCL_TILE_MIN_BYTES`,
`UK_CCL_DEV_BLOCKS`) combination on a given machine.

```bash
cd experimental/ukernel
CUDA_VISIBLE_DEVICES=6,7 bash bench/bench_shim_param_sweep.sh \
  [all_reduce|all_gather|reduce_scatter]
```

Env overrides: `SWEEP_MIN/SWEEP_MAX`, `ITERS/WARMUP`,
`LARGE_TILES_VALS`, `IPC_BATCH_VALS`, `TILE_MIN_VALS`,
`DEV_BLOCKS_VALS`, `BASE_LT/BASE_TM/BASE_IB`. Output is one line per
(config, size): `label|size|oop_time_us|oop_algbw|oop_wrong|ip_time_us|
ip_algbw|ip_wrong`, followed by a best-per-size summary (max OOP algbw).
mpirun must run with CPU binding disabled
(`--mca hwloc_base_binding_policy none`) and env knobs propagated with
`-x`, or small-message latency explodes.

## AllToAll (`alltoall_perf.cu`)

Minimal `ncclAllToAll` bandwidth benchmark. The same binary runs against
the ukernel shim or native NCCL, so the comparison is apples-to-apples
on the standard API. nccl-tests' `alltoall_perf` uses
`ncclSend`/`ncclRecv`, which the shim does not implement; this bench
uses `ncclAllToAll` directly.

```bash
cd experimental/ukernel && nvcc -O3 -arch=sm_103 -o /tmp/alltoall_perf \
  bench/alltoall_perf.cu -lnccl
mpirun -np 1 /tmp/alltoall_perf --rank=0 --nranks=N --bytes=268435456 \
  : -np 1 /tmp/alltoall_perf --rank=1 --nranks=N --bytes=268435456 \
  : ... : -np 1 /tmp/alltoall_perf --rank=N-1 --nranks=N --bytes=268435456
```

Rank 0 writes the NCCL unique id to `/tmp/uk_a2a_id`; other ranks poll
for it. Report uses the nccl-tests convention:
`algbw = (nranks-1)/nranks * total_bytes / avg_time`,
`busbw = algbw * nranks/(nranks-1)`.

## CE contention microbenchmark (`ce_contention.cu`)

Standalone 8-process alltoall that isolates copy-engine contention under
a synchronized start (see
[optimization_framework.md](optimization_framework.md), Appendix A for
results):

```bash
cd experimental/ukernel
nvcc -O3 -arch=sm_103 -o /tmp/ce_contention bench/ce_contention.cu
rm -f /tmp/ce_bar_* /tmp/ce_h_* /tmp/ce_rdy_*
for r in 0 1 2 3 4 5 6 7; do
  /tmp/ce_contention --rank $r --nranks 8 --bytes $((1<<28)) --iters 20 \
    > /tmp/ce_r$r.log 2>&1 &
done
wait
for r in 0 1 2 3 4 5 6 7; do cat /tmp/ce_r$r.log; done
```

Flags: `--serial 1` puts all 7 copies on one stream (8 concurrent
transfers instead of 56); `--sm 1` replaces `cudaMemcpyAsync` with a
vectorized copy kernel; `--batch` uses a single `cudaMemcpyBatchAsync`
(NCCL's CE-collective submission pattern).

## Spray executor benchmark

Native (non-shim) executor benchmark in `src/ccl`:

```bash
cd experimental/ukernel/src/ccl && make test_perf_spray_allreduce
CUDA_VISIBLE_DEVICES=0,1 ./test_perf_spray_allreduce --role=server --gpu 0 --dev-blocks=8 &
CUDA_VISIBLE_DEVICES=0,1 ./test_perf_spray_allreduce --role=client --gpu 1 --dev-blocks=8
# AllToAll: add --kind=alltoall
# Multi-block: --dev-blocks=1|4|8|64 to sweep SM usage
```
