# UKernel benchmarks

## Build

```bash
cd experimental/ukernel
make bench             # bench_transport + bench_gdrcopy
make transport_bench   # bench_transport only
```

## bench_transport

Transport performance benchmark over the Communicator: latency, one-way
throughput, bidirectional throughput, and payload-correctness checks
after each phase.

Flags: `--rank` `--peer-rank` `--gpu-id` `--msg-size`
`--iterations` `--warmup` `--ip` `--port` `--transport`

`--transport` accepts `auto | ipc | uccl | tcp | rdma`.

### IPC (same node, both GPUs visible to both processes)

```bash
CUDA_VISIBLE_DEVICES=6,7 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --port 6979
CUDA_VISIBLE_DEVICES=6,7 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --ip 127.0.0.1 --port 6979
```

### RDMA (same node or cross-node)

```bash
# node A
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport rdma --port 6981
# node B (replace IP with node A's address)
./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport rdma --ip <NODE_A_IP> --port 6981
```

### TCP

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --port 6982
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --ip 127.0.0.1 --port 6982
```

When launching several benchmark pairs on one host, give each pair a
unique `--port` so their bootstrap exchangers do not collide. All
reported numbers are environment-specific (GPU model, link topology,
NIC) — compare like for like on the same setup.

## bench_gdrcopy

Microbenchmark for GDRCopy host↔GPU mapping paths (pin/map bandwidth
and latency used by the device FIFOs and zero-copy signal rings):

```bash
cd experimental/ukernel
./benchmarks/bench_gdrcopy
```
