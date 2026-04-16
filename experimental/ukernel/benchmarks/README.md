# Benchmarks

This directory contains:

- `bench_transport`: transport-layer benchmark (`auto|ipc|uccl|tcp`)
- `bench_gdrcopy`: host/GDRCopy-focused benchmark

## Build

```bash
cd experimental/ukernel
make bench_transport
make bench_gdrcopy
make bench_all
```

Compatibility aliases:

```bash
make transport_bench   # alias of bench_transport
make bench             # alias of bench_all
```

## bench_transport

Transport modes:

```bash
auto
ipc
uccl
tcp
```

IPC path modes:

```bash
auto
relay
```

`transport=uccl` is the compatibility selector name for the RDMA adapter path.

### Example: two-process IPC

```bash
# server
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --ip 127.0.0.1 --port 6979 --transport ipc

# client
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --ip 127.0.0.1 --port 6979 --transport ipc
```

### Example: two-process RDMA path (`transport=uccl`)

```bash
CUDA_VISIBLE_DEVICES=6 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport uccl --port 6981
CUDA_VISIBLE_DEVICES=7 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport uccl --port 6981
```

### Example: two-process TCP

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --port 6982
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --port 6982
```

Output includes latency, one-way throughput, bidirectional throughput, and payload correctness checks.

If you run multiple benchmark pairs on one host, use unique `--port` values.
