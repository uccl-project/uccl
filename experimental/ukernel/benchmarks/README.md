## Build

NVIDIA:

```bash
cd experimental/ukernel
make transport_bench
make bench
```

## Transport benchmark

`bench_transport` is the transport performance benchmark.

Available transport modes:

```bash
auto
ipc
uccl
tcp
```

Available IPC path modes:

```bash
auto
relay
```

`--ipc-path relay` forces the same-host IPC benchmark onto the host relay path.

## Run

Server:

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --ip 127.0.0.1 --port 6979 --transport ipc
```

Client:

```bash
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --ip 127.0.0.1 --port 6979 --transport ipc
```

IPC direct:

```bash
CUDA_VISIBLE_DEVICES=6 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --ipc-path auto --port 6979
CUDA_VISIBLE_DEVICES=7 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --ipc-path auto --port 6979
```

IPC host relay:

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --ipc-path relay --port 6980
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --transport ipc --ipc-path relay --port 6980
```

UCCL:

```bash
CUDA_VISIBLE_DEVICES=6 ./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport uccl --port 6981
CUDA_VISIBLE_DEVICES=6 ./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport uccl --port 6981
```

TCP:

```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --port 6982
./bench_transport --rank 1 --peer-rank 0 --gpu-id 1 --msg-size 1048576 --iterations 1000 --warmup 100 --transport tcp --port 6982
```

The benchmark reports:
- latency
- one-way throughput
- bidirectional throughput
- payload correctness after each phase

If you launch multiple benchmark pairs on one host, give each pair a unique
`--port` so the bootstrap exchanger does not collide.
