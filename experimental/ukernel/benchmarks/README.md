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

## CCL collective benchmarks

`all_reduce_perf` and `alltoall_perf` benchmark the `ccl` executor through the
real transport and device backends. They are intended to feel similar to
`nccl-tests`: rank 0 prints one row per message size with `time/algbw/busbw`.

Example `all_reduce_perf`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --no-python \
  --nproc-per-node 3 \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  ./benchmarks/all_reduce_perf \
  -b 393216 \
  -e 1048576 \
  -f 2 \
  -w 5 \
  -n 20 \
  --transport auto \
  --tile-bytes 65536 \
  --num-flows 2 \
  --exchanger-ip 127.0.0.1 \
  --exchanger-port 29600
```

Example `alltoall_perf`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --no-python \
  --nproc-per-node 3 \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  ./benchmarks/alltoall_perf \
  -b 65536 \
  -e 1048572 \
  -f 2 \
  -w 5 \
  -n 20 \
  --transport auto \
  --tile-bytes 65536 \
  --num-flows 2 \
  --exchanger-ip 127.0.0.1 \
  --exchanger-port 29600
```

Useful options:

- `-b`, `--min-bytes`: minimum bytes per rank
- `-e`, `--max-bytes`: maximum bytes per rank
- `-f`, `--factor`: geometric growth factor
- `-w`, `--warmup-iters`: warmup iterations per size
- `-n`, `--iters`: measured iterations per size
- `--transport`: `auto`, `ipc`, `uccl`, or `tcp`
- `--tile-bytes`: planner tile size
- `--num-flows`: planner pipeline flow count
- `--threads-per-block`, `--fifo-capacity`, `--smem-size`: device backend tuning
- `--check 0|1`: disable or enable one correctness run per size

Current implementation note:

- The planner accepts very small messages, but the current `ccl` runtime is
  much more stable in benchmark mode once `allreduce` starts from at least
  `2 * nranks * tile_bytes` and `alltoall` starts from at least `tile_bytes`.
