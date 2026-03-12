# UKernel
Ultra Unified and Fine-grained Kernel

```
sudo apt-get update
sudo apt-get install -y libelf-dev
```

## transport/runtime develpment:
on AMD
```
cd experimental/ukernel
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel
make clean -f Makefile && make -j$(nproc) -f Makefile
```
> The CUDA build container uses glog 0.5 (libglog.so.0), but many host systems use glog 0.6 (libglog.so.1), causing runtime linking errors.

## test transport communicator
```
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client

# notifier version transport
CUDA_VISIBLE_DEVICES=5 ./test_main --role=server-notifier
CUDA_VISIBLE_DEVICES=5 ./test_main --role=client-notifier
```

```
CUDA_VISIBLE_DEVICES=5 python py/test_p2p.py
```


## compute develpment
on AMD
```
cd experimental/ukernel/src/compute
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel/src/compute
make clean -f Makefile && make -j$(nproc) -f Makefile
```

## test compute
```
CUDA_VISIBLE_DEVICES=5 ./test_persistent

# bench
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_full_fifo
CUDA_VISIBLE_DEVICES=5 ./benchmarks/bench_sm_fifo 83
```

## Transport Benchmark

Build the transport benchmark:
```bash
cd experimental/ukernel
make -f Makefile bench
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UKERNEL_TRANSPORT_BACKEND` | Transport backend: `uccl` or `basic`/`basicrdma`/`basic_rdma` | `uccl` |
| `UKERNEL_RDMA_CHUNK_SIZE` | RDMA chunk size in bytes | 1048576 |
| `UKERNEL_QP_COUNT` | QP count per endpoint | 1 |
| `UKERNEL_CQ_COUNT` | CQ count per endpoint | 1 |
| `UKERNEL_CQ_POLLER_THREADS` | CQ poller threads | 1 |
| `UKERNEL_CQ_DEPTH` | CQ depth | 2048 |
| `UKERNEL_MAX_RETRY_TIMES` | Max retry times | 10 |
| `UKERNEL_RESOLVE_TIMEOUT_MS` | Resolve timeout in ms | 2000 |
| `UKERNEL_QP_MAX_SEND_WR` | Max send WR per QP | 2048 |
| `UKERNEL_QP_MAX_RECV_WR` | Max receive WR per QP | 2048 |
| `UKERNEL_QP_MAX_SGE` | Max SGE per WR | 1 |
| `UHM_EXCHANGER_SERVER_IP` | Exchanger server IP | `0.0.0.0` |
| `UHM_EXCHANGER_SERVER_PORT` | Exchanger server port | 6979 |

### Usage

The benchmark requires two processes (sender and receiver) running simultaneously. Lower rank acts as sender, higher rank acts as receiver.

**Using UCCL backend (default):**
```bash
# Terminal 1 (Receiver - 先启动)
./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 10000 --ip 192.168.2.243 --port 6979
# Terminal 2 (Sender - 后启动)
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 10000 --ip 192.168.2.243 --port 6979
```

**Using BasicRDMA backend:**
```bash
# Terminal 1 (Sender - rank 0):
UKERNEL_TRANSPORT_BACKEND=basic ./bench_transport --rank 0 --peer-rank 1 --msg-size 1048576

# Terminal 2 (Receiver - rank 1):
UKERNEL_TRANSPORT_BACKEND=basic ./bench_transport --rank 1 --peer-rank 0 --msg-size 1048576
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rank` | Current process rank (0 or 1) | Required |
| `--peer-rank` | Peer process rank | Required |
| `--gpu-id` | GPU device ID | 0 |
| `--msg-size` | Message size in bytes | 1024 |
| `--iterations` | Number of test iterations | 10000 |
| `--warmup` | Number of warmup iterations | 1000 |
| `--ip` | Local IP address | 127.0.0.1 |
| `--port` | Listen port | 6979 |

### Output Metrics

The benchmark reports three test results:

1. **Latency Test (Ping-Pong)**
   - Measures round-trip latency
   - Reports: min, p50, p90, p99, max (microseconds)

2. **Throughput Test (One-way)**
   - Measures unidirectional send bandwidth
   - Reports: Total data (GB), Time (sec), Throughput (GB/s, Gbps), Messages/sec

3. **Bidirectional Throughput Test**
   - Measures simultaneous send/receive bandwidth
   - Reports: Total data (GB), Time (sec), Throughput (GB/s, Gbps)

### Examples

**Test with 1MB messages using UCCL:**
```bash
# Terminal 1
./bench_transport --rank 0 --peer-rank 1 --msg-size 1048576

# Terminal 2
./bench_transport --rank 1 --peer-rank 0 --msg-size 1048576
```

**Test with 4KB messages using BasicRDMA:**
```bash
# Terminal 1
UKERNEL_TRANSPORT_BACKEND=basic ./bench_transport --rank 0 --peer-rank 1 --msg-size 4096

# Terminal 2
UKERNEL_TRANSPORT_BACKEND=basic ./bench_transport --rank 1 --peer-rank 0 --msg-size 4096
```

**Test with custom RDMA configuration:**
```bash
export UKERNEL_RDMA_CHUNK_SIZE=524288
export UKERNEL_QP_COUNT=4
export UKERNEL_CQ_DEPTH=4096

./bench_transport --rank 0 --peer-rank 1 --msg-size 1048576
```