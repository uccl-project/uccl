## Build

on AMD
```
cd experimental/ukernel/benchmarks
make clean -f Makefile.rocm && make -j$(nproc) -f Makefile.rocm
```

on Nvidia
```
cd experimental/ukernel/benchmarks
make clean -f Makefile && make -j$(nproc) -f Makefile
```

## Transport Benchmark

Build the transport benchmark:
```bash
cd experimental/ukernel
make -f Makefile bench
```

### Usage

The benchmark requires two processes (sender and receiver) running simultaneously. Lower rank acts as sender, higher rank acts as receiver.

**Terminal 1 (Sender - rank 0):**
```bash
./bench_transport --rank 0 --peer-rank 1 --gpu-id 0 --msg-size 1048576 --iterations 10000 --ip 127.0.0.1 --port 6979
```

**Terminal 2 (Receiver - rank 1):**
```bash
./bench_transport --rank 1 --peer-rank 0 --gpu-id 0 --msg-size 1048576 --iterations 10000 --ip 127.0.0.1 --port 6979
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

### Example with Different Message Sizes

Test with 1MB messages:
```bash
# Terminal 1
./bench_transport --rank 0 --peer-rank 1 --msg-size 1048576

# Terminal 2
./bench_transport --rank 1 --peer-rank 0 --msg-size 1048576
```

Test with 4KB messages over 1000 iterations:
```bash
# Terminal 1
./bench_transport --rank 0 --peer-rank 1 --msg-size 4096 --iterations 1000

# Terminal 2
./bench_transport --rank 1 --peer-rank 0 --msg-size 4096 --iterations 1000
```